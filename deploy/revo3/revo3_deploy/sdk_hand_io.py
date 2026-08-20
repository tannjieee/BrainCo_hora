from __future__ import annotations

import asyncio
import fcntl
import hashlib
import inspect
import math
from pathlib import Path
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from .tactile import FingertipForceAdapter
from .vision_touch import VisionTouchCollector

DEG_TO_RAD = math.pi / 180.0
RAD_TO_DEG = 180.0 / math.pi
RAD_S_TO_RPM = 60.0 / (2.0 * math.pi)
JOINT_DIM = 21
STALL_BIT = 1 << 8


def _load_sdk():
    try:
        from bc_revo3_sdk import main_mod as sdk
    except ImportError as exc:
        raise RuntimeError(
            "bc-revo3-sdk 1.5.1 is required for hardware access. Run this command "
            "with /home/tan/miniconda3/envs/revo3/bin/python."
        ) from exc
    version = str(getattr(sdk, "__version__", "unknown"))
    if version != "1.5.1":
        raise RuntimeError(f"Expected bc-revo3-sdk 1.5.1, found {version}.")
    return sdk


@dataclass(frozen=True)
class Revo3SdkConfig:
    port: str | None = None
    baudrate: int = 5_000_000
    slave_id: int | None = None
    auto_detect: bool = True
    configure_tactile: bool = False
    initialize_tactile: bool = True
    use_without_retry: bool = True
    expected_hand: str = "right"
    allowed_hardware_types: tuple[str, ...] = (
        "Revo3UltraTouch",
        "Revo3UltraVisionTouch",
    )
    serial_allowlist: tuple[str, ...] = ()
    max_abs_current_ma: float = 500.0
    allowed_stall_motor_ids: tuple[int, ...] = ()
    stall_grace_s: float | None = None


class Revo3SdkHandIO:
    """bc-revo3-sdk 1.5.1 adapter; all public angles use radians."""

    def __init__(self, config: Revo3SdkConfig, tactile_cfg: dict | None = None) -> None:
        self.config = config
        self.tactile_cfg = dict(tactile_cfg or {})
        self.force_adapter = FingertipForceAdapter.from_profile(tactile_cfg)
        self.sdk = _load_sdk()
        self.ctx: Any | None = None
        self.port: str | None = config.port
        self.baudrate = int(config.baudrate)
        self.slave_id: int | None = config.slave_id
        self.touch_vendor: int | None = None
        self.device_info: Any | None = None
        self.last_motor_currents_ma: np.ndarray | None = None
        self.last_stalled_motor_ids: tuple[int, ...] = ()
        self.last_stall_durations_s = np.zeros(JOINT_DIM, dtype=np.float64)
        self._stall_started_at_s = np.full(JOINT_DIM, np.nan, dtype=np.float64)
        self.device_position_lower_rad: np.ndarray | None = None
        self.device_position_upper_rad: np.ndarray | None = None
        self.vision_touch: VisionTouchCollector | None = None
        self.tactile_source = "revo3_touch"
        self.last_tactile_age_s: float | None = None
        self._motion_lock_handles: list[Any] = []

    async def open(self) -> None:
        self._acquire_motion_locks()
        try:
            self.sdk.init_logging()
            if self.config.auto_detect and (self.port is None or self.slave_id is None):
                detected = await self.sdk.revo3_auto_detect_modbus(self.port)
                if len(detected) != 4:
                    raise RuntimeError(
                        f"Unexpected Revo3 auto-detect result: {detected!r}."
                    )
                _, detected_port, detected_baudrate, detected_slave_id = detected
                if (
                    self.config.slave_id is not None
                    and int(detected_slave_id) != self.config.slave_id
                ):
                    raise RuntimeError(
                        f"Auto-detected slave {int(detected_slave_id)}, but "
                        f"{self.config.slave_id} was explicitly requested."
                    )
                self.port = str(detected_port)
                self.baudrate = _baudrate_int(self.sdk, detected_baudrate)
                self.slave_id = int(detected_slave_id)
            if self.port is None or self.slave_id is None:
                raise RuntimeError(
                    "port and slave_id are required when auto_detect is disabled."
                )

            self.ctx = await self.sdk.modbus_open(
                self.port, self._baudrate_enum(self.baudrate)
            )
            self.device_info = await self._ctx.revo3_get_device_info(self.slave_id)
            self._validate_device_identity()
            self._set_device_position_limits(
                await self._ctx.revo3_get_all_joint_position_limits(self.slave_id)
            )
            vendor = await self._ctx.revo3_get_touch_vendor(self.slave_id)
            self.touch_vendor = _enum_int(vendor)
            if not self.config.initialize_tactile:
                self.tactile_source = "disabled"
            elif self._is_vision_touch_hardware():
                if self.config.configure_tactile:
                    raise RuntimeError(
                        "VisionTouch does not use --configure-tactile; remove that flag and "
                        "configure the five camera sensors through the VisionTouch collector."
                    )
                vision_cfg = dict(self.tactile_cfg.get("vision_touch") or {})
                if not bool(vision_cfg.get("enabled", False)):
                    raise RuntimeError(
                        "This Revo3UltraVisionTouch hand requires tactile.vision_touch.enabled=true."
                    )
                self.vision_touch = VisionTouchCollector(vision_cfg)
                await asyncio.to_thread(self.vision_touch.open)
                self.tactile_source = "vision_touch"
            else:
                if self.touch_vendor not in (1, 2):
                    raise RuntimeError(
                        f"A Pressure or Matrix tactile hand is required, got vendor={vendor}."
                    )
                if self.config.configure_tactile:
                    await self._configure_tactile()
                await self._validate_tactile_mode()
        except BaseException:
            try:
                await self.close()
            except Exception:
                pass
            raise

    async def close(self) -> None:
        try:
            vision = self.vision_touch
            self.vision_touch = None
            vision_error: BaseException | None = None
            if vision is not None:
                try:
                    await asyncio.to_thread(vision.close)
                except BaseException as exc:
                    vision_error = exc
            if self.ctx is None:
                if vision_error is not None:
                    raise vision_error
                return
            ctx = self.ctx
            try:
                result = self.sdk.modbus_close(ctx)
                if inspect.isawaitable(result):
                    await result
            finally:
                self.ctx = None
            if vision_error is not None:
                raise vision_error
        finally:
            self._release_motion_locks()

    def _acquire_motion_locks(self) -> None:
        if self._motion_lock_handles:
            raise RuntimeError("Revo3 SDK motion lock is already held by this adapter.")
        serials = sorted(
            {
                str(value).strip().upper()
                for value in self.config.serial_allowlist
                if str(value).strip()
            }
        )
        if serials:
            identities = [f"serial:{serial}" for serial in serials]
        elif self.config.port is not None:
            endpoint = str(Path(self.config.port).expanduser().resolve())
            slave = "*" if self.config.slave_id is None else str(self.config.slave_id)
            identities = [f"endpoint:{endpoint}:slave:{slave}"]
        else:
            # Auto-detection without a serial allowlist cannot identify a device
            # before opening it, so conservatively serialize that whole hand side.
            identities = [f"autodetect:{self.config.expected_hand}"]

        acquired: list[Any] = []
        try:
            for identity in identities:
                key = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]
                path = Path(f"/tmp/revo3-sdk-motion-{key}.lock")
                handle = path.open("a+", encoding="utf-8")
                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BaseException:
                    handle.close()
                    raise
                acquired.append(handle)
        except BlockingIOError as exc:
            for handle in reversed(acquired):
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                handle.close()
            raise RuntimeError(
                "Another Revo3 process holds the SDK motion lock for this hand; "
                "hardware open was refused."
            ) from exc
        except BaseException:
            for handle in reversed(acquired):
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                handle.close()
            raise
        self._motion_lock_handles = acquired

    def _release_motion_locks(self) -> None:
        handles = getattr(self, "_motion_lock_handles", [])
        self._motion_lock_handles = []
        for handle in reversed(handles):
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            finally:
                handle.close()

    async def read_position_rad(self, check_errors: bool = False) -> np.ndarray:
        if not check_errors:
            values = await self._ctx.revo3_get_all_motor_positions(self._slave_id)
            return self._sdk_vector(values, "motor positions") * DEG_TO_RAD

        status = await self._ctx.revo3_get_motor_status_data(self._slave_id)
        online_mask = _enum_int(
            await self._ctx.revo3_get_motor_online_status(self._slave_id)
        )
        required_online_mask = (1 << JOINT_DIM) - 1
        missing_online = required_online_mask & ~online_mask
        if missing_online:
            missing = [index for index in range(JOINT_DIM) if missing_online & (1 << index)]
            raise RuntimeError(
                f"Motors are offline; command suppressed for motor IDs {missing}."
            )
        statuses = np.asarray(
            [_enum_int(value) for value in status.statuses], dtype=np.int64
        )
        errors = np.asarray([_enum_int(value) for value in status.errors], dtype=np.int64)
        if statuses.shape != (JOINT_DIM,) or errors.shape != (JOINT_DIM,):
            raise RuntimeError(
                f"SDK motor statuses and errors must each contain {JOINT_DIM} values."
            )
        # Bit 11 is the normal Running state. Stall is bit 8. By default every
        # non-running bit suppresses commands. Calibration tooling may explicitly
        # permit bit 8 on named motors while all other bits remain fatal.
        statuses_without_running = statuses & ~(1 << 11)
        errors_without_running = errors & ~(1 << 11)
        stalled = np.flatnonzero((statuses | errors) & STALL_BIT)
        self.last_stalled_motor_ids = tuple(int(index) for index in stalled)
        allowed_stall_ids = tuple(int(index) for index in self.config.allowed_stall_motor_ids)
        if len(set(allowed_stall_ids)) != len(allowed_stall_ids) or any(
            index < 0 or index >= JOINT_DIM for index in allowed_stall_ids
        ):
            raise RuntimeError("allowed_stall_motor_ids must be unique IDs in M0..M20.")
        grace_s = self.config.stall_grace_s
        if grace_s is not None and (not math.isfinite(grace_s) or grace_s <= 0.0):
            raise RuntimeError("stall_grace_s must be finite and positive when configured.")
        starts = getattr(self, "_stall_started_at_s", None)
        if starts is None or np.asarray(starts).shape != (JOINT_DIM,):
            starts = np.full(JOINT_DIM, np.nan, dtype=np.float64)
            self._stall_started_at_s = starts
        now = time.monotonic()
        stalled_mask = ((statuses | errors) & STALL_BIT) != 0
        starts[~stalled_mask] = np.nan
        newly_stalled = stalled_mask & np.isnan(starts)
        starts[newly_stalled] = now
        durations = np.where(stalled_mask, now - starts, 0.0)
        self.last_stall_durations_s = durations
        if allowed_stall_ids:
            allowed = np.asarray(allowed_stall_ids, dtype=np.int64)
            if grace_s is None:
                ignored = allowed
            else:
                ignored = allowed[
                    stalled_mask[allowed] & (durations[allowed] <= grace_s)
                ]
            statuses_without_running[ignored] &= ~STALL_BIT
            errors_without_running[ignored] &= ~STALL_BIT
        failed = np.flatnonzero(statuses_without_running | errors_without_running)
        if failed.size:
            details = ", ".join(
                f"M{index}(status=0x{statuses_without_running[index]:X},"
                f" error=0x{errors_without_running[index]:X})"
                for index in failed
            )
            raise RuntimeError(f"Motor fault reported; command suppressed: {details}.")
        currents = self._sdk_vector(status.currents, "motor currents")
        self.last_motor_currents_ma = currents
        current_limit = float(self.config.max_abs_current_ma)
        if not np.isfinite(current_limit) or current_limit <= 0.0:
            raise RuntimeError("max_abs_current_ma must be finite and positive.")
        over_current = np.flatnonzero(np.abs(currents) > current_limit)
        if over_current.size:
            details = ", ".join(
                f"M{index}={currents[index]:.1f}mA" for index in over_current
            )
            raise RuntimeError(
                f"Motor current exceeds {current_limit:.1f}mA; command suppressed: {details}."
            )
        return self._sdk_vector(status.positions, "motor positions") * DEG_TO_RAD

    async def read_fingertip_forces_n(
        self,
        enforce_freshness: bool = True,
    ) -> np.ndarray:
        if self.vision_touch is not None:
            forces, age = self.vision_touch.read_latest(
                enforce_freshness=enforce_freshness
            )
            self.last_tactile_age_s = age
            return self.force_adapter.from_force_vector(forces)
        if self.touch_vendor == 1:
            summary = await self._ctx.revo3_get_touch_summary(self._slave_id)
            return self.force_adapter.from_pressure_summary(summary)
        if self.touch_vendor == 2:
            modules: dict[int, list[float]] = {}
            for module_id in self.force_adapter.matrix_tip_module_ids:
                modules[module_id] = await self._ctx.revo3_get_touch_module_data(
                    self._slave_id, module_id
                )
            return self.force_adapter.from_matrix_modules(modules)
        raise RuntimeError("Tactile vendor has not been initialized.")

    def _is_vision_touch_hardware(self) -> bool:
        expected = getattr(self.sdk.StarkHardwareType, "Revo3UltraVisionTouch", None)
        return expected is not None and _enum_int(self.device_info.hardware_type) == _enum_int(
            expected
        )

    async def read_observation(
        self,
        check_motor_errors: bool = False,
        enforce_tactile_freshness: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        # Keep accesses sequential: both operations share one Modbus connection.
        position = await self.read_position_rad(check_errors=check_motor_errors)
        forces = await self.read_fingertip_forces_n(
            enforce_freshness=enforce_tactile_freshness
        )
        return position, forces

    async def send_mit_command_rad(
        self,
        position_rad: np.ndarray,
        velocity_rad_s: np.ndarray | None = None,
        kp: float | list[float] | np.ndarray = 1.0,
        kd: float | list[float] | np.ndarray = 0.1,
        effort_ma: float | list[float] | np.ndarray = 0.0,
    ) -> None:
        self.validate_device_position(position_rad)
        position_deg = self._sdk_vector(position_rad, "position_rad") * RAD_TO_DEG
        velocity = (
            np.zeros(JOINT_DIM, dtype=np.float32)
            if velocity_rad_s is None
            else self._sdk_vector(velocity_rad_s, "velocity_rad_s")
        )
        method_name = (
            "revo3_set_all_mit_params_without_retry"
            if self.config.use_without_retry
            else "revo3_set_all_mit_params"
        )
        method = getattr(self._ctx, method_name)
        await method(
            self._slave_id,
            self._command_values(kp, "kp"),
            self._command_values(kd, "kd"),
            position_deg.tolist(),
            (velocity * RAD_S_TO_RPM).tolist(),
            self._command_values(effort_ma, "effort_ma"),
        )

    async def release_mit(self) -> None:
        """Best-effort zero-force MIT frame; position is ignored because Kp is zero."""
        zeros = [0.0] * JOINT_DIM
        await self._ctx.revo3_set_all_mit_params_without_retry(
            self._slave_id,
            zeros,
            zeros,
            zeros,
            zeros,
            zeros,
        )

    async def _configure_tactile(self) -> None:
        await self._ctx.revo3_set_all_touch_modules_enabled(self._slave_id, 0x07FF)
        if self.touch_vendor == 1:
            await self._ctx.revo3_set_touch_data_type(
                self._slave_id, self.sdk.TouchDataMode.ForceSummary
            )
            await self._ctx.revo3_set_touch_module_value_type(
                self._slave_id, self.sdk.TouchModuleValueType.Force
            )
        elif self.touch_vendor == 2:
            await self._ctx.revo3_set_matrix_touch_output_mode(
                self._slave_id, self.sdk.MatrixTouchOutputMode.Force
            )

    async def _validate_tactile_mode(self) -> None:
        enabled = int(await self._ctx.revo3_get_all_touch_modules_enabled(self._slave_id))
        required_mask = sum(
            1 << module_id for module_id in self.force_adapter.matrix_tip_module_ids
        )
        if enabled & required_mask != required_mask:
            raise RuntimeError(
                f"Fingertip tactile modules are not all enabled (mask=0x{enabled:03X}); "
                "rerun with --configure-tactile to change it explicitly."
            )
        if self.touch_vendor == 1:
            data_type = await self._ctx.revo3_get_touch_data_type(self._slave_id)
            value_type = await self._ctx.revo3_get_touch_module_value_type(self._slave_id)
            if _enum_int(data_type) != _enum_int(self.sdk.TouchDataMode.ForceSummary):
                raise RuntimeError(
                    "Pressure tactile data mode is not ForceSummary; rerun with "
                    "--configure-tactile to change it explicitly."
                )
            if _enum_int(value_type) != _enum_int(self.sdk.TouchModuleValueType.Force):
                raise RuntimeError(
                    "Pressure tactile value type is not Force; rerun with "
                    "--configure-tactile to change it explicitly."
                )
        elif self.touch_vendor == 2:
            output_mode = await self._ctx.revo3_get_matrix_touch_output_mode(self._slave_id)
            if _enum_int(output_mode) != _enum_int(self.sdk.MatrixTouchOutputMode.Force):
                raise RuntimeError(
                    "Matrix tactile output mode is not Force; rerun with "
                    "--configure-tactile to change it explicitly."
                )
            invalid_modules = []
            for module_id in self.force_adapter.matrix_tip_module_ids:
                module_mode = await self._ctx.revo3_get_matrix_touch_module_output_mode(
                    self._slave_id, module_id
                )
                if _enum_int(module_mode) != _enum_int(
                    self.sdk.MatrixTouchOutputMode.Force
                ):
                    invalid_modules.append(module_id)
            if invalid_modules:
                raise RuntimeError(
                    "Matrix fingertip modules are not all in Force mode: "
                    f"{invalid_modules}; rerun with --configure-tactile."
                )

    def _validate_device_identity(self) -> None:
        info = self.device_info
        if info is None:
            raise RuntimeError("Device identity is unavailable.")
        expected_hand = self.config.expected_hand.lower()
        hand_enum = {
            "left": self.sdk.HandType.Left,
            "right": self.sdk.HandType.Right,
        }.get(expected_hand)
        if hand_enum is None:
            raise ValueError(f"Unsupported expected_hand: {self.config.expected_hand!r}.")
        if _enum_int(info.hand_type) != _enum_int(hand_enum):
            raise RuntimeError(
                f"Connected hand type {info.hand_type} does not match expected {expected_hand}."
            )

        allowed_hardware = []
        for name in self.config.allowed_hardware_types:
            value = getattr(self.sdk.StarkHardwareType, name, None)
            if value is None:
                raise ValueError(f"Unknown allowed Revo3 hardware type: {name}.")
            allowed_hardware.append(_enum_int(value))
        if _enum_int(info.hardware_type) not in allowed_hardware:
            raise RuntimeError(
                f"Hardware type {info.hardware_type} is not an allowed 21-DoF tactile Revo3."
            )

        allowlist = tuple(str(value).strip() for value in self.config.serial_allowlist)
        if any(not value for value in allowlist) or len(set(allowlist)) != len(allowlist):
            raise ValueError("serial_allowlist entries must be non-empty and unique.")
        serial = str(getattr(info, "serial_number", "") or "").strip()
        if allowlist and not serial:
            raise RuntimeError("Device serial is empty; identity binding cannot be verified.")
        if allowlist and serial not in allowlist:
            raise RuntimeError(f"Device serial {serial!r} is not in the profile allowlist.")

    def _set_device_position_limits(self, limits: Any) -> None:
        if not isinstance(limits, (tuple, list)) or len(limits) != 2:
            raise RuntimeError("SDK joint position limits must be a [lower, upper] pair.")
        lower_deg = self._sdk_vector(limits[0], "joint lower limits")
        upper_deg = self._sdk_vector(limits[1], "joint upper limits")
        if np.any(upper_deg <= lower_deg):
            raise RuntimeError("SDK device joint position limits are invalid.")
        self.device_position_lower_rad = lower_deg * DEG_TO_RAD
        self.device_position_upper_rad = upper_deg * DEG_TO_RAD

    def validate_device_position(self, position_rad: np.ndarray) -> None:
        position = self._sdk_vector(position_rad, "position_rad")
        lower = self.device_position_lower_rad
        upper = self.device_position_upper_rad
        if lower is None or upper is None:
            raise RuntimeError("SDK device joint position limits are unavailable.")
        invalid = np.flatnonzero((position < lower) | (position > upper))
        if invalid.size:
            details = ", ".join(
                f"M{index}={position[index] * RAD_TO_DEG:.2f}deg outside device "
                f"[{lower[index] * RAD_TO_DEG:.2f},{upper[index] * RAD_TO_DEG:.2f}]"
                for index in invalid
            )
            raise ValueError(f"Command violates device-reported position limits: {details}.")

    def _baudrate_enum(self, value: int):
        mapping = {
            1_000_000: self.sdk.Baudrate.Baud1Mbps,
            2_000_000: self.sdk.Baudrate.Baud2Mbps,
            3_000_000: self.sdk.Baudrate.Baud3Mbps,
            5_000_000: self.sdk.Baudrate.Baud5Mbps,
        }
        try:
            return mapping[value]
        except KeyError as exc:
            raise ValueError(f"Unsupported Revo3 Modbus baudrate: {value}.") from exc

    @property
    def _ctx(self):
        if self.ctx is None:
            raise RuntimeError("Revo3SdkHandIO is not open.")
        return self.ctx

    @property
    def _slave_id(self) -> int:
        if self.slave_id is None:
            raise RuntimeError("Revo3 slave ID is not initialized.")
        return self.slave_id

    @staticmethod
    def _sdk_vector(value: Any, name: str) -> np.ndarray:
        vector = np.asarray(value, dtype=np.float32).reshape(-1)
        if vector.shape != (JOINT_DIM,) or not np.isfinite(vector).all():
            raise RuntimeError(f"SDK {name} must contain {JOINT_DIM} finite values.")
        return vector

    @staticmethod
    def _command_values(value: Any, name: str) -> list[float]:
        vector = np.asarray(value, dtype=np.float32).reshape(-1)
        if vector.shape == (1,):
            vector = np.repeat(vector, JOINT_DIM)
        if vector.shape != (JOINT_DIM,) or not np.isfinite(vector).all():
            raise ValueError(f"{name} must be a finite scalar or {JOINT_DIM}-element vector.")
        return [float(item) for item in vector]


def _enum_int(value: Any) -> int:
    if hasattr(value, "int_value"):
        int_value = value.int_value
        return int(int_value() if callable(int_value) else int_value)
    return int(value)


def _baudrate_int(sdk: Any, value: Any) -> int:
    if isinstance(value, int):
        return value
    mapping = {
        sdk.Baudrate.Baud1Mbps: 1_000_000,
        sdk.Baudrate.Baud2Mbps: 2_000_000,
        sdk.Baudrate.Baud3Mbps: 3_000_000,
        sdk.Baudrate.Baud5Mbps: 5_000_000,
    }
    try:
        return mapping[value]
    except (KeyError, TypeError) as exc:
        raise RuntimeError(f"Unsupported detected baudrate: {value!r}.") from exc
