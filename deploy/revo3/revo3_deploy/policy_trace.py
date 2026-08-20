from __future__ import annotations

import copy
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .policy_runner import PolicyStep


JOINT_DIM = 21
CONTACT_DIM = 5
FRAME_DIM = 47
OBS_DIM = 141
HISTORY_LEN = 30


class PolicyTraceRecorder:
    """Collect policy-loop diagnostics in memory and atomically write one NPZ.

    No file I/O is performed by :meth:`append_frame` or the timing update methods.
    The control loop can therefore release the hand and close the SDK before the
    (comparatively expensive) compressed NPZ write happens.
    """

    _SCALAR_DTYPES: dict[str, np.dtype] = {
        "step_index": np.dtype(np.int64),
        "sample_time_s": np.dtype(np.float64),
        "read_ms": np.dtype(np.float64),
        "inference_ms": np.dtype(np.float64),
        "pre_send_ms": np.dtype(np.float64),
        "write_ms": np.dtype(np.float64),
        "loop_ms": np.dtype(np.float64),
        "tactile_age_ms": np.dtype(np.float64),
        "motor_status_valid": np.dtype(np.bool_),
        "command_sent": np.dtype(np.bool_),
        "command_completed": np.dtype(np.bool_),
    }
    _VECTOR_SPECS: dict[str, tuple[tuple[int, ...], np.dtype]] = {
        "sdk_pos_rad": ((JOINT_DIM,), np.dtype(np.float32)),
        "policy_pos_rad": ((JOINT_DIM,), np.dtype(np.float32)),
        "joint_pos_unscaled": ((JOINT_DIM,), np.dtype(np.float32)),
        "input_target_policy_rad": ((JOINT_DIM,), np.dtype(np.float32)),
        "force_n": ((CONTACT_DIM,), np.dtype(np.float32)),
        "frame_raw": ((FRAME_DIM,), np.dtype(np.float32)),
        "obs_raw": ((OBS_DIM,), np.dtype(np.float32)),
        "proprio_hist_raw": ((HISTORY_LEN, FRAME_DIM), np.dtype(np.float32)),
        "onnx_action_raw": ((JOINT_DIM,), np.dtype(np.float32)),
        "action": ((JOINT_DIM,), np.dtype(np.float32)),
        "policy_target_unclipped_rad": ((JOINT_DIM,), np.dtype(np.float32)),
        "policy_target_rad": ((JOINT_DIM,), np.dtype(np.float32)),
        "target_clipped": ((JOINT_DIM,), np.dtype(np.bool_)),
        "sdk_target_rad": ((JOINT_DIM,), np.dtype(np.float32)),
        "motor_current_ma": ((JOINT_DIM,), np.dtype(np.float32)),
        "stall_mask": ((JOINT_DIM,), np.dtype(np.bool_)),
        "stall_duration_s": ((JOINT_DIM,), np.dtype(np.float64)),
    }

    def __init__(self, path: str | Path, metadata: dict[str, Any]) -> None:
        trace_path = Path(path).expanduser().resolve()
        if trace_path.suffix.lower() != ".npz":
            raise ValueError("--trace-npz must end in .npz.")
        if trace_path.exists():
            if not trace_path.is_file():
                raise ValueError(f"Trace path is not a regular file: {trace_path}")
            raise FileExistsError(
                f"Refusing to overwrite an existing policy trace: {trace_path}"
            )
        trace_path.parent.mkdir(parents=True, exist_ok=True)

        self.path = trace_path
        self.metadata = copy.deepcopy(metadata)
        self.contact_force_scale = float(self.metadata.get("contact_force_scale", 1.0))
        if not np.isfinite(self.contact_force_scale) or self.contact_force_scale <= 0.0:
            raise ValueError("trace contact_force_scale must be finite and positive")
        self.started_monotonic_s = time.monotonic()
        self._scalars: dict[str, list[Any]] = {
            name: [] for name in self._SCALAR_DTYPES
        }
        self._vectors: dict[str, list[np.ndarray]] = {
            name: [] for name in self._VECTOR_SPECS
        }
        self._loop_started_monotonic_s: list[float] = []
        self._sample_origin_monotonic_s: float | None = None
        self._saved = False

    @property
    def frame_count(self) -> int:
        return len(self._scalars["step_index"])

    def update_metadata(self, **values: Any) -> None:
        self.metadata.update(copy.deepcopy(values))

    def append_frame(
        self,
        *,
        step_index: int,
        loop_started_monotonic_s: float,
        sdk_pos_rad: np.ndarray,
        policy_pos_rad: np.ndarray,
        force_n: np.ndarray,
        result: PolicyStep,
        sdk_target_rad: np.ndarray,
        read_ms: float,
        inference_ms: float,
        tactile_age_ms: float | None,
        motor_current_ma: np.ndarray | None,
        stalled_motor_ids: tuple[int, ...],
        stall_duration_s: np.ndarray | None,
        motor_status_valid: bool,
    ) -> int:
        obs = self._network_input(result.obs_raw, (1, OBS_DIM), "obs_raw")[0]
        history = self._network_input(
            result.proprio_hist_raw,
            (1, HISTORY_LEN, FRAME_DIM),
            "proprio_hist_raw",
        )[0]
        frame = history[-1]
        if not np.array_equal(obs, history[-3:].reshape(OBS_DIM)):
            raise RuntimeError("Trace refused: obs_raw is not the last three history frames.")

        sdk_pos = self._vector(sdk_pos_rad, JOINT_DIM, "sdk_pos_rad")
        policy_pos = self._vector(policy_pos_rad, JOINT_DIM, "policy_pos_rad")
        measured_forces = self._vector(force_n, CONTACT_DIM, "force_n")
        forces = (measured_forces * self.contact_force_scale).astype(np.float32)
        if not np.array_equal(forces, frame[42:47]):
            raise RuntimeError("Trace refused: force_n differs from the actual network input.")

        onnx_action_raw = self._vector(
            result.onnx_action_raw,
            JOINT_DIM,
            "onnx_action_raw",
        )
        action = self._vector(result.action, JOINT_DIM, "action")
        policy_target_unclipped = self._vector(
            result.policy_target_unclipped_rad,
            JOINT_DIM,
            "policy_target_unclipped_rad",
        )
        policy_target = self._vector(
            result.policy_target_rad,
            JOINT_DIM,
            "policy_target_rad",
        )
        target_clipped = self._vector(
            result.target_clipped,
            JOINT_DIM,
            "target_clipped",
            dtype=np.bool_,
        )
        sdk_target = self._vector(sdk_target_rad, JOINT_DIM, "sdk_target_rad")
        currents = (
            np.full(JOINT_DIM, np.nan, dtype=np.float32)
            if motor_current_ma is None
            else self._vector(motor_current_ma, JOINT_DIM, "motor_current_ma")
        )
        stall_mask = np.zeros(JOINT_DIM, dtype=np.bool_)
        for motor_id in stalled_motor_ids:
            index = int(motor_id)
            if not 0 <= index < JOINT_DIM:
                raise RuntimeError(f"Trace received invalid stalled motor ID {motor_id}.")
            stall_mask[index] = True
        stall_durations = (
            np.full(JOINT_DIM, np.nan, dtype=np.float64)
            if stall_duration_s is None
            else self._vector(
                stall_duration_s,
                JOINT_DIM,
                "stall_duration_s",
                dtype=np.float64,
            )
        )

        loop_start = float(loop_started_monotonic_s)
        if not np.isfinite(loop_start):
            raise ValueError("loop_started_monotonic_s must be finite.")
        if self._sample_origin_monotonic_s is None:
            self._sample_origin_monotonic_s = loop_start
        relative_time = loop_start - self._sample_origin_monotonic_s
        scalar_values = {
            "step_index": int(step_index),
            "sample_time_s": relative_time,
            "read_ms": self._finite_nonnegative(read_ms, "read_ms"),
            "inference_ms": self._finite_nonnegative(inference_ms, "inference_ms"),
            "pre_send_ms": np.nan,
            "write_ms": np.nan,
            "loop_ms": np.nan,
            "tactile_age_ms": (
                np.nan
                if tactile_age_ms is None
                else self._finite_nonnegative(tactile_age_ms, "tactile_age_ms")
            ),
            "motor_status_valid": bool(motor_status_valid),
            # command_sent means that the SDK send call was entered. On a timeout,
            # the command may have reached the device, so it is set before awaiting.
            "command_sent": False,
            "command_completed": False,
        }
        vector_values = {
            "sdk_pos_rad": sdk_pos,
            "policy_pos_rad": policy_pos,
            "joint_pos_unscaled": frame[:21],
            "input_target_policy_rad": frame[21:42],
            "force_n": forces,
            "frame_raw": frame,
            "obs_raw": obs,
            "proprio_hist_raw": history,
            "onnx_action_raw": onnx_action_raw,
            "action": action,
            "policy_target_unclipped_rad": policy_target_unclipped,
            "policy_target_rad": policy_target,
            "target_clipped": target_clipped,
            "sdk_target_rad": sdk_target,
            "motor_current_ma": currents,
            "stall_mask": stall_mask,
            "stall_duration_s": stall_durations,
        }

        row = self.frame_count
        for name, value in scalar_values.items():
            self._scalars[name].append(value)
        for name, value in vector_values.items():
            self._vectors[name].append(np.asarray(value).copy())
        self._loop_started_monotonic_s.append(loop_start)
        return row

    def mark_command_sent(self, row: int, *, pre_send_ms: float) -> None:
        self._validate_row(row)
        self._scalars["pre_send_ms"][row] = self._finite_nonnegative(
            pre_send_ms,
            "pre_send_ms",
        )
        self._scalars["command_sent"][row] = True

    def mark_command_completed(self, row: int, *, write_ms: float) -> None:
        self._validate_row(row)
        self._scalars["write_ms"][row] = self._finite_nonnegative(write_ms, "write_ms")
        self._scalars["command_completed"][row] = True

    def finish_frame(self, row: int, *, loop_ms: float) -> None:
        self._validate_row(row)
        self._scalars["loop_ms"][row] = self._finite_nonnegative(loop_ms, "loop_ms")

    def finish_pending_frames(self) -> None:
        """Freeze incomplete frame timing before hardware cleanup starts."""
        self._finalize_open_frames()

    def save(
        self,
        *,
        termination_status: str,
        error: BaseException | None = None,
    ) -> Path:
        if self._saved:
            return self.path
        self._finalize_open_frames()

        metadata = copy.deepcopy(self.metadata)
        metadata.update(
            {
                "termination": {
                    "status": str(termination_status),
                    "error_type": type(error).__name__ if error is not None else None,
                    "error_message": str(error) if error is not None else None,
                },
                "frame_count": self.frame_count,
                "command_sent_frame_count": int(
                    np.count_nonzero(self._scalars["command_sent"])
                ),
                "saved_at_utc": datetime.now(timezone.utc).isoformat(),
                "array_contract": self.array_contract(),
            }
        )
        arrays: dict[str, np.ndarray] = {
            "metadata_json": np.asarray(
                json.dumps(metadata, ensure_ascii=False, sort_keys=True),
                dtype=np.str_,
            )
        }
        for name, dtype in self._SCALAR_DTYPES.items():
            arrays[name] = np.asarray(self._scalars[name], dtype=dtype)
        for name, (tail_shape, dtype) in self._VECTOR_SPECS.items():
            values = self._vectors[name]
            arrays[name] = (
                np.stack(values, axis=0).astype(dtype, copy=False)
                if values
                else np.empty((0, *tail_shape), dtype=dtype)
            )

        temporary = self.path.with_name(
            f".{self.path.name}.{os.getpid()}.partial.npz"
        )
        try:
            np.savez_compressed(temporary, **arrays)
            os.replace(temporary, self.path)
        finally:
            if temporary.exists():
                temporary.unlink()
        self._saved = True
        return self.path

    @classmethod
    def array_contract(cls) -> dict[str, Any]:
        units = {
            "step_index": "zero-based policy step within this run",
            "sample_time_s": "seconds from first recorded frame (monotonic clock)",
            "sdk_pos_rad": "radians, SDK M0..M20 order",
            "policy_pos_rad": "radians, policy joint order",
            "joint_pos_unscaled": "dimensionless, exact network frame slice [0:21]",
            "input_target_policy_rad": "radians, pre-action network frame slice [21:42]",
            "force_n": "scaled newtons in policy contact order; exact network frame slice [42:47]",
            "frame_raw": "exact pre-normalization 47-value network frame",
            "obs_raw": "exact pre-normalization ONNX obs input without batch dimension",
            "proprio_hist_raw": (
                "exact pre-normalization ONNX proprio_hist input without batch dimension"
            ),
            "onnx_action_raw": "exact finite ONNX output before deployment safety clipping",
            "action": "deployed ONNX output clipped to [-1,1]",
            "policy_target_unclipped_rad": (
                "input target plus action_scale times deployed action, before target limits"
            ),
            "policy_target_rad": "post-action integrated/clipped target, policy order",
            "target_clipped": "true where target limits clipped the integrated target",
            "sdk_target_rad": "post-transform command target, SDK M0..M20 order",
            "motor_current_ma": "SDK motor current in runtime-assumed milliamperes",
            "motor_status_valid": "true when current/stall came from this frame's status read",
            "stall_mask": "true where the SDK Stall bit was present for an SDK motor",
            "stall_duration_s": "continuous Stall-bit duration; NaN when status was not read",
            "read_ms": "loop start through observation completion",
            "inference_ms": "runner.step wall time",
            "pre_send_ms": "loop start through entry into SDK send; NaN when not sent",
            "write_ms": "SDK send await duration; NaN when not completed",
            "loop_ms": "loop start through frame completion/error cleanup snapshot",
            "tactile_age_ms": "VisionTouch sample age; NaN when unavailable",
            "command_sent": "SDK send call entered; the command may be in flight on timeout",
            "command_completed": "SDK send await returned normally",
        }
        contract: dict[str, Any] = {}
        for name, dtype in cls._SCALAR_DTYPES.items():
            contract[name] = {
                "shape": ["T"],
                "dtype": dtype.name,
                "units_or_semantics": units.get(name, "see field name"),
            }
        for name, (tail_shape, dtype) in cls._VECTOR_SPECS.items():
            contract[name] = {
                "shape": ["T", *tail_shape],
                "dtype": dtype.name,
                "units_or_semantics": units.get(name, "see field name"),
            }
        return contract

    def _finalize_open_frames(self) -> None:
        now = time.monotonic()
        for row, value in enumerate(self._scalars["loop_ms"]):
            if not np.isfinite(value):
                elapsed_ms = max(
                    0.0,
                    (now - self._loop_started_monotonic_s[row]) * 1000.0,
                )
                self._scalars["loop_ms"][row] = elapsed_ms

    def _validate_row(self, row: int) -> None:
        if not 0 <= int(row) < self.frame_count:
            raise IndexError(f"Trace row {row} is outside 0..{self.frame_count - 1}.")

    @staticmethod
    def _network_input(value: np.ndarray, shape: tuple[int, ...], name: str) -> np.ndarray:
        array = np.asarray(value)
        if array.shape != shape or array.dtype != np.float32 or not np.isfinite(array).all():
            raise ValueError(
                f"{name} must be finite float32 with shape {shape}, got {array.shape}."
            )
        return array

    @staticmethod
    def _vector(
        value: np.ndarray,
        size: int,
        name: str,
        *,
        dtype: np.dtype | type = np.float32,
    ) -> np.ndarray:
        vector = np.asarray(value, dtype=dtype).reshape(-1)
        if vector.shape != (size,) or not np.isfinite(vector).all():
            raise ValueError(f"{name} must contain {size} finite values.")
        return vector

    @staticmethod
    def _finite_nonnegative(value: float, name: str) -> float:
        number = float(value)
        if not np.isfinite(number) or number < 0.0:
            raise ValueError(f"{name} must be finite and non-negative.")
        return number
