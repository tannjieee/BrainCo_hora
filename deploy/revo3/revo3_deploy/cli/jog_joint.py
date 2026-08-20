from __future__ import annotations

import argparse
import asyncio
import sys
import time

import numpy as np

from revo3_deploy.robot_profile import Revo3Profile
from revo3_deploy.sdk_hand_io import Revo3SdkConfig, Revo3SdkHandIO


MAX_JOG_DEG = 10.0
LOW_GAIN_KP = 0.3
LOW_GAIN_KD = 0.1
ABSOLUTE_MAX_JOG_KP = 2.0
ABSOLUTE_MAX_JOG_KD = 1.0
MAX_RATE_HZ = 20.0
MIN_RAMP_S = 1.0
MAX_JOG_SPEED_DEG_S = 2.0
POSITION_MARGIN_DEG = 0.5
PASSIVE_JOINT_MARGIN_DEG = 5.0
MAX_JOG_JOINTS = 4
MAX_TOTAL_SELECTED_CURRENT_MA = 1000.0
MAX_MEASURED_LIMIT_TOLERANCE_DEG = 1.0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Perform a bounded Revo3 joint calibration jog."
    )
    parser.add_argument("--profile", required=True)
    parser.add_argument(
        "--joint",
        required=True,
        action="append",
        help="SDK motor ID such as M13; repeat the flag for a synchronized multi-joint jog.",
    )
    parser.add_argument("--delta-deg", required=True, type=float)
    parser.add_argument("--kp", type=float, default=0.2)
    parser.add_argument("--kd", type=float, default=0.05)
    parser.add_argument("--rate", type=float, default=20.0)
    parser.add_argument("--ramp-s", type=float, default=1.0)
    parser.add_argument("--hold-s", type=float, default=0.25)
    parser.add_argument(
        "--measured-limit-tolerance-deg",
        type=float,
        default=None,
        help=(
            "Encoder-only tolerance at hardware limit; command targets remain strict "
            "(maximum 1 degree)."
        ),
    )
    parser.add_argument("--port", default=None)
    parser.add_argument("--baudrate", type=int, default=None)
    parser.add_argument("--slave-id", type=lambda value: int(value, 0), default=None)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm-fixed", action="store_true")
    parser.add_argument("--confirm-empty", action="store_true")
    parser.add_argument("--confirm-estop", action="store_true")
    parser.add_argument("--confirm-release", action="store_true")
    parser.add_argument(
        "--confirm-large-jog",
        action="store_true",
        help="Additional confirmation required when abs(delta) is greater than 1 degree.",
    )
    parser.add_argument(
        "--confirm-multi-joint",
        action="store_true",
        help="Additional confirmation required when more than one joint is selected.",
    )
    parser.add_argument(
        "--confirm-high-gain",
        action="store_true",
        help="Additional confirmation required above kp=0.3 or kd=0.1.",
    )
    parser.add_argument(
        "--allow-selected-stall",
        action="store_true",
        help=(
            "Allow only Stall bit 0x100 on selected motors; current, online, "
            "position, communication, and all other fault checks remain active."
        ),
    )
    return parser


async def async_main(args: argparse.Namespace) -> int:
    confirmations = (
        args.execute,
        args.confirm_fixed,
        args.confirm_empty,
        args.confirm_estop,
        args.confirm_release,
    )
    if not all(confirmations):
        raise RuntimeError(
            "Jog refused: --execute and all four physical-safety confirmations are required."
        )

    profile = Revo3Profile.load(args.profile)
    safety = profile.safety
    delta_deg = float(args.delta_deg)
    kp = float(args.kp)
    kd = float(args.kd)
    rate_hz = float(args.rate)
    ramp_s = float(args.ramp_s)
    hold_s = float(args.hold_s)
    values = np.asarray([delta_deg, kp, kd, rate_hz, ramp_s, hold_s], dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Jog parameters must be finite.")
    if delta_deg == 0.0 or abs(delta_deg) > MAX_JOG_DEG:
        raise ValueError(f"--delta-deg must be nonzero and at most {MAX_JOG_DEG:g} deg.")
    if abs(delta_deg) > 1.0 and not args.confirm_large_jog:
        raise RuntimeError(
            "Jog refused: movement above 1 degree requires --confirm-large-jog."
        )
    configured_max_kp = min(
        float(safety.get("max_kp", ABSOLUTE_MAX_JOG_KP)),
        ABSOLUTE_MAX_JOG_KP,
    )
    configured_max_kd = min(
        float(safety.get("max_kd", ABSOLUTE_MAX_JOG_KD)),
        ABSOLUTE_MAX_JOG_KD,
    )
    if not 0.0 < kp <= configured_max_kp or not 0.0 <= kd <= configured_max_kd:
        raise ValueError(
            f"Jog gains must satisfy 0 < kp <= {configured_max_kp:g} and "
            f"0 <= kd <= {configured_max_kd:g}."
        )
    if (kp > LOW_GAIN_KP or kd > LOW_GAIN_KD) and not args.confirm_high_gain:
        raise RuntimeError(
            "Jog refused: gains above kp=0.3 or kd=0.1 require --confirm-high-gain."
        )
    if not 0.0 < rate_hz <= MAX_RATE_HZ:
        raise ValueError(f"Jog rate must be in (0,{MAX_RATE_HZ:g}] Hz.")
    required_ramp_s = max(MIN_RAMP_S, abs(delta_deg) / MAX_JOG_SPEED_DEG_S)
    if ramp_s < required_ramp_s or not 0.0 <= hold_s <= 1.0:
        raise ValueError(
            f"Jog ramp must be at least {required_ramp_s:g} s to stay below "
            f"{MAX_JOG_SPEED_DEG_S:g} deg/s, and hold must be in [0,1] s."
        )

    joint_indices = tuple(
        _parse_motor_id(value, len(profile.sdk_joint_order)) for value in args.joint
    )
    if len(set(joint_indices)) != len(joint_indices):
        raise ValueError("--joint entries must be unique.")
    if len(joint_indices) > MAX_JOG_JOINTS:
        raise ValueError(f"At most {MAX_JOG_JOINTS} joints may be jogged together.")
    if len(joint_indices) > 1 and not args.confirm_multi_joint:
        raise RuntimeError(
            "Jog refused: synchronized movement requires --confirm-multi-joint."
        )
    joint_array = np.asarray(joint_indices, dtype=np.int64)
    max_current_ma = float(safety.get("jog_max_abs_current_ma", 500.0))
    if not np.isfinite(max_current_ma) or not 0.0 < max_current_ma <= 500.0:
        raise ValueError("Calibration jog current cutoff must be in (0,500] mA.")
    if args.measured_limit_tolerance_deg is None:
        measured_tolerance = float(
            safety.get("measured_position_tolerance_rad", np.deg2rad(0.5))
        )
    else:
        measured_tolerance_deg = float(args.measured_limit_tolerance_deg)
        if (
            not np.isfinite(measured_tolerance_deg)
            or measured_tolerance_deg < 0.0
            or measured_tolerance_deg > MAX_MEASURED_LIMIT_TOLERANCE_DEG
        ):
            raise ValueError(
                "--measured-limit-tolerance-deg must be finite and in [0,1]."
            )
        measured_tolerance = float(np.deg2rad(measured_tolerance_deg))
    connection_timeout = float(safety.get("connection_timeout_s", 15.0))
    release_timeout = float(safety.get("release_timeout_s", 0.25))
    close_timeout = float(safety.get("close_timeout_s", 0.5))
    sdk_cfg = profile.sdk
    configured_slave = sdk_cfg.get("slave_id")
    serial_allowlist = tuple(
        str(value).strip() for value in sdk_cfg.get("serial_allowlist") or ()
    )
    if not serial_allowlist:
        raise RuntimeError("Jog refused: sdk.serial_allowlist must bind the physical hand.")

    io = Revo3SdkHandIO(
        Revo3SdkConfig(
            port=args.port if args.port is not None else sdk_cfg.get("port"),
            baudrate=int(
                args.baudrate
                if args.baudrate is not None
                else sdk_cfg.get("baudrate", 5_000_000)
            ),
            slave_id=(
                args.slave_id
                if args.slave_id is not None
                else (int(configured_slave) if configured_slave is not None else None)
            ),
            auto_detect=bool(sdk_cfg.get("auto_detect", True)),
            configure_tactile=False,
            initialize_tactile=False,
            use_without_retry=True,
            expected_hand=str(sdk_cfg.get("expected_hand", profile.hand)),
            allowed_hardware_types=tuple(
                str(value) for value in sdk_cfg.get("allowed_hardware_types") or ()
            ),
            serial_allowlist=serial_allowlist,
            max_abs_current_ma=max_current_ma,
            allowed_stall_motor_ids=(joint_indices if args.allow_selected_stall else ()),
        ),
        tactile_cfg=profile.tactile,
    )

    command_may_have_been_sent = False
    cleanup_errors: list[BaseException] = []
    try:
        await asyncio.wait_for(io.open(), timeout=connection_timeout)
        start = await asyncio.wait_for(
            io.read_position_rad(check_errors=True),
            timeout=0.5,
        )
        profile.validate_sdk_position(
            start,
            "jog measured start",
            tolerance_rad=measured_tolerance,
        )
        if io.device_position_lower_rad is None or io.device_position_upper_rad is None:
            raise RuntimeError("Device position limits are unavailable.")
        command_base = np.clip(
            start,
            np.maximum(profile.sdk_position_lower_rad, io.device_position_lower_rad),
            np.minimum(profile.sdk_position_upper_rad, io.device_position_upper_rad),
        ).astype(np.float32)
        delta_rad = float(np.deg2rad(delta_deg))
        final_target = command_base.copy()
        final_target[joint_array] += delta_rad
        profile.validate_sdk_position(final_target, "jog final target")
        io.validate_device_position(final_target)

        kp_values = np.zeros(21, dtype=np.float32)
        kd_values = np.zeros(21, dtype=np.float32)
        kp_values[joint_array] = kp
        kd_values[joint_array] = kd
        offsets = _build_jog_offsets(delta_rad, ramp_s, hold_s, rate_hz)
        period = 1.0 / rate_hz
        envelope_lower = start[joint_array] + min(0.0, delta_rad) - np.deg2rad(
            POSITION_MARGIN_DEG
        )
        envelope_upper = start[joint_array] + max(0.0, delta_rad) + np.deg2rad(
            POSITION_MARGIN_DEG
        )
        passive_mask = np.ones(21, dtype=bool)
        passive_mask[joint_array] = False
        selected_total_current_limit = min(
            MAX_TOTAL_SELECTED_CURRENT_MA,
            max_current_ma * len(joint_indices),
        )

        print(
            "CALIBRATION JOG ENABLED: "
            f"motors={list(joint_indices)} "
            f"joints={[profile.sdk_joint_order[index] for index in joint_indices]} "
            f"start_deg={np.array2string(np.rad2deg(start[joint_array]), precision=3)} "
            f"delta={delta_deg:+.3f}deg kp={kp:.3f} kd={kd:.3f} "
            f"per_motor_cutoff={max_current_ma:.1f}mA "
            f"selected_total_cutoff={selected_total_current_limit:.1f}mA "
            f"allow_selected_stall={args.allow_selected_stall}"
        )
        for step_index, offset in enumerate(offsets, start=1):
            loop_start = time.monotonic()
            measured = await asyncio.wait_for(
                io.read_position_rad(check_errors=True),
                timeout=min(0.04, period * 0.8),
            )
            profile.validate_sdk_position(
                measured,
                "jog measured position",
                tolerance_rad=measured_tolerance,
            )
            actual = measured[joint_array]
            invalid_selected = np.flatnonzero(
                (actual < envelope_lower) | (actual > envelope_upper)
            )
            if invalid_selected.size:
                failed = [joint_indices[index] for index in invalid_selected]
                raise RuntimeError(
                    f"Selected motors left the jog envelope: {failed}; "
                    f"delta_deg={np.rad2deg(actual - start[joint_array])}."
                )
            passive_delta = np.abs(measured - start)
            passive_index = int(np.argmax(np.where(passive_mask, passive_delta, -1.0)))
            if np.rad2deg(passive_delta[passive_index]) > PASSIVE_JOINT_MARGIN_DEG:
                raise RuntimeError(
                    f"Passive M{passive_index} moved "
                    f"{np.rad2deg(passive_delta[passive_index]):.3f} deg; release requested."
                )
            selected_currents = np.abs(io.last_motor_currents_ma[joint_array])
            if float(np.sum(selected_currents)) > selected_total_current_limit:
                raise RuntimeError(
                    "Selected motor total current exceeded the calibration limit: "
                    f"{float(np.sum(selected_currents)):.1f} mA > "
                    f"{selected_total_current_limit:.1f} mA."
                )
            target = command_base.copy()
            target[joint_array] += offset
            profile.validate_sdk_position(target, "jog command target")
            io.validate_device_position(target)
            remaining = period - (time.monotonic() - loop_start)
            if remaining <= 0.0:
                raise RuntimeError("Jog frame expired before command; release requested.")
            command_may_have_been_sent = True
            await asyncio.wait_for(
                io.send_mit_command_rad(
                    target,
                    kp=kp_values,
                    kd=kd_values,
                    effort_ma=np.zeros(21, dtype=np.float32),
                ),
                timeout=remaining,
            )
            if step_index == 1 or step_index % 5 == 0 or step_index == len(offsets):
                print(
                    f"step={step_index}/{len(offsets)} "
                    f"target_delta_deg={np.rad2deg(offset):+.3f} "
                    f"measured_delta_deg="
                    f"{np.array2string(np.rad2deg(actual - start[joint_array]), precision=3)} "
                    f"current_mA="
                    f"{np.array2string(io.last_motor_currents_ma[joint_array], precision=1)} "
                    f"stall={list(io.last_stalled_motor_ids)}"
                )
            sleep_time = period - (time.monotonic() - loop_start)
            if sleep_time > 0.0:
                await asyncio.sleep(sleep_time)
    finally:
        if command_may_have_been_sent:
            try:
                await asyncio.wait_for(io.release_mit(), timeout=release_timeout)
                print("Zero-force MIT release sent.", file=sys.stderr)
            except BaseException as exc:
                cleanup_errors.append(exc)
                print(
                    f"WARNING: zero-force release failed ({exc!r}); use the hardware E-stop.",
                    file=sys.stderr,
                )
        try:
            await asyncio.wait_for(io.close(), timeout=close_timeout)
        except BaseException as exc:
            cleanup_errors.append(exc)
            print(f"WARNING: SDK close failed: {exc!r}", file=sys.stderr)
    if cleanup_errors:
        raise RuntimeError("Calibration jog cleanup failed; use the hardware E-stop.") from cleanup_errors[0]
    return 0


def _parse_motor_id(value: str, joint_count: int) -> int:
    text = str(value).strip().upper()
    if not text.startswith("M") or not text[1:].isdigit():
        raise ValueError("--joint must be an SDK motor ID such as M13.")
    index = int(text[1:])
    if not 0 <= index < joint_count:
        raise ValueError(f"Motor ID {text} is outside M0..M{joint_count - 1}.")
    return index


def _build_jog_offsets(
    delta_rad: float,
    ramp_s: float,
    hold_s: float,
    rate_hz: float,
) -> np.ndarray:
    ramp_steps = max(1, int(np.ceil(ramp_s * rate_hz)))
    hold_steps = max(0, int(np.ceil(hold_s * rate_hz)))
    outbound = np.linspace(0.0, delta_rad, ramp_steps + 1, dtype=np.float32)[1:]
    hold = np.full(hold_steps, delta_rad, dtype=np.float32)
    inbound = np.linspace(delta_rad, 0.0, ramp_steps + 1, dtype=np.float32)[1:]
    return np.concatenate((outbound, hold, inbound))


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return asyncio.run(async_main(args))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
