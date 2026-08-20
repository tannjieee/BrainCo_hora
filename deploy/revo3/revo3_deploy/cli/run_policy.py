from __future__ import annotations

import argparse
import asyncio
import hashlib
import sys
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from revo3_deploy.policy_runner import Revo3PolicyRunner
from revo3_deploy.policy_trace import PolicyTraceRecorder
from revo3_deploy.robot_profile import Revo3Profile
from revo3_deploy.sdk_hand_io import Revo3SdkConfig, Revo3SdkHandIO


MAX_STALL_GRACE_S = 1.0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the tactile HORA ONNX policy through bc-revo3-sdk 1.5.1."
    )
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--port", default=None)
    parser.add_argument("--baudrate", type=int, default=None)
    parser.add_argument("--slave-id", type=lambda value: int(value, 0), default=None)
    parser.add_argument("--rate", type=float, default=None)
    parser.add_argument(
        "--allow-rate-override",
        action="store_true",
        help="Allow a non-contract rate for deliberate motor dry-run benchmarking only.",
    )
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--print-every", type=int, default=20)
    parser.add_argument("--provider", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--trace-npz",
        default=None,
        metavar="PATH",
        help=(
            "Record exact raw ONNX inputs/outputs, transformed joint state, tactile input, "
            "hardware status, and loop timing to one .npz. Data stays in memory until "
            "motor release and SDK close. Supported in dry-run, preflight, and motion modes; "
            "non-preflight capture requires a finite --steps value."
        ),
    )
    parser.add_argument(
        "--policy-start-delay-s",
        type=float,
        default=0.0,
        help=(
            "After SDK/VisionTouch open and validation, wait 0..120 seconds before the "
            "first policy observation so the cylinder can be placed. Dry-run only; no "
            "motor command is sent during the delay."
        ),
    )
    parser.add_argument("--kp", type=float, default=None)
    parser.add_argument("--kd", type=float, default=None)
    parser.add_argument("--effort-ma", type=float, default=None)
    parser.add_argument(
        "--enable-motion",
        action="store_true",
        help="Actually send MIT commands. Without this flag no motor command is sent.",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help=(
            "Read hardware health and print the first policy target/delta, but never send "
            "a motor command."
        ),
    )
    parser.add_argument(
        "--preflight-position-tolerance-deg",
        type=float,
        default=None,
        help=(
            "Read-only measured-position tolerance for --preflight-only (maximum 5 deg); "
            "never changes command limits."
        ),
    )
    parser.add_argument(
        "--allow-unverified-calibration",
        action="store_true",
        help="Allow motion while profile calibration.status is not verified.",
    )
    parser.add_argument(
        "--allow-stall",
        action="append",
        default=[],
        metavar="MOTOR",
        help=(
            "Allow only Stall bit 0x100 for one named motor such as M13; repeat for each "
            "confirmed false-positive motor. All other checks remain active."
        ),
    )
    parser.add_argument(
        "--stall-grace-s",
        type=float,
        default=0.0,
        help=(
            "Allow Stall 0x100 on all motors only while continuously present for no more "
            "than this duration; maximum 1.0 s. Other fault bits remain immediate."
        ),
    )
    parser.add_argument(
        "--ignore-all-stall",
        action="store_true",
        help=(
            "Ignore Stall bit 0x100 on all 21 motors for an explicitly authorized test; "
            "all other fault and safety checks remain active."
        ),
    )
    parser.add_argument(
        "--configure-tactile",
        action="store_true",
        help="Explicitly enable tactile modules and switch them to force mode.",
    )
    parser.add_argument(
        "--preposition-cache",
        default=None,
        metavar="NPY",
        help=(
            "Before policy motion, slowly move to one 21-DoF policy-order row from this "
            ".npy cache using the hand's absolute joint positions."
        ),
    )
    parser.add_argument(
        "--preposition-row",
        type=int,
        default=0,
        help="Zero-based cache row used by --preposition-cache (default: 0).",
    )
    parser.add_argument(
        "--preposition-speed-deg-s",
        type=float,
        default=2.0,
        help="Maximum per-joint preposition speed; must be in (0,2] deg/s.",
    )
    parser.add_argument(
        "--confirm-preposition",
        action="store_true",
        help="Confirm that the cache row and clear path were checked before motor motion.",
    )
    return parser


async def async_main(args: argparse.Namespace) -> int:
    policy_start_delay_s = float(args.policy_start_delay_s)
    if (
        not np.isfinite(policy_start_delay_s)
        or policy_start_delay_s < 0.0
        or policy_start_delay_s > 120.0
    ):
        raise ValueError("--policy-start-delay-s must be finite and in [0,120].")
    if policy_start_delay_s > 0.0 and (args.enable_motion or args.preflight_only):
        raise ValueError(
            "A nonzero --policy-start-delay-s is only allowed in motor dry-run mode."
        )
    if args.preposition_cache is not None:
        if not args.enable_motion or not args.confirm_preposition:
            raise ValueError(
                "--preposition-cache requires --enable-motion and --confirm-preposition."
            )
        if args.preflight_only:
            raise ValueError("--preposition-cache cannot be combined with --preflight-only.")
        if args.preposition_row < 0:
            raise ValueError("--preposition-row must be non-negative.")
        preposition_speed_deg_s = float(args.preposition_speed_deg_s)
        if (
            not np.isfinite(preposition_speed_deg_s)
            or preposition_speed_deg_s <= 0.0
            or preposition_speed_deg_s > 2.0
        ):
            raise ValueError("--preposition-speed-deg-s must be finite and in (0,2].")
    else:
        if args.confirm_preposition:
            raise ValueError("--confirm-preposition requires --preposition-cache.")
        preposition_speed_deg_s = 2.0
    if args.preflight_only and (args.enable_motion or args.configure_tactile):
        raise ValueError(
            "--preflight-only cannot be combined with --enable-motion or "
            "--configure-tactile."
        )
    if args.trace_npz is not None and args.steps is None and not args.preflight_only:
        raise ValueError(
            "--trace-npz requires a finite --steps value outside --preflight-only; "
            "the recorder intentionally keeps control-loop data in memory until cleanup."
        )
    if args.preflight_position_tolerance_deg is not None and not args.preflight_only:
        raise ValueError(
            "--preflight-position-tolerance-deg is only valid with --preflight-only."
        )
    preflight_tolerance_rad: float | None = None
    if args.preflight_position_tolerance_deg is not None:
        preflight_tolerance_deg = float(args.preflight_position_tolerance_deg)
        if (
            not np.isfinite(preflight_tolerance_deg)
            or preflight_tolerance_deg < 0.0
            or preflight_tolerance_deg > 5.0
        ):
            raise ValueError(
                "--preflight-position-tolerance-deg must be finite and in [0,5]."
            )
        preflight_tolerance_rad = float(np.deg2rad(preflight_tolerance_deg))
    if args.allow_stall and not args.enable_motion:
        raise ValueError("--allow-stall is only valid together with --enable-motion.")
    if args.ignore_all_stall and not args.enable_motion:
        raise ValueError("--ignore-all-stall is only valid together with --enable-motion.")
    stall_grace_s = float(args.stall_grace_s)
    if not np.isfinite(stall_grace_s) or not 0.0 <= stall_grace_s <= MAX_STALL_GRACE_S:
        raise ValueError("--stall-grace-s must be finite and in [0,1].")
    if stall_grace_s > 0.0 and not args.enable_motion:
        raise ValueError("--stall-grace-s is only valid together with --enable-motion.")
    if sum(bool(value) for value in (stall_grace_s > 0.0, args.allow_stall, args.ignore_all_stall)) > 1:
        raise ValueError(
            "Use only one of --stall-grace-s, per-motor --allow-stall, or "
            "--ignore-all-stall."
        )
    allowed_stall_motor_ids = tuple(_parse_motor_id(value) for value in args.allow_stall)
    if len(set(allowed_stall_motor_ids)) != len(allowed_stall_motor_ids):
        raise ValueError("--allow-stall motor IDs must be unique.")
    if stall_grace_s > 0.0 or args.ignore_all_stall:
        allowed_stall_motor_ids = tuple(range(21))
    runner = Revo3PolicyRunner.create(
        args.onnx,
        args.metadata,
        args.profile,
        provider=args.provider,
    )
    profile = runner.profile
    preposition_sdk_target = (
        _load_preposition_sdk_target(
            args.preposition_cache,
            args.preposition_row,
            profile,
        )
        if args.preposition_cache is not None
        else None
    )
    vision_cfg = dict(profile.tactile.get("vision_touch") or {})
    if args.enable_motion and bool(vision_cfg.get("enabled", False)):
        if not bool(vision_cfg.get("mapping_verified", False)):
            raise RuntimeError(
                "Motion refused: VisionTouch sensor_order is not marked mapping_verified. "
                "Press each fingertip once and confirm the five SN-to-finger mapping first."
            )
    if args.enable_motion and profile.calibration_status != "verified":
        if not args.allow_unverified_calibration:
            raise RuntimeError(
                "Motion refused: profile calibration.status is unverified. Check joint offsets, "
                "tactile calibration, and MIT gains first; then mark it verified or explicitly "
                "pass --allow-unverified-calibration."
            )

    sdk_cfg = profile.sdk
    serial_allowlist = tuple(
        str(value).strip() for value in sdk_cfg.get("serial_allowlist") or ()
    )
    if any(not value for value in serial_allowlist) or len(set(serial_allowlist)) != len(
        serial_allowlist
    ):
        raise ValueError("sdk.serial_allowlist entries must be non-empty and unique.")
    if args.enable_motion and not serial_allowlist:
        raise RuntimeError(
            "Motion refused: sdk.serial_allowlist is empty. Run the motor dry-run, verify the "
            "connected hand serial, and bind that serial in the profile first."
        )

    safety = profile.safety
    max_abs_current_ma = _positive(safety, "max_abs_current_ma", 500.0)
    configured_slave = sdk_cfg.get("slave_id")
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
            # Tactile setters are only authorized by this explicit CLI flag.
            configure_tactile=args.configure_tactile,
            use_without_retry=bool(profile.mit.get("without_retry", True)),
            expected_hand=str(sdk_cfg.get("expected_hand", profile.hand)),
            allowed_hardware_types=tuple(
                str(value) for value in sdk_cfg.get("allowed_hardware_types") or ()
            ),
            serial_allowlist=serial_allowlist,
            max_abs_current_ma=max_abs_current_ma,
            allowed_stall_motor_ids=allowed_stall_motor_ids,
            stall_grace_s=(stall_grace_s if stall_grace_s > 0.0 else None),
        ),
        tactile_cfg=profile.tactile,
    )

    mit = profile.mit
    kp = float(args.kp if args.kp is not None else mit.get("kp", 1.0))
    kd = float(args.kd if args.kd is not None else mit.get("kd", 0.1))
    effort_ma = float(
        args.effort_ma if args.effort_ma is not None else mit.get("effort_ma", 0.0)
    )
    rate_hz = float(args.rate if args.rate is not None else runner.rate_hz)
    if not np.isfinite(rate_hz) or rate_hz <= 0.0:
        raise ValueError("Policy rate must be finite and positive.")
    if not np.isclose(rate_hz, runner.rate_hz) and not args.allow_rate_override:
        raise RuntimeError(
            f"Requested {rate_hz:g} Hz differs from the {runner.rate_hz:g} Hz policy contract; "
            "pass --allow-rate-override only for deliberate dry-run benchmarking."
        )
    if args.enable_motion and not np.isclose(rate_hz, runner.rate_hz):
        raise RuntimeError(
            "Motion mode must use the policy contract rate; override is dry-run only."
        )
    if args.steps is not None and args.steps <= 0:
        raise ValueError("--steps must be positive when provided.")
    if args.print_every <= 0:
        raise ValueError("--print-every must be positive.")
    _validate_mit_settings(kp, kd, effort_ma, safety)

    period = 1.0 / rate_hz
    max_observation_age = _positive(safety, "max_observation_age_s", 0.045)
    diagnostic_io_timeout = _positive(safety, "diagnostic_io_timeout_s", 2.0)
    measured_position_tolerance = _positive(
        safety,
        "measured_position_tolerance_rad",
        0.00872665,
    )
    max_tracking_error = _positive(safety, "max_tracking_error_rad", 0.4363323)
    max_tracking_samples = int(safety.get("max_tracking_error_samples", 3))
    if max_tracking_samples <= 0:
        raise ValueError("safety.max_tracking_error_samples must be positive.")
    release_timeout = _positive(safety, "release_timeout_s", 0.25)
    close_timeout = _positive(safety, "close_timeout_s", 0.5)
    connection_timeout = _positive(safety, "connection_timeout_s", 15.0)
    initial_delta_limit = _positive(
        safety, "max_initial_command_delta_rad", 0.0872665
    )
    command_step_limit = _positive(safety, "max_command_step_rad", 0.05)
    device_target_margin = _positive(
        safety, "device_target_margin_rad", np.deg2rad(0.05)
    )

    trace_recorder = (
        PolicyTraceRecorder(
            args.trace_npz,
            _build_trace_metadata(
                args=args,
                runner=runner,
                rate_hz=rate_hz,
                kp=kp,
                kd=kd,
                effort_ma=effort_ma,
            ),
        )
        if args.trace_npz is not None
        else None
    )

    try:
        await asyncio.wait_for(io.open(), timeout=connection_timeout)
    except asyncio.TimeoutError as exc:
        error = RuntimeError(
            f"SDK connection/preflight exceeded {connection_timeout:.2f} s."
        )
        _save_trace_best_effort(trace_recorder, termination_status="error", error=error)
        raise error from exc
    except BaseException as exc:
        _save_trace_best_effort(trace_recorder, termination_status="error", error=exc)
        raise
    try:
        if io.device_position_lower_rad is None or io.device_position_upper_rad is None:
            raise RuntimeError("Connected device did not report usable joint position limits.")
        runner.apply_device_target_limits(
            io.device_position_lower_rad,
            io.device_position_upper_rad,
            margin_rad=device_target_margin,
        )
        if trace_recorder is not None:
            info = io.device_info
            trace_recorder.update_metadata(
                device={
                    "port": io.port,
                    "baudrate": io.baudrate,
                    "slave_id": io.slave_id,
                    "serial_number": str(getattr(info, "serial_number", "")),
                    "hand_type": str(getattr(info, "hand_type", "")),
                    "hardware_type": str(getattr(info, "hardware_type", "")),
                    "touch_vendor": io.touch_vendor,
                    "tactile_source": io.tactile_source,
                    "position_lower_sdk_rad": io.device_position_lower_rad.tolist(),
                    "position_upper_sdk_rad": io.device_position_upper_rad.tolist(),
                },
                effective_target_limits_policy_rad={
                    "lower": runner.builder.target_lower.tolist(),
                    "upper": runner.builder.target_upper.tolist(),
                },
            )
    except BaseException as setup_error:
        try:
            await asyncio.wait_for(io.close(), timeout=close_timeout)
        except BaseException as close_error:
            print(
                f"WARNING: SDK close after setup failure failed: {close_error!r}",
                file=sys.stderr,
            )
        _save_trace_best_effort(
            trace_recorder,
            termination_status="error",
            error=setup_error,
        )
        raise
    if args.preflight_only:
        preflight_error: BaseException | None = None
        preflight_run_error: BaseException | None = None
        try:
            await _run_read_only_preflight(
                io=io,
                runner=runner,
                profile=profile,
                timeout_s=diagnostic_io_timeout,
                initial_delta_limit_rad=initial_delta_limit,
                measured_position_tolerance_rad=(
                    preflight_tolerance_rad
                    if preflight_tolerance_rad is not None
                    else measured_position_tolerance
                ),
                trace_recorder=trace_recorder,
            )
        except BaseException as exc:
            preflight_run_error = exc
            raise
        finally:
            try:
                await asyncio.wait_for(io.close(), timeout=close_timeout)
            except BaseException as exc:
                preflight_error = exc
                print(f"WARNING: SDK close failed: {exc!r}", file=sys.stderr)
            trace_error = preflight_run_error or preflight_error
            _save_trace_best_effort(
                trace_recorder,
                termination_status="completed" if trace_error is None else "error",
                error=trace_error,
            )
        if preflight_error is not None:
            if isinstance(
                preflight_error,
                (asyncio.CancelledError, KeyboardInterrupt, SystemExit),
            ):
                raise preflight_error
            raise RuntimeError(
                "Hardware preflight finished, but SDK cleanup failed."
            ) from preflight_error
        return 0
    iteration = 0
    tracking_error_samples = 0
    previous_sdk_target: np.ndarray | None = None
    command_may_have_been_sent = False
    cleanup_errors: list[BaseException] = []
    diagnostic_stale_force_samples = 0
    diagnostic_max_force_age_s = 0.0
    run_error: BaseException | None = None
    try:
        if args.enable_motion and io.touch_vendor == 2:
            raise RuntimeError(
                "Matrix tactile motion is disabled: five dense module reads have not been shown "
                "to meet the 50 ms deadline. Benchmark it in motor dry-run mode first."
            )
        mode = "MOTION ENABLED" if args.enable_motion else "MOTOR DRY-RUN"
        info = io.device_info
        print(
            f"Connected port={io.port} slave_id={io.slave_id} "
            f"serial={getattr(info, 'serial_number', '')} hand={getattr(info, 'hand_type', '')} "
            f"hardware={getattr(info, 'hardware_type', '')} tactile_vendor={io.touch_vendor}; "
            f"tactile_source={io.tactile_source}; policy={rate_hz:.2f} Hz; mode={mode}; "
            f"allowed_stall_motors={list(allowed_stall_motor_ids)}; "
            f"stall_grace_s={stall_grace_s:.3f}; "
            f"ignore_all_stall={args.ignore_all_stall}"
        )
        print("Tactile order: thumb, index, middle, ring, little [N]")
        if policy_start_delay_s > 0.0:
            print(
                "=" * 72
                + f"\nPOLICY START DELAY {policy_start_delay_s:.1f}s: "
                "无电机命令，可在此期间放置圆柱。\n"
                + "=" * 72,
                flush=True,
            )
            await asyncio.sleep(policy_start_delay_s)
            print("Policy start delay complete; taking the first observation.", flush=True)

        if preposition_sdk_target is not None:
            profile.validate_sdk_position(
                preposition_sdk_target,
                "preposition SDK target",
            )
            io.validate_device_position(preposition_sdk_target)

            def mark_preposition_command_attempted() -> None:
                nonlocal command_may_have_been_sent
                command_may_have_been_sent = True

            await _run_preposition(
                io=io,
                profile=profile,
                sdk_target=preposition_sdk_target,
                rate_hz=rate_hz,
                speed_deg_s=preposition_speed_deg_s,
                kp=kp,
                kd=kd,
                effort_ma=effort_ma,
                read_timeout_s=max_observation_age,
                max_tracking_error_rad=max_tracking_error,
                max_tracking_samples=max_tracking_samples,
                measured_position_tolerance_rad=measured_position_tolerance,
                device_target_margin_rad=device_target_margin,
                mark_command_attempted=mark_preposition_command_attempted,
            )
            # The policy initializes its history from a fresh absolute-position read.
            # Do not carry the preposition target into the policy step-delta gate.
            previous_sdk_target = None
            tracking_error_samples = 0

        while args.steps is None or iteration < args.steps:
            loop_start = time.monotonic()
            trace_row: int | None = None
            # A traced dry-run uses the same full motor-status read as motion so
            # currents/stall state are real samples, while still sending no command.
            read_motor_status = bool(args.enable_motion or trace_recorder is not None)
            read_timeout = (
                max_observation_age if args.enable_motion else diagnostic_io_timeout
            )
            try:
                sdk_pos, forces = await asyncio.wait_for(
                    io.read_observation(
                        check_motor_errors=read_motor_status,
                        enforce_tactile_freshness=args.enable_motion,
                    ),
                    timeout=read_timeout,
                )
            except asyncio.TimeoutError as exc:
                raise RuntimeError(
                    f"Observation read exceeded {read_timeout * 1000.0:.2f} ms"
                    + ("; command suppressed." if args.enable_motion else ".")
                ) from exc
            read_finished = time.monotonic()
            observation_age = read_finished - loop_start
            if io.last_tactile_age_s is not None:
                diagnostic_max_force_age_s = max(
                    diagnostic_max_force_age_s,
                    io.last_tactile_age_s,
                )
                if (
                    not args.enable_motion
                    and io.vision_touch is not None
                    and io.last_tactile_age_s > io.vision_touch.max_sample_age_s
                ):
                    diagnostic_stale_force_samples += 1
            if args.enable_motion and observation_age > max_observation_age:
                raise RuntimeError(
                    f"Observation is stale ({observation_age * 1000.0:.2f} ms > "
                    f"{max_observation_age * 1000.0:.2f} ms); command suppressed."
                )

            if args.enable_motion:
                profile.validate_sdk_position(
                    sdk_pos,
                    "measured SDK position",
                    tolerance_rad=measured_position_tolerance,
                )
                if previous_sdk_target is not None:
                    tracking_error = float(np.max(np.abs(previous_sdk_target - sdk_pos)))
                    tracking_error_samples = (
                        tracking_error_samples + 1
                        if tracking_error > max_tracking_error
                        else 0
                    )
                    if tracking_error_samples >= max_tracking_samples:
                        raise RuntimeError(
                            f"Tracking error {tracking_error:.6f} rad exceeded "
                            f"{max_tracking_error:.6f} rad for {tracking_error_samples} samples; "
                            "command suppressed."
                        )

            policy_pos = profile.measured_sdk_to_policy(sdk_pos)
            inference_started = time.monotonic()
            result = runner.step(policy_pos, forces)
            inference_finished = time.monotonic()
            sdk_target = profile.target_policy_to_sdk(result.policy_target_rad)
            if trace_recorder is not None:
                motor_status_valid = read_motor_status
                trace_row = trace_recorder.append_frame(
                    step_index=iteration,
                    loop_started_monotonic_s=loop_start,
                    sdk_pos_rad=sdk_pos,
                    policy_pos_rad=policy_pos,
                    force_n=forces,
                    result=result,
                    sdk_target_rad=sdk_target,
                    read_ms=(read_finished - loop_start) * 1000.0,
                    inference_ms=(inference_finished - inference_started) * 1000.0,
                    tactile_age_ms=(
                        None
                        if io.last_tactile_age_s is None
                        else io.last_tactile_age_s * 1000.0
                    ),
                    motor_current_ma=(
                        io.last_motor_currents_ma if motor_status_valid else None
                    ),
                    stalled_motor_ids=(
                        io.last_stalled_motor_ids if motor_status_valid else ()
                    ),
                    stall_duration_s=(
                        io.last_stall_durations_s if motor_status_valid else None
                    ),
                    motor_status_valid=motor_status_valid,
                )
            if args.enable_motion:
                profile.validate_sdk_position(sdk_target, "SDK command target")
                if previous_sdk_target is None:
                    command_delta = float(np.max(np.abs(sdk_target - sdk_pos)))
                    allowed_delta = initial_delta_limit
                    label = "initial measured-to-target delta"
                else:
                    command_delta = float(np.max(np.abs(sdk_target - previous_sdk_target)))
                    allowed_delta = command_step_limit
                    label = "target step"
                if command_delta > allowed_delta:
                    raise RuntimeError(
                        f"Unsafe {label}: {command_delta:.6f} rad exceeds "
                        f"{allowed_delta:.6f} rad; command suppressed."
                    )

                pre_send_age = time.monotonic() - loop_start
                if pre_send_age > period:
                    raise RuntimeError(
                        f"Control frame expired before send ({pre_send_age * 1000.0:.2f} ms); "
                        "command suppressed."
                    )
                command_may_have_been_sent = True
                write_timeout = period - pre_send_age
                write_started = time.monotonic()
                if trace_recorder is not None and trace_row is not None:
                    trace_recorder.mark_command_sent(
                        trace_row,
                        pre_send_ms=pre_send_age * 1000.0,
                    )
                try:
                    await asyncio.wait_for(
                        io.send_mit_command_rad(
                            sdk_target,
                            kp=kp,
                            kd=kd,
                            effort_ma=effort_ma,
                        ),
                        timeout=write_timeout,
                    )
                except asyncio.TimeoutError as exc:
                    raise RuntimeError(
                        f"Control write exceeded the remaining {write_timeout * 1000.0:.2f} ms "
                        "frame budget; zero-force release requested."
                    ) from exc
                if trace_recorder is not None and trace_row is not None:
                    trace_recorder.mark_command_completed(
                        trace_row,
                        write_ms=(time.monotonic() - write_started) * 1000.0,
                    )
                previous_sdk_target = sdk_target.copy()
                post_send_age = time.monotonic() - loop_start
                if post_send_age > period:
                    raise RuntimeError(
                        f"Control write exceeded the {period * 1000.0:.2f} ms deadline; "
                        "zero-force release requested."
                    )

            iteration += 1
            elapsed = time.monotonic() - loop_start
            if trace_recorder is not None and trace_row is not None:
                trace_recorder.finish_frame(trace_row, loop_ms=elapsed * 1000.0)
            if iteration == 1 or iteration % args.print_every == 0:
                print(
                    f"step={iteration} loop_ms={elapsed * 1000.0:.2f} "
                    f"force_N={np.array2string(forces, precision=3)} "
                    f"action_abs_max={float(np.max(np.abs(result.action))):.3f} "
                    + (
                        f"force_age_ms={io.last_tactile_age_s * 1000.0:.1f}"
                        if io.last_tactile_age_s is not None
                        else ""
                    )
                    + (
                        f" stall={list(io.last_stalled_motor_ids)}"
                        if io.last_stalled_motor_ids
                        else ""
                    )
                )

            # Never catch up with a shortened policy interval after scheduler jitter.
            sleep_time = period - (time.monotonic() - loop_start)
            if sleep_time > 0.0:
                await asyncio.sleep(sleep_time)
        if not args.enable_motion and io.vision_touch is not None:
            print(
                "VisionTouch diagnostic: "
                f"max_force_age_ms={diagnostic_max_force_age_s * 1000.0:.1f} "
                f"stale_samples={diagnostic_stale_force_samples}/{iteration} "
                f"threshold_ms={io.vision_touch.max_sample_age_s * 1000.0:.1f}"
            )
    except BaseException as exc:
        run_error = exc
        if trace_recorder is not None:
            trace_recorder.finish_pending_frames()
        raise
    finally:
        if command_may_have_been_sent:
            try:
                await asyncio.wait_for(io.release_mit(), timeout=release_timeout)
                print(
                    "Zero-force MIT release sent; the hand may go limp and drop the object.",
                    file=sys.stderr,
                )
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
        trace_error = run_error or (cleanup_errors[0] if cleanup_errors else None)
        _save_trace_best_effort(
            trace_recorder,
            termination_status="completed" if trace_error is None else "error",
            error=trace_error,
        )
    if cleanup_errors:
        error = cleanup_errors[0]
        if isinstance(error, (asyncio.CancelledError, KeyboardInterrupt, SystemExit)):
            raise error
        raise RuntimeError(
            "Policy loop finished, but hardware cleanup failed; do not assume the hand is safe."
        ) from error
    return 0


async def _run_read_only_preflight(
    io: Revo3SdkHandIO,
    runner: Revo3PolicyRunner,
    profile: Revo3Profile,
    timeout_s: float,
    initial_delta_limit_rad: float,
    measured_position_tolerance_rad: float,
    trace_recorder: PolicyTraceRecorder | None = None,
) -> None:
    started = time.monotonic()
    try:
        sdk_pos, forces = await asyncio.wait_for(
            io.read_observation(
                check_motor_errors=True,
                enforce_tactile_freshness=True,
            ),
            timeout=timeout_s,
        )
    except asyncio.TimeoutError as exc:
        raise RuntimeError(
            f"Preflight observation exceeded {timeout_s * 1000.0:.1f} ms."
        ) from exc
    read_finished = time.monotonic()
    observation_age_s = read_finished - started
    profile.validate_sdk_position(
        sdk_pos,
        "preflight measured SDK position",
        tolerance_rad=measured_position_tolerance_rad,
    )
    policy_pos = profile.measured_sdk_to_policy(sdk_pos)
    inference_started = time.monotonic()
    result = runner.step(policy_pos, forces)
    inference_finished = time.monotonic()
    sdk_target = profile.target_policy_to_sdk(result.policy_target_rad)
    profile.validate_sdk_position(sdk_target, "preflight first SDK target")
    io.validate_device_position(sdk_target)

    currents = io.last_motor_currents_ma
    if currents is None:
        raise RuntimeError("Preflight motor currents are unavailable.")
    delta = sdk_target - sdk_pos
    max_index = int(np.argmax(np.abs(delta)))
    max_delta = float(np.max(np.abs(delta)))
    device_lower = io.device_position_lower_rad
    device_upper = io.device_position_upper_rad
    if device_lower is None or device_upper is None:
        raise RuntimeError("Preflight device limits are unavailable.")

    if trace_recorder is not None:
        trace_row = trace_recorder.append_frame(
            step_index=0,
            loop_started_monotonic_s=started,
            sdk_pos_rad=sdk_pos,
            policy_pos_rad=policy_pos,
            force_n=forces,
            result=result,
            sdk_target_rad=sdk_target,
            read_ms=(read_finished - started) * 1000.0,
            inference_ms=(inference_finished - inference_started) * 1000.0,
            tactile_age_ms=(
                None
                if io.last_tactile_age_s is None
                else io.last_tactile_age_s * 1000.0
            ),
            motor_current_ma=currents,
            stalled_motor_ids=io.last_stalled_motor_ids,
            stall_duration_s=io.last_stall_durations_s,
            motor_status_valid=True,
        )
        trace_recorder.finish_frame(
            trace_row,
            loop_ms=(time.monotonic() - started) * 1000.0,
        )

    info = io.device_info
    print("Hardware preflight: NO MOTOR COMMAND WILL BE SENT")
    print(
        f"Connected port={io.port} slave_id={io.slave_id} "
        f"serial={getattr(info, 'serial_number', '')} "
        f"hardware={getattr(info, 'hardware_type', '')} tactile_source={io.tactile_source}"
    )
    print(
        f"observation_ms={observation_age_s * 1000.0:.2f} "
        f"force_age_ms={float(io.last_tactile_age_s or 0.0) * 1000.0:.2f} "
        f"force_N={np.array2string(forces, precision=3)}"
    )
    print("motor joint measured_deg target_deg delta_deg current_mA device_deg")
    for index, joint in enumerate(profile.sdk_joint_order):
        print(
            f"M{index:02d} {joint:24s} "
            f"{np.rad2deg(sdk_pos[index]):9.3f} "
            f"{np.rad2deg(sdk_target[index]):9.3f} "
            f"{np.rad2deg(delta[index]):+9.3f} "
            f"{currents[index]:+9.1f} "
            f"[{np.rad2deg(device_lower[index]):.1f},{np.rad2deg(device_upper[index]):.1f}]"
        )
    status = "PASS" if max_delta <= initial_delta_limit_rad else "FAIL"
    print(
        f"first_delta_gate={status} max={max_delta:.6f} rad/"
        f"{np.rad2deg(max_delta):.3f} deg at M{max_index} "
        f"limit={initial_delta_limit_rad:.6f} rad/"
        f"{np.rad2deg(initial_delta_limit_rad):.3f} deg"
    )
    print(
        f"action_abs_max={float(np.max(np.abs(result.action))):.3f}; "
        f"calibration_status={profile.calibration_status}; NO MOTOR COMMAND SENT"
    )
    print(
        "NOTE: first_delta_gate checks command continuity only; it does not verify "
        "cache-pose equality or sim-to-real joint offsets."
    )


def _load_preposition_sdk_target(
    cache_path: str,
    row: int,
    profile: Revo3Profile,
) -> np.ndarray:
    path = Path(cache_path).expanduser()
    try:
        cache = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ValueError(f"Cannot load preposition cache {path}: {exc}") from exc
    if cache.ndim != 2 or cache.shape[1] < 21:
        raise ValueError(
            f"Preposition cache must have shape [N,>=21], got {cache.shape}."
        )
    if row >= cache.shape[0]:
        raise ValueError(
            f"--preposition-row {row} is outside cache with {cache.shape[0]} rows."
        )
    policy_target = np.asarray(cache[row, :21], dtype=np.float32)
    if not np.isfinite(policy_target).all():
        raise ValueError(f"Preposition cache row {row} contains NaN or infinity.")
    return profile.target_policy_to_sdk(policy_target)


def _build_preposition_targets(
    start_rad: np.ndarray,
    target_rad: np.ndarray,
    *,
    speed_deg_s: float,
    rate_hz: float,
) -> np.ndarray:
    start = np.asarray(start_rad, dtype=np.float64)
    target = np.asarray(target_rad, dtype=np.float64)
    if start.shape != (21,) or target.shape != (21,):
        raise ValueError("Preposition start and target must each contain 21 joints.")
    max_step_rad = float(np.deg2rad(speed_deg_s) / rate_hz)
    step_count = max(
        1,
        int(np.ceil(float(np.max(np.abs(target - start))) / max_step_rad)),
    )
    alpha = np.arange(1, step_count + 1, dtype=np.float64)[:, None] / step_count
    return (start[None, :] + alpha * (target - start)[None, :]).astype(np.float32)


async def _run_preposition(
    *,
    io: Revo3SdkHandIO,
    profile: Revo3Profile,
    sdk_target: np.ndarray,
    rate_hz: float,
    speed_deg_s: float,
    kp: float,
    kd: float,
    effort_ma: float,
    read_timeout_s: float,
    max_tracking_error_rad: float,
    max_tracking_samples: int,
    measured_position_tolerance_rad: float,
    device_target_margin_rad: float,
    mark_command_attempted: Callable[[], None],
) -> None:
    period = 1.0 / rate_hz
    try:
        measured = await asyncio.wait_for(
            io.read_position_rad(check_errors=True),
            timeout=read_timeout_s,
        )
    except asyncio.TimeoutError as exc:
        raise RuntimeError("Preposition initial position read timed out; no target sent.") from exc
    profile.validate_sdk_position(
        measured,
        "preposition measured SDK position",
        tolerance_rad=measured_position_tolerance_rad,
    )

    device_lower = io.device_position_lower_rad
    device_upper = io.device_position_upper_rad
    if device_lower is None or device_upper is None:
        raise RuntimeError("Preposition requires live device joint limits.")
    command_lower = (
        np.maximum(profile.sdk_position_lower_rad, device_lower)
        + device_target_margin_rad
    )
    command_upper = (
        np.minimum(profile.sdk_position_upper_rad, device_upper)
        - device_target_margin_rad
    )
    if np.any(command_upper <= command_lower):
        raise RuntimeError("Preposition command envelopes have an empty joint intersection.")
    target = np.asarray(sdk_target, dtype=np.float32)
    if np.any(target < command_lower) or np.any(target > command_upper):
        bad = np.flatnonzero((target < command_lower) | (target > command_upper))
        raise RuntimeError(
            "Preposition target is outside the inward live command envelope at motors "
            + ", ".join(f"M{int(index)}" for index in bad)
            + "."
        )

    # Absolute encoder readings define the interpolation start. Clipping here only moves a
    # noisy/outward-limit reading into the strict command envelope; it does not change zero.
    command_start = np.clip(measured, command_lower, command_upper)
    targets = _build_preposition_targets(
        command_start,
        target,
        speed_deg_s=speed_deg_s,
        rate_hz=rate_hz,
    )
    settle_steps = max(1, int(np.ceil(0.25 * rate_hz)))
    scheduled_targets = np.concatenate(
        [targets, np.repeat(target[None, :], settle_steps, axis=0)],
        axis=0,
    )
    previous_target: np.ndarray | None = None
    tracking_samples = 0
    print(
        f"Preposition: cache target over {len(targets)} frames at <= "
        f"{speed_deg_s:.2f} deg/s, then {settle_steps} settle frames; "
        "absolute motor zeros are unchanged."
    )

    for index, command in enumerate(scheduled_targets, start=1):
        frame_started = time.monotonic()
        try:
            measured = await asyncio.wait_for(
                io.read_position_rad(check_errors=True),
                timeout=read_timeout_s,
            )
        except asyncio.TimeoutError as exc:
            raise RuntimeError("Preposition position read timed out; release requested.") from exc
        profile.validate_sdk_position(
            measured,
            "preposition measured SDK position",
            tolerance_rad=measured_position_tolerance_rad,
        )
        if previous_target is not None:
            tracking_error = float(np.max(np.abs(previous_target - measured)))
            tracking_samples = tracking_samples + 1 if tracking_error > max_tracking_error_rad else 0
            if tracking_samples >= max_tracking_samples:
                raise RuntimeError(
                    f"Preposition tracking error {np.rad2deg(tracking_error):.2f} deg "
                    f"persisted for {tracking_samples} samples; release requested."
                )
        else:
            tracking_error = float(np.max(np.abs(command_start - measured)))

        profile.validate_sdk_position(command, "preposition SDK command")
        io.validate_device_position(command)
        elapsed = time.monotonic() - frame_started
        write_timeout = period - elapsed
        if write_timeout <= 0.0:
            raise RuntimeError("Preposition frame expired before send; release requested.")
        try:
            mark_command_attempted()
            await asyncio.wait_for(
                io.send_mit_command_rad(command, kp=kp, kd=kd, effort_ma=effort_ma),
                timeout=write_timeout,
            )
        except asyncio.TimeoutError as exc:
            raise RuntimeError("Preposition write missed its deadline; release requested.") from exc
        if time.monotonic() - frame_started > period:
            raise RuntimeError("Preposition frame exceeded its deadline; release requested.")
        previous_target = command.copy()

        if index == 1 or index % 20 == 0 or index == len(scheduled_targets):
            remaining = float(np.max(np.abs(target - measured)))
            currents = io.last_motor_currents_ma
            current_abs_max = (
                float(np.max(np.abs(currents))) if currents is not None else float("nan")
            )
            print(
                f"preposition_step={index}/{len(scheduled_targets)} "
                f"remaining_deg={np.rad2deg(remaining):.2f} "
                f"tracking_deg={np.rad2deg(tracking_error):.2f} "
                f"current_abs_max_mA={current_abs_max:.1f}"
                + (
                    f" stall={list(io.last_stalled_motor_ids)}"
                    if io.last_stalled_motor_ids
                    else ""
                )
            )
        sleep_time = period - (time.monotonic() - frame_started)
        if sleep_time > 0.0:
            await asyncio.sleep(sleep_time)

    try:
        final_measured = await asyncio.wait_for(
            io.read_position_rad(check_errors=True),
            timeout=read_timeout_s,
        )
    except asyncio.TimeoutError as exc:
        raise RuntimeError("Preposition arrival check timed out; release requested.") from exc
    arrival_error = np.abs(target - final_measured)
    arrival_index = int(np.argmax(arrival_error))
    arrival_max = float(arrival_error[arrival_index])
    arrival_limit = float(np.deg2rad(2.5))
    if arrival_max > arrival_limit:
        raise RuntimeError(
            f"Preposition did not arrive: M{arrival_index} error "
            f"{np.rad2deg(arrival_max):.2f} deg exceeds 2.50 deg; release requested."
        )
    print(
        f"Preposition PASS: max absolute-position error={np.rad2deg(arrival_max):.2f} deg "
        f"at M{arrival_index}; starting policy immediately."
    )


def _build_trace_metadata(
    *,
    args: argparse.Namespace,
    runner: Revo3PolicyRunner,
    rate_hz: float,
    kp: float,
    kd: float,
    effort_ma: float,
) -> dict:
    profile = runner.profile
    contract = runner.contract
    mode = (
        "preflight"
        if args.preflight_only
        else ("motion" if args.enable_motion else "dry_run")
    )
    return {
        "schema_name": "hora_policy_trace",
        "schema_version": 1,
        "source": "real",
        "mode": mode,
        # This deployment runtime currently validates only the exported tactile
        # cylinder Stage-2 ABI.  Keep these top-level fields identical to the sim
        # trace so a comparator can reject incompatible runs before loading arrays.
        "task": "cylinder",
        "policy_dt_s": 1.0 / rate_hz,
        "policy_rate_hz": rate_hz,
        "action_scale": contract.action_scale,
        "joint_order": list(contract.joint_order),
        "contact_order": list(contract.contact_order),
        "contact_force_scale": contract.contact_force_scale,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "artifacts": {
            "onnx": _trace_artifact(runner.onnx_path),
            "deploy_metadata": _trace_artifact(contract.path),
            "robot_profile": _trace_artifact(profile.path),
        },
        "software": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "onnxruntime_providers": runner.providers,
        },
        "policy_contract": {
            "policy_rate_hz": contract.policy_rate_hz,
            "runtime_rate_hz": rate_hz,
            "action_scale_rad": contract.action_scale,
            "action_semantics": "delta",
            "normalization_baked_in": contract.normalization_baked_in,
            "contact_force_scale": contract.contact_force_scale,
            "obs_shape": [1, contract.obs_dim],
            "proprio_hist_shape": [
                1,
                contract.history_len,
                contract.frame_dim,
            ],
            "action_shape": [1, contract.action_dim],
            "frame_layout": {
                "joint_pos_unscaled": [0, 21],
                "input_target_policy_rad": [21, 42],
                "force_n": [42, 47],
            },
            "history_order": "oldest_to_newest",
            "policy_joint_order": list(contract.joint_order),
            "sdk_joint_order": list(profile.sdk_joint_order),
            "contact_order": list(contract.contact_order),
            "policy_to_sdk_permutation": profile.policy_to_sdk_perm.tolist(),
            "sdk_to_policy_permutation": profile.sdk_to_policy_perm.tolist(),
            "joint_lower_policy_rad": profile.joint_lower_policy.tolist(),
            "joint_upper_policy_rad": profile.joint_upper_policy.tolist(),
            "profile_target_lower_policy_rad": profile.target_lower_policy.tolist(),
            "profile_target_upper_policy_rad": profile.target_upper_policy.tolist(),
            "sim2real_offset_sdk_rad": profile.sdk_offset_rad.tolist(),
        },
        "runtime": {
            "provider_requested": args.provider,
            "steps_requested": args.steps,
            "motion_enabled": bool(args.enable_motion),
            "preflight_only": bool(args.preflight_only),
            "policy_start_delay_s": float(args.policy_start_delay_s),
            "kp": kp,
            "kd": kd,
            "effort_ma": effort_ma,
            "calibration_status": profile.calibration_status,
            "allow_unverified_calibration": bool(args.allow_unverified_calibration),
            "allowed_stall_motor_ids": list(args.allow_stall),
            "stall_grace_s": float(args.stall_grace_s),
            "ignore_all_stall": bool(args.ignore_all_stall),
            "preposition_cache": (
                str(Path(args.preposition_cache).expanduser().resolve())
                if args.preposition_cache is not None
                else None
            ),
            "preposition_row": (
                int(args.preposition_row) if args.preposition_cache is not None else None
            ),
        },
        "tactile_profile": profile.tactile,
        "safety_profile": profile.safety,
        "trace_semantics": {
            "sample": "one pre-action policy observation and its resulting action/target",
            "network_inputs": "raw float32 before ONNX-baked normalization",
            "command_sent": (
                "true once the SDK send call is entered; a timeout may still mean the device "
                "received the command"
            ),
            "command_completed": "true only when the SDK send await returned normally",
            "dry_run_motor_status": (
                "when tracing, dry-run uses the same checked full motor-status read as motion "
                "so current/stall arrays are sampled without sending commands"
            ),
        },
    }


def _trace_artifact(path: str | Path) -> dict[str, str]:
    resolved = Path(path).expanduser().resolve()
    digest = hashlib.sha256()
    with resolved.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(resolved), "sha256": digest.hexdigest()}


def _save_trace_best_effort(
    recorder: PolicyTraceRecorder | None,
    *,
    termination_status: str,
    error: BaseException | None,
) -> None:
    if recorder is None:
        return
    try:
        path = recorder.save(termination_status=termination_status, error=error)
        print(f"Policy trace saved: {path} ({recorder.frame_count} frames)")
    except Exception as trace_error:
        print(
            f"WARNING: policy trace could not be saved: {trace_error!r}",
            file=sys.stderr,
        )


def _positive(cfg: dict, key: str, default: float) -> float:
    value = float(cfg.get(key, default))
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"safety.{key} must be finite and positive.")
    return value


def _validate_mit_settings(kp: float, kd: float, effort_ma: float, safety: dict) -> None:
    values = np.asarray([kp, kd, effort_ma], dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("MIT kp, kd, and effort must be finite.")
    max_kp = _positive(safety, "max_kp", 2.0)
    max_kd = _positive(safety, "max_kd", 1.0)
    max_effort = float(safety.get("max_abs_effort_ma", 0.0))
    if not np.isfinite(max_effort) or max_effort < 0.0:
        raise ValueError("safety.max_abs_effort_ma must be finite and non-negative.")
    if not 0.0 <= kp <= min(max_kp, 10.0):
        raise ValueError(f"MIT kp={kp} is outside the configured safe range [0,{max_kp}].")
    if not 0.0 <= kd <= min(max_kd, 10.0):
        raise ValueError(f"MIT kd={kd} is outside the configured safe range [0,{max_kd}].")
    if abs(effort_ma) > min(max_effort, 1024.0):
        raise ValueError(
            f"MIT effort={effort_ma}mA exceeds configured absolute limit {max_effort}mA."
        )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return asyncio.run(async_main(args))
    except KeyboardInterrupt:
        return 130


def _parse_motor_id(value: str) -> int:
    text = str(value).strip().upper()
    if not text.startswith("M") or not text[1:].isdigit():
        raise ValueError("--allow-stall must name a motor such as M13.")
    index = int(text[1:])
    if not 0 <= index < 21:
        raise ValueError("--allow-stall motor must be in M0..M20.")
    return index


if __name__ == "__main__":
    raise SystemExit(main())
