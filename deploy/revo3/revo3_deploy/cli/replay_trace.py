from __future__ import annotations

import argparse
import asyncio
import hashlib
import math
from pathlib import Path
import sys
import time

import numpy as np

from revo3_deploy.replay_trace import ReplayTrace
from revo3_deploy.robot_profile import Revo3Profile
from revo3_deploy.sdk_hand_io import Revo3SdkConfig, Revo3SdkHandIO


_monotonic = time.monotonic


MAX_SELECTED_JOINTS = 4
# The deployment profile is fixed at 20 Hz, so this permits one uninterrupted
# recorded-rate replay of up to 30 seconds.  Preposition ticks are planned and
# checked separately and do not consume trace frames.
MAX_EXECUTE_FRAMES = 600
CONTROL_RATE_HZ = 20.0
MAX_REPLAY_SPEED_DEG_S = 2.0
MAX_EXECUTION_TICKS = 1200
CONFIRM_EXCURSION_DEG = 1.0
MAX_EXCURSION_DEG = 10.0
MAX_CURRENT_MA = 500.0
MAX_TOTAL_SELECTED_CURRENT_MA = 1000.0
MAX_MEASURED_TOLERANCE_RAD = math.radians(3.0)
MAX_EXPLICIT_MEASURED_TOLERANCE_DEG = 5.0
MAX_LIVE_MEASURED_TOLERANCE_RAD = math.radians(0.25)
MAX_PASSIVE_LIVE_LIMIT_OUTLIER_RAD = math.radians(5.0)
MAX_PASSIVE_MOVEMENT_RAD = math.radians(5.0)
MAX_ANCHORED_PASSIVE_MOVEMENT_RAD = math.radians(1.0)
MAX_TRACKING_ERROR_RAD = math.radians(5.0)
MAX_TRACKING_ERROR_SAMPLES = 3
MAX_ANCHORED_EXCURSION_DEG = 3.0
MAX_ANCHORED_MEASURED_EXCURSION_DEG = 3.5
MAX_ANCHORED_TRACKING_ERROR_RAD = math.radians(2.0)
MAX_ANCHORED_TRACKING_ERROR_SAMPLES = 2
LOW_GAIN_KP = 1.0#0.3
LOW_GAIN_KD = 0.1
MAX_FINAL_HOLD_S = 120.0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect or replay a validated simulator policy-target trace on Revo3. "
            "No ONNX inference is performed; motion is disabled unless --execute and "
            "all confirmations are present."
        )
    )
    parser.add_argument("--trace-npz", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument(
        "--checkpoint",
        default=None,
        help=(
            "Checkpoint whose SHA256 must match the trace metadata; required for "
            "hardware preflight/replay."
        ),
    )
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument(
        "--trajectory-source",
        choices=("target", "measured"),
        default="target",
        help=(
            "Replay policy_target_rad (default) or the noiseless simulator measured "
            "joint trajectory policy_pos_rad. Both use metadata policy order and are "
            "permuted to SDK M0..M20 order before hardware checks."
        ),
    )
    parser.add_argument(
        "--recorded-rate",
        action="store_true",
        help=(
            "Replay measured endpoints directly at the trace policy rate. This disables "
            "diagnostic interpolation, source-step, excursion, plan-duration, and inward-"
            "margin gates while retaining device limits, current, online, communication, "
            "tracking, and non-ignored fault checks."
        ),
    )
    parser.add_argument(
        "--preposition-to-first",
        action="store_true",
        help=(
            "Before recorded-rate playback, interpolate from the fresh hardware pose "
            "to measured row 0 using --max-speed-deg-s, then begin the trace clock."
        ),
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=None,
        help=(
            "Number of non-terminal trace rows. Required for --execute and limited "
            "to 600 rows (30 seconds at the required 20 Hz profile rate); offline "
            "inspection otherwise uses the remainder of the trace."
        ),
    )
    parser.add_argument(
        "--joint",
        action="append",
        default=[],
        metavar="MOTOR_OR_JOINT",
        help=(
            "Controlled joint: SDK motor M0..M20, policy index P0..P20, or exact "
            "joint name. Repeat for at most four joints."
        ),
    )
    parser.add_argument(
        "--all-joints",
        action="store_true",
        help=(
            "Select all 21 joints. Execution is available only for measured trajectories "
            "with --confirm-full-hand and all ordinary motion confirmations."
        ),
    )
    parser.add_argument(
        "--max-speed-deg-s",
        type=float,
        default=2.0,
        help=(
            "Maximum selected-joint command speed. Recorded targets are retained as "
            "exact endpoints and interpolated at 20 Hz (default/max 2 deg/s)."
        ),
    )
    parser.add_argument("--kp", type=float, default=0.2)
    parser.add_argument("--kd", type=float, default=0.05)
    parser.add_argument("--print-every", type=int, default=1)
    parser.add_argument(
        "--hold-final-s",
        type=float,
        default=0.0,
        help=(
            "Keep sending the final checked target for up to 120 seconds before "
            "zero-force release. Intended for visual fixed-pose calibration."
        ),
    )
    parser.add_argument("--port", default=None)
    parser.add_argument("--baudrate", type=int, default=None)
    parser.add_argument("--slave-id", type=lambda value: int(value, 0), default=None)
    parser.add_argument(
        "--measured-limit-tolerance-deg",
        type=float,
        default=None,
        help=(
            "Explicit encoder-reading tolerance outside configured SDK limits, up to "
            "5 degrees. Command targets and live device limits remain strict."
        ),
    )
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Connect and compare live positions to the first target without sending commands.",
    )
    parser.add_argument(
        "--anchor-current",
        "--rebase-to-current",
        dest="anchor_current",
        action="store_true",
        help=(
            "For a single selected joint, replay the trace displacement relative to "
            "a fresh live start instead of pursuing the simulator's absolute cache pose. "
            "This is only for joint mapping/direction diagnosis."
        ),
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--allow-unverified-calibration", action="store_true")
    parser.add_argument("--confirm-fixed", action="store_true")
    parser.add_argument("--confirm-clear-path", action="store_true")
    parser.add_argument("--confirm-estop", action="store_true")
    parser.add_argument("--confirm-release", action="store_true")
    parser.add_argument(
        "--confirm-mapping",
        action="store_true",
        help="Confirm that the printed policy-index to SDK-motor mapping was reviewed.",
    )
    parser.add_argument("--confirm-high-gain", action="store_true")
    parser.add_argument(
        "--confirm-recorded-rate",
        action="store_true",
        help="Explicitly confirm direct measured replay at the recorded policy rate.",
    )
    parser.add_argument(
        "--confirm-preposition",
        action="store_true",
        help="Confirm the path from the live pose to the first measured endpoint.",
    )
    parser.add_argument(
        "--confirm-hold",
        action="store_true",
        help="Confirm visual inspection while the final commanded pose remains energized.",
    )
    parser.add_argument(
        "--confirm-full-hand",
        action="store_true",
        help=(
            "Explicitly confirm simultaneous 21-joint measured-trajectory motion. "
            "Required with --all-joints --execute."
        ),
    )
    parser.add_argument(
        "--ignore-all-stall",
        action="store_true",
        help=(
            "Ignore only Stall bit 0x100 on all motors during hardware preflight or an "
            "explicitly authorized execution. Current, limit, tracking, communication, "
            "and all other fault checks remain active."
        ),
    )
    parser.add_argument(
        "--confirm-large-excursion",
        action="store_true",
        help="Required when any selected joint moves more than 1 degree from live start.",
    )
    parser.add_argument(
        "--confirm-current-anchor",
        action="store_true",
        help=(
            "Confirm that current-anchored replay changes the absolute simulator "
            "trajectory and is only being used as a single-joint mapping probe."
        ),
    )
    parser.add_argument(
        "--allow-passive-limit-outlier",
        action="store_true",
        help=(
            "In current-anchor mode only, explicitly allow an unselected zero-gain "
            "encoder reading to sit at most 5 degrees beyond a device-reported limit. "
            "Selected joints and all command targets remain strictly bounded."
        ),
    )
    # Internal provenance binding used by the joint-order session. These are
    # hidden from the ordinary replay UI because the session, not an operator,
    # creates the expected digests.
    parser.add_argument("--expected-trace-sha256", help=argparse.SUPPRESS)
    parser.add_argument("--expected-checkpoint-sha256", help=argparse.SUPPRESS)
    parser.add_argument("--expected-profile-sha256", help=argparse.SUPPRESS)
    return parser


async def async_main(args: argparse.Namespace) -> int:
    if args.preflight and args.execute:
        raise ValueError("--preflight and --execute are mutually exclusive.")
    if (args.preflight or args.execute) and args.checkpoint is None:
        raise RuntimeError(
            "Hardware preflight/replay requires --checkpoint so its SHA256 can be "
            "matched to the trace."
        )

    _verify_expected_artifact_hashes(args)
    profile = Revo3Profile.load(args.profile)
    trace = ReplayTrace.load(args.trace_npz, profile, checkpoint_path=args.checkpoint)
    # Catch replacement during deserialization. Later path changes cannot alter
    # the profile and trace objects already materialized for this execution.
    _verify_expected_artifact_hashes(args)
    rows = trace.select(args.start_frame, args.frames)
    selected_sdk = _resolve_sdk_indices(profile, args.joint, args.all_joints)
    if args.anchor_current and not (args.preflight or args.execute):
        raise ValueError("--anchor-current requires --preflight or --execute.")
    if args.confirm_current_anchor and not (args.anchor_current and args.execute):
        raise ValueError(
            "--confirm-current-anchor is only valid with --anchor-current --execute."
        )
    if args.allow_passive_limit_outlier and not args.anchor_current:
        raise ValueError(
            "--allow-passive-limit-outlier is only valid with --anchor-current."
        )
    if args.recorded_rate and args.trajectory_source != "measured":
        raise ValueError("--recorded-rate requires --trajectory-source measured.")
    if args.recorded_rate and args.anchor_current:
        raise ValueError("--recorded-rate does not support --anchor-current.")
    if args.preposition_to_first and not args.recorded_rate:
        raise ValueError("--preposition-to-first requires --recorded-rate.")
    if args.confirm_recorded_rate and not (args.recorded_rate and args.execute):
        raise ValueError(
            "--confirm-recorded-rate is only valid with --recorded-rate --execute."
        )
    if args.confirm_preposition and not (args.preposition_to_first and args.execute):
        raise ValueError(
            "--confirm-preposition is only valid with --preposition-to-first --execute."
        )
    hold_final_s = float(args.hold_final_s)
    if not np.isfinite(hold_final_s) or not 0.0 <= hold_final_s <= MAX_FINAL_HOLD_S:
        raise ValueError(f"--hold-final-s must be finite and in [0,{MAX_FINAL_HOLD_S:g}].")
    if hold_final_s > 0.0 and not args.execute:
        raise ValueError("--hold-final-s requires --execute.")
    if args.confirm_hold and not (args.execute and hold_final_s > 0.0):
        raise ValueError("--confirm-hold is only valid with --execute --hold-final-s > 0.")
    if args.execute and hold_final_s > 0.0 and not args.confirm_hold:
        raise RuntimeError("Final-pose hold requires --confirm-hold.")
    if args.measured_limit_tolerance_deg is not None:
        requested_measured_tolerance = float(args.measured_limit_tolerance_deg)
        if (
            not np.isfinite(requested_measured_tolerance)
            or requested_measured_tolerance < 0.0
            or requested_measured_tolerance > MAX_EXPLICIT_MEASURED_TOLERANCE_DEG
        ):
            raise ValueError(
                "--measured-limit-tolerance-deg must be finite and in [0,5]."
            )
    if args.ignore_all_stall and not (args.preflight or args.execute):
        raise ValueError(
            "--ignore-all-stall is only valid with --preflight or --execute."
        )
    if args.confirm_full_hand and not (args.execute and args.all_joints):
        raise ValueError(
            "--confirm-full-hand is only valid with --all-joints --execute."
        )
    display_sdk = selected_sdk or tuple(range(len(profile.sdk_joint_order)))
    trajectory_policy = trace.trajectory_policy_rad(args.trajectory_source)
    _print_trace_summary(
        trace,
        profile,
        rows,
        display_sdk,
        selected_sdk,
        args.trajectory_source,
        trajectory_policy,
    )

    if not args.preflight and not args.execute:
        print("OFFLINE INSPECTION ONLY: no hardware was opened and no command was sent.")
        return 0

    if not selected_sdk:
        raise ValueError(
            "Hardware preflight/replay requires --joint (recommended) or --all-joints."
        )
    if len(selected_sdk) > MAX_SELECTED_JOINTS and not args.all_joints:
        raise ValueError(f"At most {MAX_SELECTED_JOINTS} joints may be selected.")
    if args.execute and args.all_joints and args.trajectory_source != "measured":
        raise RuntimeError(
            "21-joint execution is limited to --trajectory-source measured; target-trace "
            "execution remains limited to four selected joints."
        )
    if args.execute and args.all_joints and not args.confirm_full_hand:
        raise RuntimeError(
            "21-joint measured replay requires --confirm-full-hand."
        )
    if args.execute and args.recorded_rate and not args.confirm_recorded_rate:
        raise RuntimeError(
            "Recorded-rate measured replay requires --confirm-recorded-rate."
        )
    if args.execute and args.preposition_to_first and not args.confirm_preposition:
        raise RuntimeError(
            "Recorded-rate preposition requires --confirm-preposition."
        )
    if args.anchor_current:
        if args.all_joints or len(selected_sdk) != 1:
            raise RuntimeError(
                "Current-anchored replay is limited to exactly one --joint for mapping "
                "diagnosis."
            )
    if args.execute and args.frames is None:
        raise RuntimeError("Replay refused: --execute requires an explicit --frames value.")
    if args.execute and len(rows) > MAX_EXECUTE_FRAMES:
        raise ValueError(
            f"A single hardware replay is limited to {MAX_EXECUTE_FRAMES} frames."
        )

    max_speed_deg_s = float(args.max_speed_deg_s)
    kp = float(args.kp)
    kd = float(args.kd)
    numeric = np.asarray([max_speed_deg_s, kp, kd], dtype=np.float64)
    if not np.isfinite(numeric).all():
        raise ValueError("Replay speed and gains must be finite.")
    if max_speed_deg_s <= 0.0 or (
        not args.recorded_rate and max_speed_deg_s > MAX_REPLAY_SPEED_DEG_S
    ):
        raise ValueError(
            f"--max-speed-deg-s must be positive and, outside --recorded-rate, no "
            f"greater than {MAX_REPLAY_SPEED_DEG_S:g}."
        )
    max_kp = min(float(profile.safety.get("max_kp", 2.0)), 2.0)
    max_kd = min(float(profile.safety.get("max_kd", 1.0)), 1.0)
    if not 0.0 < kp <= max_kp or not 0.0 <= kd <= max_kd:
        raise ValueError(
            f"Replay gains must satisfy 0 < kp <= {max_kp:g} and 0 <= kd <= {max_kd:g}."
        )
    if (kp > LOW_GAIN_KP or kd > LOW_GAIN_KD) and not args.confirm_high_gain:
        raise RuntimeError(
            "Replay refused: gains above kp=0.3 or kd=0.1 require --confirm-high-gain."
        )
    if args.print_every <= 0:
        raise ValueError("--print-every must be positive.")

    if args.execute:
        confirmations = (
            args.confirm_fixed,
            args.confirm_clear_path,
            args.confirm_estop,
            args.confirm_release,
            args.confirm_mapping,
        )
        if not all(confirmations):
            raise RuntimeError(
                "Replay refused: --execute and all five physical/mapping confirmations "
                "are required."
            )
        if args.anchor_current and not args.confirm_current_anchor:
            raise RuntimeError(
                "Replay refused: --anchor-current --execute requires "
                "--confirm-current-anchor."
            )
        if (
            profile.calibration_status != "verified"
            and not args.allow_unverified_calibration
        ):
            raise RuntimeError(
                "Replay refused: calibration.status is not verified; use the bounded "
                "single-joint procedure first or explicitly pass "
                "--allow-unverified-calibration."
            )

    safety = profile.safety
    max_current_ma = (
        float(safety.get("max_abs_current_ma", MAX_CURRENT_MA))
        if args.recorded_rate
        else min(
            float(safety.get("jog_max_abs_current_ma", MAX_CURRENT_MA)),
            MAX_CURRENT_MA,
        )
    )
    if not np.isfinite(max_current_ma) or max_current_ma <= 0.0:
        raise ValueError("Replay current cutoff must be finite and positive.")
    connection_timeout = _positive(safety, "connection_timeout_s", 15.0)
    release_timeout = _positive(safety, "release_timeout_s", 0.25)
    close_timeout = _positive(safety, "close_timeout_s", 0.5)
    target_margin = (
        0.0
        if args.recorded_rate
        else _positive(
            safety,
            "device_target_margin_rad",
            math.radians(0.05),
        )
    )
    initial_delta_limit = _positive(
        safety,
        "max_initial_command_delta_rad",
        math.radians(5.0),
    )
    step_delta_limit = _positive(
        safety,
        "max_command_step_rad",
        0.05,
    )
    if args.measured_limit_tolerance_deg is None:
        measured_tolerance = min(
            _nonnegative(
                safety,
                "measured_position_tolerance_rad",
                math.radians(0.5),
            ),
            MAX_MEASURED_TOLERANCE_RAD,
        )
    else:
        measured_tolerance_deg = float(args.measured_limit_tolerance_deg)
        if (
            not np.isfinite(measured_tolerance_deg)
            or measured_tolerance_deg < 0.0
            or measured_tolerance_deg > MAX_EXPLICIT_MEASURED_TOLERANCE_DEG
        ):
            raise ValueError(
                "--measured-limit-tolerance-deg must be finite and in [0,5]."
            )
        measured_tolerance = math.radians(measured_tolerance_deg)

    sdk_cfg = profile.sdk
    configured_slave = sdk_cfg.get("slave_id")
    serial_allowlist = tuple(
        str(value).strip() for value in sdk_cfg.get("serial_allowlist") or ()
    )
    if not serial_allowlist:
        raise RuntimeError("Replay refused: sdk.serial_allowlist must bind the physical hand.")

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
            use_without_retry=bool(profile.mit.get("without_retry", True)),
            expected_hand=str(sdk_cfg.get("expected_hand", profile.hand)),
            allowed_hardware_types=tuple(
                str(value) for value in sdk_cfg.get("allowed_hardware_types") or ()
            ),
            serial_allowlist=serial_allowlist,
            max_abs_current_ma=max_current_ma,
            allowed_stall_motor_ids=(tuple(range(21)) if args.ignore_all_stall else ()),
        ),
        tactile_cfg=profile.tactile,
    )

    command_may_have_been_sent = False
    run_error: BaseException | None = None
    cleanup_errors: list[BaseException] = []
    try:
        await asyncio.wait_for(io.open(), timeout=connection_timeout)
        measured = await asyncio.wait_for(
            io.read_position_rad(check_errors=True),
            timeout=0.5,
        )
        lower, upper = _effective_sdk_envelope(profile, io, target_margin)
        live_lower, live_upper = _live_sdk_envelope(io, target_margin)
        trace_sdk_targets = _sdk_targets(profile, trajectory_policy[rows])
        if args.recorded_rate:
            # Canonicalize only floating-point residue around a nominal zero.
            trace_sdk_targets[np.abs(trace_sdk_targets) < 1.0e-7] = 0.0
        baseline_policy = (
            trace.target_before_policy_rad[rows[:1]]
            if args.trajectory_source == "target"
            else trajectory_policy[rows[:1]]
        )
        trace_baseline_sdk = _sdk_targets(profile, baseline_policy)[0]
        if args.anchor_current:
            _validate_anchored_measured_position(
                profile=profile,
                io=io,
                measured=measured,
                selected_sdk=selected_sdk,
                selected_lower=lower,
                selected_upper=upper,
                allow_passive_limit_outlier=args.allow_passive_limit_outlier,
            )
            _print_passive_nominal_outliers(profile, io, measured, selected_sdk)
            sdk_targets = _anchor_selected_targets(
                trace_sdk_targets,
                trace_baseline_sdk,
                measured,
                selected_sdk,
            )
            print(
                "target_mode=current_anchored: selected targets preserve trace "
                "displacement but do not pursue the absolute cache pose."
            )
        else:
            profile.validate_sdk_position(
                measured,
                "replay measured start",
                tolerance_rad=measured_tolerance,
            )
            sdk_targets = trace_sdk_targets
        _validate_selected_targets(
            profile,
            sdk_targets,
            selected_sdk,
            lower,
            upper,
            step_delta_limit,
            rows,
            enforce_step_limit=not args.recorded_rate,
        )
        initial_delta = np.abs(sdk_targets[0, selected_sdk] - measured[list(selected_sdk)])
        _print_live_alignment(profile, selected_sdk, measured, sdk_targets[0], initial_delta)
        max_excursion = _validate_live_excursion(
            profile,
            sdk_targets,
            selected_sdk,
            measured,
            math.inf if args.preposition_to_first else initial_delta_limit,
            require_confirmation=args.execute,
            confirmed=args.confirm_large_excursion,
            max_excursion_deg=(
                math.inf
                if args.recorded_rate
                else (
                    MAX_ANCHORED_EXCURSION_DEG
                    if args.anchor_current
                    else MAX_EXCURSION_DEG
                )
            ),
        )
        print("first_delta_gate=PASS")
        if args.recorded_rate:
            trace_plan = sdk_targets[:, list(selected_sdk)]
            if args.preposition_to_first:
                selected = list(selected_sdk)
                preposition_start = np.clip(
                    measured[selected],
                    lower[selected],
                    upper[selected],
                )
                preposition_plan, _ = _build_interpolated_targets(
                    preposition_start,
                    trace_plan[:1],
                    math.radians(max_speed_deg_s) / CONTROL_RATE_HZ,
                )
                plan_targets = np.vstack((preposition_plan, trace_plan[1:]))
            else:
                plan_targets = trace_plan
        else:
            plan_targets, _ = _build_interpolated_targets(
                measured[list(selected_sdk)],
                sdk_targets[:, list(selected_sdk)],
                math.radians(max_speed_deg_s) / CONTROL_RATE_HZ,
            )
        if not args.recorded_rate and plan_targets.shape[0] > MAX_EXECUTION_TICKS:
            raise RuntimeError(
                f"Replay needs {plan_targets.shape[0]} safety ticks "
                f"({plan_targets.shape[0] / CONTROL_RATE_HZ:.1f}s), above the "
                f"{MAX_EXECUTION_TICKS / CONTROL_RATE_HZ:.1f}s diagnostic limit."
            )
        print(
            f"bounded_plan: mode={'recorded_rate' if args.recorded_rate else 'interpolated'} "
            f"max_excursion={math.degrees(max_excursion):.3f}deg "
            f"speed={'trace' if args.recorded_rate else f'{max_speed_deg_s:g}deg/s'} "
            f"ticks={plan_targets.shape[0]} "
            f"duration={plan_targets.shape[0] / CONTROL_RATE_HZ:.3f}s"
        )
        if args.preposition_to_first:
            print(
                f"preposition_to_first=enabled speed<={max_speed_deg_s:g}deg/s "
                f"then_trace_rate={trace.policy_rate_hz:g}Hz"
            )

        if args.preflight:
            print("PREFLIGHT ONLY: hardware health/mapping checked; no command was sent.")
        else:
            # Once execution begins, always request a zero-force release in finally,
            # including when the first checked read or command call raises.
            command_may_have_been_sent = True
            await _execute_replay(
                args=args,
                io=io,
                profile=profile,
                trace=trace,
                rows=rows,
                selected_sdk=selected_sdk,
                sdk_targets=trace_sdk_targets,
                lower=lower,
                upper=upper,
                measured_tolerance=measured_tolerance,
                initial_delta_limit=initial_delta_limit,
                max_speed_deg_s=max_speed_deg_s,
                confirm_large_excursion=args.confirm_large_excursion,
                kp=kp,
                kd=kd,
                max_current_ma=max_current_ma,
                anchor_current=args.anchor_current,
                trace_baseline_sdk=trace_baseline_sdk,
                step_delta_limit=step_delta_limit,
                live_lower=live_lower,
                live_upper=live_upper,
                allow_passive_limit_outlier=args.allow_passive_limit_outlier,
                recorded_rate=args.recorded_rate,
                preposition_to_first=args.preposition_to_first,
                hold_final_s=hold_final_s,
            )
    except BaseException as exc:
        run_error = exc
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
        details = "; ".join(repr(error) for error in cleanup_errors)
        cleanup_error = RuntimeError(
            f"Replay cleanup failed ({details}); use the hardware E-stop."
        )
        if run_error is not None:
            raise cleanup_error from run_error
        raise cleanup_error from cleanup_errors[0]
    if run_error is not None:
        raise run_error
    return 0


async def _execute_replay(
    *,
    args: argparse.Namespace,
    io: Revo3SdkHandIO,
    profile: Revo3Profile,
    trace: ReplayTrace,
    rows: np.ndarray,
    selected_sdk: tuple[int, ...],
    sdk_targets: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    measured_tolerance: float,
    initial_delta_limit: float,
    max_speed_deg_s: float,
    confirm_large_excursion: bool,
    kp: float,
    kd: float,
    max_current_ma: float,
    anchor_current: bool = False,
    trace_baseline_sdk: np.ndarray | None = None,
    step_delta_limit: float = 0.05,
    live_lower: np.ndarray | None = None,
    live_upper: np.ndarray | None = None,
    allow_passive_limit_outlier: bool = False,
    recorded_rate: bool = False,
    preposition_to_first: bool = False,
    hold_final_s: float = 0.0,
) -> None:
    period = 1.0 / CONTROL_RATE_HZ
    kp_values = np.zeros(21, dtype=np.float32)
    kd_values = np.zeros(21, dtype=np.float32)
    kp_values[list(selected_sdk)] = kp
    kd_values[list(selected_sdk)] = kd
    effort = np.zeros(21, dtype=np.float32)
    selected_total_current_limit = (
        math.inf
        if recorded_rate
        else min(
            MAX_TOTAL_SELECTED_CURRENT_MA,
            max_current_ma * len(selected_sdk),
        )
    )
    passive_movement_limit = (
        MAX_ANCHORED_PASSIVE_MOVEMENT_RAD
        if anchor_current
        else MAX_PASSIVE_MOVEMENT_RAD
    )
    passive_lower = lower if live_lower is None else live_lower
    passive_upper = upper if live_upper is None else live_upper

    # This fresh checked sample is the authoritative execution baseline. The
    # earlier preflight-style printout is intentionally not trusted for motion.
    fresh_cycle_start = _monotonic()
    start_measured, tracking_error_samples = await _read_checked_health(
        io=io,
        profile=profile,
        selected_sdk=selected_sdk,
        measured_tolerance=measured_tolerance,
        selected_total_current_limit=selected_total_current_limit,
        start_measured=None,
        previous_target=None,
        tracking_error_samples=0,
        timeout_s=period * 0.8,
        anchor_current=anchor_current,
        selected_lower=lower,
        selected_upper=upper,
        passive_movement_limit=passive_movement_limit,
        allow_passive_limit_outlier=allow_passive_limit_outlier,
        recorded_rate=recorded_rate,
    )
    execution_targets = sdk_targets
    if anchor_current:
        if trace_baseline_sdk is None:
            raise RuntimeError("Current-anchored replay is missing its trace baseline.")
        execution_targets = _anchor_selected_targets(
            sdk_targets,
            trace_baseline_sdk,
            start_measured,
            selected_sdk,
        )
    _validate_selected_targets(
        profile,
        execution_targets,
        selected_sdk,
        lower,
        upper,
        step_delta_limit,
        rows,
        enforce_step_limit=not recorded_rate,
    )
    max_excursion = _validate_live_excursion(
        profile,
        execution_targets,
        selected_sdk,
        start_measured,
        math.inf if preposition_to_first else initial_delta_limit,
        require_confirmation=True,
        confirmed=confirm_large_excursion,
        max_excursion_deg=(
            math.inf
            if recorded_rate
            else (MAX_ANCHORED_EXCURSION_DEG if anchor_current else MAX_EXCURSION_DEG)
        ),
    )
    if recorded_rate:
        trace_plan = execution_targets[:, list(selected_sdk)].astype(
            np.float32,
            copy=False,
        )
        if preposition_to_first:
            selected = list(selected_sdk)
            preposition_start = np.clip(
                start_measured[selected],
                lower[selected],
                upper[selected],
            )
            preposition_plan, _ = _build_interpolated_targets(
                preposition_start,
                trace_plan[:1],
                math.radians(max_speed_deg_s) / CONTROL_RATE_HZ,
            )
            plan_targets = np.vstack((preposition_plan, trace_plan[1:]))
            plan_source_offsets = np.concatenate(
                (
                    np.full(max(0, preposition_plan.shape[0] - 1), -1, dtype=np.int64),
                    np.arange(trace_plan.shape[0], dtype=np.int64),
                )
            )
        else:
            plan_targets = trace_plan
            plan_source_offsets = np.arange(plan_targets.shape[0], dtype=np.int64)
    else:
        plan_targets, plan_source_offsets = _build_interpolated_targets(
            start_measured[list(selected_sdk)],
            execution_targets[:, list(selected_sdk)],
            math.radians(max_speed_deg_s) / CONTROL_RATE_HZ,
        )
    if not recorded_rate and plan_targets.shape[0] > MAX_EXECUTION_TICKS:
        raise RuntimeError(
            f"Fresh replay plan needs {plan_targets.shape[0]} safety ticks, above "
            f"the {MAX_EXECUTION_TICKS}-tick diagnostic limit. No target was sent."
        )

    print(
        "TRACE REPLAY ENABLED: "
        f"target_mode={'recorded_rate' if recorded_rate else ('current_anchored' if anchor_current else 'absolute_sim')} "
        f"frames={len(rows)} safety_rate={CONTROL_RATE_HZ:g}Hz "
        f"speed={'trace' if recorded_rate else f'{max_speed_deg_s:g}deg/s'} "
        f"ticks={plan_targets.shape[0]} "
        f"duration={plan_targets.shape[0] / CONTROL_RATE_HZ:.3f}s "
        f"max_excursion={math.degrees(max_excursion):.3f}deg "
        f"motors={list(selected_sdk)} kp={kp:g} kd={kd:g} "
        f"per_motor_cutoff={max_current_ma:g}mA "
        f"selected_total_cutoff="
        f"{'disabled' if math.isinf(selected_total_current_limit) else f'{selected_total_current_limit:g}mA'}"
    )
    if preposition_to_first:
        print(
            f"PREPOSITION ENABLED: speed<={max_speed_deg_s:g}deg/s; recorded trace "
            f"starts after reaching source row {int(rows[0])}."
        )
    if hold_final_s > 0.0:
        print(
            f"FINAL POSE HOLD ENABLED: duration={hold_final_s:g}s; "
            "health checks remain active until zero-force release."
        )
    print("execute_first_delta_gate=PASS (fresh checked sample)")

    previous_target: np.ndarray | None = None
    measured = start_measured
    for tick_index, selected_target in enumerate(plan_targets):
        loop_start = fresh_cycle_start if tick_index == 0 else _monotonic()
        if tick_index > 0:
            measured, tracking_error_samples = await _read_checked_health(
                io=io,
                profile=profile,
                selected_sdk=selected_sdk,
                measured_tolerance=measured_tolerance,
                selected_total_current_limit=selected_total_current_limit,
                start_measured=start_measured,
                previous_target=previous_target,
                tracking_error_samples=tracking_error_samples,
                timeout_s=period * 0.8,
                anchor_current=anchor_current,
                selected_lower=lower,
                selected_upper=upper,
                passive_movement_limit=passive_movement_limit,
                allow_passive_limit_outlier=allow_passive_limit_outlier,
                recorded_rate=recorded_rate,
            )

        command = np.clip(measured, passive_lower, passive_upper).astype(np.float32)
        command[list(selected_sdk)] = selected_target
        if anchor_current:
            _validate_selected_command(
                profile,
                command,
                selected_sdk,
                lower,
                upper,
                "current-anchored replay command",
            )
        else:
            profile.validate_sdk_position(command, "replay command target")
        io.validate_device_position(command)
        remaining = period - (_monotonic() - loop_start)
        if remaining <= 0.0:
            raise RuntimeError("Replay frame expired before command; release requested.")
        await asyncio.wait_for(
            io.send_mit_command_rad(
                command,
                kp=kp_values,
                kd=kd_values,
                effort_ma=effort,
            ),
            timeout=remaining,
        )
        previous_target = command.copy()

        source_offset = int(plan_source_offsets[tick_index])
        is_source_endpoint = (
            tick_index == plan_targets.shape[0] - 1
            or int(plan_source_offsets[tick_index + 1]) != source_offset
        )
        source_number = source_offset + 1
        if source_offset >= 0 and is_source_endpoint and (
            source_number == 1
            or source_number % args.print_every == 0
            or source_number == len(rows)
        ):
            row = int(rows[source_offset])
            selected = list(selected_sdk)
            print(
                f"replay={source_number}/{len(rows)} tick={tick_index + 1}/"
                f"{plan_targets.shape[0]} source_row={row} "
                f"step={int(trace.step_index[row])} "
                "target_deg="
                f"{np.array2string(np.rad2deg(command[selected]), precision=3)} "
                "measured_deg="
                f"{np.array2string(np.rad2deg(measured[selected]), precision=3)} "
                "current_mA="
                f"{np.array2string(io.last_motor_currents_ma[selected], precision=1)}"
            )
        sleep_time = period - (_monotonic() - loop_start)
        if sleep_time > 0.0:
            await asyncio.sleep(sleep_time)

    hold_ticks = int(math.ceil(hold_final_s * CONTROL_RATE_HZ))
    for hold_index in range(hold_ticks):
        loop_start = _monotonic()
        measured, tracking_error_samples = await _read_checked_health(
            io=io,
            profile=profile,
            selected_sdk=selected_sdk,
            measured_tolerance=measured_tolerance,
            selected_total_current_limit=selected_total_current_limit,
            start_measured=start_measured,
            previous_target=previous_target,
            tracking_error_samples=tracking_error_samples,
            timeout_s=period * 0.8,
            anchor_current=anchor_current,
            selected_lower=lower,
            selected_upper=upper,
            passive_movement_limit=passive_movement_limit,
            allow_passive_limit_outlier=allow_passive_limit_outlier,
            recorded_rate=recorded_rate,
        )
        remaining = period - (_monotonic() - loop_start)
        if remaining <= 0.0:
            raise RuntimeError("Final-pose hold frame expired; release requested.")
        await asyncio.wait_for(
            io.send_mit_command_rad(
                previous_target,
                kp=kp_values,
                kd=kd_values,
                effort_ma=effort,
            ),
            timeout=remaining,
        )
        if hold_index == 0 or (hold_index + 1) % int(CONTROL_RATE_HZ) == 0:
            print(
                f"hold={hold_index + 1}/{hold_ticks} "
                f"measured_deg={np.array2string(np.rad2deg(measured[list(selected_sdk)]), precision=3)}"
            )
        sleep_time = period - (_monotonic() - loop_start)
        if sleep_time > 0.0:
            await asyncio.sleep(sleep_time)

    # The final endpoint gets a full 20 Hz interval followed by one more checked
    # status sample before zero-force release.
    final_measured, _ = await _read_checked_health(
        io=io,
        profile=profile,
        selected_sdk=selected_sdk,
        measured_tolerance=measured_tolerance,
        selected_total_current_limit=selected_total_current_limit,
        start_measured=start_measured,
        previous_target=previous_target,
        tracking_error_samples=tracking_error_samples,
        timeout_s=period * 0.8,
        anchor_current=anchor_current,
        selected_lower=lower,
        selected_upper=upper,
        passive_movement_limit=passive_movement_limit,
        allow_passive_limit_outlier=allow_passive_limit_outlier,
        recorded_rate=recorded_rate,
    )
    print(
        "post_send_health=PASS final_measured_deg="
        f"{np.array2string(np.rad2deg(final_measured[list(selected_sdk)]), precision=3)}"
    )


async def _read_checked_health(
    *,
    io: Revo3SdkHandIO,
    profile: Revo3Profile,
    selected_sdk: tuple[int, ...],
    measured_tolerance: float,
    selected_total_current_limit: float,
    start_measured: np.ndarray | None,
    previous_target: np.ndarray | None,
    tracking_error_samples: int,
    timeout_s: float,
    anchor_current: bool,
    selected_lower: np.ndarray,
    selected_upper: np.ndarray,
    passive_movement_limit: float,
    allow_passive_limit_outlier: bool,
    recorded_rate: bool = False,
) -> tuple[np.ndarray, int]:
    measured = await asyncio.wait_for(
        io.read_position_rad(check_errors=True),
        timeout=timeout_s,
    )
    if anchor_current:
        _validate_anchored_measured_position(
            profile=profile,
            io=io,
            measured=measured,
            selected_sdk=selected_sdk,
            selected_lower=selected_lower,
            selected_upper=selected_upper,
            allow_passive_limit_outlier=allow_passive_limit_outlier,
        )
    else:
        profile.validate_sdk_position(
            measured,
            "replay measured position",
            tolerance_rad=measured_tolerance,
        )
    selected_current = np.abs(io.last_motor_currents_ma[list(selected_sdk)])
    if float(np.sum(selected_current)) > selected_total_current_limit:
        raise RuntimeError(
            "Selected motor total current exceeded the replay limit: "
            f"{float(np.sum(selected_current)):.1f}mA > "
            f"{selected_total_current_limit:.1f}mA."
        )
    if previous_target is not None:
        tracking_error_limit = (
            float(profile.safety.get("max_tracking_error_rad", MAX_TRACKING_ERROR_RAD))
            if recorded_rate
            else (
                MAX_ANCHORED_TRACKING_ERROR_RAD
                if anchor_current
                else MAX_TRACKING_ERROR_RAD
            )
        )
        tracking_error_sample_limit = (
            int(profile.safety.get("max_tracking_error_samples", MAX_TRACKING_ERROR_SAMPLES))
            if recorded_rate
            else (
                MAX_ANCHORED_TRACKING_ERROR_SAMPLES
                if anchor_current
                else MAX_TRACKING_ERROR_SAMPLES
            )
        )
        tracking_error = float(
            np.max(
                np.abs(
                    previous_target[list(selected_sdk)]
                    - measured[list(selected_sdk)]
                )
            )
        )
        tracking_error_samples = (
            tracking_error_samples + 1
            if tracking_error > tracking_error_limit
            else 0
        )
        if tracking_error_samples >= tracking_error_sample_limit:
            raise RuntimeError(
                "Selected-joint tracking error exceeded "
                f"{math.degrees(tracking_error_limit):.1f}deg for "
                f"{tracking_error_samples} samples; release requested."
            )
    if start_measured is not None:
        if anchor_current:
            selected_excursion = np.abs(
                measured[list(selected_sdk)] - start_measured[list(selected_sdk)]
            )
            failed = int(np.argmax(selected_excursion))
            if selected_excursion[failed] > math.radians(
                MAX_ANCHORED_MEASURED_EXCURSION_DEG
            ):
                sdk_index = selected_sdk[failed]
                raise RuntimeError(
                    f"Selected M{sdk_index} measured excursion reached "
                    f"{math.degrees(float(selected_excursion[failed])):.3f}deg, "
                    f"above the {MAX_ANCHORED_MEASURED_EXCURSION_DEG:g}deg "
                    "current-anchor safety envelope; release requested."
                )
        passive_mask = np.ones(21, dtype=np.bool_)
        passive_mask[list(selected_sdk)] = False
        passive_delta = np.where(
            passive_mask,
            np.abs(measured - start_measured),
            -1.0,
        )
        passive_index = int(np.argmax(passive_delta))
        if passive_delta[passive_index] > passive_movement_limit:
            raise RuntimeError(
                f"Passive M{passive_index} moved "
                f"{math.degrees(float(passive_delta[passive_index])):.3f}deg; "
                "release requested."
            )
    return measured, tracking_error_samples


def _validate_live_excursion(
    profile: Revo3Profile,
    sdk_targets: np.ndarray,
    selected_sdk: tuple[int, ...],
    measured: np.ndarray,
    initial_delta_limit: float,
    *,
    require_confirmation: bool,
    confirmed: bool,
    max_excursion_deg: float = MAX_EXCURSION_DEG,
) -> float:
    selected = list(selected_sdk)
    initial_delta = np.abs(sdk_targets[0, selected] - measured[selected])
    if float(np.max(initial_delta)) > initial_delta_limit:
        failed = int(np.argmax(initial_delta))
        sdk_index = selected_sdk[failed]
        raise RuntimeError(
            "First replay target is not continuous with the measured pose: "
            f"M{sdk_index} delta={math.degrees(float(initial_delta[failed])):.3f}deg "
            f"> {math.degrees(initial_delta_limit):.3f}deg. No command was sent."
        )
    excursion = np.abs(sdk_targets[:, selected] - measured[selected])
    max_excursion = float(np.max(excursion))
    if max_excursion > math.radians(max_excursion_deg):
        row_offset, selected_offset = np.unravel_index(
            int(np.argmax(excursion)),
            excursion.shape,
        )
        sdk_index = selected_sdk[int(selected_offset)]
        raise RuntimeError(
            f"Replay row {row_offset} moves M{sdk_index} "
            f"({profile.sdk_joint_order[sdk_index]}) "
            f"{math.degrees(max_excursion):.3f}deg from live start, above the "
            f"{max_excursion_deg:g}deg diagnostic limit. No command was sent."
        )
    if (
        require_confirmation
        and max_excursion > math.radians(CONFIRM_EXCURSION_DEG)
        and not confirmed
    ):
        raise RuntimeError(
            f"Replay reaches {math.degrees(max_excursion):.3f}deg from live start; "
            f"movement above {CONFIRM_EXCURSION_DEG:g}deg requires "
            "--confirm-large-excursion. No command was sent."
        )
    return max_excursion


def _build_interpolated_targets(
    start_selected_rad: np.ndarray,
    source_targets_rad: np.ndarray,
    max_step_rad: float,
) -> tuple[np.ndarray, np.ndarray]:
    start = np.asarray(start_selected_rad, dtype=np.float64).reshape(-1)
    source = np.asarray(source_targets_rad, dtype=np.float64)
    if (
        source.ndim != 2
        or source.shape[1] != start.shape[0]
        or source.shape[0] == 0
        or not np.isfinite(start).all()
        or not np.isfinite(source).all()
    ):
        raise ValueError("Replay interpolation inputs have invalid shapes or values.")
    step = float(max_step_rad)
    if not np.isfinite(step) or step <= 0.0:
        raise ValueError("Replay interpolation max step must be finite and positive.")

    planned: list[np.ndarray] = []
    source_offsets: list[int] = []
    previous = start.copy()
    for source_offset, endpoint in enumerate(source):
        delta = endpoint - previous
        steps = max(1, int(math.ceil(float(np.max(np.abs(delta))) / step - 1.0e-12)))
        for tick in range(1, steps + 1):
            planned.append(previous + (float(tick) / float(steps)) * delta)
            source_offsets.append(source_offset)
        previous = endpoint.copy()
    targets = np.asarray(planned, dtype=np.float32)
    if targets.shape[0] > 1 and float(np.max(np.abs(np.diff(targets, axis=0)))) > step + 1.0e-7:
        raise RuntimeError("Internal replay interpolation exceeded its speed bound.")
    return targets, np.asarray(source_offsets, dtype=np.int64)


def _resolve_sdk_indices(
    profile: Revo3Profile,
    selectors: list[str],
    all_joints: bool,
) -> tuple[int, ...]:
    if all_joints and selectors:
        raise ValueError("--all-joints cannot be combined with --joint.")
    if all_joints:
        return tuple(range(len(profile.sdk_joint_order)))
    resolved: list[int] = []
    for selector in selectors:
        text = str(selector).strip()
        upper = text.upper()
        if upper.startswith("M") and upper[1:].isdigit():
            index = int(upper[1:])
            if not 0 <= index < len(profile.sdk_joint_order):
                raise ValueError(f"SDK motor {text!r} is outside M0..M20.")
        elif upper.startswith("P") and upper[1:].isdigit():
            policy_index = int(upper[1:])
            if not 0 <= policy_index < len(profile.policy_joint_order):
                raise ValueError(f"Policy index {text!r} is outside P0..P20.")
            joint = profile.policy_joint_order[policy_index]
            index = profile.sdk_joint_order.index(joint)
        else:
            try:
                index = profile.sdk_joint_order.index(text)
            except ValueError as exc:
                raise ValueError(
                    f"Unknown joint selector {text!r}; use M0..M20, P0..P20, or an exact name."
                ) from exc
        if index in resolved:
            raise ValueError("--joint entries must resolve to unique SDK motors.")
        resolved.append(index)
    return tuple(resolved)


def _print_trace_summary(
    trace: ReplayTrace,
    profile: Revo3Profile,
    rows: np.ndarray,
    display_sdk: tuple[int, ...],
    selected_sdk: tuple[int, ...],
    trajectory_source: str,
    trajectory_policy_rad: np.ndarray,
) -> None:
    policy_targets = trajectory_policy_rad[rows]
    sdk_targets = _sdk_targets(profile, policy_targets)
    baseline_policy = (
        trace.target_before_policy_rad[rows[:1]]
        if trajectory_source == "target"
        else policy_targets[:1]
    )
    sdk_baseline = _sdk_targets(profile, baseline_policy)[0]
    selected_set = set(selected_sdk)
    print(f"trace={trace.path}")
    print(
        f"checkpoint={trace.metadata.get('checkpoint', '')} "
        f"sha256={trace.metadata['checkpoint_sha256']}"
    )
    print(
        f"task={trace.metadata.get('task', '')} cache_row="
        f"{trace.metadata.get('cache_row_actual')} recorded_frames={trace.frame_count} "
        f"selected_rows=[{int(rows[0])},{int(rows[-1]) + 1}) rate={trace.policy_rate_hz:g}Hz "
        f"trajectory_source={trajectory_source}"
    )
    print(
        "sel Pidx SDK joint offset_deg first_sdk_deg min_sdk_deg "
        "max_sdk_deg first_delta_deg span_deg max_step_deg"
    )
    for sdk_index in display_sdk:
        joint = profile.sdk_joint_order[sdk_index]
        policy_index = profile.policy_joint_order.index(joint)
        values = sdk_targets[:, sdk_index]
        max_step = (
            float(np.max(np.abs(np.diff(values)))) if values.shape[0] > 1 else 0.0
        )
        marker = "*" if sdk_index in selected_set else "-"
        print(
            f" {marker}  P{policy_index:02d} M{sdk_index:02d} {joint:<28} "
            f"{math.degrees(float(profile.sdk_offset_rad[sdk_index])):10.3f} "
            f"{math.degrees(float(values[0])):9.3f} "
            f"{math.degrees(float(np.min(values))):8.3f} "
            f"{math.degrees(float(np.max(values))):8.3f} "
            f"{math.degrees(float(values[0] - sdk_baseline[sdk_index])):15.3f} "
            f"{math.degrees(float(np.ptp(values))):8.3f} "
            f"{math.degrees(max_step):12.3f}"
        )
    print(
        "Mapping note: configuration consistency does not prove the physical motor order; "
        "visually verify one low-gain joint at a time, especially thumb M16..M20."
    )


def _sdk_targets(profile: Revo3Profile, policy_targets: np.ndarray) -> np.ndarray:
    return (
        policy_targets[:, profile.policy_to_sdk_perm]
        + profile.sdk_offset_rad.reshape(1, -1)
    ).astype(np.float32)


def _anchor_selected_targets(
    trace_sdk_targets: np.ndarray,
    trace_baseline_sdk: np.ndarray,
    live_start_sdk: np.ndarray,
    selected_sdk: tuple[int, ...],
) -> np.ndarray:
    targets = np.asarray(trace_sdk_targets, dtype=np.float32)
    baseline = np.asarray(trace_baseline_sdk, dtype=np.float32).reshape(-1)
    live_start = np.asarray(live_start_sdk, dtype=np.float32).reshape(-1)
    if (
        targets.ndim != 2
        or targets.shape[0] == 0
        or targets.shape[1] != 21
        or baseline.shape != (21,)
        or live_start.shape != (21,)
        or not selected_sdk
        or not np.isfinite(targets).all()
        or not np.isfinite(baseline).all()
        or not np.isfinite(live_start).all()
    ):
        raise ValueError("Current-anchor inputs have invalid shapes or values.")
    selected = list(selected_sdk)
    displacement = (
        targets[:, selected].astype(np.float64)
        - baseline[selected].astype(np.float64).reshape(1, -1)
    )
    anchored = targets.copy()
    anchored[:, selected] = (
        live_start[selected].astype(np.float64).reshape(1, -1) + displacement
    ).astype(np.float32)
    return anchored


def _live_sdk_envelope(
    io: Revo3SdkHandIO,
    margin_rad: float,
) -> tuple[np.ndarray, np.ndarray]:
    if io.device_position_lower_rad is None or io.device_position_upper_rad is None:
        raise RuntimeError("Live device position limits are unavailable.")
    lower = (io.device_position_lower_rad + margin_rad).astype(np.float32)
    upper = (io.device_position_upper_rad - margin_rad).astype(np.float32)
    invalid = np.flatnonzero(upper <= lower)
    if invalid.size:
        raise RuntimeError(
            "Live command envelope is empty after its inward margin at "
            f"{invalid.tolist()}."
        )
    return lower, upper


def _effective_sdk_envelope(
    profile: Revo3Profile,
    io: Revo3SdkHandIO,
    margin_rad: float,
) -> tuple[np.ndarray, np.ndarray]:
    live_lower, live_upper = _live_sdk_envelope(io, margin_rad)
    lower = np.maximum(
        profile.sdk_position_lower_rad,
        live_lower,
    ).astype(np.float32)
    upper = np.minimum(
        profile.sdk_position_upper_rad,
        live_upper,
    ).astype(np.float32)
    invalid = np.flatnonzero(upper <= lower)
    if invalid.size:
        raise RuntimeError(
            "Static/live command-limit intersection is empty at "
            f"{invalid.tolist()}."
        )
    return lower, upper


def _validate_anchored_measured_position(
    *,
    profile: Revo3Profile,
    io: Revo3SdkHandIO,
    measured: np.ndarray,
    selected_sdk: tuple[int, ...],
    selected_lower: np.ndarray,
    selected_upper: np.ndarray,
    allow_passive_limit_outlier: bool = False,
) -> None:
    position = np.asarray(measured, dtype=np.float32).reshape(-1)
    if position.shape != (21,) or not np.isfinite(position).all():
        raise ValueError("Replay measured position must contain 21 finite values.")
    if io.device_position_lower_rad is None or io.device_position_upper_rad is None:
        raise RuntimeError("Live device position limits are unavailable.")
    live_below = position < (
        io.device_position_lower_rad - MAX_LIVE_MEASURED_TOLERANCE_RAD
    )
    live_above = position > (
        io.device_position_upper_rad + MAX_LIVE_MEASURED_TOLERANCE_RAD
    )
    selected_mask = np.zeros(21, dtype=np.bool_)
    selected_mask[list(selected_sdk)] = True
    selected_live_invalid = np.flatnonzero((live_below | live_above) & selected_mask)
    if selected_live_invalid.size:
        details = ", ".join(
            f"M{index}={math.degrees(float(position[index])):.2f}deg outside live "
            f"[{math.degrees(float(io.device_position_lower_rad[index])):.2f},"
            f"{math.degrees(float(io.device_position_upper_rad[index])):.2f}]deg"
            for index in selected_live_invalid
        )
        raise RuntimeError(
            "Current-anchored selected-joint measurements violate device-reported "
            f"limits: {details}. No command was sent."
        )
    passive_live_invalid = np.flatnonzero(
        (live_below | live_above) & ~selected_mask
    )
    if passive_live_invalid.size:
        outlier = np.maximum(
            io.device_position_lower_rad - position,
            position - io.device_position_upper_rad,
        )
        excessive = passive_live_invalid[
            outlier[passive_live_invalid] > MAX_PASSIVE_LIVE_LIMIT_OUTLIER_RAD
        ]
        details = ", ".join(
            f"M{index}={math.degrees(float(position[index])):.2f}deg "
            f"({math.degrees(float(outlier[index])):.2f}deg beyond live range)"
            for index in passive_live_invalid
        )
        if excessive.size:
            raise RuntimeError(
                "Passive encoder readings exceed the hard 5deg live-limit outlier "
                f"cap: {details}. No command was sent."
            )
        if not allow_passive_limit_outlier:
            raise RuntimeError(
                "Passive zero-gain encoder readings sit outside device-reported "
                f"limits: {details}. Re-run only after review with "
                "--allow-passive-limit-outlier; no command was sent."
            )
    _validate_selected_command(
        profile,
        position,
        selected_sdk,
        selected_lower,
        selected_upper,
        "current-anchored measured position",
    )


def _validate_selected_command(
    profile: Revo3Profile,
    position: np.ndarray,
    selected_sdk: tuple[int, ...],
    lower: np.ndarray,
    upper: np.ndarray,
    name: str,
) -> None:
    values = np.asarray(position, dtype=np.float32).reshape(-1)
    if values.shape != (21,) or not np.isfinite(values).all():
        raise ValueError(f"{name} must contain 21 finite values.")
    selected = np.asarray(selected_sdk, dtype=np.int64)
    invalid_mask = (values[selected] < lower[selected]) | (
        values[selected] > upper[selected]
    )
    invalid_offsets = np.flatnonzero(invalid_mask)
    if invalid_offsets.size:
        details = ", ".join(
            f"M{int(selected[offset])}={math.degrees(float(values[selected[offset]])):.2f}deg "
            f"outside [{math.degrees(float(lower[selected[offset]])):.2f},"
            f"{math.degrees(float(upper[selected[offset]])):.2f}]deg"
            for offset in invalid_offsets
        )
        raise RuntimeError(
            f"{name} violates the selected-joint inward command envelope: "
            f"{details}. No command was sent."
        )


def _print_passive_nominal_outliers(
    profile: Revo3Profile,
    io: Revo3SdkHandIO,
    measured: np.ndarray,
    selected_sdk: tuple[int, ...],
) -> None:
    passive = np.ones(21, dtype=np.bool_)
    passive[list(selected_sdk)] = False
    position = np.asarray(measured, dtype=np.float32)
    invalid = np.flatnonzero(
        passive
        & (
            (position < profile.sdk_position_lower_rad)
            | (position > profile.sdk_position_upper_rad)
        )
    )
    for index in invalid:
        live_text = "unavailable"
        live_outlier = False
        if io.device_position_lower_rad is not None and io.device_position_upper_rad is not None:
            live_text = (
                f"[{math.degrees(float(io.device_position_lower_rad[index])):.2f},"
                f"{math.degrees(float(io.device_position_upper_rad[index])):.2f}]deg"
            )
            live_outlier = bool(
                position[index] < io.device_position_lower_rad[index]
                or position[index] > io.device_position_upper_rad[index]
            )
        outlier_text = (
            " Explicit passive live-limit exception is active; its command slot "
            "will be clipped into the live envelope."
            if live_outlier
            else ""
        )
        print(
            "WARNING: passive zero-gain "
            f"M{index} ({profile.sdk_joint_order[index]}) measured "
            f"{math.degrees(float(position[index])):.2f}deg outside nominal "
            f"[{math.degrees(float(profile.sdk_position_lower_rad[index])):.2f},"
            f"{math.degrees(float(profile.sdk_position_upper_rad[index])):.2f}]deg; "
            f"device_live={live_text}. It will receive kp=kd=effort=0."
            f"{outlier_text}",
            file=sys.stderr,
        )


def _validate_selected_targets(
    profile: Revo3Profile,
    sdk_targets: np.ndarray,
    selected_sdk: tuple[int, ...],
    lower: np.ndarray,
    upper: np.ndarray,
    step_delta_limit: float,
    source_rows: np.ndarray,
    *,
    enforce_step_limit: bool = True,
) -> None:
    selected = list(selected_sdk)
    values = sdk_targets[:, selected]
    invalid = np.argwhere((values < lower[selected]) | (values > upper[selected]))
    if invalid.size:
        row_offset, selected_offset = (int(value) for value in invalid[0])
        sdk_index = selected[selected_offset]
        value = float(values[row_offset, selected_offset])
        raise RuntimeError(
            f"Replay target source row {int(source_rows[row_offset])} for M{sdk_index} "
            f"({profile.sdk_joint_order[sdk_index]}) is {math.degrees(value):.3f}deg, "
            "outside the inward static/live command envelope. No command was sent."
        )
    if enforce_step_limit and values.shape[0] > 1:
        deltas = np.abs(np.diff(values, axis=0))
        maximum = float(np.max(deltas))
        if maximum > step_delta_limit:
            row_offset, selected_offset = np.unravel_index(
                int(np.argmax(deltas)),
                deltas.shape,
            )
            sdk_index = selected[int(selected_offset)]
            raise RuntimeError(
                "Replay target source step "
                f"{int(source_rows[row_offset])}->{int(source_rows[row_offset + 1])} "
                f"for M{sdk_index} "
                f"is {math.degrees(maximum):.3f}deg, above the configured "
                f"{math.degrees(step_delta_limit):.3f}deg gate. No command was sent."
            )


def _print_live_alignment(
    profile: Revo3Profile,
    selected_sdk: tuple[int, ...],
    measured: np.ndarray,
    first_target: np.ndarray,
    delta: np.ndarray,
) -> None:
    print("motor policy joint measured_deg first_target_deg abs_delta_deg")
    for offset, sdk_index in enumerate(selected_sdk):
        joint = profile.sdk_joint_order[sdk_index]
        policy_index = profile.policy_joint_order.index(joint)
        print(
            f"M{sdk_index:02d} P{policy_index:02d} {joint:<28} "
            f"{math.degrees(float(measured[sdk_index])):12.3f} "
            f"{math.degrees(float(first_target[sdk_index])):16.3f} "
            f"{math.degrees(float(delta[offset])):13.3f}"
        )


def _verify_expected_artifact_hashes(args: argparse.Namespace) -> None:
    bindings = (
        (
            "trace",
            getattr(args, "expected_trace_sha256", None),
            getattr(args, "trace_npz", None),
        ),
        (
            "checkpoint",
            getattr(args, "expected_checkpoint_sha256", None),
            getattr(args, "checkpoint", None),
        ),
        (
            "profile",
            getattr(args, "expected_profile_sha256", None),
            getattr(args, "profile", None),
        ),
    )
    for label, expected_value, path_value in bindings:
        if expected_value is None:
            continue
        expected = str(expected_value).lower()
        if len(expected) != 64 or any(
            char not in "0123456789abcdef" for char in expected
        ):
            raise ValueError(
                f"Expected {label} SHA256 is not a 64-character hex digest."
            )
        if path_value is None:
            raise ValueError(f"Expected {label} SHA256 was supplied without its path.")
        path = Path(path_value).expanduser().resolve()
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        actual = digest.hexdigest()
        if actual != expected:
            raise ValueError(
                f"Bound {label} SHA256 mismatch: expected {expected}, got {actual}."
            )


def _positive(mapping: dict, key: str, default: float) -> float:
    value = float(mapping.get(key, default))
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{key} must be finite and positive.")
    return value


def _nonnegative(mapping: dict, key: str, default: float) -> float:
    value = float(mapping.get(key, default))
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(f"{key} must be finite and non-negative.")
    return value


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return asyncio.run(async_main(args))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
