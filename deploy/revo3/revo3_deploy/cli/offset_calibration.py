from __future__ import annotations

import argparse
from pathlib import Path
import tempfile

import numpy as np
import yaml

from revo3_deploy.replay_trace import ReplayTrace
from revo3_deploy.robot_profile import Revo3Profile


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create auditable candidate Revo3 sim-to-real joint-offset profiles "
            "without modifying the source profile."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    for name in ("init", "show"):
        command = subparsers.add_parser(name)
        _add_pose_arguments(command)
        if name == "init":
            command.add_argument("--output-profile", required=True)

    adjust = subparsers.add_parser("adjust")
    adjust.add_argument("--profile", required=True)
    adjust.add_argument("--output-profile", required=True)
    adjust.add_argument(
        "--add",
        action="append",
        default=[],
        metavar="MOTOR=DEG",
        help="Add a signed degree correction, for example --add M16=+2.5.",
    )
    adjust.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="MOTOR=DEG",
        help="Set an absolute candidate offset in degrees, for example --set M16=8.0.",
    )
    return parser


def _add_pose_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--profile", required=True)
    parser.add_argument("--trace-npz", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument(
        "--trajectory-source",
        choices=("measured", "target"),
        default="measured",
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "adjust":
        return _adjust(args)
    return _show_or_init(args)


def _show_or_init(args: argparse.Namespace) -> int:
    profile = Revo3Profile.load(args.profile)
    trace = ReplayTrace.load(args.trace_npz, profile, checkpoint_path=args.checkpoint)
    frame = int(args.frame)
    if frame < 0 or frame >= trace.usable_frame_count:
        raise ValueError(
            f"--frame {frame} is outside 0..{trace.usable_frame_count - 1}."
        )
    policy_pose = trace.trajectory_policy_rad(args.trajectory_source)[frame]
    sim_sdk = policy_pose[profile.policy_to_sdk_perm]
    candidate_sdk = sim_sdk + profile.sdk_offset_rad
    _validate_pose(profile, candidate_sdk)
    _print_pose(profile, frame, args.trajectory_source, sim_sdk, candidate_sdk)

    if args.command == "init":
        cfg = _read_yaml(args.profile)
        offset = dict(cfg.get("sim2real_joint_offset") or {})
        offset.update(
            {
                "order": "sdk_joint_order",
                "units": "radians",
                "values": [float(value) for value in profile.sdk_offset_rad],
                "calibration_trace": str(Path(args.trace_npz).expanduser().resolve()),
                "calibration_frame": frame,
                "calibration_source": args.trajectory_source,
            }
        )
        cfg["sim2real_joint_offset"] = offset
        calibration = dict(cfg.get("calibration") or {})
        calibration["status"] = "unverified"
        cfg["calibration"] = calibration
        output = _write_new_yaml(args.output_profile, cfg)
        print(f"candidate_profile={output}")
    return 0


def _adjust(args: argparse.Namespace) -> int:
    if not args.add and not args.set:
        raise ValueError("adjust requires at least one --add or --set entry.")
    profile = Revo3Profile.load(args.profile)
    offsets = profile.sdk_offset_rad.astype(np.float64, copy=True)
    for expression in args.set:
        index, value_deg = _parse_edit(expression, len(offsets))
        offsets[index] = np.deg2rad(value_deg)
    for expression in args.add:
        index, value_deg = _parse_edit(expression, len(offsets))
        offsets[index] += np.deg2rad(value_deg)
    if not np.isfinite(offsets).all():
        raise ValueError("Candidate offsets must be finite.")

    cfg = _read_yaml(args.profile)
    offset = dict(cfg.get("sim2real_joint_offset") or {})
    offset.update(
        {
            "order": "sdk_joint_order",
            "units": "radians",
            "values": [float(value) for value in offsets],
        }
    )
    cfg["sim2real_joint_offset"] = offset
    calibration = dict(cfg.get("calibration") or {})
    calibration["status"] = "unverified"
    cfg["calibration"] = calibration
    output = _write_new_yaml(args.output_profile, cfg)
    print("SDK joint candidate_offset_deg")
    for index, (joint, value) in enumerate(zip(profile.sdk_joint_order, offsets)):
        print(f"M{index:02d} {joint:<28} {np.rad2deg(value):+10.3f}")
    print(f"candidate_profile={output}")
    return 0


def _parse_edit(expression: str, joint_count: int) -> tuple[int, float]:
    text = str(expression).strip()
    if "=" not in text:
        raise ValueError(f"Offset edit must use MOTOR=DEG, got {text!r}.")
    motor, raw_value = text.split("=", 1)
    motor = motor.strip().upper()
    if not motor.startswith("M") or not motor[1:].isdigit():
        raise ValueError(f"Offset motor must be M0..M{joint_count - 1}, got {motor!r}.")
    index = int(motor[1:])
    if not 0 <= index < joint_count:
        raise ValueError(f"Offset motor {motor} is outside M0..M{joint_count - 1}.")
    value = float(raw_value)
    if not np.isfinite(value):
        raise ValueError("Offset edit degrees must be finite.")
    return index, value


def _validate_pose(profile: Revo3Profile, candidate_sdk: np.ndarray) -> None:
    profile.validate_sdk_position(candidate_sdk, "candidate calibration pose")


def _print_pose(
    profile: Revo3Profile,
    frame: int,
    source: str,
    sim_sdk: np.ndarray,
    candidate_sdk: np.ndarray,
) -> None:
    print(f"calibration_frame={frame} trajectory_source={source}")
    print("SDK Policy joint sim_deg offset_deg candidate_sdk_deg")
    for sdk_index, joint in enumerate(profile.sdk_joint_order):
        policy_index = profile.policy_joint_order.index(joint)
        print(
            f"M{sdk_index:02d} P{policy_index:02d} {joint:<28} "
            f"{np.rad2deg(sim_sdk[sdk_index]):9.3f} "
            f"{np.rad2deg(profile.sdk_offset_rad[sdk_index]):+10.3f} "
            f"{np.rad2deg(candidate_sdk[sdk_index]):12.3f}"
        )


def _read_yaml(path: str | Path) -> dict:
    source = Path(path).expanduser().resolve()
    with source.open("r", encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream) or {}
    if not isinstance(cfg, dict):
        raise ValueError("Profile root must be a YAML mapping.")
    return cfg


def _write_new_yaml(path: str | Path, cfg: dict) -> Path:
    output = Path(path).expanduser().resolve()
    if output.exists():
        raise FileExistsError(
            f"Refusing to overwrite candidate profile {output}; choose a new versioned name."
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=output.parent,
        prefix=f".{output.name}.",
        suffix=".tmp",
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
        yaml.safe_dump(cfg, stream, sort_keys=False, allow_unicode=True)
    temporary.replace(output)
    return output


if __name__ == "__main__":
    raise SystemExit(main())
