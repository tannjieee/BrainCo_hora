from __future__ import annotations

import argparse
import hashlib
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np

from revo3_deploy.policy_runner import Revo3PolicyRunner
from revo3_deploy.sdk_hand_io import _load_sdk
from revo3_deploy.vision_touch import VisionTouchCollector


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate the tactile HORA ONNX artifact without connecting to hardware."
    )
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--provider", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument(
        "--check-sdk",
        action="store_true",
        help=(
            "Also verify bc-revo3-sdk 1.5.1 and, when configured, the pyvitaisdk "
            "API plus local VisionTouch force-model files."
        ),
    )
    parser.add_argument(
        "--joint-pos-npy",
        default=None,
        help="Optional grasp-cache .npy; select a row with --joint-pos-row.",
    )
    parser.add_argument(
        "--joint-pos-row",
        type=int,
        default=0,
        help="Zero-based row selected from --joint-pos-npy (default: 0).",
    )
    parser.add_argument(
        "--print-alignment",
        action="store_true",
        help="Print the exact joint permutation and input/output scaling contract.",
    )
    parser.add_argument(
        "--contact-forces",
        nargs=5,
        type=float,
        default=[0.0] * 5,
        metavar=("THUMB", "INDEX", "MIDDLE", "RING", "LITTLE"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.steps <= 0:
        raise ValueError("--steps must be positive.")
    if args.joint_pos_row < 0:
        raise ValueError("--joint-pos-row must be non-negative.")
    if args.joint_pos_row != 0 and args.joint_pos_npy is None:
        raise ValueError("--joint-pos-row requires --joint-pos-npy.")

    runner = Revo3PolicyRunner.create(
        args.onnx,
        args.metadata,
        args.profile,
        provider=args.provider,
    )
    position = _initial_position(runner, args.joint_pos_npy, args.joint_pos_row)
    contacts = np.asarray(args.contact_forces, dtype=np.float32)

    durations_ms: list[float] = []
    result = None
    for _ in range(args.steps):
        started = time.perf_counter()
        result = runner.step(position, contacts)
        durations_ms.append((time.perf_counter() - started) * 1000.0)
    assert result is not None

    onnx_path = Path(args.onnx).expanduser().resolve()
    sdk_version = _validate_sdk_surface() if args.check_sdk else "not checked"
    vision_cfg = dict(runner.profile.tactile.get("vision_touch") or {})
    vision_version = "not configured"
    if bool(vision_cfg.get("enabled", False)):
        vision_version = (
            _validate_vision_touch_surface(vision_cfg)
            if args.check_sdk
            else "not checked"
        )
    print("Validation passed")
    print(f"  ONNX: {onnx_path}")
    print(f"  SHA256: {_sha256(onnx_path)}")
    print(f"  providers: {runner.providers}")
    print(f"  bc-revo3-sdk: {sdk_version}")
    print(f"  pyvitaisdk4bc/VisionTouch models: {vision_version}")
    print(
        "  ABI: "
        f"obs[1,{runner.contract.obs_dim}] + "
        f"proprio_hist[1,{runner.contract.history_len},{runner.contract.frame_dim}] "
        f"-> action[1,{runner.contract.action_dim}]"
    )
    print(f"  rate/action_scale: {runner.rate_hz:g} Hz / {runner.contract.action_scale:.9f}")
    print(
        "  inference ms: "
        f"mean={np.mean(durations_ms):.3f}, p95={np.percentile(durations_ms, 95):.3f}, "
        f"max={np.max(durations_ms):.3f}"
    )
    print(f"  final action: {np.array2string(result.action, precision=4)}")
    print(f"  final target rad: {np.array2string(result.policy_target_rad, precision=4)}")
    if args.print_alignment:
        _print_alignment_report(runner, position, contacts)
    if runner.profile.calibration_status != "verified":
        print(
            "  NOTE: profile calibration is unverified; offline validation does not "
            "authorize motion."
        )
    return 0


def _initial_position(
    runner: Revo3PolicyRunner,
    cache_path: str | None,
    row: int,
) -> np.ndarray:
    if cache_path is None:
        return (
            (runner.profile.joint_lower_policy + runner.profile.joint_upper_policy) * 0.5
        ).astype(np.float32)
    cache = np.load(Path(cache_path).expanduser().resolve(), allow_pickle=False)
    if cache.ndim != 2 or cache.shape[0] == 0 or cache.shape[1] < runner.contract.action_dim:
        raise ValueError("--joint-pos-npy must contain at least one row with 21 values.")
    if row >= cache.shape[0]:
        raise ValueError(
            f"--joint-pos-row {row} is outside cache with {cache.shape[0]} rows."
        )
    position = np.asarray(cache[row, : runner.contract.action_dim], dtype=np.float32)
    if not np.isfinite(position).all():
        raise ValueError("--joint-pos-npy contains non-finite joint positions.")
    return position


def _print_alignment_report(
    runner: Revo3PolicyRunner,
    position: np.ndarray,
    contacts: np.ndarray,
) -> None:
    profile = runner.profile
    contract = runner.contract
    q_unscaled = (
        2.0 * position - contract.joint_upper_rad - contract.joint_lower_rad
    ) / (contract.joint_upper_rad - contract.joint_lower_rad)
    print("Alignment report")
    print("  raw frame[47] = q_unscaled[21] + current_target_rad[21] + force_N[5]")
    print("  obs = frames[t-2:t+1] flattened oldest->newest; hist = frames[t-29:t+1]")
    print("  RunningMeanStd is inside ONNX; deployment must not normalize again")
    print(
        f"  action target increment = action / 24 rad = action * "
        f"{np.rad2deg(contract.action_scale):.6f} deg per 20 Hz frame"
    )
    print(
        "  force order [N]: "
        + ", ".join(contract.contact_order)
        + f"; supplied={np.array2string(contacts, precision=4)}"
    )
    print("  Pidx SDK joint train_deg target_deg offset_deg position_deg q_unscaled")
    for policy_index, joint in enumerate(contract.joint_order):
        sdk_index = profile.sdk_joint_order.index(joint)
        print(
            f"  P{policy_index:02d} M{sdk_index:02d} {joint:24s} "
            f"[{np.rad2deg(contract.joint_lower_rad[policy_index]):7.3f},"
            f"{np.rad2deg(contract.joint_upper_rad[policy_index]):7.3f}] "
            f"[{np.rad2deg(profile.target_lower_policy[policy_index]):7.3f},"
            f"{np.rad2deg(profile.target_upper_policy[policy_index]):7.3f}] "
            f"{np.rad2deg(profile.policy_offset_rad[policy_index]):+8.3f} "
            f"{np.rad2deg(position[policy_index]):9.3f} "
            f"{q_unscaled[policy_index]:+10.5f}"
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_sdk_surface() -> str:
    sdk = _load_sdk()
    required_module = (
        "init_logging",
        "revo3_auto_detect_modbus",
        "modbus_open",
        "modbus_close",
        "DeviceContext",
        "Baudrate",
        "HandType",
        "StarkHardwareType",
        "TouchDataMode",
        "TouchModuleValueType",
        "MatrixTouchOutputMode",
    )
    required_context = (
        "revo3_get_device_info",
        "revo3_get_touch_vendor",
        "revo3_get_all_touch_modules_enabled",
        "revo3_get_touch_data_type",
        "revo3_get_touch_module_value_type",
        "revo3_get_matrix_touch_output_mode",
        "revo3_get_matrix_touch_module_output_mode",
        "revo3_get_all_motor_positions",
        "revo3_get_motor_status_data",
        "revo3_get_motor_online_status",
        "revo3_get_all_joint_position_limits",
        "revo3_get_touch_summary",
        "revo3_get_touch_module_data",
        "revo3_set_all_touch_modules_enabled",
        "revo3_set_touch_data_type",
        "revo3_set_touch_module_value_type",
        "revo3_set_matrix_touch_output_mode",
        "revo3_set_all_mit_params",
        "revo3_set_all_mit_params_without_retry",
    )
    missing = [name for name in required_module if not hasattr(sdk, name)]
    context = getattr(sdk, "DeviceContext", None)
    if context is not None:
        missing.extend(name for name in required_context if not hasattr(context, name))
    if missing:
        raise RuntimeError(f"bc-revo3-sdk is missing required APIs: {missing}.")
    return str(sdk.__version__)


def _validate_vision_touch_surface(config: dict) -> str:
    collector = VisionTouchCollector(config)
    try:
        import pyvitaisdk
    except ImportError as exc:
        raise RuntimeError(
            "VisionTouch is enabled, but pyvitaisdk4bc is not installed."
        ) from exc
    required = ("VTSDataType", "VTSDeviceFinder", "VTSensor")
    missing_api = [name for name in required if not hasattr(pyvitaisdk, name)]
    if missing_api:
        raise RuntimeError(f"pyvitaisdk is missing required APIs: {missing_api}.")
    missing_models = [
        collector.model_dir / serial / f"{serial}.onnx.enc"
        for serial in collector.sensor_serials
        if not (collector.model_dir / serial / f"{serial}.onnx.enc").is_file()
    ]
    if missing_models:
        raise RuntimeError(f"VisionTouch force-model files are missing: {missing_models}.")
    try:
        package_version = version("pyvitaisdk4bc")
    except PackageNotFoundError:
        package_version = "installed (distribution version unavailable)"
    mapping = "verified" if collector.mapping_verified else "UNVERIFIED mapping"
    return f"{package_version}; 5 models present; {mapping}"


if __name__ == "__main__":
    raise SystemExit(main())
