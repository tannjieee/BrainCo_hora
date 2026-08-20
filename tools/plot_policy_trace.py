#!/usr/bin/env python3
"""Plot joint, force, action, and object-state curves from a policy trace NPZ."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


JOINT_DIM = 21
FINGER_DIM = 5
DEFAULT_CONTACT_ORDER = ("thumb", "index", "middle", "ring", "little")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Plot all 21 joint angles and all five fingertip forces from a HORA "
            "simulator or hardware policy-trace NPZ."
        )
    )
    parser.add_argument("trace_npz", help="Input policy-trace .npz file")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for PNG files (default: next to the input trace)",
    )
    parser.add_argument(
        "--angle-unit",
        choices=("deg", "rad"),
        default="deg",
        help="Displayed joint-angle unit (default: deg)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=160,
        help="PNG resolution in dots per inch (default: 160)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Also open the Matplotlib windows after saving the PNG files",
    )
    return parser


def _metadata(payload: Any) -> dict[str, Any]:
    if "metadata_json" not in payload.files:
        return {}
    raw = np.asarray(payload["metadata_json"])
    if raw.shape != ():
        raise ValueError("metadata_json must be a scalar JSON string")
    decoded = json.loads(str(raw.item()))
    if not isinstance(decoded, dict):
        raise ValueError("metadata_json must decode to an object")
    return decoded


def _matrix(payload: Any, name: str, columns: int) -> np.ndarray:
    if name not in payload.files:
        raise ValueError(f"Trace is missing required array {name!r}")
    value = np.asarray(payload[name], dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != columns:
        raise ValueError(
            f"Trace array {name!r} must have shape (T,{columns}), got {value.shape}"
        )
    if value.shape[0] == 0:
        raise ValueError(f"Trace array {name!r} contains no frames")
    if not np.isfinite(value).all():
        raise ValueError(f"Trace array {name!r} contains non-finite values")
    return value


def _time_axis(payload: Any, frame_count: int) -> np.ndarray:
    if "sample_time_s" not in payload.files:
        return np.arange(frame_count, dtype=np.float64)
    time_s = np.asarray(payload["sample_time_s"], dtype=np.float64)
    if time_s.shape != (frame_count,):
        raise ValueError(
            "Trace array 'sample_time_s' must have shape "
            f"({frame_count},), got {time_s.shape}"
        )
    if not np.isfinite(time_s).all() or np.any(np.diff(time_s) < 0.0):
        raise ValueError("Trace sample_time_s must be finite and non-decreasing")
    return time_s


def _labels(metadata: dict[str, Any], key: str, count: int, prefix: str) -> list[str]:
    raw = metadata.get(key)
    if isinstance(raw, list) and len(raw) == count:
        return [str(item).removeprefix("right_").replace("_joint", "") for item in raw]
    return [f"{prefix}{index:02d}" for index in range(count)]


def _plot_joint_angles(
    plt: Any,
    time_s: np.ndarray,
    measured_rad: np.ndarray,
    target_rad: np.ndarray | None,
    joint_names: list[str],
    angle_unit: str,
    title: str,
) -> Any:
    scale = 180.0 / np.pi if angle_unit == "deg" else 1.0
    unit_text = "deg" if angle_unit == "deg" else "rad"
    figure, axes = plt.subplots(7, 3, figsize=(15, 20), sharex=True)
    for joint_index, axis in enumerate(axes.flat):
        axis.plot(
            time_s,
            measured_rad[:, joint_index] * scale,
            color="#1565c0",
            linewidth=1.8,
            label="measured",
        )
        if target_rad is not None:
            axis.plot(
                time_s,
                target_rad[:, joint_index] * scale,
                color="#ef6c00",
                linewidth=1.4,
                linestyle="--",
                label="target",
            )
        axis.set_title(f"P{joint_index:02d}  {joint_names[joint_index]}", fontsize=9)
        axis.set_ylabel(unit_text)
        axis.grid(True, alpha=0.3)
        if joint_index == 0:
            axis.legend(loc="best", fontsize=8)
    for axis in axes[-1, :]:
        axis.set_xlabel("time (s)")
    figure.suptitle(f"{title}\nAll policy-order joint angles", fontsize=15)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    return figure


def _plot_fingertip_forces(
    plt: Any,
    time_s: np.ndarray,
    force_n: np.ndarray,
    contact_names: list[str],
    title: str,
) -> Any:
    colors = ("#7b1fa2", "#1976d2", "#388e3c", "#f57c00", "#d32f2f")
    figure, axes = plt.subplots(5, 1, figsize=(12, 12), sharex=True)
    for finger_index, axis in enumerate(axes):
        axis.plot(
            time_s,
            force_n[:, finger_index],
            color=colors[finger_index],
            linewidth=2.0,
        )
        axis.set_title(f"F{finger_index}  {contact_names[finger_index]}", fontsize=10)
        axis.set_ylabel("force (N)")
        axis.grid(True, alpha=0.3)
    axes[-1].set_xlabel("time (s)")
    figure.suptitle(f"{title}\nFive fingertip forces", fontsize=15)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    return figure


def _plot_actions(
    plt: Any,
    time_s: np.ndarray,
    action: np.ndarray,
    joint_names: list[str],
    title: str,
) -> Any:
    figure, axes = plt.subplots(7, 3, figsize=(15, 20), sharex=True, sharey=True)
    for joint_index, axis in enumerate(axes.flat):
        axis.plot(
            time_s,
            action[:, joint_index],
            color="#6a1b9a",
            linewidth=1.4,
        )
        axis.axhline(0.0, color="#424242", linewidth=0.7, alpha=0.5)
        axis.set_ylim(-1.08, 1.08)
        axis.set_title(f"P{joint_index:02d}  {joint_names[joint_index]}", fontsize=9)
        axis.set_ylabel("action")
        axis.grid(True, alpha=0.3)
    for axis in axes[-1, :]:
        axis.set_xlabel("time (s)")
    figure.suptitle(f"{title}\nClipped policy actions", fontsize=15)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    return figure


def _optional_vector(payload: Any, name: str, frame_count: int) -> np.ndarray | None:
    if name not in payload.files:
        return None
    value = np.asarray(payload[name], dtype=np.float64)
    if value.shape != (frame_count,):
        raise ValueError(
            f"Trace array {name!r} must have shape ({frame_count},), got {value.shape}"
        )
    if not np.isfinite(value).all():
        raise ValueError(f"Trace array {name!r} contains non-finite values")
    return value


def _plot_object_diagnostics(
    plt: Any,
    time_s: np.ndarray,
    angular_velocity: np.ndarray,
    linear_velocity: np.ndarray,
    tilt_deg: np.ndarray | None,
    xy_drift_mm: np.ndarray | None,
    z_drift_mm: np.ndarray | None,
    reward: np.ndarray | None,
    title: str,
) -> Any:
    figure, axes = plt.subplots(5, 1, figsize=(13, 14), sharex=True)
    colors = ("#d32f2f", "#388e3c", "#1976d2")
    labels = ("x", "y", "z")
    for index in range(3):
        axes[0].plot(
            time_s,
            angular_velocity[:, index],
            color=colors[index],
            linewidth=1.5,
            label=labels[index],
        )
        axes[1].plot(
            time_s,
            linear_velocity[:, index],
            color=colors[index],
            linewidth=1.5,
            label=labels[index],
        )
    axes[0].set_ylabel("rad/s")
    axes[0].set_title("Cylinder angular velocity")
    axes[0].legend(ncol=3, fontsize=8)
    axes[1].set_ylabel("m/s")
    axes[1].set_title("Cylinder linear velocity")
    axes[1].legend(ncol=3, fontsize=8)

    if tilt_deg is not None:
        axes[2].plot(time_s, tilt_deg, color="#ef6c00", linewidth=1.7)
    axes[2].set_ylabel("deg")
    axes[2].set_title("Cylinder tilt")

    if xy_drift_mm is not None:
        axes[3].plot(time_s, xy_drift_mm, color="#00838f", linewidth=1.6, label="xy")
    if z_drift_mm is not None:
        axes[3].plot(time_s, z_drift_mm, color="#ad1457", linewidth=1.6, label="z")
    axes[3].set_ylabel("mm")
    axes[3].set_title("Cylinder drift")
    if xy_drift_mm is not None or z_drift_mm is not None:
        axes[3].legend(fontsize=8)

    if reward is not None:
        axes[4].plot(time_s, reward, color="#5d4037", linewidth=1.4)
    axes[4].set_ylabel("reward")
    axes[4].set_title("Per-step reward")
    axes[4].set_xlabel("time (s)")
    for axis in axes:
        axis.grid(True, alpha=0.3)
    figure.suptitle(f"{title}\nObject and reward diagnostics", fontsize=15)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    return figure


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    trace_path = Path(args.trace_npz).expanduser().resolve()
    if not trace_path.is_file():
        raise FileNotFoundError(trace_path)
    if args.dpi <= 0:
        raise ValueError("--dpi must be positive")

    try:
        import matplotlib

        if not args.show:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "Matplotlib is required; run this tool with a Python environment that "
            "provides matplotlib."
        ) from exc

    with np.load(trace_path, allow_pickle=False) as payload:
        metadata = _metadata(payload)
        measured_rad = _matrix(payload, "policy_pos_rad", JOINT_DIM)
        force_n = _matrix(payload, "force_n", FINGER_DIM)
        if force_n.shape[0] != measured_rad.shape[0]:
            raise ValueError("policy_pos_rad and force_n frame counts differ")
        target_rad = None
        if "policy_target_rad" in payload.files:
            target_rad = _matrix(payload, "policy_target_rad", JOINT_DIM)
            if target_rad.shape[0] != measured_rad.shape[0]:
                raise ValueError("policy_pos_rad and policy_target_rad frame counts differ")
        time_s = _time_axis(payload, measured_rad.shape[0])
        action = (
            _matrix(payload, "action", JOINT_DIM)
            if "action" in payload.files
            else None
        )
        angular_velocity = (
            _matrix(payload, "object_angvel_rad_s", 3)
            if "object_angvel_rad_s" in payload.files
            else None
        )
        linear_velocity = (
            _matrix(payload, "object_linvel_m_s", 3)
            if "object_linvel_m_s" in payload.files
            else None
        )
        tilt_deg = _optional_vector(payload, "extra__cylinder_tilt_deg", measured_rad.shape[0])
        xy_drift_mm = _optional_vector(payload, "extra__xy_drift_mm", measured_rad.shape[0])
        z_drift_mm = _optional_vector(payload, "extra__z_drift_mm", measured_rad.shape[0])
        reward = _optional_vector(payload, "reward", measured_rad.shape[0])

    joint_names = _labels(metadata, "joint_order", JOINT_DIM, "joint_")
    contact_names = _labels(metadata, "contact_order", FINGER_DIM, "finger_")
    if contact_names == [f"finger_{index:02d}" for index in range(FINGER_DIM)]:
        contact_names = list(DEFAULT_CONTACT_ORDER)

    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir is not None
        else trace_path.parent
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = trace_path.stem
    joint_path = output_dir / f"{stem}.joint_angles.png"
    force_path = output_dir / f"{stem}.fingertip_forces.png"
    action_path = output_dir / f"{stem}.actions.png"
    object_path = output_dir / f"{stem}.object_diagnostics.png"
    plot_title = f"{trace_path.name} ({measured_rad.shape[0]} frames)"

    joint_figure = _plot_joint_angles(
        plt,
        time_s,
        measured_rad,
        target_rad,
        joint_names,
        args.angle_unit,
        plot_title,
    )
    force_figure = _plot_fingertip_forces(
        plt,
        time_s,
        force_n,
        contact_names,
        plot_title,
    )
    action_figure = (
        _plot_actions(plt, time_s, action, joint_names, plot_title)
        if action is not None
        else None
    )
    object_figure = (
        _plot_object_diagnostics(
            plt,
            time_s,
            angular_velocity,
            linear_velocity,
            tilt_deg,
            xy_drift_mm,
            z_drift_mm,
            reward,
            plot_title,
        )
        if angular_velocity is not None and linear_velocity is not None
        else None
    )
    joint_figure.savefig(joint_path, dpi=args.dpi, bbox_inches="tight")
    force_figure.savefig(force_path, dpi=args.dpi, bbox_inches="tight")
    if action_figure is not None:
        action_figure.savefig(action_path, dpi=args.dpi, bbox_inches="tight")
    if object_figure is not None:
        object_figure.savefig(object_path, dpi=args.dpi, bbox_inches="tight")

    print(f"Joint-angle plot: {joint_path}")
    print(f"Fingertip-force plot: {force_path}")
    if action_figure is not None:
        print(f"Action plot: {action_path}")
    if object_figure is not None:
        print(f"Object-diagnostics plot: {object_path}")
    if args.show:
        plt.show()
    else:
        plt.close(joint_figure)
        plt.close(force_figure)
        if action_figure is not None:
            plt.close(action_figure)
        if object_figure is not None:
            plt.close(object_figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
