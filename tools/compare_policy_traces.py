#!/usr/bin/env python3
"""Compare unified simulator and real-robot policy traces.

The comparison is deliberately split into two questions:

1. Did sim and real build the same policy inputs and command pipeline?
2. Does a supplied ONNX reproduce each trace's recorded action from its own raw
   inputs?

The first comparison is step-aligned and reports differences; it does not assume
that independently evolving sim/real trajectories should remain numerically
identical.  The optional ONNX replay is an exact inference-parity check.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


CORE_KEYS = (
    "policy_pos_rad",
    "joint_pos_unscaled",
    "input_target_policy_rad",
    "force_n",
    "frame_raw",
    "obs_raw",
    "proprio_hist_raw",
    "obs_normalized",
    "proprio_hist_normalized",
    "onnx_action_raw",
    "action",
    "policy_target_rad",
)

METADATA_CONTRACT_KEYS = (
    "schema_name",
    "schema_version",
    "task",
    "policy_dt_s",
    "policy_rate_hz",
    "action_scale",
    "joint_order",
    "contact_order",
)


@dataclass(frozen=True)
class Trace:
    path: Path
    arrays: dict[str, np.ndarray]
    metadata: dict[str, Any]

    @property
    def length(self) -> int:
        return int(self.arrays["step_index"].shape[0])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare unified sim and real policy traces and optionally replay ONNX."
    )
    parser.add_argument("sim_trace", help="Simulator .npz trace.")
    parser.add_argument("real_trace", help="Real-robot .npz trace.")
    parser.add_argument(
        "--keys",
        default="",
        help="Comma-separated arrays to compare. Default: shared core policy arrays.",
    )
    parser.add_argument(
        "--all-common",
        action="store_true",
        help="Compare every shared numeric per-frame array with matching trailing shape.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help="Limit comparison/replay to the first N aligned steps (0 means all).",
    )
    parser.add_argument(
        "--onnx",
        default="",
        help="Optional Stage-2 ONNX used to replay obs_raw/proprio_hist_raw for both traces.",
    )
    parser.add_argument(
        "--replay-atol",
        type=float,
        default=1.0e-5,
        help="Absolute tolerance reported for ONNX replay versus recorded action.",
    )
    parser.add_argument(
        "--strict-replay",
        action="store_true",
        help="Exit with status 2 if ONNX replay exceeds --replay-atol.",
    )
    parser.add_argument(
        "--report-json",
        default="",
        metavar="PATH",
        help="Optionally save the machine-readable comparison report.",
    )
    return parser


def _load_trace(value: str) -> Trace:
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        with np.load(path, allow_pickle=False) as archive:
            arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
    except (OSError, ValueError) as exc:
        raise ValueError(f"Cannot load trace {path}: {exc}") from exc

    if "metadata_json" not in arrays:
        raise ValueError(f"Trace has no metadata_json: {path}")
    try:
        metadata_text = str(np.asarray(arrays.pop("metadata_json")).reshape(()).item())
        metadata = json.loads(metadata_text)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"Trace metadata_json is invalid: {path}: {exc}") from exc
    if not isinstance(metadata, dict):
        raise ValueError(f"Trace metadata_json must decode to a mapping: {path}")

    if "step_index" not in arrays:
        raise ValueError(f"Trace has no step_index: {path}")
    steps = np.asarray(arrays["step_index"])
    if steps.ndim != 1 or not np.issubdtype(steps.dtype, np.integer):
        raise ValueError(f"step_index must be a 1-D integer array: {path}")
    if steps.size == 0:
        raise ValueError(f"Trace contains no frames: {path}")
    if np.unique(steps).size != steps.size:
        raise ValueError(f"step_index contains duplicates: {path}")

    length = int(steps.shape[0])
    for name, array in arrays.items():
        if name.endswith("_rms_mean") or name.endswith("_rms_var"):
            continue
        if array.ndim == 0:
            continue
        if array.shape[0] != length:
            raise ValueError(
                f"Per-frame array {name!r} has leading size {array.shape[0]}, "
                f"expected {length}: {path}"
            )
    return Trace(path=path, arrays=arrays, metadata=metadata)


def _aligned_indices(
    sim: Trace, real: Trace, max_steps: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sim_steps = sim.arrays["step_index"].astype(np.int64, copy=False)
    real_steps = real.arrays["step_index"].astype(np.int64, copy=False)
    common = np.intersect1d(sim_steps, real_steps, assume_unique=True)
    if common.size == 0:
        raise ValueError("The traces have no common step_index values.")
    if max_steps > 0:
        common = common[:max_steps]
    sim_lookup = {int(step): index for index, step in enumerate(sim_steps.tolist())}
    real_lookup = {int(step): index for index, step in enumerate(real_steps.tolist())}
    sim_indices = np.asarray([sim_lookup[int(step)] for step in common], dtype=np.int64)
    real_indices = np.asarray([real_lookup[int(step)] for step in common], dtype=np.int64)
    return common, sim_indices, real_indices


def _metadata_contract_report(sim: Trace, real: Trace) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for key in METADATA_CONTRACT_KEYS:
        sim_value = sim.metadata.get(key)
        real_value = real.metadata.get(key)
        report[key] = {
            "match": sim_value == real_value,
            "sim": sim_value,
            "real": real_value,
        }
    sim_onnx_sha = _metadata_onnx_sha256(sim.metadata)
    real_onnx_sha = _metadata_onnx_sha256(real.metadata)
    report["onnx_sha256"] = {
        "match": bool(sim_onnx_sha and real_onnx_sha and sim_onnx_sha == real_onnx_sha),
        "sim": sim_onnx_sha,
        "real": real_onnx_sha,
    }
    return report


def _metadata_onnx_sha256(metadata: dict[str, Any]) -> str | None:
    top_level = metadata.get("onnx_sha256")
    if isinstance(top_level, str) and top_level:
        return top_level.lower()
    artifacts = metadata.get("artifacts")
    if isinstance(artifacts, dict):
        onnx = artifacts.get("onnx")
        if isinstance(onnx, dict):
            digest = onnx.get("sha256")
            if isinstance(digest, str) and digest:
                return digest.lower()
    return None


def _invariant_result(
    actual: np.ndarray,
    expected: np.ndarray,
    *,
    atol: float = 0.0,
) -> dict[str, Any]:
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    if actual.shape != expected.shape:
        return {
            "pass": False,
            "reason": "shape mismatch",
            "actual_shape": list(actual.shape),
            "expected_shape": list(expected.shape),
        }
    if actual.dtype == np.bool_ or expected.dtype == np.bool_:
        failed = np.not_equal(actual, expected)
        max_abs_error = float(np.any(failed))
    else:
        actual64 = actual.astype(np.float64, copy=False)
        expected64 = expected.astype(np.float64, copy=False)
        both_finite = np.isfinite(actual64) & np.isfinite(expected64)
        difference = np.abs(actual64 - expected64)
        failed = (~both_finite) | (both_finite & (difference > atol))
        finite_difference = difference[np.isfinite(difference)]
        max_abs_error = (
            float(np.max(finite_difference)) if finite_difference.size else math.nan
        )
    return {
        "pass": not bool(np.any(failed)),
        "failed_count": int(np.count_nonzero(failed)),
        "element_count": int(actual.size),
        "max_abs_error": max_abs_error,
        "atol": float(atol),
    }


def _trace_invariant_report(trace: Trace) -> dict[str, Any]:
    arrays = trace.arrays
    checks: dict[str, Any] = {}

    def require(names: tuple[str, ...], check_name: str) -> bool:
        missing = [name for name in names if name not in arrays]
        if missing:
            checks[check_name] = {"pass": False, "reason": f"missing arrays: {missing}"}
            return False
        return True

    if require(("obs_raw", "proprio_hist_raw"), "obs_equals_history_tail3"):
        history = arrays["proprio_hist_raw"]
        expected_obs = history[:, -3:, :].reshape(history.shape[0], -1)
        checks["obs_equals_history_tail3"] = _invariant_result(
            arrays["obs_raw"], expected_obs
        )

    if require(("frame_raw", "proprio_hist_raw"), "frame_equals_history_last"):
        checks["frame_equals_history_last"] = _invariant_result(
            arrays["frame_raw"], arrays["proprio_hist_raw"][:, -1, :]
        )

    frame_slices = (
        ("joint_pos_unscaled", slice(0, 21)),
        ("input_target_policy_rad", slice(21, 42)),
        ("force_n", slice(42, 47)),
    )
    for array_name, frame_slice in frame_slices:
        check_name = f"frame_slice_equals_{array_name}"
        if require(("frame_raw", array_name), check_name):
            checks[check_name] = _invariant_result(
                arrays["frame_raw"][:, frame_slice], arrays[array_name]
            )

    target_names = (
        "input_target_policy_rad",
        "action",
        "policy_target_unclipped_rad",
    )
    if require(target_names, "target_unclipped_formula"):
        action_scale = trace.metadata.get("action_scale")
        if not isinstance(action_scale, (int, float)) or not np.isfinite(action_scale):
            checks["target_unclipped_formula"] = {
                "pass": False,
                "reason": "metadata action_scale is missing/non-finite",
            }
        else:
            expected_unclipped = (
                arrays["input_target_policy_rad"]
                + float(action_scale) * arrays["action"]
            )
            checks["target_unclipped_formula"] = _invariant_result(
                arrays["policy_target_unclipped_rad"],
                expected_unclipped,
                atol=2.0e-6,
            )

    if require(("action",), "action_within_clip"):
        action = np.asarray(arrays["action"])
        excess = np.maximum(np.abs(action.astype(np.float64)) - 1.0, 0.0)
        checks["action_within_clip"] = {
            "pass": bool(np.isfinite(action).all() and np.max(excess, initial=0.0) <= 1.0e-7),
            "failed_count": int(np.count_nonzero((~np.isfinite(action)) | (np.abs(action) > 1.0 + 1.0e-7))),
            "element_count": int(action.size),
            "max_excess": float(np.max(excess, initial=0.0)),
        }

    if "onnx_action_raw" in arrays and "action" in arrays:
        onnx_action_atol = (
            1.0e-5 if str(trace.metadata.get("source", "")).lower() == "sim" else 1.0e-7
        )
        checks["action_equals_clipped_onnx_action_raw"] = _invariant_result(
            arrays["action"],
            np.clip(arrays["onnx_action_raw"], -1.0, 1.0),
            atol=onnx_action_atol,
        )

    clip_names = ("policy_target_unclipped_rad", "policy_target_rad", "target_clipped")
    if require(clip_names, "target_clipped_mask"):
        unclipped = arrays["policy_target_unclipped_rad"]
        target = arrays["policy_target_rad"]
        expected_mask = np.abs(
            target.astype(np.float64) - unclipped.astype(np.float64)
        ) > 1.0e-7
        checks["target_clipped_mask"] = _invariant_result(
            arrays["target_clipped"].astype(np.bool_), expected_mask
        )

        unchanged_actual = np.where(~expected_mask, target, unclipped)
        checks["target_unchanged_where_not_clipped"] = _invariant_result(
            unchanged_actual, unclipped, atol=1.0e-7
        )

    bounds_names = ("joint_lower_policy_rad", "joint_upper_policy_rad")
    if all(name in arrays for name in bounds_names) and all(
        name in arrays for name in clip_names
    ):
        expected_target = np.clip(
            arrays["policy_target_unclipped_rad"],
            arrays["joint_lower_policy_rad"],
            arrays["joint_upper_policy_rad"],
        )
        checks["target_equals_limit_clip"] = _invariant_result(
            arrays["policy_target_rad"], expected_target, atol=2.0e-6
        )

    return {
        "valid": all(bool(item.get("pass")) for item in checks.values()),
        "checks": checks,
    }


def _source_summary(values: np.ndarray) -> dict[str, float | int]:
    values64 = np.asarray(values, dtype=np.float64)
    finite = values64[np.isfinite(values64)]
    if finite.size == 0:
        return {"finite_count": 0, "mean": math.nan, "std": math.nan, "min": math.nan, "max": math.nan}
    return {
        "finite_count": int(finite.size),
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
    }


def _difference_summary(
    sim_values: np.ndarray,
    real_values: np.ndarray,
    aligned_steps: np.ndarray,
) -> dict[str, Any]:
    if sim_values.shape != real_values.shape:
        raise ValueError(f"Aligned shapes differ: {sim_values.shape} versus {real_values.shape}")
    delta = np.asarray(sim_values, dtype=np.float64) - np.asarray(real_values, dtype=np.float64)
    finite_mask = np.isfinite(delta)
    finite_delta = delta[finite_mask]
    if finite_delta.size == 0:
        return {
            "finite_count": 0,
            "finite_fraction": 0.0,
            "mean_signed": math.nan,
            "mean_abs": math.nan,
            "rmse": math.nan,
            "p95_abs": math.nan,
            "max_abs": math.nan,
            "max_step_index": None,
            "max_component_index": None,
        }

    abs_delta = np.abs(delta)
    masked_abs = np.where(finite_mask, abs_delta, -np.inf)
    flat_index = int(np.argmax(masked_abs))
    full_index = np.unravel_index(flat_index, delta.shape)
    frame_offset = int(full_index[0])
    component_index = [int(index) for index in full_index[1:]]
    return {
        "finite_count": int(finite_delta.size),
        "finite_fraction": float(finite_delta.size / delta.size),
        "mean_signed": float(np.mean(finite_delta)),
        "mean_abs": float(np.mean(np.abs(finite_delta))),
        "rmse": float(np.sqrt(np.mean(np.square(finite_delta)))),
        "p95_abs": float(np.percentile(np.abs(finite_delta), 95.0)),
        "max_abs": float(np.max(np.abs(finite_delta))),
        "max_step_index": int(aligned_steps[frame_offset]),
        "max_component_index": component_index,
    }


def _select_keys(sim: Trace, real: Trace, args: argparse.Namespace) -> list[str]:
    if args.keys:
        requested = [name.strip() for name in args.keys.split(",") if name.strip()]
        if not requested:
            raise ValueError("--keys did not contain an array name.")
        missing = [
            name
            for name in requested
            if name not in sim.arrays or name not in real.arrays
        ]
        if missing:
            raise ValueError(f"Requested arrays are not shared by both traces: {missing}")
        return requested

    selected = [name for name in CORE_KEYS if name in sim.arrays and name in real.arrays]
    if args.all_common:
        for name in sorted(set(sim.arrays) & set(real.arrays)):
            if name in selected or name == "step_index":
                continue
            sim_array = sim.arrays[name]
            real_array = real.arrays[name]
            if (
                np.issubdtype(sim_array.dtype, np.number)
                and np.issubdtype(real_array.dtype, np.number)
                and sim_array.ndim > 0
                and real_array.ndim > 0
                and sim_array.shape[1:] == real_array.shape[1:]
            ):
                selected.append(name)
    if not selected:
        raise ValueError("The traces share none of the core policy arrays.")
    return selected


def _compare_arrays(
    sim: Trace,
    real: Trace,
    keys: list[str],
    aligned_steps: np.ndarray,
    sim_indices: np.ndarray,
    real_indices: np.ndarray,
) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for name in keys:
        sim_array = sim.arrays[name]
        real_array = real.arrays[name]
        if not np.issubdtype(sim_array.dtype, np.number) or not np.issubdtype(
            real_array.dtype, np.number
        ):
            report[name] = {"skipped": "non-numeric array"}
            continue
        if sim_array.shape[1:] != real_array.shape[1:]:
            report[name] = {
                "skipped": "trailing shapes differ",
                "sim_shape": list(sim_array.shape),
                "real_shape": list(real_array.shape),
            }
            continue
        sim_values = sim_array[sim_indices]
        real_values = real_array[real_indices]
        report[name] = {
            "shape_per_step": list(sim_array.shape[1:]),
            "sim": _source_summary(sim_values),
            "real": _source_summary(real_values),
            "difference_sim_minus_real": _difference_summary(
                sim_values, real_values, aligned_steps
            ),
        }
    return report


def _replay_trace(
    trace: Trace,
    session: Any,
    max_steps: int,
    replay_atol: float,
) -> dict[str, Any]:
    for name in ("obs_raw", "proprio_hist_raw", "action"):
        if name not in trace.arrays:
            raise ValueError(f"ONNX replay requires {name!r} in {trace.path}")
    count = trace.length if max_steps <= 0 else min(trace.length, max_steps)
    obs = np.asarray(trace.arrays["obs_raw"][:count], dtype=np.float32)
    history = np.asarray(trace.arrays["proprio_hist_raw"][:count], dtype=np.float32)
    expected_name = "onnx_action_raw" if "onnx_action_raw" in trace.arrays else "action"
    expected = np.asarray(trace.arrays[expected_name][:count], dtype=np.float32)
    if not np.isfinite(obs).all() or not np.isfinite(history).all():
        raise ValueError(f"Raw ONNX inputs contain NaN/Inf: {trace.path}")

    replay_chunks: list[np.ndarray] = []
    chunk_size = 256
    for start in range(0, count, chunk_size):
        stop = min(count, start + chunk_size)
        output = session.run(
            ["action"],
            {"obs": obs[start:stop], "proprio_hist": history[start:stop]},
        )[0]
        replay_chunks.append(np.asarray(output, dtype=np.float32))
    replay = np.concatenate(replay_chunks, axis=0)
    steps = np.asarray(trace.arrays["step_index"][:count], dtype=np.int64)
    difference = _difference_summary(replay, expected, steps)
    max_abs = float(difference["max_abs"])
    return {
        "frames": int(count),
        "expected_array": expected_name,
        "within_atol": bool(np.isfinite(max_abs) and max_abs <= replay_atol),
        "atol": float(replay_atol),
        "replay_minus_recorded_action": difference,
    }


def _create_onnx_session(path_value: str) -> Any:
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError(
            "--onnx requires onnxruntime in the current environment."
        ) from exc
    path = Path(path_value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    input_names = [item.name for item in session.get_inputs()]
    output_names = [item.name for item in session.get_outputs()]
    if input_names != ["obs", "proprio_hist"] or output_names != ["action"]:
        raise ValueError(
            f"Unexpected ONNX contract: inputs={input_names}, outputs={output_names}."
        )
    return session


def _print_report(report: dict[str, Any]) -> None:
    print(
        f"Aligned {report['aligned_frames']} frames: "
        f"step {report['first_step_index']}..{report['last_step_index']}"
    )
    print("Metadata contract:")
    for key, item in report["metadata_contract"].items():
        marker = "OK" if item["match"] else "MISMATCH"
        print(f"  {marker:8s} {key}")
        if not item["match"]:
            print(f"             sim={item['sim']!r}")
            print(f"             real={item['real']!r}")

    print("\nPer-trace internal invariants:")
    for source, invariant_report in report["trace_invariants"].items():
        marker = "PASS" if invariant_report["valid"] else "FAIL"
        print(f"  {source:4s} {marker}")
        for name, item in invariant_report["checks"].items():
            check_marker = "OK" if item.get("pass") else "FAIL"
            detail = (
                f" max_abs={item['max_abs_error']:.6g}"
                if isinstance(item.get("max_abs_error"), (int, float))
                else ""
            )
            reason = f" ({item['reason']})" if "reason" in item else ""
            print(f"       {check_marker:4s} {name}{detail}{reason}")

    print("\nStep-aligned arrays (sim - real):")
    print("  array                              mean_abs         rmse      p95_abs      max_abs  max_step")
    for name, item in report["arrays"].items():
        if "skipped" in item:
            print(f"  {name:34s} SKIP: {item['skipped']}")
            continue
        diff = item["difference_sim_minus_real"]
        print(
            f"  {name:34s} "
            f"{diff['mean_abs']:12.5g} {diff['rmse']:12.5g} "
            f"{diff['p95_abs']:12.5g} {diff['max_abs']:12.5g} "
            f"{str(diff['max_step_index']):>9s}"
        )

    replay = report.get("onnx_replay")
    if replay:
        print("\nONNX replay against each trace's recorded action:")
        for source, item in replay.items():
            diff = item["replay_minus_recorded_action"]
            marker = "PASS" if item["within_atol"] else "FAIL"
            print(
                f"  {source:4s} {marker}: max_abs={diff['max_abs']:.9g}, "
                f"mean_abs={diff['mean_abs']:.9g}, atol={item['atol']:.3g}, "
                f"expected={item['expected_array']}"
            )

    print(
        "\nInterpretation: exact ONNX replay isolates inference/export errors. "
        "Step-aligned sim-real deltas also include sensor calibration, initial-state, "
        "contact, timing, actuator, and accumulated trajectory differences."
    )


def main() -> int:
    args = build_parser().parse_args()
    if args.max_steps < 0:
        raise ValueError("--max-steps must be non-negative.")
    if not np.isfinite(args.replay_atol) or args.replay_atol < 0.0:
        raise ValueError("--replay-atol must be finite and non-negative.")
    if args.strict_replay and not args.onnx:
        raise ValueError("--strict-replay requires --onnx.")

    sim = _load_trace(args.sim_trace)
    real = _load_trace(args.real_trace)
    aligned_steps, sim_indices, real_indices = _aligned_indices(
        sim, real, args.max_steps
    )
    keys = _select_keys(sim, real, args)

    report: dict[str, Any] = {
        "sim_trace": str(sim.path),
        "real_trace": str(real.path),
        "sim_frames": sim.length,
        "real_frames": real.length,
        "aligned_frames": int(aligned_steps.size),
        "first_step_index": int(aligned_steps[0]),
        "last_step_index": int(aligned_steps[-1]),
        "metadata_contract": _metadata_contract_report(sim, real),
        "trace_invariants": {
            "sim": _trace_invariant_report(sim),
            "real": _trace_invariant_report(real),
        },
        "arrays": _compare_arrays(
            sim,
            real,
            keys,
            aligned_steps,
            sim_indices,
            real_indices,
        ),
    }

    strict_failed = False
    if args.onnx:
        session = _create_onnx_session(args.onnx)
        replay = {
            "sim": _replay_trace(sim, session, args.max_steps, args.replay_atol),
            "real": _replay_trace(real, session, args.max_steps, args.replay_atol),
        }
        report["onnx_replay"] = replay
        strict_failed = not all(item["within_atol"] for item in replay.values())

    _print_report(report)
    if args.report_json:
        report_path = Path(args.report_json).expanduser().resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"Report JSON: {report_path}")
    invariants_failed = not all(
        item["valid"] for item in report["trace_invariants"].values()
    )
    metadata_failed = not all(
        item["match"] for item in report["metadata_contract"].values()
    )
    return 2 if metadata_failed or invariants_failed or (args.strict_replay and strict_failed) else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
