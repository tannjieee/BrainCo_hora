#!/usr/bin/env python3
"""Validate a completed Revo3 grasp cache without starting Isaac Sim."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True, help="Final .npy cache to validate.")
    parser.add_argument(
        "--expected-count",
        type=int,
        required=True,
        help="Exact number of cache rows required.",
    )
    parser.add_argument(
        "--quaternion-atol",
        type=float,
        default=1.0e-3,
        help="Maximum absolute error allowed for quaternion norms (default: 1e-3).",
    )
    args = parser.parse_args()
    if args.expected_count <= 0:
        parser.error("--expected-count must be greater than 0")
    if args.quaternion_atol <= 0.0:
        parser.error("--quaternion-atol must be greater than 0")
    return args


def validate_cache(cache_path: Path, expected_count: int, quaternion_atol: float) -> None:
    if not cache_path.is_file():
        raise ValueError(f"final cache does not exist: {cache_path}")

    states = np.load(cache_path, allow_pickle=False)
    expected_shape = (expected_count, 28)
    if states.shape != expected_shape:
        raise ValueError(f"expected shape {expected_shape}, found {states.shape}")
    if not np.issubdtype(states.dtype, np.number) or np.iscomplexobj(states):
        raise ValueError(f"cache must contain real numeric values, found dtype {states.dtype}")
    if not np.isfinite(states).all():
        bad_count = int((~np.isfinite(states)).sum())
        raise ValueError(f"cache contains {bad_count} non-finite values")

    quaternion_norms = np.linalg.norm(states[:, -4:].astype(np.float64), axis=1)
    norm_errors = np.abs(quaternion_norms - 1.0)
    if not np.all(norm_errors <= quaternion_atol):
        worst_index = int(np.argmax(norm_errors))
        raise ValueError(
            "object quaternion is not unit length at row "
            f"{worst_index}: norm={quaternion_norms[worst_index]:.8f}, "
            f"error={norm_errors[worst_index]:.3e}, atol={quaternion_atol:.3e}"
        )

    print(
        f"[VALID] {cache_path}: shape={states.shape}, dtype={states.dtype}, "
        f"finite=yes, quat_norm=[{quaternion_norms.min():.8f}, {quaternion_norms.max():.8f}]",
        flush=True,
    )


def main() -> int:
    args = parse_args()
    try:
        validate_cache(args.cache, args.expected_count, args.quaternion_atol)
    except Exception as error:
        print(f"[INVALID] {args.cache}: {error}", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
