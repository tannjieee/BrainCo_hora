from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .robot_profile import Revo3Profile


TRACE_SCHEMA_NAME = "hora_policy_trace"
TRACE_SCHEMA_VERSION = 1
JOINT_DIM = 21
TRACE_ATOL_RAD = 2.0e-5
LIMIT_ABI_ATOL_RAD = 1.0e-6


@dataclass(frozen=True)
class ReplayTrace:
    """Validated simulator targets suitable for bounded hardware replay."""

    path: Path
    metadata: dict[str, Any]
    step_index: np.ndarray
    sample_time_s: np.ndarray
    action: np.ndarray
    policy_pos_rad: np.ndarray
    target_before_policy_rad: np.ndarray
    policy_target_rad: np.ndarray
    done: np.ndarray

    @classmethod
    def load(
        cls,
        path: str | Path,
        profile: Revo3Profile,
        checkpoint_path: str | Path | None = None,
    ) -> "ReplayTrace":
        trace_path = Path(path).expanduser().resolve()
        if not trace_path.is_file():
            raise FileNotFoundError(trace_path)

        with np.load(trace_path, allow_pickle=False) as payload:
            metadata = _load_metadata(payload)
            _validate_metadata(metadata, profile)

            target = _matrix(payload, "policy_target_rad", JOINT_DIM)
            frame_count = int(target.shape[0])
            if frame_count == 0:
                raise ValueError("Replay trace contains no frames.")
            policy_pos = _matrix(payload, "policy_pos_rad", JOINT_DIM, frame_count)
            action = _matrix(payload, "action", JOINT_DIM, frame_count)
            target_before = _matrix(
                payload,
                "target_before_policy_rad",
                JOINT_DIM,
                frame_count,
            )
            lower = _matrix(
                payload,
                "joint_lower_policy_rad",
                JOINT_DIM,
                frame_count,
            )
            upper = _matrix(
                payload,
                "joint_upper_policy_rad",
                JOINT_DIM,
                frame_count,
            )
            if np.any(upper <= lower):
                raise ValueError("Replay trace contains invalid policy joint limits.")
            expected_lower = profile.joint_lower_policy.reshape(1, -1)
            expected_upper = profile.joint_upper_policy.reshape(1, -1)
            if not np.allclose(
                lower,
                expected_lower,
                rtol=0.0,
                atol=LIMIT_ABI_ATOL_RAD,
            ) or not np.allclose(
                upper,
                expected_upper,
                rtol=0.0,
                atol=LIMIT_ABI_ATOL_RAD,
            ):
                raise ValueError(
                    "Replay trace joint limits differ from the robot profile training limits."
                )
            if np.any(np.abs(action) > 1.0 + 1.0e-6):
                raise ValueError("Replay action is outside the required [-1,1] range.")
            action_scale = float(metadata["action_scale"])
            expected_target = np.clip(
                target_before + action_scale * np.clip(action, -1.0, 1.0),
                lower,
                upper,
            )
            target_error = float(np.max(np.abs(expected_target - target)))
            if target_error > TRACE_ATOL_RAD:
                raise ValueError(
                    "policy_target_rad does not match target_before + action_scale * "
                    f"action followed by clipping (max error {target_error:.9g} rad)."
                )
            if frame_count > 1:
                continuity_error = float(
                    np.max(np.abs(target_before[1:] - target[:-1]))
                )
                if continuity_error > TRACE_ATOL_RAD:
                    raise ValueError(
                        "Replay target_before sequence is discontinuous "
                        f"(max error {continuity_error:.9g} rad)."
                    )

            step_index = _vector(
                payload,
                "step_index",
                frame_count,
                dtype=np.int64,
            )
            if frame_count > 1 and not np.array_equal(
                np.diff(step_index),
                np.ones(frame_count - 1, dtype=np.int64),
            ):
                raise ValueError("Replay trace step_index must be contiguous.")
            if int(step_index[0]) != 0:
                raise ValueError("Replay trace step_index must start at zero.")
            sample_time = _vector(
                payload,
                "sample_time_s",
                frame_count,
                dtype=np.float64,
            )
            if frame_count > 1:
                expected_dt = 1.0 / float(metadata["policy_rate_hz"])
                actual_dt = np.diff(sample_time)
                if not np.allclose(actual_dt, expected_dt, rtol=0.0, atol=1.0e-7):
                    raise ValueError(
                        "Replay trace sample_time_s does not match metadata policy_rate_hz."
                    )
            if not np.isclose(float(sample_time[0]), 0.0, rtol=0.0, atol=1.0e-12):
                raise ValueError("Replay trace sample_time_s must start at zero.")
            done = _binary_vector(payload, "done", frame_count)
            next_state_is_reset = _binary_vector(
                payload,
                "next_state_is_reset",
                frame_count,
            )
            if not np.array_equal(done, next_state_is_reset):
                raise ValueError("Replay done and next_state_is_reset arrays disagree.")
            terminal_rows = np.flatnonzero(done)
            if terminal_rows.size > 1 or (
                terminal_rows.size == 1 and int(terminal_rows[0]) != frame_count - 1
            ):
                raise ValueError(
                    "Replay trace crosses an episode reset; export one uninterrupted episode."
                )

        if checkpoint_path is not None:
            checkpoint = Path(checkpoint_path).expanduser().resolve()
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
            expected_sha = str(metadata.get("checkpoint_sha256", "")).lower()
            actual_sha = _sha256(checkpoint)
            if not expected_sha or actual_sha != expected_sha:
                raise ValueError(
                    "Replay trace checkpoint SHA256 does not match the requested checkpoint."
                )

        return cls(
            path=trace_path,
            metadata=metadata,
            step_index=step_index,
            sample_time_s=sample_time,
            action=action.astype(np.float32, copy=False),
            policy_pos_rad=policy_pos.astype(np.float32, copy=False),
            target_before_policy_rad=target_before.astype(np.float32, copy=False),
            policy_target_rad=target.astype(np.float32, copy=False),
            done=done,
        )

    @property
    def frame_count(self) -> int:
        return int(self.policy_target_rad.shape[0])

    @property
    def usable_frame_count(self) -> int:
        """Terminal actions are excluded from hardware replay."""
        terminal = np.flatnonzero(self.done)
        return int(terminal[0]) if terminal.size else self.frame_count

    @property
    def policy_rate_hz(self) -> float:
        return float(self.metadata["policy_rate_hz"])

    def trajectory_policy_rad(self, source: str) -> np.ndarray:
        if source == "target":
            return self.policy_target_rad
        if source == "measured":
            return self.policy_pos_rad
        raise ValueError(f"Unknown replay trajectory source {source!r}.")

    def select(self, start_frame: int, frames: int | None) -> np.ndarray:
        start = int(start_frame)
        if start < 0:
            raise ValueError("--start-frame must be non-negative.")
        usable = self.usable_frame_count
        if start >= usable:
            raise ValueError(
                f"--start-frame {start} is outside {usable} non-terminal replay frames."
            )
        if frames is None:
            stop = usable
        else:
            count = int(frames)
            if count <= 0:
                raise ValueError("--frames must be positive.")
            stop = start + count
            if stop > usable:
                raise ValueError(
                    f"Requested rows [{start},{stop}) exceed {usable} non-terminal frames."
                )
        return np.arange(start, stop, dtype=np.int64)


def _load_metadata(payload: Any) -> dict[str, Any]:
    if "metadata_json" not in payload.files:
        raise ValueError("Replay trace is missing metadata_json.")
    raw = np.asarray(payload["metadata_json"])
    if raw.shape != ():
        raise ValueError("Replay trace metadata_json must be a scalar string.")
    try:
        metadata = json.loads(str(raw.item()))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("Replay trace metadata_json is invalid JSON.") from exc
    if not isinstance(metadata, dict):
        raise ValueError("Replay trace metadata_json must decode to a mapping.")
    return metadata


def _validate_metadata(metadata: dict[str, Any], profile: Revo3Profile) -> None:
    if metadata.get("schema_name") != TRACE_SCHEMA_NAME:
        raise ValueError(f"Expected trace schema {TRACE_SCHEMA_NAME!r}.")
    if int(metadata.get("schema_version", -1)) != TRACE_SCHEMA_VERSION:
        raise ValueError(f"Expected trace schema version {TRACE_SCHEMA_VERSION}.")
    if metadata.get("source") != "sim":
        raise ValueError("Only simulator traces may be used for hardware replay.")
    if metadata.get("command") != "tools/dump_runtime_actions.py":
        raise ValueError("Replay trace was not produced by tools/dump_runtime_actions.py.")
    if metadata.get("action_semantics") != "delta":
        raise ValueError("Replay trace action_semantics must be 'delta'.")
    if metadata.get("target_units") != "radians":
        raise ValueError("Replay trace target_units must be 'radians'.")
    action_clip = tuple(float(value) for value in metadata.get("action_clip") or ())
    if action_clip != (-1.0, 1.0):
        raise ValueError("Replay trace action_clip must be [-1,1].")
    units = metadata.get("units")
    if not isinstance(units, dict) or units.get("joint_position") != "rad":
        raise ValueError("Replay trace joint-position units must be radians.")
    joint_order = tuple(str(value) for value in metadata.get("joint_order") or ())
    if joint_order != profile.policy_joint_order:
        raise ValueError("Replay trace joint_order differs from the robot profile.")

    rate = float(metadata.get("policy_rate_hz", 0.0))
    scale = float(metadata.get("action_scale", 0.0))
    if not np.isfinite([rate, scale]).all() or rate <= 0.0 or scale <= 0.0:
        raise ValueError("Replay trace policy rate and action scale must be finite and positive.")
    if not np.isclose(rate, profile.default_rate_hz, rtol=0.0, atol=1.0e-9):
        raise ValueError(
            f"Replay rate {rate:g} Hz differs from profile rate {profile.default_rate_hz:g} Hz."
        )
    if not np.isclose(scale, profile.action_scale, rtol=0.0, atol=1.0e-12):
        raise ValueError("Replay trace action_scale differs from the robot profile.")
    checkpoint_sha = str(metadata.get("checkpoint_sha256", ""))
    if len(checkpoint_sha) != 64 or any(
        char not in "0123456789abcdefABCDEF" for char in checkpoint_sha
    ):
        raise ValueError("Replay trace is missing a valid checkpoint_sha256.")


def _matrix(
    payload: Any,
    name: str,
    columns: int,
    rows: int | None = None,
) -> np.ndarray:
    if name not in payload.files:
        raise ValueError(f"Replay trace is missing required array {name!r}.")
    value = np.asarray(payload[name], dtype=np.float64)
    expected_rows = value.shape[0] if rows is None and value.ndim == 2 else rows
    if value.ndim != 2 or value.shape != (expected_rows, columns):
        row_text = "T" if rows is None else str(rows)
        raise ValueError(
            f"Replay array {name!r} must have shape ({row_text},{columns}), "
            f"got {value.shape}."
        )
    if not np.isfinite(value).all():
        raise ValueError(f"Replay array {name!r} contains non-finite values.")
    return value


def _vector(
    payload: Any,
    name: str,
    rows: int,
    dtype: np.dtype | type,
) -> np.ndarray:
    if name not in payload.files:
        raise ValueError(f"Replay trace is missing required array {name!r}.")
    raw = np.asarray(payload[name])
    if raw.shape != (rows,):
        raise ValueError(
            f"Replay array {name!r} must have shape ({rows},), got {raw.shape}."
        )
    if raw.dtype.kind in "fc" and not np.isfinite(raw).all():
        raise ValueError(f"Replay array {name!r} contains non-finite values.")
    return raw.astype(dtype, copy=False)


def _binary_vector(payload: Any, name: str, rows: int) -> np.ndarray:
    raw = _vector(payload, name, rows, dtype=np.int64)
    if not np.isin(raw, (0, 1)).all():
        raise ValueError(f"Replay array {name!r} must contain only 0/1 values.")
    return raw.astype(np.bool_, copy=False)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
