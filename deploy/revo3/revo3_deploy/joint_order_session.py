from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any
from uuid import uuid4
from datetime import datetime, timezone

import numpy as np

from .replay_trace import ReplayTrace
from .robot_profile import Revo3Profile


SESSION_SCHEMA_NAME = "revo3_joint_order_session"
SESSION_SCHEMA_VERSION = 1


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: str | Path) -> str:
    resolved = Path(path).expanduser().resolve()
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_joint_probe_plan(
    trace: ReplayTrace,
    profile: Revo3Profile,
    *,
    min_delta_deg: float = 1.0,
    max_delta_deg: float = 2.5,
) -> list[dict[str, Any]]:
    minimum = float(min_delta_deg)
    maximum = float(max_delta_deg)
    if (
        not np.isfinite([minimum, maximum]).all()
        or minimum <= 0.0
        or maximum <= minimum
        or maximum > 3.0
    ):
        raise ValueError(
            "Probe delta window must be finite with 0 < min < max <= 3 degrees."
        )

    usable = trace.usable_frame_count
    delta_deg = np.rad2deg(
        trace.policy_target_rad[:usable] - trace.target_before_policy_rad[:usable]
    )
    joints: list[dict[str, Any]] = []
    for policy_index, joint_name in enumerate(profile.policy_joint_order):
        sdk_index = profile.sdk_joint_order.index(joint_name)
        values = delta_deg[:, policy_index]
        candidates: list[dict[str, Any]] = []
        for direction in (1, -1):
            valid = np.flatnonzero(
                (np.sign(values) == direction)
                & (np.abs(values) >= minimum)
                & (np.abs(values) <= maximum)
            )
            if valid.size == 0:
                continue
            local = int(np.argmax(np.abs(values[valid])))
            row = int(valid[local])
            applied = float(values[row])
            raw_delta = float(
                np.rad2deg(profile.action_scale * trace.action[row, policy_index])
            )
            candidates.append(
                {
                    "row": row,
                    "step_index": int(trace.step_index[row]),
                    "delta_deg": applied,
                    "direction": "positive" if applied > 0.0 else "negative",
                    "sim_target_clipped": bool(abs(applied - raw_delta) > 1.0e-4),
                }
            )
        candidates.sort(key=lambda item: (-abs(item["delta_deg"]), item["row"]))
        joints.append(
            {
                "policy_index": policy_index,
                "sdk_index": sdk_index,
                "joint_name": joint_name,
                "state": "planned" if candidates else "unavailable",
                "candidates": candidates,
                "attempts": [],
            }
        )
    return joints


def create_session(
    *,
    path: str | Path,
    trace_path: str | Path,
    checkpoint_path: str | Path,
    profile_path: str | Path,
    trace: ReplayTrace,
    profile: Revo3Profile,
    min_delta_deg: float,
    max_delta_deg: float,
) -> dict[str, Any]:
    session_path = Path(path).expanduser().resolve()
    if session_path.exists():
        raise FileExistsError(
            f"Joint-order session already exists and will not be overwritten: {session_path}"
        )
    trace_resolved = Path(trace_path).expanduser().resolve()
    checkpoint_resolved = Path(checkpoint_path).expanduser().resolve()
    profile_resolved = Path(profile_path).expanduser().resolve()
    serial_allowlist = [
        str(value).strip() for value in profile.sdk.get("serial_allowlist") or ()
    ]
    if len(serial_allowlist) != 1 or not serial_allowlist[0]:
        raise RuntimeError(
            "Joint-order sessions require exactly one non-empty SDK serial allowlist "
            "entry so preflight and execution are bound to the same physical hand."
        )
    session = {
        "schema_name": SESSION_SCHEMA_NAME,
        "schema_version": SESSION_SCHEMA_VERSION,
        "session_id": str(uuid4()),
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "state": "active",
        "artifacts": {
            "trace": {
                "path": str(trace_resolved),
                "sha256": sha256_file(trace_resolved),
            },
            "checkpoint": {
                "path": str(checkpoint_resolved),
                "sha256": sha256_file(checkpoint_resolved),
            },
            "profile": {
                "path": str(profile_resolved),
                "sha256": sha256_file(profile_resolved),
            },
        },
        "trace_checkpoint_sha256": trace.metadata["checkpoint_sha256"],
        "device_expected": {
            "hand": profile.hand,
            "serial_allowlist": serial_allowlist,
        },
        "probe_config": {
            "frames": 1,
            "anchor_current": True,
            "kp": 0.2,
            "kd": 0.05,
            "max_speed_deg_s": 2.0,
            "min_delta_deg": float(min_delta_deg),
            "max_delta_deg": float(max_delta_deg),
        },
        "joints": build_joint_probe_plan(
            trace,
            profile,
            min_delta_deg=min_delta_deg,
            max_delta_deg=max_delta_deg,
        ),
    }
    save_session(session_path, session, require_existing=False)
    return session


def load_session(path: str | Path) -> dict[str, Any]:
    session_path = Path(path).expanduser().resolve()
    with session_path.open("r", encoding="utf-8") as handle:
        session = json.load(handle)
    if session.get("schema_name") != SESSION_SCHEMA_NAME:
        raise ValueError("Unsupported joint-order session schema name.")
    if int(session.get("schema_version", -1)) != SESSION_SCHEMA_VERSION:
        raise ValueError("Unsupported joint-order session schema version.")
    if not isinstance(session.get("joints"), list) or len(session["joints"]) != 21:
        raise ValueError("Joint-order session must contain exactly 21 joints.")
    return session


def save_session(
    path: str | Path,
    session: dict[str, Any],
    *,
    require_existing: bool = True,
) -> None:
    session_path = Path(path).expanduser().resolve()
    if require_existing and not session_path.is_file():
        raise FileNotFoundError(session_path)
    session_path.parent.mkdir(parents=True, exist_ok=True)
    session["updated_at"] = utc_now()
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{session_path.name}.",
        suffix=".tmp",
        dir=session_path.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(session, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        if require_existing:
            os.replace(temporary_path, session_path)
        else:
            try:
                os.link(temporary_path, session_path)
            except FileExistsError as exc:
                raise FileExistsError(
                    "Joint-order session appeared concurrently and will not be "
                    f"overwritten: {session_path}"
                ) from exc
            temporary_path.unlink()
        directory_fd = os.open(session_path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def verify_session_artifacts(session: dict[str, Any]) -> None:
    for label in ("trace", "checkpoint", "profile"):
        artifact = session["artifacts"][label]
        actual = sha256_file(artifact["path"])
        expected = str(artifact["sha256"])
        if actual != expected:
            raise ValueError(
                f"Session {label} SHA256 mismatch: expected {expected}, got {actual}."
            )


def validate_session_plan(
    session: dict[str, Any],
    trace: ReplayTrace,
    profile: Revo3Profile,
) -> None:
    config = session.get("probe_config") or {}
    fixed_contract = {
        "frames": 1,
        "anchor_current": True,
        "kp": 0.2,
        "kd": 0.05,
        "max_speed_deg_s": 2.0,
    }
    for key, expected in fixed_contract.items():
        if config.get(key) != expected:
            raise ValueError(f"Session probe_config {key} was modified.")
    expected_plan = build_joint_probe_plan(
        trace,
        profile,
        min_delta_deg=float(config["min_delta_deg"]),
        max_delta_deg=float(config["max_delta_deg"]),
    )
    allowed_states = {
        "planned",
        "unavailable",
        "armed",
        "observation_pending",
        "passed",
        "blocked",
    }
    for expected, actual in zip(expected_plan, session["joints"], strict=True):
        for key in ("policy_index", "sdk_index", "joint_name"):
            if actual.get(key) != expected[key]:
                raise ValueError(f"Session joint plan {key} was modified.")
        if actual.get("state") not in allowed_states:
            raise ValueError("Session joint state is invalid.")
        if not isinstance(actual.get("attempts"), list):
            raise ValueError("Session joint attempts must be a list.")
        actual_candidates = actual.get("candidates")
        if not isinstance(actual_candidates, list) or len(actual_candidates) != len(
            expected["candidates"]
        ):
            raise ValueError("Session joint candidate count was modified.")
        for expected_candidate, actual_candidate in zip(
            expected["candidates"],
            actual_candidates,
            strict=True,
        ):
            for key in ("row", "step_index", "direction", "sim_target_clipped"):
                if actual_candidate.get(key) != expected_candidate[key]:
                    raise ValueError(f"Session candidate {key} was modified.")
            if not np.isclose(
                float(actual_candidate.get("delta_deg")),
                float(expected_candidate["delta_deg"]),
                rtol=0.0,
                atol=1.0e-7,
            ):
                raise ValueError("Session candidate delta_deg was modified.")


def resolve_session_joint(
    session: dict[str, Any],
    selector: str,
) -> dict[str, Any]:
    text = str(selector).strip()
    upper = text.upper()
    matches: list[dict[str, Any]] = []
    for joint in session["joints"]:
        if upper == f"P{int(joint['policy_index']):02d}":
            matches.append(joint)
        elif upper == f"P{int(joint['policy_index'])}":
            matches.append(joint)
        elif upper == f"M{int(joint['sdk_index']):02d}":
            matches.append(joint)
        elif upper == f"M{int(joint['sdk_index'])}":
            matches.append(joint)
        elif text == joint["joint_name"]:
            matches.append(joint)
    unique = {int(item["policy_index"]): item for item in matches}
    if len(unique) != 1:
        raise ValueError(
            f"Unknown or ambiguous joint selector {selector!r}; use P0..P20, "
            "M0..M20, or an exact joint name."
        )
    return next(iter(unique.values()))


def select_candidate(joint: dict[str, Any], row: int | None) -> dict[str, Any]:
    candidates = list(joint.get("candidates") or ())
    if not candidates:
        raise RuntimeError(
            f"P{joint['policy_index']:02d}/M{joint['sdk_index']:02d} has no safe "
            "visible single-frame candidate."
        )
    if row is None:
        return candidates[0]
    matches = [item for item in candidates if int(item["row"]) == int(row)]
    if len(matches) != 1:
        available = [int(item["row"]) for item in candidates]
        raise ValueError(
            f"Row {row} is not a planned candidate for P{joint['policy_index']:02d}; "
            f"available rows are {available}."
        )
    return matches[0]
