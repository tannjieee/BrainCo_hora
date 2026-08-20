from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml


@dataclass(frozen=True)
class Revo3Profile:
    path: Path
    hand: str
    policy_joint_order: tuple[str, ...]
    sdk_joint_order: tuple[str, ...]
    joint_lower_policy: np.ndarray
    joint_upper_policy: np.ndarray
    policy_to_sdk_perm: np.ndarray
    sdk_to_policy_perm: np.ndarray
    sdk_offset_rad: np.ndarray
    policy_offset_rad: np.ndarray
    sdk_position_lower_rad: np.ndarray
    sdk_position_upper_rad: np.ndarray
    target_lower_policy: np.ndarray
    target_upper_policy: np.ndarray
    joint_limit_scale: float
    action_scale: float
    default_rate_hz: float
    sdk: dict
    mit: dict
    tactile: dict
    safety: dict
    calibration_status: str

    @classmethod
    def load(
        cls,
        path: str | Path,
        expected_policy_order: tuple[str, ...] | list[str] | None = None,
        expected_limit_scale: float | None = None,
        expected_action_scale: float | None = None,
        expected_contact_order: tuple[str, ...] | list[str] | None = None,
    ) -> "Revo3Profile":
        profile_path = Path(path).expanduser().resolve()
        with profile_path.open("r", encoding="utf-8") as stream:
            cfg = yaml.safe_load(stream) or {}

        policy_order = tuple(str(name) for name in cfg.get("policy_joint_order") or [])
        sdk_order = tuple(str(name) for name in cfg.get("sdk_joint_order") or [])
        _validate_order(policy_order, "policy_joint_order")
        _validate_order(sdk_order, "sdk_joint_order")
        if set(policy_order) != set(sdk_order):
            raise ValueError("policy_joint_order and sdk_joint_order must contain the same joints.")
        if expected_policy_order is not None and policy_order != tuple(expected_policy_order):
            raise ValueError("Deployment metadata joint order differs from the robot profile.")

        limit_scale = float(cfg.get("policy_joint_limit_scale", 1.0))
        if not np.isfinite(limit_scale) or limit_scale <= 0.0:
            raise ValueError("policy_joint_limit_scale must be finite and positive.")
        if expected_limit_scale is not None and not np.isclose(limit_scale, expected_limit_scale):
            raise ValueError(
                f"Profile joint-limit scale {limit_scale} differs from metadata "
                f"{expected_limit_scale}."
            )
        action_scale = float(cfg.get("action_scale", 0.0))
        if not np.isfinite(action_scale) or action_scale <= 0.0:
            raise ValueError("action_scale must be finite and positive.")
        if expected_action_scale is not None and not np.isclose(
            action_scale, expected_action_scale
        ):
            raise ValueError(
                f"Profile action scale {action_scale} differs from metadata "
                f"{expected_action_scale}."
            )

        limits = cfg.get("joint_limits") or {}
        lower: list[float] = []
        upper: list[float] = []
        for joint in policy_order:
            joint_limits = limits.get(joint)
            if not isinstance(joint_limits, dict):
                raise ValueError(f"Missing joint_limits for {joint}.")
            lower.append(float(joint_limits["lower"]) * limit_scale)
            upper.append(float(joint_limits["upper"]) * limit_scale)
        lower_array = np.asarray(lower, dtype=np.float32)
        upper_array = np.asarray(upper, dtype=np.float32)
        if not np.isfinite(lower_array).all() or not np.isfinite(upper_array).all():
            raise ValueError("Scaled joint limits must be finite.")
        if np.any(upper_array <= lower_array):
            raise ValueError("Scaled joint limits are invalid.")

        policy_to_sdk = np.asarray([policy_order.index(name) for name in sdk_order], dtype=np.int64)
        sdk_to_policy = np.asarray([sdk_order.index(name) for name in policy_order], dtype=np.int64)
        sdk_offset = _load_offset(cfg, len(policy_order))
        policy_offset = sdk_offset[sdk_to_policy]

        sdk_position_limits = cfg.get("sdk_position_limits_deg") or {}
        sdk_lower: list[float] = []
        sdk_upper: list[float] = []
        for joint in sdk_order:
            bounds = sdk_position_limits.get(joint)
            if not isinstance(bounds, dict):
                raise ValueError(f"Missing sdk_position_limits_deg for {joint}.")
            sdk_lower.append(float(bounds["lower"]))
            sdk_upper.append(float(bounds["upper"]))
        sdk_lower_rad = np.deg2rad(np.asarray(sdk_lower, dtype=np.float32))
        sdk_upper_rad = np.deg2rad(np.asarray(sdk_upper, dtype=np.float32))
        if not np.isfinite(sdk_lower_rad).all() or not np.isfinite(sdk_upper_rad).all():
            raise ValueError("SDK position limits must be finite.")
        if np.any(sdk_upper_rad <= sdk_lower_rad):
            raise ValueError("SDK position limits are invalid.")

        # The network was trained with the scaled simulation limits, while commands
        # must also stay inside the independent SDK motor-coordinate envelope.
        # Keep the training bounds for q_unscaled, but integrate targets inside the
        # intersection expressed in policy coordinates.
        physical_lower_policy = (
            sdk_lower_rad[sdk_to_policy] - policy_offset
        ).astype(np.float32)
        physical_upper_policy = (
            sdk_upper_rad[sdk_to_policy] - policy_offset
        ).astype(np.float32)
        target_lower_policy = np.maximum(lower_array, physical_lower_policy)
        target_upper_policy = np.minimum(upper_array, physical_upper_policy)
        if np.any(target_upper_policy <= target_lower_policy):
            raise ValueError(
                "Training and SDK position limits have an empty target intersection."
            )

        tactile = dict(cfg.get("tactile") or {})
        tactile_order = tuple(str(name) for name in tactile.get("output_order") or [])
        if expected_contact_order is not None and tactile_order != tuple(expected_contact_order):
            raise ValueError("Tactile profile output_order differs from deployment metadata.")

        calibration = dict(cfg.get("calibration") or {})
        default_rate_hz = float(cfg.get("default_rate_hz", 20.0))
        if not np.isfinite(default_rate_hz) or default_rate_hz <= 0.0:
            raise ValueError("default_rate_hz must be finite and positive.")
        return cls(
            path=profile_path,
            hand=str(cfg.get("hand", "right")),
            policy_joint_order=policy_order,
            sdk_joint_order=sdk_order,
            joint_lower_policy=lower_array,
            joint_upper_policy=upper_array,
            policy_to_sdk_perm=policy_to_sdk,
            sdk_to_policy_perm=sdk_to_policy,
            sdk_offset_rad=sdk_offset,
            policy_offset_rad=policy_offset,
            sdk_position_lower_rad=sdk_lower_rad,
            sdk_position_upper_rad=sdk_upper_rad,
            target_lower_policy=target_lower_policy,
            target_upper_policy=target_upper_policy,
            joint_limit_scale=limit_scale,
            action_scale=action_scale,
            default_rate_hz=default_rate_hz,
            sdk=dict(cfg.get("sdk") or {}),
            mit=dict(cfg.get("mit") or {}),
            tactile=tactile,
            safety=dict(cfg.get("safety") or {}),
            calibration_status=str(calibration.get("status", "unverified")),
        )

    def measured_sdk_to_policy(self, sdk_pos_rad: np.ndarray) -> np.ndarray:
        sdk_pos_rad = self._vector(sdk_pos_rad, "sdk_pos_rad")
        return (sdk_pos_rad[self.sdk_to_policy_perm] - self.policy_offset_rad).astype(np.float32)

    def target_policy_to_sdk(self, policy_target_rad: np.ndarray) -> np.ndarray:
        policy_target_rad = self._vector(policy_target_rad, "policy_target_rad")
        return (policy_target_rad[self.policy_to_sdk_perm] + self.sdk_offset_rad).astype(np.float32)

    def validate_sdk_position(
        self,
        sdk_position_rad: np.ndarray,
        name: str,
        tolerance_rad: float = 0.0,
    ) -> np.ndarray:
        position = self._vector(sdk_position_rad, name)
        tolerance = float(tolerance_rad)
        if not np.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("SDK position tolerance must be finite and non-negative.")
        below = position < self.sdk_position_lower_rad - tolerance
        above = position > self.sdk_position_upper_rad + tolerance
        invalid = np.flatnonzero(below | above)
        if invalid.size:
            details = ", ".join(
                f"M{index}={np.rad2deg(position[index]):.2f}deg outside "
                f"[{np.rad2deg(self.sdk_position_lower_rad[index]):.2f},"
                f"{np.rad2deg(self.sdk_position_upper_rad[index]):.2f}]"
                for index in invalid
            )
            raise ValueError(
                f"{name} violates SDK hardware limits with "
                f"{np.rad2deg(tolerance):.2f}deg tolerance: {details}."
            )
        return position

    def _vector(self, value: np.ndarray, name: str) -> np.ndarray:
        vector = np.asarray(value, dtype=np.float32).reshape(-1)
        expected = len(self.policy_joint_order)
        if vector.shape != (expected,) or not np.isfinite(vector).all():
            raise ValueError(f"{name} must contain {expected} finite values.")
        return vector


def _validate_order(order: tuple[str, ...], name: str) -> None:
    if len(order) != 21:
        raise ValueError(f"{name} must contain 21 joints, got {len(order)}.")
    if len(set(order)) != len(order):
        raise ValueError(f"{name} contains duplicate joints.")


def _load_offset(cfg: dict, joint_dim: int) -> np.ndarray:
    offset_cfg = cfg.get("sim2real_joint_offset") or {}
    if not offset_cfg:
        return np.zeros(joint_dim, dtype=np.float32)
    if offset_cfg.get("order") != "sdk_joint_order":
        raise ValueError("sim2real_joint_offset.order must be sdk_joint_order.")
    if str(offset_cfg.get("units", "radians")) != "radians":
        raise ValueError("sim2real_joint_offset values must use radians.")
    values = np.asarray(offset_cfg.get("values") or [], dtype=np.float32).reshape(-1)
    if values.shape != (joint_dim,) or not np.isfinite(values).all():
        raise ValueError(f"sim2real_joint_offset.values must contain {joint_dim} finite values.")
    return values
