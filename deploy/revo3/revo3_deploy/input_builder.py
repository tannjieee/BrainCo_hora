from __future__ import annotations

from collections import deque

import numpy as np

from .contract import PolicyContract


class Stage2InputBuilder:
    """Build raw tactile Stage-2 observations in chronological order."""

    def __init__(
        self,
        contract: PolicyContract,
        joint_lower: np.ndarray,
        joint_upper: np.ndarray,
        target_lower: np.ndarray | None = None,
        target_upper: np.ndarray | None = None,
    ) -> None:
        self.contract = contract
        self.joint_dim = contract.action_dim
        self.contact_dim = len(contract.contact_order)
        self.joint_lower = self._joint_vector(joint_lower, "joint_lower")
        self.joint_upper = self._joint_vector(joint_upper, "joint_upper")
        if np.any(self.joint_upper <= self.joint_lower):
            raise ValueError("Every joint upper limit must be greater than its lower limit.")
        self.target_lower = self._joint_vector(
            self.joint_lower if target_lower is None else target_lower,
            "target_lower",
        )
        self.target_upper = self._joint_vector(
            self.joint_upper if target_upper is None else target_upper,
            "target_upper",
        )
        if np.any(self.target_upper <= self.target_lower):
            raise ValueError("Every target upper limit must be greater than its lower limit.")
        if np.any(self.target_lower < self.joint_lower) or np.any(
            self.target_upper > self.joint_upper
        ):
            raise ValueError("Target limits must be contained inside training joint limits.")

        self.current_target: np.ndarray | None = None
        self._frames: deque[np.ndarray] = deque(maxlen=contract.history_len)

    def set_target_limits(self, lower: np.ndarray, upper: np.ndarray) -> None:
        """Tighten target limits before the history/current target is initialized."""
        if self.current_target is not None or self._frames:
            raise RuntimeError("Target limits must be set before the input history is initialized.")
        lower = self._joint_vector(lower, "target_lower")
        upper = self._joint_vector(upper, "target_upper")
        if np.any(upper <= lower):
            raise ValueError("Every target upper limit must be greater than its lower limit.")
        if np.any(lower < self.joint_lower) or np.any(upper > self.joint_upper):
            raise ValueError("Target limits must be contained inside training joint limits.")
        self.target_lower = lower
        self.target_upper = upper

    def reset(
        self,
        joint_pos: np.ndarray,
        contact_forces: np.ndarray,
        target: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        joint_pos = self._joint_vector(joint_pos, "joint_pos")
        contact_forces = self._contact_vector(contact_forces)
        target = joint_pos if target is None else self._joint_vector(target, "target")
        target = self._clip_target(target)
        frame = self._build_frame(joint_pos, target, contact_forces)

        self.current_target = target.copy()
        self._frames.clear()
        for _ in range(self.contract.history_len):
            self._frames.append(frame.copy())
        return self.policy_inputs()

    def observe(self, joint_pos: np.ndarray, contact_forces: np.ndarray) -> dict[str, np.ndarray]:
        joint_pos = self._joint_vector(joint_pos, "joint_pos")
        contact_forces = self._contact_vector(contact_forces)
        if self.current_target is None:
            return self.reset(joint_pos, contact_forces)
        self._frames.append(self._build_frame(joint_pos, self.current_target, contact_forces))
        return self.policy_inputs()

    def action_to_target(self, action: np.ndarray) -> np.ndarray:
        if self.current_target is None:
            raise RuntimeError("Call reset() before action_to_target().")
        action = np.clip(self._joint_vector(action, "action"), -1.0, 1.0)
        target = self.current_target + self.contract.action_scale * action
        self.current_target = self._clip_target(target)
        return self.current_target.copy()

    def policy_inputs(self) -> dict[str, np.ndarray]:
        if len(self._frames) != self.contract.history_len:
            raise RuntimeError(
                f"History is not initialized: expected {self.contract.history_len} frames, "
                f"got {len(self._frames)}."
            )
        history = np.stack(tuple(self._frames), axis=0).astype(np.float32, copy=False)
        return {
            "obs": history[-self.contract.obs_window :].reshape(1, self.contract.obs_dim),
            "proprio_hist": history.reshape(
                1, self.contract.history_len, self.contract.frame_dim
            ),
        }

    def _build_frame(
        self,
        joint_pos: np.ndarray,
        target: np.ndarray,
        contact_forces: np.ndarray,
    ) -> np.ndarray:
        joint_pos_unscaled = (
            2.0 * joint_pos - self.joint_upper - self.joint_lower
        ) / (self.joint_upper - self.joint_lower)
        scaled_contact_forces = contact_forces * self.contract.contact_force_scale
        frame = np.concatenate((joint_pos_unscaled, target, scaled_contact_forces), axis=0)
        if frame.shape != (self.contract.frame_dim,):
            raise RuntimeError(f"Built frame has unexpected shape {frame.shape}.")
        return frame.astype(np.float32, copy=False)

    def _clip_target(self, target: np.ndarray) -> np.ndarray:
        return np.clip(target, self.target_lower, self.target_upper).astype(np.float32)

    def _joint_vector(self, value: np.ndarray, name: str) -> np.ndarray:
        vector = np.asarray(value, dtype=np.float32).reshape(-1)
        if vector.shape != (self.joint_dim,):
            raise ValueError(f"{name} must have shape ({self.joint_dim},), got {vector.shape}.")
        if not np.isfinite(vector).all():
            raise ValueError(f"{name} contains NaN or infinity.")
        return vector

    def _contact_vector(self, value: np.ndarray) -> np.ndarray:
        vector = np.asarray(value, dtype=np.float32).reshape(-1)
        if vector.shape != (self.contact_dim,):
            raise ValueError(
                f"contact_forces must have shape ({self.contact_dim},), got {vector.shape}."
            )
        if not np.isfinite(vector).all() or np.any(vector < 0.0):
            raise ValueError("contact_forces must contain finite, non-negative values in newtons.")
        return vector
