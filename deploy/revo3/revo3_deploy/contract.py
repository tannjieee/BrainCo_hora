from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml


@dataclass(frozen=True)
class TensorSpec:
    name: str
    shape: tuple[str | int, ...]
    dtype: str


@dataclass(frozen=True)
class PolicyContract:
    """Validated subset of the deploy metadata needed by the runtime."""

    path: Path
    inputs: tuple[TensorSpec, ...]
    outputs: tuple[TensorSpec, ...]
    joint_order: tuple[str, ...]
    contact_order: tuple[str, ...]
    contact_force_scale: float
    obs_dim: int
    history_len: int
    frame_dim: int
    action_dim: int
    obs_window: int
    action_scale: float
    policy_rate_hz: float
    joint_limit_scale: float
    joint_lower_rad: np.ndarray
    joint_upper_rad: np.ndarray
    normalization_baked_in: bool

    @classmethod
    def load(cls, path: str | Path) -> "PolicyContract":
        meta_path = Path(path).expanduser().resolve()
        with meta_path.open("r", encoding="utf-8") as stream:
            cfg = yaml.safe_load(stream) or {}

        io_cfg = _require_mapping(cfg, "io_contract")
        deploy_cfg = _require_mapping(cfg, "deploy_observation_contract")
        normalization_cfg = _require_mapping(cfg, "normalization")

        inputs = tuple(_tensor_spec(item, "input") for item in _require_list(io_cfg, "inputs"))
        outputs = tuple(_tensor_spec(item, "output") for item in _require_list(io_cfg, "outputs"))
        by_name = {spec.name: spec for spec in inputs}
        output_by_name = {spec.name: spec for spec in outputs}
        if set(by_name) != {"obs", "proprio_hist"}:
            raise ValueError(f"Expected ONNX inputs obs/proprio_hist, got {sorted(by_name)}.")
        if set(output_by_name) != {"action"}:
            raise ValueError(f"Expected one action output, got {sorted(output_by_name)}.")

        obs_dim = _fixed_dim(by_name["obs"], 1)
        history_len = _fixed_dim(by_name["proprio_hist"], 1)
        frame_dim = _fixed_dim(by_name["proprio_hist"], 2)
        action_dim = _fixed_dim(output_by_name["action"], 1)

        obs_cfg = _require_mapping(deploy_cfg, "obs")
        history_cfg = _require_mapping(deploy_cfg, "proprio_hist")
        obs_window = int(obs_cfg.get("window_frames", 0))
        if obs_dim != obs_window * frame_dim:
            raise ValueError(
                f"obs dimension {obs_dim} is not obs_window({obs_window}) * frame_dim({frame_dim})."
            )
        if int(history_cfg.get("window_frames", 0)) != history_len:
            raise ValueError("proprio_hist window length disagrees with its tensor shape.")

        joint_order = tuple(str(name) for name in deploy_cfg.get("joint_order") or [])
        if len(joint_order) != action_dim or len(set(joint_order)) != action_dim:
            raise ValueError(f"joint_order must contain {action_dim} unique joints.")

        single_frame = _require_list(deploy_cfg, "single_frame_layout")
        if any(not isinstance(item, dict) for item in single_frame):
            raise ValueError("single_frame_layout entries must be mappings.")
        frame_by_name = {str(item.get("name", "")): item for item in single_frame}
        expected_frame_names = {
            "joint_pos_unscaled",
            "cur_targets",
            "contact_forces",
        }
        if len(single_frame) != len(expected_frame_names) or set(
            frame_by_name
        ) != expected_frame_names:
            raise ValueError(
                f"single_frame_layout must contain exactly {sorted(expected_frame_names)}."
            )
        joint_cfg = frame_by_name["joint_pos_unscaled"]
        target_cfg = frame_by_name["cur_targets"]
        contact_cfg = frame_by_name["contact_forces"]
        if _slice(joint_cfg, "joint_pos_unscaled") != (0, action_dim):
            raise ValueError("joint_pos_unscaled must occupy frame slice [0,21].")
        if str(joint_cfg.get("units", "")) != "dimensionless":
            raise ValueError("joint_pos_unscaled must be dimensionless.")
        if _slice(target_cfg, "cur_targets") != (action_dim, action_dim * 2):
            raise ValueError("cur_targets must occupy frame slice [21,42].")
        if str(target_cfg.get("units", "")) != "radians":
            raise ValueError("cur_targets must use radians.")
        contact_order = tuple(str(name) for name in contact_cfg.get("order") or [])
        contact_slice = _slice(contact_cfg, "contact_forces")
        if len(contact_slice) != 2 or contact_slice[1] - contact_slice[0] != len(contact_order):
            raise ValueError("contact_forces order and slice disagree.")
        if str(contact_cfg.get("units", "")) != "newtons":
            raise ValueError("contact_forces must use newtons.")
        contact_force_scale = float(contact_cfg.get("input_scale", 1.0))
        if not np.isfinite(contact_force_scale) or contact_force_scale <= 0.0:
            raise ValueError("contact_forces input_scale must be finite and positive.")
        if frame_dim != action_dim * 2 + len(contact_order):
            raise ValueError(
                "Only [joint_pos, current_target, contact_forces] frames are supported."
            )
        if contact_slice != (action_dim * 2, frame_dim):
            raise ValueError("contact_forces must be the final five values of each raw frame.")

        normalization_baked_in = bool(normalization_cfg.get("baked_in_onnx", False))
        if not normalization_baked_in:
            raise ValueError("This runtime requires normalization to be baked into the ONNX graph.")
        if str(io_cfg.get("action_semantics", "")) != "delta":
            raise ValueError("Only delta action semantics are supported.")
        action_clip = tuple(float(value) for value in io_cfg.get("action_clip") or ())
        if action_clip != (-1.0, 1.0):
            raise ValueError("The Stage-2 action output must be clipped to [-1,1].")
        if str(io_cfg.get("target_units", "")) != "radians":
            raise ValueError("Stage-2 target units must be radians.")
        if int(io_cfg.get("chunk_size", 0)) != 1 or int(
            io_cfg.get("n_action_steps", 0)
        ) != 1:
            raise ValueError("This runtime requires one action per policy step.")

        dof_limits = _require_mapping(deploy_cfg, "dof_limits")
        joint_limit_scale = float(dof_limits["scaled_by"])
        unscaled_lower = _finite_vector(
            dof_limits.get("unscaled_lower_rad"), action_dim, "unscaled_lower_rad"
        )
        unscaled_upper = _finite_vector(
            dof_limits.get("unscaled_upper_rad"), action_dim, "unscaled_upper_rad"
        )
        joint_lower = _finite_vector(
            dof_limits.get("scaled_lower_rad"), action_dim, "scaled_lower_rad"
        )
        joint_upper = _finite_vector(
            dof_limits.get("scaled_upper_rad"), action_dim, "scaled_upper_rad"
        )
        if np.any(unscaled_upper <= unscaled_lower) or np.any(joint_upper <= joint_lower):
            raise ValueError("Metadata joint limits are invalid.")
        if not np.allclose(
            joint_lower,
            unscaled_lower * joint_limit_scale,
            rtol=0.0,
            atol=1e-6,
        ) or not np.allclose(
            joint_upper,
            unscaled_upper * joint_limit_scale,
            rtol=0.0,
            atol=1e-6,
        ):
            raise ValueError("Scaled metadata joint limits disagree with scaled_by.")
        contract = cls(
            path=meta_path,
            inputs=inputs,
            outputs=outputs,
            joint_order=joint_order,
            contact_order=contact_order,
            contact_force_scale=contact_force_scale,
            obs_dim=obs_dim,
            history_len=history_len,
            frame_dim=frame_dim,
            action_dim=action_dim,
            obs_window=obs_window,
            action_scale=float(io_cfg["action_scale"]),
            policy_rate_hz=float(io_cfg["policy_rate_hz"]),
            joint_limit_scale=joint_limit_scale,
            joint_lower_rad=joint_lower,
            joint_upper_rad=joint_upper,
            normalization_baked_in=normalization_baked_in,
        )
        contract._validate_current_stage2_abi()
        return contract

    def input_spec(self, name: str) -> TensorSpec:
        for spec in self.inputs:
            if spec.name == name:
                return spec
        raise KeyError(name)

    def _validate_current_stage2_abi(self) -> None:
        scalars = (
            self.action_scale,
            self.policy_rate_hz,
            self.joint_limit_scale,
        )
        if not np.isfinite(scalars).all() or any(value <= 0.0 for value in scalars):
            raise ValueError(
                "action_scale, policy_rate_hz, and joint-limit scale must be finite and positive."
            )
        if not np.isclose(self.action_scale, 1.0 / 24.0, rtol=0.0, atol=1e-12):
            raise ValueError("This Stage-2 policy requires action_scale=1/24 rad.")
        if not np.isclose(self.policy_rate_hz, 20.0, rtol=0.0, atol=1e-9):
            raise ValueError("This Stage-2 policy requires a 20 Hz policy rate.")
        if not np.isclose(self.joint_limit_scale, 0.9, rtol=0.0, atol=1e-12):
            raise ValueError("This Stage-2 policy requires joint-limit scale 0.9.")
        expected = (self.obs_dim, self.history_len, self.frame_dim, self.action_dim)
        if expected != (141, 30, 47, 21):
            raise ValueError(
                "This runtime currently supports the tactile HORA Stage-2 ABI "
                f"(141, 30, 47, 21), got {expected}."
            )
        expected_contact_order = (
            "thumb_DIP",
            "index_DIP",
            "middle_DIP",
            "ring_DIP",
            "little_DIP",
        )
        if self.contact_order != expected_contact_order:
            raise ValueError(
                "The tactile Stage-2 ABI requires contact order "
                f"{expected_contact_order}, got {self.contact_order}."
            )


def _tensor_spec(value: Any, kind: str) -> TensorSpec:
    if not isinstance(value, dict):
        raise ValueError(f"Each {kind} spec must be a mapping.")
    name = str(value.get("name", ""))
    shape = value.get("shape")
    if not name or not isinstance(shape, list):
        raise ValueError(f"Invalid {kind} tensor spec: {value!r}.")
    parsed_shape: list[str | int] = []
    for dim in shape:
        parsed_shape.append(int(dim) if isinstance(dim, int) else str(dim))
    return TensorSpec(name=name, shape=tuple(parsed_shape), dtype=str(value.get("dtype", "")))


def _fixed_dim(spec: TensorSpec, index: int) -> int:
    try:
        dim = spec.shape[index]
    except IndexError as exc:
        raise ValueError(f"Tensor {spec.name} has invalid shape {spec.shape}.") from exc
    if not isinstance(dim, int) or dim <= 0:
        raise ValueError(f"Tensor {spec.name} dimension {index} must be a positive integer.")
    return dim


def _slice(cfg: dict[str, Any], name: str) -> tuple[int, ...]:
    try:
        values = tuple(int(value) for value in cfg.get("slice") or ())
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} has an invalid slice.") from exc
    if len(values) != 2 or values[1] <= values[0]:
        raise ValueError(f"{name} has an invalid slice.")
    return values


def _finite_vector(value: Any, size: int, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64).reshape(-1)
    if vector.shape != (size,) or not np.isfinite(vector).all():
        raise ValueError(f"Metadata {name} must contain {size} finite values.")
    return vector.astype(np.float32)


def _require_mapping(cfg: dict[str, Any], key: str) -> dict[str, Any]:
    value = cfg.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Metadata is missing mapping {key!r}.")
    return value


def _require_list(cfg: dict[str, Any], key: str) -> list[Any]:
    value = cfg.get(key)
    if not isinstance(value, list):
        raise ValueError(f"Metadata is missing list {key!r}.")
    return value
