from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort

from .contract import PolicyContract, TensorSpec
from .input_builder import Stage2InputBuilder
from .robot_profile import Revo3Profile


@dataclass(frozen=True)
class PolicyStep:
    onnx_action_raw: np.ndarray
    action: np.ndarray
    policy_target_unclipped_rad: np.ndarray
    policy_target_rad: np.ndarray
    target_clipped: np.ndarray
    # Exact, raw float32 tensors passed to ONNX.  The batch dimension is retained
    # so a recorded step can be replayed without reconstructing history state.
    obs_raw: np.ndarray
    proprio_hist_raw: np.ndarray


class Revo3PolicyRunner:
    """Validated ONNX inference plus delta-target integration."""

    def __init__(
        self,
        onnx_path: str | Path,
        metadata_path: str | Path,
        profile: Revo3Profile,
        provider: str = "cpu",
    ) -> None:
        self.onnx_path = Path(onnx_path).expanduser().resolve()
        if not self.onnx_path.is_file():
            raise FileNotFoundError(self.onnx_path)
        self.contract = PolicyContract.load(metadata_path)
        if self.contract.joint_order != profile.policy_joint_order:
            raise ValueError("Policy metadata joint order differs from the robot profile.")
        if not np.allclose(
            self.contract.joint_lower_rad,
            profile.joint_lower_policy,
            rtol=0.0,
            atol=1e-7,
        ) or not np.allclose(
            self.contract.joint_upper_rad,
            profile.joint_upper_policy,
            rtol=0.0,
            atol=1e-7,
        ):
            raise ValueError(
                "Profile joint limits differ from the numeric training limits in metadata; "
                "joint_pos_unscaled and target clipping would be mis-scaled."
            )
        self.profile = profile

        self.builder = Stage2InputBuilder(
            contract=self.contract,
            joint_lower=profile.joint_lower_policy,
            joint_upper=profile.joint_upper_policy,
            target_lower=profile.target_lower_policy,
            target_upper=profile.target_upper_policy,
        )
        self.session = self._create_session(provider)
        self._validate_onnx_contract()
        self.initialized = False

    @classmethod
    def create(
        cls,
        onnx_path: str | Path,
        metadata_path: str | Path,
        profile_path: str | Path,
        provider: str = "cpu",
    ) -> "Revo3PolicyRunner":
        contract = PolicyContract.load(metadata_path)
        profile = Revo3Profile.load(
            profile_path,
            expected_policy_order=contract.joint_order,
            expected_limit_scale=contract.joint_limit_scale,
            expected_action_scale=contract.action_scale,
            expected_contact_order=contract.contact_order,
        )
        return cls(onnx_path, metadata_path, profile, provider=provider)

    @property
    def rate_hz(self) -> float:
        return self.contract.policy_rate_hz

    @property
    def providers(self) -> list[str]:
        return self.session.get_providers()

    def apply_device_target_limits(
        self,
        device_lower_sdk_rad: np.ndarray,
        device_upper_sdk_rad: np.ndarray,
        margin_rad: float,
    ) -> None:
        """Intersect policy targets with live SDK limits using an inward margin."""
        margin = float(margin_rad)
        if not np.isfinite(margin) or margin < 0.0:
            raise ValueError("Device target margin must be finite and non-negative.")
        lower_sdk = np.asarray(device_lower_sdk_rad, dtype=np.float32).reshape(-1)
        upper_sdk = np.asarray(device_upper_sdk_rad, dtype=np.float32).reshape(-1)
        if (
            lower_sdk.shape != (self.contract.action_dim,)
            or upper_sdk.shape != (self.contract.action_dim,)
            or not np.isfinite(lower_sdk).all()
            or not np.isfinite(upper_sdk).all()
        ):
            raise ValueError("Live SDK limits must contain 21 finite lower/upper values.")
        lower_policy = self.profile.measured_sdk_to_policy(lower_sdk + margin)
        upper_policy = self.profile.measured_sdk_to_policy(upper_sdk - margin)
        effective_lower = np.maximum(self.profile.target_lower_policy, lower_policy)
        effective_upper = np.minimum(self.profile.target_upper_policy, upper_policy)
        self.builder.set_target_limits(effective_lower, effective_upper)

    def reset(
        self,
        measured_policy_pos_rad: np.ndarray,
        fingertip_forces_n: np.ndarray,
    ) -> None:
        self.builder.reset(measured_policy_pos_rad, fingertip_forces_n)
        self.initialized = True

    def step(
        self,
        measured_policy_pos_rad: np.ndarray,
        fingertip_forces_n: np.ndarray,
    ) -> PolicyStep:
        if self.initialized:
            inputs = self.builder.observe(measured_policy_pos_rad, fingertip_forces_n)
        else:
            inputs = self.builder.reset(measured_policy_pos_rad, fingertip_forces_n)
            self.initialized = True

        for name, value in inputs.items():
            if value.dtype != np.float32 or not np.isfinite(value).all():
                raise RuntimeError(f"Policy input {name} is not finite float32.")
        onnx_action_raw = np.asarray(
            self.session.run(["action"], inputs)[0][0],
            dtype=np.float32,
        ).reshape(-1)
        if onnx_action_raw.shape != (self.contract.action_dim,) or not np.isfinite(
            onnx_action_raw
        ).all():
            raise RuntimeError(
                f"Policy produced invalid action with shape {onnx_action_raw.shape}."
            )
        action = np.clip(onnx_action_raw, -1.0, 1.0).astype(np.float32)
        if self.builder.current_target is None:
            raise RuntimeError("Policy input builder lost its current target.")
        target_unclipped = (
            self.builder.current_target + self.contract.action_scale * action
        ).astype(np.float32)
        target_clipped = (target_unclipped < self.builder.target_lower) | (
            target_unclipped > self.builder.target_upper
        )
        target = self.builder.action_to_target(action)
        return PolicyStep(
            onnx_action_raw=onnx_action_raw.copy(),
            action=action.copy(),
            policy_target_unclipped_rad=target_unclipped.copy(),
            policy_target_rad=target.copy(),
            target_clipped=target_clipped.copy(),
            obs_raw=inputs["obs"].copy(),
            proprio_hist_raw=inputs["proprio_hist"].copy(),
        )

    def _create_session(self, provider: str) -> ort.InferenceSession:
        provider = provider.lower()
        available = ort.get_available_providers()
        if provider == "cpu":
            providers = ["CPUExecutionProvider"]
        elif provider == "cuda":
            if "CUDAExecutionProvider" not in available:
                raise RuntimeError(
                    f"CUDAExecutionProvider is unavailable; installed providers: {available}."
                )
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            raise ValueError("provider must be 'cpu' or 'cuda'.")

        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        return ort.InferenceSession(str(self.onnx_path), sess_options=options, providers=providers)

    def _validate_onnx_contract(self) -> None:
        actual_inputs = self.session.get_inputs()
        actual_outputs = self.session.get_outputs()
        expected_inputs = list(self.contract.inputs)
        expected_outputs = list(self.contract.outputs)
        if [item.name for item in actual_inputs] != [item.name for item in expected_inputs]:
            raise ValueError(
                f"ONNX inputs {[item.name for item in actual_inputs]} differ from metadata "
                f"{[item.name for item in expected_inputs]}."
            )
        if [item.name for item in actual_outputs] != [item.name for item in expected_outputs]:
            raise ValueError(
                f"ONNX outputs {[item.name for item in actual_outputs]} differ from metadata "
                f"{[item.name for item in expected_outputs]}."
            )
        for actual, expected in zip(
            actual_inputs + actual_outputs,
            expected_inputs + expected_outputs,
        ):
            self._validate_tensor(actual.name, tuple(actual.shape), actual.type, expected)

    @staticmethod
    def _validate_tensor(
        name: str,
        actual_shape: tuple[object, ...],
        actual_type: str,
        expected: TensorSpec,
    ) -> None:
        if len(actual_shape) != len(expected.shape):
            raise ValueError(
                f"ONNX tensor {name} shape {actual_shape} differs from {expected.shape}."
            )
        for index, (actual_dim, expected_dim) in enumerate(zip(actual_shape, expected.shape)):
            if isinstance(expected_dim, int) and actual_dim != expected_dim:
                raise ValueError(
                    f"ONNX tensor {name} dimension {index} is {actual_dim}, "
                    f"expected {expected_dim}."
                )
            if not isinstance(expected_dim, int):
                if index != 0:
                    raise ValueError(f"Only the batch dimension may be dynamic for tensor {name}.")
                if isinstance(actual_dim, int):
                    raise ValueError(
                        f"ONNX tensor {name} batch is fixed at {actual_dim}; "
                        "metadata requires dynamic."
                    )
        expected_type = {"float32": "tensor(float)"}.get(expected.dtype)
        if expected_type is None or actual_type != expected_type:
            raise ValueError(
                f"ONNX tensor {name} type {actual_type} differs from metadata {expected.dtype}."
            )
