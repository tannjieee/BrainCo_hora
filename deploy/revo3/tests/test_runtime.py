from __future__ import annotations

import argparse
import asyncio
from dataclasses import replace
import hashlib
import json
import signal
import tempfile
import time
import unittest
from unittest.mock import patch
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from revo3_deploy.cli.jog_joint import (
    _build_jog_offsets,
    async_main as jog_async_main,
    build_parser as jog_build_parser,
)
from revo3_deploy.cli.joint_order import (
    _run_with_signal_cleanup,
    async_main as joint_order_async_main,
    build_parser as joint_order_build_parser,
)
from revo3_deploy.cli.offset_calibration import main as offset_calibration_main
from revo3_deploy.cli.replay_trace import (
    _anchor_selected_targets,
    _build_interpolated_targets,
    _execute_replay,
    _resolve_sdk_indices,
    _validate_anchored_measured_position,
    async_main as replay_async_main,
    build_parser as replay_build_parser,
)
from revo3_deploy.cli.run_policy import (
    _build_preposition_targets,
    _load_preposition_sdk_target,
    async_main,
    build_parser,
)
from revo3_deploy.contract import PolicyContract, TensorSpec
from revo3_deploy.input_builder import Stage2InputBuilder
from revo3_deploy.joint_order_session import (
    build_joint_probe_plan,
    create_session,
    load_session,
    verify_session_artifacts,
)
from revo3_deploy.policy_runner import PolicyStep, Revo3PolicyRunner
from revo3_deploy.policy_trace import PolicyTraceRecorder
from revo3_deploy.replay_trace import ReplayTrace
from revo3_deploy.robot_profile import Revo3Profile
from revo3_deploy.sdk_hand_io import Revo3SdkConfig, Revo3SdkHandIO
from revo3_deploy.tactile import FingertipForceAdapter
from revo3_deploy.vision_touch import VisionTouchCollector


REPO_ROOT = Path(__file__).resolve().parents[3]
FIXTURE_META = Path(__file__).resolve().parent / "fixtures/stage2_meta.yaml"
ACTUAL_META = REPO_ROOT / "outputs/revo3_right/onnx/cylinder_stage2.deploy_meta.yaml"
ONNX = REPO_ROOT / "outputs/revo3_right/onnx/cylinder_stage2.onnx"
PROFILE = REPO_ROOT / "deploy/revo3/config/revo3_right.yaml"


def _write_replay_trace_fixture(
    path: Path,
    profile: Revo3Profile,
    *,
    checkpoint_sha256: str = "0" * 64,
    terminal: bool = False,
    tamper_target: bool = False,
    tamper_limits: bool = False,
    tamper_continuity: bool = False,
    round_limit_ulp: bool = False,
) -> np.ndarray:
    frame_count = 4
    lower = np.repeat(
        profile.joint_lower_policy.reshape(1, -1),
        frame_count,
        axis=0,
    ).astype(np.float32)
    upper = np.repeat(
        profile.joint_upper_policy.reshape(1, -1),
        frame_count,
        axis=0,
    ).astype(np.float32)
    base = ((profile.joint_lower_policy + profile.joint_upper_policy) * 0.5).astype(
        np.float32
    )
    action = np.zeros((frame_count, 21), dtype=np.float32)
    action[:, 0] = np.asarray([0.1, -0.2, 0.3, -0.1], dtype=np.float32)
    action[:, 14] = np.asarray([-0.2, 0.1, 0.2, -0.1], dtype=np.float32)
    target_before = np.empty_like(action)
    target = np.empty_like(action)
    for frame in range(frame_count):
        target_before[frame] = base if frame == 0 else target[frame - 1]
        target[frame] = np.clip(
            target_before[frame] + profile.action_scale * action[frame],
            lower[frame],
            upper[frame],
        )
    policy_pos = target.copy()
    policy_pos[:, 5] += np.asarray([0.0, 0.001, -0.001, 0.002], dtype=np.float32)
    if tamper_target:
        target[1, 5] += np.float32(1.0e-3)
    if tamper_limits:
        lower[:, 0] -= np.float32(1.0e-2)
    if round_limit_ulp:
        for _ in range(3):
            upper = np.nextafter(upper, np.float32(-np.inf))
    if tamper_continuity:
        target_before[2, 5] += np.float32(1.0e-3)
        for frame in range(2, frame_count):
            if frame > 2:
                target_before[frame] = target[frame - 1]
            target[frame] = np.clip(
                target_before[frame] + profile.action_scale * action[frame],
                lower[frame],
                upper[frame],
            )

    done = np.zeros(frame_count, dtype=np.bool_)
    if terminal:
        done[-1] = True
    metadata = {
        "schema_name": "hora_policy_trace",
        "schema_version": 1,
        "source": "sim",
        "task": "cylinder",
        "joint_order": list(profile.policy_joint_order),
        "policy_rate_hz": profile.default_rate_hz,
        "action_semantics": "delta",
        "action_scale": profile.action_scale,
        "action_clip": [-1.0, 1.0],
        "target_units": "radians",
        "units": {"joint_position": "rad"},
        "checkpoint": "fixture.ckpt",
        "checkpoint_sha256": checkpoint_sha256,
        "cache_row_actual": 7942,
        "command": "tools/dump_runtime_actions.py",
    }
    np.savez(
        path,
        metadata_json=np.asarray(json.dumps(metadata), dtype=np.str_),
        step_index=np.arange(frame_count, dtype=np.int64),
        sample_time_s=np.arange(frame_count, dtype=np.float64)
        / profile.default_rate_hz,
        action=action,
        policy_pos_rad=policy_pos,
        target_before_policy_rad=target_before,
        policy_target_rad=target,
        joint_lower_policy_rad=lower,
        joint_upper_policy_rad=upper,
        done=done,
        next_state_is_reset=done,
    )
    return target


class RuntimeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.contract = PolicyContract.load(FIXTURE_META)
        cls.profile = Revo3Profile.load(
            PROFILE,
            expected_policy_order=cls.contract.joint_order,
            expected_limit_scale=cls.contract.joint_limit_scale,
            expected_action_scale=cls.contract.action_scale,
            expected_contact_order=cls.contract.contact_order,
        )

    def test_contract(self) -> None:
        self.assertEqual((self.contract.obs_dim, self.contract.history_len), (141, 30))
        self.assertEqual((self.contract.frame_dim, self.contract.action_dim), (47, 21))
        self.assertEqual(len(self.contract.contact_order), 5)
        self.assertAlmostEqual(self.contract.contact_force_scale, 0.1)
        self.assertTrue(self.contract.normalization_baked_in)
        np.testing.assert_allclose(
            self.contract.joint_lower_rad,
            self.profile.joint_lower_policy,
            atol=1e-7,
        )
        np.testing.assert_allclose(
            self.contract.joint_upper_rad,
            self.profile.joint_upper_policy,
            atol=1e-7,
        )
        self.assertAlmostEqual(self.contract.action_scale, 1.0 / 24.0)
        self.assertAlmostEqual(self.contract.policy_rate_hz, 20.0)
        with self.assertRaisesRegex(ValueError, "batch is fixed"):
            Revo3PolicyRunner._validate_tensor(
                "obs",
                (1, 141),
                "tensor(float)",
                TensorSpec("obs", ("B", 141), "float32"),
            )

    def test_profile_scaled_limits_and_roundtrip(self) -> None:
        self.assertAlmostEqual(float(self.profile.joint_lower_policy[0]), -0.23562, places=6)
        self.assertAlmostEqual(float(self.profile.joint_upper_policy[4]), 1.72791, places=6)
        self.assertTrue(
            np.all(self.profile.target_lower_policy >= self.profile.joint_lower_policy)
        )
        self.assertTrue(
            np.all(self.profile.target_upper_policy <= self.profile.joint_upper_policy)
        )
        sdk = np.linspace(-0.2, 1.2, 21, dtype=np.float32)
        policy = self.profile.measured_sdk_to_policy(sdk)
        np.testing.assert_allclose(self.profile.target_policy_to_sdk(policy), sdk, atol=1e-7)
        self.assertEqual(
            self.profile.sdk_joint_order,
            (
                "right_little_MPR_joint", "right_little_MCP_joint",
                "right_little_PIP_joint", "right_little_DIP_joint",
                "right_ring_MPR_joint", "right_ring_MCP_joint",
                "right_ring_PIP_joint", "right_ring_DIP_joint",
                "right_middle_MPR_joint", "right_middle_MCP_joint",
                "right_middle_PIP_joint", "right_middle_DIP_joint",
                "right_index_MPR_joint", "right_index_MCP_joint",
                "right_index_PIP_joint", "right_index_DIP_joint",
                "right_thumb_MCP_joint", "right_thumb_PIP_joint",
                "right_thumb_DIP_joint", "right_thumb_CMP_joint",
                "right_thumb_CMR_joint",
            ),
        )
        self.profile.validate_sdk_position(np.zeros(21, dtype=np.float32), "test")
        invalid = np.zeros(21, dtype=np.float32)
        invalid[1] = -0.1
        with self.assertRaisesRegex(ValueError, "hardware limits"):
            self.profile.validate_sdk_position(invalid, "test")
        encoder_noise = np.zeros(21, dtype=np.float32)
        encoder_noise[1] = -0.005
        with self.assertRaisesRegex(ValueError, "hardware limits"):
            self.profile.validate_sdk_position(encoder_noise, "strict command")
        self.profile.validate_sdk_position(
            encoder_noise,
            "measured position",
            tolerance_rad=0.00872665,
        )

    def test_replay_trace_loads_valid_npz(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sim_trace.npz"
            expected_target = _write_replay_trace_fixture(path, self.profile)

            trace = ReplayTrace.load(path, self.profile)

        self.assertEqual(trace.frame_count, 4)
        self.assertEqual(trace.usable_frame_count, 4)
        self.assertEqual(trace.policy_rate_hz, 20.0)
        np.testing.assert_array_equal(trace.step_index, np.arange(4, dtype=np.int64))
        np.testing.assert_allclose(trace.policy_target_rad, expected_target, atol=0.0)
        self.assertEqual(trace.policy_pos_rad.shape, (4, 21))
        np.testing.assert_array_equal(
            trace.trajectory_policy_rad("measured"),
            trace.policy_pos_rad,
        )
        np.testing.assert_array_equal(trace.select(1, 2), [1, 2])

    def test_replay_trace_accepts_float32_limit_abi_rounding(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rounded_limits_trace.npz"
            _write_replay_trace_fixture(path, self.profile, round_limit_ulp=True)
            trace = ReplayTrace.load(path, self.profile)
        self.assertEqual(trace.frame_count, 4)

    def test_joint_order_plan_uses_applied_delta_and_signed_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            trace_path = Path(directory) / "sim_trace.npz"
            _write_replay_trace_fixture(trace_path, self.profile)
            trace = ReplayTrace.load(trace_path, self.profile)
            plan = build_joint_probe_plan(
                trace,
                self.profile,
                min_delta_deg=0.2,
                max_delta_deg=1.0,
            )

        p00 = plan[0]
        self.assertEqual((p00["policy_index"], p00["sdk_index"]), (0, 12))
        self.assertEqual(p00["candidates"][0]["row"], 2)
        self.assertGreater(p00["candidates"][0]["delta_deg"], 0.0)
        self.assertEqual(p00["candidates"][1]["row"], 1)
        self.assertLess(p00["candidates"][1]["delta_deg"], 0.0)
        self.assertEqual(plan[5]["state"], "unavailable")

    def test_joint_order_session_hashes_and_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "fixture.ckpt"
            checkpoint.write_bytes(b"joint order session checkpoint")
            checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            trace_path = root / "sim_trace.npz"
            _write_replay_trace_fixture(
                trace_path,
                self.profile,
                checkpoint_sha256=checkpoint_sha,
            )
            trace = ReplayTrace.load(trace_path, self.profile, checkpoint)
            session_path = root / "session.json"
            create_session(
                path=session_path,
                trace_path=trace_path,
                checkpoint_path=checkpoint,
                profile_path=PROFILE,
                trace=trace,
                profile=self.profile,
                min_delta_deg=0.2,
                max_delta_deg=1.0,
            )
            session = load_session(session_path)
            verify_session_artifacts(session)
            with self.assertRaises(FileExistsError):
                create_session(
                    path=session_path,
                    trace_path=trace_path,
                    checkpoint_path=checkpoint,
                    profile_path=PROFILE,
                    trace=trace,
                    profile=self.profile,
                    min_delta_deg=0.2,
                    max_delta_deg=1.0,
                )
            multi_serial_profile = replace(
                self.profile,
                sdk={
                    **self.profile.sdk,
                    "serial_allowlist": ["HAND-A", "HAND-B"],
                },
            )
            with self.assertRaisesRegex(RuntimeError, "exactly one"):
                create_session(
                    path=root / "multi_serial_session.json",
                    trace_path=trace_path,
                    checkpoint_path=checkpoint,
                    profile_path=PROFILE,
                    trace=trace,
                    profile=multi_serial_profile,
                    min_delta_deg=0.2,
                    max_delta_deg=1.0,
                )
            checkpoint.write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "checkpoint SHA256 mismatch"):
                verify_session_artifacts(session)

    def test_joint_order_probe_requires_one_time_challenge_and_records_match(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "fixture.ckpt"
            checkpoint.write_bytes(b"joint order probe checkpoint")
            checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            trace_path = root / "sim_trace.npz"
            _write_replay_trace_fixture(
                trace_path,
                self.profile,
                checkpoint_sha256=checkpoint_sha,
            )
            trace = ReplayTrace.load(trace_path, self.profile, checkpoint)
            session_path = root / "session.json"
            create_session(
                path=session_path,
                trace_path=trace_path,
                checkpoint_path=checkpoint,
                profile_path=PROFILE,
                trace=trace,
                profile=self.profile,
                min_delta_deg=0.2,
                max_delta_deg=1.0,
            )
            args = joint_order_build_parser().parse_args(
                [
                    "probe",
                    "--session", str(session_path),
                    "--joint", "P00",
                    "--allow-unverified-calibration",
                ]
            )
            replay_calls: list[argparse.Namespace] = []

            async def fake_replay(namespace) -> int:
                replay_calls.append(namespace)
                return 0

            with patch(
                "revo3_deploy.cli.joint_order.secrets.token_hex",
                side_effect=["abcd", "ef01"],
            ):
                responses = iter(
                    [
                        "ARM SN=BCUVR1205J2600002 P00 M12 ROW002 +0.716deg "
                        "UNVERIFIED_CALIBRATION ABCD",
                        "MATCH EF01",
                    ]
                )
                result = asyncio.run(
                    joint_order_async_main(
                        args,
                        prompt=lambda _: next(responses),
                        replay_runner=fake_replay,
                    )
                )

            self.assertEqual(result, 0)
            self.assertEqual(len(replay_calls), 2)
            self.assertTrue(replay_calls[0].preflight)
            self.assertFalse(replay_calls[0].execute)
            self.assertTrue(replay_calls[1].execute)
            self.assertEqual(replay_calls[1].start_frame, 2)
            self.assertEqual(replay_calls[1].joint, ["P00"])
            self.assertTrue(replay_calls[1].confirm_current_anchor)
            self.assertEqual(
                replay_calls[1].expected_trace_sha256,
                hashlib.sha256(trace_path.read_bytes()).hexdigest(),
            )
            self.assertEqual(
                replay_calls[1].expected_checkpoint_sha256,
                checkpoint_sha,
            )
            session = load_session(session_path)
            self.assertEqual(session["joints"][0]["state"], "passed")
            attempt = session["joints"][0]["attempts"][0]
            self.assertEqual(attempt["execution"]["status"], "passed_released_closed")
            self.assertEqual(attempt["observation"]["verdict"], "passed")

    def test_joint_order_probe_challenge_mismatch_sends_no_motion(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "fixture.ckpt"
            checkpoint.write_bytes(b"joint order declined checkpoint")
            checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            trace_path = root / "sim_trace.npz"
            _write_replay_trace_fixture(
                trace_path,
                self.profile,
                checkpoint_sha256=checkpoint_sha,
            )
            trace = ReplayTrace.load(trace_path, self.profile, checkpoint)
            session_path = root / "session.json"
            create_session(
                path=session_path,
                trace_path=trace_path,
                checkpoint_path=checkpoint,
                profile_path=PROFILE,
                trace=trace,
                profile=self.profile,
                min_delta_deg=0.2,
                max_delta_deg=1.0,
            )
            args = joint_order_build_parser().parse_args(
                [
                    "probe",
                    "--session", str(session_path),
                    "--joint", "P00",
                    "--allow-unverified-calibration",
                ]
            )
            replay_modes: list[str] = []

            async def fake_replay(namespace) -> int:
                replay_modes.append("execute" if namespace.execute else "preflight")
                return 0

            result = asyncio.run(
                joint_order_async_main(
                    args,
                    prompt=lambda _: "wrong approval",
                    replay_runner=fake_replay,
                )
            )
            self.assertEqual(result, 2)
            self.assertEqual(replay_modes, ["preflight"])
            session = load_session(session_path)
            self.assertEqual(session["joints"][0]["state"], "planned")
            replay_modes.clear()
            with patch(
                "revo3_deploy.cli.joint_order.TtyPrompter",
                side_effect=RuntimeError("controlling TTY required"),
            ):
                with self.assertRaisesRegex(RuntimeError, "controlling TTY"):
                    asyncio.run(
                        joint_order_async_main(
                            args,
                            replay_runner=fake_replay,
                        )
                    )
            self.assertEqual(replay_modes, [])

    def test_joint_order_old_approval_nonce_cannot_record_match(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "fixture.ckpt"
            checkpoint.write_bytes(b"joint order independent verdict nonce")
            checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            trace_path = root / "sim_trace.npz"
            _write_replay_trace_fixture(
                trace_path,
                self.profile,
                checkpoint_sha256=checkpoint_sha,
            )
            trace = ReplayTrace.load(trace_path, self.profile, checkpoint)
            session_path = root / "session.json"
            create_session(
                path=session_path,
                trace_path=trace_path,
                checkpoint_path=checkpoint,
                profile_path=PROFILE,
                trace=trace,
                profile=self.profile,
                min_delta_deg=0.2,
                max_delta_deg=1.0,
            )
            args = joint_order_build_parser().parse_args(
                [
                    "probe",
                    "--session", str(session_path),
                    "--joint", "P00",
                    "--allow-unverified-calibration",
                ]
            )

            async def fake_replay(_namespace) -> int:
                return 0

            responses = iter(
                [
                    "ARM SN=BCUVR1205J2600002 P00 M12 ROW002 +0.716deg "
                    "UNVERIFIED_CALIBRATION ABCD",
                    "MATCH ABCD",
                ]
            )
            with patch(
                "revo3_deploy.cli.joint_order.secrets.token_hex",
                side_effect=["abcd", "ef01"],
            ):
                result = asyncio.run(
                    joint_order_async_main(
                        args,
                        prompt=lambda _: next(responses),
                        replay_runner=fake_replay,
                    )
                )

            self.assertEqual(result, 3)
            session = load_session(session_path)
            self.assertEqual(session["state"], "blocked")
            observation = session["joints"][0]["attempts"][0]["observation"]
            self.assertFalse(observation["response_valid"])
            self.assertEqual(observation["nonce"], "EF01")

    def test_joint_order_probe_revalidates_artifacts_after_approval(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "fixture.ckpt"
            checkpoint.write_bytes(b"joint order artifact binding checkpoint")
            checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            trace_path = root / "sim_trace.npz"
            _write_replay_trace_fixture(
                trace_path,
                self.profile,
                checkpoint_sha256=checkpoint_sha,
            )
            trace = ReplayTrace.load(trace_path, self.profile, checkpoint)
            session_path = root / "session.json"
            create_session(
                path=session_path,
                trace_path=trace_path,
                checkpoint_path=checkpoint,
                profile_path=PROFILE,
                trace=trace,
                profile=self.profile,
                min_delta_deg=0.2,
                max_delta_deg=1.0,
            )
            args = joint_order_build_parser().parse_args(
                [
                    "probe",
                    "--session", str(session_path),
                    "--joint", "P00",
                    "--allow-unverified-calibration",
                ]
            )
            replay_modes: list[str] = []

            async def fake_replay(namespace) -> int:
                replay_modes.append("execute" if namespace.execute else "preflight")
                return 0

            def mutate_then_approve(_: str) -> str:
                checkpoint.write_bytes(b"changed while operator reviewed")
                return (
                    "ARM SN=BCUVR1205J2600002 P00 M12 ROW002 +0.716deg "
                    "UNVERIFIED_CALIBRATION ABCD"
                )

            with patch(
                "revo3_deploy.cli.joint_order.secrets.token_hex",
                side_effect=["abcd", "ef01"],
            ):
                with self.assertRaisesRegex(ValueError, "checkpoint SHA256 mismatch"):
                    asyncio.run(
                        joint_order_async_main(
                            args,
                            prompt=mutate_then_approve,
                            replay_runner=fake_replay,
                        )
                    )

            self.assertEqual(replay_modes, ["preflight"])
            session = load_session(session_path)
            self.assertEqual(session["state"], "blocked")
            self.assertEqual(
                session["joints"][0]["attempts"][0]["execution"]["status"],
                "artifact_revalidation_failed",
            )

    def test_joint_order_signals_cancel_and_wait_for_cleanup(self) -> None:
        async def scenario(signum: signal.Signals) -> tuple[int, bool]:
            loop = asyncio.get_running_loop()
            handlers: dict[signal.Signals, tuple[object, tuple[object, ...]]] = {}
            started = asyncio.Event()
            cleaned = False

            async def fake_runner(_args) -> int:
                nonlocal cleaned
                started.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    cleaned = True

            def add_handler(signum, callback, *callback_args) -> None:
                handlers[signum] = (callback, callback_args)

            with patch.object(loop, "add_signal_handler", side_effect=add_handler), patch.object(
                loop,
                "remove_signal_handler",
                return_value=True,
            ):
                task = asyncio.create_task(
                    _run_with_signal_cleanup(
                        SimpleNamespace(),
                        runner=fake_runner,
                    )
                )
                await started.wait()
                callback, callback_args = handlers[signum]
                callback(*callback_args)
                result = await task
            return result, cleaned

        for signum in (
            signal.SIGINT,
            signal.SIGQUIT,
            signal.SIGTSTP,
            signal.SIGTERM,
            signal.SIGHUP,
        ):
            with self.subTest(signum=signum):
                result, cleaned = asyncio.run(scenario(signum))
                self.assertEqual(result, 128 + signum)
                self.assertTrue(cleaned)

    def test_replay_trace_verifies_checkpoint_sha256(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "fixture.ckpt"
            checkpoint.write_bytes(b"known replay checkpoint fixture")
            checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            trace_path = root / "sim_trace.npz"
            _write_replay_trace_fixture(
                trace_path,
                self.profile,
                checkpoint_sha256=checkpoint_sha,
            )

            trace = ReplayTrace.load(
                trace_path,
                self.profile,
                checkpoint_path=checkpoint,
            )
            self.assertEqual(trace.metadata["checkpoint_sha256"], checkpoint_sha)

            wrong_checkpoint = root / "wrong.ckpt"
            wrong_checkpoint.write_bytes(b"different checkpoint")
            with self.assertRaisesRegex(ValueError, "checkpoint SHA256"):
                ReplayTrace.load(
                    trace_path,
                    self.profile,
                    checkpoint_path=wrong_checkpoint,
                )

    def test_replay_joint_selectors_resolve_to_sdk_motor_order(self) -> None:
        self.assertEqual(
            _resolve_sdk_indices(
                self.profile,
                ["M13", "P0", "right_thumb_CMP_joint"],
                False,
            ),
            (13, 12, 19),
        )
        self.assertEqual(
            _resolve_sdk_indices(self.profile, [], True),
            tuple(range(21)),
        )
        with self.assertRaisesRegex(ValueError, "unique SDK motors"):
            _resolve_sdk_indices(self.profile, ["M12", "P0"], False)

    def test_replay_trace_rejects_tampered_policy_target(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tampered_trace.npz"
            _write_replay_trace_fixture(path, self.profile, tamper_target=True)

            with self.assertRaisesRegex(ValueError, "policy_target_rad does not match"):
                ReplayTrace.load(path, self.profile)

    def test_replay_trace_rejects_tampered_limits_and_target_continuity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cases = (
                ("limits", {"tamper_limits": True}, "joint limits differ"),
                (
                    "continuity",
                    {"tamper_continuity": True},
                    "target_before sequence is discontinuous",
                ),
            )
            for name, options, error_pattern in cases:
                with self.subTest(name=name):
                    path = root / f"{name}.npz"
                    _write_replay_trace_fixture(path, self.profile, **options)
                    with self.assertRaisesRegex(ValueError, error_pattern):
                        ReplayTrace.load(path, self.profile)

    def test_replay_trace_never_selects_terminal_row(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "terminal_trace.npz"
            _write_replay_trace_fixture(path, self.profile, terminal=True)
            trace = ReplayTrace.load(path, self.profile)

        self.assertEqual(trace.frame_count, 4)
        self.assertEqual(trace.usable_frame_count, 3)
        np.testing.assert_array_equal(trace.select(0, None), [0, 1, 2])
        with self.assertRaisesRegex(ValueError, "outside 3 non-terminal"):
            trace.select(3, 1)

    def test_replay_execute_refuses_missing_frames_or_confirmations_before_hardware(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "fixture.ckpt"
            checkpoint.write_bytes(b"execute guard checkpoint")
            checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            trace_path = root / "sim_trace.npz"
            _write_replay_trace_fixture(
                trace_path,
                self.profile,
                checkpoint_sha256=checkpoint_sha,
            )
            base = [
                "--trace-npz",
                str(trace_path),
                "--profile",
                str(PROFILE),
                "--checkpoint",
                str(checkpoint),
                "--execute",
                "--joint",
                "M13",
            ]
            cases = (
                (base, "explicit --frames"),
                (base + ["--frames", "1"], "all five physical/mapping confirmations"),
            )
            for argv, error_pattern in cases:
                with self.subTest(error_pattern=error_pattern):
                    args = replay_build_parser().parse_args(argv)
                    with patch(
                        "revo3_deploy.cli.replay_trace.Revo3SdkHandIO"
                    ) as hand_io:
                        with self.assertRaisesRegex(RuntimeError, error_pattern):
                            asyncio.run(replay_async_main(args))
                        hand_io.assert_not_called()

    def test_replay_hardware_mode_requires_checkpoint_before_hardware(self) -> None:
        args = replay_build_parser().parse_args(
            [
                "--trace-npz", "unused.npz",
                "--profile", str(PROFILE),
                "--preflight",
                "--joint", "M13",
            ]
        )
        with patch("revo3_deploy.cli.replay_trace.Revo3SdkHandIO") as hand_io:
            with self.assertRaisesRegex(RuntimeError, "requires --checkpoint"):
                asyncio.run(replay_async_main(args))
            hand_io.assert_not_called()

    def test_full_hand_execute_requires_measured_source_and_confirmation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "fixture.ckpt"
            checkpoint.write_bytes(b"full hand guard checkpoint")
            checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            trace_path = root / "sim_trace.npz"
            _write_replay_trace_fixture(
                trace_path,
                self.profile,
                checkpoint_sha256=checkpoint_sha,
            )
            base = [
                "--trace-npz", str(trace_path),
                "--profile", str(PROFILE),
                "--checkpoint", str(checkpoint),
                "--execute",
                "--frames", "1",
                "--all-joints",
            ]
            cases = (
                (base, "limited to --trajectory-source measured"),
                (
                    base + ["--trajectory-source", "measured"],
                    "requires --confirm-full-hand",
                ),
            )
            for argv, error_pattern in cases:
                with self.subTest(error_pattern=error_pattern):
                    args = replay_build_parser().parse_args(argv)
                    with patch(
                        "revo3_deploy.cli.replay_trace.Revo3SdkHandIO"
                    ) as hand_io:
                        with self.assertRaisesRegex(RuntimeError, error_pattern):
                            asyncio.run(replay_async_main(args))
                        hand_io.assert_not_called()

    def test_ignore_all_stall_requires_hardware_mode(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            trace_path = Path(directory) / "sim_trace.npz"
            _write_replay_trace_fixture(trace_path, self.profile)
            args = replay_build_parser().parse_args(
                [
                    "--trace-npz", str(trace_path),
                    "--profile", str(PROFILE),
                    "--joint", "M13",
                    "--ignore-all-stall",
                ]
            )
            with patch("revo3_deploy.cli.replay_trace.Revo3SdkHandIO") as hand_io:
                with self.assertRaisesRegex(ValueError, "--preflight or --execute"):
                    asyncio.run(replay_async_main(args))
                hand_io.assert_not_called()

    def test_recorded_rate_argument_guards_precede_hardware(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "fixture.ckpt"
            checkpoint.write_bytes(b"recorded rate guard checkpoint")
            checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            trace_path = root / "sim_trace.npz"
            _write_replay_trace_fixture(
                trace_path,
                self.profile,
                checkpoint_sha256=checkpoint_sha,
            )
            common = [
                "--trace-npz", str(trace_path),
                "--profile", str(PROFILE),
            ]
            cases = (
                (common + ["--recorded-rate"], "requires --trajectory-source measured"),
                (
                    common + ["--preposition-to-first"],
                    "requires --recorded-rate",
                ),
                (
                    common + [
                        "--checkpoint", str(checkpoint),
                        "--trajectory-source", "measured",
                        "--recorded-rate",
                        "--frames", "1",
                        "--joint", "M13",
                        "--execute",
                    ],
                    "requires --confirm-recorded-rate",
                ),
            )
            for argv, error_pattern in cases:
                with self.subTest(error_pattern=error_pattern):
                    args = replay_build_parser().parse_args(argv)
                    with patch(
                        "revo3_deploy.cli.replay_trace.Revo3SdkHandIO"
                    ) as hand_io:
                        with self.assertRaisesRegex((ValueError, RuntimeError), error_pattern):
                            asyncio.run(replay_async_main(args))
                        hand_io.assert_not_called()

    def test_final_pose_hold_requires_execute_and_confirmation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "fixture.ckpt"
            checkpoint.write_bytes(b"final hold guard checkpoint")
            checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            trace_path = root / "sim_trace.npz"
            _write_replay_trace_fixture(
                trace_path,
                self.profile,
                checkpoint_sha256=checkpoint_sha,
            )
            cases = (
                (
                    [
                        "--trace-npz", str(trace_path),
                        "--profile", str(PROFILE),
                        "--joint", "M13",
                        "--hold-final-s", "30",
                    ],
                    "requires --execute",
                ),
                (
                    [
                        "--trace-npz", str(trace_path),
                        "--profile", str(PROFILE),
                        "--checkpoint", str(checkpoint),
                        "--joint", "M13",
                        "--frames", "1",
                        "--execute",
                        "--hold-final-s", "30",
                    ],
                    "requires --confirm-hold",
                ),
            )
            for argv, pattern in cases:
                with self.subTest(pattern=pattern):
                    args = replay_build_parser().parse_args(argv)
                    with patch("revo3_deploy.cli.replay_trace.Revo3SdkHandIO") as hand_io:
                        with self.assertRaisesRegex((ValueError, RuntimeError), pattern):
                            asyncio.run(replay_async_main(args))
                        hand_io.assert_not_called()

    def test_replay_rejects_excessive_measured_limit_tolerance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            trace_path = Path(directory) / "sim_trace.npz"
            _write_replay_trace_fixture(trace_path, self.profile)
            args = replay_build_parser().parse_args(
                [
                    "--trace-npz", str(trace_path),
                    "--profile", str(PROFILE),
                    "--measured-limit-tolerance-deg", "5.1",
                ]
            )
            with patch("revo3_deploy.cli.replay_trace.Revo3SdkHandIO") as hand_io:
                with self.assertRaisesRegex(ValueError, "finite and in \\[0,5\\]"):
                    asyncio.run(replay_async_main(args))
                hand_io.assert_not_called()

    def test_offset_calibration_creates_versioned_candidate_profiles(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trace_path = root / "sim_trace.npz"
            _write_replay_trace_fixture(trace_path, self.profile)
            candidate_v1 = root / "offset_v01.yaml"
            result = offset_calibration_main(
                [
                    "init",
                    "--profile", str(PROFILE),
                    "--trace-npz", str(trace_path),
                    "--frame", "0",
                    "--output-profile", str(candidate_v1),
                ]
            )
            self.assertEqual(result, 0)
            self.assertTrue(candidate_v1.is_file())
            candidate_v1_profile = Revo3Profile.load(candidate_v1)
            np.testing.assert_allclose(
                candidate_v1_profile.sdk_offset_rad,
                self.profile.sdk_offset_rad,
                atol=0.0,
            )

            candidate_v2 = root / "offset_v02.yaml"
            result = offset_calibration_main(
                [
                    "adjust",
                    "--profile", str(candidate_v1),
                    "--add", "M16=+2.5",
                    "--set", "M13=-1.25",
                    "--output-profile", str(candidate_v2),
                ]
            )
            self.assertEqual(result, 0)
            candidate_v2_profile = Revo3Profile.load(candidate_v2)
            self.assertAlmostEqual(
                float(np.rad2deg(candidate_v2_profile.sdk_offset_rad[16])),
                2.5,
                places=5,
            )
            self.assertAlmostEqual(
                float(np.rad2deg(candidate_v2_profile.sdk_offset_rad[13])),
                -1.25,
                places=5,
            )
            with self.assertRaises(FileExistsError):
                offset_calibration_main(
                    [
                        "adjust",
                        "--profile", str(candidate_v1),
                        "--add", "M16=+1",
                        "--output-profile", str(candidate_v2),
                    ]
                )

    def test_replay_bound_artifact_hash_mismatch_precedes_hardware(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "fixture.ckpt"
            checkpoint.write_bytes(b"bound replay checkpoint")
            checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            trace_path = root / "sim_trace.npz"
            _write_replay_trace_fixture(
                trace_path,
                self.profile,
                checkpoint_sha256=checkpoint_sha,
            )
            args = replay_build_parser().parse_args(
                [
                    "--trace-npz", str(trace_path),
                    "--profile", str(PROFILE),
                    "--checkpoint", str(checkpoint),
                    "--preflight",
                    "--joint", "M13",
                    "--expected-trace-sha256", "0" * 64,
                ]
            )
            with patch("revo3_deploy.cli.replay_trace.Revo3SdkHandIO") as hand_io:
                with self.assertRaisesRegex(ValueError, "Bound trace SHA256 mismatch"):
                    asyncio.run(replay_async_main(args))
                hand_io.assert_not_called()

    def test_replay_current_anchor_guards_run_before_hardware(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "fixture.ckpt"
            checkpoint.write_bytes(b"current anchor guard checkpoint")
            checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            trace_path = root / "sim_trace.npz"
            _write_replay_trace_fixture(
                trace_path,
                self.profile,
                checkpoint_sha256=checkpoint_sha,
            )
            offline_anchor_args = replay_build_parser().parse_args(
                [
                    "--trace-npz", str(trace_path),
                    "--profile", str(PROFILE),
                    "--checkpoint", str(checkpoint),
                    "--joint", "M13",
                    "--anchor-current",
                ]
            )
            with patch("revo3_deploy.cli.replay_trace.Revo3SdkHandIO") as hand_io:
                with self.assertRaisesRegex(ValueError, "requires --preflight"):
                    asyncio.run(replay_async_main(offline_anchor_args))
                hand_io.assert_not_called()

            execute_args = replay_build_parser().parse_args(
                [
                    "--trace-npz", str(trace_path),
                    "--profile", str(PROFILE),
                    "--checkpoint", str(checkpoint),
                    "--frames", "1",
                    "--joint", "M13",
                    "--anchor-current",
                    "--execute",
                    "--confirm-fixed",
                    "--confirm-clear-path",
                    "--confirm-estop",
                    "--confirm-release",
                    "--confirm-mapping",
                ]
            )
            with patch("revo3_deploy.cli.replay_trace.Revo3SdkHandIO") as hand_io:
                with self.assertRaisesRegex(RuntimeError, "confirm-current-anchor"):
                    asyncio.run(replay_async_main(execute_args))
                hand_io.assert_not_called()

            all_joint_args = replay_build_parser().parse_args(
                [
                    "--trace-npz", str(trace_path),
                    "--profile", str(PROFILE),
                    "--checkpoint", str(checkpoint),
                    "--all-joints",
                    "--anchor-current",
                    "--preflight",
                ]
            )
            with patch("revo3_deploy.cli.replay_trace.Revo3SdkHandIO") as hand_io:
                with self.assertRaisesRegex(RuntimeError, "exactly one"):
                    asyncio.run(replay_async_main(all_joint_args))
                hand_io.assert_not_called()

    def test_replay_interpolation_preserves_endpoints_and_speed_bound(self) -> None:
        start = np.asarray([0.0, 0.0], dtype=np.float32)
        source = np.asarray(
            [[0.003, -0.001], [0.001, 0.002]],
            dtype=np.float32,
        )
        max_step = float(np.deg2rad(0.1))
        plan, source_offsets = _build_interpolated_targets(start, source, max_step)
        all_steps = np.diff(np.vstack((start, plan)), axis=0)
        self.assertLessEqual(float(np.max(np.abs(all_steps))), max_step + 1e-7)
        np.testing.assert_allclose(plan[np.flatnonzero(source_offsets == 0)[-1]], source[0])
        np.testing.assert_allclose(plan[-1], source[-1])

    def test_replay_current_anchor_preserves_first_action_and_trace_deltas(self) -> None:
        baseline = self.profile.target_policy_to_sdk(
            (self.profile.joint_lower_policy + self.profile.joint_upper_policy) * 0.5
        )
        targets = np.repeat(baseline.reshape(1, -1), 3, axis=0)
        targets[:, 13] += np.deg2rad([2.0, -1.0, 0.5])
        live_start = baseline.copy()
        live_start[13] -= np.deg2rad(20.0)

        anchored = _anchor_selected_targets(targets, baseline, live_start, (13,))

        self.assertAlmostEqual(
            float(np.rad2deg(anchored[0, 13] - live_start[13])),
            2.0,
            places=5,
        )
        np.testing.assert_allclose(
            np.diff(anchored[:, 13]),
            np.diff(targets[:, 13]),
            atol=1e-7,
        )
        np.testing.assert_array_equal(anchored[:, 14], targets[:, 14])

    def test_replay_current_anchor_separates_passive_and_selected_limits(self) -> None:
        profile = self.profile
        measured = self.profile.sdk_position_lower_rad.copy()
        measured[13] = np.deg2rad(20.0)
        measured[14] = np.deg2rad(-3.03)

        class FakeIO:
            device_position_lower_rad = profile.sdk_position_lower_rad.copy()
            device_position_upper_rad = profile.sdk_position_upper_rad.copy()

        io = FakeIO()
        io.device_position_lower_rad[14] = np.deg2rad(-12.0)
        _validate_anchored_measured_position(
            profile=self.profile,
            io=io,
            measured=measured,
            selected_sdk=(13,),
            selected_lower=self.profile.sdk_position_lower_rad,
            selected_upper=self.profile.sdk_position_upper_rad,
        )

        with self.assertRaisesRegex(RuntimeError, "selected-joint inward"):
            _validate_anchored_measured_position(
                profile=self.profile,
                io=io,
                measured=measured,
                selected_sdk=(14,),
                selected_lower=self.profile.sdk_position_lower_rad,
                selected_upper=self.profile.sdk_position_upper_rad,
            )

        io.device_position_lower_rad[14] = 0.0
        with self.assertRaisesRegex(RuntimeError, "device-reported limits"):
            _validate_anchored_measured_position(
                profile=self.profile,
                io=io,
                measured=measured,
                selected_sdk=(13,),
                selected_lower=self.profile.sdk_position_lower_rad,
                selected_upper=self.profile.sdk_position_upper_rad,
            )
        _validate_anchored_measured_position(
            profile=self.profile,
            io=io,
            measured=measured,
            selected_sdk=(13,),
            selected_lower=self.profile.sdk_position_lower_rad,
            selected_upper=self.profile.sdk_position_upper_rad,
            allow_passive_limit_outlier=True,
        )

    def test_replay_current_anchor_uses_fresh_start_and_zero_gain_passive_slots(self) -> None:
        profile = self.profile
        policy_baseline = (
            self.profile.joint_lower_policy + self.profile.joint_upper_policy
        ) * 0.5
        trace_baseline = self.profile.target_policy_to_sdk(policy_baseline)
        trace_endpoint = trace_baseline.copy()
        trace_endpoint[13] += np.deg2rad(2.0)
        fresh_start = trace_baseline.copy()
        fresh_start[13] -= np.deg2rad(20.0)
        fresh_start[14] = np.deg2rad(-3.03)
        selected_sdk = (13,)
        rows = np.asarray([0], dtype=np.int64)
        trace = SimpleNamespace(step_index=np.asarray([0], dtype=np.int64))

        class FakeIO:
            def __init__(self, *, runaway: bool = False) -> None:
                self.position = fresh_start.copy()
                self.runaway = runaway
                self.device_position_lower_rad = profile.sdk_position_lower_rad.copy()
                self.device_position_upper_rad = profile.sdk_position_upper_rad.copy()
                self.last_motor_currents_ma = np.zeros(21, dtype=np.float32)
                self.sent: list[
                    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
                ] = []

            async def read_position_rad(self, check_errors: bool = False) -> np.ndarray:
                return self.position.copy()

            async def send_mit_command_rad(self, position_rad, *, kp, kd, effort_ma) -> None:
                command = np.asarray(position_rad, dtype=np.float32).copy()
                kp_array = np.asarray(kp).copy()
                kd_array = np.asarray(kd).copy()
                effort_array = np.asarray(effort_ma).copy()
                self.sent.append((command, kp_array, kd_array, effort_array))
                active = kp_array > 0.0
                self.position[active] = command[active]
                if self.runaway:
                    self.position[13] = fresh_start[13] + np.deg2rad(4.0)

            def validate_device_position(self, position_rad) -> None:
                value = np.asarray(position_rad)
                if np.any(value < self.device_position_lower_rad) or np.any(
                    value > self.device_position_upper_rad
                ):
                    raise ValueError("fake live device command limit")

        async def no_sleep(_: float) -> None:
            return None

        io = FakeIO()
        io.profile = self.profile
        live_lower = io.device_position_lower_rad + np.deg2rad(0.05)
        live_upper = io.device_position_upper_rad - np.deg2rad(0.05)
        with patch("revo3_deploy.cli.replay_trace.asyncio.sleep", new=no_sleep):
            asyncio.run(
                _execute_replay(
                    args=SimpleNamespace(print_every=1),
                    io=io,
                    profile=self.profile,
                    trace=trace,
                    rows=rows,
                    selected_sdk=selected_sdk,
                    sdk_targets=trace_endpoint.reshape(1, -1),
                    lower=self.profile.sdk_position_lower_rad,
                    upper=self.profile.sdk_position_upper_rad,
                    measured_tolerance=np.deg2rad(1.0),
                    initial_delta_limit=np.deg2rad(5.0),
                    max_speed_deg_s=2.0,
                    confirm_large_excursion=True,
                    kp=0.2,
                    kd=0.05,
                    max_current_ma=500.0,
                    anchor_current=True,
                    trace_baseline_sdk=trace_baseline,
                    step_delta_limit=np.deg2rad(3.0),
                    live_lower=live_lower,
                    live_upper=live_upper,
                    allow_passive_limit_outlier=True,
                )
            )

        self.assertGreater(len(io.sent), 0)
        final_command, final_kp, final_kd, final_effort = io.sent[-1]
        self.assertAlmostEqual(
            float(np.rad2deg(final_command[13] - fresh_start[13])),
            2.0,
            places=4,
        )
        self.assertAlmostEqual(float(np.rad2deg(final_command[14])), 0.05, places=3)
        self.assertGreater(float(final_kp[13]), 0.0)
        self.assertEqual(float(final_kp[14]), 0.0)
        self.assertEqual(float(final_kd[14]), 0.0)
        self.assertEqual(float(final_effort[14]), 0.0)

        stale_io = FakeIO()
        stale_io.profile = self.profile
        with patch(
            "revo3_deploy.cli.replay_trace._monotonic",
            side_effect=[0.0, 0.1],
        ):
            with self.assertRaisesRegex(RuntimeError, "expired before command"):
                asyncio.run(
                    _execute_replay(
                        args=SimpleNamespace(print_every=1),
                        io=stale_io,
                        profile=self.profile,
                        trace=trace,
                        rows=rows,
                        selected_sdk=selected_sdk,
                        sdk_targets=trace_endpoint.reshape(1, -1),
                        lower=self.profile.sdk_position_lower_rad,
                        upper=self.profile.sdk_position_upper_rad,
                        measured_tolerance=np.deg2rad(1.0),
                        initial_delta_limit=np.deg2rad(5.0),
                        max_speed_deg_s=2.0,
                        confirm_large_excursion=True,
                        kp=0.2,
                        kd=0.05,
                        max_current_ma=500.0,
                        anchor_current=True,
                        trace_baseline_sdk=trace_baseline,
                        step_delta_limit=np.deg2rad(3.0),
                        live_lower=live_lower,
                        live_upper=live_upper,
                        allow_passive_limit_outlier=True,
                    )
                )
        self.assertEqual(stale_io.sent, [])

        runaway_io = FakeIO(runaway=True)
        runaway_io.profile = self.profile
        with patch("revo3_deploy.cli.replay_trace.asyncio.sleep", new=no_sleep):
            with self.assertRaisesRegex(RuntimeError, "measured excursion"):
                asyncio.run(
                    _execute_replay(
                        args=SimpleNamespace(print_every=1),
                        io=runaway_io,
                        profile=self.profile,
                        trace=trace,
                        rows=rows,
                        selected_sdk=selected_sdk,
                        sdk_targets=trace_endpoint.reshape(1, -1),
                        lower=self.profile.sdk_position_lower_rad,
                        upper=self.profile.sdk_position_upper_rad,
                        measured_tolerance=np.deg2rad(1.0),
                        initial_delta_limit=np.deg2rad(5.0),
                        max_speed_deg_s=2.0,
                        confirm_large_excursion=True,
                        kp=0.2,
                        kd=0.05,
                        max_current_ma=500.0,
                        anchor_current=True,
                        trace_baseline_sdk=trace_baseline,
                        step_delta_limit=np.deg2rad(3.0),
                        live_lower=live_lower,
                        live_upper=live_upper,
                        allow_passive_limit_outlier=True,
                    )
                )
        self.assertEqual(len(runaway_io.sent), 1)

    def test_replay_execute_uses_fresh_gate_and_post_send_health_read(self) -> None:
        policy_start = (
            self.profile.joint_lower_policy + self.profile.joint_upper_policy
        ) * 0.5
        sdk_endpoint = self.profile.target_policy_to_sdk(policy_start)
        selected_sdk = (13,)
        rows = np.asarray([0], dtype=np.int64)
        trace = SimpleNamespace(
            step_index=np.asarray([0], dtype=np.int64),
        )

        class FakeIO:
            def __init__(self, position: np.ndarray) -> None:
                self.position = position.copy()
                self.last_motor_currents_ma = np.zeros(21, dtype=np.float32)
                self.sent: list[np.ndarray] = []
                self.read_count = 0

            async def read_position_rad(self, check_errors: bool = False) -> np.ndarray:
                self.read_count += 1
                return self.position.copy()

            async def send_mit_command_rad(self, position_rad, **kwargs) -> None:
                command = np.asarray(position_rad, dtype=np.float32).copy()
                self.sent.append(command)
                self.position = command

            def validate_device_position(self, position_rad) -> None:
                self.profile.validate_sdk_position(
                    np.asarray(position_rad, dtype=np.float32),
                    "fake device command",
                )

        async def no_sleep(_: float) -> None:
            return None

        safe_start = sdk_endpoint.copy()
        safe_start[13] -= np.deg2rad(0.2)
        io = FakeIO(safe_start)
        io.profile = self.profile
        with patch("revo3_deploy.cli.replay_trace.asyncio.sleep", new=no_sleep):
            asyncio.run(
                _execute_replay(
                    args=SimpleNamespace(print_every=1),
                    io=io,
                    profile=self.profile,
                    trace=trace,
                    rows=rows,
                    selected_sdk=selected_sdk,
                    sdk_targets=sdk_endpoint.reshape(1, -1),
                    lower=self.profile.sdk_position_lower_rad,
                    upper=self.profile.sdk_position_upper_rad,
                    measured_tolerance=np.deg2rad(1.0),
                    initial_delta_limit=np.deg2rad(5.0),
                    max_speed_deg_s=2.0,
                    confirm_large_excursion=False,
                    kp=0.2,
                    kd=0.05,
                    max_current_ma=500.0,
                )
            )
        self.assertGreater(len(io.sent), 0)
        self.assertEqual(io.read_count, len(io.sent) + 1)
        sent_steps = np.diff(
            np.asarray([safe_start[13]] + [float(command[13]) for command in io.sent])
        )
        self.assertLessEqual(float(np.max(np.abs(sent_steps))), np.deg2rad(0.1) + 1e-7)
        np.testing.assert_allclose(io.sent[-1][13], sdk_endpoint[13], atol=1e-7)

        drifted_start = sdk_endpoint.copy()
        drifted_start[13] -= np.deg2rad(6.0)
        drifted_io = FakeIO(drifted_start)
        drifted_io.profile = self.profile
        with self.assertRaisesRegex(RuntimeError, "not continuous"):
            asyncio.run(
                _execute_replay(
                    args=SimpleNamespace(print_every=1),
                    io=drifted_io,
                    profile=self.profile,
                    trace=trace,
                    rows=rows,
                    selected_sdk=selected_sdk,
                    sdk_targets=sdk_endpoint.reshape(1, -1),
                    lower=self.profile.sdk_position_lower_rad,
                    upper=self.profile.sdk_position_upper_rad,
                    measured_tolerance=np.deg2rad(1.0),
                    initial_delta_limit=np.deg2rad(5.0),
                    max_speed_deg_s=2.0,
                    confirm_large_excursion=True,
                    kp=0.2,
                    kd=0.05,
                    max_current_ma=500.0,
                )
            )
        self.assertEqual(drifted_io.sent, [])

        selected_four = (1, 5, 9, 13)
        over_current_io = FakeIO(sdk_endpoint)
        over_current_io.profile = self.profile
        over_current_io.last_motor_currents_ma[list(selected_four)] = 300.0
        with self.assertRaisesRegex(RuntimeError, "total current exceeded"):
            asyncio.run(
                _execute_replay(
                    args=SimpleNamespace(print_every=1),
                    io=over_current_io,
                    profile=self.profile,
                    trace=trace,
                    rows=rows,
                    selected_sdk=selected_four,
                    sdk_targets=sdk_endpoint.reshape(1, -1),
                    lower=self.profile.sdk_position_lower_rad,
                    upper=self.profile.sdk_position_upper_rad,
                    measured_tolerance=np.deg2rad(1.0),
                    initial_delta_limit=np.deg2rad(5.0),
                    max_speed_deg_s=2.0,
                    confirm_large_excursion=True,
                    kp=0.2,
                    kd=0.05,
                    max_current_ma=500.0,
                )
            )
        self.assertEqual(over_current_io.sent, [])

    def test_input_builder_layout_and_history(self) -> None:
        builder = Stage2InputBuilder(
            self.contract,
            self.profile.joint_lower_policy,
            self.profile.joint_upper_policy,
        )
        position = (
            self.profile.joint_lower_policy + self.profile.joint_upper_policy
        ) * 0.5
        contacts = np.arange(1, 6, dtype=np.float32)
        inputs = builder.reset(position, contacts)
        self.assertEqual(inputs["obs"].shape, (1, 141))
        self.assertEqual(inputs["proprio_hist"].shape, (1, 30, 47))
        np.testing.assert_allclose(inputs["proprio_hist"][0, 0, :21], 0.0, atol=1e-6)
        np.testing.assert_allclose(inputs["proprio_hist"][0, 0, 21:42], position)
        np.testing.assert_allclose(inputs["proprio_hist"][0, 0, 42:47], contacts * 0.1)

        next_contacts = contacts + 10.0
        inputs = builder.observe(position, next_contacts)
        np.testing.assert_allclose(inputs["proprio_hist"][0, -2, 42:47], contacts * 0.1)
        np.testing.assert_allclose(inputs["proprio_hist"][0, -1, 42:47], next_contacts * 0.1)
        np.testing.assert_allclose(inputs["obs"][0, -5:], next_contacts * 0.1)

    def test_q_unscaled_is_not_clipped(self) -> None:
        builder = Stage2InputBuilder(
            self.contract,
            self.profile.joint_lower_policy,
            self.profile.joint_upper_policy,
        )
        position = self.profile.joint_upper_policy + 0.1
        inputs = builder.reset(position, np.zeros(5, dtype=np.float32))
        self.assertTrue(np.all(inputs["proprio_hist"][0, -1, :21] > 1.0))

    def test_delta_action_is_scaled_and_clamped(self) -> None:
        builder = Stage2InputBuilder(
            self.contract,
            self.profile.joint_lower_policy,
            self.profile.joint_upper_policy,
            self.profile.target_lower_policy,
            self.profile.target_upper_policy,
        )
        position = self.profile.target_upper_policy - 0.01
        builder.reset(position, np.zeros(5, dtype=np.float32))
        target = builder.action_to_target(np.ones(21, dtype=np.float32))
        np.testing.assert_allclose(target, self.profile.target_upper_policy, atol=1e-7)

        builder.reset(
            (self.profile.joint_lower_policy + self.profile.joint_upper_policy) * 0.5,
            np.zeros(5, dtype=np.float32),
        )
        before = builder.current_target.copy()
        target = builder.action_to_target(np.full(21, 0.5, dtype=np.float32))
        np.testing.assert_allclose(
            target - before,
            np.full(21, self.contract.action_scale * 0.5, dtype=np.float32),
            atol=1e-7,
        )

    def test_policy_step_exposes_exact_raw_onnx_inputs(self) -> None:
        runner = object.__new__(Revo3PolicyRunner)
        runner.contract = self.contract
        runner.profile = self.profile
        runner.builder = Stage2InputBuilder(
            self.contract,
            self.profile.joint_lower_policy,
            self.profile.joint_upper_policy,
            self.profile.target_lower_policy,
            self.profile.target_upper_policy,
        )
        expected_action = np.linspace(-2.0, 2.0, 21, dtype=np.float32)

        class FakeSession:
            @staticmethod
            def run(output_names, inputs):
                self.assertEqual(output_names, ["action"])
                self.assertEqual(inputs["obs"].shape, (1, 141))
                self.assertEqual(inputs["proprio_hist"].shape, (1, 30, 47))
                return [expected_action[None, :]]

        runner.session = FakeSession()
        runner.initialized = False
        position = self.profile.target_upper_policy - 0.01
        contacts = np.arange(1, 6, dtype=np.float32)
        result = runner.step(position, contacts)

        self.assertEqual(result.obs_raw.shape, (1, 141))
        self.assertEqual(result.proprio_hist_raw.shape, (1, 30, 47))
        self.assertEqual(result.obs_raw.dtype, np.float32)
        self.assertEqual(result.proprio_hist_raw.dtype, np.float32)
        np.testing.assert_array_equal(
            result.obs_raw[0],
            result.proprio_hist_raw[0, -3:].reshape(141),
        )
        np.testing.assert_array_equal(
            result.proprio_hist_raw[0, -1, 42:47],
            contacts * 0.1,
        )
        np.testing.assert_array_equal(result.onnx_action_raw, expected_action)
        np.testing.assert_array_equal(result.action, np.clip(expected_action, -1.0, 1.0))
        np.testing.assert_allclose(
            result.policy_target_unclipped_rad,
            position + self.contract.action_scale * result.action,
            atol=1e-7,
        )
        self.assertEqual(result.target_clipped.dtype, np.bool_)
        np.testing.assert_array_equal(
            result.target_clipped,
            (result.policy_target_unclipped_rad < runner.builder.target_lower)
            | (result.policy_target_unclipped_rad > runner.builder.target_upper),
        )
        self.assertTrue(np.any(result.target_clipped))

        saved_obs = result.obs_raw.copy()
        saved_history = result.proprio_hist_raw.copy()
        runner.step(position + 0.01, contacts + 10.0)
        np.testing.assert_array_equal(result.obs_raw, saved_obs)
        np.testing.assert_array_equal(result.proprio_hist_raw, saved_history)

    def test_policy_trace_shapes_atomic_save_and_error_metadata(self) -> None:
        position = (
            self.profile.joint_lower_policy + self.profile.joint_upper_policy
        ) * 0.5
        contacts = np.arange(1, 6, dtype=np.float32)
        builder = Stage2InputBuilder(
            self.contract,
            self.profile.joint_lower_policy,
            self.profile.joint_upper_policy,
            self.profile.target_lower_policy,
            self.profile.target_upper_policy,
        )
        inputs = builder.reset(position, contacts)
        action = np.zeros(21, dtype=np.float32)
        result = PolicyStep(
            onnx_action_raw=action.copy(),
            action=action,
            policy_target_unclipped_rad=position.copy(),
            policy_target_rad=position.copy(),
            target_clipped=np.zeros(21, dtype=np.bool_),
            obs_raw=inputs["obs"].copy(),
            proprio_hist_raw=inputs["proprio_hist"].copy(),
        )
        sdk_position = self.profile.target_policy_to_sdk(position)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.npz"
            recorder = PolicyTraceRecorder(
                path,
                {
                    "schema_name": "hora_policy_trace",
                    "schema_version": 1,
                    "source": "real",
                    "mode": "dry_run",
                    "contact_force_scale": self.contract.contact_force_scale,
                },
            )
            row = recorder.append_frame(
                step_index=0,
                loop_started_monotonic_s=recorder.started_monotonic_s + 0.001,
                sdk_pos_rad=sdk_position,
                policy_pos_rad=position,
                force_n=contacts,
                result=result,
                sdk_target_rad=sdk_position,
                read_ms=1.2,
                inference_ms=0.4,
                tactile_age_ms=2.5,
                motor_current_ma=np.arange(21, dtype=np.float32),
                stalled_motor_ids=(3,),
                stall_duration_s=np.linspace(0.0, 0.2, 21, dtype=np.float64),
                motor_status_valid=True,
            )
            recorder.finish_frame(row, loop_ms=2.0)
            error = RuntimeError("diagnostic stop")
            saved_path = recorder.save(termination_status="error", error=error)

            self.assertEqual(saved_path, path.resolve())
            self.assertTrue(path.is_file())
            self.assertEqual(list(Path(directory).glob("*.partial.npz")), [])
            with np.load(path, allow_pickle=False) as trace:
                expected_shapes = {
                    "step_index": (1,),
                    "sample_time_s": (1,),
                    "sdk_pos_rad": (1, 21),
                    "policy_pos_rad": (1, 21),
                    "joint_pos_unscaled": (1, 21),
                    "input_target_policy_rad": (1, 21),
                    "force_n": (1, 5),
                    "frame_raw": (1, 47),
                    "obs_raw": (1, 141),
                    "proprio_hist_raw": (1, 30, 47),
                    "onnx_action_raw": (1, 21),
                    "action": (1, 21),
                    "policy_target_unclipped_rad": (1, 21),
                    "policy_target_rad": (1, 21),
                    "target_clipped": (1, 21),
                    "sdk_target_rad": (1, 21),
                    "motor_current_ma": (1, 21),
                    "stall_mask": (1, 21),
                    "stall_duration_s": (1, 21),
                    "command_sent": (1,),
                    "command_completed": (1,),
                }
                for name, shape in expected_shapes.items():
                    self.assertEqual(trace[name].shape, shape, name)
                self.assertEqual(float(trace["sample_time_s"][0]), 0.0)
                np.testing.assert_array_equal(trace["obs_raw"][0], inputs["obs"][0])
                np.testing.assert_array_equal(
                    trace["proprio_hist_raw"][0],
                    inputs["proprio_hist"][0],
                )
                self.assertFalse(bool(trace["command_sent"][0]))
                self.assertFalse(bool(trace["command_completed"][0]))
                self.assertTrue(bool(trace["motor_status_valid"][0]))
                np.testing.assert_array_equal(
                    trace["motor_current_ma"][0],
                    np.arange(21, dtype=np.float32),
                )
                self.assertTrue(bool(trace["stall_mask"][0, 3]))
                metadata = json.loads(str(trace["metadata_json"].item()))
                self.assertEqual(metadata["schema_name"], "hora_policy_trace")
                self.assertEqual(metadata["termination"]["status"], "error")
                self.assertEqual(metadata["termination"]["error_type"], "RuntimeError")
                self.assertEqual(
                    metadata["termination"]["error_message"],
                    "diagnostic stop",
                )
                self.assertEqual(metadata["frame_count"], 1)
                self.assertEqual(metadata["command_sent_frame_count"], 0)

            with self.assertRaisesRegex(FileExistsError, "Refusing to overwrite"):
                PolicyTraceRecorder(path, {})

    def test_live_device_limits_tighten_policy_target_before_reset(self) -> None:
        runner = object.__new__(Revo3PolicyRunner)
        runner.contract = self.contract
        runner.profile = self.profile
        runner.builder = Stage2InputBuilder(
            self.contract,
            self.profile.joint_lower_policy,
            self.profile.joint_upper_policy,
            self.profile.target_lower_policy,
            self.profile.target_upper_policy,
        )
        lower_sdk = self.profile.sdk_position_lower_rad.copy()
        upper_sdk = self.profile.sdk_position_upper_rad.copy()
        upper_sdk[0] = np.deg2rad(10.0)
        runner.apply_device_target_limits(
            lower_sdk,
            upper_sdk,
            margin_rad=np.deg2rad(0.05),
        )
        policy_index = int(self.profile.policy_to_sdk_perm[0])
        self.assertLessEqual(
            float(runner.builder.target_upper[policy_index]),
            float(np.deg2rad(9.95)) + 1e-6,
        )

    def test_tactile_pressure_and_matrix_units(self) -> None:
        adapter = FingertipForceAdapter.from_profile(self.profile.tactile)
        summary = np.zeros(42, dtype=np.float32)
        for start, end in adapter.pressure_tip_slices:
            summary[start:end] = 1000.0
        np.testing.assert_allclose(adapter.from_pressure_summary(summary), [3.0] * 5)

        modules = {module_id: [1000.0, 2000.0] for module_id in adapter.matrix_tip_module_ids}
        np.testing.assert_allclose(adapter.from_matrix_modules(modules), [0.3] * 5)
        np.testing.assert_allclose(adapter.from_force_vector([1, 2, 3, 4, 5]), [1, 2, 3, 4, 5])
        with self.assertRaisesRegex(ValueError, "non-negative"):
            adapter.from_force_vector([1, 2, -3, 4, 5])

    def test_vision_touch_config_force_vector_and_freshness(self) -> None:
        config = dict(self.profile.tactile["vision_touch"])
        collector = VisionTouchCollector(config)
        self.assertEqual(len(collector.sensor_serials), 5)
        self.assertTrue(collector.mapping_verified)
        self.assertEqual(
            collector.sensor_serials,
            (
                "WTUVL2198X260001A",
                "WTUVL3197X260010B",
                "WTUVL3194X26000EB",
                "WTUVL3195X26000EE",
                "WTUVL3197X2600106",
            ),
        )

        token = object()

        class FakeSensor:
            @staticmethod
            def collect_sensor_data(data_type):
                return {data_type: np.asarray([3.0, 4.0, 0.0, 1.0, 2.0, 3.0])}

        force6d = collector._read_sensor_force(FakeSensor(), token)
        self.assertAlmostEqual(float(np.linalg.norm(force6d[:3])), 5.0)

        collector._latest_forces = np.arange(5, dtype=np.float32)
        collector._latest_timestamp = time.monotonic()
        values, age = collector.read_latest()
        np.testing.assert_allclose(values, np.arange(5, dtype=np.float32))
        self.assertLess(age, collector.max_sample_age_s)

        collector._latest_timestamp = time.monotonic() - collector.max_sample_age_s * 2.0
        with self.assertRaisesRegex(RuntimeError, "stale"):
            collector.read_latest()
        stale_values, stale_age = collector.read_latest(enforce_freshness=False)
        np.testing.assert_allclose(stale_values, np.arange(5, dtype=np.float32))
        self.assertGreater(stale_age, collector.max_sample_age_s)

        duplicate = dict(config)
        duplicate["sensor_order"] = ["same"] * 5
        with self.assertRaisesRegex(ValueError, "unique"):
            VisionTouchCollector(duplicate)

    def test_preflight_rejects_any_motor_or_tactile_write_flag(self) -> None:
        args = build_parser().parse_args(
            [
                "--onnx", "unused.onnx",
                "--metadata", "unused.yaml",
                "--profile", "unused.yaml",
                "--preflight-only",
                "--enable-motion",
            ]
        )
        with self.assertRaisesRegex(ValueError, "cannot be combined"):
            asyncio.run(async_main(args))

    def test_trace_requires_finite_steps_outside_preflight(self) -> None:
        args = build_parser().parse_args(
            [
                "--onnx", "unused.onnx",
                "--metadata", "unused.yaml",
                "--profile", "unused.yaml",
                "--trace-npz", "unused.npz",
            ]
        )
        with self.assertRaisesRegex(ValueError, "requires a finite --steps"):
            asyncio.run(async_main(args))

    def test_policy_start_delay_parser_and_bounds(self) -> None:
        base = [
            "--onnx", "unused.onnx",
            "--metadata", "unused.yaml",
            "--profile", "unused.yaml",
        ]
        self.assertEqual(build_parser().parse_args(base).policy_start_delay_s, 0.0)
        self.assertEqual(
            build_parser().parse_args(
                base + ["--policy-start-delay-s", "12.5"]
            ).policy_start_delay_s,
            12.5,
        )
        for invalid in ("-0.01", "120.01", "nan", "inf"):
            args = build_parser().parse_args(
                base + ["--policy-start-delay-s", invalid]
            )
            with self.subTest(invalid=invalid), self.assertRaisesRegex(
                ValueError,
                "finite and in",
            ):
                asyncio.run(async_main(args))

    def test_policy_start_delay_is_dry_run_only(self) -> None:
        base = [
            "--onnx", "unused.onnx",
            "--metadata", "unused.yaml",
            "--profile", "unused.yaml",
            "--policy-start-delay-s", "1",
        ]
        for forbidden_mode in ("--enable-motion", "--preflight-only"):
            args = build_parser().parse_args(base + [forbidden_mode])
            with self.subTest(mode=forbidden_mode), self.assertRaisesRegex(
                ValueError,
                "only allowed in motor dry-run mode",
            ):
                asyncio.run(async_main(args))

    def test_preposition_requires_explicit_motion_confirmation(self) -> None:
        args = build_parser().parse_args(
            [
                "--onnx", "unused.onnx",
                "--metadata", "unused.yaml",
                "--profile", "unused.yaml",
                "--preposition-cache", "cache.npy",
            ]
        )
        with self.assertRaisesRegex(ValueError, "requires --enable-motion"):
            asyncio.run(async_main(args))

    def test_preposition_cache_mapping_and_speed(self) -> None:
        cache = REPO_ROOT / "cache/revo3_right_grasp_cylinder.npy"
        if not cache.exists():
            self.skipTest("training cache is not available")
        target = _load_preposition_sdk_target(str(cache), 7942, self.profile)
        offset_deg = np.rad2deg(self.profile.sdk_offset_rad)
        self.assertAlmostEqual(
            float(np.rad2deg(target[0])),
            -8.075 + float(offset_deg[0]),
            places=3,
        )
        self.assertAlmostEqual(
            float(np.rad2deg(target[20])),
            64.429 + float(offset_deg[20]),
            places=3,
        )

        start = target - np.deg2rad(np.linspace(0.0, 10.0, 21))
        frames = _build_preposition_targets(
            start,
            target,
            speed_deg_s=2.0,
            rate_hz=20.0,
        )
        np.testing.assert_allclose(frames[-1], target, atol=1e-7)
        first_delta = frames[0] - start
        later_delta = np.diff(frames, axis=0)
        self.assertLessEqual(float(np.max(np.abs(first_delta))), np.deg2rad(0.1) + 1e-7)
        self.assertLessEqual(float(np.max(np.abs(later_delta))), np.deg2rad(0.1) + 1e-7)

    def test_preflight_position_tolerance_is_read_only_and_bounded(self) -> None:
        args = build_parser().parse_args(
            [
                "--onnx", "unused.onnx",
                "--metadata", "unused.yaml",
                "--profile", "unused.yaml",
                "--preflight-position-tolerance-deg", "3",
            ]
        )
        with self.assertRaisesRegex(ValueError, "only valid with"):
            asyncio.run(async_main(args))

        args = build_parser().parse_args(
            [
                "--onnx", "unused.onnx",
                "--metadata", "unused.yaml",
                "--profile", "unused.yaml",
                "--preflight-only",
                "--preflight-position-tolerance-deg", "5.1",
            ]
        )
        with self.assertRaisesRegex(ValueError, "finite and in"):
            asyncio.run(async_main(args))

    def test_policy_stall_override_requires_motion(self) -> None:
        args = build_parser().parse_args(
            [
                "--onnx", str(ONNX),
                "--metadata", str(ACTUAL_META),
                "--profile", str(PROFILE),
                "--allow-stall", "M13",
            ]
        )
        with self.assertRaisesRegex(ValueError, "only valid together"):
            asyncio.run(async_main(args))

        args = build_parser().parse_args(
            [
                "--onnx", str(ONNX),
                "--metadata", str(ACTUAL_META),
                "--profile", str(PROFILE),
                "--ignore-all-stall",
            ]
        )
        with self.assertRaisesRegex(ValueError, "only valid together"):
            asyncio.run(async_main(args))

    def test_policy_rejects_stall_grace_above_one_second(self) -> None:
        args = build_parser().parse_args(
            [
                "--onnx", str(ONNX),
                "--metadata", str(ACTUAL_META),
                "--profile", str(PROFILE),
                "--enable-motion",
                "--stall-grace-s", "1.01",
            ]
        )
        with self.assertRaisesRegex(ValueError, "finite and in"):
            asyncio.run(async_main(args))

    def test_calibration_jog_plan_is_bounded_and_returns_to_zero(self) -> None:
        offsets = _build_jog_offsets(np.deg2rad(1.0), 1.0, 0.25, 20.0)
        self.assertEqual(offsets.shape, (45,))
        self.assertAlmostEqual(float(np.rad2deg(offsets[0])), 0.05, places=5)
        self.assertAlmostEqual(float(np.rad2deg(np.max(offsets))), 1.0, places=5)
        self.assertAlmostEqual(float(offsets[-1]), 0.0, places=7)
        self.assertLessEqual(
            float(np.max(np.abs(np.diff(offsets)))),
            float(np.deg2rad(0.05)) + 1e-7,
        )
        large_offsets = _build_jog_offsets(np.deg2rad(10.0), 5.0, 0.25, 20.0)
        self.assertEqual(large_offsets.shape, (205,))
        self.assertAlmostEqual(float(np.rad2deg(np.max(large_offsets))), 10.0, places=4)
        self.assertLessEqual(
            float(np.rad2deg(np.max(np.abs(np.diff(large_offsets))))),
            0.1 + 1e-5,
        )

    def test_multi_joint_jog_requires_extra_confirmation(self) -> None:
        args = jog_build_parser().parse_args(
            [
                "--profile", str(PROFILE),
                "--joint", "M1",
                "--joint", "M5",
                "--delta-deg", "1",
                "--execute",
                "--confirm-fixed",
                "--confirm-empty",
                "--confirm-estop",
                "--confirm-release",
            ]
        )
        with self.assertRaisesRegex(RuntimeError, "synchronized movement"):
            asyncio.run(jog_async_main(args))

    def test_high_gain_jog_requires_extra_confirmation(self) -> None:
        args = jog_build_parser().parse_args(
            [
                "--profile", str(PROFILE),
                "--joint", "M13",
                "--delta-deg", "1",
                "--kp", "2.0",
                "--kd", "0.25",
                "--execute",
                "--confirm-fixed",
                "--confirm-empty",
                "--confirm-estop",
                "--confirm-release",
            ]
        )
        with self.assertRaisesRegex(RuntimeError, "confirm-high-gain"):
            asyncio.run(jog_async_main(args))

    def test_jog_rejects_excessive_measured_limit_tolerance(self) -> None:
        args = jog_build_parser().parse_args(
            [
                "--profile", str(PROFILE),
                "--joint", "M13",
                "--delta-deg", "1",
                "--measured-limit-tolerance-deg", "1.1",
                "--execute",
                "--confirm-fixed",
                "--confirm-empty",
                "--confirm-estop",
                "--confirm-release",
            ]
        )
        with self.assertRaisesRegex(ValueError, "must be finite and in"):
            asyncio.run(jog_async_main(args))

    @unittest.skipUnless(
        ONNX.is_file() and ACTUAL_META.is_file(),
        "exported ONNX artifact is not present",
    )
    def test_real_onnx_runner(self) -> None:
        runner = Revo3PolicyRunner.create(ONNX, ACTUAL_META, PROFILE)
        position = (
            runner.profile.joint_lower_policy + runner.profile.joint_upper_policy
        ) * 0.5
        result = runner.step(position, np.zeros(5, dtype=np.float32))
        self.assertEqual(result.onnx_action_raw.shape, (21,))
        self.assertEqual(result.action.shape, (21,))
        self.assertEqual(result.policy_target_unclipped_rad.shape, (21,))
        self.assertEqual(result.target_clipped.shape, (21,))
        self.assertEqual(result.obs_raw.shape, (1, 141))
        self.assertEqual(result.proprio_hist_raw.shape, (1, 30, 47))
        self.assertTrue(np.isfinite(result.action).all())
        self.assertTrue(np.all(np.abs(result.action) <= 1.0))


class SdkStatusTest(unittest.IsolatedAsyncioTestCase):
    async def test_sdk_motion_lock_conflicts_for_overlapping_serial_allowlists(self) -> None:
        first = object.__new__(Revo3SdkHandIO)
        first.config = Revo3SdkConfig(serial_allowlist=("TEST-LOCK-A", "TEST-LOCK-B"))
        first._motion_lock_handles = []
        second = object.__new__(Revo3SdkHandIO)
        second.config = Revo3SdkConfig(serial_allowlist=("test-lock-a",))
        second._motion_lock_handles = []

        first._acquire_motion_locks()
        try:
            with self.assertRaisesRegex(RuntimeError, "holds the SDK motion lock"):
                second._acquire_motion_locks()
        finally:
            first._release_motion_locks()

        second._acquire_motion_locks()
        second._release_motion_locks()

    async def test_running_bit_is_not_treated_as_fault(self) -> None:
        class FakeContext:
            async def revo3_get_motor_online_status(self, slave_id):
                return (1 << 21) - 1

            async def revo3_get_motor_status_data(self, slave_id):
                self.slave_id = slave_id
                return SimpleNamespace(
                    statuses=[1 << 11] * 21,
                    errors=[0] * 21,
                    currents=[0.1] * 21,
                    positions=[0.0] * 21,
                )

        io = object.__new__(Revo3SdkHandIO)
        io.ctx = FakeContext()
        io.slave_id = 127
        io.config = Revo3SdkConfig(max_abs_current_ma=500.0)
        io.last_motor_currents_ma = None
        position = await io.read_position_rad(check_errors=True)
        np.testing.assert_allclose(position, np.zeros(21, dtype=np.float32))

    async def test_real_fault_is_rejected(self) -> None:
        class FakeContext:
            async def revo3_get_motor_online_status(self, slave_id):
                return (1 << 21) - 1

            async def revo3_get_motor_status_data(self, slave_id):
                return SimpleNamespace(
                    statuses=[0] * 21,
                    errors=[1] + [0] * 20,
                    currents=[0.1] * 21,
                    positions=[0.0] * 21,
                )

        io = object.__new__(Revo3SdkHandIO)
        io.ctx = FakeContext()
        io.slave_id = 127
        io.config = Revo3SdkConfig(max_abs_current_ma=500.0)
        io.last_motor_currents_ma = None
        with self.assertRaisesRegex(RuntimeError, "Motor fault"):
            await io.read_position_rad(check_errors=True)

    async def test_stall_status_is_rejected(self) -> None:
        class FakeContext:
            async def revo3_get_motor_online_status(self, slave_id):
                return (1 << 21) - 1

            async def revo3_get_motor_status_data(self, slave_id):
                return SimpleNamespace(
                    statuses=[1 << 8] + [1 << 11] * 20,
                    errors=[0] * 21,
                    currents=[0.1] * 21,
                    positions=[0.0] * 21,
                )

        io = object.__new__(Revo3SdkHandIO)
        io.ctx = FakeContext()
        io.slave_id = 127
        io.config = Revo3SdkConfig(max_abs_current_ma=500.0)
        io.last_motor_currents_ma = None
        with self.assertRaisesRegex(RuntimeError, "status=0x100"):
            await io.read_position_rad(check_errors=True)

    async def test_explicit_selected_stall_is_allowed_but_other_faults_are_not(self) -> None:
        class FakeContext:
            async def revo3_get_motor_online_status(self, slave_id):
                return (1 << 21) - 1

            async def revo3_get_motor_status_data(self, slave_id):
                return SimpleNamespace(
                    statuses=[1 << 8] + [1 << 11] * 20,
                    errors=[1 << 8] + [0] * 20,
                    currents=[25.0] + [0.0] * 20,
                    positions=[1.0] + [0.0] * 20,
                )

        io = object.__new__(Revo3SdkHandIO)
        io.ctx = FakeContext()
        io.slave_id = 127
        io.config = Revo3SdkConfig(
            max_abs_current_ma=500.0,
            allowed_stall_motor_ids=(0,),
        )
        io.last_motor_currents_ma = None
        measured = await io.read_position_rad(check_errors=True)
        self.assertEqual(io.last_stalled_motor_ids, (0,))
        self.assertAlmostEqual(float(np.rad2deg(measured[0])), 1.0, places=5)

        io.config = Revo3SdkConfig(
            max_abs_current_ma=500.0,
            allowed_stall_motor_ids=(1,),
        )
        with self.assertRaisesRegex(RuntimeError, "M0.*status=0x100"):
            await io.read_position_rad(check_errors=True)

    async def test_stall_grace_requires_continuous_one_second(self) -> None:
        class FakeContext:
            async def revo3_get_motor_online_status(self, slave_id):
                return (1 << 21) - 1

            async def revo3_get_motor_status_data(self, slave_id):
                return SimpleNamespace(
                    statuses=[1 << 8] + [0] * 20,
                    errors=[1 << 8] + [0] * 20,
                    currents=[25.0] + [0.0] * 20,
                    positions=[0.0] * 21,
                )

        io = object.__new__(Revo3SdkHandIO)
        io.ctx = FakeContext()
        io.slave_id = 127
        io.config = Revo3SdkConfig(
            max_abs_current_ma=500.0,
            allowed_stall_motor_ids=tuple(range(21)),
            stall_grace_s=1.0,
        )
        io.last_motor_currents_ma = None
        with patch(
            "revo3_deploy.sdk_hand_io.time.monotonic",
            side_effect=[10.0, 11.0, 11.01],
        ):
            await io.read_position_rad(check_errors=True)
            await io.read_position_rad(check_errors=True)
            with self.assertRaisesRegex(RuntimeError, "status=0x100"):
                await io.read_position_rad(check_errors=True)

    async def test_over_current_is_rejected(self) -> None:
        class FakeContext:
            async def revo3_get_motor_online_status(self, slave_id):
                return (1 << 21) - 1

            async def revo3_get_motor_status_data(self, slave_id):
                return SimpleNamespace(
                    statuses=[0] * 21,
                    errors=[0] * 21,
                    currents=[501.0] + [0.0] * 20,
                    positions=[0.0] * 21,
                )

        io = object.__new__(Revo3SdkHandIO)
        io.ctx = FakeContext()
        io.slave_id = 127
        io.config = Revo3SdkConfig(max_abs_current_ma=500.0)
        io.last_motor_currents_ma = None
        with self.assertRaisesRegex(RuntimeError, "current exceeds"):
            await io.read_position_rad(check_errors=True)

    async def test_offline_motor_is_rejected(self) -> None:
        class FakeContext:
            async def revo3_get_motor_online_status(self, slave_id):
                return ((1 << 21) - 1) & ~(1 << 7)

            async def revo3_get_motor_status_data(self, slave_id):
                return SimpleNamespace(
                    statuses=[0] * 21,
                    errors=[0] * 21,
                    currents=[0.0] * 21,
                    positions=[0.0] * 21,
                )

        io = object.__new__(Revo3SdkHandIO)
        io.ctx = FakeContext()
        io.slave_id = 127
        io.config = Revo3SdkConfig(max_abs_current_ma=500.0)
        io.last_motor_currents_ma = None
        with self.assertRaisesRegex(RuntimeError, r"motor IDs \[7\]"):
            await io.read_position_rad(check_errors=True)

    def test_device_identity_and_serial_are_enforced(self) -> None:
        fake_sdk = SimpleNamespace(
            HandType=SimpleNamespace(Left=0, Right=1),
            StarkHardwareType=SimpleNamespace(
                Revo3UltraTouch=21,
                Revo3UltraVisionTouch=22,
            ),
        )
        io = object.__new__(Revo3SdkHandIO)
        io.sdk = fake_sdk
        io.config = Revo3SdkConfig(
            expected_hand="right",
            allowed_hardware_types=("Revo3UltraTouch",),
            serial_allowlist=("RIGHT-001",),
        )
        io.device_info = SimpleNamespace(
            hand_type=1,
            hardware_type=21,
            serial_number="RIGHT-001",
        )
        io._validate_device_identity()

        io.device_info.hand_type = 0
        with self.assertRaisesRegex(RuntimeError, "does not match"):
            io._validate_device_identity()

        io.device_info.hand_type = 1
        io.device_info.serial_number = ""
        with self.assertRaisesRegex(RuntimeError, "serial is empty"):
            io._validate_device_identity()

    def test_device_reported_position_limits_are_enforced(self) -> None:
        io = object.__new__(Revo3SdkHandIO)
        io.device_position_lower_rad = None
        io.device_position_upper_rad = None
        io._set_device_position_limits(([0.0] * 21, [90.0] * 21))
        io.validate_device_position(np.zeros(21, dtype=np.float32))
        invalid = np.zeros(21, dtype=np.float32)
        invalid[20] = np.deg2rad(91.0)
        with self.assertRaisesRegex(ValueError, "device-reported"):
            io.validate_device_position(invalid)

    async def test_release_uses_zero_force_without_retry(self) -> None:
        class FakeContext:
            calls = []

            async def revo3_set_all_mit_params_without_retry(self, *args):
                self.calls.append(args)

        io = object.__new__(Revo3SdkHandIO)
        io.ctx = FakeContext()
        io.slave_id = 127
        await io.release_mit()
        self.assertEqual(len(io.ctx.calls), 1)
        call = io.ctx.calls[0]
        self.assertEqual(call[0], 127)
        for values in call[1:]:
            self.assertEqual(values, [0.0] * 21)


if __name__ == "__main__":
    unittest.main()
