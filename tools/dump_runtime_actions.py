#!/usr/bin/env python3
"""Dump runtime action sequences for real-robot replay.

Runs Stage1 (PPO) or Stage2 (ProprioAdapt) policy for N episodes, records per-frame:
  raw_action: policy clamped delta output mu (before delta integration, in [-1,1])
  targets:    cur_targets — delta-accumulated + joint-limit clamped position target
  jointpos:   measured joint angles from hand.data.joint_pos (absolute, radians)

Output: one .txt per episode per type, with headers.

Gotcha — torque control does NOT update joint_pos_target during steps.
  cur_targets is the authoritative position target used in the PD formula
  torques = p_gain*(cur_targets - pos) - d_gain*vel.
"""

from __future__ import annotations

import argparse
import copy
import datetime
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("HORA_SKIP_SIM_CLOSE", "1")

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="cylinder", choices=["ball", "cylinder"])
parser.add_argument("--algo", type=str, default="auto", choices=["auto", "PPO", "ProprioAdapt"])
parser.add_argument("--train_cfg", type=str, default="Revo3HandHora")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--cache_file", type=str, default="", help="Override grasp cache filename under cache/.")
parser.add_argument(
    "--cache-row",
    type=int,
    default=None,
    metavar="N",
    help=(
        "Use exactly grasp-cache row N for env 0. The default remains seeded random "
        "sampling. Useful for matching a real preposition row."
    ),
)
parser.add_argument("--usd", type=str, default="", help="Override hand USD path.")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--episodes", type=int, default=5, help="Number of full episodes to dump.")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--output_dir", type=str, default="outputs/revo3_right/action_dump")
parser.add_argument(
    "--onnx",
    type=str,
    default="",
    help="Optional Stage-2 ONNX; compare every simulator raw obs/history action against it.",
)
parser.add_argument(
    "--max_frames",
    type=int,
    default=0,
    help="Optional early stop per episode for short parity checks (0 runs a full episode).",
)
parser.add_argument(
    "--episode-length-s",
    type=float,
    default=None,
    metavar="SECONDS",
    help=(
        "Override the simulator episode time limit for this export only. Set this "
        "longer than max_frames / policy_rate to avoid an auto-reset on the final row."
    ),
)
parser.add_argument(
    "--trace-npz",
    type=str,
    default="",
    metavar="PATH",
    help=(
        "Optional unified per-frame NPZ trace for sim/real comparison. "
        "Requires --episodes 1; '.npz' is appended when omitted."
    ),
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

from omegaconf import OmegaConf

from hora.algo.padapt.padapt import ProprioAdapt
from hora.algo.ppo.ppo import PPO
from hora.tasks.isaaclab import HoraCompatWrapper, Revo3HandHoraEnv, Revo3HandHoraEnvCfg
from hora.tasks.isaaclab.assets import (
    BALL_OBJECT_CFG, CYLINDER_OBJECT_CFG,
    REVO3_HAND_BALL_CFG, REVO3_HAND_CYLINDER_CFG,
)
from hora.utils.misc import set_np_formatting, set_seed

_TASK_ROBOT_CFG = {"ball": REVO3_HAND_BALL_CFG, "cylinder": REVO3_HAND_CYLINDER_CFG}
_TASK_OBJECT_CFG = {"ball": BALL_OBJECT_CFG, "cylinder": CYLINDER_OBJECT_CFG}
_TASK_CACHE = {
    "ball": "cache/revo3_right_grasp_ball",
    "cylinder": "cache/revo3_right_grasp_cylinder",
}


def _is_stage2_checkpoint(path: str) -> bool:
    return path.endswith(".ckpt") or ("stage2_nn" in path)


def _resolve_algo() -> str:
    if args.algo != "auto":
        return args.algo
    return "ProprioAdapt" if _is_stage2_checkpoint(args.checkpoint) else "PPO"


def _build_full_config(seed: int, algo: str):
    cfg_path = os.path.join(os.path.dirname(__file__), "..", "configs", "train", f"{args.train_cfg}.yaml")
    cfg_path = os.path.abspath(cfg_path)
    train_cfg = OmegaConf.load(cfg_path)
    train_cfg.algo = algo
    train_cfg.load_path = os.path.abspath(args.checkpoint)
    train_cfg.ppo.num_actors = args.num_envs
    train_cfg.ppo.priv_info = True
    train_cfg.ppo.proprio_adapt = algo == "ProprioAdapt"

    rl_device = getattr(args, "device", None) or "cuda:0"
    return OmegaConf.create(
        {
            "rl_device": rl_device,
            "test": True,
            "seed": seed,
            "train": train_cfg,
        }
    )


def _build_env_cfg(seed: int):
    env_cfg = Revo3HandHoraEnvCfg()
    if args.episode_length_s is not None:
        env_cfg.episode_length_s = float(args.episode_length_s)
    env_cfg.robot_cfg = _TASK_ROBOT_CFG.get(args.task, REVO3_HAND_CYLINDER_CFG)
    env_cfg.object_cfg = _TASK_OBJECT_CFG.get(args.task, CYLINDER_OBJECT_CFG)
    env_cfg.grasp_cache_path = _TASK_CACHE.get(args.task, 'cache/revo3_right_grasp_cylinder')
    if args.cache_file:
        env_cfg.grasp_cache_path = f"cache/{args.cache_file.replace('.npy', '')}"
    if args.usd:
        usd_path = os.path.abspath(args.usd)
        if not os.path.exists(usd_path):
            raise FileNotFoundError(f"--usd path not found: {usd_path}")
        env_cfg.robot_cfg = copy.deepcopy(env_cfg.robot_cfg)
        if env_cfg.robot_cfg.spawn is None or not hasattr(env_cfg.robot_cfg.spawn, "usd_path"):
            raise RuntimeError("env_cfg.robot_cfg.spawn has no usd_path to override.")
        env_cfg.robot_cfg.spawn.usd_path = usd_path

    env_cfg.gravity_curriculum = False
    env_cfg.sim.gravity = (0.0, 0.0, -9.81)  # full gravity for test/play
    env_cfg.scene.num_envs = args.num_envs
    if hasattr(env_cfg, "seed"):
        env_cfg.seed = seed
    if hasattr(env_cfg.sim, "device") and getattr(args, "device", None):
        env_cfg.sim.device = args.device
    return env_cfg


def _fmt_vec(values: np.ndarray) -> str:
    return "[" + ", ".join(f"{float(v):+.6f}" for v in values.tolist()) + "]"


TRACE_SCHEMA_NAME = "hora_policy_trace"
TRACE_SCHEMA_VERSION = 1
CONTACT_ORDER = ["thumb_DIP", "index_DIP", "middle_DIP", "ring_DIP", "little_DIP"]


def _np_copy(value: Any, dtype: np.dtype | type | None = None) -> np.ndarray:
    """Detach a tensor/array from mutable simulator buffers."""
    if isinstance(value, torch.Tensor):
        array = value.detach().cpu().numpy()
    else:
        array = np.asarray(value)
    if dtype is not None:
        array = array.astype(dtype, copy=False)
    return np.array(array, copy=True)


def _append_trace(trace: dict[str, list[np.ndarray]], name: str, value: Any) -> None:
    trace.setdefault(name, []).append(_np_copy(value))


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _trace_output_path(value: str) -> Path:
    path = Path(value).expanduser()
    if path.suffix.lower() != ".npz":
        path = Path(f"{path}.npz")
    path = path.resolve()
    if path.exists():
        if not path.is_file():
            raise ValueError(f"Trace path is not a regular file: {path}")
        raise FileExistsError(f"Refusing to overwrite an existing policy trace: {path}")
    return path


def _package_version(*names: str) -> str | None:
    for name in names:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return None


def _nearest_cache_row(
    cache: torch.Tensor | None,
    joint_pos: torch.Tensor,
    n_dof: int,
) -> tuple[int | None, float | None]:
    if cache is None or cache.ndim != 2 or cache.shape[0] == 0 or cache.shape[1] < n_dof:
        return None, None
    with torch.inference_mode():
        errors = torch.amax(torch.abs(cache[:, :n_dof] - joint_pos.reshape(1, n_dof)), dim=1)
        row = int(torch.argmin(errors).detach().cpu().item())
        max_error = float(errors[row].detach().cpu().item())
    return row, max_error


def _scalar_info(info: dict[str, Any]) -> dict[str, float]:
    scalars: dict[str, float] = {}
    for name, value in info.items():
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                continue
            scalars[name] = float(value.detach().cpu().item())
        elif isinstance(value, (int, float, np.integer, np.floating)):
            scalars[name] = float(value)
    return scalars


def _safe_extra_name(name: str) -> str:
    return "extra__" + "".join(ch if ch.isalnum() else "_" for ch in name).strip("_")


def _contact_force_vectors_w(raw_env: Revo3HandHoraEnv) -> np.ndarray:
    """Return the five object-filtered world-frame contact force vectors."""
    try:
        forces = torch.stack(
            [sensor.data.force_matrix_w[0, 0, 0, :] for sensor in raw_env._contact_sensor],
            dim=0,
        )
        return _np_copy(torch.nan_to_num(forces), np.float32)
    except (AttributeError, IndexError, RuntimeError):
        return np.full((len(CONTACT_ORDER), 3), np.nan, dtype=np.float32)


def _network_diagnostics(
    agent: PPO | ProprioAdapt,
    algo: str,
    input_dict: dict[str, torch.Tensor],
    priv_info: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Recompute inspectable network intermediates without changing model state."""
    model = agent.model
    with torch.inference_mode():
        teacher_latent_pre = model.env_mlp(priv_info)
        teacher_latent = torch.tanh(teacher_latent_pre)
        if algo == "ProprioAdapt":
            student_latent_pre = model.adapt_tconv(input_dict["proprio_hist"])
            student_latent = torch.tanh(student_latent_pre)
            policy_latent = student_latent
        else:
            student_latent_pre = torch.full_like(teacher_latent_pre, float("nan"))
            student_latent = torch.full_like(teacher_latent, float("nan"))
            policy_latent = teacher_latent

        actor_input = torch.cat([input_dict["obs"], policy_latent], dim=-1)
        actor_features = model.actor_mlp(actor_input)
        diagnostic_mu = model.mu(actor_features)
        value = model.value(actor_features)

        teacher_actor_input = torch.cat([input_dict["obs"], teacher_latent], dim=-1)
        teacher_features = model.actor_mlp(teacher_actor_input)
        teacher_action_mu = model.mu(teacher_features)

    return {
        "student_latent_pre_tanh": student_latent_pre,
        "student_latent": student_latent,
        "teacher_latent_pre_tanh": teacher_latent_pre,
        "teacher_latent": teacher_latent,
        "actor_input": actor_input,
        "actor_features": actor_features,
        "diagnostic_action_mu": diagnostic_mu,
        "teacher_action_mu": teacher_action_mu,
        "value": value,
    }


def _object_state(raw_env: Revo3HandHoraEnv) -> dict[str, np.ndarray]:
    return {
        "pose": np.concatenate(
            (
                _np_copy(raw_env.object_pos[0], np.float32),
                _np_copy(raw_env.object_rot[0], np.float32),
            )
        ),
        "linvel": _np_copy(raw_env.object_linvel[0], np.float32),
        "angvel": _np_copy(raw_env.object_angvel[0], np.float32),
    }


def _trace_metadata(
    *,
    algo: str,
    env_cfg: Revo3HandHoraEnvCfg,
    raw_env: Revo3HandHoraEnv,
    joint_names: list[str],
    dt_policy: float,
    ort_session: Any | None,
) -> dict[str, Any]:
    checkpoint = os.path.abspath(args.checkpoint)
    object_spawn = env_cfg.object_cfg.spawn
    geometry: dict[str, float] = {}
    for name in ("radius", "height"):
        value = getattr(object_spawn, name, None)
        if value is not None:
            geometry[f"{name}_m"] = float(value)

    software: dict[str, Any] = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "isaaclab": _package_version("isaaclab"),
        "isaacsim": _package_version("isaacsim"),
        "onnxruntime": _package_version("onnxruntime") if ort_session is not None else None,
        "onnxruntime_providers": ort_session.get_providers() if ort_session is not None else [],
    }

    return {
        "schema_name": TRACE_SCHEMA_NAME,
        "schema_version": TRACE_SCHEMA_VERSION,
        "source": "sim",
        "task": args.task,
        "algo": algo,
        "checkpoint": checkpoint,
        "checkpoint_sha256": _sha256(checkpoint),
        "onnx": os.path.abspath(args.onnx) if args.onnx else "",
        "onnx_sha256": _sha256(os.path.abspath(args.onnx)) if args.onnx else "",
        "software": software,
        "seed": int(args.seed),
        "policy_dt_s": float(dt_policy),
        "policy_rate_hz": float(1.0 / dt_policy),
        "episode_length_s": float(env_cfg.episode_length_s),
        "physics_dt_s": float(env_cfg.sim.dt),
        "decimation": int(env_cfg.decimation),
        "action_semantics": "delta",
        "action_scale": float(env_cfg.action_scale),
        "action_clip": [-float(env_cfg.clip_actions), float(env_cfg.clip_actions)],
        "target_units": "radians",
        "joint_order": joint_names,
        "contact_order": CONTACT_ORDER,
        "units": {
            "joint_position": "rad",
            "joint_velocity": "rad/s",
            "force": "N",
            "torque": "N*m",
            "object_position": "m",
            "object_quaternion": "wxyz",
        },
        "single_frame_layout": {
            "joint_pos_unscaled": [0, 21],
            "input_target_policy_rad": [21, 42],
            "force_n": [42, 47],
        },
        "obs_frame_order": ["t-2", "t-1", "t"],
        "history_order": "oldest_to_newest_ending_at_t",
        "row_phase": (
            "Each row contains the state/input sampled at t, the action and integrated target "
            "computed from that input, then the simulator state observed after the 0.05 s "
            "transition. If next_state_is_reset is true, next_* is the auto-reset state rather "
            "than the terminal pre-reset state."
        ),
        "joint_position_input": (
            "policy_pos_rad is the noiseless simulator joint state at t; joint_pos_unscaled and "
            "frame_raw contain the actual policy input, including per-step joint noise and the "
            "per-episode joint zero offset."
        ),
        "normalization": (
            "obs_normalized and proprio_hist_normalized are the actual checkpoint model inputs. "
            "The exported ONNX accepts obs_raw/proprio_hist_raw and applies these normalizers internally."
        ),
        "object_geometry": geometry,
        "randomization": {
            "joint_noise_scale_rad": float(env_cfg.joint_noise_scale),
            "joint_zero_offset_scale_rad": float(env_cfg.joint_zero_offset_scale),
            "pd_gains": bool(env_cfg.randomize_pd_gains),
            "friction": bool(env_cfg.randomize_friction),
            "mass": bool(env_cfg.randomize_mass),
            "com": bool(env_cfg.randomize_com),
            "external_force_scale": float(env_cfg.force_scale),
        },
        "tactile": {
            "enabled": bool(env_cfg.enable_tactile),
            "continuous": not bool(env_cfg.binary_contact),
            "contact_latency": float(env_cfg.contact_latency),
            "force_input_scale": float(env_cfg.contact_force_scale),
            "force_noise_std_scaled": float(env_cfg.contact_force_noise_std),
            "object_filtered": True,
            "sampling": "latest contact-sensor force at each policy observation",
        },
        "next_state_on_done": "Isaac Lab auto-reset state; terminal state is not exposed after step()",
        "cache_row_requested": int(args.cache_row) if args.cache_row is not None else None,
        "command": "tools/dump_runtime_actions.py",
    }


def _save_trace_npz(
    path: Path,
    trace: dict[str, list[np.ndarray]],
    extra_rows: list[dict[str, float]],
    metadata: dict[str, Any],
    agent: PPO | ProprioAdapt,
) -> None:
    payload: dict[str, np.ndarray] = {
        name: np.stack(values, axis=0) for name, values in trace.items()
    }
    payload["metadata_json"] = np.asarray(
        json.dumps(metadata, ensure_ascii=False, sort_keys=True), dtype=np.str_
    )

    # Normalizer state is immutable during evaluation and belongs once per trace,
    # rather than being repeated for every frame.
    payload["obs_rms_mean"] = _np_copy(agent.running_mean_std.running_mean, np.float64)
    payload["obs_rms_var"] = _np_copy(agent.running_mean_std.running_var, np.float64)
    if hasattr(agent, "sa_mean_std"):
        payload["proprio_hist_rms_mean"] = _np_copy(
            agent.sa_mean_std.running_mean, np.float64
        )
        payload["proprio_hist_rms_var"] = _np_copy(
            agent.sa_mean_std.running_var, np.float64
        )

    extra_key_map: dict[str, str] = {}
    for info_name in sorted({key for row in extra_rows for key in row}):
        array_name = _safe_extra_name(info_name)
        extra_key_map[array_name] = info_name
        payload[array_name] = np.asarray(
            [row.get(info_name, np.nan) for row in extra_rows], dtype=np.float32
        )
    metadata["extra_key_map"] = extra_key_map
    payload["metadata_json"] = np.asarray(
        json.dumps(metadata, ensure_ascii=False, sort_keys=True), dtype=np.str_
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{time.monotonic_ns()}.partial.npz"
    )
    try:
        with temporary.open("xb") as stream:
            np.savez_compressed(stream, **payload)
        # Hard-link publication is atomic and, unlike os.replace(), refuses to
        # overwrite a target created by another process after the early check.
        os.link(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main():
    if args.num_envs != 1:
        raise ValueError("This script is for single-env play dump. Please use --num_envs 1.")
    if args.max_frames < 0:
        raise ValueError("--max_frames must be non-negative.")
    if args.episode_length_s is not None and (
        not np.isfinite(args.episode_length_s) or args.episode_length_s <= 0.0
    ):
        raise ValueError("--episode-length-s must be finite and positive.")
    if args.trace_npz and args.episodes != 1:
        raise ValueError("--trace-npz records one unambiguous episode; please pass --episodes 1.")
    if args.cache_row is not None and args.cache_row < 0:
        raise ValueError("--cache-row must be non-negative.")
    trace_path = _trace_output_path(args.trace_npz) if args.trace_npz else None

    set_np_formatting()
    seed = set_seed(args.seed)
    algo = _resolve_algo()
    if args.onnx and algo != "ProprioAdapt":
        raise ValueError("--onnx parity is only supported for the ProprioAdapt Stage-2 policy.")
    full_config = _build_full_config(seed, algo)
    env_cfg = _build_env_cfg(seed)
    full_config.train.ppo.priv_info_dim = int(env_cfg.priv_info_dim)

    raw_env = Revo3HandHoraEnv(
        cfg=env_cfg,
        render_mode=None if getattr(args, "headless", False) else "human",
    )
    # Keep the original full cache for nearest-row reporting. A one-row view
    # makes the existing random sampler deterministically select the requested
    # row without changing default behavior when --cache-row is omitted.
    grasp_cache_reference = raw_env.saved_grasping_states
    if args.cache_row is not None:
        if grasp_cache_reference is None:
            raise ValueError("--cache-row requires an available grasp cache.")
        cache_rows = int(grasp_cache_reference.shape[0])
        if args.cache_row >= cache_rows:
            raise ValueError(
                f"--cache-row {args.cache_row} is outside the cache with {cache_rows} rows."
            )
        raw_env.saved_grasping_states = grasp_cache_reference[
            args.cache_row : args.cache_row + 1
        ]
        raw_env.bucket_grasp = 1
    env = HoraCompatWrapper(raw_env)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%m%d_%H%M%S")

    agent_cls = PPO if algo == "PPO" else ProprioAdapt
    agent = agent_cls(env, str(output_dir), full_config=full_config)
    agent.restore_test(full_config.train.load_path)
    agent.set_eval()

    ort_session = None
    parity_max_abs_error = 0.0
    if args.onnx:
        import onnxruntime as ort

        onnx_path = os.path.abspath(args.onnx)
        if not os.path.isfile(onnx_path):
            raise FileNotFoundError(f"--onnx path not found: {onnx_path}")
        ort_session = ort.InferenceSession(
            onnx_path,
            providers=["CPUExecutionProvider"],
        )
        print(f"[INFO] Simulator raw-input ONNX parity enabled: {onnx_path}")

    dt_policy = float(env_cfg.decimation) * float(env_cfg.sim.dt)
    joint_names = list(raw_env.hand.data.joint_names)
    n_actions = int(raw_env.cfg.action_space)
    trace_metadata = (
        _trace_metadata(
            algo=algo,
            env_cfg=env_cfg,
            raw_env=raw_env,
            joint_names=joint_names,
            dt_policy=dt_policy,
            ort_session=ort_session,
        )
        if trace_path is not None
        else None
    )

    print(f"[INFO] Dump start: algo={algo}, episodes={args.episodes}, dt={dt_policy:.6f}s, action_dim={n_actions}")
    print("[INFO] Joint order: " + ", ".join(f"{i}:{name}" for i, name in enumerate(joint_names)), flush=True)

    header = [
        f"# task={args.task}",
        f"# algo={algo}",
        f"# checkpoint={os.path.abspath(args.checkpoint)}",
        f"# policy_dt_sec={dt_policy:.6f}",
        f"# policy_hz={1.0 / dt_policy:.6f}",
        f"# action_semantics=delta",
        f"# action_formula=target=prev_target+(1/24)*raw_action then clamp(joint_limits)",
        f"# raw_action_definition=policy clamped delta output mu (pre delta-integration, [-1,1])",
        f"# target_definition=cur_targets (delta-accumulated + joint-limit clamped, used in PD formula)",
        f"# jointpos_definition=hand.data.joint_pos (absolute joint angles, rad)",
        "# joint_order=" + ", ".join(f"{i}:{name}" for i, name in enumerate(joint_names)),
        "",
    ]

    for ep in range(args.episodes):
        obs_dict = env.reset()
        if trace_metadata is not None:
            nearest_row, nearest_error = _nearest_cache_row(
                grasp_cache_reference,
                raw_env.hand.data.joint_pos[0],
                n_actions,
            )
            trace_metadata["initial_cache_nearest_row"] = nearest_row
            trace_metadata["max_abs_q_error_rad"] = nearest_error
            trace_metadata["cache_row_actual"] = (
                int(args.cache_row) if args.cache_row is not None else nearest_row
            )
        raw_lines: list[str] = []
        target_lines: list[str] = []
        jointpos_lines: list[str] = []
        trace: dict[str, list[np.ndarray]] = {}
        trace_extra_rows: list[dict[str, float]] = []
        ep_frame = 0
        ep_done = False

        while not ep_done:
            step_start_ns = time.monotonic_ns()
            obs_raw_t = obs_dict["obs"].detach()
            hist_raw_t = obs_dict["proprio_hist"].detach()
            priv_info_t = obs_dict["priv_info"].detach()
            frame_raw_t = hist_raw_t[:, -1, :]
            policy_pos_pre_t = raw_env.hand.data.joint_pos[0].detach().clone()
            joint_vel_pre_t = raw_env.hand.data.joint_vel[0].detach().clone()
            object_pre = _object_state(raw_env)
            contact_vectors_pre = _contact_force_vectors_w(raw_env)
            target_before_t = raw_env.prev_targets[0].detach().clone()

            if algo == "PPO":
                obs_normalized_t = agent.running_mean_std(obs_raw_t)
                input_dict = {
                    "obs": obs_normalized_t,
                    "priv_info": priv_info_t,
                }
                hist_normalized_t = torch.full_like(hist_raw_t, float("nan"))
            else:
                obs_normalized_t = agent.running_mean_std(obs_raw_t)
                hist_normalized_t = agent.sa_mean_std(hist_raw_t)
                input_dict = {
                    "obs": obs_normalized_t,
                    "proprio_hist": hist_normalized_t,
                }

            inference_start_ns = time.monotonic_ns()
            with torch.inference_mode():
                mu_unclipped = agent.model.act_inference(input_dict)
                diagnostics = _network_diagnostics(agent, algo, input_dict, priv_info_t)
                mu = torch.clamp(mu_unclipped, -1.0, 1.0)
            inference_end_ns = time.monotonic_ns()

            # Record policy raw output BEFORE step
            raw_action = mu[0].detach().cpu().numpy()
            onnx_action: np.ndarray | None = None
            if ort_session is not None:
                raw_inputs = {
                    "obs": obs_raw_t.detach().cpu().numpy().astype(np.float32),
                    "proprio_hist": hist_raw_t
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32),
                }
                onnx_action = ort_session.run(["action"], raw_inputs)[0][0]
                frame_error = float(np.max(np.abs(raw_action - onnx_action)))
                parity_max_abs_error = max(parity_max_abs_error, frame_error)

            action_scale = float(env_cfg.action_scale)
            target_unclipped = target_before_t + action_scale * mu[0]
            target_expected = torch.clamp(
                target_unclipped,
                raw_env.hand_dof_lower_limits[0],
                raw_env.hand_dof_upper_limits[0],
            )

            obs_dict, rewards, dones, info = env.step(mu)
            step_end_ns = time.monotonic_ns()

            # Read our computed position target (authoritative for torque control)
            cur_targets = raw_env.cur_targets[0].detach().cpu().numpy()
            # Actual joint angles
            jointpos = raw_env.hand.data.joint_pos[0].detach().cpu().numpy()

            t_sec = ep_frame * dt_policy
            reward_scalar = float(rewards[0].detach().cpu().item())
            done_scalar = int(dones[0].detach().cpu().item())
            terminated_scalar = int(raw_env.reset_terminated[0].detach().cpu().item())
            truncated_scalar = int(raw_env.reset_time_outs[0].detach().cpu().item())

            if trace_path is not None:
                n_dof = n_actions
                joint_pos_unscaled_t = frame_raw_t[0, :n_dof]
                input_target_t = frame_raw_t[0, n_dof : 2 * n_dof]
                force_t = frame_raw_t[0, 2 * n_dof : 2 * n_dof + len(CONTACT_ORDER)]
                input_joint_pos_rad_t = (
                    0.5
                    * (
                        joint_pos_unscaled_t
                        * (
                            raw_env.hand_dof_upper_limits[0]
                            - raw_env.hand_dof_lower_limits[0]
                        )
                        + raw_env.hand_dof_upper_limits[0]
                        + raw_env.hand_dof_lower_limits[0]
                    )
                )
                object_next = _object_state(raw_env)
                torque_t = getattr(raw_env, "torques", None)
                if torque_t is None:
                    torque_t = torch.full_like(raw_env.hand.data.joint_pos, float("nan"))

                _append_trace(trace, "step_index", np.int64(ep_frame))
                _append_trace(trace, "sample_time_s", np.float64(ep_frame * dt_policy))
                _append_trace(trace, "host_step_start_ns", np.int64(step_start_ns))
                _append_trace(trace, "host_inference_start_ns", np.int64(inference_start_ns))
                _append_trace(trace, "host_inference_end_ns", np.int64(inference_end_ns))
                _append_trace(trace, "host_step_end_ns", np.int64(step_end_ns))
                _append_trace(trace, "policy_pos_rad", policy_pos_pre_t,)
                _append_trace(trace, "next_policy_pos_rad", raw_env.hand.data.joint_pos[0])
                _append_trace(trace, "joint_vel_rad_s", joint_vel_pre_t)
                _append_trace(trace, "next_joint_vel_rad_s", raw_env.hand.data.joint_vel[0])
                _append_trace(trace, "joint_pos_unscaled", joint_pos_unscaled_t)
                _append_trace(trace, "input_joint_pos_rad", input_joint_pos_rad_t)
                _append_trace(
                    trace,
                    "input_joint_noise_rad",
                    input_joint_pos_rad_t - policy_pos_pre_t,
                )
                _append_trace(trace, "input_target_policy_rad", input_target_t)
                _append_trace(trace, "force_n", force_t)
                _append_trace(trace, "contact_force_vector_w_n", contact_vectors_pre)
                _append_trace(trace, "frame_raw", frame_raw_t[0])
                _append_trace(trace, "obs_raw", obs_raw_t[0])
                _append_trace(trace, "proprio_hist_raw", hist_raw_t[0])
                _append_trace(trace, "obs_normalized", obs_normalized_t[0])
                _append_trace(trace, "proprio_hist_normalized", hist_normalized_t[0])
                _append_trace(trace, "action_mu_unclipped", mu_unclipped[0])
                _append_trace(trace, "checkpoint_action", mu[0])
                _append_trace(trace, "action", mu[0])
                _append_trace(trace, "target_before_policy_rad", target_before_t)
                _append_trace(trace, "policy_target_unclipped_rad", target_unclipped)
                _append_trace(trace, "policy_target_rad", target_expected)
                _append_trace(trace, "runtime_target_after_step_rad", raw_env.cur_targets[0])
                _append_trace(
                    trace,
                    "target_clipped",
                    torch.abs(target_expected - target_unclipped) > 1.0e-8,
                )
                if onnx_action is not None:
                    _append_trace(trace, "onnx_action_raw", onnx_action)
                    _append_trace(trace, "onnx_action", onnx_action)
                _append_trace(trace, "privileged_info", priv_info_t[0])
                for name, value in diagnostics.items():
                    _append_trace(trace, name, value[0])
                _append_trace(
                    trace,
                    "diagnostic_action_max_abs_error",
                    torch.max(torch.abs(diagnostics["diagnostic_action_mu"][0] - mu_unclipped[0])),
                )
                _append_trace(trace, "object_pose", object_pre["pose"])
                _append_trace(trace, "object_linvel_m_s", object_pre["linvel"])
                _append_trace(trace, "object_angvel_rad_s", object_pre["angvel"])
                _append_trace(trace, "next_object_pose", object_next["pose"])
                _append_trace(trace, "next_object_linvel_m_s", object_next["linvel"])
                _append_trace(trace, "next_object_angvel_rad_s", object_next["angvel"])
                _append_trace(trace, "torque_nm", torque_t[0])
                _append_trace(trace, "p_gain", raw_env.p_gain[0])
                _append_trace(trace, "d_gain", raw_env.d_gain[0])
                _append_trace(trace, "joint_lower_policy_rad", raw_env.hand_dof_lower_limits[0])
                _append_trace(trace, "joint_upper_policy_rad", raw_env.hand_dof_upper_limits[0])
                _append_trace(trace, "reward", np.float32(reward_scalar))
                _append_trace(trace, "done", np.uint8(done_scalar))
                _append_trace(trace, "terminated", np.uint8(terminated_scalar))
                _append_trace(trace, "truncated", np.uint8(truncated_scalar))
                _append_trace(trace, "next_state_is_reset", np.uint8(done_scalar))
                _append_trace(
                    trace,
                    "obs_hist_tail_max_abs_error",
                    torch.max(torch.abs(obs_raw_t.reshape(1, 3, -1) - hist_raw_t[:, -3:, :])),
                )
                trace_extra_rows.append(_scalar_info(info))

            raw_lines.append(
                f"frame={ep_frame:03d} t={t_sec:6.3f}s reward={reward_scalar:+.6f} done={done_scalar} "
                f"raw_action={_fmt_vec(raw_action)}"
            )
            target_lines.append(
                f"frame={ep_frame:03d} t={t_sec:6.3f}s reward={reward_scalar:+.6f} done={done_scalar} "
                f"target={_fmt_vec(cur_targets)}"
            )
            jointpos_lines.append(
                f"frame={ep_frame:03d} t={t_sec:6.3f}s reward={reward_scalar:+.6f} done={done_scalar} "
                f"jointpos={_fmt_vec(jointpos)}"
            )

            ep_frame += 1
            if dones[0].item() or (args.max_frames > 0 and ep_frame >= args.max_frames):
                ep_done = True

        # Write per-episode files
        stem = f"ep{ep:02d}_{args.task}_{algo.lower()}_{ts}"
        (output_dir / f"{stem}.raw_action.txt").write_text("\n".join(header + raw_lines) + "\n", encoding="utf-8")
        (output_dir / f"{stem}.target.txt").write_text("\n".join(header + target_lines) + "\n", encoding="utf-8")
        (output_dir / f"{stem}.jointpos.txt").write_text("\n".join(header + jointpos_lines) + "\n", encoding="utf-8")
        print(f"[OK] Episode {ep+1}/{args.episodes}: {ep_frame} frames "
              f"-> {stem}.{{raw_action,target,jointpos}}.txt", flush=True)

        if trace_path is not None:
            assert trace_metadata is not None
            trace_metadata["frames"] = int(ep_frame)
            trace_metadata["episode_index"] = int(ep)
            trace_metadata["object_mass_kg"] = float(
                raw_env.priv_info_buf[0, 4].detach().cpu().item()
            )
            trace_metadata["object_com_local_m"] = _np_copy(
                raw_env.priv_info_buf[0, 5:8], np.float32
            ).tolist()
            trace_metadata["friction_randomization_scale"] = float(
                raw_env.priv_info_buf[0, 3].detach().cpu().item()
            )
            _save_trace_npz(
                trace_path,
                trace,
                trace_extra_rows,
                trace_metadata,
                agent,
            )
            print(f"[OK] Unified policy trace: {trace_path}", flush=True)

    if ort_session is not None:
        print(
            f"[OK] Simulator raw obs/history -> checkpoint vs ONNX max_abs_error="
            f"{parity_max_abs_error:.9g}",
            flush=True,
        )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("\n[ERROR] Action dump failed. Full traceback:", flush=True)
        traceback.print_exc()
        raise
    finally:
        if os.getenv("HORA_SKIP_SIM_CLOSE", "0") == "1":
            print("[INFO] Skip simulation_app.close() due to HORA_SKIP_SIM_CLOSE=1", flush=True)
        else:
            simulation_app.close()
