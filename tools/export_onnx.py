#!/usr/bin/env python3
"""Export Stage2 ProprioAdapt checkpoint to ONNX.

Stage2 IO:
  obs[B,141] + proprio_hist[B,30,47] -> action[B,21]

The exported ONNX bakes in both RunningMeanStd normalizers:
  - running_mean_std for obs
  - sa_mean_std for proprio_hist

Deploy code must therefore feed raw, simulator-style observations, not already
normalized tensors. See the generated *.deploy_meta.yaml for the exact layout.
"""

from __future__ import annotations

import argparse
import ast
import datetime
import re
import sys
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf

# Ensure repo root is importable when running: `python tools/export_onnx.py ...`
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hora.algo.models.models import ActorCritic
from hora.algo.models.running_mean_std import RunningMeanStd


DEFAULT_ACTOR_UNITS = [512, 256, 128]
DEFAULT_PRIV_UNITS = [256, 128, 8]
DEFAULT_PRIV_DIM = 8
DEFAULT_OBS_DIM = 141
DEFAULT_ACTION_DIM = 21
DEFAULT_PROP_HIST_LEN = 30
DEFAULT_OBS_PER_STEP = 47
OBS_WINDOW = 3
CONTACT_DIM = 5
ACTION_SCALE = 1.0 / 24.0
DOF_LIMITS_SCALE = 0.9

RIGHT_HAND_JOINT_ORDER = [
    "right_index_MPR_joint",
    "right_little_MPR_joint",
    "right_middle_MPR_joint",
    "right_ring_MPR_joint",
    "right_thumb_CMP_joint",
    "right_index_MCP_joint",
    "right_little_MCP_joint",
    "right_middle_MCP_joint",
    "right_ring_MCP_joint",
    "right_thumb_CMR_joint",
    "right_index_PIP_joint",
    "right_little_PIP_joint",
    "right_middle_PIP_joint",
    "right_ring_PIP_joint",
    "right_thumb_MCP_joint",
    "right_index_DIP_joint",
    "right_little_DIP_joint",
    "right_middle_DIP_joint",
    "right_ring_DIP_joint",
    "right_thumb_PIP_joint",
    "right_thumb_DIP_joint",
]

# Exact unscaled limits used by assets/urdf/urdf/revo3_right.urdf and the
# revo3_right USD consumed by the training environment, in runtime joint order.
# The environment multiplies both sides by DOF_LIMITS_SCALE before unscale(),
# target integration, and clamp. Keep the numeric values in deploy metadata so
# a runtime profile cannot silently change network input scaling.
RIGHT_HAND_JOINT_LOWER_RAD = [
    -0.2618, -0.2618, -0.2618, -0.2618, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
]
RIGHT_HAND_JOINT_UPPER_RAD = [
    0.2618, 0.2618, 0.2618, 0.2618, 1.9199,
    1.4835, 1.4835, 1.4835, 1.4835, 2.0071,
    1.4835, 1.4835, 1.4835, 1.4835, 0.8727,
    1.4835, 1.4835, 1.4835, 1.4835, 1.4835, 1.4835,
]


def _find_config_for_checkpoint(ckpt_path: Path) -> Path | None:
    # expected: run_dir/stage2_nn/model_best.ckpt
    run_dir = ckpt_path.parent.parent if ckpt_path.parent.name.endswith("_nn") else ckpt_path.parent
    candidates = sorted(run_dir.glob("config_*.yaml"))
    return candidates[-1] if candidates else None


def _find_run_dir_for_checkpoint(ckpt_path: Path) -> Path:
    # expected: outputs/revo3_right/<run_name>/stage2_nn/*.ckpt
    if ckpt_path.parent.name.endswith("_nn"):
        return ckpt_path.parent.parent
    return ckpt_path.parent


def _load_config(path: Path | None) -> Any | None:
    if path is None or not path.exists():
        return None
    return OmegaConf.load(str(path))


def _get_cfg_value(cfg: Any | None, key_path: str, default: Any) -> Any:
    if cfg is None:
        return default
    node: Any = cfg
    for key in key_path.split("."):
        if isinstance(node, dict) and key in node:
            node = node[key]
            continue
        if hasattr(node, key):
            node = getattr(node, key)
            continue
        return default
    return node


def _build_net_config(
    cfg: Any | None,
    obs_dim: int,
    actions_num: int,
    priv_info_dim: int,
    obs_per_step: int,
) -> dict[str, Any]:
    actor_units = list(_get_cfg_value(cfg, "train.network.mlp.units", DEFAULT_ACTOR_UNITS))
    priv_units = list(_get_cfg_value(cfg, "train.network.priv_mlp.units", DEFAULT_PRIV_UNITS))
    priv_dim = int(priv_info_dim)
    priv_info = True
    proprio_adapt = True

    return {
        "actor_units": actor_units,
        "priv_mlp_units": priv_units,
        "actions_num": actions_num,
        "input_shape": (obs_dim,),
        "priv_info": priv_info,
        "proprio_adapt": proprio_adapt,
        "priv_info_dim": priv_dim,
        "obs_per_step": int(obs_per_step),
    }


class Stage2ExportWrapper(torch.nn.Module):
    def __init__(self, model: ActorCritic, rms_obs: RunningMeanStd, rms_hist: RunningMeanStd):
        super().__init__()
        self.model = model
        self.rms_obs = rms_obs
        self.rms_hist = rms_hist

    def forward(self, obs: torch.Tensor, proprio_hist: torch.Tensor) -> torch.Tensor:
        obs_n = self.rms_obs(obs)
        hist_n = self.rms_hist(proprio_hist)
        mu = self.model.act_inference({"obs": obs_n, "proprio_hist": hist_n})
        return torch.clamp(mu, -1.0, 1.0)


def _save_deploy_meta(
    meta_path: Path,
    *,
    checkpoint: Path,
    onnx_path: Path,
    cfg_path: Path | None,
    obs_dim: int,
    action_dim: int,
    prop_hist_len: int,
    obs_per_step: int,
    obs_window: int,
    dynamic_batch: bool,
    normalize_baked_in: bool,
    policy_rate: float,
    chunk_size: int,
    n_action_steps: int,
    contact_force_scale: float,
    runtime_reference: dict[str, Any],
) -> None:
    dof_count = int(action_dim)
    contact_start = dof_count * 2
    contact_end = contact_start + CONTACT_DIM
    meta = {
        "export": {
            "stage": "stage2",
            "policy_type": "ProprioAdapt",
            "checkpoint": str(checkpoint),
            "onnx": str(onnx_path),
            "config_yaml": str(cfg_path) if cfg_path is not None else "",
            "normalization_baked_in": bool(normalize_baked_in),
        },
        "io_contract": {
            "inputs": [
                {"name": "obs", "shape": ["B", obs_dim], "dtype": "float32"},
                {"name": "proprio_hist", "shape": ["B", prop_hist_len, obs_per_step], "dtype": "float32"},
            ],
            "not_inputs": [
                {
                    "name": "priv_info",
                    "reason": "Stage2 deploy does not receive privileged information. During training only, env_mlp(priv_info) is used as the teacher latent target for adapt_tconv.",
                }
            ],
            "outputs": [{"name": "action", "shape": ["B", action_dim], "dtype": "float32"}],
            "action_semantics": "delta",
            "action_formula": "cur_targets = prev_targets + action_scale * action, then clamp to scaled joint limits",
            "action_scale": float(ACTION_SCALE),
            "action_clip": [-1.0, 1.0],
            "target_units": "radians",
            "policy_rate_hz": float(policy_rate),
            "chunk_size": int(chunk_size),
            "n_action_steps": int(n_action_steps),
            "dynamic_batch": dynamic_batch,
        },
        "deploy_observation_contract": {
            "important": "Feed raw observations with this layout. Do not apply RunningMeanStd outside the ONNX graph.",
            "stage2_summary": "Stage2 keeps the Stage1 tactile actor observation ABI at 141 dims and replaces env_mlp(priv_info) with adapt_tconv(proprio_hist). Both inputs require the five measured fingertip contact-force channels.",
            "joint_order": RIGHT_HAND_JOINT_ORDER,
            "joint_order_source": "Isaac Lab runtime hand.data.joint_names; same order is used by obs, proprio_hist, action, cur_targets, and joint limits.",
            "dof_count": dof_count,
            "dof_limits": {
                "source": "Isaac Lab hand.root_physx_view.get_dof_limits()",
                "scaled_by": float(DOF_LIMITS_SCALE),
                "lower": "hand_dof_lower_limits = runtime_lower_limits * 0.9",
                "upper": "hand_dof_upper_limits = runtime_upper_limits * 0.9",
                "unscaled_lower_rad": RIGHT_HAND_JOINT_LOWER_RAD,
                "unscaled_upper_rad": RIGHT_HAND_JOINT_UPPER_RAD,
                "scaled_lower_rad": [
                    float(value * DOF_LIMITS_SCALE)
                    for value in RIGHT_HAND_JOINT_LOWER_RAD
                ],
                "scaled_upper_rad": [
                    float(value * DOF_LIMITS_SCALE)
                    for value in RIGHT_HAND_JOINT_UPPER_RAD
                ],
            },
            "single_frame_layout": [
                {
                    "name": "joint_pos_unscaled",
                    "slice": [0, dof_count],
                    "size": dof_count,
                    "units": "dimensionless",
                    "formula": "(2 * joint_pos_rad - joint_upper_rad - joint_lower_rad) / (joint_upper_rad - joint_lower_rad)",
                    "training_noise": "Stage2 training inherited per-step joint noise and a per-episode joint-zero offset before unscale; deploy should use measured joint_pos without adding synthetic noise or offsets.",
                },
                {
                    "name": "cur_targets",
                    "slice": [dof_count, dof_count * 2],
                    "size": dof_count,
                    "units": "radians",
                    "formula": "previous cur_targets after delta integration and joint-limit clamp",
                },
                {
                    "name": "contact_forces",
                    "slice": [contact_start, contact_end],
                    "size": CONTACT_DIM,
                    "units": "newtons",
                    "input_scale": float(contact_force_scale),
                    "order": ["thumb_DIP", "index_DIP", "middle_DIP", "ring_DIP", "little_DIP"],
                    "sim_formula": "max(0, input_scale * norm(current object-filtered fingertip contact force) + Gaussian training noise), sampled once per 20 Hz policy step, with optional latency hold",
                    "deploy_formula": "input_scale * measured fingertip force magnitude; do not add training noise",
                },
            ],
            "obs": {
                "shape": ["B", obs_dim],
                "window_frames": int(obs_window),
                "frame_order": ["t-2", "t-1", "t"],
                "flattening": "concat three 47-dim frames in chronological order: [frame(t-2), frame(t-1), frame(t)]",
                "construction_steps": [
                    "For each policy step, build one 47-dim raw frame as joint_pos_unscaled[21] + cur_targets[21] + contact_forces[5].",
                    "Append the frame to a 3-frame chronological obs window.",
                    "Flatten the window to 141 dims.",
                    "Preserve all five measured contact-force values in every frame.",
                ],
                "stage2_contact_rule": "contact_forces are required: Stage1, Stage2, and deployment all use the same tactile channels.",
                "privileged_info_rule": "Do not append or provide priv_info to obs. Stage2 ONNX has no priv_info input.",
                "contact_slices": [
                    [frame * obs_per_step + contact_start, frame * obs_per_step + contact_end]
                    for frame in range(int(obs_window))
                ],
            },
            "proprio_hist": {
                "shape": ["B", prop_hist_len, obs_per_step],
                "window_frames": int(prop_hist_len),
                "frame_order": "oldest to newest, ending at current frame t",
                "per_frame_layout": "same 47-dim single_frame_layout as above",
                "construction_steps": [
                    "Maintain a 30-frame chronological history of raw 47-dim frames.",
                    "Each history frame uses joint_pos_unscaled[21] + cur_targets[21] + contact_forces[5].",
                    "Populate contact_forces[42:47] from the five real fingertip tactile channels after applying metadata input_scale.",
                ],
                "contact_rule": "Tactile contact values are required and must use the same order, units, calibration, and sampling rate as the actor observation.",
            },
            "reset_initialization": {
                "history_fill": "On reset, fill all history frames with the current joint_pos_unscaled, current cur_targets, and current contact values.",
                "cur_targets_initial_value": "initial joint position in radians, usually sampled from grasp cache or assets.py init_joint_pos",
            },
        },
        "normalization": {
            "baked_in_onnx": normalize_baked_in,
            "obs_normalizer": "checkpoint['running_mean_std']",
            "proprio_hist_normalizer": "checkpoint['sa_mean_std']",
        },
        "runtime_reference": runtime_reference,
    }
    with meta_path.open("w", encoding="utf-8") as f:
        f.write(OmegaConf.to_yaml(OmegaConf.create(meta)))


def _shape_list_from_state(state: dict[str, Any], key: str) -> list[int] | None:
    tensor = state.get(key)
    if tensor is None or not hasattr(tensor, "shape"):
        return None
    return [int(x) for x in tuple(tensor.shape)]


def _require_stage2_checkpoint(checkpoint: dict[str, Any], ckpt_path: Path) -> None:
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise RuntimeError(f"Expected a Stage2 checkpoint dict with a 'model' key: {ckpt_path}")
    if checkpoint.get("tactile_required") is False:
        raise RuntimeError(
            f"Checkpoint was trained for a no-tactile ABI, but this exporter now targets tactile Stage2: {ckpt_path}"
        )
    if "tactile_required" not in checkpoint:
        print(
            "[WARN] Checkpoint predates tactile ABI metadata; verify that actor obs and proprio_hist used real contacts.",
            flush=True,
        )
    model_state = checkpoint["model"]
    if not isinstance(model_state, dict):
        raise RuntimeError(f"Checkpoint 'model' is not a state_dict: {ckpt_path}")
    required_model_keys = [
        "adapt_tconv.channel_transform.0.weight",
        "env_mlp.mlp.0.weight",
        "actor_mlp.mlp.0.weight",
    ]
    missing_model = [key for key in required_model_keys if key not in model_state]
    if missing_model:
        raise RuntimeError(
            f"Checkpoint does not look like a Stage2 ProprioAdapt checkpoint. "
            f"Missing model keys {missing_model}: {ckpt_path}"
        )
    missing_top = [key for key in ["running_mean_std", "sa_mean_std"] if key not in checkpoint]
    if missing_top:
        raise RuntimeError(
            f"Stage2 export requires checkpoint normalizers {missing_top}. "
            f"Use a full stage2 .ckpt checkpoint: {ckpt_path}"
        )


def _validate_stage2_shapes(
    *,
    obs_dim: int,
    prop_hist_len: int,
    obs_per_step: int,
    action_dim: int,
    priv_info_dim: int,
    checkpoint: dict[str, Any],
) -> None:
    expected_obs_per_step = action_dim * 2 + CONTACT_DIM
    if obs_dim != OBS_WINDOW * obs_per_step:
        raise ValueError(
            f"Stage2 obs_dim mismatch: obs_dim={obs_dim}, but OBS_WINDOW({OBS_WINDOW}) "
            f"* obs_per_step({obs_per_step}) = {OBS_WINDOW * obs_per_step}."
        )
    if obs_per_step != expected_obs_per_step:
        raise ValueError(
            f"Stage2 obs_per_step mismatch: got {obs_per_step}, expected "
            f"action_dim*2+{CONTACT_DIM} = {expected_obs_per_step}."
        )
    if prop_hist_len <= 0:
        raise ValueError(f"prop_hist_len must be positive, got {prop_hist_len}.")

    model_state = checkpoint["model"]
    actor_in = int(model_state["actor_mlp.mlp.0.weight"].shape[1])
    expected_actor_in = obs_dim + int(model_state["env_mlp.mlp.4.weight"].shape[0])
    if actor_in != expected_actor_in:
        raise ValueError(
            f"Actor input mismatch in checkpoint: actor_mlp input={actor_in}, "
            f"expected obs_dim({obs_dim}) + priv_embed_dim = {expected_actor_in}."
        )
    adapt_obs_per_step = int(model_state["adapt_tconv.channel_transform.0.weight"].shape[1])
    if adapt_obs_per_step != obs_per_step:
        raise ValueError(
            f"Adapt input mismatch in checkpoint: adapt_tconv obs_per_step={adapt_obs_per_step}, "
            f"but metadata obs_per_step={obs_per_step}."
        )
    env_priv_dim = int(model_state["env_mlp.mlp.0.weight"].shape[1])
    if env_priv_dim != priv_info_dim:
        raise ValueError(
            f"Priv info dim mismatch: env_mlp expects {env_priv_dim}, inferred/configured {priv_info_dim}."
        )


def _infer_priv_info_dim_from_ckpt(checkpoint: dict[str, Any]) -> int | None:
    state = checkpoint.get("model") if isinstance(checkpoint, dict) else None
    if state is None:
        state = checkpoint
    if not isinstance(state, dict):
        return None
    weight = state.get("env_mlp.mlp.0.weight")
    if weight is None or not hasattr(weight, "shape"):
        return None
    return int(weight.shape[1])


def _extract_joint_pos_from_assets(task_key: str) -> dict[str, float] | None:
    assets_path = REPO_ROOT / "hora" / "tasks" / "isaaclab" / "assets.py"
    try:
        text = assets_path.read_text(encoding="utf-8")
    except Exception:
        return None

    if task_key == "cylinder":
        marker = "REVO3_HAND_CYLINDER_CFG"
    elif task_key == "ball":
        marker = "REVO3_HAND_BALL_CFG"
    else:
        return None

    pattern = rf"{marker}.*?joint_pos=\{{(.*?)\n\s*\}},"
    match = re.search(pattern, text, re.S)
    if not match:
        return None
    dict_str = "{" + match.group(1) + "}"
    try:
        joint_pos = ast.literal_eval(dict_str)
    except Exception:
        return None
    return {str(k): float(v) for k, v in joint_pos.items()}


def _resolve_init_joint_pos(run_dir: Path) -> dict[str, float] | None:
    name = run_dir.name.lower()
    if "cylinder" in name:
        return _extract_joint_pos_from_assets("cylinder")
    if "ball" in name:
        return _extract_joint_pos_from_assets("ball")
    return None


def _resolve_scale_keys_from_config(cfg: Any | None) -> list[float]:
    scale_list = _get_cfg_value(cfg, "env_runtime.randomize_scale_list", None)
    if scale_list is not None:
        parsed = [float(s) for s in scale_list]
        if len(parsed) > 0:
            return parsed
    return []


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Stage2 checkpoint to ONNX.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to stage2 .ckpt.")
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Output ONNX path. If empty, use policy_MMDDHH.onnx.",
    )
    parser.add_argument("--config", type=str, default="", help="Optional config_*.yaml. Auto-resolved if empty.")
    parser.add_argument("--obs_dim", type=int, default=DEFAULT_OBS_DIM)
    parser.add_argument("--action_dim", type=int, default=DEFAULT_ACTION_DIM)
    parser.add_argument("--prop_hist_len", type=int, default=DEFAULT_PROP_HIST_LEN)
    parser.add_argument("--policy_rate", type=float, default=20.0, help="Policy control frequency in Hz.")
    parser.add_argument("--chunk_size", type=int, default=1, help="Chunk size for action playback.")
    parser.add_argument("--n_action_steps", type=int, default=1, help="Number of action steps per chunk.")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--no_dynamic_batch", action="store_true", help="Disable dynamic batch axis.")
    parser.add_argument(
        "--meta_output",
        type=str,
        default="",
        help="Optional deploy meta yaml path. Default: <output_without_ext>.deploy_meta.yaml",
    )
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint).expanduser().resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    run_dir = _find_run_dir_for_checkpoint(ckpt_path)
    onnx_dir = Path("outputs") / "revo3_right" / "onnx"
    output_name = args.output.strip() if isinstance(args.output, str) else ""
    if not output_name:
        # Default: outputs/revo3_right/onnx/policy_MMDDHH.onnx
        out_path = (onnx_dir / f"policy_{datetime.datetime.now().strftime('%m%d%H')}.onnx").resolve()
    else:
        out_arg_path = Path(output_name).expanduser()
        if out_arg_path.is_absolute() or out_arg_path.parent != Path("."):
            out_path = out_arg_path.resolve()
        else:
            # If only a filename is given, still place it under outputs/revo3_right/onnx/
            out_path = (onnx_dir / out_arg_path.name).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cfg_path = Path(args.config).expanduser().resolve() if args.config else _find_config_for_checkpoint(ckpt_path)
    cfg = _load_config(cfg_path)
    if cfg_path is None:
        print("[WARN] config_*.yaml not found near checkpoint; using script defaults.")
    else:
        print(f"[INFO] Using config: {cfg_path}")

    checkpoint = torch.load(str(ckpt_path), map_location="cpu")
    _require_stage2_checkpoint(checkpoint, ckpt_path)
    checkpoint_observation_contract = checkpoint.get("observation_contract") or {}
    inferred_priv_dim = _infer_priv_info_dim_from_ckpt(checkpoint)
    priv_info_dim = int(
        inferred_priv_dim
        if inferred_priv_dim is not None
        else _get_cfg_value(cfg, "train.ppo.priv_info_dim", DEFAULT_PRIV_DIM)
    )

    obs_dim = int(args.obs_dim)
    prop_hist_len = int(args.prop_hist_len)
    rms_obs_shape = None
    if "running_mean_std" in checkpoint and isinstance(checkpoint["running_mean_std"], dict):
        rms_obs_shape = _shape_list_from_state(checkpoint["running_mean_std"], "running_mean")
        if rms_obs_shape:
            if obs_dim == DEFAULT_OBS_DIM:
                obs_dim = int(rms_obs_shape[0])
            elif obs_dim != int(rms_obs_shape[0]):
                print(
                    f"[WARN] obs_dim={obs_dim} does not match checkpoint RMS shape {rms_obs_shape}",
                    flush=True,
                )

    obs_per_step = obs_dim // 3
    rms_hist_shape = None
    if "sa_mean_std" in checkpoint and isinstance(checkpoint["sa_mean_std"], dict):
        rms_hist_shape = _shape_list_from_state(checkpoint["sa_mean_std"], "running_mean")
        if rms_hist_shape:
            if prop_hist_len == DEFAULT_PROP_HIST_LEN:
                prop_hist_len = int(rms_hist_shape[0])
            obs_per_step = int(rms_hist_shape[1])
            if prop_hist_len != int(rms_hist_shape[0]) or obs_per_step != int(rms_hist_shape[1]):
                print(
                    f"[WARN] proprio_hist shape mismatch: expected {rms_hist_shape}, "
                    f"got [{prop_hist_len}, {obs_per_step}]",
                    flush=True,
                )

    actions_num = int(args.action_dim)
    scale_keys = _resolve_scale_keys_from_config(cfg)

    if obs_per_step != (obs_dim // 3):
        print(
            f"[WARN] obs_per_step={obs_per_step} differs from obs_dim/3={obs_dim // 3}. "
            "Check stage2 obs configuration.",
            flush=True,
        )
    _validate_stage2_shapes(
        obs_dim=obs_dim,
        prop_hist_len=prop_hist_len,
        obs_per_step=obs_per_step,
        action_dim=actions_num,
        priv_info_dim=priv_info_dim,
        checkpoint=checkpoint,
    )

    net_config = _build_net_config(
        cfg,
        obs_dim=obs_dim,
        actions_num=actions_num,
        priv_info_dim=priv_info_dim,
        obs_per_step=obs_per_step,
    )
    model = ActorCritic(net_config).cpu().eval()
    rms_obs = RunningMeanStd((obs_dim,)).cpu().eval()
    if "model" in checkpoint:
        model.load_state_dict(checkpoint["model"], strict=True)
    else:
        model.load_state_dict(checkpoint, strict=True)

    if "running_mean_std" in checkpoint:
        rms_obs.load_state_dict(checkpoint["running_mean_std"])
    else:
        print("[WARN] running_mean_std missing in checkpoint; obs normalization will use default stats.")

    dynamic_batch = not args.no_dynamic_batch
    dynamic_axes: dict[str, dict[int, str]] | None = None
    if dynamic_batch:
        dynamic_axes = {"action": {0: "B"}}

    rms_hist = RunningMeanStd((prop_hist_len, obs_per_step)).cpu().eval()
    if "sa_mean_std" in checkpoint:
        rms_hist.load_state_dict(checkpoint["sa_mean_std"])
        if isinstance(checkpoint["sa_mean_std"], dict):
            rms_hist_shape = _shape_list_from_state(checkpoint["sa_mean_std"], "running_mean")
    else:
        print("[WARN] sa_mean_std missing in checkpoint; proprio_hist normalization will use default stats.")

    expected_obs = [obs_dim]
    expected_hist = [prop_hist_len, obs_per_step]
    if rms_obs_shape is not None and rms_obs_shape != expected_obs:
        print(f"[WARN] running_mean_std shape mismatch: ckpt={rms_obs_shape}, expected={expected_obs}")
    if rms_hist_shape is not None and rms_hist_shape != expected_hist:
        print(f"[WARN] sa_mean_std shape mismatch: ckpt={rms_hist_shape}, expected={expected_hist}")
    print(
        f"[INFO] Stage2 reference -> RMS(obs)={expected_obs}, RMS(hist)={expected_hist}, "
        f"joints={len(RIGHT_HAND_JOINT_ORDER)}, scale_keys={scale_keys if scale_keys else 'unknown'}",
        flush=True,
    )

    wrapper = Stage2ExportWrapper(model, rms_obs, rms_hist).eval()
    obs = torch.zeros((1, obs_dim), dtype=torch.float32)
    proprio_hist = torch.zeros((1, prop_hist_len, obs_per_step), dtype=torch.float32)
    input_names = ["obs", "proprio_hist"]
    output_names = ["action"]
    if dynamic_batch:
        dynamic_axes = {
            "obs": {0: "B"},
            "proprio_hist": {0: "B"},
            "action": {0: "B"},
        }
    torch.onnx.export(
        wrapper,
        (obs, proprio_hist),
        str(out_path),
        opset_version=args.opset,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        do_constant_folding=True,
    )

    meta_path = (
        Path(args.meta_output).expanduser().resolve()
        if args.meta_output
        else out_path.with_suffix(".deploy_meta.yaml")
    )
    _save_deploy_meta(
        meta_path,
        checkpoint=ckpt_path,
        onnx_path=out_path,
        cfg_path=cfg_path,
        obs_dim=obs_dim,
        action_dim=actions_num,
        prop_hist_len=prop_hist_len,
        obs_per_step=obs_per_step,
        obs_window=3,
        dynamic_batch=dynamic_batch,
        normalize_baked_in=True,
        policy_rate=float(args.policy_rate),
        chunk_size=int(args.chunk_size),
        n_action_steps=int(args.n_action_steps),
        contact_force_scale=float(
            checkpoint_observation_contract.get(
                "contact_force_scale",
                _get_cfg_value(cfg, "env_runtime.contact_force_scale", 1.0),
            )
        ),
        runtime_reference={
            "scale_keys": scale_keys,
            "running_mean_std_obs_shape": rms_obs_shape if rms_obs_shape is not None else [obs_dim],
            "running_mean_std_hist_shape": (
                rms_hist_shape if rms_hist_shape is not None else [prop_hist_len, obs_per_step]
            ),
            "joint_order_source": "Isaac Lab runtime hand.data.joint_names",
            "joint_order_right_hand": RIGHT_HAND_JOINT_ORDER,
            "init_joint_pos": _resolve_init_joint_pos(run_dir),
        },
    )

    print(f"[OK] Exported ONNX: {out_path}")
    print(f"[OK] Export metadata: {meta_path}")


if __name__ == "__main__":
    main()
