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
import os
import sys
import traceback
from pathlib import Path

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
parser.add_argument("--usd", type=str, default="", help="Override hand USD path.")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--episodes", type=int, default=5, help="Number of full episodes to dump.")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--output_dir", type=str, default="outputs/revo3_right/action_dump")
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


def main():
    if args.num_envs != 1:
        raise ValueError("This script is for single-env play dump. Please use --num_envs 1.")

    set_np_formatting()
    seed = set_seed(args.seed)
    algo = _resolve_algo()
    full_config = _build_full_config(seed, algo)
    env_cfg = _build_env_cfg(seed)
    full_config.train.ppo.priv_info_dim = int(env_cfg.priv_info_dim)

    raw_env = Revo3HandHoraEnv(
        cfg=env_cfg,
        render_mode=None if getattr(args, "headless", False) else "human",
    )
    env = HoraCompatWrapper(raw_env)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%m%d_%H%M%S")

    agent_cls = PPO if algo == "PPO" else ProprioAdapt
    agent = agent_cls(env, str(output_dir), full_config=full_config)
    agent.restore_test(full_config.train.load_path)
    agent.set_eval()

    dt_policy = float(env_cfg.decimation) * float(env_cfg.sim.dt)
    joint_names = list(raw_env.hand.data.joint_names)
    n_actions = int(raw_env.cfg.action_space)

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
        raw_lines: list[str] = []
        target_lines: list[str] = []
        jointpos_lines: list[str] = []
        ep_frame = 0
        ep_done = False

        while not ep_done:
            if algo == "PPO":
                input_dict = {
                    "obs": agent.running_mean_std(obs_dict["obs"]),
                    "priv_info": obs_dict["priv_info"],
                }
            else:
                input_dict = {
                    "obs": agent.running_mean_std(obs_dict["obs"]),
                    "proprio_hist": agent.sa_mean_std(obs_dict["proprio_hist"].detach()),
                }

            mu = agent.model.act_inference(input_dict)
            mu = torch.clamp(mu, -1.0, 1.0)

            # Record policy raw output BEFORE step
            raw_action = mu[0].detach().cpu().numpy()

            obs_dict, rewards, dones, _ = env.step(mu)

            # Read our computed position target (authoritative for torque control)
            cur_targets = raw_env.cur_targets[0].detach().cpu().numpy()
            # Actual joint angles
            jointpos = raw_env.hand.data.joint_pos[0].detach().cpu().numpy()

            t_sec = ep_frame * dt_policy
            reward_scalar = float(rewards[0].detach().cpu().item())
            done_scalar = int(dones[0].detach().cpu().item())

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
            if dones[0].item():
                ep_done = True

        # Write per-episode files
        stem = f"ep{ep:02d}_{args.task}_{algo.lower()}_{ts}"
        (output_dir / f"{stem}.raw_action.txt").write_text("\n".join(header + raw_lines) + "\n", encoding="utf-8")
        (output_dir / f"{stem}.target.txt").write_text("\n".join(header + target_lines) + "\n", encoding="utf-8")
        (output_dir / f"{stem}.jointpos.txt").write_text("\n".join(header + jointpos_lines) + "\n", encoding="utf-8")
        print(f"[OK] Episode {ep+1}/{args.episodes}: {ep_frame} frames "
              f"-> {stem}.{{raw_action,target,jointpos}}.txt", flush=True)


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
