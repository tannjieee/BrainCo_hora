"""Visualize initial hand pose and object position from assets.py configs.

Two modes:
  --physics off (default): render only, freeze sim. Check visual pose.
  --physics on: step zero actions, print obj_z/hand_z every 20 steps. Test passive stability.

Task selection: --task ball|cylinder picks the correct robot_cfg (REVO3_HAND_*_CFG)
  and object_cfg (BALL_OBJECT_CFG|CYLINDER_OBJECT_CFG).

Gotcha — joint pose override: after env.reset(), writes cfg.init_state.joint_pos directly
  to sim via write_joint_state_to_sim. This is needed because USD may have baked-in default
  joint positions that differ from assets.py. The env's init_joint_pos is built from the
  same source, but the manual override ensures the render matches exactly.
"""

import argparse
import copy
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument(
    "--task",
    type=str,
    default="ball",
    choices=["ball", "cylinder"],
    help="Task variant: ball or cylinder",
)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--cache", action="store_true", help="Load grasp cache by task instead of assets.py init pose.")
parser.add_argument("--cache_file", type=str, default="", help="Override cache filename under cache/; implies --cache.")
parser.add_argument(
    "--sequential_cache",
    action="store_true",
    help="Map environment i to cache row i (modulo cache size) for deterministic batch validation.",
)
parser.add_argument("--usd", type=str, default="", help="Override hand USD path.")
parser.add_argument(
    "--physics",
    action="store_true",
    help="Step physics with zero actions, useful for checking if assets.py init pose is stable.",
)
parser.add_argument("--steps", type=int, default=0, help="Stop after this many physics steps; 0 runs until closed.")
parser.add_argument("--gravity", type=float, default=None, help="Override downward gravity magnitude in m/s².")
parser.add_argument("--settle_steps", type=int, default=20, help="Steps excluded from stable-phase tilt reporting.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

if args.steps < 0:
    parser.error("--steps must be greater than or equal to 0")
if args.gravity is not None and args.gravity < 0:
    parser.error("--gravity must be greater than or equal to 0")
if args.settle_steps < 0:
    parser.error("--settle_steps must be greater than or equal to 0")

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch

from hora.tasks.isaaclab import Revo3HandHoraEnv, Revo3HandHoraEnvCfg
from hora.tasks.isaaclab.assets import (
    BALL_OBJECT_CFG, CYLINDER_OBJECT_CFG,
    REVO3_HAND_BALL_CFG, REVO3_HAND_CYLINDER_CFG,
)

env_cfg = Revo3HandHoraEnvCfg()

# Select correct robot_cfg and object_cfg based on task
_TASK_ROBOT_CFG = {"ball": REVO3_HAND_BALL_CFG, "cylinder": REVO3_HAND_CYLINDER_CFG}
_TASK_OBJECT_CFG = {"ball": BALL_OBJECT_CFG, "cylinder": CYLINDER_OBJECT_CFG}
env_cfg.robot_cfg = _TASK_ROBOT_CFG.get(args.task, REVO3_HAND_CYLINDER_CFG)
env_cfg.object_cfg = _TASK_OBJECT_CFG.get(args.task, CYLINDER_OBJECT_CFG)

_TASK_CACHE = {"ball": "cache/revo3_right_grasp_ball", "cylinder": "cache/revo3_right_grasp_cylinder"}
use_cache = args.cache or bool(args.cache_file)
if args.sequential_cache and not use_cache:
    parser.error("--sequential_cache requires --cache or --cache_file")
if use_cache:
    if args.cache_file:
        cache_path = f"cache/{args.cache_file.removesuffix('.npy')}"
    else:
        cache_path = _TASK_CACHE.get(args.task, _TASK_CACHE["cylinder"])
    env_cfg.grasp_cache_path = cache_path
else:
    cache_path = "none"
    env_cfg.grasp_cache_path = "__nonexistent__"  # force fallback to init_joint_pos

if args.usd:
    usd_path = os.path.abspath(args.usd)
    if not os.path.exists(usd_path):
        raise FileNotFoundError(f"--usd path not found: {usd_path}")
    env_cfg.robot_cfg = copy.deepcopy(env_cfg.robot_cfg)
    if env_cfg.robot_cfg.spawn is None or not hasattr(env_cfg.robot_cfg.spawn, "usd_path"):
        raise RuntimeError("env_cfg.robot_cfg.spawn has no usd_path to override.")
    env_cfg.robot_cfg.spawn.usd_path = usd_path

env_cfg.scene.num_envs = args.num_envs
env_cfg.grasp_cache_sequential = args.sequential_cache
# Keep interactive inspection from timing out and changing to a different
# cached row. Automated checks use --steps to provide their own finite horizon.
env_cfg.episode_length_s = 9999.0
env_cfg.randomize_mass = False
env_cfg.randomize_com = False
env_cfg.randomize_friction = False
env_cfg.randomize_pd_gains = False
env_cfg.gravity_curriculum = False
env_cfg.force_scale = 0.0
env_cfg.random_force_prob_scalar = 0.0
if args.gravity is not None:
    env_cfg.sim.gravity = (0.0, 0.0, -args.gravity)

print(f"[VIEW] Task: {args.task}")
print(f"[VIEW] Cache: {cache_path if use_cache else 'none (assets.py init pose)'}")
print(f"[VIEW] Gravity: {abs(env_cfg.sim.gravity[2]):g} m/s² downward")
if args.usd:
    print(f"[VIEW] Hand USD override: {os.path.abspath(args.usd)}")

print("[VIEW] Creating environment...", flush=True)
env = Revo3HandHoraEnv(env_cfg, render_mode=None if args.headless else "human")
print("[VIEW] Environment created; resetting...", flush=True)
env.reset()
print("[VIEW] Reset complete.", flush=True)

# Override USD baked-in defaults with assets.py init_state.joint_pos
_init_joint_pos = env_cfg.robot_cfg.init_state.joint_pos
if _init_joint_pos and not use_cache:
    dof_pos = torch.zeros((env.num_envs, env.num_hand_dofs), device=env.device)
    for joint_name, joint_val in _init_joint_pos.items():
        if joint_name in env.hand.joint_names:
            idx = env.hand.joint_names.index(joint_name)
            dof_pos[:, idx] = float(joint_val)
    env.hand.write_joint_state_to_sim(dof_pos, torch.zeros_like(dof_pos))
    env.hand.set_joint_position_target(dof_pos)

# Print actual joint positions to verify assets.py changes took effect.
print("[VIEW] Reading joint state...", flush=True)
joint_names = list(env.hand.joint_names)
print(f"[VIEW] Found {len(joint_names)} joints.", flush=True)
joint_pos = env.hand.data.joint_pos[0].detach().cpu().numpy()
print("[VIEW] Joint state copied to CPU.", flush=True)
print(
    "[VIEW] Actual joint positions after reset (rad): "
    + ", ".join(f"{float(pos):+.4f}" for pos in joint_pos),
    flush=True,
)

zero_actions = torch.zeros((env.num_envs, env_cfg.action_space), device=env.device)
initial_obj_pos = env.object.data.root_pos_w.clone()
initial_obj_z = initial_obj_pos[:, 2]


def cylinder_axis_tilt_deg() -> torch.Tensor:
    """Unsigned angle between the cylinder long axis and world Z."""
    quat = env.object.data.root_quat_w
    quat_x = quat[:, 1]
    quat_y = quat[:, 2]
    alignment = torch.abs(1.0 - 2.0 * (quat_x.square() + quat_y.square()))
    return torch.rad2deg(torch.acos(torch.clamp(alignment, 0.0, 1.0)))

if not args.physics:
    print("\n[VIEW] Frozen render mode.")
    print("  Showing assets.py hand init pose + assets.py object init pos.")
    print("  Add --physics to step zero actions and test passive stability.\n")
    env.sim._physics_context.enabled = False  # freeze physics, render only
    while simulation_app.is_running():
        env.sim.render()
else:
    print("\n[PHYSICS] Stepping with zero actions.", flush=True)
    print("  Testing whether the selected initial pose can hold the object without policy action.", flush=True)
    print("  obj_z printed every 20 steps. Hand z printed for reference.\n", flush=True)
    step = 0
    termination_count = 0
    timeout_count = 0
    max_abs_z_drift = 0.0
    max_horizontal_drift = 0.0
    max_stable_horizontal_drift = 0.0
    max_axis_tilt_deg = 0.0
    max_stable_axis_tilt_deg = 0.0
    while simulation_app.is_running():
        with torch.inference_mode():
            _, _, terminated, truncated, _ = env.step(zero_actions)
        step += 1
        termination_count += int(terminated.sum().item())
        timeout_count += int(truncated.sum().item())
        obj_z = env.object.data.root_pos_w[:, 2]
        max_abs_z_drift = max(
            max_abs_z_drift, float(torch.max(torch.abs(obj_z - initial_obj_z)).item())
        )
        horizontal_drift = torch.norm(
            env.object.data.root_pos_w[:, :2] - initial_obj_pos[:, :2], dim=-1
        )
        max_horizontal_drift = max(
            max_horizontal_drift, float(horizontal_drift.max().item())
        )
        axis_tilt_deg = cylinder_axis_tilt_deg()
        max_axis_tilt_deg = max(max_axis_tilt_deg, float(axis_tilt_deg.max().item()))
        if step > args.settle_steps:
            max_stable_horizontal_drift = max(
                max_stable_horizontal_drift, float(horizontal_drift.max().item())
            )
            max_stable_axis_tilt_deg = max(
                max_stable_axis_tilt_deg, float(axis_tilt_deg.max().item())
            )
        if step % 20 == 0:
            hand_z = env.hand.data.root_pos_w[:, 2]
            print(
                f"  step={step:4d}  "
                f"obj_z={obj_z[0]:.4f}  "
                f"obj_z_range=[{obj_z.min():.4f}, {obj_z.max():.4f}]  "
                f"xy_drift_max={1000.0 * horizontal_drift.max():.2f}mm  "
                f"tilt_range=[{axis_tilt_deg.min():.2f}, {axis_tilt_deg.max():.2f}]deg  "
                f"hand_z={hand_z[0]:.4f}  "
                f"diff={obj_z[0] - hand_z[0]:.4f}"
            )
        if args.steps and step >= args.steps:
            print(
                f"\n[RESULT] steps={step} terminations={termination_count} timeouts={timeout_count} "
                f"max_abs_z_drift={max_abs_z_drift:.6f}m "
                f"max_xy_drift={1000.0 * max_horizontal_drift:.2f}mm "
                f"max_stable_xy_drift={1000.0 * max_stable_horizontal_drift:.2f}mm "
                f"max_axis_tilt={max_axis_tilt_deg:.2f}deg "
                f"max_stable_axis_tilt={max_stable_axis_tilt_deg:.2f}deg",
                flush=True,
            )
            break

env.close()
simulation_app.close()
