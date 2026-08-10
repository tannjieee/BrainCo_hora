#!/usr/bin/env python3
"""Training entry point for Stage1 (PPO) and Stage2 (ProprioAdapt).

Task selection: --task ball|cylinder selects robot_cfg, object_cfg, and grasp cache.
  robot_cfg and object_cfg are chosen from assets.py (not env_cfg.py class defaults).

Cache path: {grasp_cache_path}.npy under cache/. Override with --cache_file.

Gotcha — num_envs × horizon_length must be >= minibatch_size and exactly
  divisible by it.  The current 16-step horizon permits 2048, 4096, ... envs.

Gotcha — tactile deployment: Stage1, Stage2 actor obs, and Stage2
  proprio_hist all retain the same five fingertip contact-force channels.
"""

import argparse
import copy
import datetime
import os
import subprocess
import traceback

os.environ.setdefault("HORA_SKIP_SIM_CLOSE", "1")

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, default='cylinder', choices=['ball', 'cylinder'])
parser.add_argument('--algo', type=str, default='PPO', choices=['PPO', 'ProprioAdapt'])
parser.add_argument('--train_cfg', type=str, default='Revo3HandHora')
parser.add_argument('--output_name', type=str, default='debug')
parser.add_argument('--checkpoint', type=str, default='')
parser.add_argument('--cache_file', type=str, default='', help='Override grasp cache filename under cache/.')
parser.add_argument('--usd', type=str, default='', help='Override hand USD path.')
parser.add_argument('--num_envs', type=int, default=16384)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument(
    '--max_agent_steps', type=int, default=None,
    help='Override train.ppo.max_agent_steps (useful when resuming beyond the original training budget).',
)
parser.add_argument('--test', action='store_true')
parser.add_argument(
    '--test_steps', type=int, default=0,
    help='Finite full-gravity checkpoint evaluation length in policy steps; 0 keeps interactive play.',
)
parser.add_argument('--video', action='store_true', help='Record one test video from the viewport camera.')
parser.add_argument(
    '--video_seconds', type=float, default=10.0,
    help='Recorded video duration in simulation seconds (default: 10).',
)
parser.add_argument(
    '--video_dir', type=str, default='outputs/revo3_right/videos',
    help='Directory for recorded MP4 files.',
)
parser.add_argument(
    '--real-time', dest='real_time', action='store_true',
    help='Pace test playback to the environment control rate when the machine is fast enough.',
)
parser.add_argument(
    '--camera_eye', type=float, nargs=3, metavar=('X', 'Y', 'Z'), default=None,
    help='Viewport camera position in world coordinates.',
)
parser.add_argument(
    '--camera_lookat', type=float, nargs=3, metavar=('X', 'Y', 'Z'), default=None,
    help='Viewport camera target in world coordinates.',
)
parser.add_argument('--force_overwrite', action='store_true')
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

if args.video:
    args.enable_cameras = True


def _is_stage2_checkpoint(path: str) -> bool:
    if not path:
        return False
    return path.endswith('.ckpt') or 'stage2_nn' in path


def _default_output_name() -> str:
    if args.algo == 'PPO':
        return 'run1_continue' if args.checkpoint else f'run_{args.task}'
    # Stage2 warm-start/resume: keep stage1_nn and stage2_nn in one run dir.
    checkpoint_run_dir = os.path.dirname(os.path.dirname(os.path.abspath(args.checkpoint)))
    return os.path.basename(checkpoint_run_dir) if checkpoint_run_dir else f'run_{args.task}'


_auto_output_name = args.output_name == 'debug'
if not args.test and _auto_output_name:
    args.output_name = _default_output_name()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

from omegaconf import OmegaConf
from termcolor import cprint

from hora.algo.padapt.padapt import ProprioAdapt
from hora.algo.ppo.ppo import PPO
from hora.tasks.isaaclab import HoraCompatWrapper, Revo3HandHoraEnv, Revo3HandHoraEnvCfg
from hora.tasks.isaaclab.assets import (
    BALL_OBJECT_CFG, CYLINDER_OBJECT_CFG,
    REVO3_HAND_BALL_CFG, REVO3_HAND_CYLINDER_CFG,
)
from hora.utils.misc import set_np_formatting, set_seed


_ALGO_MAP = {
    'PPO': PPO,
    'ProprioAdapt': ProprioAdapt,
}

_TASK_ROBOT_CFG = {'ball': REVO3_HAND_BALL_CFG, 'cylinder': REVO3_HAND_CYLINDER_CFG}
_TASK_OBJECT_CFG = {'ball': BALL_OBJECT_CFG, 'cylinder': CYLINDER_OBJECT_CFG}
_TASK_CACHE = {
    'ball': 'cache/revo3_right_grasp_ball',
    'cylinder': 'cache/revo3_right_grasp_cylinder',
}

def _build_full_config(seed: int):
    cfg_path = os.path.join(os.path.dirname(__file__), 'configs', 'train', f'{args.train_cfg}.yaml')
    train_cfg = OmegaConf.load(cfg_path)
    train_cfg.algo = args.algo
    train_cfg.load_path = os.path.abspath(args.checkpoint) if args.checkpoint else ''
    train_cfg.ppo.output_name = args.output_name
    minibatch = train_cfg.ppo.minibatch_size
    min_envs = minibatch // train_cfg.ppo.horizon_length
    if (
        not args.test
        and args.algo == 'PPO'
        and (args.num_envs < min_envs or (args.num_envs * train_cfg.ppo.horizon_length) % minibatch != 0)
    ):
        raise ValueError(
            f"num_envs ({args.num_envs}) must be >= {min_envs} and num_envs*horizon must be divisible "
            f"by minibatch_size ({minibatch}). Valid num_envs: {', '.join(str(i) for i in range(min_envs, 20000, min_envs))}..."
        )
    train_cfg.ppo.num_actors = args.num_envs
    if args.max_agent_steps is not None:
        if args.max_agent_steps <= 0:
            raise ValueError('--max_agent_steps must be positive')
        train_cfg.ppo.max_agent_steps = args.max_agent_steps
    train_cfg.ppo.priv_info = True
    train_cfg.ppo.proprio_adapt = args.algo == 'ProprioAdapt'

    rl_device = getattr(args, 'device', None) or 'cuda:0'
    return OmegaConf.create({
        'rl_device': rl_device,
        'test': args.test,
        'seed': seed,
        'train': train_cfg,
    })


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

    env_cfg.scene.num_envs = args.num_envs


    if hasattr(env_cfg, 'seed'):
        env_cfg.seed = seed
    if hasattr(env_cfg.sim, 'device') and getattr(args, 'device', None):
        env_cfg.sim.device = args.device
    return env_cfg


def _save_run_metadata(output_dif: str, full_config) -> None:
    date = str(datetime.datetime.now().strftime('%m%d_%H%M%S'))
    with open(os.path.join(output_dif, f'gitdiff_{date}.patch'), 'w', encoding='utf-8') as f:
        try:
            result = subprocess.run(
                ['git', 'diff', '--binary', 'HEAD'],
                check=True,
                capture_output=True,
                text=True,
            )
            f.write(result.stdout)
        except (OSError, subprocess.CalledProcessError) as exc:
            f.write(f'# Unable to capture git diff: {exc}\n')
    config_name = f'config_{date}.yaml'

    with open(os.path.join(output_dif, config_name), 'w', encoding='utf-8') as f:
        f.write(OmegaConf.to_yaml(full_config))


def _attach_env_runtime_to_config(full_config, env_cfg) -> None:
    full_config.env_runtime = OmegaConf.create(
        {
            'grasp_cache_path': str(env_cfg.grasp_cache_path),
            'enable_tactile': bool(env_cfg.enable_tactile),
            'enable_contact_in_obs': bool(env_cfg.enable_contact_in_obs),
            'contact_order': ['thumb_DIP', 'index_DIP', 'middle_DIP', 'ring_DIP', 'little_DIP'],
            'policy_dt': float(env_cfg.decimation * env_cfg.sim.dt),
            'gravity': tuple(float(v) for v in env_cfg.sim.gravity),
        }
    )


def main():
    if args.test and not args.checkpoint:
        raise ValueError('--test requires --checkpoint')
    if args.video and not args.test:
        raise ValueError('--video is only supported together with --test')
    if args.video_seconds <= 0:
        raise ValueError('--video_seconds must be positive')
    if args.algo == 'ProprioAdapt' and not args.checkpoint:
        raise ValueError('ProprioAdapt training requires --checkpoint')

    set_np_formatting()
    seed = set_seed(args.seed)
    full_config = _build_full_config(seed)

    cprint('Start Building the Environment', 'green', attrs=['bold'])
    env_cfg = _build_env_cfg(seed)
    if args.camera_eye is not None:
        env_cfg.viewer.eye = tuple(args.camera_eye)
    if args.camera_lookat is not None:
        env_cfg.viewer.lookat = tuple(args.camera_lookat)
    if args.algo == 'ProprioAdapt':
        # Tactile deployment contract: keep the five fingertip-force channels
        # in both the frozen actor observation and the adaptation history.
        env_cfg.enable_contact_in_obs = True
        env_cfg.gravity_curriculum = False     # Stage2: actor frozen from Stage1, must train at full gravity
        env_cfg.sim.gravity = (0.0, 0.0, -9.81)
    if args.test:
        env_cfg.gravity_curriculum = False
        env_cfg.sim.gravity = (0.0, 0.0, -9.81)  # full gravity for test/play
    raw_env = Revo3HandHoraEnv(
        cfg=env_cfg,
        render_mode='rgb_array' if args.video else (None if getattr(args, 'headless', False) else 'human'),
    )
    video_steps = 0
    if args.video:
        import gymnasium as gym

        video_steps = max(1, round(args.video_seconds / raw_env.step_dt))
        video_dir = os.path.abspath(args.video_dir)
        os.makedirs(video_dir, exist_ok=True)
        name_prefix = f'{args.task}_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}'
        raw_env = gym.wrappers.RecordVideo(
            raw_env,
            video_folder=video_dir,
            step_trigger=lambda step: step == 0,
            video_length=video_steps,
            name_prefix=name_prefix,
            disable_logger=True,
        )
        print(
            f'[INFO] Recording {video_steps} policy steps ({args.video_seconds:g} s) to {video_dir}',
            flush=True,
        )
    env = HoraCompatWrapper(raw_env)

    # By default Stage2 warm-start and resume stay beside the source checkpoint.
    # An explicit output name still creates a separate run directory.
    if args.algo == 'ProprioAdapt' and _auto_output_name:
        output_dif = os.path.dirname(os.path.dirname(os.path.abspath(args.checkpoint)))
    else:
        output_dif = os.path.join('outputs', 'revo3_right', args.output_name)
    os.makedirs(output_dif, exist_ok=True)
    algo_name = str(full_config.train.algo)
    if algo_name not in _ALGO_MAP:
        raise ValueError(f"Unsupported algo: {algo_name}. Available: {list(_ALGO_MAP.keys())}")
    agent = _ALGO_MAP[algo_name](env, output_dif, full_config=full_config)

    if args.test:
        try:
            agent.restore_test(full_config.train.load_path)
            agent.test(
                max_steps=video_steps if args.video else args.test_steps,
                real_time=args.real_time,
            )
        finally:
            if args.video:
                env.close()
    else:
        best_ckpt_path = os.path.join(
            output_dif,
            'stage1_nn' if full_config.train.algo == 'PPO' else 'stage2_nn',
            'best.pth' if full_config.train.algo == 'PPO' else 'model_best.ckpt',
        )
        stage2_resume = (
            full_config.train.algo == 'ProprioAdapt'
            and _is_stage2_checkpoint(args.checkpoint)
        )
        if os.path.exists(best_ckpt_path) and not stage2_resume:
            if args.force_overwrite:
                print(f"[INFO] --force_overwrite enabled, continue and overwrite in {output_dif}", flush=True)
            else:
                user_input = input(
                    f'are you intentionally going to overwrite files in {output_dif}, type yes to continue \n'
                )
                if user_input != 'yes':
                    return

            # A fresh PPO run must not leave an old, pre-gating best.pth in
            # place: Stage2 could otherwise consume it before the new policy
            # completes its 9.81 m/s² evaluation. Preserve it as a backup.
            if full_config.train.algo == 'PPO' and not args.checkpoint:
                timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                archived_path = best_ckpt_path.replace('.pth', f'.pre_retrain_{timestamp}.pth')
                os.replace(best_ckpt_path, archived_path)
                print(f"[INFO] Archived stale best checkpoint to: {archived_path}", flush=True)
        elif stage2_resume:
            print(f"[INFO] Resuming Stage2 in existing run directory: {output_dif}", flush=True)

        _attach_env_runtime_to_config(full_config, env_cfg)
        _save_run_metadata(output_dif, full_config)
        agent.restore_train(full_config.train.load_path)
        agent.train()


if __name__ == '__main__':
    try:
        main()
    except Exception:
        print("\n[ERROR] Training terminated with an exception. Full traceback:", flush=True)
        traceback.print_exc()
        raise
    finally:
        if os.getenv("HORA_SKIP_SIM_CLOSE", "0") == "1":
            print("[INFO] Skip simulation_app.close() due to HORA_SKIP_SIM_CLOSE=1", flush=True)
        else:
            simulation_app.close()
