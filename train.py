#!/usr/bin/env python3
"""Training entry point for Stage1 (PPO) and Stage2 (ProprioAdapt).

Task selection: --task selects robot_cfg, object_cfg, reward constraints, and grasp cache
from the shared object registry.  This includes ball/cylinder and packaged USD objects.

Cache path: {grasp_cache_path}.npy under cache/. Override with --cache_file.

Gotcha — num_envs × horizon_length must be >= minibatch_size and exactly
  divisible by it. Override --minibatch_size for smaller diagnostic runs.

Gotcha — tactile deployment: Stage1, Stage2 actor obs, and Stage2
  proprio_hist all retain the same five fingertip contact-force channels.
"""

import argparse
import copy
import datetime
import hashlib
import math
import os
import subprocess
import traceback

os.environ.setdefault("HORA_SKIP_SIM_CLOSE", "1")

from isaaclab.app import AppLauncher
from hora.object_registry import OBJECT_TASK_NAMES, get_object_task_spec


parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, default='cylinder', choices=OBJECT_TASK_NAMES)
parser.add_argument('--algo', type=str, default='PPO', choices=['PPO', 'ProprioAdapt'])
parser.add_argument('--train_cfg', type=str, default='Revo3HandHora')
parser.add_argument('--output_name', type=str, default='debug')
parser.add_argument('--checkpoint', type=str, default='')
parser.add_argument('--object_orientation', choices=('auto', 'none', 'rotation6d'), default='auto',
                    help='Stage1 privileged object orientation. Auto enables 6D orientation for new duck '
                         'policies and duck weights-only warm starts; eval/strict resume/Stage2 follow the checkpoint.')
parser.add_argument(
    '--weights_only',
    action='store_true',
    help=(
        'Warm-start PPO from checkpoint actor/observation normalization only; reset the '
        'value head, value normalization, optimizer, counters and best-checkpoint scores.'
    ),
)
parser.add_argument('--cache_file', type=str, default='', help='Override grasp cache filename under cache/.')
parser.add_argument('--usd', type=str, default='', help='Override hand USD path.')
parser.add_argument('--num_envs', type=int, default=None, help='Default: 2048 for PPO, 16384 for Stage2.')
parser.add_argument('--minibatch_size', type=int, default=None, help='PPO minibatch override; must divide num_envs * horizon_length.')
parser.add_argument('--horizon_length', type=int, default=None, help='PPO rollout steps per environment.')
parser.add_argument('--gamma', type=float, default=None, help='PPO reward discount in (0, 1).')
parser.add_argument('--episode_length_s', type=float, default=None, help='Episode time limit in seconds; also applies to evaluation.')
parser.add_argument('--physics_hz', type=int, choices=(120, 240), default=240,
                    help='Physics/PD frequency; decimation is adjusted to keep the policy at 20 Hz.')
parser.add_argument('--seed', type=int, default=42)
budget_args = parser.add_mutually_exclusive_group()
budget_args.add_argument(
    '--max_agent_steps', type=int, default=None,
    help='Override train.ppo.max_agent_steps (useful when resuming beyond the original training budget).',
)
budget_args.add_argument('--max_iterations', type=int, default=None,
                         help='Total PPO iterations; budget is num_envs * horizon_length * max_iterations.')
parser.add_argument('--save_frequency', type=int, default=None,
                    help='Save a numbered checkpoint and last.pth every N PPO iterations.')
gravity_args = parser.add_mutually_exclusive_group()
gravity_args.add_argument(
    '--fixed_train_gravity',
    type=float,
    default=None,
    metavar='M_S2',
    help=(
        'Lock PPO training to this downward gravity magnitude. The value is reapplied after '
        'checkpoint restore so a saved curriculum gravity cannot override it.'
    ),
)
gravity_args.add_argument(
    '--initial_train_gravity', type=float, default=None, metavar='M_S2',
    help='Explicit PPO curriculum starting gravity for fresh/weights-only runs; overrides the strawberry full-gravity default.',
)
parser.add_argument('--test', action='store_true')
parser.add_argument('--finger_gait', action='store_true',
                    help='Opt in to support-gated rotation, persistent blocked-action cost and full-turn bonus (PPO).')
parser.add_argument('--rotation_speed', type=float, default=None,
                    help='Positive speed target (rad/s) for --finger_gait; includes overspeed attenuation.')
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

if args.finger_gait and args.algo != 'PPO':
    parser.error('--finger_gait is currently a PPO Stage1 reward option')
if args.rotation_speed is not None:
    if not args.finger_gait or not math.isfinite(args.rotation_speed) or args.rotation_speed <= 0:
        parser.error('--rotation_speed requires --finger_gait and a finite positive value')

if args.num_envs is None:
    args.num_envs = 2048 if args.algo == 'PPO' else 16384
if args.num_envs <= 0:
    parser.error('--num_envs must be positive')
if args.minibatch_size is not None and args.minibatch_size <= 0:
    parser.error('--minibatch_size must be positive')
if args.horizon_length is not None and args.horizon_length <= 0:
    parser.error('--horizon_length must be positive')
if args.gamma is not None and (not math.isfinite(args.gamma) or not 0 < args.gamma < 1):
    parser.error('--gamma must be finite and in (0, 1)')
if (args.horizon_length is not None or args.gamma is not None) and args.algo != 'PPO':
    parser.error('--horizon_length and --gamma are PPO options')
if args.episode_length_s is not None and (not math.isfinite(args.episode_length_s) or args.episode_length_s <= 0):
    parser.error('--episode_length_s must be finite and positive')
if args.max_iterations is not None and (args.max_iterations <= 0 or args.algo != 'PPO'):
    parser.error('--max_iterations requires PPO and a positive iteration count')
if args.save_frequency is not None and (args.save_frequency <= 0 or args.algo != 'PPO'):
    parser.error('--save_frequency requires PPO and a positive interval')
if args.initial_train_gravity is not None:
    if not math.isfinite(args.initial_train_gravity) or not 0.0 < args.initial_train_gravity <= 9.81:
        parser.error('--initial_train_gravity must be finite and in (0, 9.81]')
    if args.test or args.algo != 'PPO' or (args.checkpoint and not args.weights_only):
        parser.error('--initial_train_gravity is only for fresh/weights-only PPO training')
if args.task == 'strawberry' and args.algo == 'PPO' and not args.test:
    if args.fixed_train_gravity is None and args.initial_train_gravity is None:
        args.fixed_train_gravity = 9.81
        print('[INFO] Strawberry Stage1 defaults to fixed 9.81 m/s^2, matching grasp collection.', flush=True)
if args.fixed_train_gravity is not None:
    if not math.isfinite(args.fixed_train_gravity) or args.fixed_train_gravity <= 0.0:
        parser.error('--fixed_train_gravity must be finite and greater than 0')
    if args.test:
        parser.error('--fixed_train_gravity is for training; --test already evaluates at 9.81 m/s^2')
    if args.algo != 'PPO':
        parser.error('--fixed_train_gravity is only supported for PPO Stage1 training')

if args.weights_only:
    if not args.checkpoint:
        parser.error('--weights_only requires --checkpoint')
    if args.test:
        parser.error('--weights_only is only supported for training')
    if args.algo != 'PPO':
        parser.error('--weights_only is only supported for PPO Stage1 training')

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
    configure_env_for_object_task,
)
from hora.utils.misc import set_np_formatting, set_seed
from hora.utils.privileged_observations import checkpoint_privileged_dim, resolve_privileged_dim


_ALGO_MAP = {
    'PPO': PPO,
    'ProprioAdapt': ProprioAdapt,
}

def _build_full_config(seed: int):
    cfg_path = os.path.join(os.path.dirname(__file__), 'configs', 'train', f'{args.train_cfg}.yaml')
    train_cfg = OmegaConf.load(cfg_path)
    train_cfg.algo = args.algo
    train_cfg.load_path = os.path.abspath(args.checkpoint) if args.checkpoint else ''
    train_cfg.resume_mode = 'weights_only' if args.weights_only else 'strict'
    train_cfg.ppo.output_name = args.output_name
    if args.horizon_length is not None:
        train_cfg.ppo.horizon_length = args.horizon_length
    if args.gamma is not None:
        train_cfg.ppo.gamma = args.gamma
    if args.minibatch_size is not None:
        train_cfg.ppo.minibatch_size = args.minibatch_size
    minibatch = train_cfg.ppo.minibatch_size
    min_envs = math.ceil(minibatch / train_cfg.ppo.horizon_length)
    if (
        not args.test
        and args.algo == 'PPO'
        and (args.num_envs < min_envs or (args.num_envs * train_cfg.ppo.horizon_length) % minibatch != 0)
    ):
        raise ValueError(
            f"num_envs ({args.num_envs}) must be >= {min_envs} and num_envs*horizon must be divisible "
            f"by minibatch_size ({minibatch}); use --minibatch_size for smaller runs."
        )
    train_cfg.ppo.num_actors = args.num_envs
    if args.rotation_speed is not None:
        train_cfg.ppo.full_gravity_min_target_angvel = .5 * args.rotation_speed
    if args.max_agent_steps is not None:
        if args.max_agent_steps <= 0:
            raise ValueError('--max_agent_steps must be positive')
        train_cfg.ppo.max_agent_steps = args.max_agent_steps
    if args.max_iterations is not None:
        train_cfg.ppo.max_iterations = args.max_iterations
        train_cfg.ppo.max_agent_steps = args.num_envs * train_cfg.ppo.horizon_length * args.max_iterations
    if args.save_frequency is not None:
        train_cfg.ppo.save_frequency = args.save_frequency
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
    configure_env_for_object_task(env_cfg, args.task)
    env_cfg.sim.dt = 1.0 / args.physics_hz
    env_cfg.decimation = args.physics_hz // 20
    if args.episode_length_s is not None:
        env_cfg.episode_length_s = args.episode_length_s
    env_cfg.finger_gait = args.finger_gait
    if args.rotation_speed is not None:
        env_cfg.target_angvel = args.rotation_speed
        env_cfg.stable_rotation_min_angvel = .5 * args.rotation_speed
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
    if args.algo == 'PPO':
        env_cfg.prop_hist_len = 3  # Only Stage2 needs the 30-frame adaptation history.
    if not args.test and not os.path.isfile(f'{env_cfg.grasp_cache_path}.npy'):
        raise FileNotFoundError(f'Training requires grasp cache: {env_cfg.grasp_cache_path}.npy')
    if args.headless:
        env_cfg.sim.render_interval = env_cfg.decimation

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
    object_spec = get_object_task_spec(env_cfg.object_task)
    with open(f'{env_cfg.grasp_cache_path}.npy', 'rb') as cache_file:
        cache_sha256 = hashlib.file_digest(cache_file, 'sha256').hexdigest()
    full_config.env_runtime = OmegaConf.create(
        {
            'object': object_spec.metadata(),
            'grasp_cache_path': str(env_cfg.grasp_cache_path),
            'grasp_cache_sha256': cache_sha256,
            'enable_tactile': bool(env_cfg.enable_tactile),
            'enable_contact_in_obs': bool(env_cfg.enable_contact_in_obs),
            'privileged_observation': {
                'dim': int(env_cfg.priv_info_dim),
                'object_orientation': 'rotation6d_world_xy' if env_cfg.priv_info_dim == 24 else 'none',
            },
            'contact_order': ['thumb_DIP', 'index_DIP', 'middle_DIP', 'ring_DIP', 'little_DIP'],
            'policy_dt': float(env_cfg.decimation * env_cfg.sim.dt),
            'physics_dt': float(env_cfg.sim.dt),
            'decimation': int(env_cfg.decimation),
            'episode_length_s': float(env_cfg.episode_length_s),
            'gravity': tuple(float(v) for v in env_cfg.sim.gravity),
            'action_scale': float(env_cfg.action_scale),
            'force_scale': float(env_cfg.force_scale),
            'finger_gait': {
                'enabled': bool(env_cfg.finger_gait),
                'version': 2,
                'blocked_push_scale': float(env_cfg.gait_blocked_push_scale),
                'limit_grace_steps': int(env_cfg.gait_limit_grace_steps),
                'limit_recovery_margin': float(env_cfg.gait_limit_recovery_margin),
                'contact_debounce_steps': int(env_cfg.gait_contact_debounce_steps),
                'speed_reward': 'peaked' if env_cfg.finger_gait else 'saturating',
                'safe_z_m': float(env_cfg.gait_safe_z_m),
                'max_z_m': float(env_cfg.gait_max_z_m),
                'max_down_speed': float(env_cfg.gait_max_down_speed),
                'full_turn_bonus': float(env_cfg.gait_full_turn_bonus),
                'contact_threshold_n': float(env_cfg.contact_threshold),
                'axis_tilt_tolerance': float(env_cfg.object_axis_tilt_tolerance),
                'action_scale': float(env_cfg.action_scale),
                'rotate_scale': float(env_cfg.rotate_reward_scale),
                'target_angvel': float(env_cfg.target_angvel),
            },
            'reward': {
                'target_angvel': float(env_cfg.target_angvel),
                'stable_rotation_min_angvel': float(env_cfg.stable_rotation_min_angvel),
                'rotate': float(env_cfg.rotate_reward_scale),
                'stable_rotation_bonus': float(env_cfg.stable_rotation_bonus_scale),
                'alive': float(env_cfg.alive_reward_scale),
                'drop': float(env_cfg.drop_penalty_scale),
                'xy_drift_deadzone_m': float(env_cfg.xy_drift_deadzone),
                'self_collision_force_threshold_n': float(env_cfg.self_collision_force_threshold),
                'self_collision_force_tolerance_n': float(env_cfg.self_collision_force_tolerance),
                'self_collision': float(env_cfg.self_collision_penalty_scale),
                'torque_normalization': float(env_cfg.torque_normalization),
                'torque': float(env_cfg.torque_penalty_scale),
                'work': float(env_cfg.work_penalty_scale),
            },
            'domain_randomization': {
                'friction_mode': 'multiply_authored_static_dynamic',
                'pd_gain_scale': (
                    float(env_cfg.randomize_p_gain_scale_lower),
                    float(env_cfg.randomize_p_gain_scale_upper),
                ),
                'friction_scale': (
                    float(env_cfg.randomize_friction_scale_lower),
                    float(env_cfg.randomize_friction_scale_upper),
                ),
                'com_m': (float(env_cfg.randomize_com_lower), float(env_cfg.randomize_com_upper)),
                'mass_kg': (float(env_cfg.randomize_mass_lower), float(env_cfg.randomize_mass_upper)),
            },
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
    checkpoint_dim = None
    if args.checkpoint:
        import torch
        checkpoint_dim = checkpoint_privileged_dim(torch.load(args.checkpoint, map_location='cpu', weights_only=True))
    env_cfg.priv_info_dim = resolve_privileged_dim(
        args.object_orientation, args.task, checkpoint_dim, args.weights_only)
    full_config.train.ppo.priv_info_dim = env_cfg.priv_info_dim
    print(f'[INFO] Privileged observation: {env_cfg.priv_info_dim} dims; '
          f'object orientation={"rotation6d_world_xy" if env_cfg.priv_info_dim == 24 else "none"}', flush=True)
    if args.initial_train_gravity is not None:
        env_cfg.sim.gravity = (0.0, 0.0, -float(args.initial_train_gravity))
    if args.camera_eye is not None:
        env_cfg.viewer.eye = tuple(args.camera_eye)
    if args.camera_lookat is not None:
        env_cfg.viewer.lookat = tuple(args.camera_lookat)
    if args.fixed_train_gravity is not None:
        fixed_gravity = float(args.fixed_train_gravity)
        # Keep the curriculum tracker enabled so PPO still receives a rolling
        # height-reset rate and can gate best_full_gravity checkpoints. Setting
        # both the floor (sim.gravity) and target to the same value makes the
        # physical gravity immutable even when the tracker requests a rollback.
        env_cfg.gravity_curriculum = True
        env_cfg.gravity_curriculum_target = fixed_gravity
        env_cfg.sim.gravity = (0.0, 0.0, -fixed_gravity)
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
            _attach_env_runtime_to_config(full_config, env_cfg)
            agent.env_runtime = OmegaConf.to_container(full_config.env_runtime, resolve=True)
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
        agent.env_runtime = OmegaConf.to_container(full_config.env_runtime, resolve=True)
        if args.weights_only:
            agent.restore_weights_only(full_config.train.load_path)
        else:
            agent.restore_train(full_config.train.load_path)
        if args.algo == 'PPO' and args.max_iterations is not None:
            remaining = max(0, args.max_iterations - agent.epoch_num)
            agent.max_agent_steps = agent.agent_steps + remaining * agent.batch_size
            full_config.train.ppo.max_agent_steps = agent.max_agent_steps
        _save_run_metadata(output_dif, full_config)
        if args.fixed_train_gravity is not None:
            fixed_gravity = float(args.fixed_train_gravity)
            base_env = env._base_env
            # Strict resume restores the checkpoint's curriculum gravity
            # (0.05 for the sapota run). The explicit training override must
            # win after all checkpoint state has been loaded.
            base_env.set_gravity_magnitude(fixed_gravity)
            base_env._gravity_reset_sum.zero_()
            base_env._gravity_window_steps = 0
            base_env._gravity_window_reset_rate = 1.0
            if hasattr(agent, 'full_gravity_epochs'):
                agent.full_gravity_epochs = 0
            print(
                f'[INFO] Fixed PPO training gravity: {fixed_gravity:.3f} m/s^2 downward '
                '(checkpoint gravity overridden; curriculum metrics retained)',
                flush=True,
            )
        if args.algo == 'PPO' and env_cfg.gravity_curriculum:
            # Lower bound only: assume every curriculum window succeeds, and
            # account for full-gravity qualification. Warn, but allow bounded
            # smoke runs that intentionally cannot produce a best checkpoint.
            base_env = env._base_env
            target = max(env_cfg.gravity_curriculum_target, agent.full_gravity_magnitude - agent.full_gravity_tolerance)
            if env_cfg.gravity_curriculum_target < agent.full_gravity_magnitude - agent.full_gravity_tolerance:
                print('[WARN] Configured gravity target is below the full-gravity best-checkpoint gate.', flush=True)
            gravity_windows = math.ceil(max(0.0, target - base_env._gravity_magnitude) / env_cfg.gravity_curriculum_step)
            policy_steps = env_cfg.gravity_curriculum_warmup_steps + max(1, gravity_windows) * env_cfg.gravity_curriculum_window
            policy_steps += agent.full_gravity_eval_epochs * agent.horizon_length
            required_steps = policy_steps * env.num_envs
            if agent.max_agent_steps - agent.agent_steps < required_steps:
                print(f'[WARN] Remaining budget is below the optimistic full-gravity qualification bound ({required_steps:,} agent steps); best.pth may not be produced.', flush=True)
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
