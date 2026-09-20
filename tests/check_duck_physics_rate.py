"""Paired physics-rate validation: replay identical initial grasps and actions.

Run 240 Hz first with --reference, then 120 Hz without it, using the same directory.
Both rates keep 20 Hz policy control, authored materials, nominal dynamics and PD gains.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument('--physics_hz', type=int, choices=(120, 240), required=True)
parser.add_argument('--reference', action='store_true')
parser.add_argument('--checkpoint', required=True)
parser.add_argument('--result_dir', required=True)
parser.add_argument('--num_envs', type=int, default=512)
parser.add_argument('--randomized', action='store_true', help='Use the training dynamics randomization ranges.')
parser.add_argument('--seed', type=int, default=12345)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import numpy as np
import torch
from omegaconf import OmegaConf
from hora.algo.models.models import ActorCritic
from hora.algo.models.running_mean_std import RunningMeanStd
from hora.tasks.isaaclab import HoraCompatWrapper, Revo3HandHoraEnv, Revo3HandHoraEnvCfg
from hora.tasks.isaaclab.assets import configure_env_for_object_task
from hora.utils.privileged_observations import checkpoint_privileged_dim


def main():
    destination = Path(args.result_dir)
    destination.mkdir(parents=True, exist_ok=True)
    cfg = Revo3HandHoraEnvCfg()
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    cfg.priv_info_dim = checkpoint_privileged_dim(checkpoint)
    configure_env_for_object_task(cfg, 'rubber_duck')
    cfg.scene.num_envs = args.num_envs
    cfg.sim.device = args.device
    cfg.sim.dt = 1 / args.physics_hz
    cfg.decimation = args.physics_hz // 20
    cfg.sim.render_interval = cfg.decimation
    cfg.sim.gravity = (0., 0., -9.81)
    cfg.gravity_curriculum = False
    cfg.episode_length_s = 20
    cfg.finger_gait = True
    cfg.target_angvel = .5
    cfg.stable_rotation_min_angvel = .25
    cfg.prop_hist_len = 3
    cfg.grasp_cache_path = 'cache/revo3_right_grasp_rubber_duck_gait_v2'
    cfg.joint_noise_scale = 0
    cfg.randomize_pd_gains = cfg.randomize_friction = cfg.randomize_com = cfg.randomize_mass = args.randomized
    cfg.contact_sensor_noise = cfg.contact_latency = 0
    cfg.force_scale = 0
    cfg.seed = args.seed
    env = Revo3HandHoraEnv(cfg)
    wrapper = HoraCompatWrapper(env)
    assert abs(env.step_dt - .05) < 1e-9
    net = OmegaConf.load(ROOT / 'configs/train/Revo3HandHora.yaml').network
    model = ActorCritic(dict(actions_num=21, input_shape=(141,), actor_units=list(net.mlp.units),
                            priv_mlp_units=list(net.priv_mlp.units), priv_info=True,
                            proprio_adapt=False, priv_info_dim=cfg.priv_info_dim)).to(env.device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    rms = RunningMeanStd((141,)).to(env.device)
    rms.load_state_dict(checkpoint['running_mean_std'])
    rms.eval()
    report = dict(physics_hz=args.physics_hz, policy_hz=20, num_envs=args.num_envs,
                  priv_info_dim=cfg.priv_info_dim,
                  randomized=args.randomized, seed=args.seed,
                  checkpoint_sha256=hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
                  cache_sha256=hashlib.sha256(Path(cfg.grasp_cache_path + '.npy').read_bytes()).hexdigest(),
                  phases={})
    with torch.no_grad():
        for phase, steps in [('hold', 100), ('replay', 200)]:
            torch.manual_seed(args.seed)
            np.random.seed(args.seed)
            obs = wrapper.reset()
            initial = torch.cat([env.grasp_joint_pos, env.object_default_pose], -1).cpu().numpy()
            fingerprint = hashlib.sha256(initial.tobytes()).hexdigest()
            dynamics = torch.cat([env.p_gain, env.d_gain, env.priv_info_buf[:, 3:8]], -1).cpu().numpy()
            dynamics_fingerprint = hashlib.sha256(dynamics.tobytes()).hexdigest()
            recording = destination / f'{phase}_actions.npy'
            if not args.reference:
                actions_recorded = np.load(recording, allow_pickle=False)
                assert actions_recorded.shape == (steps, args.num_envs, 21)
            else:
                actions_recorded = []
            alive = torch.ones(args.num_envs, dtype=torch.bool, device=env.device)
            first_drop = torch.full((args.num_envs,), steps * .05, device=env.device)
            tracking_sum = torch.zeros((), device=env.device)
            force_change_sum = torch.zeros_like(tracking_sum)
            switches = torch.zeros_like(tracking_sum)
            count = torch.zeros_like(tracking_sum)
            previous_force = torch.zeros((args.num_envs, 5), device=env.device)
            max_force = torch.zeros_like(tracking_sum)
            torch.cuda.synchronize(env.device)
            started = time.perf_counter()
            for step in range(steps):
                if args.reference:
                    action = (torch.zeros((args.num_envs, 21), device=env.device) if phase == 'hold'
                              else model.act_inference({'obs': rms(obs['obs']), 'priv_info': obs['priv_info']}).clamp(-1, 1))
                    actions_recorded.append(action.cpu().numpy())
                else:
                    action = torch.from_numpy(actions_recorded[step]).to(env.device)
                obs, _, _, _ = wrapper.step(action)
                failed = env.reset_terminated.bool()
                newly_failed = alive & failed
                first_drop[newly_failed] = (step + 1) * .05
                alive &= ~failed
                force = env._raw_object_contact_forces()
                error = (env.cur_targets - env.hand_dof_pos).square().mean(-1)
                assert torch.isfinite(force).all() and torch.isfinite(error).all(), 'Nonfinite dynamics'
                # Exclude first 1 s settling and all frames after the first failure/reset.
                valid = alive & (step >= 20)
                tracking_sum += error[valid].sum()
                force_change_sum += (force - previous_force).abs().mean(-1)[valid].sum()
                switches += ((force > .05) != (previous_force > .05)).float().mean(-1)[valid].sum()
                count += valid.sum()
                if valid.any():
                    max_force = torch.maximum(max_force, force[valid].max())
                previous_force = force.clone()
            torch.cuda.synchronize(env.device)
            elapsed = time.perf_counter() - started
            if args.reference:
                np.save(recording, np.stack(actions_recorded))
            report['phases'][phase] = dict(initial_sha256=fingerprint, dynamics_sha256=dynamics_fingerprint,
                survival=float(alive.float().mean()), first_drop_mean_s=float(first_drop.mean()),
                tracking_rms_rad=float((tracking_sum / count.clamp_min(1)).sqrt()),
                force_change_mean_n=float(force_change_sum / count.clamp_min(1)),
                contact_switch_fraction=float(switches / count.clamp_min(1)),
                max_force_n=float(max_force), valid_frames=int(count), elapsed_s=elapsed)
            print(phase, json.dumps(report['phases'][phase]), flush=True)
    if not args.reference:
        reference = json.loads((destination / 'rate_240.json').read_text())
        assert reference['cache_sha256'] == report['cache_sha256']
        assert reference['checkpoint_sha256'] == report['checkpoint_sha256']
        checks = {}
        for phase in report['phases']:
            a, b = reference['phases'][phase], report['phases'][phase]
            checks[phase] = dict(
                identical_initial_state=a['initial_sha256'] == b['initial_sha256'],
                identical_dynamics=a['dynamics_sha256'] == b['dynamics_sha256'],
                survival=b['survival'] >= a['survival'] - .02,
                first_drop=b['first_drop_mean_s'] >= .95 * a['first_drop_mean_s'],
                tracking=b['tracking_rms_rad'] <= 1.25 * a['tracking_rms_rad'] + 1e-4,
                force_change=b['force_change_mean_n'] <= 1.3 * a['force_change_mean_n'] + .005,
                contact_switch=b['contact_switch_fraction'] <= 1.3 * a['contact_switch_fraction'] + .005,
                force_peak=b['max_force_n'] <= 2 * a['max_force_n'] + .1,
                enough_frames=b['valid_frames'] >= .9 * a['valid_frames'] and a['valid_frames'] > 0)
        report['acceptance_checks'] = checks
        report['passed'] = all(all(check.values()) for check in checks.values())
    (destination / f'rate_{args.physics_hz}.json').write_text(json.dumps(report, indent=2) + '\n')
    print('PHYSICS_RATE_REPORT_SAVED', flush=True)
    env.close()


try:
    main()
finally:
    app.close()
