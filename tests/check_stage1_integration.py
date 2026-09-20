"""Bounded Isaac Lab regression; all PPO outputs go to a temporary directory.

Run: ~/IsaacLab/isaaclab.sh -p tests/check_stage1_integration.py
"""
import hashlib
from pathlib import Path
import sys
import tempfile
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from isaaclab.app import AppLauncher

app = AppLauncher(headless=True).app
import torch
from omegaconf import OmegaConf
from hora.algo.ppo.ppo import PPO
from hora.tasks.isaaclab import HoraCompatWrapper, Revo3HandHoraEnv, Revo3HandHoraEnvCfg
from hora.tasks.isaaclab.assets import configure_env_for_object_task
from hora.utils.privileged_observations import object_rotation_6d

env = None
agent = None
exit_code = 1
try:
    cfg = Revo3HandHoraEnvCfg()
    configure_env_for_object_task(cfg, 'strawberry')
    cfg.scene.num_envs = 8
    cfg.prop_hist_len = 3
    cfg.priv_info_dim = 24
    cfg.sim.render_interval = cfg.decimation
    cfg.sim.gravity = (0.0, 0.0, -9.81)
    cfg.gravity_curriculum = False
    cfg.finger_gait = True
    cfg.joint_noise_scale = 0.0
    cache_path = ROOT / f'{cfg.grasp_cache_path}.npy'
    cache_hash = hashlib.sha256(cache_path.read_bytes()).hexdigest()
    env = Revo3HandHoraEnv(cfg)
    wrapper = HoraCompatWrapper(env)
    # no_grad rather than inference_mode: PPO backprop must consume these buffers.
    with torch.no_grad():
        obs = wrapper.reset()
        for asset in (env.hand, env.object):
            mats = asset.root_physx_view.get_material_properties()
            expected = .5 * env.priv_info_buf[:, 3].cpu()
            torch.testing.assert_close(mats[..., 0], expected[:, None].expand_as(mats[..., 0]))
            torch.testing.assert_close(mats[..., 1], mats[..., 0])
        assert obs['obs'].shape == (8, 141)
        assert obs['priv_info'].shape == (8, 24)
        torch.testing.assert_close(obs['priv_info'][:, 18:], object_rotation_6d(env.object_rot))
        assert obs['proprio_hist'].shape == (8, 3, 47)
        actions = torch.zeros(8, 21, device=env.device)
        for _ in range(3):
            obs, _, _, _ = wrapper.step(actions)
        assert torch.isfinite(env.gait_angle).all()
        gait_before = env.gait_angle.clone()
        env._get_observations()
        torch.testing.assert_close(env.gait_angle, gait_before)
        assert 'gait/support_gate' in env.extras
        previous_history = env.proprio_hist_buf.clone()
        previous_index = env._obs_history_index
        pose = env.object.data.root_state_w[:, :7].clone()
        pose[1:3, 2] += .10
        env.object.write_root_pose_to_sim(pose)
        env.episode_length_buf[:2] = env.max_episode_length - 1
        obs, reward, done, info = wrapper.step(actions)
        assert info['time_outs'].tolist() == [True, False, False, False, False, False, False, False]
        assert bool(env.reset_terminated[1]) and bool(env.reset_time_outs[1])
        assert bool(env.reset_terminated[2]) and not bool(env.reset_time_outs[2])
        assert not env.gait_angle[:3].any()
        assert not env.gait_limit_age[:3].any()
        assert not env.gait_contacts[:3].any()
        terminal = info['terminal_observation']
        assert terminal['priv_info'].shape == (1, 24)
        torch.testing.assert_close(terminal['priv_info'][:, 18:].reshape(-1, 2, 3).norm(dim=-1), torch.ones(1, 2, device=env.device))
        torch.testing.assert_close(obs['priv_info'][:, 18:], object_rotation_6d(env.object_rot))
        assert terminal['env_ids'].tolist() == [0]
        final_history = terminal['obs'].reshape(1, 3, 47)
        torch.testing.assert_close(final_history[0, :2], previous_history[0, -2:])
        assert env._obs_history_index == (previous_index + 1) % 3
        torch.testing.assert_close(obs['obs'][3].reshape(3, 47)[:2], previous_history[3, -2:])
        assert float(terminal['priv_info'][0, :3].norm()) > 1e-6
        assert float(obs['priv_info'][0, :3].norm()) < 1e-5
        terminal_copy = terminal['obs'].clone()
        _, _, _, info = wrapper.step(actions)
        assert info['terminal_observation'] is None
        torch.testing.assert_close(terminal['obs'], terminal_copy)
        print('[PASS] Friction, timeout/drop masks, pre-reset state, chronological history, and stale snapshot clearing.', flush=True)

    train_cfg = OmegaConf.load(ROOT / 'configs/train/Revo3HandHora.yaml')
    train_cfg.ppo.num_actors = 8
    train_cfg.ppo.priv_info_dim = cfg.priv_info_dim
    train_cfg.ppo.minibatch_size = 64
    train_cfg.ppo.horizon_length = 16
    full_cfg = OmegaConf.create({'rl_device': env.device, 'test': False, 'train': train_cfg})
    with tempfile.TemporaryDirectory(prefix='strawberry-ppo-regression-') as output:
        agent = PPO(wrapper, output, full_cfg)
        # Frequent time limits exercise the bootstrap path throughout the update.
        env.cfg.episode_length_s = .15
        agent.obs = wrapper.reset()
        before = agent.model.mu.weight.detach().clone()
        stats = agent.train_epoch()
        assert agent.agent_steps == 128
        assert agent.value_mean_std.count.item() == 129  # One return fit, not two.
        assert not torch.equal(before, agent.model.mu.weight)
        assert all(torch.isfinite(p).all() for p in agent.model.parameters())
        assert all(torch.isfinite(loss) for group in stats[:5] for loss in group)
        agent.writer.close()
        print('[PASS] PPO rollout + 3 optimization passes; finite gradients/losses/parameters, correct agent-step and RMS counts.', flush=True)

    with torch.no_grad():
        env.cfg.episode_length_s = 20.0
        env.scene.env_origins[0, 2] = .25
        env._reset_idx(torch.tensor([0], device=env.device))
        torch.testing.assert_close(env.reset_height_lower[0], env.object_pos[0, 2] - .02)
        torch.testing.assert_close(env.reset_height_upper[0], env.object_pos[0, 2] + .02)
        env.cfg.enable_contact_in_obs = False
        env.episode_length_buf[0] = env.max_episode_length - 1
        _, _, _, info = wrapper.step(actions)
        assert not info['terminal_observation']['obs'].reshape(-1, 3, 47)[:, :, 42:].any()
        print('[PASS] Nonzero-Z environment origin and no-tactile terminal observation.', flush=True)
    assert hashlib.sha256(cache_path.read_bytes()).hexdigest() == cache_hash
    print('[PASS] Original strawberry cache unchanged.', flush=True)
    exit_code = 0
except BaseException:
    traceback.print_exc()
    raise
finally:
    if agent is not None:
        agent.writer.close()
    if env is not None:
        env.close()
    import omni.kit.app
    omni.kit.app.get_app().post_quit(exit_code)
    app.close()
