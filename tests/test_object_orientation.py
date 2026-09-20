"""Orientation encoding, legacy ABI, and functional warm-start regressions."""
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import torch
from omegaconf import OmegaConf

from hora.algo.ppo.ppo import PPO
from hora.utils.privileged_observations import (
    object_rotation_6d, checkpoint_privileged_dim, resolve_privileged_dim,
    warmstart_privileged_state, PRIV_INPUT_WEIGHT,
)


class OrientationTests(unittest.TestCase):
    def test_absolute_yaw_axes_and_quaternion_sign(self):
        a = math.sqrt(.5)
        q = torch.tensor([[1., 0, 0, 0], [a, 0, 0, a], [0., 0, 0, 1.]])
        expected = torch.tensor([[1., 0, 0, 0, 1, 0], [0., 1, 0, -1, 0, 0], [-1., 0, 0, 0, -1, 0]])
        torch.testing.assert_close(object_rotation_6d(q), expected, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(object_rotation_6d(q), object_rotation_6d(-q))
        torch.testing.assert_close(object_rotation_6d(q), object_rotation_6d(3*q))

    def test_pi_boundary_is_continuous_and_axes_are_orthonormal(self):
        yaw = torch.tensor([math.pi - 1e-5, -math.pi + 1e-5])
        q = torch.stack([torch.cos(yaw/2), torch.zeros(2), torch.zeros(2), torch.sin(yaw/2)], -1)
        f = object_rotation_6d(q)
        self.assertLess(float((f[0]-f[1]).abs().max()), 3e-5)
        q = torch.randn(100, 4)
        f = object_rotation_6d(q)
        x, y = f[:, :3], f[:, 3:]
        torch.testing.assert_close(x.norm(dim=-1), torch.ones(100))
        torch.testing.assert_close(y.norm(dim=-1), torch.ones(100))
        torch.testing.assert_close((x*y).sum(-1), torch.zeros(100), atol=1e-6, rtol=0)

    def test_layout_selection_and_legacy_metadata(self):
        self.assertEqual(resolve_privileged_dim('auto', 'rubber_duck'), 24)
        self.assertEqual(resolve_privileged_dim('auto', 'cylinder'), 18)
        for task in ('rubber_duck', 'cylinder'):
            for dim in (18, 24):
                self.assertEqual(resolve_privileged_dim('auto', task, dim), dim)
        self.assertEqual(resolve_privileged_dim('auto', 'rubber_duck', 18, True), 24)
        self.assertEqual(resolve_privileged_dim('none', 'rubber_duck', 18, True), 18)
        for mode, source, warm in [('rotation6d', 18, False), ('none', 24, False), ('none', 24, True)]:
            with self.assertRaises(ValueError):
                resolve_privileged_dim(mode, 'rubber_duck', source, warm)
        checkpoint = {'model': {PRIV_INPUT_WEIGHT: torch.ones(2, 18)}}
        self.assertEqual(checkpoint_privileged_dim(checkpoint), 18)
        expanded = warmstart_privileged_state(checkpoint, 24)
        self.assertEqual(checkpoint['model'][PRIV_INPUT_WEIGHT].shape[1], 18)
        self.assertFalse(expanded[PRIV_INPUT_WEIGHT][:, 18:].any())
        checkpoint['priv_info_dim'] = 24
        with self.assertRaisesRegex(RuntimeError, 'metadata'):
            checkpoint_privileged_dim(checkpoint)

    def test_real_ppo_warmstart_preserves_actor_and_learns_new_inputs(self):
        env = SimpleNamespace(
            action_space=gym.spaces.Box(-1., 1., shape=(21,), dtype=np.float32),
            observation_space=gym.spaces.Box(-np.inf, np.inf, shape=(141,), dtype=np.float32))
        config = OmegaConf.load(Path(__file__).resolve().parents[1] / 'configs/train/Revo3HandHora.yaml')
        config.ppo.num_actors = 2
        config.ppo.minibatch_size = 32
        full = OmegaConf.create({'rl_device': 'cpu', 'test': False, 'train': config})
        with tempfile.TemporaryDirectory() as directory:
            old = PPO(env, directory + '/old', full)
            full.train.ppo.priv_info_dim = 24
            new = PPO(env, directory + '/new', full)
            try:
                old.running_mean_std.update(torch.randn(32, 141))
                old.agent_steps, old.epoch_num = 1234, 17
                source = directory + '/old/source'
                old.save(source)
                source += '.pth'
                with self.assertRaisesRegex(RuntimeError, 'mismatch'):
                    new.restore_train(source)
                with self.assertRaisesRegex(RuntimeError, 'layout mismatch'):
                    new.restore_test(source)
                new.restore_weights_only(source)
                old.model.eval()
                new.model.eval()
                obs = torch.randn(8, 141)
                priv = torch.randn(8, 18)
                rotation = object_rotation_6d(torch.randn(8, 4))
                old_input = {'obs': old.running_mean_std(obs, update_stats=False), 'priv_info': priv}
                new_input = {'obs': new.running_mean_std(obs, update_stats=False),
                             'priv_info': torch.cat([priv, rotation], -1)}
                torch.testing.assert_close(old.model.act_inference(old_input), new.model.act_inference(new_input))
                self.assertEqual((new.agent_steps, new.epoch_num), (0, 0))
                self.assertFalse(new.optimizer.state)
                self.assertFalse(new.model.value.weight.any())
                # Expanded columns are zero initially but have a real gradient.
                mu = new.model._actor_critic(new_input)[0]
                (mu - 1).square().mean().backward()
                columns = new.model.env_mlp.mlp[0].weight
                self.assertGreater(float(columns.grad[:, 18:].abs().sum()), 0)
                new.optimizer.step()
                self.assertGreater(float(columns[:, 18:].abs().sum()), 0)
                new.save(directory + '/expanded')
                new.restore_test(directory + '/expanded.pth')
                new.restore_train(directory + '/expanded.pth')
            finally:
                old.writer.close()
                new.writer.close()


if __name__ == '__main__':
    unittest.main()
