"""CPU regressions: python -m unittest discover -s tests -p test_stage1_core.py -v"""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import gymnasium as gym
import numpy as np
import torch
from omegaconf import OmegaConf

from hora.algo.models.running_mean_std import RunningMeanStd
from hora.algo.ppo.experience import ExperienceBuffer
from hora.algo.ppo.ppo import PPO
from hora.utils.grasp_cache import load_grasp_cache


class NormalizationTests(unittest.TestCase):
    def test_single_statistics_for_values_and_returns(self):
        rms = RunningMeanStd((1,))
        values = torch.zeros(128, 1)
        returns = torch.full_like(values, 10.0)
        rms.update(returns)
        count = rms.count.clone()
        normalized_values = rms(values, update_stats=False)
        normalized_returns = rms(returns, update_stats=False)
        torch.testing.assert_close(rms.count, count)
        # Use values within the clipping range to verify the shared affine map.
        torch.testing.assert_close(normalized_returns, rms(returns, update_stats=False))
        torch.testing.assert_close(normalized_values, rms(values, update_stats=False))
        self.assertLess(float(normalized_values.mean()), float(normalized_returns.mean()))

    def test_singleton_and_inverse_do_not_corrupt_statistics(self):
        rms = RunningMeanStd((1,))
        rms.update(torch.tensor([[3.0]]))
        self.assertTrue(torch.isfinite(rms.running_var).all())
        count = rms.count.clone()
        rms(torch.tensor([[0.0]]), unnorm=True)
        torch.testing.assert_close(rms.count, count)

    def test_population_moments_match_direct_calculation(self):
        rms = RunningMeanStd((1,))
        rms.update(torch.tensor([[2.0], [4.0]]))
        # Includes the original prior: count=1, mean=0, variance=1.
        torch.testing.assert_close(rms.running_mean, torch.tensor([2.0], dtype=torch.float64))
        torch.testing.assert_close(rms.running_var, torch.tensor([3.0], dtype=torch.float64))
        self.assertEqual(rms.count.item(), 3)


class BootstrapTests(unittest.TestCase):
    def setUp(self):
        self.agent = object.__new__(PPO)
        self.agent.reward_scale = 0.01
        self.agent.gamma = 0.99
        self.agent.value_bootstrap = True
        self.agent.model_value = lambda terminal: terminal['priv_info']

    def test_only_final_state_of_pure_timeout_bootstraps(self):
        rewards = torch.ones(3, 1)
        infos = {'time_outs': torch.tensor([True, False, False]), 'terminal_observation': {
            'env_ids': torch.tensor([0]), 'priv_info': torch.tensor([[7.0]]),
        }}
        result = self.agent._bootstrap_timeouts(rewards, infos)
        torch.testing.assert_close(result, torch.tensor([[6.94], [.01], [.01]]))
        torch.testing.assert_close(rewards, torch.ones_like(rewards))
        storage = ExperienceBuffer(3, 1, 3, 3, 1, 1, 1, 'cpu')
        storage.update_data('rewards', 0, result)
        storage.update_data('dones', 0, torch.tensor([1, 1, 0], dtype=torch.uint8))
        storage.computer_return(torch.full((3, 1), 100.0), .99, .95)
        torch.testing.assert_close(storage.storage_dict['returns'][0], torch.tensor([[6.94], [.01], [99.01]]))

    def test_missing_or_mismatched_final_state_fails(self):
        with self.assertRaises(RuntimeError):
            self.agent._bootstrap_timeouts(torch.zeros(2, 1), {'time_outs': torch.tensor([True, False])})
        with self.assertRaises(RuntimeError):
            self.agent._bootstrap_timeouts(torch.zeros(2, 1), {
                'time_outs': torch.tensor([False, True]),
                'terminal_observation': {'env_ids': torch.tensor([0]), 'priv_info': torch.ones(1, 1)},
            })

    def test_disabled_bootstrap_does_not_require_terminal_state(self):
        self.agent.value_bootstrap = False
        result = self.agent._bootstrap_timeouts(torch.ones(2, 1), {'time_outs': torch.ones(2, dtype=torch.bool)})
        torch.testing.assert_close(result, torch.full((2, 1), .01))


class CacheTests(unittest.TestCase):
    def test_valid_and_invalid_cache_formats(self):
        valid = np.zeros((2, 28), dtype=np.float32)
        valid[:, -1] = 1.0
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'cache.npy'
            np.save(path, valid)
            np.testing.assert_array_equal(load_grasp_cache(path, 21), valid)
            invalid_quat = valid.copy()
            invalid_quat[0, -1] = 0.0
            nonfinite = valid.copy()
            nonfinite[0, 0] = np.nan
            for bad in (valid[:0], valid[:, :-1], valid[0], invalid_quat, nonfinite):
                np.save(path, bad)
                with self.assertRaises(ValueError):
                    load_grasp_cache(path, 21)


class RolloutLoggingTests(unittest.TestCase):
    def test_batched_metrics_preserve_scaling_counts_and_reused_buffers(self):
        env = SimpleNamespace(
            action_space=gym.spaces.Box(-1., 1., shape=(21,), dtype=np.float32),
            observation_space=gym.spaces.Box(-np.inf, np.inf, shape=(141,), dtype=np.float32))
        config = OmegaConf.load(Path(__file__).resolve().parents[1] / 'configs/train/Revo3HandHora.yaml')
        config.ppo.num_actors = 2
        config.ppo.horizon_length = 2
        config.ppo.minibatch_size = 4
        full = OmegaConf.create({'rl_device': 'cpu', 'test': False, 'train': config})
        observation = {'obs': torch.zeros(2, 141), 'priv_info': torch.zeros(2, 18)}
        shared = torch.tensor(0.)
        calls = []
        def step(actions):
            calls.append(actions.clone())
            t = len(calls)
            shared.fill_(10*t)
            info = {'rew/example': shared, 'plain': shared, 'python_value': t,
                    'vector_ignored': torch.ones(2),
                    'gait/completed_count': torch.tensor(1. if t == 1 else 3.),
                    'gait/completed_one_turn_count': torch.tensor(1. if t == 1 else 0.),
                    'gait/completed_net_turns_sum': torch.tensor(2. if t == 1 else 6.),
                    'gait/completed_turns_sum': torch.tensor(1.)}
            return observation, torch.ones(2), torch.zeros(2, dtype=torch.uint8), info
        env.step = step
        with tempfile.TemporaryDirectory() as directory:
            agent = PPO(env, directory, full)
            try:
                agent.obs = observation
                original_act = agent.model_act
                def act(obs):
                    result = original_act(obs)
                    result['actions'].fill_(2. if not calls else .5)
                    return result
                agent.model_act = act
                with torch.no_grad():
                    agent.play_steps()
                metrics = agent.extra_info
                self.assertAlmostEqual(metrics['rew/example'], .15, places=6)
                self.assertAlmostEqual(metrics['plain'], 15.)
                self.assertAlmostEqual(metrics['python_value'], 1.5)
                self.assertAlmostEqual(metrics['gait/completed_one_turn_rate'], .25)
                self.assertAlmostEqual(metrics['gait/completed_mean_net_turns'], 2.)
                self.assertAlmostEqual(metrics['gait/completed_mean_turns'], .5)
                self.assertAlmostEqual(metrics['policy/action_saturation_rate'], .5)
                self.assertNotIn('vector_ignored', metrics)
                self.assertTrue(all(isinstance(v, float) for v in metrics.values()))
                torch.testing.assert_close(calls[0], torch.ones(2, 21))
                self.assertEqual(agent.agent_steps, 4)
            finally:
                agent.writer.close()


class CheckpointTests(unittest.TestCase):
    def test_roundtrip_rejects_changed_cache_or_pose(self):
        env = SimpleNamespace(
            action_space=gym.spaces.Box(-1.0, 1.0, shape=(21,), dtype=np.float32),
            observation_space=gym.spaces.Box(-np.inf, np.inf, shape=(141,), dtype=np.float32),
        )
        config = OmegaConf.load(Path(__file__).resolve().parents[1] / 'configs/train/Revo3HandHora.yaml')
        config.ppo.num_actors = 2
        config.ppo.minibatch_size = 32
        full = OmegaConf.create({'rl_device': 'cpu', 'test': False, 'train': config})
        with tempfile.TemporaryDirectory() as directory:
            agent = PPO(env, directory, full)
            try:
                agent.env_runtime = {'object': {'task': 'strawberry', 'scale': 1.4641}, 'grasp_cache_sha256': 'original'}
                path = str(Path(directory) / 'checkpoint')
                agent.agent_steps = 128
                agent.best_rewards = 4.0
                agent.save(path)
                original_bytes = Path(path + '.pth').read_bytes()
                with patch('hora.algo.ppo.ppo.torch.save', side_effect=OSError('simulated write failure')):
                    with self.assertRaises(OSError):
                        agent.save(path)
                self.assertEqual(Path(path + '.pth').read_bytes(), original_bytes)
                self.assertFalse(list(Path(directory).glob('*.tmp')))
                agent.agent_steps = 0
                agent.restore_train(path + '.pth')
                self.assertEqual(agent.agent_steps, 128)
                self.assertEqual(agent.best_rewards, 4.0)
                agent.restore_test(path + '.pth')
                agent.env_runtime['finger_gait'] = {'enabled': True, 'version': 1}
                with self.assertRaisesRegex(RuntimeError, 'reward mismatch'):
                    agent.restore_train(path + '.pth')
                agent.env_runtime.pop('finger_gait')
                agent.env_runtime['grasp_cache_sha256'] = 'changed'
                with self.assertRaisesRegex(RuntimeError, 'mismatch'):
                    agent.restore_train(path + '.pth')
                with self.assertRaisesRegex(RuntimeError, 'mismatch'):
                    agent.restore_test(path + '.pth')
                agent.env_runtime['grasp_cache_sha256'] = 'original'
                agent.env_runtime['object']['scale'] = 2.0
                with self.assertRaisesRegex(RuntimeError, 'mismatch'):
                    agent.restore_train(path + '.pth')
                agent.env_runtime['object']['scale'] = 1.4641
                legacy = torch.load(path + '.pth', weights_only=True)
                legacy.pop('training_semantics_version')
                torch.save(legacy, path + '.legacy.pth')
                agent.restore_train(path + '.legacy.pth')
                self.assertEqual(agent.best_rewards, -10000)
                self.assertEqual(agent.full_gravity_epochs, 0)
                # One final non-periodic update must still reach last.pth.
                agent.agent_steps = 0
                agent.epoch_num = 0
                agent.max_agent_steps = 1
                agent.save_freq = 50
                agent.save_best_after = 1000
                agent.env.reset = lambda: {}
                def train_epoch():
                    agent.agent_steps += agent.batch_size
                    return [], [], [], [], [], .1, .1
                agent.train_epoch = train_epoch
                agent.write_stats = lambda *a: None
                agent._print_epoch_log = lambda **k: None
                agent.train()
                final = torch.load(Path(directory) / 'stage1_nn/last.pth', weights_only=True)
                self.assertEqual(final['epoch_num'], 1)
                self.assertEqual(final['agent_steps'], agent.batch_size)
                # Resume after two updates collected with a larger historical batch.
                # A four-iteration cap still means exactly two further updates.
                agent.epoch_num = 2
                agent.agent_steps = 128
                agent.max_iterations = 4
                agent.max_agent_steps = 1  # stale transition cap must not win
                agent.train()
                resumed = torch.load(Path(directory) / 'stage1_nn/last.pth', weights_only=True)
                self.assertEqual(resumed['epoch_num'], 4)
                self.assertEqual(resumed['agent_steps'], 128 + 2 * agent.batch_size)
            finally:
                agent.writer.close()


if __name__ == '__main__':
    unittest.main()
