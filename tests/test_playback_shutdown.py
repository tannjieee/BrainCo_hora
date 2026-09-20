"""Playback cancellation must stop stepping without reporting a full evaluation."""
import contextlib
import io
from types import SimpleNamespace
import unittest

import torch

from hora.algo.ppo.ppo import PPO
from hora.algo.padapt.padapt import ProprioAdapt


class PlaybackShutdownTests(unittest.TestCase):
    def test_window_close_stops_both_players_without_complete_eval_metrics(self):
        for algorithm in (PPO, ProprioAdapt):
            for allowed_steps in (0, 1):
                with self.subTest(algorithm=algorithm.__name__, allowed_steps=allowed_steps):
                    calls = []
                    observation = {key: torch.zeros(1, 3) for key in ('obs', 'priv_info', 'proprio_hist')}

                    def step(_actions):
                        calls.append(1)
                        return observation, torch.zeros(1), torch.zeros(1), {}

                    agent = algorithm.__new__(algorithm)
                    agent.device, agent.num_actors = 'cpu', 1
                    agent.set_eval = lambda: None
                    agent.env = SimpleNamespace(
                        reset=lambda: observation, step=step, step_dt=.05,
                        reset_terminated=torch.zeros(1, dtype=torch.bool),
                    )
                    agent.running_mean_std = agent.sa_mean_std = lambda value: value
                    agent.model = SimpleNamespace(act_inference=lambda _obs: torch.zeros(1, 21))
                    output = io.StringIO()
                    with contextlib.redirect_stdout(output):
                        agent.test(max_steps=5, should_continue=lambda: len(calls) < allowed_steps)
                    self.assertEqual(len(calls), allowed_steps)
                    self.assertIn(f'closed after {allowed_steps} policy steps', output.getvalue())
                    self.assertNotIn('[FULL-GRAVITY EVAL]', output.getvalue())


if __name__ == '__main__':
    unittest.main()
