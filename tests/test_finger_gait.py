import math
import unittest

import torch

from hora.utils.finger_gait import (
    signed_axis_increment, new_turns, blocked_push, support_gate, debounce_contacts,
    directed_speed_reward,
)


def quat(degrees):
    a = torch.tensor(degrees) * math.pi / 360
    return torch.stack([a.cos(), torch.zeros_like(a), torch.zeros_like(a), a.sin()], -1)


class GaitTests(unittest.TestCase):
    def test_wrap_sign_reverse_and_off_axis(self):
        prev, cur = quat([179., 0., 20., 0.]), quat([-179., 0., 10., 90.])
        cur[1] *= -1
        axes = torch.tensor([[0., 0., 1.]]).repeat(4, 1)
        axes[-1] = torch.tensor([1., 0., 0.])
        result = signed_axis_increment(prev, cur, axes)
        torch.testing.assert_close(result, torch.tensor([2., 0., -10., 0.]) * math.pi / 180)

    def test_cumulative_turns_no_reverse_farming_and_subset_reset(self):
        high = torch.zeros(2)
        high, gained = new_turns(torch.tensor([2*math.pi+.01, 0.]), high)
        torch.testing.assert_close(gained, torch.tensor([1., 0.]))
        high, gained = new_turns(torch.tensor([1., -2*math.pi]), high)
        torch.testing.assert_close(gained, torch.zeros(2))
        high, gained = new_turns(torch.tensor([2*math.pi+.01, -1.]), high)
        torch.testing.assert_close(gained, torch.zeros(2))
        high[1] = 0
        high, gained = new_turns(torch.tensor([4*math.pi+.01, 2*math.pi+.01]), high)
        torch.testing.assert_close(gained, torch.ones(2))

    def test_limit_cost_only_for_persistent_outward_push(self):
        previous = torch.tensor([[1., 0., 1., 0., .5]])
        requested = torch.tensor([[1.1, -.1, 1., .1, .6]])
        lo, hi = torch.zeros_like(previous), torch.ones_like(previous)
        age = torch.zeros_like(previous, dtype=torch.long)
        for _ in range(4):
            age, cost = blocked_push(previous, requested, lo, hi, age, .1, 4)
            torch.testing.assert_close(cost, torch.zeros(1))
        age, cost = blocked_push(previous, requested, lo, hi, age, .1, 4)
        torch.testing.assert_close(cost, torch.tensor([.4]))
        age, cost = blocked_push(previous, torch.tensor([[.9,.1,1.,0.,.5]]), lo, hi, age, .1, 4)
        self.assertFalse(age.any())
        torch.testing.assert_close(cost, torch.zeros(1))

    def test_support_gate_allows_two_fingers_and_cuts_falling_progress(self):
        gate = support_gate(torch.tensor([0., .02, 0., 0., .0125]),
                            torch.tensor([0., 0., .05, 0., 0.]),
                            torch.tensor([2, 5, 5, 1, 3]), .005, .02, .05)
        torch.testing.assert_close(gate, torch.tensor([1., 0., 0., 0., .5]))

    def test_contact_debounce_and_hysteresis(self):
        state = torch.zeros(1, 2, dtype=torch.bool)
        age = torch.zeros(1, 2, dtype=torch.long)
        def step(values):
            nonlocal state, age
            state, age, on, off = debounce_contacts(torch.tensor([values]), state, age, .05, .025, 2)
            return on, off
        step([.06, .06]); step([0., .06])
        self.assertEqual(state.tolist(), [[False, True]])
        step([0., .03])  # Hysteresis retains an existing contact.
        self.assertEqual(state.tolist(), [[False, True]])
        step([0., 0.]); on, off = step([0., 0.])
        self.assertEqual(off.tolist(), [[False, True]])
        self.assertFalse(on.any())

    def test_pause_and_tiny_retreat_do_not_renew_limit_grace(self):
        lo, hi = torch.zeros(1, 1), torch.ones(1, 1)
        age = torch.full((1, 1), 5, dtype=torch.long)
        for target in (1., .999):
            age, cost = blocked_push(hi, torch.full_like(hi, target), lo, hi, age, .1, 4)
            self.assertEqual(age.item(), 5)
            self.assertEqual(cost.item(), 0)
        age, cost = blocked_push(hi, hi + .1, lo, hi, age, .1, 4)
        self.assertGreater(cost.item(), .99)
        age, cost = blocked_push(hi, hi - .1, lo, hi, age, .1, 4)
        self.assertEqual(age.item(), 0)

    def test_speed_target_is_a_peak_and_reverse_remains_penalized(self):
        speed = torch.tensor([-1., -.25, 0., .25, .5, .75, 1., 2.])
        torch.testing.assert_close(directed_speed_reward(speed, .5),
                                  torch.tensor([-1., -.5, 0., .5, 1., .5, 0., 0.]))


if __name__ == '__main__':
    unittest.main()
