import math
import unittest

import numpy as np
import torch

from hora.utils.grasp_sampling import GraspQuota, rotate_about_axis, valid_grasp_success


class GraspSamplingTests(unittest.TestCase):
    def test_easy_groups_cannot_exhaust_cache_budget(self):
        quota = GraspQuota(10, 2, 2)
        self.assertEqual(quota.accept([0] * 100).sum(), 3)
        self.assertFalse(quota.complete)
        self.assertEqual(quota.accept([0, 1, 1, 1, 2, 2, 3, 3]).sum(), 7)
        self.assertTrue(quota.complete)
        self.assertFalse(quota.accept([0, 1, 2, 3]).any())
        np.testing.assert_array_equal(quota.counts, [3, 3, 2, 2])

    def test_invalid_coverage_or_bucket_fails(self):
        with self.assertRaises(ValueError):
            GraspQuota(3, 2, 2)
        quota = GraspQuota(4, 2, 2)
        with self.assertRaises(ValueError):
            quota.accept([-1])

    def test_world_axis_rotation_preserves_normalized_quaternion(self):
        q = torch.tensor([[1., 0., 0., 0.], [1., 0., 0., 0.]])
        axis = torch.tensor([[0., 0., 1.], [1., 0., 0.]])
        rotated = rotate_about_axis(q, axis, torch.tensor([math.pi, math.pi / 2]))
        torch.testing.assert_close(rotated.norm(dim=-1), torch.ones(2))
        torch.testing.assert_close(rotated[0], torch.tensor([0., 0., 0., 1.]), atol=1e-6, rtol=0)
        torch.testing.assert_close(rotated[1], torch.tensor([math.sqrt(.5), math.sqrt(.5), 0., 0.]))
        restored = rotate_about_axis(rotated, axis, torch.tensor([-math.pi, -math.pi / 2]))
        torch.testing.assert_close(restored, q, atol=1e-6, rtol=0)

    def test_final_frame_failure_and_unselected_env_never_cached(self):
        ok = valid_grasp_success(torch.tensor([100, 100, 100, 100, 99]), 100,
                                 torch.tensor([False, True, False, False, False]),
                                 torch.tensor([True, True, False, True, True]),
                                 torch.tensor([0, 1, 2, 4]))
        self.assertEqual(ok.tolist(), [True, False, False, False, False])


if __name__ == '__main__':
    unittest.main()
