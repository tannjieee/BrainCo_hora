"""Simulator-independent stratification for physically validated grasp candidates."""
import numpy as np
import torch


class GraspQuota:
    """Do not let easy yaw/contact groups fill the entire grasp cache."""

    def __init__(self, total, yaw_bins, groups):
        count = yaw_bins * groups
        if total < count or min(yaw_bins, groups) < 1:
            raise ValueError('target_count must cover every yaw/contact group')
        self.targets = np.full(count, total // count, dtype=np.int64)
        self.targets[:total % count] += 1
        self.counts = np.zeros(count, dtype=np.int64)

    def accept(self, buckets):
        buckets = np.asarray(buckets, dtype=np.int64)
        if buckets.ndim != 1 or np.any((buckets < 0) | (buckets >= len(self.targets))):
            raise ValueError('invalid grasp bucket')
        keep = np.zeros(len(buckets), dtype=bool)
        for bucket in np.unique(buckets):
            candidates = np.flatnonzero(buckets == bucket)
            take = candidates[:max(0, int(self.targets[bucket] - self.counts[bucket]))]
            keep[take] = True
            self.counts[bucket] += len(take)
        return keep

    @property
    def complete(self):
        return bool(np.array_equal(self.counts, self.targets))


def rotate_about_axis(quat_wxyz, axis, angles):
    """Left-multiply by a world-axis rotation, preserving the initial axis tilt."""
    axis = torch.nn.functional.normalize(axis, dim=-1)
    rw = (angles / 2).cos().unsqueeze(-1)
    rv = axis * (angles / 2).sin().unsqueeze(-1)
    qw, qv = quat_wxyz[..., :1], quat_wxyz[..., 1:]
    return torch.cat((rw * qw - (rv * qv).sum(-1, keepdim=True),
                      rw * qv + qw * rv + torch.cross(rv, qv, dim=-1)), -1)


def valid_grasp_success(episode_lengths, max_length, terminated, valid, reset_ids):
    """A final-frame failure must never be cached as a successful timeout."""
    selected = torch.zeros_like(terminated, dtype=torch.bool)
    selected[reset_ids] = True
    return selected & (episode_lengths >= max_length) & ~terminated & valid
