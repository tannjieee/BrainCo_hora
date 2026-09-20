"""Simulator-independent, batched finger-gait diagnostics/reward primitives."""
import math

import torch


def signed_axis_increment(previous, current, axis):
    """Shortest world-frame quaternion increment projected on a unit axis.

    Quaternions use wxyz. Sign flips are equivalent rotations. Sampling must
    resolve rotations smaller than pi per control step; resets must seed prev.
    """
    previous = torch.nn.functional.normalize(previous, dim=-1)
    current = torch.nn.functional.normalize(current, dim=-1)
    pw, pv = previous[..., :1], previous[..., 1:]
    cw, cv = current[..., :1], current[..., 1:]
    w = cw * pw + (cv * pv).sum(-1, keepdim=True)
    v = pw * cv - cw * pv - torch.cross(cv, pv, dim=-1)
    sign = torch.where(w < 0, -1., 1.)
    w, v = w * sign, v * sign
    norm = v.norm(dim=-1, keepdim=True)
    angle = 2 * torch.atan2(norm, w.clamp_min(0))
    rotvec = v * torch.where(norm > 1e-7, angle / norm.clamp_min(1e-7), 2.)
    return (rotvec * axis).sum(-1)


def new_turns(net_angle, high_water):
    """Reward a new signed full-turn high-water mark only once per episode."""
    reached = torch.floor(net_angle.clamp_min(0) / (2 * math.pi))
    gained = (reached - high_water).clamp_min(0)
    return torch.maximum(high_water, reached), gained


def blocked_push(previous, requested, lower, upper, age, action_scale, grace_steps,
                 recovery_margin=.05):
    """Charge outward overflow until real target clearance has been restored.

    Pausing or making a tiny inward motion must not refresh the grace period.
    Holding/retracting still incurs zero cost; only outward overflow is charged.
    """
    clipped = torch.maximum(torch.minimum(requested, upper), lower)
    overflow = (requested - clipped).abs() / action_scale
    moving_outward = ((requested > upper) & (requested > previous)) | (
        (requested < lower) & (requested < previous))
    margin = torch.minimum(torch.full_like(lower, recovery_margin), (upper - lower) / 2)
    recovered = (requested >= lower + margin) & (requested <= upper - margin)
    age = torch.where(recovered, torch.zeros_like(age),
                      torch.where(moving_outward, (age + 1).clamp_max(grace_steps + 1), age))
    penalty = (overflow.clamp(max=1).square() * (age > grace_steps)).mean(-1)
    return age, penalty


def directed_speed_reward(angular_velocity, target_speed):
    """Peak at target speed; slowing the target must not reward overspeed equally."""
    ratio = angular_velocity / target_speed
    return ratio.clamp(-1, 1) - (ratio - 1).clamp(0, 1)


def support_gate(z_drift, downward_speed, contact_count, safe_z, max_z, max_down_speed):
    """Positive progress only; allow a free finger while at least two support."""
    height = ((max_z - z_drift) / (max_z - safe_z)).clamp(0, 1)
    velocity = (1 - downward_speed.clamp_min(0) / max_down_speed).clamp(0, 1)
    return height * velocity * (contact_count >= 2).to(z_drift.dtype)


def debounce_contacts(force, state, age, on_threshold, off_threshold, frames):
    """Hysteresis + consecutive-frame debounce; no contact-change reward."""
    desired = torch.where(state, force > off_threshold, force >= on_threshold)
    age = torch.where(desired != state, age + 1, torch.zeros_like(age))
    changed = age >= frames
    updated = torch.where(changed, desired, state)
    recontacts = changed & updated
    releases = changed & ~updated
    return updated, torch.where(changed, torch.zeros_like(age), age), recontacts, releases
