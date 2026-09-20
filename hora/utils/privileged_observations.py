"""Stage1 privileged-observation layout and explicit legacy-policy migration."""
import torch


LEGACY_PRIV_DIM = 18
ORIENTATION_PRIV_DIM = 24
PRIV_INPUT_WEIGHT = 'env_mlp.mlp.0.weight'


def object_rotation_6d(quaternion):
    """World directions of object-local X then Y, from a wxyz quaternion.

    Absolute orientation (not relative to the reset pose) distinguishes asymmetric
    objects at different initial yaw angles. q and -q yield identical features.
    """
    q = torch.nn.functional.normalize(quaternion, dim=-1)
    w, x, y, z = q.unbind(-1)
    return torch.stack((1 - 2 * (y*y + z*z), 2 * (x*y + w*z),
                        2 * (x*z - w*y), 2 * (x*y - w*z),
                        1 - 2 * (x*x + z*z), 2 * (y*z + w*x)), dim=-1)


def checkpoint_privileged_dim(checkpoint):
    weight = checkpoint['model'][PRIV_INPUT_WEIGHT]
    dim = weight.shape[1]
    declared = checkpoint.get('priv_info_dim', dim)
    if int(declared) != dim:
        raise RuntimeError('Checkpoint privileged-observation metadata disagrees with model weights')
    if dim not in (LEGACY_PRIV_DIM, ORIENTATION_PRIV_DIM):
        raise RuntimeError(f'Unsupported privileged-observation layout: {dim}')
    return dim


def resolve_privileged_dim(mode, task, checkpoint_dim=None, weights_only=False):
    """Preserve old eval/resume layouts; new duck policies include orientation."""
    if mode == 'auto':
        if checkpoint_dim is not None and (not weights_only or task != 'rubber_duck'):
            target = checkpoint_dim
        else:
            target = ORIENTATION_PRIV_DIM if task == 'rubber_duck' else LEGACY_PRIV_DIM
    elif mode in ('none', 'rotation6d'):
        target = ORIENTATION_PRIV_DIM if mode == 'rotation6d' else LEGACY_PRIV_DIM
    else:
        raise ValueError(f'Unknown object orientation mode: {mode}')
    if target not in (LEGACY_PRIV_DIM, ORIENTATION_PRIV_DIM):
        raise ValueError(f'Unsupported privileged-observation layout: {target}')
    if checkpoint_dim is not None and checkpoint_dim != target:
        if not (weights_only and checkpoint_dim == LEGACY_PRIV_DIM and target == ORIENTATION_PRIV_DIM):
            raise ValueError('Changing the observation layout requires --weights_only '
                             'with an 18 -> 24 dimensional orientation expansion; '
                             'use --object_orientation auto for unchanged eval/resume')
    return target


def warmstart_privileged_state(checkpoint, target_dim):
    """Append zero input columns so the expanded actor initially matches exactly.

    Only a weights-only warm start may use this. Optimizer and critic state must
    be reset by the caller; strict restore must never expand weights implicitly.
    """
    source_dim = checkpoint_privileged_dim(checkpoint)
    state = dict(checkpoint['model'])
    if source_dim == target_dim:
        return state
    if (source_dim, target_dim) != (LEGACY_PRIV_DIM, ORIENTATION_PRIV_DIM):
        raise RuntimeError(f'Unsupported observation migration: {source_dim} -> {target_dim}')
    old = state[PRIV_INPUT_WEIGHT]
    expanded = old.new_zeros(old.shape[0], target_dim)
    expanded[:, :source_dim] = old
    state[PRIV_INPUT_WEIGHT] = expanded
    return state
