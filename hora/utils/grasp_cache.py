"""Simulator-independent validation for the shared xyzw grasp-cache format."""
from pathlib import Path

import numpy as np


def validate_object_runtime(previous, current):
    """Reject accidental checkpoint/task/cache mixing, ignoring machine paths."""
    if not previous or not current:
        return
    old_object, new_object = previous.get('object', {}), current.get('object', {})
    keys = ('task', 'scale', 'object_init_pos_m', 'object_init_quat_wxyz',
            'rotation_axis_local', 'target_axis_world', 'hand_joint_pos_rad',
            'axis_bidirectional', 'enforce_axis_alignment', 'axis_tilt_tolerance_deg')
    if (any(old_object.get(k) != new_object.get(k) for k in keys)
            or previous.get('grasp_cache_sha256') != current.get('grasp_cache_sha256')):
        raise RuntimeError('Checkpoint task/pose/cache mismatch; select the matching --task and --cache_file. '
                           'For intentional training transfer use --weights_only in a new run.')


def load_grasp_cache(path, num_dofs):
    path = Path(path)
    data = np.load(path, allow_pickle=False)
    if data.ndim != 2 or data.shape[0] == 0 or data.shape[1] != num_dofs + 7:
        raise ValueError(f'{path}: expected nonempty (N, {num_dofs + 7}) grasp cache, got {data.shape}')
    if data.dtype.kind not in 'fi' or not np.isfinite(data).all():
        raise ValueError(f'{path}: grasp cache must contain only finite numeric values')
    norms = np.linalg.norm(data[:, num_dofs + 3:num_dofs + 7].astype(np.float64), axis=1)
    if not np.allclose(norms, 1.0, rtol=0.0, atol=1e-3):
        raise ValueError(f'{path}: grasp-cache xyzw quaternions must be unit length')
    data = data.astype(np.float32)
    if not np.isfinite(data).all():
        raise ValueError(f'{path}: grasp cache overflows float32')
    return data
