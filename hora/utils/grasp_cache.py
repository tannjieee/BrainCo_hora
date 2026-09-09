"""Simulator-independent validation for the shared xyzw grasp-cache format."""
from pathlib import Path

import numpy as np


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
