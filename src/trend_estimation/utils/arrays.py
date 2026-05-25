from __future__ import annotations

import numpy as np


def as_1d_float_array(y, *, name: str = "y") -> np.ndarray:
    """Convert input to a finite one-dimensional float array."""
    arr = np.asarray(y, dtype=float)
    if arr.ndim != 1:
        arr = arr.ravel()
    if arr.size == 0:
        raise ValueError(f"{name} must not be empty.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values.")
    return arr


def indices_from_slice_or_array(idx, n_obs: int) -> np.ndarray:
    """Return integer positions from a slice, boolean mask, or integer array."""
    if idx is None:
        return np.arange(n_obs)
    if isinstance(idx, slice):
        return np.arange(n_obs)[idx]
    arr = np.asarray(idx)
    if arr.dtype == bool:
        if arr.size != n_obs:
            raise ValueError("Boolean index has incompatible length.")
        return np.flatnonzero(arr)
    return arr.astype(int)
