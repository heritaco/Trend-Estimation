from __future__ import annotations

import numpy as np
from .difference import difference_matrix


def smoothing_matrix(n_obs: int, order: int, lambda_: float) -> np.ndarray:
    """Return ``(I + lambda D.T D)^-1`` for small to medium arrays."""
    D = difference_matrix(n_obs, order)
    A = np.eye(n_obs) + float(lambda_) * (D.T @ D)
    return np.linalg.inv(A)
