from __future__ import annotations

from math import comb
import numpy as np


def difference_coefficients(order: int) -> np.ndarray:
    """Return coefficients for the forward difference of a given order."""
    if order < 0:
        raise ValueError("order must be nonnegative.")
    return np.array([(-1) ** (order - k) * comb(order, k) for k in range(order + 1)], dtype=float)


def difference_matrix(n_obs: int, order: int) -> np.ndarray:
    """Construct the finite-difference matrix of shape ``(n_obs-order, n_obs)``.

    For ``order=0`` the identity matrix is returned.
    """
    n_obs = int(n_obs)
    order = int(order)
    if n_obs <= 0:
        raise ValueError("n_obs must be positive.")
    if order < 0:
        raise ValueError("order must be nonnegative.")
    if order >= n_obs and order != 0:
        raise ValueError("order must be smaller than n_obs.")
    if order == 0:
        return np.eye(n_obs)
    K = np.zeros((n_obs - order, n_obs), dtype=float)
    coeffs = difference_coefficients(order)
    for r in range(n_obs - order):
        K[r, r : r + order + 1] = coeffs
    return K
