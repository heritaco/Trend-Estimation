from __future__ import annotations

import numpy as np

from .difference import difference_matrix


def penalty_eigenvalues(n_obs: int, order: int) -> np.ndarray:
    """Eigenvalues of ``D.T @ D`` for the finite-difference penalty."""
    D = difference_matrix(n_obs, order)
    return np.linalg.eigvalsh(D.T @ D)


def effective_degrees_of_freedom(lambda_: float, n_obs: int, order: int) -> float:
    """Return ``trace((I + lambda D.T D)^-1)``."""
    eigvals = penalty_eigenvalues(n_obs, order)
    return float(np.sum(1.0 / (1.0 + float(lambda_) * eigvals)))


def lambda_to_smoothness(lambda_: float, n_obs: int, order: int) -> float:
    """Map a penalty parameter to Guerrero's normalized smoothness index."""
    lambda_ = float(lambda_)
    if lambda_ < 0:
        raise ValueError("lambda_ must be nonnegative.")
    if order == 0:
        return lambda_ / (1.0 + lambda_)
    tr = effective_degrees_of_freedom(lambda_, n_obs, order)
    s_raw = 1.0 - tr / n_obs
    s_max = 1.0 - order / n_obs
    return float(s_raw / s_max) if s_max > 0 else 0.0


def smoothness_to_lambda(
    smoothness: float,
    n_obs: int,
    order: int,
    *,
    tol: float = 1e-11,
    max_iter: int = 100,
) -> float:
    """Map smoothness in ``[0,1)`` to ``lambda_`` using bisection."""
    smoothness = float(smoothness)
    if smoothness <= 0:
        return 0.0
    if smoothness >= 1.0:
        smoothness = 0.999999
    if order == 0:
        return smoothness / (1.0 - smoothness)

    eigvals = penalty_eigenvalues(n_obs, order)
    s_max = 1.0 - order / n_obs
    target = smoothness * s_max

    def s_raw(lmb: float) -> float:
        return 1.0 - float(np.sum(1.0 / (1.0 + lmb * eigvals))) / n_obs

    lo, hi = 0.0, 1.0
    while s_raw(hi) < target and hi < 1e16:
        hi *= 10.0
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        val = s_raw(mid)
        if abs(val - target) < tol:
            return float(mid)
        if val < target:
            lo = mid
        else:
            hi = mid
    return float(0.5 * (lo + hi))
