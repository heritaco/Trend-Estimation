from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .pure import PurePenalizedSolver
from trend_estimation.utils.arrays import as_1d_float_array


@dataclass
class PureTrendDerivatives:
    """Trend and its first two derivatives with respect to ``lambda_``."""

    trend: np.ndarray
    first: np.ndarray
    second: np.ndarray


def pure_trend_derivatives(y, order: int, lambda_: float) -> PureTrendDerivatives:
    r"""Differentiate the pure penalized smoother analytically.

    If ``Q = D.T @ D`` and ``S = (I + lambda Q)^(-1)``, then

    ``t = S y``, ``t' = -S Q t``, and ``t'' = 2 S Q S Q t``.

    The implementation uses the eigendecomposition already natural to the
    finite-difference penalty, avoiding explicit matrix inversion.
    """

    y = as_1d_float_array(y)
    lambda_ = float(lambda_)
    if lambda_ < 0:
        raise ValueError("lambda_ must be nonnegative.")

    solver = PurePenalizedSolver(len(y), int(order))
    delta = solver.eigvals
    q = solver.eigvecs
    spectral_y = q.T @ y
    alpha = 1.0 / (1.0 + lambda_ * delta)

    trend = q @ (alpha * spectral_y)
    first = q @ ((-delta * alpha**2) * spectral_y)
    second = q @ ((2.0 * delta**2 * alpha**3) * spectral_y)
    return PureTrendDerivatives(trend=trend, first=first, second=second)


def mse_from_prediction_derivatives(
    target,
    prediction,
    prediction_first,
    prediction_second,
) -> tuple[float, float, float]:
    r"""Return MSE and its first two derivatives for a differentiable prediction.

    For residual ``r = target - prediction(lambda)``,

    ``f' = -(2/n) r.T prediction'`` and
    ``f'' = (2/n)(prediction'.T prediction' - r.T prediction'')``.
    """

    target = as_1d_float_array(target)
    prediction = as_1d_float_array(prediction)
    prediction_first = as_1d_float_array(prediction_first)
    prediction_second = as_1d_float_array(prediction_second)
    if not (
        target.size
        == prediction.size
        == prediction_first.size
        == prediction_second.size
    ):
        raise ValueError("All inputs must have the same length.")
    if target.size == 0:
        raise ValueError("Inputs must not be empty.")

    residual = target - prediction
    n = float(target.size)
    value = float((residual @ residual) / n)
    first = float((-2.0 / n) * (residual @ prediction_first))
    second = float((2.0 / n) * ((prediction_first @ prediction_first) - (residual @ prediction_second)))
    return value, first, second
