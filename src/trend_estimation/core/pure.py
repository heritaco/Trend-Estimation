from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .difference import difference_matrix
from .smoothness import lambda_to_smoothness, smoothness_to_lambda
from trend_estimation.utils.arrays import as_1d_float_array


@dataclass
class PureSolverResult:
    """Result of the pure quadratic finite-difference smoother."""

    trend: np.ndarray
    lambda_: float
    smoothness: float
    diag_smoother: np.ndarray


class PurePenalizedSolver:
    r"""Spectral solver for

    .. math::
        \min_t \|y-t\|_2^2 + \lambda \|D_d t\|_2^2.

    The solution is ``(I + lambda D.T @ D)^(-1) y``.  This class deliberately
    contains no drift term; the Guerrero formulation lives in
    :class:`GuerreroSpectralSolver`.
    """

    def __init__(self, n_obs: int, order: int):
        self.n_obs = int(n_obs)
        self.order = int(order)
        if self.n_obs <= 0:
            raise ValueError("n_obs must be positive.")
        if self.order < 0:
            raise ValueError("order must be nonnegative.")
        self.D = difference_matrix(self.n_obs, self.order)
        self.penalty = self.D.T @ self.D
        self.eigvals, self.eigvecs = np.linalg.eigh(self.penalty)

    def lambda_from_s(self, smoothness: float) -> float:
        return smoothness_to_lambda(smoothness, self.n_obs, self.order)

    def smoothness_from_lambda(self, lambda_: float) -> float:
        return lambda_to_smoothness(lambda_, self.n_obs, self.order)

    def fit_for_lambda(self, y, lambda_: float) -> PureSolverResult:
        y = as_1d_float_array(y)
        if y.size != self.n_obs:
            raise ValueError(f"y has length {y.size}; expected {self.n_obs}.")
        lambda_ = float(lambda_)
        if lambda_ < 0:
            raise ValueError("lambda_ must be nonnegative.")

        alpha = 1.0 / (1.0 + lambda_ * self.eigvals)
        spectral_y = self.eigvecs.T @ y
        trend = self.eigvecs @ (alpha * spectral_y)
        diag_smoother = (self.eigvecs**2) @ alpha
        return PureSolverResult(
            trend=trend,
            lambda_=lambda_,
            smoothness=self.smoothness_from_lambda(lambda_),
            diag_smoother=diag_smoother,
        )

    def fit_for_s(self, y, smoothness: float) -> PureSolverResult:
        return self.fit_for_lambda(y, self.lambda_from_s(float(smoothness)))


def pure_penalized_solution(y, order: int, lambda_: float) -> np.ndarray:
    """Return only the fitted pure penalized trend."""

    y = as_1d_float_array(y)
    return PurePenalizedSolver(len(y), order).fit_for_lambda(y, lambda_).trend
