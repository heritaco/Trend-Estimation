from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .difference import difference_matrix
from trend_estimation.utils.arrays import as_1d_float_array


@dataclass
class SolverResult:
    trend: np.ndarray
    m_hat: float
    lambda_: float
    sigma2_hat: float
    diag_smoother: np.ndarray
    smoothness: float


class GuerreroSpectralSolver:
    """Spectral solver for Guerrero-style penalized trend smoothing."""

    def __init__(self, n_obs: int, order: int):
        self.n_obs = int(n_obs)
        self.order = int(order)
        self.D = difference_matrix(self.n_obs, self.order)
        self.DT = self.D.T
        self.penalty = self.DT @ self.D
        eigvals, eigvecs = np.linalg.eigh(self.penalty)
        self.eigvals = eigvals
        self.eigvecs = eigvecs
        self.DT1 = self.DT @ np.ones(self.n_obs - self.order if self.order > 0 else self.n_obs)

    @property
    def s_max(self) -> float:
        return 1.0 - self.order / self.n_obs if self.order > 0 else 1.0

    def lambda_from_s(self, smoothness: float) -> float:
        smoothness = float(smoothness)
        if smoothness <= 0:
            return 0.0
        if smoothness >= 1.0:
            smoothness = 0.999999
        if self.order == 0:
            return smoothness / (1.0 - smoothness)

        target = smoothness * self.s_max

        def s_raw(lambda_value: float) -> float:
            return 1.0 - float(np.sum(1.0 / (1.0 + lambda_value * self.eigvals))) / self.n_obs

        lo, hi = 0.0, 1.0
        while s_raw(hi) < target and hi < 1e16:
            hi *= 10.0
        for _ in range(100):
            mid = 0.5 * (lo + hi)
            val = s_raw(mid)
            if abs(val - target) < 1e-11:
                return float(mid)
            if val < target:
                lo = mid
            else:
                hi = mid
        return float(0.5 * (lo + hi))

    def smoothness_from_lambda(self, lambda_: float) -> float:
        lambda_ = float(lambda_)
        if lambda_ < 0:
            raise ValueError("lambda_ must be nonnegative.")
        if self.order == 0:
            return lambda_ / (1.0 + lambda_)
        tr = float(np.sum(1.0 / (1.0 + lambda_ * self.eigvals)))
        s_raw = 1.0 - tr / self.n_obs
        return float(s_raw / self.s_max) if self.s_max > 0 else 0.0

    def fit_for_lambda(
        self,
        y,
        lambda_: float,
        *,
        estimate_drift: bool = True,
        m_tol: float = 1e-10,
        max_m_iter: int = 120,
    ) -> SolverResult:
        y = as_1d_float_array(y)
        if y.size != self.n_obs:
            raise ValueError(f"y has length {y.size}; expected {self.n_obs}.")
        lambda_ = float(lambda_)
        if lambda_ < 0:
            raise ValueError("lambda_ must be nonnegative.")

        if estimate_drift:
            m_hat = float(np.mean(self.D @ y))
        else:
            m_hat = 0.0

        q = self.eigvecs
        denom = 1.0 + lambda_ * self.eigvals
        for _ in range(max_m_iter):
            rhs = y + lambda_ * m_hat * self.DT1
            z = (q.T @ rhs) / denom
            trend = q @ z
            if not estimate_drift:
                break
            m_new = float(np.mean(self.D @ trend))
            if abs(m_new - m_hat) < m_tol:
                m_hat = m_new
                break
            m_hat = m_new

        alpha = 1.0 / denom
        diag_smoother = (q**2) @ alpha
        residuals = y - trend
        penalty_residuals = (self.D @ trend) - m_hat
        dof = max(1, self.n_obs - self.order - 1)
        sigma2_hat = float((residuals @ residuals + lambda_ * (penalty_residuals @ penalty_residuals)) / dof)
        smoothness = self.smoothness_from_lambda(lambda_)
        return SolverResult(trend, m_hat, lambda_, sigma2_hat, diag_smoother, smoothness)

    def fit_for_s(self, y, smoothness: float, **kwargs) -> SolverResult:
        lambda_ = self.lambda_from_s(smoothness)
        return self.fit_for_lambda(y, lambda_, **kwargs)


def penalized_solution(y, order: int, lambda_: float, *, estimate_drift: bool = True) -> np.ndarray:
    """Convenience function returning only the penalized trend."""
    y = as_1d_float_array(y)
    solver = GuerreroSpectralSolver(len(y), order)
    return solver.fit_for_lambda(y, lambda_, estimate_drift=estimate_drift).trend
