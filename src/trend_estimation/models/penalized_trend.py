from __future__ import annotations

from functools import lru_cache
import numpy as np

from .base import BaseTrendEstimator, TrendFitResult
from trend_estimation.core.solvers import GuerreroSpectralSolver
from trend_estimation.forecasting.extrapolation import forecast_trend
from trend_estimation.utils.arrays import as_1d_float_array


@lru_cache(maxsize=32)
def _cached_solver(n_obs: int, order: int) -> GuerreroSpectralSolver:
    return GuerreroSpectralSolver(int(n_obs), int(order))


class PenalizedTrend(BaseTrendEstimator):
    """Guerrero-style penalized least-squares trend estimator."""

    def __init__(self, order: int = 2, smoothness: float | None = 0.75, lambda_: float | None = None, estimate_drift: bool = True):
        self.order = int(order)
        self.smoothness = smoothness
        self.lambda_ = lambda_
        self.estimate_drift = bool(estimate_drift)

    def fit(self, y, X=None):
        y = as_1d_float_array(y)
        self.y_ = y
        self.n_obs_ = y.size
        self.solver_ = _cached_solver(y.size, self.order)
        if self.lambda_ is None:
            if self.smoothness is None:
                raise ValueError("Either smoothness or lambda_ must be provided.")
            lambda_value = self.solver_.lambda_from_s(float(self.smoothness))
        else:
            lambda_value = float(self.lambda_)
        solver_result = self.solver_.fit_for_lambda(y, lambda_value, estimate_drift=self.estimate_drift)
        self.trend_ = solver_result.trend
        self.fitted_values_ = self.trend_
        self.residuals_ = y - self.trend_
        self.m_hat_ = solver_result.m_hat
        self.lambda_ = solver_result.lambda_
        self.smoothness_ = solver_result.smoothness
        self.sigma2_hat_ = solver_result.sigma2_hat
        self.fit_result_ = TrendFitResult(
            y_=self.y_,
            trend_=self.trend_,
            residuals_=self.residuals_,
            fitted_values_=self.fitted_values_,
            order_=self.order,
            lambda_=self.lambda_,
            smoothness_=self.smoothness_,
            metadata_={"m_hat": self.m_hat_, "sigma2_hat": self.sigma2_hat_},
        )
        return self

    def forecast(self, steps: int) -> np.ndarray:
        if not hasattr(self, "trend_"):
            raise RuntimeError("Call fit before forecast.")
        return forecast_trend(self.trend_, self.order, self.m_hat_, int(steps))
