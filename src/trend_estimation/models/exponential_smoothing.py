from __future__ import annotations

import numpy as np

from .base import BaseTrendEstimator, TrendFitResult
from trend_estimation.utils.arrays import as_1d_float_array


class ExponentialSmoothingTrend(BaseTrendEstimator):
    """Simple exponential smoothing trend baseline.

    Parameters
    ----------
    alpha:
        Smoothing parameter in ``(0, 1]``. Larger values react more strongly
        to recent observations.
    initial:
        Optional initial level. If ``None``, the first observation is used.
    """

    def __init__(self, alpha: float = 0.2, initial: float | None = None):
        alpha = float(alpha)
        if not (0.0 < alpha <= 1.0):
            raise ValueError("alpha must be in (0, 1].")
        self.alpha = alpha
        self.initial = initial

    def fit(self, y, X=None):
        y = as_1d_float_array(y)
        if y.size == 0:
            raise ValueError("y must contain at least one observation.")
        trend = np.empty_like(y, dtype=float)
        trend[0] = float(y[0] if self.initial is None else self.initial)
        for i in range(1, y.size):
            trend[i] = self.alpha * y[i] + (1.0 - self.alpha) * trend[i - 1]

        self.y_ = y
        self.n_obs_ = y.size
        self.trend_ = trend
        self.fitted_values_ = trend
        self.residuals_ = y - trend
        self.last_level_ = float(trend[-1])
        self.fit_result_ = TrendFitResult(
            y_=y,
            trend_=trend,
            residuals_=self.residuals_,
            fitted_values_=self.fitted_values_,
            metadata_={"alpha": self.alpha, "initial": self.initial},
        )
        return self

    def forecast(self, steps: int) -> np.ndarray:
        if not hasattr(self, "last_level_"):
            raise RuntimeError("Call fit before forecast.")
        return np.full(int(steps), self.last_level_, dtype=float)
