from __future__ import annotations

import numpy as np

from .base import BaseTrendEstimator, TrendFitResult
from trend_estimation.utils.arrays import as_1d_float_array


class MovingAverageTrend(BaseTrendEstimator):
    """Trailing moving-average trend estimator.

    Parameters
    ----------
    window:
        Number of most recent observations used in each trailing average.
    min_periods:
        Minimum number of observations required at the beginning of the
        sample. If ``None``, it defaults to 1.

    Notes
    -----
    This is intentionally simple. It is a baseline for comparative studies,
    not a claim that moving averages are optimal trend estimators.
    """

    def __init__(self, window: int = 20, min_periods: int | None = None):
        if int(window) < 1:
            raise ValueError("window must be >= 1.")
        self.window = int(window)
        self.min_periods = min_periods

    def fit(self, y, X=None):
        y = as_1d_float_array(y)
        n = y.size
        min_periods = 1 if self.min_periods is None else int(self.min_periods)
        if min_periods < 1:
            raise ValueError("min_periods must be >= 1.")
        if min_periods > self.window:
            raise ValueError("min_periods cannot exceed window.")

        trend = np.empty(n, dtype=float)
        csum = np.r_[0.0, np.cumsum(y)]
        for i in range(n):
            start = max(0, i - self.window + 1)
            count = i - start + 1
            if count < min_periods:
                trend[i] = np.nan
            else:
                trend[i] = (csum[i + 1] - csum[start]) / count

        if np.isnan(trend).any():
            first_valid = np.flatnonzero(np.isfinite(trend))[0]
            trend[:first_valid] = trend[first_valid]

        self.y_ = y
        self.n_obs_ = n
        self.trend_ = trend
        self.fitted_values_ = trend
        self.residuals_ = y - trend
        self.last_level_ = float(trend[-1])
        self.fit_result_ = TrendFitResult(
            y_=y,
            trend_=trend,
            residuals_=self.residuals_,
            fitted_values_=self.fitted_values_,
            metadata_={"window": self.window, "min_periods": min_periods},
        )
        return self

    def forecast(self, steps: int) -> np.ndarray:
        if not hasattr(self, "last_level_"):
            raise RuntimeError("Call fit before forecast.")
        return np.full(int(steps), self.last_level_, dtype=float)
