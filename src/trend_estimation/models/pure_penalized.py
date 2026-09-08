from __future__ import annotations

from functools import lru_cache

from trend_estimation.core.pure import PurePenalizedSolver
from trend_estimation.forecasting.extrapolation import forecast_trend
from trend_estimation.models.base import BaseTrendEstimator, TrendFitResult
from trend_estimation.utils.arrays import as_1d_float_array


@lru_cache(maxsize=32)
def _cached_pure_solver(n_obs: int, order: int) -> PurePenalizedSolver:
    return PurePenalizedSolver(int(n_obs), int(order))


class PurePenalizedTrend(BaseTrendEstimator):
    """Pure quadratic finite-difference trend estimator.

    This model solves ``||y-t||^2 + lambda ||D_d t||^2`` with no Guerrero drift
    term.  Its out-of-sample continuation imposes zero ``order``-th difference,
    which yields the polynomial continuation implied by the penalty order.
    """

    def __init__(
        self,
        order: int = 2,
        smoothness: float | None = 0.75,
        lambda_: float | None = None,
    ):
        self.order = int(order)
        self.smoothness = smoothness
        self.lambda_ = lambda_

    def fit(self, y, X=None):
        y = as_1d_float_array(y)
        self.y_ = y
        self.n_obs_ = y.size
        self.solver_ = _cached_pure_solver(y.size, self.order)
        if self.lambda_ is None:
            if self.smoothness is None:
                raise ValueError("Either smoothness or lambda_ must be provided.")
            lambda_value = self.solver_.lambda_from_s(float(self.smoothness))
        else:
            lambda_value = float(self.lambda_)

        result = self.solver_.fit_for_lambda(y, lambda_value)
        self.trend_ = result.trend
        self.fitted_values_ = self.trend_
        self.residuals_ = y - self.trend_
        self.lambda_ = result.lambda_
        self.smoothness_ = result.smoothness
        self.fit_result_ = TrendFitResult(
            y_=self.y_,
            trend_=self.trend_,
            residuals_=self.residuals_,
            fitted_values_=self.fitted_values_,
            order_=self.order,
            lambda_=self.lambda_,
            smoothness_=self.smoothness_,
            metadata_={"model": "pure_penalized"},
        )
        return self

    def forecast(self, steps: int):
        if not hasattr(self, "trend_"):
            raise RuntimeError("Call fit before forecast.")
        return forecast_trend(self.trend_, self.order, 0.0, int(steps))
