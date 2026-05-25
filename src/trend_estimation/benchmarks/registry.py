from __future__ import annotations

from trend_estimation.forecasting.polynomial import PolynomialTrendForecaster
from trend_estimation.models.exponential_smoothing import ExponentialSmoothingTrend
from trend_estimation.models.hp_filter import HPTrend
from trend_estimation.models.moving_average import MovingAverageTrend
from trend_estimation.models.whittaker import WhittakerTrend
from trend_estimation.selection.time_weighted import TimeWeightedValidationSelector
from trend_estimation.selection.train_val import TrainValidationSelector


def default_benchmark_models() -> dict[str, object]:
    """Return a small default model dictionary for synthetic benchmarks."""
    return {
        "penalized_trainval": TrainValidationSelector(orders=[1, 2, 3]),
        "penalized_timeweighted": TimeWeightedValidationSelector(orders=[1, 2, 3]),
        "moving_average_20": MovingAverageTrend(window=20),
        "exp_smoothing_02": ExponentialSmoothingTrend(alpha=0.2),
        "hp_1600": HPTrend(lambda_=1600.0),
        "whittaker_d2_s075": WhittakerTrend(order=2, smoothness=0.75),
        "poly_degree_2": PolynomialTrendForecaster(degree=2),
    }
