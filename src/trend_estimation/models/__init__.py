from trend_estimation.models.base import BaseTrendEstimator, TrendFitResult
from trend_estimation.models.penalized_trend import PenalizedTrend
from trend_estimation.models.hp_filter import HPTrend
from trend_estimation.models.whittaker import WhittakerTrend
from trend_estimation.models.moving_average import MovingAverageTrend
from trend_estimation.models.exponential_smoothing import ExponentialSmoothingTrend
from trend_estimation.forecasting.polynomial import PolynomialTrendForecaster

__all__ = [
    "BaseTrendEstimator",
    "TrendFitResult",
    "PenalizedTrend",
    "HPTrend",
    "WhittakerTrend",
    "MovingAverageTrend",
    "ExponentialSmoothingTrend",
    "PolynomialTrendForecaster",
]
