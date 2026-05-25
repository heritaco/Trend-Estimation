from .base import BaseTrendEstimator, TrendFitResult
from .penalized_trend import PenalizedTrend
from .hp_filter import HPTrend
from trend_estimation.forecasting.polynomial import PolynomialTrendForecaster

__all__ = ["BaseTrendEstimator", "TrendFitResult", "PenalizedTrend", "HPTrend", "PolynomialTrendForecaster"]
