from .base import ForecastResult
from .extrapolation import forecast_trend, build_polynomial_from_tail
from .polynomial import PolynomialTrendForecaster

__all__ = ["ForecastResult", "forecast_trend", "build_polynomial_from_tail", "PolynomialTrendForecaster"]
