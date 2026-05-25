from trend_estimation.core import (
    difference_matrix,
    difference_coefficients,
    lambda_to_smoothness,
    smoothness_to_lambda,
    effective_degrees_of_freedom,
    GuerreroSpectralSolver,
    penalized_solution,
    roughness,
)
from trend_estimation.models import BaseTrendEstimator, TrendFitResult, PenalizedTrend, HPTrend, PolynomialTrendForecaster
from trend_estimation.selection import (
    BaseTrendSelector,
    BaseSelectionCriterion,
    SelectionResult,
    golden_local,
    find_all_local_minima,
    TrainValidationSelector,
    TimeWeightedValidationSelector,
    SmoothnessSelector,
)
from trend_estimation.validation import mse_loss, weighted_mse, TimeWeightedValidationLoss, make_time_weights, train_val_test_split, train_val_test_split_indices
from trend_estimation.forecasting import ForecastResult, forecast_trend, build_polynomial_from_tail
from trend_estimation.metrics import mae, mse, rmse, mape, smape, error_metrics_table, compare_error_tables
from trend_estimation.datasets import SyntheticTrendData, make_polynomial_trend_series, make_noisy_trend_series, make_piecewise_trend_series
from trend_estimation.plotting import plot_smoothed_series, plot_train_val_test_split, plot_validation_curve, plot_forecasted_trend, plot_metrics_table, set_style

__all__ = [
    "difference_matrix", "difference_coefficients", "lambda_to_smoothness", "smoothness_to_lambda",
    "effective_degrees_of_freedom", "GuerreroSpectralSolver", "penalized_solution", "roughness",
    "BaseTrendEstimator", "TrendFitResult", "PenalizedTrend", "HPTrend", "PolynomialTrendForecaster",
    "BaseTrendSelector", "BaseSelectionCriterion", "SelectionResult", "golden_local", "find_all_local_minima",
    "TrainValidationSelector", "TimeWeightedValidationSelector", "SmoothnessSelector",
    "mse_loss", "weighted_mse", "TimeWeightedValidationLoss", "make_time_weights",
    "train_val_test_split", "train_val_test_split_indices", "ForecastResult", "forecast_trend", "build_polynomial_from_tail",
    "mae", "mse", "rmse", "mape", "smape", "error_metrics_table", "compare_error_tables",
    "SyntheticTrendData", "make_polynomial_trend_series", "make_noisy_trend_series", "make_piecewise_trend_series",
    "plot_smoothed_series", "plot_train_val_test_split", "plot_validation_curve", "plot_forecasted_trend",
    "plot_metrics_table", "set_style",
]

__version__ = "0.1.0"
