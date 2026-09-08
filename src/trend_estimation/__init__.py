from trend_estimation.core import (
    difference_matrix,
    difference_coefficients,
    lambda_to_smoothness,
    smoothness_to_lambda,
    effective_degrees_of_freedom,
    GuerreroSpectralSolver,
    penalized_solution,
    PurePenalizedSolver,
    PureSolverResult,
    pure_penalized_solution,
    PureTrendDerivatives,
    pure_trend_derivatives,
    mse_from_prediction_derivatives,
    roughness,
)
from trend_estimation.models import (
    BaseTrendEstimator,
    TrendFitResult,
    PenalizedTrend,
    GuerreroTrend,
    PurePenalizedTrend,
    HPTrend,
    WhittakerTrend,
    MovingAverageTrend,
    ExponentialSmoothingTrend,
    PolynomialTrendForecaster,
)
from trend_estimation.selection import (
    BaseTrendSelector,
    BaseSelectionCriterion,
    SelectionResult,
    golden_local,
    find_all_local_minima,
    TrainValidationSelector,
    TimeWeightedValidationSelector,
    SmoothnessSelector,
    LambdaOptimizationResult,
    minimize_over_log_lambda,
    newton_stationary_log_lambda,
)
from trend_estimation.validation import (
    mse_loss,
    weighted_mse,
    TimeWeightedValidationLoss,
    make_time_weights,
    train_val_test_split,
    train_val_test_split_indices,
    RollingOriginSplit,
    rolling_origin_splits,
)
from trend_estimation.forecasting import ForecastResult, forecast_trend, build_polynomial_from_tail
from trend_estimation.metrics import mae, mse, rmse, mape, smape, error_metrics_table, compare_error_tables, roughness_d
from trend_estimation.datasets import (
    SyntheticTrendData,
    make_polynomial_trend_series,
    make_noisy_trend_series,
    make_piecewise_trend_series,
    make_sinusoidal_trend_series,
    make_local_linear_trend_series,
    make_structural_break_series,
)
from trend_estimation.plotting import (
    plot_smoothed_series,
    plot_train_val_test_split,
    plot_validation_curve,
    plot_forecasted_trend,
    plot_metrics_table,
    plot_benchmark_forecasts,
    plot_benchmark_metrics,
    set_style,
)
from trend_estimation.benchmarks import (
    BenchmarkResult,
    BenchmarkRunner,
    run_benchmark,
    benchmark_metrics_table,
    default_benchmark_models,
)

__all__ = [
    "difference_matrix", "difference_coefficients", "lambda_to_smoothness", "smoothness_to_lambda",
    "effective_degrees_of_freedom", "GuerreroSpectralSolver", "penalized_solution",
    "PurePenalizedSolver", "PureSolverResult", "pure_penalized_solution", "PureTrendDerivatives",
    "pure_trend_derivatives", "mse_from_prediction_derivatives", "roughness",
    "BaseTrendEstimator", "TrendFitResult", "PenalizedTrend", "GuerreroTrend", "PurePenalizedTrend",
    "HPTrend", "WhittakerTrend", "MovingAverageTrend", "ExponentialSmoothingTrend",
    "PolynomialTrendForecaster", "BaseTrendSelector", "BaseSelectionCriterion", "SelectionResult",
    "golden_local", "find_all_local_minima", "TrainValidationSelector", "TimeWeightedValidationSelector",
    "SmoothnessSelector", "LambdaOptimizationResult", "minimize_over_log_lambda",
    "newton_stationary_log_lambda", "mse_loss", "weighted_mse", "TimeWeightedValidationLoss",
    "make_time_weights", "train_val_test_split", "train_val_test_split_indices", "RollingOriginSplit",
    "rolling_origin_splits", "ForecastResult", "forecast_trend", "build_polynomial_from_tail",
    "mae", "mse", "rmse", "mape", "smape", "error_metrics_table", "compare_error_tables", "roughness_d",
    "SyntheticTrendData", "make_polynomial_trend_series", "make_noisy_trend_series", "make_piecewise_trend_series",
    "make_sinusoidal_trend_series", "make_local_linear_trend_series", "make_structural_break_series",
    "plot_smoothed_series", "plot_train_val_test_split", "plot_validation_curve", "plot_forecasted_trend",
    "plot_metrics_table", "plot_benchmark_forecasts", "plot_benchmark_metrics", "set_style",
    "BenchmarkResult", "BenchmarkRunner", "run_benchmark", "benchmark_metrics_table", "default_benchmark_models",
]

__version__ = "0.2.0a0"
