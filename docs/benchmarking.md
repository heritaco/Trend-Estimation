# Benchmarking trend estimators

The benchmarking layer compares trend estimators under a common temporal protocol.

## Protocol

1. Fit on train.
2. Forecast validation.
3. If the object is a selector, choose hyperparameters using validation.
4. Refit on train + validation.
5. Forecast test.
6. Report validation and test metrics.

## Current baseline models

- `MovingAverageTrend`
- `ExponentialSmoothingTrend`
- `HPTrend`
- `WhittakerTrend`
- `PolynomialTrendForecaster`
- `TrainValidationSelector`
- `TimeWeightedValidationSelector`

## Example

```python
import trend_estimation as td

data = td.make_structural_break_series(n_obs=180)
train_idx, val_idx, test_idx = td.train_val_test_split_indices(len(data.y))

models = td.default_benchmark_models()
result = td.BenchmarkRunner(models).run(
    data.y,
    train_idx=train_idx,
    val_idx=val_idx,
    test_idx=test_idx,
    true_trend=data.true_trend,
)

print(result.metrics_)
```

## Planned methods

Kalman filters, ARIMA trend proxies, STL/LOESS, and smoothing splines remain planned and should not be reported as implemented until added and tested.
