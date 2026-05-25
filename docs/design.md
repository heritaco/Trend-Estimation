# Design document

## Core decomposition

The package separates five layers:

1. `core/`: linear operators, smoothness maps and solvers.
2. `models/`: estimator classes with `fit` and `forecast` methods.
3. `validation/`: losses, temporal weights and split utilities.
4. `selection/`: hyperparameter selection over `order`, `smoothness` and `lambda_`.
5. `plotting/`, `metrics/`, `datasets/`, `forecasting/`: analysis utilities outside the fitting core.

## Origin of the implemented pieces

### From `06 Multiple Minima for TrainVal`

Implemented as:

- `selection.minima.golden_local`
- `selection.minima.find_all_local_minima`
- `selection.train_val.TrainValidationSelector`
- `plotting.validation_curves.plot_validation_curve`

The notebook/script idea became a selection strategy, not a parallel pipeline.

### From `07 Timeweighted Validation Loss`

Implemented as:

- `validation.time_weights.make_time_weights`
- `validation.losses.weighted_mse`
- `selection.time_weighted.TimeWeightedValidationSelector`

The time-weighted logic is now a loss variant reused by the common train-validation selector.

## Adding a new trend model

Subclass `BaseTrendEstimator` and implement:

- `fit(y, X=None)`
- `forecast(steps)`
- optionally `get_params()` and `set_params()`

Then expose it in `src/trend_estimation/__init__.py`.

## Adding a new selection criterion

Subclass `BaseTrendSelector` or add a criterion object with a `score(...)` method. Keep model fitting separate from loss evaluation.

## Adding a new validation loss

Add a pure function or class under `validation/losses.py`. It should accept arrays and return a scalar. It should not fit models or plot.

## Future extensions

- ARIMA/Kalman: add models under `models/` and noise assumptions under `noise/`.
- Multivariate trends: add estimators under `multivariate/`; keep input validation compatible with `(n_obs,)` and `(n_obs, n_series)` arrays.
- Segmented smoothness: add breakpoint and piecewise penalty logic under `segments/`.
