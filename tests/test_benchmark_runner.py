import trend_estimation as td


def test_benchmark_runner_synthetic():
    data = td.make_sinusoidal_trend_series(n_obs=120, noise_std=0.2, random_state=1)
    train_idx, val_idx, test_idx = td.train_val_test_split_indices(len(data.y), 0.55, 0.25)
    models = {
        "ma": td.MovingAverageTrend(window=8),
        "exp": td.ExponentialSmoothingTrend(alpha=0.25),
        "hp": td.HPTrend(lambda_=100.0),
        "whittaker": td.WhittakerTrend(order=2, smoothness=0.7),
        "poly2": td.PolynomialTrendForecaster(degree=2),
    }
    result = td.BenchmarkRunner(models=models, metrics=["MAE", "RMSE", "SMAPE"]).run(
        data.y,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        true_trend=data.true_trend,
    )
    assert not result.metrics_.empty
    assert {"model", "phase", "target", "n_obs", "MAE", "RMSE", "SMAPE"}.issubset(result.metrics_.columns)
    assert "test" in set(result.metrics_["phase"])
    assert "true_trend" in set(result.metrics_["target"])
    assert set(result.forecasts_.keys()) == set(models.keys())


def test_default_benchmark_models_runs_small_subset():
    data = td.make_polynomial_trend_series(n_obs=100, noise_std=0.3, random_state=2)
    models = {
        "moving_average_20": td.default_benchmark_models()["moving_average_20"],
        "exp_smoothing_02": td.default_benchmark_models()["exp_smoothing_02"],
    }
    result = td.run_benchmark(models, data.y, true_trend=data.true_trend)
    assert not result.metrics_.empty
