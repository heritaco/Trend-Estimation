import trend_estimation as td


def test_time_weighted_selector_runs():
    data = td.make_polynomial_trend_series(n_obs=80, noise_std=0.2)
    selector = td.TimeWeightedValidationSelector(orders=[1, 2], smoothness_grid=[0.2, 0.5, 0.8])
    result = selector.fit(data.y, train_idx=slice(0, 50), val_idx=slice(50, 65))
    assert result.best_model_ is not None
