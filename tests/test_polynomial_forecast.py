import trend_estimation as td
import numpy as np


def test_polynomial_forecaster_train_and_trainval():
    data = td.make_polynomial_trend_series(n_obs=70, noise_std=0.1)
    r1 = td.PolynomialTrendForecaster(degree=2, fit_on='train').fit_forecast(data.y, train_idx=slice(0,40), test_idx=slice(50,70), steps=20)
    r2 = td.PolynomialTrendForecaster(degree=2, fit_on='train_validation').fit_forecast(data.y, train_idx=slice(0,40), val_idx=slice(40,50), test_idx=slice(50,70), steps=20)
    assert len(r1.forecast_values_) == 20
    assert len(r2.forecast_values_) == 20


def test_polynomial_forecaster_uses_target_indices_for_validation():
    y = [float(i) for i in range(10)]

    result = td.PolynomialTrendForecaster(degree=1, fit_on="train").fit_forecast(
        y,
        train_idx=slice(0, 6),
        val_idx=slice(6, 8),
        test_idx=slice(6, 8),
        steps=2,
    )

    assert list(result.forecast_index_) == [6.0, 7.0]
    np.testing.assert_allclose(result.forecast_values_, [6.0, 7.0])
