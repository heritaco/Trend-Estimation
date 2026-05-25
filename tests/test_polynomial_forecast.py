import trend_estimation as td


def test_polynomial_forecaster_train_and_trainval():
    data = td.make_polynomial_trend_series(n_obs=70, noise_std=0.1)
    r1 = td.PolynomialTrendForecaster(degree=2, fit_on='train').fit_forecast(data.y, train_idx=slice(0,40), test_idx=slice(50,70), steps=20)
    r2 = td.PolynomialTrendForecaster(degree=2, fit_on='train_validation').fit_forecast(data.y, train_idx=slice(0,40), val_idx=slice(40,50), test_idx=slice(50,70), steps=20)
    assert len(r1.forecast_values_) == 20
    assert len(r2.forecast_values_) == 20
