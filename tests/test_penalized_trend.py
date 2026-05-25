import trend_estimation as td


def test_penalized_trend_fit_and_forecast():
    data = td.make_polynomial_trend_series(n_obs=60, noise_std=0.1)
    model = td.PenalizedTrend(order=2, smoothness=0.7).fit(data.y)
    assert model.trend_.shape == data.y.shape
    assert model.residuals_.shape == data.y.shape
    assert len(model.forecast(5)) == 5
