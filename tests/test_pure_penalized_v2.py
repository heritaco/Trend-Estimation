import numpy as np

import trend_estimation as td


def test_pure_lambda_zero_recovers_observations():
    y = np.array([1.0, 2.0, 1.5, 3.0, 2.5])
    fitted = td.PurePenalizedTrend(order=2, lambda_=0.0).fit(y)
    assert np.allclose(fitted.trend_, y)


def test_pure_and_guerrero_are_explicit_distinct_models():
    y = np.linspace(0.0, 2.0, 20) ** 2
    pure = td.PurePenalizedTrend(order=2, lambda_=10.0).fit(y)
    guerrero = td.GuerreroTrend(order=2, lambda_=10.0).fit(y)
    assert pure.trend_.shape == guerrero.trend_.shape == y.shape
    assert pure.fit_result_.metadata_["model"] == "pure_penalized"


def test_pure_forecast_has_zero_order_difference():
    y = np.array([0.0, 1.0, 4.0, 9.0, 16.0, 25.0])
    model = td.PurePenalizedTrend(order=2, lambda_=1.0).fit(y)
    forecast = model.forecast(5)
    joined = np.concatenate([model.trend_[-2:], forecast])
    assert np.allclose(np.diff(joined, n=2), 0.0, atol=1e-10)
