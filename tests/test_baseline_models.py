import numpy as np
import trend_estimation as td


def test_baseline_models_fit_and_forecast():
    y = np.linspace(0, 1, 80) + 0.05 * np.sin(np.arange(80))
    models = [
        td.MovingAverageTrend(window=5),
        td.ExponentialSmoothingTrend(alpha=0.3),
        td.HPTrend(lambda_=100.0),
        td.WhittakerTrend(order=2, smoothness=0.7),
    ]
    for model in models:
        fitted = model.fit(y)
        assert fitted.trend_.shape == y.shape
        assert fitted.residuals_.shape == y.shape
        forecast = fitted.forecast(7)
        assert forecast.shape == (7,)
        assert np.all(np.isfinite(forecast))
