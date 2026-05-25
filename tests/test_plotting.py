import matplotlib
matplotlib.use('Agg')
import trend_estimation as td


def test_plotting_returns_figures():
    data = td.make_polynomial_trend_series(n_obs=50, noise_std=0.1)
    model = td.PenalizedTrend(order=2, smoothness=0.7).fit(data.y)
    assert td.plot_smoothed_series(data.y, model.trend_).__class__.__name__ == 'Figure'
    assert td.plot_train_val_test_split(data.y, slice(0,30), slice(30,40), slice(40,50)).__class__.__name__ == 'Figure'
    assert td.plot_forecasted_trend(data.y, model.trend_, model.forecast(5)).__class__.__name__ == 'Figure'
