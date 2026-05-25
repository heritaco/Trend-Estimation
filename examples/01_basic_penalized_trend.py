import trend_estimation as td

data = td.make_polynomial_trend_series(n_obs=120, degree=2, noise_std=0.25, random_state=123)
model = td.PenalizedTrend(order=2, smoothness=0.75).fit(data.y)
fig = td.plot_smoothed_series(data.y, model.trend_)
print(model.lambda_, model.smoothness_)
