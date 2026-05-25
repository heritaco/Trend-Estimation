import trend_estimation as td

data = td.make_polynomial_trend_series(n_obs=120, random_state=123)
model = td.PenalizedTrend(order=2, smoothness=0.7).fit(data.y[:90])
forecast = model.forecast(30)
fig = td.plot_forecasted_trend(data.y[:90], model.trend_, forecast, origin="train_end")
