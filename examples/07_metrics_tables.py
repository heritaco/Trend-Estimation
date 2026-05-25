import trend_estimation as td

data = td.make_polynomial_trend_series(n_obs=150, degree=2, noise_std=0.3, random_state=123)
model = td.PenalizedTrend(order=2, smoothness=0.7).fit(data.y[:100])
forecast = model.forecast(50)
poly = td.PolynomialTrendForecaster(degree=2, fit_on="train")
poly_result = poly.fit_forecast(data.y, train_idx=slice(0, 100), test_idx=slice(100, 150), steps=50)
table = td.compare_error_tables(
    targets={"observed_series": data.y[100:150], "true_trend": data.true_trend[100:150], "polynomial_trend": poly_result.forecast_values_},
    predictions={"PenalizedTrend": forecast},
)
print(table)
fig = td.plot_metrics_table(table)
