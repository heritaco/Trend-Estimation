import trend_estimation as td

data = td.make_polynomial_trend_series(n_obs=120, random_state=123)
fig = td.plot_train_val_test_split(data.y, slice(0, 70), slice(70, 95), slice(95, 120))
