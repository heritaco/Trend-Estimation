import trend_estimation as td

data = td.make_polynomial_trend_series(n_obs=140, degree=2, noise_std=0.3, random_state=123)
selector = td.TrainValidationSelector(orders=[1, 2, 3], smoothness_grid=[0.1, 0.3, 0.5, 0.7, 0.9])
selection = selector.fit(data.y, train_idx=slice(0, 90), val_idx=slice(90, 115))
print(selection.best_order_, selection.best_smoothness_, selection.best_score_)
fig = td.plot_validation_curve(selection.validation_curve_, minima=selection.all_minima_)
