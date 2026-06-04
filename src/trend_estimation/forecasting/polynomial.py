from __future__ import annotations

import numpy as np
from .base import ForecastResult
from trend_estimation.utils.arrays import as_1d_float_array, indices_from_slice_or_array


class PolynomialTrendForecaster:
    """Polynomial benchmark forecaster for diagnostics and comparisons."""

    def __init__(self, degree: int = 2, fit_on: str = "train"):
        self.degree = int(degree)
        self.fit_on = fit_on

    def _fit_indices(self, n_obs: int, train_idx=None, val_idx=None, test_idx=None) -> np.ndarray:
        if self.fit_on == "train":
            if train_idx is None:
                raise ValueError("train_idx is required when fit_on='train'.")
            return indices_from_slice_or_array(train_idx, n_obs)
        if self.fit_on == "validation":
            if val_idx is None:
                raise ValueError("val_idx is required when fit_on='validation'.")
            return indices_from_slice_or_array(val_idx, n_obs)
        if self.fit_on == "train_validation":
            if train_idx is None or val_idx is None:
                raise ValueError("train_idx and val_idx are required when fit_on='train_validation'.")
            return np.r_[indices_from_slice_or_array(train_idx, n_obs), indices_from_slice_or_array(val_idx, n_obs)]
        if self.fit_on == "all_observed":
            return np.arange(n_obs)
        raise ValueError(f"Unknown fit_on={self.fit_on!r}.")

    def fit_forecast(self, y, *, train_idx=None, val_idx=None, test_idx=None, steps: int | None = None) -> ForecastResult:
        y = as_1d_float_array(y)
        n_obs = y.size
        fit_idx = self._fit_indices(n_obs, train_idx, val_idx, test_idx)
        x_fit = fit_idx.astype(float)
        self.coefficients_ = np.polyfit(x_fit, y[fit_idx], deg=self.degree)
        poly = np.poly1d(self.coefficients_)
        fitted_values = poly(np.arange(n_obs, dtype=float))
        if steps is None:
            if test_idx is not None:
                steps = len(indices_from_slice_or_array(test_idx, n_obs))
            else:
                steps = 1
        steps = int(steps)
        target_idx = None
        if test_idx is not None:
            target_idx = indices_from_slice_or_array(test_idx, n_obs)
        elif self.fit_on == "train" and val_idx is not None:
            target_idx = indices_from_slice_or_array(val_idx, n_obs)

        if target_idx is not None and target_idx.size > 0:
            forecast_index = target_idx[:steps].astype(float)
            if forecast_index.size < steps:
                extra = np.arange(
                    target_idx[-1] + 1,
                    target_idx[-1] + 1 + steps - forecast_index.size,
                    dtype=float,
                )
                forecast_index = np.r_[forecast_index, extra]
        else:
            start = int(np.max(fit_idx)) + 1
            forecast_index = np.arange(start, start + steps, dtype=float)
        forecast_values = poly(forecast_index)
        self.degree_ = self.degree
        self.fit_on_ = self.fit_on
        self.fitted_values_ = np.asarray(fitted_values, dtype=float)
        self.forecast_values_ = np.asarray(forecast_values, dtype=float)
        self.forecast_index_ = forecast_index
        self.residuals_ = y - self.fitted_values_
        return ForecastResult(
            fitted_values_=self.fitted_values_,
            forecast_values_=self.forecast_values_,
            forecast_index_=self.forecast_index_,
            origin_=self.fit_on,
            steps_=steps,
            model_name_="PolynomialTrendForecaster",
            metadata_={"degree": self.degree, "fit_on": self.fit_on},
        )
