from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np
import pandas as pd

from trend_estimation.benchmarks.results import BenchmarkResult
from trend_estimation.forecasting.polynomial import PolynomialTrendForecaster
from trend_estimation.metrics.errors import error_metrics_table
from trend_estimation.selection.base import BaseTrendSelector
from trend_estimation.utils.arrays import as_1d_float_array, indices_from_slice_or_array
from trend_estimation.validation.splits import train_val_test_split_indices


class BenchmarkRunner:
    """Run a common train/validation/test protocol for trend estimators.

    Parameters
    ----------
    models:
        Mapping from model name to estimator/selector object. Objects can be
        ordinary estimators with ``fit`` and ``forecast`` methods, trend
        selectors returning a ``SelectionResult``, or ``PolynomialTrendForecaster``.
    metrics:
        Metric names to keep in the final table. The underlying computation uses
        ``error_metrics_table`` and then filters columns if requested.
    refit_on_train_val:
        If ``True``, fit the selected/fixed model again on train+validation
        before forecasting the test segment. This is the usual final-evaluation
        protocol after validation-based selection.
    continue_on_error:
        If ``True``, store model errors and continue the benchmark.
    """

    def __init__(
        self,
        models: dict[str, Any],
        metrics: list[str] | tuple[str, ...] | None = None,
        *,
        refit_on_train_val: bool = True,
        continue_on_error: bool = True,
    ):
        self.models = dict(models)
        self.metrics = list(metrics) if metrics is not None else ["MAE", "MSE", "RMSE", "MAPE", "SMAPE"]
        self.refit_on_train_val = bool(refit_on_train_val)
        self.continue_on_error = bool(continue_on_error)

    def run(
        self,
        y,
        *,
        train_idx=None,
        val_idx=None,
        test_idx=None,
        true_trend=None,
        frac_train: float = 0.6,
        frac_val: float = 0.2,
    ) -> BenchmarkResult:
        y = as_1d_float_array(y)
        n_obs = y.size
        if train_idx is None or val_idx is None or test_idx is None:
            train_idx, val_idx, test_idx = train_val_test_split_indices(n_obs, frac_train, frac_val)

        train_pos = indices_from_slice_or_array(train_idx, n_obs)
        val_pos = indices_from_slice_or_array(val_idx, n_obs)
        test_pos = indices_from_slice_or_array(test_idx, n_obs)
        train_val_pos = np.r_[train_pos, val_pos]

        y_train = y[train_pos]
        y_val = y[val_pos]
        y_test = y[test_pos]
        true_trend_arr = None if true_trend is None else as_1d_float_array(true_trend)

        metric_tables: list[pd.DataFrame] = []
        forecasts: dict[str, dict[str, Any]] = {}
        fitted_trends: dict[str, dict[str, Any]] = {}
        model_params: dict[str, dict[str, Any]] = {}
        errors: dict[str, str] = {}

        for name, model in self.models.items():
            try:
                out = self._run_one_model(
                    name=name,
                    model=model,
                    y=y,
                    y_train=y_train,
                    y_val=y_val,
                    y_test=y_test,
                    train_idx=train_idx,
                    val_idx=val_idx,
                    test_idx=test_idx,
                    train_pos=train_pos,
                    val_pos=val_pos,
                    test_pos=test_pos,
                    train_val_pos=train_val_pos,
                    true_trend=true_trend_arr,
                )
                metric_tables.extend(out["tables"])
                forecasts[name] = out["forecasts"]
                fitted_trends[name] = out["fitted_trends"]
                model_params[name] = out["params"]
            except Exception as exc:  # pragma: no cover - exercised by user workflows
                errors[name] = f"{type(exc).__name__}: {exc}"
                if not self.continue_on_error:
                    raise

        metrics_df = pd.concat(metric_tables, ignore_index=True) if metric_tables else pd.DataFrame()
        if not metrics_df.empty:
            front = ["model", "phase", "target", "n_obs"]
            keep_metrics = [m for m in self.metrics if m in metrics_df.columns]
            metrics_df = metrics_df[[c for c in front if c in metrics_df.columns] + keep_metrics]

        result = BenchmarkResult(
            metrics_=metrics_df,
            forecasts_=forecasts,
            fitted_trends_=fitted_trends,
            model_params_=model_params,
            errors_=errors,
            metadata_={
                "n_obs": n_obs,
                "n_train": len(train_pos),
                "n_val": len(val_pos),
                "n_test": len(test_pos),
                "refit_on_train_val": self.refit_on_train_val,
            },
        )
        self.result_ = result
        return result

    def _run_one_model(self, **kwargs) -> dict[str, Any]:
        model = kwargs["model"]
        if isinstance(model, PolynomialTrendForecaster):
            return self._run_polynomial(**kwargs)
        if isinstance(model, BaseTrendSelector):
            return self._run_selector(**kwargs)
        return self._run_estimator(**kwargs)

    def _run_selector(self, **kwargs) -> dict[str, Any]:
        name = kwargs["name"]
        selector = deepcopy(kwargs["model"])
        y = kwargs["y"]
        y_val = kwargs["y_val"]
        y_test = kwargs["y_test"]
        train_idx = kwargs["train_idx"]
        val_idx = kwargs["val_idx"]
        train_pos = kwargs["train_pos"]
        val_pos = kwargs["val_pos"]
        test_pos = kwargs["test_pos"]
        train_val_pos = kwargs["train_val_pos"]
        true_trend = kwargs["true_trend"]

        selection = selector.fit(y, train_idx=train_idx, val_idx=val_idx, test_idx=kwargs["test_idx"])
        val_forecast = selection.best_model_.forecast(len(y_val))
        val_trend_target = None if true_trend is None else true_trend[val_pos]

        params = {
            "kind": "selector",
            "best_order": getattr(selection, "best_order_", None),
            "guerrero_smoothness": getattr(selection, "best_smoothness_", None),
            "best_lambda": getattr(selection, "best_lambda_", None),
            "best_score": getattr(selection, "best_score_", None),
        }

        if self.refit_on_train_val:
            final_model = deepcopy(selection.best_model_).fit(y[train_val_pos])
        else:
            final_model = selection.best_model_
        test_forecast = final_model.forecast(len(y_test))
        test_trend_target = None if true_trend is None else true_trend[test_pos]

        return self._pack_outputs(
            name=name,
            val_forecast=val_forecast,
            test_forecast=test_forecast,
            y_val=y_val,
            y_test=y_test,
            val_trend_target=val_trend_target,
            test_trend_target=test_trend_target,
            fitted_val_model=selection.best_model_,
            fitted_test_model=final_model,
            params=params,
        )

    def _run_estimator(self, **kwargs) -> dict[str, Any]:
        name = kwargs["name"]
        model = deepcopy(kwargs["model"])
        original_model = deepcopy(model)
        y = kwargs["y"]
        y_train = kwargs["y_train"]
        y_val = kwargs["y_val"]
        y_test = kwargs["y_test"]
        val_pos = kwargs["val_pos"]
        test_pos = kwargs["test_pos"]
        train_val_pos = kwargs["train_val_pos"]
        true_trend = kwargs["true_trend"]

        val_model = model.fit(y_train)
        val_forecast = val_model.forecast(len(y_val))
        if self.refit_on_train_val:
            final_model = original_model.fit(y[train_val_pos])
        else:
            final_model = val_model
        test_forecast = final_model.forecast(len(y_test))

        return self._pack_outputs(
            name=name,
            val_forecast=val_forecast,
            test_forecast=test_forecast,
            y_val=y_val,
            y_test=y_test,
            val_trend_target=None if true_trend is None else true_trend[val_pos],
            test_trend_target=None if true_trend is None else true_trend[test_pos],
            fitted_val_model=val_model,
            fitted_test_model=final_model,
            params={"kind": "estimator", **original_model.get_params()},
        )

    def _run_polynomial(self, **kwargs) -> dict[str, Any]:
        name = kwargs["name"]
        model = deepcopy(kwargs["model"])
        y = kwargs["y"]
        y_val = kwargs["y_val"]
        y_test = kwargs["y_test"]
        train_idx = kwargs["train_idx"]
        val_idx = kwargs["val_idx"]
        test_idx = kwargs["test_idx"]
        val_pos = kwargs["val_pos"]
        test_pos = kwargs["test_pos"]
        true_trend = kwargs["true_trend"]

        val_poly = PolynomialTrendForecaster(degree=model.degree, fit_on="train")
        val_result = val_poly.fit_forecast(y, train_idx=train_idx, val_idx=val_idx, test_idx=val_idx, steps=len(y_val))
        val_forecast = val_result.forecast_values_

        test_poly = PolynomialTrendForecaster(degree=model.degree, fit_on="train_validation")
        test_result = test_poly.fit_forecast(y, train_idx=train_idx, val_idx=val_idx, test_idx=test_idx, steps=len(y_test))
        test_forecast = test_result.forecast_values_

        return self._pack_outputs(
            name=name,
            val_forecast=val_forecast,
            test_forecast=test_forecast,
            y_val=y_val,
            y_test=y_test,
            val_trend_target=None if true_trend is None else true_trend[val_pos],
            test_trend_target=None if true_trend is None else true_trend[test_pos],
            fitted_val_model=val_poly,
            fitted_test_model=test_poly,
            params={"kind": "polynomial_forecaster", "degree": model.degree},
        )

    def _pack_outputs(
        self,
        *,
        name: str,
        val_forecast,
        test_forecast,
        y_val,
        y_test,
        val_trend_target,
        test_trend_target,
        fitted_val_model,
        fitted_test_model,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        tables = []
        tables.append(self._metric_table(y_val, val_forecast, model=name, phase="validation", target="observed_series"))
        tables.append(self._metric_table(y_test, test_forecast, model=name, phase="test", target="observed_series"))
        if val_trend_target is not None:
            tables.append(self._metric_table(val_trend_target, val_forecast, model=name, phase="validation", target="true_trend"))
        if test_trend_target is not None:
            tables.append(self._metric_table(test_trend_target, test_forecast, model=name, phase="test", target="true_trend"))

        return {
            "tables": tables,
            "forecasts": {"validation": np.asarray(val_forecast), "test": np.asarray(test_forecast)},
            "fitted_trends": {
                "validation_model": getattr(fitted_val_model, "trend_", getattr(fitted_val_model, "fitted_values_", None)),
                "test_model": getattr(fitted_test_model, "trend_", getattr(fitted_test_model, "fitted_values_", None)),
            },
            "params": params,
        }

    @staticmethod
    def _metric_table(y_true, y_pred, *, model: str, phase: str, target: str) -> pd.DataFrame:
        table = error_metrics_table(y_true, y_pred, reference_name=target, model_name=model)
        table = table.rename(columns={"reference": "target"})
        table.insert(1, "phase", phase)
        return table


def run_benchmark(models: dict[str, Any], y, **kwargs) -> BenchmarkResult:
    """Convenience function around :class:`BenchmarkRunner`."""
    runner_kwargs = {k: kwargs.pop(k) for k in list(kwargs.keys()) if k in {"metrics", "refit_on_train_val", "continue_on_error"}}
    return BenchmarkRunner(models=models, **runner_kwargs).run(y, **kwargs)
