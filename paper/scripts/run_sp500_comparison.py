from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import trend_estimation as td


PAPER_DIR = Path(__file__).resolve().parents[1]
TABLES_DIR = PAPER_DIR / "tables"
OUTPUTS_DIR = PAPER_DIR / "outputs"
DEFAULT_PROCESSED = PAPER_DIR / "data" / "processed" / "sp500_log_prices.csv"

REQUIRED_COLUMNS = [
    "model",
    "phase",
    "target",
    "n_obs",
    "MAE",
    "MSE",
    "RMSE",
    "MAPE",
    "SMAPE",
    "best_order",
    "guerrero_smoothness",
    "best_lambda",
    "window",
    "alpha",
    "degree",
    "notes",
]

GUERRERO_MODELS = {
    "TrainValidationSelector",
    "TimeWeightedValidationSelector",
    "HPTrend",
    "WhittakerTrend",
}


def _metrics_row(
    *,
    model: str,
    phase: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    best_order: float | None = None,
    guerrero_smoothness: float | None = None,
    best_lambda: float | None = None,
    window: float | None = None,
    alpha: float | None = None,
    degree: float | None = None,
    notes: str = "",
) -> dict[str, Any]:
    return {
        "model": model,
        "phase": phase,
        "target": "observed_series",
        "n_obs": int(min(len(y_true), len(y_pred))),
        "MAE": td.mae(y_true, y_pred),
        "MSE": td.mse(y_true, y_pred),
        "RMSE": td.rmse(y_true, y_pred),
        "MAPE": td.mape(y_true, y_pred),
        "SMAPE": td.smape(y_true, y_pred),
        "best_order": best_order,
        "guerrero_smoothness": guerrero_smoothness,
        "best_lambda": best_lambda,
        "window": window,
        "alpha": alpha,
        "degree": degree,
        "notes": notes,
    }


def _param_row(model: str, params: dict[str, Any], notes: str = "") -> dict[str, Any]:
    row = {column: np.nan for column in REQUIRED_COLUMNS}
    row.update({"model": model, "phase": "final", "target": "observed_series", "notes": notes})
    for key in ("best_order", "guerrero_smoothness", "best_lambda"):
        if key in params:
            row[key] = params[key]
    row.update(params)
    return row


def _model_order_for_roughness(params: dict[str, Any]) -> int:
    value = params.get("best_order", params.get("order", 2))
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return 2
    return int(value)


def _base_models() -> dict[str, object]:
    return {
        "TrainValidationSelector": td.TrainValidationSelector(
            orders=[1, 2, 3],
            smoothness_grid=np.linspace(0.05, 0.98, 21),
            detect_multiple_minima=True,
            refine=False,
        ),
        "TimeWeightedValidationSelector": td.TimeWeightedValidationSelector(
            orders=[1, 2, 3],
            smoothness_grid=np.linspace(0.05, 0.98, 21),
            detect_multiple_minima=True,
            refine=False,
            decay=0.95,
        ),
        "HPTrend": td.HPTrend(lambda_=1600.0),
        "WhittakerTrend": td.WhittakerTrend(order=2, smoothness=0.75),
        "MovingAverageTrend": td.MovingAverageTrend(window=60),
        "ExponentialSmoothingTrend": td.ExponentialSmoothingTrend(alpha=0.20),
        "PolynomialTrendForecaster": td.PolynomialTrendForecaster(degree=2),
    }


def _run_selector(
    name: str,
    selector: td.BaseTrendSelector,
    y: np.ndarray,
    train_idx: slice,
    val_idx: slice,
    test_idx: slice,
) -> dict[str, Any]:
    y_train = y[train_idx]
    y_val = y[val_idx]
    y_train_val = y[: test_idx.start]
    y_test = y[test_idx]

    selection = selector.fit(y, train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)
    val_pred = selection.best_model_.forecast(len(y_val))
    final_model = td.PenalizedTrend(
        order=selection.best_order_,
        smoothness=selection.best_smoothness_,
    ).fit(y_train_val)
    test_pred = final_model.forecast(len(y_test))
    params = {
        "kind": "selector",
        "best_order": selection.best_order_,
        "guerrero_smoothness": selection.best_smoothness_,
        "best_lambda": final_model.lambda_,
        "validation_best_lambda": selection.best_lambda_,
        "validation_best_score": selection.best_score_,
    }
    return {
        "validation_prediction": val_pred,
        "test_prediction": test_pred,
        "fitted_trend": final_model.trend_,
        "params": params,
        "validation_curve": selection.validation_curve_,
        "minima": pd.DataFrame(selection.all_minima_),
        "notes": f"Selected on validation after fitting {len(y_train)} training observations.",
    }


def _run_estimator(
    name: str,
    estimator: td.BaseTrendEstimator,
    y: np.ndarray,
    train_idx: slice,
    val_idx: slice,
    test_idx: slice,
) -> dict[str, Any]:
    y_train = y[train_idx]
    y_val = y[val_idx]
    y_train_val = y[: test_idx.start]
    y_test = y[test_idx]

    val_model = deepcopy(estimator).fit(y_train)
    val_pred = val_model.forecast(len(y_val))
    final_model = deepcopy(estimator).fit(y_train_val)
    test_pred = final_model.forecast(len(y_test))
    params = {"kind": "estimator", **deepcopy(estimator).get_params()}
    if name in GUERRERO_MODELS:
        params.setdefault("best_order", getattr(final_model.fit_result_, "order_", None))
        params.setdefault("best_lambda", getattr(final_model.fit_result_, "lambda_", None))
        params["guerrero_smoothness"] = getattr(final_model.fit_result_, "smoothness_", None)
        params.pop("smoothness", None)
    return {
        "validation_prediction": val_pred,
        "test_prediction": test_pred,
        "fitted_trend": final_model.trend_,
        "params": params,
        "validation_curve": None,
        "minima": None,
        "notes": "Fixed-parameter baseline; no validation hyperparameter search.",
    }


def _run_polynomial(
    name: str,
    model: td.PolynomialTrendForecaster,
    y: np.ndarray,
    train_idx: slice,
    val_idx: slice,
    test_idx: slice,
) -> dict[str, Any]:
    val_result = td.PolynomialTrendForecaster(degree=model.degree, fit_on="train").fit_forecast(
        y,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=val_idx,
        steps=val_idx.stop - val_idx.start,
    )
    test_model = td.PolynomialTrendForecaster(degree=model.degree, fit_on="train_validation")
    test_result = test_model.fit_forecast(
        y,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        steps=test_idx.stop - test_idx.start,
    )
    return {
        "validation_prediction": val_result.forecast_values_,
        "test_prediction": test_result.forecast_values_,
        "fitted_trend": test_result.fitted_values_[: test_idx.start],
        "params": {"kind": "polynomial_forecaster", "degree": model.degree},
        "validation_curve": None,
        "minima": None,
        "notes": "Polynomial fitted on train for validation and train+validation for test.",
    }


def run_comparison(
    *,
    processed_csv: Path = DEFAULT_PROCESSED,
    tables_dir: Path = TABLES_DIR,
    outputs_dir: Path = OUTPUTS_DIR,
) -> dict[str, pd.DataFrame]:
    tables_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(processed_csv, parse_dates=["date"])
    y = data["log_price"].to_numpy(dtype=float)
    dates = data["date"]
    train_idx, val_idx, test_idx = td.train_val_test_split_indices(len(y), 0.6, 0.2)
    y_val = y[val_idx]
    y_test = y[test_idx]

    metric_rows: list[dict[str, Any]] = []
    parameter_rows: list[dict[str, Any]] = []
    roughness_rows: list[dict[str, Any]] = []
    forecast_rows: list[pd.DataFrame] = []
    trend_rows: list[pd.DataFrame] = []

    for name, model in _base_models().items():
        if isinstance(model, td.BaseTrendSelector):
            output = _run_selector(name, model, y, train_idx, val_idx, test_idx)
        elif isinstance(model, td.PolynomialTrendForecaster):
            output = _run_polynomial(name, model, y, train_idx, val_idx, test_idx)
        else:
            output = _run_estimator(name, model, y, train_idx, val_idx, test_idx)

        params = output["params"]
        notes = output["notes"]
        metric_rows.append(
            _metrics_row(
                model=name,
                phase="validation",
                y_true=y_val,
                y_pred=output["validation_prediction"],
                best_order=params.get("best_order"),
                guerrero_smoothness=params.get("guerrero_smoothness"),
                best_lambda=params.get("best_lambda"),
                window=params.get("window"),
                alpha=params.get("alpha"),
                degree=params.get("degree"),
                notes=notes,
            )
        )
        metric_rows.append(
            _metrics_row(
                model=name,
                phase="test",
                y_true=y_test,
                y_pred=output["test_prediction"],
                best_order=params.get("best_order"),
                guerrero_smoothness=params.get("guerrero_smoothness"),
                best_lambda=params.get("best_lambda"),
                window=params.get("window"),
                alpha=params.get("alpha"),
                degree=params.get("degree"),
                notes=notes,
            )
        )
        parameter_rows.append(_param_row(name, params, notes=notes))

        order = _model_order_for_roughness(params)
        fitted_trend = np.asarray(output["fitted_trend"], dtype=float)
        roughness_rows.append(
            {
                "model": name,
                "phase": "train_validation",
                "target": "observed_series",
                "n_obs": len(fitted_trend),
                "roughness_order": order,
                "realized_roughness": td.roughness_d(fitted_trend, order=order),
                "test_RMSE": td.rmse(y_test, output["test_prediction"]),
                "best_order": params.get("best_order"),
                "guerrero_smoothness": params.get("guerrero_smoothness"),
                "best_lambda": params.get("best_lambda"),
                "notes": notes,
            }
        )

        forecast_rows.append(
            pd.DataFrame(
                {
                    "model": name,
                    "phase": "validation",
                    "date": dates.iloc[val_idx].to_numpy(),
                    "observed": y_val,
                    "prediction": output["validation_prediction"],
                }
            )
        )
        forecast_rows.append(
            pd.DataFrame(
                {
                    "model": name,
                    "phase": "test",
                    "date": dates.iloc[test_idx].to_numpy(),
                    "observed": y_test,
                    "prediction": output["test_prediction"],
                }
            )
        )
        trend_rows.append(
            pd.DataFrame(
                {
                    "model": name,
                    "date": dates.iloc[: test_idx.start].to_numpy(),
                    "trend": fitted_trend,
                }
            )
        )

        curve = output["validation_curve"]
        if curve is not None:
            safe_name = name.lower().replace("selector", "")
            pd.DataFrame(curve).rename(columns={"smoothness": "guerrero_smoothness"}).to_csv(
                outputs_dir / f"{safe_name}_validation_curve.csv",
                index=False,
            )
            pd.DataFrame(output["minima"]).rename(
                columns={"smoothness": "guerrero_smoothness", "lambda": "best_lambda"}
            ).to_csv(outputs_dir / f"{safe_name}_validation_minima.csv", index=False)

    full_metrics = pd.DataFrame(metric_rows)
    full_metrics = full_metrics[REQUIRED_COLUMNS]
    selection_metrics = full_metrics[full_metrics["phase"] == "validation"].reset_index(drop=True)
    test_metrics = full_metrics[full_metrics["phase"] == "test"].reset_index(drop=True)
    model_parameters = pd.DataFrame(parameter_rows)
    roughness_metrics = pd.DataFrame(roughness_rows)

    selection_metrics.to_csv(tables_dir / "selection_metrics.csv", index=False)
    test_metrics.to_csv(tables_dir / "test_metrics.csv", index=False)
    full_metrics.to_csv(tables_dir / "full_comparison_metrics.csv", index=False)
    model_parameters.to_csv(tables_dir / "model_parameters.csv", index=False)
    roughness_metrics.to_csv(tables_dir / "roughness_metrics.csv", index=False)
    pd.concat(forecast_rows, ignore_index=True).to_csv(outputs_dir / "forecasts.csv", index=False)
    pd.concat(trend_rows, ignore_index=True).to_csv(outputs_dir / "fitted_trends.csv", index=False)
    data.to_csv(outputs_dir / "analysis_series.csv", index=False)

    split_metadata = {
        "n_obs": len(y),
        "n_train": train_idx.stop - train_idx.start,
        "n_validation": val_idx.stop - val_idx.start,
        "n_test": test_idx.stop - test_idx.start,
        "train_start": str(dates.iloc[train_idx.start].date()),
        "train_end": str(dates.iloc[train_idx.stop - 1].date()),
        "validation_start": str(dates.iloc[val_idx.start].date()),
        "validation_end": str(dates.iloc[val_idx.stop - 1].date()),
        "test_start": str(dates.iloc[test_idx.start].date()),
        "test_end": str(dates.iloc[test_idx.stop - 1].date()),
    }
    (outputs_dir / "split_metadata.json").write_text(json.dumps(split_metadata, indent=2))

    return {
        "selection_metrics": selection_metrics,
        "test_metrics": test_metrics,
        "full_comparison_metrics": full_metrics,
        "model_parameters": model_parameters,
        "roughness_metrics": roughness_metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the S&P 500 trend comparison.")
    parser.add_argument("--processed-csv", type=Path, default=DEFAULT_PROCESSED)
    args = parser.parse_args()
    run_comparison(processed_csv=args.processed_csv)
    print("Saved comparison tables and outputs.")


if __name__ == "__main__":
    main()
