from __future__ import annotations

import numpy as np
import pandas as pd


def _paired(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    n = min(y_true.size, y_pred.size)
    if n == 0:
        raise ValueError("Inputs must contain at least one paired observation.")
    return y_true[:n], y_pred[:n]


def mae(y_true, y_pred) -> float:
    y_true, y_pred = _paired(y_true, y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def mse(y_true, y_pred) -> float:
    y_true, y_pred = _paired(y_true, y_pred)
    return float(np.mean((y_true - y_pred) ** 2))


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mse(y_true, y_pred)))


def mape(y_true, y_pred) -> float:
    y_true, y_pred = _paired(y_true, y_pred)
    denom = np.maximum(np.abs(y_true), np.finfo(float).eps)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)


def smape(y_true, y_pred) -> float:
    y_true, y_pred = _paired(y_true, y_pred)
    denom = np.maximum((np.abs(y_true) + np.abs(y_pred)) / 2.0, np.finfo(float).eps)
    return float(np.mean(np.abs(y_true - y_pred) / denom) * 100.0)


def error_metrics_table(y_true, y_pred, *, reference_name: str = "target", model_name: str = "model") -> pd.DataFrame:
    yt, yp = _paired(y_true, y_pred)
    return pd.DataFrame([{
        "model": model_name,
        "reference": reference_name,
        "n_obs": len(yt),
        "MAE": mae(yt, yp),
        "MSE": mse(yt, yp),
        "RMSE": rmse(yt, yp),
        "MAPE": mape(yt, yp),
        "SMAPE": smape(yt, yp),
    }])


def compare_error_tables(targets: dict[str, object], predictions: dict[str, object]) -> pd.DataFrame:
    tables = []
    for target_name, y_true in targets.items():
        for model_name, y_pred in predictions.items():
            tables.append(error_metrics_table(y_true, y_pred, reference_name=target_name, model_name=model_name))
    if not tables:
        return pd.DataFrame(columns=["target", "model", "n_obs", "MAE", "MSE", "RMSE", "MAPE", "SMAPE"])
    out = pd.concat(tables, ignore_index=True)
    out = out.rename(columns={"reference": "target"})
    return out[["target", "model", "n_obs", "MAE", "MSE", "RMSE", "MAPE", "SMAPE"]]
