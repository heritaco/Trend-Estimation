from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


def plot_forecasted_trend(y, fitted_trend, forecast_trend, *, train_idx=None, val_idx=None, test_idx=None, index=None, forecast_index=None, origin="train_end", title="Forecasted trend", save_path=None, ax=None):
    y = np.asarray(y, dtype=float).ravel()
    fitted_trend = np.asarray(fitted_trend, dtype=float).ravel()
    forecast_trend = np.asarray(forecast_trend, dtype=float).ravel()
    if index is None:
        index = np.arange(len(y))
    if forecast_index is None:
        forecast_index = np.arange(len(y), len(y) + len(forecast_trend))
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    ax.plot(index, y, label="observed", linewidth=1.0)
    ax.plot(index[:len(fitted_trend)], fitted_trend, label="fitted trend", linewidth=2.0)
    ax.plot(forecast_index, forecast_trend, label=f"forecast ({origin})", linestyle="--", linewidth=2.0)
    ax.set_title(title)
    ax.set_xlabel("index")
    ax.set_ylabel("value")
    ax.legend()
    ax.grid(True, linestyle=":", alpha=0.4)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path)
    return fig
