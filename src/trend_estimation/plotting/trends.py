from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


def plot_smoothed_series(y=None, trend=None, *, index=None, residuals=None, title="Smoothed trend", show_residuals=False, save_path=None, ax=None):
    """Plot observed series and smoothed trend. Returns a Matplotlib Figure."""
    if trend is None and hasattr(y, "trend_"):
        result = y
        y = getattr(result, "y_", None)
        trend = result.trend_
        residuals = getattr(result, "residuals_", residuals)
    y = np.asarray(y, dtype=float).ravel()
    trend = np.asarray(trend, dtype=float).ravel()
    if index is None:
        index = np.arange(len(y))
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    ax.plot(index[:len(y)], y, label="observed", linewidth=1.0)
    ax.plot(index[:len(trend)], trend, label="trend", linewidth=2.0)
    ax.set_title(title)
    ax.set_xlabel("index")
    ax.set_ylabel("value")
    ax.legend()
    ax.grid(True, linestyle=":", alpha=0.4)
    if show_residuals and residuals is not None:
        ax2 = ax.twinx()
        ax2.plot(index[:len(residuals)], residuals, label="residuals", linewidth=0.8, alpha=0.5)
        ax2.set_ylabel("residual")
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path)
    return fig
