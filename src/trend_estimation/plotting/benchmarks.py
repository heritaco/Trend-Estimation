from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from trend_estimation.benchmarks.results import BenchmarkResult


def plot_benchmark_forecasts(
    result: BenchmarkResult,
    y,
    *,
    phase: str = "test",
    max_models: int | None = None,
    title: str = "Benchmark forecasts",
    save_path=None,
    ax=None,
):
    """Plot observed series and benchmark forecasts for one phase."""
    y = np.asarray(y, dtype=float).ravel()
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4.5))
    else:
        fig = ax.figure

    ax.plot(np.arange(len(y)), y, label="observed", linewidth=1.0)
    count = 0
    start = len(y)
    for name, fc_dict in result.forecasts_.items():
        if phase not in fc_dict:
            continue
        forecast = np.asarray(fc_dict[phase], dtype=float).ravel()
        x = np.arange(start, start + len(forecast))
        ax.plot(x, forecast, linestyle="--", linewidth=1.5, label=name)
        count += 1
        if max_models is not None and count >= max_models:
            break
    ax.set_title(title)
    ax.set_xlabel("index")
    ax.set_ylabel("value")
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.legend(fontsize=8)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path)
    return fig


def plot_benchmark_metrics(
    result: BenchmarkResult,
    *,
    metric: str = "RMSE",
    phase: str = "test",
    target: str = "observed_series",
    title: str | None = None,
    save_path=None,
    ax=None,
):
    """Plot a bar chart of one benchmark metric."""
    table = result.metrics_
    if table.empty:
        raise ValueError("BenchmarkResult.metrics_ is empty.")
    sub = table[(table["phase"] == phase) & (table["target"] == target)].copy()
    if metric not in sub.columns:
        raise ValueError(f"Metric {metric!r} not found in result table.")
    sub = sub.sort_values(metric)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4.5))
    else:
        fig = ax.figure
    ax.bar(sub["model"].astype(str), sub[metric].astype(float))
    ax.set_title(title or f"{metric} by model ({phase}, {target})")
    ax.set_xlabel("model")
    ax.set_ylabel(metric)
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, axis="y", linestyle=":", alpha=0.4)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path)
    return fig
