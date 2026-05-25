from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd


def plot_validation_curve(validation_curve, minima=None, *, x="smoothness", y="score", title="Validation curve", save_path=None, ax=None):
    """Plot validation score curves and optional local minima."""
    df = pd.DataFrame(validation_curve)
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    if "order" in df.columns:
        for order, sub in df.groupby("order"):
            sub = sub.sort_values(x)
            ax.plot(sub[x], sub[y], label=f"order={order}")
    else:
        df = df.sort_values(x)
        ax.plot(df[x], df[y], label=y)
    if minima is not None and len(minima) > 0:
        m = pd.DataFrame(minima)
        sx = x if x in m.columns else "smoothness"
        sy = y if y in m.columns else "score"
        ax.scatter(m[sx], m[sy], marker="*", s=80, label="local minima")
    ax.set_title(title)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.legend()
    ax.grid(True, linestyle=":", alpha=0.4)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path)
    return fig
