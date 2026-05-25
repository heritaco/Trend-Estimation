from __future__ import annotations

import matplotlib.pyplot as plt


def set_style():
    """Apply a lightweight Matplotlib style without adding seaborn as a dependency."""
    plt.rcParams.update({
        "axes.grid": True,
        "grid.linestyle": ":",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.figsize": (8, 4.5),
        "legend.frameon": True,
    })
