from __future__ import annotations

import matplotlib.pyplot as plt


def set_style():
    """Apply a lightweight Matplotlib style without adding seaborn as a dependency."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": [
            "Latin Modern Roman",
            "Computer Modern Roman",
            "CMU Serif",
            "DejaVu Serif",
        ],
        "mathtext.fontset": "cm",
        "mathtext.rm": "serif",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8,
        "axes.grid": True,
        "grid.linestyle": ":",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.figsize": (8, 4.5),
        "legend.frameon": True,
    })
