from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from trend_estimation.utils.arrays import indices_from_slice_or_array


def plot_train_val_test_split(y, train_idx, val_idx, test_idx=None, *, index=None, title="Train / validation / test split", save_path=None, ax=None):
    y = np.asarray(y, dtype=float).ravel()
    n = len(y)
    if index is None:
        index = np.arange(n)
    tr = indices_from_slice_or_array(train_idx, n)
    va = indices_from_slice_or_array(val_idx, n)
    te = indices_from_slice_or_array(test_idx, n) if test_idx is not None else np.array([], dtype=int)
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    ax.plot(index[tr], y[tr], label="train", linewidth=1.2)
    ax.plot(index[va], y[va], label="validation", linewidth=1.2)
    if te.size:
        ax.plot(index[te], y[te], label="test", linewidth=1.2)
    for pos in [tr[-1] if tr.size else None, va[-1] if va.size else None]:
        if pos is not None:
            ax.axvline(index[pos], linestyle="--", linewidth=1.0)
    ax.set_title(title)
    ax.set_xlabel("index")
    ax.set_ylabel("value")
    ax.legend()
    ax.grid(True, linestyle=":", alpha=0.4)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path)
    return fig
