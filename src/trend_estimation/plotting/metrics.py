from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd


def plot_metrics_table(metrics_table, *, title="Forecast error metrics", save_path=None):
    df = pd.DataFrame(metrics_table)
    fig, ax = plt.subplots(figsize=(max(6, 1.2 * len(df.columns)), max(2, 0.5 * (len(df) + 1))))
    ax.axis("off")
    ax.set_title(title)
    table = ax.table(cellText=df.round(4).astype(str).values, colLabels=df.columns, loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.2)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path)
    return fig
