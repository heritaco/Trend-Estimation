from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

import trend_estimation as td


PAPER_DIR = Path(__file__).resolve().parents[1]
FIGURES_DIR = PAPER_DIR / "figures"
TABLES_DIR = PAPER_DIR / "tables"
OUTPUTS_DIR = PAPER_DIR / "outputs"


def _save(fig: plt.Figure, stem: str, figures_dir: Path) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures_dir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(figures_dir / f"{stem}.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def _date_series(df: pd.DataFrame, column: str = "date") -> pd.Series:
    return pd.to_datetime(df[column])


def _set_validation_curve_ylim(ax: plt.Axes, curve: pd.DataFrame) -> None:
    """Keep dominated validation panels readable when one order explodes."""
    if "order" not in curve or "score" not in curve:
        return

    scores = pd.to_numeric(curve["score"], errors="coerce")
    orders = curve["order"]
    valid = pd.DataFrame({"order": orders, "score": scores}).dropna()
    if valid.empty:
        return

    order_max = valid.groupby("order")["score"].max().sort_values()
    visible = valid["score"]
    if len(order_max) > 1:
        largest = float(order_max.iloc[-1])
        second_largest = float(order_max.iloc[-2])
        if second_largest > 0 and largest / second_largest > 20:
            visible = valid.loc[valid["order"] != order_max.index[-1], "score"]

    lower = min(0.0, float(visible.min()))
    upper = float(visible.max())
    if upper <= lower:
        return

    padding = 0.08 * (upper - lower)
    ax.set_ylim(lower, upper + padding)


def make_figures(
    *,
    figures_dir: Path = FIGURES_DIR,
    tables_dir: Path = TABLES_DIR,
    outputs_dir: Path = OUTPUTS_DIR,
) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(outputs_dir / "analysis_series.csv", parse_dates=["date"])
    forecasts = pd.read_csv(outputs_dir / "forecasts.csv", parse_dates=["date"])
    trends = pd.read_csv(outputs_dir / "fitted_trends.csv", parse_dates=["date"])
    test_metrics = pd.read_csv(tables_dir / "test_metrics.csv")
    roughness = pd.read_csv(tables_dir / "roughness_metrics.csv")
    split = json.loads((outputs_dir / "split_metadata.json").read_text())

    n_train = int(split["n_train"])
    n_val = int(split["n_validation"])
    n_obs = int(split["n_obs"])
    train_idx = slice(0, n_train)
    val_idx = slice(n_train, n_train + n_val)
    test_idx = slice(n_train + n_val, n_obs)

    td.set_style()
    dates = data["date"].to_numpy()
    y = data["log_price"].to_numpy()

    fig = td.plot_train_val_test_split(
        y,
        train_idx,
        val_idx,
        test_idx,
        index=dates,
        title="S&P 500 log prices: temporal split",
    )
    fig.axes[0].set_ylabel("log price")
    _save(fig, "sp500_train_val_test_split", figures_dir)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(data["date"], data["log_price"], color="black", linewidth=1.0, label="observed")
    for model, sub in trends.groupby("model"):
        ax.plot(sub["date"], sub["trend"], linewidth=1.1, label=model)
    ax.axvline(data["date"].iloc[test_idx.start], color="0.3", linestyle="--", linewidth=1.0)
    ax.set_title("In-sample trend estimates before the test period")
    ax.set_xlabel("date")
    ax.set_ylabel("log price")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, linestyle=":", alpha=0.35)
    fig.tight_layout()
    _save(fig, "sp500_smoothed_trends", figures_dir)

    curve_paths = sorted(outputs_dir.glob("*_validation_curve.csv"))
    fig, axes = plt.subplots(1, max(1, len(curve_paths)), figsize=(6 * max(1, len(curve_paths)), 4))
    if not isinstance(axes, (list, tuple)):
        axes = [axes] if len(curve_paths) == 1 else list(axes.ravel())
    for ax, curve_path in zip(axes, curve_paths):
        minima_path = curve_path.with_name(curve_path.name.replace("_curve", "_minima"))
        curve = pd.read_csv(curve_path)
        minima = pd.read_csv(minima_path) if minima_path.exists() else None
        title = curve_path.stem.replace("_validation_curve", "").replace("_", " ").title()
        td.plot_validation_curve(
            curve,
            minima=minima,
            x="guerrero_smoothness",
            title=title,
            ax=ax,
        )
        _set_validation_curve_ylim(ax, curve)
        ax.set_xlabel("Guerrero smoothness index")
    fig.tight_layout()
    _save(fig, "sp500_validation_curves", figures_dir)

    fig, ax = plt.subplots(figsize=(10, 5))
    train_val = data.iloc[: test_idx.start]
    test = data.iloc[test_idx]
    ax.plot(train_val["date"], train_val["log_price"], color="0.45", linewidth=1.0, label="observed train+validation")
    ax.plot(test["date"], test["log_price"], color="black", linewidth=1.5, label="observed test")
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    for i, (model, sub) in enumerate(forecasts[forecasts["phase"] == "test"].groupby("model")):
        color = color_cycle[i % len(color_cycle)] if color_cycle else None
        past = trends[trends["model"] == model]
        if not past.empty:
            ax.plot(
                past["date"],
                past["trend"],
                color=color,
                linewidth=1.0,
                alpha=0.2,
                label="_nolegend_",
            )
        ax.plot(sub["date"], sub["prediction"], color=color, linewidth=1.1, label=model)
    ax.set_title("Test-set forecasts")
    ax.set_xlabel("date")
    ax.set_ylabel("log price")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, linestyle=":", alpha=0.35)
    fig.tight_layout()
    _save(fig, "sp500_forecasts_test", figures_dir)

    ordered = test_metrics.sort_values("RMSE")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(ordered["model"], ordered["RMSE"], color="#4c78a8")
    ax.set_title("Test RMSE by method")
    ax.set_xlabel("RMSE")
    fig.tight_layout()
    _save(fig, "sp500_test_rmse_barplot", figures_dir)

    ordered = test_metrics.sort_values("SMAPE")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(ordered["model"], ordered["SMAPE"], color="#f58518")
    ax.set_title("Test SMAPE by method")
    ax.set_xlabel("SMAPE")
    fig.tight_layout()
    _save(fig, "sp500_test_smape_barplot", figures_dir)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(roughness["realized_roughness"], roughness["test_RMSE"], color="#54a24b")
    for _, row in roughness.iterrows():
        ax.annotate(
            row["model"],
            (row["realized_roughness"], row["test_RMSE"]),
            fontsize=7,
            xytext=(4, 3),
            textcoords="offset points",
        )
    ax.set_xscale("log")
    ax.set_title("Realized roughness against test RMSE")
    ax.set_xlabel("Realized roughness")
    ax.set_ylabel("test RMSE")
    ax.grid(True, linestyle=":", alpha=0.35)
    fig.tight_layout()
    _save(fig, "sp500_roughness_vs_error", figures_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate paper figures.")
    parser.parse_args()
    make_figures()
    print("Saved figures.")


if __name__ == "__main__":
    main()
