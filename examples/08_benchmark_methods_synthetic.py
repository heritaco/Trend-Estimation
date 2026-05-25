"""Benchmark trend estimators on a synthetic series with known true trend."""

from pathlib import Path

import matplotlib.pyplot as plt
import trend_estimation as td


def main():
    data = td.make_structural_break_series(n_obs=180, noise_std=0.35, random_state=42)
    train_idx, val_idx, test_idx = td.train_val_test_split_indices(len(data.y), 0.55, 0.25)

    models = {
        "penalized_trainval": td.TrainValidationSelector(orders=[1, 2, 3]),
        "timeweighted": td.TimeWeightedValidationSelector(orders=[1, 2, 3]),
        "moving_average_12": td.MovingAverageTrend(window=12),
        "exp_smoothing_025": td.ExponentialSmoothingTrend(alpha=0.25),
        "hp_100": td.HPTrend(lambda_=100.0),
        "whittaker_d2": td.WhittakerTrend(order=2, smoothness=0.75),
        "poly2": td.PolynomialTrendForecaster(degree=2),
    }

    result = td.BenchmarkRunner(models=models, metrics=["MAE", "RMSE", "SMAPE"]).run(
        data.y,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        true_trend=data.true_trend,
    )

    print(td.benchmark_metrics_table(result, phase="test", target="observed_series"))
    print("\nErrors:", result.errors_)

    out_dir = Path("examples/outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    result.metrics_.to_csv(out_dir / "benchmark_methods_synthetic_metrics.csv", index=False)

    fig1 = td.plot_benchmark_metrics(result, metric="RMSE", phase="test", target="observed_series")
    fig1.savefig(out_dir / "benchmark_test_rmse.png", dpi=160)
    plt.close(fig1)

    fig2 = td.plot_train_val_test_split(data.y, train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)
    fig2.savefig(out_dir / "benchmark_train_val_test_split.png", dpi=160)
    plt.close(fig2)


if __name__ == "__main__":
    main()
