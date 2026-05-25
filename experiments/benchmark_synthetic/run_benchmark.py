from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import trend_estimation as td


def make_data(scenario: str, n_obs: int, noise_std: float, seed: int):
    if scenario == "polynomial":
        return td.make_polynomial_trend_series(n_obs=n_obs, degree=2, noise_std=noise_std, random_state=seed)
    if scenario == "piecewise":
        return td.make_piecewise_trend_series(n_obs=n_obs, noise_std=noise_std, random_state=seed)
    if scenario == "sinusoidal":
        return td.make_sinusoidal_trend_series(n_obs=n_obs, noise_std=noise_std, random_state=seed)
    if scenario == "local_linear":
        return td.make_local_linear_trend_series(n_obs=n_obs, observation_noise_std=noise_std, random_state=seed)
    if scenario == "structural_break":
        return td.make_structural_break_series(n_obs=n_obs, noise_std=noise_std, random_state=seed)
    raise ValueError(f"Unknown scenario: {scenario}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="structural_break", choices=["polynomial", "piecewise", "sinusoidal", "local_linear", "structural_break"])
    parser.add_argument("--n-obs", type=int, default=180)
    parser.add_argument("--noise-std", type=float, default=0.35)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="experiments/benchmark_synthetic/outputs")
    args = parser.parse_args()

    data = make_data(args.scenario, args.n_obs, args.noise_std, args.seed)
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

    result = td.run_benchmark(
        models,
        data.y,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        true_trend=data.true_trend,
        metrics=["MAE", "RMSE", "SMAPE"],
    )

    out_dir = Path(args.output_dir) / args.scenario
    out_dir.mkdir(parents=True, exist_ok=True)
    result.metrics_.to_csv(out_dir / "metrics.csv", index=False)

    fig = td.plot_benchmark_metrics(result, metric="RMSE", phase="test", target="true_trend")
    fig.savefig(out_dir / "rmse_test_true_trend.png", dpi=160)
    plt.close(fig)

    fig = td.plot_train_val_test_split(data.y, train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)
    fig.savefig(out_dir / "train_val_test_split.png", dpi=160)
    plt.close(fig)

    print(result.metrics_)
    if result.errors_:
        print("Errors:", result.errors_)


if __name__ == "__main__":
    main()
