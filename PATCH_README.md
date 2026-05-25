# trend_estimation benchmark patch

This patch adds a benchmarking layer for comparative trend-estimation experiments.
It is intended to be copied on top of the current `Trend-Estimation` project.

## What it adds

- Baseline models:
  - `MovingAverageTrend`
  - `ExponentialSmoothingTrend`
  - `HPTrend`
  - `WhittakerTrend`
- Benchmarking layer:
  - `BenchmarkRunner`
  - `BenchmarkResult`
  - `run_benchmark`
  - `benchmark_metrics_table`
  - `plot_benchmark_forecasts`
  - `plot_benchmark_metrics`
- Synthetic datasets:
  - `make_sinusoidal_trend_series`
  - `make_local_linear_trend_series`
  - `make_structural_break_series`
- Tests:
  - `tests/test_baseline_models.py`
  - `tests/test_benchmark_runner.py`
- Example:
  - `examples/08_benchmark_methods_synthetic.py`
- Experiment:
  - `experiments/benchmark_synthetic/run_benchmark.py`

## How to apply

From the root of your current project:

```powershell
# Option 1: manually copy the folders in this patch over the repo root.
# Option 2: from PowerShell, after unzipping this patch:
Copy-Item -Recurse -Force .\trend_estimation_benchmark_patch\* .\
```

Then reinstall and test:

```powershell
conda activate trend_estimation
python -m pip install -e ".[dev]"
python -m pytest
python examples/08_benchmark_methods_synthetic.py
```

## Notes

This patch intentionally does not implement Kalman, ARIMA, STL, or LOESS yet. Those should remain in the roadmap until the baseline layer is stable.
