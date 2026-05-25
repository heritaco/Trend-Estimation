# Synthetic benchmark experiment

This experiment compares trend estimators on synthetic series where the true trend is known.

Run from the repository root:

```powershell
python experiments/benchmark_synthetic/run_benchmark.py --scenario structural_break
python experiments/benchmark_synthetic/run_benchmark.py --scenario sinusoidal
python experiments/benchmark_synthetic/run_benchmark.py --scenario local_linear
```

Outputs are written to:

```text
experiments/benchmark_synthetic/outputs/
```
