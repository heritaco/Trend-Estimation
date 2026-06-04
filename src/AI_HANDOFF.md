# AI Handoff: Source Code

The source package is under `src/trend_estimation/`. Recent code work supported
the paper pipeline without changing the core estimator mathematics or the
train/validation/test protocol.

## Key Changes

- `trend_estimation.metrics.smoothness`
  - Added `roughness_d`.
  - Exposed the helper through `metrics/__init__.py` and package exports.
- `trend_estimation.core.solvers`
  - Optimized Guerrero smoothness and lambda mapping using cached eigenvalues.
  - Preserve the finite-difference penalized least-squares mathematics.
- `trend_estimation.models.penalized_trend`
  - Uses cached solvers for repeated fits.
- `trend_estimation.forecasting.polynomial`
  - Fixed validation/test forecast indexing.
- `trend_estimation.benchmarks.runner`
  - Uses `guerrero_smoothness` in exported params instead of
    `best_smoothness`.
- `trend_estimation.plotting.style`
  - Sets a LaTeX-like serif font stack for Matplotlib figures:
    Latin Modern, Computer Modern, CMU Serif, and DejaVu Serif fallback.

## Tests

Run from the repository root:

```powershell
conda run -n trend_estimation python -m pytest
```

Recent test additions:

- `tests/test_smoothness_metrics.py`
- `tests/test_polynomial_forecast.py`

The last successful test run reported `18 passed`.

## Implementation Constraints

- Keep `guerrero_smoothness` only for finite-difference penalized smoothers.
- Keep `realized_roughness` as the diagnostic for all fitted trends.
- Do not change the benchmark set unless the user explicitly asks.
- Do not change the temporal protocol unless the user explicitly asks.
- Avoid reverting uncommitted user/generated changes.
