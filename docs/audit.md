# Repository audit

This template was generated from a cleaned project and four uploaded legacy scripts.

## Preserved material

- `legacy/original_project_snapshot/py_10_multiple_minima.py`
- `legacy/original_project_snapshot/py_11_multiple_minima.py`
- `legacy/original_project_snapshot/py12_timeweighted.py`
- `legacy/original_project_snapshot/style.py`

## Code classified as central

The central reusable components extracted into `src/trend_estimation` are:

1. `difference_matrix` for finite-difference penalties.
2. A spectral Guerrero-style solver based on eigendecomposition of `D.T @ D`.
3. `lambda_from_s` / smoothness-index conversion.
4. Forecasting by the polynomial implied by constant `d`-th differences.
5. Local-minimum detection over smoothness grids.
6. Train-validation and time-weighted validation selectors.

## Experiments

The legacy experiment scripts were copied to:

- `experiments/06_multiple_minima_trainval/`
- `experiments/07_timeweighted_validation_loss/`

Each folder includes an `outputs/` directory for reproducible outputs.

## Outputs/results

No large output files were generated in this template. Future generated figures/tables should go under `experiments/*/outputs/`.

## Obsolete or risky patterns isolated from library code

- `plt.show()` inside reusable functions.
- CSV writes inside plotting functions.
- Downloading Yahoo Finance data as part of the core logic.
- Hard-coded ticker examples.
- Mixing solver, validation, plotting and output export in one script.

The new library code avoids those patterns.
