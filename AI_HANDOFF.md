# AI Handoff

This repository is `Trend-Estimation`. Recent work focused on making the S&P
500 paper pipeline reproducible and mathematically precise about Guerrero
smoothness versus realized roughness.

## Current State

- The paper subproject lives in `paper/` and compiles to `paper/build/main.pdf`.
- The paper uses section files under `paper/sections/`.
- Generated figures live in `paper/figures/`.
- Generated tables live in `paper/tables/`.
- The main code package lives in `src/trend_estimation/`.
- The conda environment used for validation is `trend_estimation`.

## Important Terminology

- `guerrero_smoothness` is only for finite-difference penalized least-squares
  smoothers:
  - `TrainValidationSelector`
  - `TimeWeightedValidationSelector`
  - `HPTrend`
  - `WhittakerTrend`
- Non-Guerrero methods must leave `guerrero_smoothness` blank/NA:
  - `MovingAverageTrend`
  - `ExponentialSmoothingTrend`
  - `PolynomialTrendForecaster`
- `realized_roughness` is an output diagnostic and can be computed for every
  fitted trend.
- Do not rename realized roughness as smoothness.

## Validation Commands

Run these from the repository root:

```powershell
conda run -n trend_estimation python -m pytest
conda run -n trend_estimation python paper/build_paper.py --ticker ^GSPC --start 2010-01-01 --no-compile
conda run -n trend_estimation latexmk -pdf -interaction=nonstopmode -outdir=paper/build paper/main.tex
```

The last successful pytest run reported `18 passed`.

## Notable Modified Areas

- `paper/` was added as a reproducible paper pipeline.
- `src/trend_estimation/metrics/smoothness.py` adds realized roughness helpers.
- `src/trend_estimation/core/solvers.py` was optimized with cached spectral
  smoothness/lambda calculations.
- `src/trend_estimation/models/penalized_trend.py` uses cached solvers.
- `src/trend_estimation/forecasting/polynomial.py` fixed forecast indexing.
- `src/trend_estimation/plotting/style.py` now uses a LaTeX-like serif font
  stack for figures.

## Caution

The worktree contains many uncommitted changes, including generated artifacts.
Do not revert files unless the user explicitly asks.
