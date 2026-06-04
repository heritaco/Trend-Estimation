# AI Handoff: Paper

The paper is a reproducible S&P 500 trend-estimation comparison. The main file
is `paper/main.tex`; sections are in `paper/sections/`.

## Build Outputs

- Final PDF: `paper/build/main.pdf`
- Figures: `paper/figures/*.pdf` and `paper/figures/*.png`
- Tables: `paper/tables/*.csv` and `paper/tables/*.tex`
- Script outputs: `paper/outputs/`

## Build Commands

From the repository root:

```powershell
conda run -n trend_estimation python paper/build_paper.py --ticker ^GSPC --start 2010-01-01 --no-compile
conda run -n trend_estimation latexmk -pdf -interaction=nonstopmode -outdir=paper/build paper/main.tex
```

To regenerate only figures:

```powershell
conda run -n trend_estimation python paper/scripts/make_figures.py
```

## Recent Paper Changes

- Added a `sections/` structure.
- Added a detailed `algorithm.tex` section explaining the numerical methods:
  finite-difference penalty, spectral solve, Guerrero trace mapping,
  bisection, validation search, time-weighted validation, forecasting, and the
  final protocol.
- Added bibliography entries with DOI URLs where available.
- Added explanations for every figure and table in `sections/results.tex`.
- Wrapped generated table snippets in proper LaTeX `table` environments with
  captions and labels.
- Figure 3 validation curves use a robust y-limit so the order-three curve does
  not flatten the lower-order curves.

## Required Terminology

- Use "Guerrero smoothness index" only for selector-based penalized methods and
  fixed penalized baselines where an implied index exists.
- Use "realized roughness" for the diagnostic computed from fitted trends.
- Do not imply moving average, exponential smoothing, or polynomial forecasting
  has a native Guerrero smoothness parameter.

## Key Files

- `paper/scripts/run_sp500_comparison.py`
- `paper/scripts/make_tables.py`
- `paper/scripts/make_figures.py`
- `paper/sections/algorithm.tex`
- `paper/sections/results.tex`
- `paper/biblio.bib`
