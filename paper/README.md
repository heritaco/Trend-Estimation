# S&P 500 Comparative Paper Subproject

This folder contains a reproducible comparative study that uses the local
`trend_estimation` Python library as an installed dependency.

The separation is intentional:

- `src/trend_estimation/` contains reusable library code.
- `paper/` contains S&P 500-specific data handling, experiments, figures, tables, and LaTeX.

The data source is `yfinance`, used for research and educational purposes. The
pipeline caches downloaded data locally so the study can be rerun without
downloading every time.

## Setup

From the repository root:

```bash
conda activate trend_estimation
python -m pip install -e ".[dev,finance]"
```

## Build

```bash
python paper/build_paper.py --ticker ^GSPC --start 2010-01-01 --no-compile
```

Useful options:

```bash
python paper/build_paper.py --skip-download --no-compile
python paper/build_paper.py --force-download --no-compile
```

The default cache files for the S&P 500 study are:

```text
paper/data/raw/sp500_2010_present.csv
paper/data/processed/sp500_log_prices.csv
```

## LaTeX

If `latexmk` is installed, compile from the repository root with:

```bash
latexmk -pdf -interaction=nonstopmode -outdir=paper/build paper/main.tex
```

Fallback sequence:

```bash
pdflatex -output-directory=paper/build paper/main.tex
bibtex paper/build/main
pdflatex -output-directory=paper/build paper/main.tex
pdflatex -output-directory=paper/build paper/main.tex
```

The master build script attempts compilation only when `--no-compile` is not
passed and `latexmk` is available.

## Outputs

The comparison writes CSV and LaTeX tables to `paper/tables/`, figures to
`paper/figures/`, and intermediate experiment artifacts to `paper/outputs/`.

The real S&P 500 data do not provide an observed true trend. The evaluation is
therefore based on out-of-sample forecasts of the observed log-price series and
on realized roughness diagnostics, not on true-trend recovery. Guerrero
smoothness is reported only for finite-difference penalized smoothers where the
index is mathematically native.
