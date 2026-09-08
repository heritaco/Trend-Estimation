# Papers

This project maintains two complementary manuscripts built from the same tested codebase.

## `formal/`

The research manuscript. It should be concise, literature-grounded, and organized around claims that can be supported by derivations or reproducible experiments. It is the only manuscript intended for eventual journal submission.

Current working theme: penalized trend estimation, chronological smoothness selection, and the comparison between forecast-optimal and downstream decision-optimal smoothing in financial time series.

## `tutorial/`

A step-by-step companion for readers who do not already know penalized smoothing, matrix derivatives, temporal validation, or numerical hyperparameter optimization. It may contain slower derivations, intuition, examples, implementation notes, and warnings that would be too pedagogical for the formal article.

The tutorial must remain mathematically consistent with the formal manuscript and with `src/trend_estimation/`. It should never introduce a different estimator merely to simplify exposition without explicitly saying so.

## Model naming convention

Two estimators are first-class throughout both papers:

1. **Pure penalized trend**

   \[
   \widehat t_{\lambda,d}=(I+\lambda D_d^\top D_d)^{-1}y.
   \]

   Implemented by `PurePenalizedTrend` / `PurePenalizedSolver`.

2. **Guerrero-style penalized trend with drift**

   \[
   \widehat\tau_{\lambda,d}
   =(I+\lambda D_d^\top D_d)^{-1}
   \left(y+\lambda\widehat mD_d^\top\mathbf1\right).
   \]

   Implemented by `GuerreroTrend` / `GuerreroSpectralSolver`. `PenalizedTrend` remains a compatibility name for the older API.

The distinction matters because the Guerrero drift is re-estimated from the fitted trend and therefore depends on \(\lambda\). The analytic derivative implemented for the pure smoother must not be reported as the derivative of the Guerrero formulation.

## Historical S&P 500 comparison

The previous root-level S&P 500 manuscript and its sections are retained temporarily as source material while the two new manuscripts absorb anything still useful. Its exact pre-refactor state is permanently preserved on branch `archive/pre-research-v2`.

Generated LaTeX files and large regenerable outputs are not source material and are ignored by Git going forward.

## Build

From the repository root:

```bash
latexmk -pdf -interaction=nonstopmode -outdir=paper/build paper/formal/main.tex
latexmk -pdf -interaction=nonstopmode -outdir=paper/build paper/tutorial/main.tex
```

The reusable estimators must be implemented and tested in `src/`; paper-specific scripts may orchestrate experiments but should not duplicate estimator mathematics.
