# Roadmap

## Implemented

- Penalized trend estimator.
- Guerrero-style spectral solver with drift estimation.
- Smoothness-index to `lambda_` conversion.
- Train-validation selector.
- Multiple-local-minimum detection.
- Time-weighted validation selector.
- Polynomial extrapolation from trend tail.
- Plotting functions returning `matplotlib.figure.Figure`.
- Error metric tables.
- Synthetic datasets.

## Experimental

- The legacy scripts in `experiments/` remain useful for reproducing earlier exploratory work.
- The precise statistical interpretation of the drift and endpoint behavior should be documented more rigorously before submission.

## Planned

- GCV selector.
- AICc/BIC selectors.
- Rolling-origin and blocked CV.
- AR(1), ARMA/ARIMA and state-space noise models.
- Kalman smoothing.
- Multivariate penalized trends.
- Segmented smoothness.
- Benchmark suite against `statsmodels`, `sktime`, and `pybaselines`.

## Not implemented

- Full multivariate model.
- Full ARIMA/state-space models.
- Automatic JOSS paper generation.
- Production-quality benchmark claims.
