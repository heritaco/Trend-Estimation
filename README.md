# Trend Estimation

`trend_estimation` is a research-oriented Python library for penalized trend estimation, temporal validation, forecasting, and downstream decision experiments in time series.

The project now treats two related smoothers as first-class models rather than conflating them.

## Model A: pure finite-difference penalization

For observations \(y\in\mathbb{R}^n\), difference operator \(D_d\), and \(\lambda\ge 0\),

\[
\widehat t_{\lambda,d}
=\arg\min_t\{\|y-t\|_2^2+\lambda\|D_dt\|_2^2\}
=(I+\lambda D_d^\top D_d)^{-1}y.
\]

This model has especially simple analytic sensitivity with respect to \(\lambda\):

\[
\frac{\partial \widehat t}{\partial\lambda}=-SQS y,
\qquad
\frac{\partial^2 \widehat t}{\partial\lambda^2}=2SQSQS y,
\]

where \(Q=D_d^\top D_d\) and \(S=(I+\lambda Q)^{-1}\).

Use `PurePenalizedTrend` when the goal is a transparent quadratic smoother, analytic derivatives, or experiments with differentiable hyperparameter selection.

## Model B: Guerrero-style smoother with drift

The original project is retained as a separate model:

\[
\widehat\tau_{\lambda,d}
=(I+\lambda D_d^\top D_d)^{-1}
\left(y+\lambda\widehat m D_d^\top\mathbf 1\right),
\]

with \(\widehat m\) estimated from the fitted trend. Because \(\widehat m=\widehat m(\lambda)\), its full derivative requires differentiating the coupled fixed-point/system rather than copying the pure-model derivative.

Use `GuerreroTrend` for this formulation. `PenalizedTrend` remains as a backward-compatible alias.

## Research direction

The current research program separates three questions:

1. **Estimation:** how should a smooth trend be defined and computed?
2. **Hyperparameter selection:** should \(\lambda\) be chosen by GCV, temporal forecast loss, or analytic/numerical optimization?
3. **Decision-aware selection:** does the \(\lambda\) that minimizes forecast error differ from the \(\lambda\) that maximizes downstream financial utility?

All financial validation must be chronological. Future observations may be used to score a fitted model, but never to construct the fitted trend used to predict them.

## Installation

```bash
conda env create -f environment.yml
conda activate trend_estimation
python -m pip install -e ".[dev,finance]"
pytest
```

## Minimal examples

```python
import trend_estimation as td

series = td.make_polynomial_trend_series(
    n_obs=160, degree=2, noise_std=0.4, random_state=123
)
y = series.y

pure = td.PurePenalizedTrend(order=2, lambda_=100.0).fit(y)
guerrero = td.GuerreroTrend(order=2, smoothness=0.75).fit(y)

sensitivity = td.pure_trend_derivatives(y, order=2, lambda_=100.0)
print(sensitivity.first.shape)
```

For temporal selection:

```python
selector = td.TrainValidationSelector(orders=[1, 2, 3])
selection = selector.fit(y, train_idx=slice(0, 100), val_idx=slice(100, 130))
```

The selector fits on the training prefix and evaluates forecasts on the later validation block; validation observations are not included in the fit.

## Repository layout

```text
src/trend_estimation/   reusable library code
tests/                  unit and mathematical-consistency tests
paper/formal/           compact research manuscript
paper/tutorial/         step-by-step pedagogical companion
experiments/            reproducible research experiments
legacy/                 historically important snapshots
```

The exact pre-refactor state is preserved on branch `archive/pre-research-v2`.

## Status

Implemented:

- finite-difference operators;
- pure penalized smoother;
- Guerrero-style penalized smoother with drift;
- spectral solution machinery;
- analytic first and second \(\lambda\)-derivatives for the pure smoother;
- temporal train/validation selection;
- rolling-origin split generation;
- log-\(\lambda\) numerical optimization utilities;
- polynomial continuation implied by finite differences;
- benchmark models, metrics, plotting, synthetic datasets, and tests.

Next research tasks:

- derive/implement implicit differentiation for the Guerrero fixed-point system;
- implement GCV and blocked/rolling CV selectors;
- benchmark Brent/Newton/grid selection in log-\(\lambda\) space;
- add trend-filtering and state-space baselines;
- formalize financial signals and portfolio objectives;
- compare forecast-optimal and decision-optimal smoothing.

## References

The package is conceptually related to Guerrero (2007), Hodrick--Prescott filtering, Whittaker--Henderson smoothing, smoothing splines, and trend filtering. The software itself should not be interpreted as claiming novelty for those classical components; the research contribution is evaluated at the level of the full estimation-selection-decision pipeline.
