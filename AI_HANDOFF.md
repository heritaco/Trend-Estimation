# AI Handoff

This repository is `Trend-Estimation`. The current research direction generalizes the original Guerrero-style project into an experimental library for penalized trend estimation, chronological hyperparameter selection, analytic sensitivity, and eventual decision-aware financial optimization.

## Safety snapshot

The exact repository state before the research-v2 refactor is preserved on branch:

`archive/pre-research-v2`

Do not rewrite that branch.

## Canonical models

### Pure penalized trend

\[
\widehat t_{\lambda,d}=(I+\lambda D_d^\top D_d)^{-1}y.
\]

Code:

- `src/trend_estimation/core/pure.py`
- `src/trend_estimation/models/pure_penalized.py`
- `src/trend_estimation/core/derivatives.py`

The implemented derivatives with respect to lambda belong to this model.

### Guerrero trend with estimated drift

\[
\widehat\tau_{\lambda,d}
=(I+\lambda D_d^\top D_d)^{-1}
\left(y+\lambda\widehat mD_d^\top\mathbf1\right).
\]

Code:

- `src/trend_estimation/core/solvers.py`
- `src/trend_estimation/models/penalized_trend.py`
- explicit research name: `GuerreroTrend`
- backward-compatible name: `PenalizedTrend`

Important: the solver re-estimates `m_hat` from the fitted trend. Therefore `m_hat = m_hat(lambda)` and the pure-model derivative cannot be copied to the Guerrero model. Full Guerrero sensitivity is a pending implicit-differentiation task.

## Validation invariant

Financial experiments must be chronological. At forecast origin `T`, the fit may use only observations available through `T`. Future observations may score a forecast but may never participate in constructing the fitted trend.

Use `rolling_origin_splits` for repeated chronological evaluation.

## Lambda optimization

Prefer `theta = log(lambda)` for numerical optimization. Utilities live in `src/trend_estimation/selection/numerical.py`.

## Papers

Two manuscripts are canonical:

- `paper/formal/main.tex`: compact research paper.
- `paper/tutorial/main.tex`: step-by-step pedagogical companion.

They must describe the same estimators and notation. The tutorial can be slower and more explicit but must not silently simplify the formal model.

The previous S&P 500 manuscript remains in the root of `paper/` temporarily as migration material; its exact original state is also in the archive branch.

## Testing priorities

Before trusting a new optimization result:

1. run the full test suite;
2. compare analytic derivatives with centered finite differences;
3. verify chronological split boundaries;
4. ensure experiments identify whether they use the pure or Guerrero model;
5. separate forecast-loss results from downstream decision/portfolio results.

## Next research tasks

- implement implicit differentiation for the Guerrero fixed-point system;
- implement GCV and rolling/blocked CV selectors;
- compare grid, Brent, and Newton selection in log-lambda space;
- add trend-filtering/state-space baselines;
- define financial trend signals and portfolio objectives;
- test whether `lambda_forecast` and `lambda_decision` differ materially.
