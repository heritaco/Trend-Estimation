# Research v2 design

## Scientific separation

The repository now distinguishes two estimators that share the same finite-difference penalty matrix but solve different models.

### Pure penalized model

\[
\widehat t_{\lambda,d}=(I+\lambda D_d^\top D_d)^{-1}y.
\]

This is the preferred model for analytic sensitivity experiments because the dependence on \(\lambda\) is explicit. The first two derivatives are implemented in `core/derivatives.py` and must be tested against centered finite differences.

### Guerrero model with drift

\[
\widehat\tau_{\lambda,d}
=(I+\lambda D_d^\top D_d)^{-1}
\left(y+\lambda\widehat mD_d^\top\mathbf1\right).
\]

The original iterative drift estimate is retained in `GuerreroSpectralSolver`. Since \(\widehat m\) is re-estimated from the trend, it is a function of \(\lambda\). A future derivative implementation must differentiate the coupled fixed-point/system, preferably by implicit differentiation. Do not reuse the pure derivative as an approximation without labelling it as such.

## Validation invariant

At every forecast origin \(T\), fitting code may access only observations with indices \(\le T\). Validation/test observations may be read only after the prediction has been produced. Any selector violating this invariant is invalid for the financial experiments.

`rolling_origin_splits` provides expanding or rolling chronological windows. Future selectors should consume these splits rather than inventing independent conventions.

## Hyperparameter optimization

Numerical search should normally use \(\theta=\log\lambda\). The library exposes:

- bounded scalar minimization in log-lambda space;
- Newton stationary-point iterations given \((f,f',f'')\) with respect to lambda.

Grid search remains useful as a diagnostic and as a robust benchmark; it should not be silently replaced when reproducing older experiments.

## Manuscripts

- `paper/formal/`: research article; compact claims, proofs/derivations, experiments, limitations.
- `paper/tutorial/`: pedagogical companion; full derivations, terminology, implementation map, examples.

Both manuscripts must use the same symbols and estimator definitions. The tutorial may be longer but must not change the mathematics.

## Planned layers

1. estimator correctness and derivative checks;
2. lambda-selection benchmarks (grid, Brent, Newton, GCV, rolling forecast loss);
3. classical trend baselines and trend filtering;
4. financial signal definition;
5. downstream portfolio objective;
6. comparison of forecast-optimal and decision-optimal smoothing.

Portfolio code should not be added to the core estimator modules. It belongs in a future `portfolio/` package so statistical estimation and decision logic remain separable and testable.
