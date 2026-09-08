from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.optimize import minimize_scalar


@dataclass
class LambdaOptimizationResult:
    lambda_: float
    theta_: float
    objective_: float
    converged_: bool
    n_iter_: int


def minimize_over_log_lambda(
    objective: Callable[[float], float],
    *,
    log_bounds: tuple[float, float] = (-12.0, 20.0),
    xatol: float = 1e-8,
) -> LambdaOptimizationResult:
    """Minimize a scalar objective over ``lambda > 0`` using ``theta=log(lambda)``.

    The public objective receives ``lambda``; positivity is guaranteed by the
    exponential reparameterization.
    """

    lo, hi = map(float, log_bounds)
    if not lo < hi:
        raise ValueError("log_bounds must satisfy lower < upper.")

    def objective_theta(theta: float) -> float:
        return float(objective(float(np.exp(theta))))

    result = minimize_scalar(
        objective_theta,
        bounds=(lo, hi),
        method="bounded",
        options={"xatol": float(xatol)},
    )
    theta = float(result.x)
    return LambdaOptimizationResult(
        lambda_=float(np.exp(theta)),
        theta_=theta,
        objective_=float(result.fun),
        converged_=bool(result.success),
        n_iter_=int(getattr(result, "nit", 0)),
    )


def newton_stationary_log_lambda(
    value_grad_hess: Callable[[float], tuple[float, float, float]],
    initial_lambda: float,
    *,
    log_bounds: tuple[float, float] = (-12.0, 20.0),
    tol: float = 1e-10,
    max_iter: int = 50,
) -> LambdaOptimizationResult:
    r"""Find a stationary point with Newton iterations in log-lambda space.

    ``value_grad_hess(lambda)`` must return ``(f, f', f'')`` with derivatives
    taken with respect to ``lambda``. If ``g(theta)=f(exp(theta))``, then

    ``g' = lambda f'`` and ``g'' = lambda f' + lambda^2 f''``.
    """

    initial_lambda = float(initial_lambda)
    if initial_lambda <= 0:
        raise ValueError("initial_lambda must be positive.")
    lo, hi = map(float, log_bounds)
    theta = float(np.clip(np.log(initial_lambda), lo, hi))
    converged = False
    value = np.nan

    for iteration in range(1, int(max_iter) + 1):
        lambda_ = float(np.exp(theta))
        value, grad, hess = map(float, value_grad_hess(lambda_))
        g1 = lambda_ * grad
        g2 = lambda_ * grad + lambda_ * lambda_ * hess
        if abs(g1) <= tol:
            converged = True
            break
        if not np.isfinite(g2) or abs(g2) <= np.finfo(float).eps:
            break
        step = g1 / g2
        candidate = float(np.clip(theta - step, lo, hi))
        if abs(candidate - theta) <= tol:
            theta = candidate
            converged = True
            break
        theta = candidate

    lambda_ = float(np.exp(theta))
    value = float(value_grad_hess(lambda_)[0])
    return LambdaOptimizationResult(
        lambda_=lambda_,
        theta_=theta,
        objective_=value,
        converged_=converged,
        n_iter_=iteration,
    )
