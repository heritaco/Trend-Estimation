from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


@dataclass
class SyntheticTrendData:
    y: np.ndarray
    true_trend: np.ndarray
    index: np.ndarray
    metadata: dict = field(default_factory=dict)


def make_polynomial_trend_series(
    n_obs: int = 200,
    degree: int = 2,
    coefficients=None,
    noise_std: float = 1.0,
    random_state: int | None = 123,
    return_true_trend: bool = True,
) -> SyntheticTrendData:
    """Generate a noisy polynomial trend series with known true trend."""
    rng = np.random.default_rng(random_state)
    x = np.linspace(-1.0, 1.0, int(n_obs))
    if coefficients is None:
        coefficients = np.ones(int(degree) + 1)
        coefficients[:-1] *= 0.5
    p = np.poly1d(coefficients)
    true_trend = np.asarray(p(x), dtype=float)
    y = true_trend + rng.normal(0.0, noise_std, size=int(n_obs))
    return SyntheticTrendData(y=y, true_trend=true_trend, index=np.arange(int(n_obs)), metadata={"degree": degree, "noise_std": noise_std})


def make_noisy_trend_series(n_obs: int = 200, noise_std: float = 1.0, random_state: int | None = 123) -> SyntheticTrendData:
    return make_polynomial_trend_series(n_obs=n_obs, degree=2, noise_std=noise_std, random_state=random_state)


def make_piecewise_trend_series(n_obs: int = 200, noise_std: float = 1.0, random_state: int | None = 123) -> SyntheticTrendData:
    rng = np.random.default_rng(random_state)
    x = np.arange(int(n_obs), dtype=float)
    mid = int(n_obs) // 2
    true_trend = np.empty(int(n_obs), dtype=float)
    true_trend[:mid] = 0.02 * x[:mid]
    true_trend[mid:] = 0.02 * x[mid] + 0.08 * (x[mid:] - x[mid])
    y = true_trend + rng.normal(0.0, noise_std, size=int(n_obs))
    return SyntheticTrendData(y=y, true_trend=true_trend, index=np.arange(int(n_obs)), metadata={"type": "piecewise"})
