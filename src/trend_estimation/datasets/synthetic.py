from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


@dataclass
class SyntheticTrendData:
    y: np.ndarray
    true_trend: np.ndarray
    index: np.ndarray
    metadata: dict = field(default_factory=dict)


def _rng(random_state):
    return np.random.default_rng(random_state)


def make_polynomial_trend_series(
    n_obs: int = 200,
    degree: int = 2,
    coefficients=None,
    noise_std: float = 1.0,
    random_state: int | None = 123,
    return_true_trend: bool = True,
) -> SyntheticTrendData:
    """Generate a noisy polynomial trend series with known true trend."""
    rng = _rng(random_state)
    n_obs = int(n_obs)
    x = np.linspace(-1.0, 1.0, n_obs)
    if coefficients is None:
        coefficients = np.ones(int(degree) + 1)
        coefficients[:-1] *= 0.5
    p = np.poly1d(coefficients)
    true_trend = np.asarray(p(x), dtype=float)
    y = true_trend + rng.normal(0.0, noise_std, size=n_obs)
    return SyntheticTrendData(
        y=y,
        true_trend=true_trend,
        index=np.arange(n_obs),
        metadata={"type": "polynomial", "degree": degree, "noise_std": noise_std},
    )


def make_noisy_trend_series(n_obs: int = 200, noise_std: float = 1.0, random_state: int | None = 123) -> SyntheticTrendData:
    return make_polynomial_trend_series(n_obs=n_obs, degree=2, noise_std=noise_std, random_state=random_state)


def make_piecewise_trend_series(n_obs: int = 200, noise_std: float = 1.0, random_state: int | None = 123) -> SyntheticTrendData:
    """Generate a piecewise-linear trend with one slope change."""
    rng = _rng(random_state)
    n_obs = int(n_obs)
    x = np.arange(n_obs, dtype=float)
    mid = n_obs // 2
    true_trend = np.empty(n_obs, dtype=float)
    true_trend[:mid] = 0.02 * x[:mid]
    true_trend[mid:] = 0.02 * x[mid] + 0.08 * (x[mid:] - x[mid])
    y = true_trend + rng.normal(0.0, noise_std, size=n_obs)
    return SyntheticTrendData(y=y, true_trend=true_trend, index=np.arange(n_obs), metadata={"type": "piecewise", "noise_std": noise_std})


def make_sinusoidal_trend_series(
    n_obs: int = 240,
    amplitude: float = 1.0,
    period: float = 80.0,
    slope: float = 0.01,
    noise_std: float = 0.5,
    random_state: int | None = 123,
) -> SyntheticTrendData:
    """Generate a smooth sinusoidal trend plus optional linear drift."""
    rng = _rng(random_state)
    n_obs = int(n_obs)
    t = np.arange(n_obs, dtype=float)
    true_trend = slope * t + amplitude * np.sin(2.0 * np.pi * t / float(period))
    y = true_trend + rng.normal(0.0, noise_std, size=n_obs)
    return SyntheticTrendData(y=y, true_trend=true_trend, index=np.arange(n_obs), metadata={"type": "sinusoidal", "period": period, "noise_std": noise_std})


def make_local_linear_trend_series(
    n_obs: int = 240,
    slope_noise_std: float = 0.01,
    observation_noise_std: float = 0.5,
    random_state: int | None = 123,
) -> SyntheticTrendData:
    """Generate a local-linear trend with a random-walk slope."""
    rng = _rng(random_state)
    n_obs = int(n_obs)
    slope = np.zeros(n_obs, dtype=float)
    true_trend = np.zeros(n_obs, dtype=float)
    slope[0] = 0.02
    for t in range(1, n_obs):
        slope[t] = slope[t - 1] + rng.normal(0.0, slope_noise_std)
        true_trend[t] = true_trend[t - 1] + slope[t]
    y = true_trend + rng.normal(0.0, observation_noise_std, size=n_obs)
    return SyntheticTrendData(
        y=y,
        true_trend=true_trend,
        index=np.arange(n_obs),
        metadata={"type": "local_linear", "slope_noise_std": slope_noise_std, "observation_noise_std": observation_noise_std},
    )


def make_structural_break_series(
    n_obs: int = 240,
    break_point: int | None = None,
    pre_slope: float = 0.01,
    post_slope: float = -0.03,
    level_shift: float = 2.0,
    noise_std: float = 0.5,
    random_state: int | None = 123,
) -> SyntheticTrendData:
    """Generate a trend with a level and slope break."""
    rng = _rng(random_state)
    n_obs = int(n_obs)
    bp = n_obs // 2 if break_point is None else int(break_point)
    bp = max(1, min(bp, n_obs - 1))
    t = np.arange(n_obs, dtype=float)
    true_trend = np.empty(n_obs, dtype=float)
    true_trend[:bp] = pre_slope * t[:bp]
    true_trend[bp:] = pre_slope * t[bp] + level_shift + post_slope * (t[bp:] - t[bp])
    y = true_trend + rng.normal(0.0, noise_std, size=n_obs)
    return SyntheticTrendData(
        y=y,
        true_trend=true_trend,
        index=np.arange(n_obs),
        metadata={"type": "structural_break", "break_point": bp, "noise_std": noise_std},
    )
