from __future__ import annotations

from math import comb
import numpy as np


def forecast_trend(trend, order: int, m_hat: float, steps: int) -> np.ndarray:
    """Forecast by recursively imposing a constant ``order``-th difference."""
    trend = np.asarray(trend, dtype=float).ravel()
    steps = int(steps)
    if steps <= 0:
        return np.array([], dtype=float)
    order = int(order)
    if order < 0:
        raise ValueError("order must be nonnegative.")
    if order == 0:
        return np.full(steps, float(m_hat))
    d_eff = min(order, trend.size)
    last = trend[-d_eff:].copy()
    coeffs = np.array([(-1) ** (d_eff - k) * comb(d_eff, k) for k in range(d_eff)], dtype=float)
    out = np.empty(steps, dtype=float)
    for i in range(steps):
        out[i] = float(m_hat) - float(coeffs @ last)
        last[:-1] = last[1:]
        last[-1] = out[i]
    return out


def build_polynomial_from_tail(trend, order: int, m_hat: float, n_total: int, n_fit: int) -> np.ndarray:
    """Build the global polynomial implied by constant finite difference and a right tail."""
    trend = np.asarray(trend, dtype=float).ravel()
    if order <= 0:
        return np.full(n_total, float(m_hat), dtype=float)
    d_eff = min(int(order), int(n_fit), trend.size)
    poly = np.empty(int(n_total), dtype=float)
    j0 = int(n_fit) - 1
    start = j0 - d_eff + 1
    poly[start:j0 + 1] = trend[-d_eff:]

    back_coeffs = np.array([(-1) ** (d_eff - k) * comb(d_eff, k) for k in range(1, d_eff + 1)], dtype=float)
    coef0 = (-1) ** d_eff
    for j in range(start - 1, -1, -1):
        poly[j] = (float(m_hat) - float(back_coeffs @ poly[j + 1:j + 1 + d_eff])) * coef0

    fwd_coeffs = np.array([(-1) ** (d_eff - k) * comb(d_eff, k) for k in range(d_eff)], dtype=float)
    for i in range(j0 + 1, int(n_total)):
        poly[i] = float(m_hat) - float(fwd_coeffs @ poly[i - d_eff:i])
    return poly
