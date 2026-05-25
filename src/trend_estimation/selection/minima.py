from __future__ import annotations

from typing import Callable
import numpy as np


def golden_local(J: Callable[[float], float], a: float, b: float, n_iter: int = 30) -> tuple[float, float]:
    """Golden-section minimization on a local bracket."""
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    invphi = 1.0 / phi
    c = b - invphi * (b - a)
    d = a + invphi * (b - a)
    fc = J(c)
    fd = J(d)
    for _ in range(n_iter):
        if fc < fd:
            b = d
            d = c
            fd = fc
            c = b - invphi * (b - a)
            fc = J(c)
        else:
            a = c
            c = d
            fc = fd
            d = a + invphi * (b - a)
            fd = J(d)
    x = 0.5 * (a + b)
    return float(x), float(J(x))


def find_all_local_minima(
    J: Callable[[float], float],
    grid,
    *,
    refine: bool = True,
    refine_iter: int = 30,
) -> tuple[np.ndarray, np.ndarray]:
    """Detect and optionally refine all local minima on a one-dimensional grid."""
    grid = np.asarray(grid, dtype=float)
    values = np.array([J(float(x)) for x in grid], dtype=float)
    cmp = values.copy()
    cmp[~np.isfinite(cmp)] = np.inf
    candidates: list[int] = []
    n = len(grid)
    for i in range(1, n - 1):
        if np.isfinite(cmp[i]) and cmp[i] <= cmp[i - 1] and cmp[i] <= cmp[i + 1]:
            candidates.append(i)
    if n >= 2 and np.isfinite(cmp[0]) and cmp[0] <= cmp[1]:
        candidates.append(0)
    if n >= 2 and np.isfinite(cmp[-1]) and cmp[-1] <= cmp[-2]:
        candidates.append(n - 1)

    xs, fs = [], []
    for i in candidates:
        if refine and n > 1:
            if 0 < i < n - 1:
                a, b = grid[i - 1], grid[i + 1]
            elif i == 0:
                a, b = grid[0], grid[1]
            else:
                a, b = grid[-2], grid[-1]
            x, f = golden_local(J, float(a), float(b), n_iter=refine_iter)
        else:
            x, f = float(grid[i]), float(values[i])
        xs.append(x)
        fs.append(f)
    order = np.argsort(xs)
    return np.asarray(xs)[order], np.asarray(fs)[order]
