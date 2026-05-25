from __future__ import annotations

import numpy as np
from .difference import difference_matrix


def roughness(trend, order: int) -> float:
    """Return squared finite-difference roughness ``||D^order trend||^2``."""
    trend = np.asarray(trend, dtype=float).ravel()
    D = difference_matrix(trend.size, order)
    v = D @ trend
    return float(v @ v)
