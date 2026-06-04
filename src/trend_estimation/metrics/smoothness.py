from __future__ import annotations

from trend_estimation.core.penalties import roughness


def roughness_d(trend, order: int = 2) -> float:
    """Return squared finite-difference roughness ``||D^order trend||_2^2``."""
    return roughness(trend, order=order)
