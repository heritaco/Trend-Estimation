from __future__ import annotations

from .penalized_trend import PenalizedTrend


class HPTrend(PenalizedTrend):
    """Hodrick-Prescott-style trend as the order-2 penalized trend special case."""

    def __init__(self, lambda_: float = 1600.0):
        super().__init__(order=2, smoothness=None, lambda_=lambda_, estimate_drift=False)
