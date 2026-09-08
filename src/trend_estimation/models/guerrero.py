from __future__ import annotations

from trend_estimation.models.penalized_trend import PenalizedTrend


class GuerreroTrend(PenalizedTrend):
    """Explicit name for the original Guerrero-style smoother with drift.

    `PenalizedTrend` is retained for backward compatibility. New research code
    should prefer this name whenever the drift formulation is intended, so that
    experiments cannot confuse it with `PurePenalizedTrend`.
    """

    pass
