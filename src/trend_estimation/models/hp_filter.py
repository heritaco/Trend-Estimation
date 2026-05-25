from __future__ import annotations

from .penalized_trend import PenalizedTrend


class HPTrend(PenalizedTrend):
    """Hodrick-Prescott-style trend as an order-2 penalized trend.

    Notes
    -----
    This class uses the package's quadratic penalized solver with
    ``order=2`` and no estimated drift. It is included as a baseline for
    comparisons, not as a full replacement for every econometric convention
    around the HP filter.
    """

    def __init__(self, lambda_: float = 1600.0):
        super().__init__(order=2, smoothness=None, lambda_=float(lambda_), estimate_drift=False)

    def get_params(self):
        return {"lambda_": self.lambda_}
