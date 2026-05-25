from __future__ import annotations

from .penalized_trend import PenalizedTrend


class WhittakerTrend(PenalizedTrend):
    """Whittaker-Henderson-style quadratic penalized trend baseline.

    Parameters
    ----------
    order:
        Difference order used in the roughness penalty.
    lambda_:
        Penalty parameter. Larger values imply smoother trends.
    smoothness:
        Optional smoothness index. If ``lambda_`` is omitted, this is converted
        to a penalty parameter by the package smoothness map.

    Notes
    -----
    The implementation sets ``estimate_drift=False``. This corresponds to the
    classical quadratic penalty around zero differences, in contrast with the
    Guerrero-style estimator where a drift term can be estimated.
    """

    def __init__(self, order: int = 2, lambda_: float | None = None, smoothness: float | None = 0.75):
        super().__init__(order=order, smoothness=smoothness, lambda_=lambda_, estimate_drift=False)

    def get_params(self):
        return {"order": self.order, "lambda_": self.lambda_, "smoothness": self.smoothness}
