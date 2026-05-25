from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any
import numpy as np


@dataclass
class TrendFitResult:
    y_: np.ndarray
    trend_: np.ndarray
    residuals_: np.ndarray
    fitted_values_: np.ndarray
    order_: int | None = None
    lambda_: float | None = None
    smoothness_: float | None = None
    index_: np.ndarray | None = None
    metadata_: dict[str, Any] = field(default_factory=dict)


class BaseTrendEstimator(ABC):
    """Minimal estimator interface for trend models."""

    @abstractmethod
    def fit(self, y, X=None):
        raise NotImplementedError

    @abstractmethod
    def forecast(self, steps: int):
        raise NotImplementedError

    def predict(self, *args, **kwargs):
        if not hasattr(self, "trend_"):
            raise RuntimeError("Estimator must be fitted before predict().")
        return self.trend_

    def get_params(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if not k.endswith("_")}

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self
