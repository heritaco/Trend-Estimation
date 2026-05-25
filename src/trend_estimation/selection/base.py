from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class SelectionResult:
    best_model_: Any
    best_order_: int
    best_lambda_: float
    best_smoothness_: float
    best_score_: float
    all_minima_: list[dict[str, Any]] = field(default_factory=list)
    validation_curve_: Any = None
    models_: dict[Any, Any] = field(default_factory=dict)


class BaseTrendSelector(ABC):
    @abstractmethod
    def fit(self, y, **kwargs) -> SelectionResult:
        raise NotImplementedError


class BaseSelectionCriterion:
    def score(self, y, trend, residuals, model, **kwargs) -> float:
        raise NotImplementedError
