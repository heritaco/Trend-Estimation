from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import numpy as np


@dataclass
class ForecastResult:
    fitted_values_: np.ndarray
    forecast_values_: np.ndarray
    forecast_index_: np.ndarray | None = None
    origin_: str | None = None
    steps_: int | None = None
    model_name_: str | None = None
    metadata_: dict[str, Any] = field(default_factory=dict)
