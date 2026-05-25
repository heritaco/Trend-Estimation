from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass
class BenchmarkResult:
    """Container returned by :class:`BenchmarkRunner`."""

    metrics_: pd.DataFrame
    forecasts_: dict[str, dict[str, Any]] = field(default_factory=dict)
    fitted_trends_: dict[str, dict[str, Any]] = field(default_factory=dict)
    model_params_: dict[str, dict[str, Any]] = field(default_factory=dict)
    errors_: dict[str, str] = field(default_factory=dict)
    metadata_: dict[str, Any] = field(default_factory=dict)
