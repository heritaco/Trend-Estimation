from __future__ import annotations

import pandas as pd

from trend_estimation.benchmarks.results import BenchmarkResult


def benchmark_metrics_table(result: BenchmarkResult, *, phase: str | None = None, target: str | None = None) -> pd.DataFrame:
    """Return a filtered benchmark metrics table."""
    table = result.metrics_.copy()
    if phase is not None and not table.empty:
        table = table[table["phase"] == phase]
    if target is not None and not table.empty:
        table = table[table["target"] == target]
    return table.reset_index(drop=True)
