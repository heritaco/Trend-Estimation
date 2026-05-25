import numpy as np
from trend_estimation.selection.minima import find_all_local_minima


def test_find_minimum_artificial():
    grid = np.linspace(0, 1, 101)
    xs, fs = find_all_local_minima(lambda x: (x - 0.4) ** 2, grid)
    assert len(xs) >= 1
    assert abs(xs[0] - 0.4) < 0.02
