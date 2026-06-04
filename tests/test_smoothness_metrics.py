import numpy as np

import trend_estimation as td


def test_roughness_d_matches_squared_second_difference():
    trend = np.array([0.0, 1.0, 4.0, 9.0])

    assert td.roughness_d(trend, order=2) == 8.0


def test_roughness_d_matches_core_roughness():
    trend = np.linspace(0.0, 1.0, 10) ** 3

    assert td.roughness_d(trend, order=3) == td.roughness(trend, order=3)
