import numpy as np
import trend_estimation as td


def test_solver_length():
    y = np.linspace(0, 1, 30)
    trend = td.penalized_solution(y, order=2, lambda_=1.0)
    assert trend.shape == y.shape
