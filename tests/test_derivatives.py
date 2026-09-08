import numpy as np

import trend_estimation as td


def test_pure_analytic_derivatives_match_centered_finite_differences():
    rng = np.random.default_rng(123)
    y = rng.normal(size=24)
    order = 2
    lambda_ = 3.0
    h = 1e-4

    analytic = td.pure_trend_derivatives(y, order=order, lambda_=lambda_)
    plus = td.pure_penalized_solution(y, order, lambda_ + h)
    minus = td.pure_penalized_solution(y, order, lambda_ - h)

    first_fd = (plus - minus) / (2.0 * h)
    second_fd = (plus - 2.0 * analytic.trend + minus) / (h * h)

    assert np.allclose(analytic.first, first_fd, rtol=2e-5, atol=2e-7)
    assert np.allclose(analytic.second, second_fd, rtol=2e-3, atol=2e-5)


def test_mse_derivative_helper_matches_scalar_finite_difference():
    target = np.array([0.0, 1.0, 2.0])

    def prediction(lambda_):
        return np.array([lambda_, lambda_**2, 2.0 * lambda_])

    lambda_ = 0.7
    pred = prediction(lambda_)
    pred_first = np.array([1.0, 2.0 * lambda_, 2.0])
    pred_second = np.array([0.0, 2.0, 0.0])
    value, first, second = td.mse_from_prediction_derivatives(
        target, pred, pred_first, pred_second
    )

    h = 1e-5
    f = lambda lmb: float(np.mean((target - prediction(lmb)) ** 2))
    first_fd = (f(lambda_ + h) - f(lambda_ - h)) / (2.0 * h)
    second_fd = (f(lambda_ + h) - 2.0 * f(lambda_) + f(lambda_ - h)) / (h * h)

    assert np.isclose(value, f(lambda_))
    assert np.isclose(first, first_fd, rtol=1e-6, atol=1e-8)
    assert np.isclose(second, second_fd, rtol=1e-4, atol=1e-5)
