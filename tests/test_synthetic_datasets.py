import trend_estimation as td


def test_synthetic_same_length():
    data = td.make_polynomial_trend_series(n_obs=55)
    assert len(data.y) == len(data.true_trend) == 55
