import trend_estimation as td


def test_smoothness_round_trip():
    lam = td.smoothness_to_lambda(0.5, 40, 2)
    s = td.lambda_to_smoothness(lam, 40, 2)
    assert abs(s - 0.5) < 1e-6
