import numpy as np

import trend_estimation as td


def test_rolling_origin_splits_are_chronological():
    splits = td.rolling_origin_splits(20, initial_train=10, horizon=3, step=2)
    assert len(splits) == 4
    for split in splits:
        assert split.train.stop <= split.validation.start
        assert split.validation.stop <= 20


def test_rolling_window_keeps_requested_width():
    splits = td.rolling_origin_splits(
        20,
        initial_train=8,
        horizon=2,
        step=3,
        expanding=False,
        train_window=5,
    )
    assert all((s.train.stop - s.train.start) == 5 for s in splits)


def test_log_lambda_minimization_recovers_positive_optimum():
    result = td.minimize_over_log_lambda(
        lambda lmb: (np.log(lmb) - np.log(3.0)) ** 2,
        log_bounds=(-5.0, 5.0),
    )
    assert result.converged_
    assert np.isclose(result.lambda_, 3.0, rtol=1e-5)
