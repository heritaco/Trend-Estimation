from __future__ import annotations

import numpy as np
from trend_estimation.utils.arrays import as_1d_float_array


def train_val_test_split_indices(n_obs: int, frac_train: float = 0.6, frac_val: float = 0.2):
    n_obs = int(n_obs)
    n_train = int(frac_train * n_obs)
    n_val = int(frac_val * n_obs)
    n_train = max(1, min(n_train, n_obs - 2))
    n_val = max(1, min(n_val, n_obs - n_train - 1))
    return slice(0, n_train), slice(n_train, n_train + n_val), slice(n_train + n_val, n_obs)


def train_val_test_split(y, frac_train: float = 0.6, frac_val: float = 0.2):
    y = as_1d_float_array(y)
    train_idx, val_idx, test_idx = train_val_test_split_indices(len(y), frac_train, frac_val)
    return y[train_idx], y[val_idx], y[test_idx], train_idx, val_idx, test_idx
