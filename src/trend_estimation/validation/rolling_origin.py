from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RollingOriginSplit:
    train: slice
    validation: slice


def rolling_origin_splits(
    n_obs: int,
    initial_train: int,
    horizon: int,
    *,
    step: int = 1,
    expanding: bool = True,
    train_window: int | None = None,
) -> list[RollingOriginSplit]:
    """Create chronological train/validation splits with no look-ahead.

    Parameters
    ----------
    n_obs:
        Total observations.
    initial_train:
        Number of observations available at the first forecast origin.
    horizon:
        Number of future observations scored at each origin.
    step:
        Number of observations by which the origin advances.
    expanding:
        If True, training always starts at zero. If False, a rolling window is
        used and ``train_window`` defaults to ``initial_train``.
    train_window:
        Width of the rolling training window when ``expanding=False``.
    """

    n_obs = int(n_obs)
    initial_train = int(initial_train)
    horizon = int(horizon)
    step = int(step)
    if n_obs <= 0 or initial_train <= 0 or horizon <= 0 or step <= 0:
        raise ValueError("n_obs, initial_train, horizon, and step must be positive.")
    if initial_train + horizon > n_obs:
        raise ValueError("The first validation horizon must fit inside the sample.")

    if not expanding:
        if train_window is None:
            train_window = initial_train
        train_window = int(train_window)
        if train_window <= 0:
            raise ValueError("train_window must be positive.")

    splits: list[RollingOriginSplit] = []
    origin = initial_train
    while origin + horizon <= n_obs:
        if expanding:
            train_start = 0
        else:
            train_start = max(0, origin - train_window)
        splits.append(
            RollingOriginSplit(
                train=slice(train_start, origin),
                validation=slice(origin, origin + horizon),
            )
        )
        origin += step
    return splits
