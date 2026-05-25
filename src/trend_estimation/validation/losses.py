from __future__ import annotations

import numpy as np
from .time_weights import make_time_weights


def mse_loss(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    if y_true.size != y_pred.size:
        raise ValueError("y_true and y_pred must have the same length.")
    return float(np.mean((y_true - y_pred) ** 2))


def weighted_mse(y_true, y_pred, weights=None, *, weight_scheme: str = "uniform", decay: float = 0.95) -> float:
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    if y_true.size != y_pred.size:
        raise ValueError("y_true and y_pred must have the same length.")
    if weights is None:
        weights = make_time_weights(y_true.size, weight_scheme, decay=decay)
    else:
        weights = np.asarray(weights, dtype=float).ravel()
        weights = weights / np.sum(weights)
    return float(np.sum(weights * (y_true - y_pred) ** 2))


class TimeWeightedValidationLoss:
    """Callable time-weighted validation loss."""

    def __init__(self, weight_scheme: str = "exponential", decay: float = 0.95, custom_weights=None):
        self.weight_scheme = weight_scheme
        self.decay = decay
        self.custom_weights = custom_weights

    def __call__(self, y_true, y_pred) -> float:
        if self.weight_scheme == "custom":
            return weighted_mse(y_true, y_pred, weights=self.custom_weights)
        return weighted_mse(y_true, y_pred, weight_scheme=self.weight_scheme, decay=self.decay)
