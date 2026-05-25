from __future__ import annotations

from typing import Callable
import numpy as np
import pandas as pd

from .base import BaseTrendSelector, SelectionResult
from .minima import find_all_local_minima
from trend_estimation.models.penalized_trend import PenalizedTrend
from trend_estimation.utils.arrays import as_1d_float_array, indices_from_slice_or_array
from trend_estimation.validation.splits import train_val_test_split_indices
from trend_estimation.validation.losses import mse_loss


class TrainValidationSelector(BaseTrendSelector):
    """Select order and smoothness by validation loss over a temporal split."""

    def __init__(
        self,
        orders=(1, 2, 3),
        smoothness_grid=None,
        *,
        detect_multiple_minima: bool = True,
        refine: bool = True,
        refine_iter: int = 30,
        loss: Callable | None = None,
        frac_train: float = 0.6,
        frac_val: float = 0.2,
    ):
        self.orders = list(orders)
        self.smoothness_grid = smoothness_grid
        self.detect_multiple_minima = detect_multiple_minima
        self.refine = refine
        self.refine_iter = int(refine_iter)
        self.loss = loss or mse_loss
        self.frac_train = frac_train
        self.frac_val = frac_val

    def _default_grid(self):
        if self.smoothness_grid is None:
            return np.linspace(1e-3, 0.999, 120)
        return np.asarray(self.smoothness_grid, dtype=float)

    def fit(self, y, *, train_idx=None, val_idx=None, test_idx=None) -> SelectionResult:
        y = as_1d_float_array(y)
        n_obs = y.size
        if train_idx is None or val_idx is None:
            train_idx, val_idx, test_idx_auto = train_val_test_split_indices(n_obs, self.frac_train, self.frac_val)
            if test_idx is None:
                test_idx = test_idx_auto
        train_pos = indices_from_slice_or_array(train_idx, n_obs)
        val_pos = indices_from_slice_or_array(val_idx, n_obs)
        y_train = y[train_pos]
        y_val = y[val_pos]
        if len(y_val) == 0:
            raise ValueError("Validation set must not be empty.")

        grid = self._default_grid()
        rows: list[dict] = []
        minima: list[dict] = []
        models: dict[tuple[int, float], PenalizedTrend] = {}

        def score_for(order: int, smoothness: float) -> float:
            model = PenalizedTrend(order=order, smoothness=float(smoothness)).fit(y_train)
            forecast = model.forecast(len(y_val))
            return float(self.loss(y_val, forecast))

        for order in self.orders:
            for s in grid:
                try:
                    score = score_for(order, float(s))
                except Exception:
                    score = np.inf
                rows.append({"order": order, "smoothness": float(s), "score": score})

            if self.detect_multiple_minima:
                J = lambda s, order=order: score_for(order, float(s))
                s_min, j_min = find_all_local_minima(J, grid, refine=self.refine, refine_iter=self.refine_iter)
            else:
                sub = [r for r in rows if r["order"] == order]
                best = min(sub, key=lambda r: r["score"])
                s_min = np.array([best["smoothness"]])
                j_min = np.array([best["score"]])
            for s, score in zip(s_min, j_min):
                model = PenalizedTrend(order=order, smoothness=float(s)).fit(y_train)
                key = (order, float(s))
                models[key] = model
                minima.append({
                    "order": order,
                    "smoothness": float(s),
                    "lambda": float(model.lambda_),
                    "score": float(score),
                })

        curve = pd.DataFrame(rows)
        best = min(minima, key=lambda r: r["score"])
        best_key = (best["order"], best["smoothness"])
        self.best_model_ = models[best_key]
        self.best_order_ = best["order"]
        self.best_smoothness_ = best["smoothness"]
        self.best_lambda_ = best["lambda"]
        self.best_score_ = best["score"]
        self.all_minima_ = minima
        self.validation_curve_ = curve
        self.models_ = models
        return SelectionResult(
            best_model_=self.best_model_,
            best_order_=self.best_order_,
            best_lambda_=self.best_lambda_,
            best_smoothness_=self.best_smoothness_,
            best_score_=self.best_score_,
            all_minima_=self.all_minima_,
            validation_curve_=self.validation_curve_,
            models_=self.models_,
        )
