from __future__ import annotations


def check_order(order: int) -> int:
    order = int(order)
    if order < 0:
        raise ValueError("order must be nonnegative.")
    return order


def check_smoothness(smoothness: float) -> float:
    smoothness = float(smoothness)
    if not (0.0 <= smoothness < 1.0):
        raise ValueError("smoothness must be in [0, 1).")
    return smoothness
