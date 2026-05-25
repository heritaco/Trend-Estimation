from __future__ import annotations

import numpy as np


def make_time_weights(n_obs: int, scheme: str = "uniform", *, decay: float = 0.95, custom=None) -> np.ndarray:
    """Create normalized time weights. Later observations get larger weights for linear/exponential."""
    n_obs = int(n_obs)
    if n_obs <= 0:
        raise ValueError("n_obs must be positive.")
    if scheme == "uniform":
        weights = np.ones(n_obs, dtype=float)
    elif scheme == "linear":
        weights = np.arange(1, n_obs + 1, dtype=float)
    elif scheme == "exponential":
        if not (0 < decay <= 1):
            raise ValueError("decay must be in (0, 1].")
        # Oldest observation gets decay^(n-1), newest gets 1.
        weights = decay ** np.arange(n_obs - 1, -1, -1, dtype=float)
    elif scheme == "custom":
        if custom is None:
            raise ValueError("custom weights must be provided for scheme='custom'.")
        weights = np.asarray(custom, dtype=float).ravel()
        if weights.size != n_obs:
            raise ValueError("custom weights have incompatible length.")
    else:
        raise ValueError(f"Unknown weight scheme {scheme!r}.")
    if np.any(weights < 0) or not np.all(np.isfinite(weights)):
        raise ValueError("weights must be finite and nonnegative.")
    total = float(np.sum(weights))
    if total <= 0:
        raise ValueError("weights must have positive sum.")
    return weights / total
