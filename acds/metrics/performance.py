"""
Performance metrics for model evaluation.
"""

import numpy as np


def nrmse(preds: np.ndarray, target: np.ndarray) -> float:
    """Compute Normalized Root Mean Squared Error.

    Args:
        preds: Predictions array.
        target: Ground truth array (must match preds shape).

    Returns:
        NRMSE value (lower is better).
    """
    assert preds.shape == target.shape, "Predictions and target must have the same shape"
    mse = np.mean(np.square(preds - target))
    norm = np.sqrt(np.mean(np.square(target)))
    return np.sqrt(mse) / (norm + 1e-9)


__all__ = ["nrmse"]
