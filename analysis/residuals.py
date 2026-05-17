"""Residual calculations."""

from __future__ import annotations

import numpy as np


def residuals(observed: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    """Compute observed minus predicted residuals."""
    return np.asarray(observed, dtype=float) - np.asarray(predicted, dtype=float)
