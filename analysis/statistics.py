"""Common statistical metrics for cosmology outputs."""

from __future__ import annotations

import numpy as np


def rmse(res: np.ndarray) -> float:
    """Root mean squared error."""
    arr = np.asarray(res, dtype=float)
    return float(np.sqrt(np.mean(np.square(arr))))


def chi2_diag(res: np.ndarray, sigma: np.ndarray) -> float:
    """Diagonal chi-square."""
    r = np.asarray(res, dtype=float)
    s = np.asarray(sigma, dtype=float)
    return float(np.sum(np.square(r / s)))
