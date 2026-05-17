"""Information criteria wrappers."""

from __future__ import annotations

import cwsf_pipeline as cw


def aic_bic_from_loglike(n_obs: int, n_params: int, loglike_max: float) -> tuple[float, float]:
    """Compute (AIC, BIC) for maximum Gaussian log-likelihood."""
    return cw.max_aic_bic(int(n_obs), int(n_params), float(loglike_max))
