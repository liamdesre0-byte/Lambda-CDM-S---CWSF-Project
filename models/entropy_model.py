"""Entropy-blend model observables."""

from __future__ import annotations

import numpy as np

import cwsf_pipeline as cw


def hubble_blend(z: np.ndarray, h0: float, omega_m: float, t_crit: float, k: float, eng: cw.CosmoInterpEngine) -> np.ndarray:
    """Compute blend-model H(z) from ccomplet2-aligned physics."""
    return cw.c2_H_blend(np.asarray(z, dtype=float), float(h0), float(omega_m), float(t_crit), float(k), eng)


def distance_modulus_blend(
    z: np.ndarray,
    h0: float,
    omega_m: float,
    t_crit: float,
    k: float,
    eng: cw.CosmoInterpEngine,
) -> np.ndarray:
    """Compute blend-model distance modulus."""
    return cw.mu_blend(np.asarray(z, dtype=float), float(h0), float(omega_m), float(t_crit), float(k), eng)
