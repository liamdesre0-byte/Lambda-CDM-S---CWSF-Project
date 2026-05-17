"""Flat LCDM observables."""

from __future__ import annotations

import numpy as np

import cwsf_pipeline as cw


def hubble_lcdm(z: np.ndarray, h0: float, omega_m: float) -> np.ndarray:
    """Compute H(z) for flat LCDM."""
    return cw.Hz(np.asarray(z, dtype=float), float(h0), float(omega_m))


def distance_modulus_lcdm(z: np.ndarray, h0: float, omega_m: float, eng: cw.CosmoInterpEngine) -> np.ndarray:
    """Compute LCDM distance modulus."""
    return cw.mu_lcdm(np.asarray(z, dtype=float), float(h0), float(omega_m), eng)
