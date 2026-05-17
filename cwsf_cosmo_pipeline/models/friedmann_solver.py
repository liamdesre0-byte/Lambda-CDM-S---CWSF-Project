"""Background interpolation/solver accessors."""

from __future__ import annotations

import cwsf_pipeline as cw


def build_engine(zp_hi_dm: float = 4.0, z_age: float = 2.5) -> cw.CosmoInterpEngine:
    """Create a cached cosmology interpolation engine."""
    return cw.CosmoInterpEngine(float(zp_hi_dm), float(z_age))
