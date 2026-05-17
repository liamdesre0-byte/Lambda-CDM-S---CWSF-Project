"""MCMC driver wrappers around the legacy full pipeline."""

from __future__ import annotations

import cwsf_pipeline as cw


def run_mcmc_pipeline() -> None:
    """Run the full CWSF SN/BAO/CMB-style inference workflow."""
    cw.main()
