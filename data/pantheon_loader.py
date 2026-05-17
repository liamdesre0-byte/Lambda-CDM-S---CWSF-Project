"""Pantheon+ SH0ES data loading helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

import cwsf_pipeline as cw


def load_pantheon_training(outdir: Path | str = "cwsf_output") -> pd.DataFrame:
    """Download (if needed) and load the Pantheon+ SH0ES training sample."""
    out_path = Path(outdir)
    out_path.mkdir(parents=True, exist_ok=True)
    pant_path = out_path / "Pantheon+SH0ES.dat"
    cw.ensure_download(cw.PANT_URL, pant_path)
    return cw.load_pantheon(pant_path)
