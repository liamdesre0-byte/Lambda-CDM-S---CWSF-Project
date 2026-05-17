"""DES holdout loading helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

import cwsf_pipeline as cw


def load_des_holdout(outdir: Path | str = "cwsf_output") -> pd.DataFrame:
    """Download (if needed) and load DES Dovekie holdout data."""
    out_path = Path(outdir)
    out_path.mkdir(parents=True, exist_ok=True)
    des_path = out_path / "DES-Dovekie_HD.csv"
    cw.ensure_download(cw.DES_URL, des_path)
    return cw.load_des(des_path)
