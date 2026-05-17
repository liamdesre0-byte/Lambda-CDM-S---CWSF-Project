"""Posterior chain summarization utilities."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def summarize_chain(path: Path | str) -> pd.DataFrame:
    """Return 16/50/84th percentile summaries for numeric chain columns."""
    chain = pd.read_csv(Path(path))
    cols = [c for c in chain.columns if pd.api.types.is_numeric_dtype(chain[c])]
    rows: list[dict[str, float | str]] = []
    for col in cols:
        vals = pd.to_numeric(chain[col], errors="coerce").dropna().to_numpy()
        if vals.size == 0:
            continue
        q16, q50, q84 = [float(x) for x in pd.Series(vals).quantile([0.16, 0.5, 0.84])]
        rows.append({"parameter": col, "q16": q16, "median": q50, "q84": q84})
    return pd.DataFrame(rows)
