"""Corner plot helper for posterior CSV files."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

import cwsf_pipeline as cw


def write_corner_plot(chain_csv: Path | str, out_png: Path | str) -> Path | None:
    """Create a corner plot if the optional dependency is available."""
    if not cw.HAVE_CORNER:
        return None
    chain = pd.read_csv(Path(chain_csv))
    columns = [c for c in ["H0", "Omega_m", "t_crit", "k"] if c in chain.columns]
    if not columns:
        return None
    fig = cw.corner.corner(chain[columns].to_numpy(), labels=columns, show_titles=True)  # type: ignore[union-attr]
    out_path = Path(out_png)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path
