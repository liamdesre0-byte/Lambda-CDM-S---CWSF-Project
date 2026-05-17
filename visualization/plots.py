"""Basic plot wrappers from the legacy pipeline."""

from __future__ import annotations

from pathlib import Path

import cwsf_pipeline as cw


def plot_lcdm_q_curve(outdir: Path | str, h0: float = 72.8, omega_m: float = 0.30) -> Path:
    """Write LCDM deceleration plot into the figure directory."""
    out_path = Path(outdir)
    out_path.mkdir(parents=True, exist_ok=True)
    fig_path = out_path / "q_deceleration_lcdm.png"
    cw.plot_q_deceleration_lcdm(fig_path, float(h0), float(omega_m))
    return fig_path
