"""Publication-grade figure export wrapper."""

from __future__ import annotations

from pathlib import Path

import cwsf_pipeline as cw


def generate_all_figures(outdir: Path | str = "cwsf_output") -> dict:
    """Generate publication figure bundle from pipeline artifacts."""
    return cw.generate_publication_paper_figures(Path(outdir))
