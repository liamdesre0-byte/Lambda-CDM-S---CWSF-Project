"""BAO/CMB compressed likelihood helper loaders."""

from __future__ import annotations

from pathlib import Path

import cwsf_pipeline as cw


def load_external_joint_pack(outdir: Path | str = "cwsf_output"):
    """Build the optional external BAO/CMB compressed likelihood pack."""
    out_path = Path(outdir)
    out_path.mkdir(parents=True, exist_ok=True)
    cw.copy_default_likelihood_templates(out_path)
    return cw.build_external_joint_pack(out_path)
