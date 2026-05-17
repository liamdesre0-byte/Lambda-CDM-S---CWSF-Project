"""Monte Carlo sensitivity helpers from ccomplet2ee."""

from __future__ import annotations

from pathlib import Path

import ccomplet2ee as c2


def run_monte_carlo_report(outdir: Path | str = "ee_output", mc_runs: int = 300, seed: int = 42) -> int:
    """Execute the ccomplet2ee sensitivity/MC report generation."""
    out_path = Path(outdir)
    out_path.mkdir(parents=True, exist_ok=True)
    return int(c2.main(["--out", str(out_path), "--mc", str(int(mc_runs)), "--seed", str(int(seed))]))
