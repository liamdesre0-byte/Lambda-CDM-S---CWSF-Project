"""Top-level orchestrator for the modular CWSF repository."""

from __future__ import annotations

import argparse
from pathlib import Path

import nbody_simulations as nbody
from simulations.mcmc_sampler import run_mcmc_pipeline
from simulations.monte_carlo import run_monte_carlo_report
from visualization.publication_figures import generate_all_figures


def parse_args() -> argparse.Namespace:
    """Parse command-line options for the top-level workflow runner."""
    parser = argparse.ArgumentParser(description="CWSF modular workflow driver.")
    parser.add_argument(
        "--task",
        default="all",
        choices=["all", "pipeline", "nbody", "ee", "figures"],
        help="Select which stage to run.",
    )
    parser.add_argument("--outdir", default="cwsf_output", help="Pipeline output directory.")
    parser.add_argument("--ee-outdir", default="ee_output", help="ccomplet2ee output directory.")
    parser.add_argument("--mc-runs", type=int, default=300, help="Monte Carlo runs for ccomplet2ee report.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for ccomplet2ee report.")
    return parser.parse_args()


def main() -> int:
    """Run selected tasks in a clean, reproducible order."""
    args = parse_args()
    outdir = Path(args.outdir)
    ee_outdir = Path(args.ee_outdir)

    if args.task in ("all", "pipeline"):
        run_mcmc_pipeline()

    if args.task in ("all", "nbody"):
        nbody.main(["--outdir", str(outdir)])

    if args.task in ("all", "ee"):
        run_monte_carlo_report(outdir=ee_outdir, mc_runs=args.mc_runs, seed=args.seed)

    if args.task in ("all", "figures"):
        generate_all_figures(outdir=outdir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
