"""
Sequential grid search over CWSF_ALPHA_H_ENHANCE × CWSF_T_TRANSITION_MIN_GYR.

Runs ``python -u cwsf_pipeline.py`` for each combo so MCMC progress bars and prints
stream live (do not use capture_output).

Usage (from this directory):
    python tune_blend.py

Quick smoke test: set QUICK=1 in the environment or edit base_env below.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PIPELINE = ROOT / "cwsf_pipeline.py"

# --- MCMC / suite toggles (override with env QUICK=1 for fast iteration) ---
_quick = os.getenv("QUICK", "").strip() in ("1", "true", "yes", "on")

base_env: dict[str, str] = {
    "PYTHONUNBUFFERED": "1",
    "CWSF_N_WALKERS": "16" if _quick else "48",
    "CWSF_N_BURN": "100" if _quick else "2000",
    "CWSF_N_PROD": "200" if _quick else "5000",
    "CWSF_N_FRAMEWORK_SEEDS": "1",
    "CWSF_RUN_CV": "0",
    "CWSF_RUN_NESTED": "0",
    "CWSF_ROBUSTNESS": "0",
    "CWSF_VALIDATION_SUITE": "0",
    "CWSF_PUBLICATION_SUITE": "0",
}

# Grid: α lock-time sweep (includes 9.5 Gyr from the physics note + 9.0 baseline + 10.0)
ALPHAS = [0.040, 0.020, 0.010, 0.0]
LOCKS_GYR = [9.0, 9.5, 10.0]


def main() -> int:
    if not PIPELINE.is_file():
        print(f"Missing pipeline script: {PIPELINE}", file=sys.stderr)
        return 1

    print(
        f"tune_blend: QUICK={_quick} walkers={base_env['CWSF_N_WALKERS']} "
        f"burn={base_env['CWSF_N_BURN']} prod={base_env['CWSF_N_PROD']}"
    )
    print(f"tune_blend: {len(ALPHAS)} alphas × {len(LOCKS_GYR)} locks = {len(ALPHAS) * len(LOCKS_GYR)} runs\n")

    merged_base = {**os.environ, **base_env}

    for alpha in ALPHAS:
        for lock in LOCKS_GYR:
            tag_a = f"{alpha:g}".replace(".", "p")
            tag_l = f"{lock:g}".replace(".", "p")
            outdir = ROOT / f"tune_a{tag_a}_l{tag_l}"
            outdir.mkdir(parents=True, exist_ok=True)

            env = {
                **merged_base,
                "CWSF_OUTDIR": str(outdir),
                "CWSF_ALPHA_H_ENHANCE": str(alpha),
                "CWSF_T_TRANSITION_MIN_GYR": str(lock),
            }

            print("\n" + "=" * 72)
            print(f"=== alpha={alpha}  T_lock={lock} Gyr  ->  {outdir.name} ===")
            print("=" * 72 + "\n", flush=True)

            cmd = [sys.executable, "-u", str(PIPELINE)]
            rc = subprocess.run(cmd, cwd=str(ROOT), env=env).returncode

            if rc != 0:
                print(f"\n*** tune_blend: run FAILED with exit code {rc} ***\n", flush=True)
                return rc

    print("\n=== ALL DONE ===\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
