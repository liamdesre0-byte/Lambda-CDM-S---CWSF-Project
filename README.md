# CWSF Cosmology Pipeline

Modularized research codebase for your CWSF workflow, based on:
- `cwsf_pipeline.py` (full inference + publication pipeline)
- `ccomplet2ee.py` (entropy-extended analysis and Streamlit tooling; aligned to your `ccomplet2.py` direction)
- `nbody_simulations.py` (posterior-driven PM and diagnostic simulation stack; aligned to your `nbody_sims.py` work)

## Project Structure

```text
cwsf_cosmo_pipeline/
├── main.py
├── cwsf_pipeline.py
├── ccomplet2ee.py
├── nbody_simulations.py
├── data/
├── models/
├── simulations/
├── analysis/
├── visualization/
└── app/
```

The large single-file scripts are kept as reproducibility baselines, while the new package folders provide cleaner interfaces for GitHub presentation and future maintenance.

## Quick Start

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the full orchestrated flow:

```bash
python main.py --task all
```

## Common Tasks

- Run only the core cosmology pipeline:
  - `python main.py --task pipeline`
- Run only posterior-driven N-body diagnostics:
  - `python main.py --task nbody --outdir cwsf_output`
- Run only ccomplet2ee sensitivity + MC study:
  - `python main.py --task ee --ee-outdir ee_output --mc-runs 300 --seed 42`
- Generate publication figures from existing outputs:
  - `python main.py --task figures --outdir cwsf_output`

## Notes

- Keep `cwsf_pipeline.py`, `ccomplet2ee.py`, and `nbody_simulations.py` in the repository as your reference implementations.
- Keep outputs (`cwsf_output/`, `ee_output/`) out of version control unless you are intentionally publishing a small, curated artifact set.
