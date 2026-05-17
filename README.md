# CWSF Cosmology Pipeline

This repository contains all the computational work used to explore Lambda-CDM+S. In this project, the core physics were defined as solutions to the Friedmann Equations. The parameters involved in the solutions were explored through a Monte Carlo Ensemble, and then constrained through a Markov Chain Monte Carlo using data from Type Ia Supernovae, the Cosmic Microwave Background, Galaxies, and the Baryon Acoustic Oscillations through the Planck, Pantheon+, DESI, DES and SDSS datasets. The codebase is split into several modular components:
- `cwsf_pipeline.py` (Full Bayesian Inference)
- `ccomplet2ee.py` (Core physics of the Entropy-driven model and Streamlit interactive modelling)
- `nbody_simulations.py` (N-Body cosmological simulations based on the posteriors)

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
- Run only posterior-driven N-body Simulation:
  - `python main.py --task nbody --outdir cwsf_output`
- Run only ccomplet2ee parameter exploration:
  - `python main.py --task ee --ee-outdir ee_output --mc-runs 300 --seed 42`
- Generate figures from posterior outputs:
  - `python main.py --task figures --outdir cwsf_output`
