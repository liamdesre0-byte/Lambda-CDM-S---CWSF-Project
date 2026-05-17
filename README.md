# ΛCDM+S Computational Cosmology Framework

A computational cosmology project exploring whether the accelerated expansion of the universe can be reproduced and explained by entropy growth associated with the Hubble Horizon of the universe without requiring dark energy as a fundamental component.

This repository contains the full simulation pipeline, cosmological models, Monte Carlo analyses, posterior predictive checks, and visualization tools used for my 2026 Canada-Wide Science Fair (CWSF) project.

---

# Scientific Motivation

One of the most important discoveries in science is that the universe is expanding, and that this expansion is accelerating.

The current standard cosmological model, known as ΛCDM (Lambda Cold Dark Matter), explains this acceleration using dark energy. However, despite decades of research, the physical nature of dark energy remains unknown.

In recent years, increasingly precise astronomical observations have revealed several tensions within ΛCDM, particularly involving:
- the Hubble constant (H_0),
- late-universe supernova measurements,
- and discrepancies between early-universe and late-universe observations.

This project explores whether thermodynamic principles associated with black hole horizons can provide an alternative explanation for cosmic acceleration.

---

# Core Idea

Both black holes and the observable universe possess horizons.

Since black hole entropy is proportional to horizon area through the Bekenstein-Hawking relation:
S ∝ A. 
This raises the possibility that the Hubble Horizon of the universe may also obey this relation.

The central hypothesis explored in this project is:

> increasing entropy in the cosmic horizon may naturally drive accelerated cosmic expansion.

Under this framework:
- the early universe behaves like standard gravity-dominated ΛCDM,
- while the late universe transitions toward an entropy-driven expansion regime, hence the name of the new proposed model:
- ΛCDM+S
where S represents the thermodynamic entropy contribution.

---

# Repository Structure

```text
cwsf_cosmo_pipeline/
│
├── analysis/
├── app/
├── cwsf_output/
├── data/
├── figures/
├── figures_mc_smoke/
├── figures_mc_test/
├── figures_mc_test2/
├── models/
├── report/
├── simulations/
├── visualization/
│
├── LICENSE
├── README.md
│
├── ccomplet2ee.py
├── cwsf_pipeline.py
├── mcplots.py
├── nbody_simulations.py
├── sigmoid_dataset.py
├── main.py
│
├── requirements-complete2ee.txt
└── requirements.txt
```

---

# Main Components

## `cwsf_pipeline.py`

Primary cosmology analysis pipeline.

Handles:
- cosmological evolution calculations,
- Markov Chain Monte Carlo Posterior Sampling,
- residual analysis,
- and figure generation.

---

## `nbody_simulations.py`

Numerical large-scale cosmological simulations used to study emergent structure and horizon evolution under modified expansion behavior.

---

## `ccomplet2ee.py`

Extended cosmological computation engine used for:
- parameter evaluation,
- and entropy-driven Hubble Horizon evolution calculations.

---

## `mcplots.py`

Monte Carlo visualization utilities for:
- posterior distributions,
- parameter evolution,
- uncertainty regions,
- and convergence diagnostics.

---

## `sigmoid_dataset.py`

Dataset used to determine the sensitivity of parameters in the proposed transitioning framework.

---

# Mathematical Framework

The project numerically evaluates modified Friedmann evolution equations describing cosmic expansion through time.

The framework was designed so that:

- the early universe reduces to standard gravity-dominated expansion,
- while the late universe transitions into an entropy-dominated accelerating regime.

The transition behavior is modeled through a smooth, continouous Sigmoid Function rather than through abrupt phase changes.

The equations are solved numerically because no complete analytical solution exists for the coupled system explored here.

---

# Computational Methods

The cosmological parameter space was explored using large-scale Monte Carlo simulations in 4-dimensional parameter space.

The simulations:
- randomly sample 4 cosmological parameters,
- evolve the cosmic horizon numerically,
- and compare predictions against observational datasets.

The project used:
- Monte Carlo methods,
- Markov Chain Monte Carlo (MCMC),
- posterior predictive simulations,
- and residual density comparisons.

The computational pipeline explored approximately:

```text
~10,000,000 simulated samples
```

across 4-dimensional parameter space.

---

# Observational Constraints

Model parameters were constrained using observational cosmology datasets including:
- Type Ia supernova measurements from the Pantheon+ and DES SN5 dataset,
- luminosity distance observations from the SH0ES dataset,
- and early-universe expansion constraints from the Planck and SDSS datasets.

---

# Posterior Predictive Results

The analysis suggests that:
- most cosmological parameters remain broadly consistent with ΛCDM,
- while the preferred Hubble constant shifts toward larger late-universe values.

Representative values explored include:

| Parameter | Approximate Value |
|---|---|
| Standard ΛCDM \(H_0\) | 67 km/s/Mpc |
| ΛCDM+S  \(H_0\) | 72.8 km/s/Mpc |

A larger \(H_0\) predicts:
- a faster present-day expansion rate,
- brighter late-universe supernovae,
- and slightly smaller inferred cosmological distances.

Interestingly, observed DES-SN5 supernova residuals appear slightly brighter than standard ΛCDM predictions, producing agreement with the model’s predicted residual direction.

---

# Important Interpretation Notes

This project does **not** claim to falsify or disprove ΛCDM.

Instead, the work explores whether an entropy-based cosmic expansion can:
- reproduce observational behavior,
- and reduce certain late-universe tensions.

---

# Installation

Clone the repository:

```bash
git clone https://github.com/your-username/Lambda-CDM-Entropy.git
cd Lambda-CDM-Entropy
```

Install dependencies:

```bash
pip install -r requirements.txt
```

or:

```bash
pip install -r requirements-complete2ee.txt
```

---

# Running the Pipeline

Main execution:

```bash
python main.py
```

Individual modules:

```bash
python cwsf_pipeline.py
python nbody_simulations.py
python mcplots.py
```

---

# Outputs

Running the pipeline may generate:
- posterior predictive plots,
- MCMC chains,
- cosmological evolution figures,
- Monte Carlo diagnostics,
- residual analyses,
- simulation outputs and figures,
- and processed datasets.

Outputs are generally written to:

```text
figures/
cwsf_output/
report/
```

---

# Reproducibility

For reproducibility, preserve:
- package versions,
- Python version,
- random seeds,
- posterior initialization settings,
- observational dataset versions,
- preprocessing methods,
- and plotting configurations.

---

# Citation

If referencing this project:

```text
Liam Desreu
ΛCDM+S Computational Cosmology Framework
Canada-Wide Science Fair (CWSF), 2026
```

---

# License

This project is released for educational and research purposes under the MIT License.

See `LICENSE` for details.
