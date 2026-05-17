#!/usr/bin/env python3
r"""
CWSF horizon-area blend vs flat :math:`\Lambda`CDM — publishable single-file pipeline
====================================================================================

Scientific goal
---------------
Test a **horizon-area blend** model (gravity-dominated early branch + thermodynamic late
branch, joined by a sigmoid in cosmic time) against **flat** :math:`\Lambda`\ **CDM** on
supernova distance moduli. Training: **Pantheon+ SH0ES** (diagonal errors by default).
Hold-out validation: **DES Dovekie HD** (DES-SN5YR) — **not** included in the likelihood;
used only for posterior predictive / point-prediction checks at posterior summaries.

The blend is **not** tuned to match :math:`\Lambda`\ **CDM** by construction; the comparison
is empirical (training fit metrics + independent hold-out behavior).

Mathematical model (aligned with slide-deck framing; implementation detail below)
-------------------------------------------------------------------------------
**Branch areas** (normalized by a shared :math:`A_0`; only ratios enter):

- Gravity-dominated: :math:`A_{\mathrm{gr}}(t) = A_0\, (t/t_0)^{4/3}`.
- Thermodynamic: :math:`A_{\mathrm{thermo}}(t) =
  A_0\, \exp\bigl(2 H_\Lambda (t-t_0)\bigr)`, with
  :math:`H_\Lambda = H_0 \sqrt{\Omega_\Lambda}`.

**Sigmoid:** :math:`w(t) = 1/\bigl(1 + \exp(-k(t-t_{\mathrm{crit}}))\bigr)`.

**Blend:** :math:`A_{\mathrm{blend}}(t) = (1-w)\,A_{\mathrm{gr}} + w\,A_{\mathrm{thermo}}`.

**Residual ratio (area diagnostic):** :math:`r(t)=\log_{10}(A_{\mathrm{blend}}/A_{\mathrm{LCDM}})` with the
same branch definitions as ``ccomplet2.py`` (early lock + sigmoid blend in :math:`\log_{10}A` space).

**Default blend distance modulus (``CWSF_BLEND_PHYSICS_MODE=ccomplet2``):** self-consistent luminosity
distance from the ported ``ccomplet2`` background — build :math:`a(t)` from blended areas, calibrate and
integrate :math:`H(z)`, then

.. math::

    \mu_{\mathrm{blend}}(z) = 5\log_{10}\bigl(d_L(z)/(10\,\mathrm{pc})\bigr) + M,

with **M** the usual nuisance magnitude. **No** LCDM spline backbone is used for this path.

**Legacy mode** (``CWSF_BLEND_PHYSICS_MODE=legacy_residual``): older pipeline with
:math:`\mu=\mu_{\mathrm{LCDM}}+2.5\,r` using LCDM :math:`t(z)` and a closed-form :math:`\chi` area proxy for
:math:`A_{\mathrm{LCDM}}` (diagnostic / backward compatibility only).

Flat :math:`\Lambda`\ **CDM** with radiation is retained for the **LCDM** model chain, LCDM comparison
plots, and high-:math:`z` extensions of the blend :math:`H(z)` table. Optional compressed BAO / CMB Gaussian
blocks are **off** unless you set the documented environment flags.

Flatness convention (critical)
------------------------------
Do **not** treat :math:`\Omega_m` and :math:`\Omega_\Lambda` as independent when flatness holds.

.. math::

    \Omega_\Lambda = 1 - \Omega_m - \Omega_r(H_0), \quad
    \Omega_r(H_0) = 2.469\times 10^{-5}\, h^{-2}\, (1 + 0.2271\times 3.046),\quad
    h = H_0/100.

Free parameters fitted here
---------------------------
**Flat LCDM:** :math:`H_0,\ \Omega_m,\ M`.

**Blend:** :math:`H_0,\ \Omega_m,\ t_{\mathrm{crit}},\ k,\ M` (:math:`\Omega_\Lambda` derived).

**Slide-deck Gaussian prior centers** (truncated normals on physical bounds):

- :math:`H_0 = 72.8\ \mathrm{km\,s^{-1}\,Mpc^{-1}}` (:math:`\sigma=2`).
- :math:`\Omega_m = 0.30` (:math:`\sigma=0.03`).
- :math:`t_{\mathrm{crit}} = 15.9` Gyr (:math:`\sigma=1.5` Gyr).
- :math:`k = 0.37\ \mathrm{Gyr^{-1}}` (:math:`\sigma=0.07\ \mathrm{Gyr^{-1}}`).
- :math:`M`: wide Gaussian (implementation uses generous truncated normal).

Hard bounds: :math:`H_0\in[55,85]`, :math:`\Omega_m\in[0.05,0.55]`,
:math:`t_{\mathrm{crit}}\in[5,40]`\,Gyr, :math:`k\in[0.01,2.0]\ \mathrm{Gyr^{-1}}`,
:math:`M` in a broad magnitude interval (see ``BND``). If posterior mass piles on a boundary,
inspect outputs and treat parameters as poorly identified — do **not** over-interpret bounds.

Data
----
- **Training:** Pantheon+ SH0ES ``MU_SH0ES`` / ``MU_SH0ES_ERR_DIAG`` at ``zCMB`` (URL in
  ``PANT_URL``). Calibrators optionally filtered (``CWSF_KEEP_CALIBRATORS``).
- **Hold-out:** DES Dovekie HD (``zHD``, ``MU``, combined errors) — ``DES_URL``.

Inference
---------
**emcee** affine-invariant ensemble MCMC: default **48** walkers, **3** independent chains
(different RNG seeds), **2000** burn-in, **5000** production steps per chain, base **seed
2026** (``RNG_SEED``). **Nuisance** :math:`M` is **profiled out** of the Gaussian diagonal
likelihood by default (``CWSF_PROFILE_M=1``), which breaks the :math:`H_0`–:math:`M`
degeneracy and reduces fitted dimensions to :math:`(H_0,\Omega_m)` vs
:math:`(H_0,\Omega_m,t_{\mathrm{crit}},k)`.

In **legacy_residual** mode the horizon residual was anchored so :math:`r(z=0)=0` relative to a
LCDM-area proxy; the default **ccomplet2** path does **not** use that construction for :math:`\mu(z)`.

Cached **LCDM** comoving distance / cosmic age splines per :math:`(H_0,\Omega_m)` and cached **blend**
:math:`(z,H,D_M,t)` tables per :math:`(H_0,\Omega_m,t_{\mathrm{crit}},k)` avoid rebuilding interpolators
on every likelihood evaluation.

Convergence: multi-chain and split-chain :math:`\hat R`, autocorrelation time, conservative
ESS, acceptance fraction; ``summary.json`` warns when :math:`\hat R` exceeds
``CWSF_RHAT_TARGET`` (default **1.05**) or ESS falls below ``CWSF_MIN_ESS``.

Likelihoods & scores (training diagonal Gaussian default)
-------------------------------------------------------
- :math:`\chi^2`, reduced :math:`\chi^2` with correct dof (:math:`N - k`).
- Maximum log-likelihood, **AIC**, **BIC**, **WAIC** from posterior subsample of pointwise
  log-likelihoods.

**DES:** RMSE, :math:`\chi^2` (diag), bias, residual std; compare to predictions at posterior
**medians** (and optionally at approximate max-like draw); not used during fitting.

Model comparison rhetoric
-------------------------
Claim the blend is "better" than LCDM **only if** DES hold-out **RMSE** improves **and**
AIC/BIC do not incur a severe penalty for the extra parameters; otherwise report the blend as
a physically motivated **alternative** with comparable or weaker predictive scores.

Artifacts (under ``OUTDIR``, default ``./cwsf_output``)
-------------------------------------------------------
Cleaned Pantheon+/DES CSVs; ``mcmc_chain_lcdm.csv`` / ``mcmc_chain_blend.csv``;
``predictive_*.csv``; ``summary.json`` (metadata, posterior summaries, ICs, convergence,
URLs, interpretation); figures under ``figures/`` — Hubble/residual bands, corners (if
**corner** installed), predictive overlays, traces as implemented in ``main()``.

Environment knobs
-----------------
Core: ``CWSF_OUTDIR``, ``CWSF_REFETCH``, ``CWSF_Z_NODES``, ``CWSF_PROFILE_M``, ``CWSF_N_CHAINS``, ``CWSF_N_WALKERS``,
``CWSF_N_BURN``, ``CWSF_N_PROD``, ``CWSF_RHAT_TARGET``, ``CWSF_MIN_ESS``, ``CWSF_WAIC_DRAWS``, ``CWSF_LONG_TABLE_DRAWS``,
``CWSF_KEEP_CALIBRATORS``, ``CWSF_LEGACY_PRIORS`` (wide exploratory box), ``CWSF_USE_COV`` (full Pantheon STATONLY covariance path).

Blend background tuning (``c2_`` / ``CosmoInterpEngine``): ``CWSF_ALPHA_H_ENHANCE`` (late-time ``H(z)`` boost; **default 0.010** after calibration tuning),
``CWSF_T_TRANSITION_MIN_GYR`` (early-universe lock; default **9.0**),
``CWSF_BLEND_CALIBRATION`` — ``two_point`` (match ΛCDM at ``t=t_lock`` and today; default) or ``lsq_loga`` (least-squares linear fit of ``log a_\mathrm{LCDM}`` vs ``log a_\mathrm{blend,raw}`` on ``t\in[t_\mathrm{lock},T_\mathrm{ref}]``, diagnostic / alternate physics).

Multi-seed / robustness: ``CWSF_N_FRAMEWORK_SEEDS`` (independent emcee reruns, default 5);

Publication PNG bundle **without MCMC** (reads ``OUTDIR`` CSV/JSON): ``python cwsf_pipeline.py paper-figures [OUTDIR]`` writes ``figures_for_paper/*.png``.

Cross-validation: ``CWSF_RUN_CV``, ``CWSF_CV_FOLDS``, ``CWSF_CV_REPEATS``, ``CWSF_CV_N_BURN``, ``CWSF_CV_N_PROD``,
``CWSF_CV_N_CHAINS``, ``CWSF_CV_N_WALKERS``, ``CWSF_CV_SEED``;

Nested sampling: ``CWSF_RUN_NESTED``, ``CWSF_NESTED_NLIVE``, ``CWSF_NESTED_DLOGZ`` (requires **dynesty**);

Systematics-style shorts: ``CWSF_ROBUSTNESS``, ``CWSF_ROBUST_SHORT_BURN``, ``CWSF_ROBUST_SHORT_PROD``;

Other: ``CWSF_PANTHEON_COV_URL``, ``CWSF_REDSHIFT_SLICES`` (training bin count for residual tables / plots),
``CWSF_H_CONSISTENCY_DIAG`` (default **1**: save **figures/diagnostic_Hz_splinechi_vs_Friedmann.png**).

Optional **embedded** Gaussian BAO / compressed CMB extensions (same file; additive on the log-posterior, **same** SN priors and sampler dimensions):
``CWSF_USE_BAO``, ``CWSF_USE_CMB`` (each ``0``/``1``; **defaults ``1``** for joint SN+BAO+CMB-style inference — turn off for SN-only studies), ``CWSF_BAO_JSON``, ``CWSF_CMB_SHIFT_JSON`` (optional override paths; if absent, **in-memory** dicts ``_TEMPLATE_*_DICT`` are used),
``CWSF_OMEGA_B_H2`` (fixed baryon density for r_d / CMB shift mapping; **not** a free parameter). ``CWSF_WRITE_LIKELIHOOD_TEMPLATE_JSON=1`` optionally writes JSON copies of those dicts to ``OUTDIR`` for inspection only.

Galaxy-scale MOND benchmark (**never** mixed into cosmology): ``CWSF_SPARC_DIR`` pointing to a folder of SPARC ``*rotmod*.dat`` files.

**Distance ladder:** comoving/spline $d_L$ is uniformly rescaled at construction time so $d_L(z\\!\\to\\!0)\\approx c z/H_0$
(``anchor_luminosity_distance_lowz_hubbles`` inside ``mu_lcdm_shape``), shared by LCDM and blend baselines.
**Blend-only:** ``mu_blend_shape`` rejects catastrophic distance moduli ($\\mu\\notin[20,50]$ or non-finite) with NaNs, which
forces $-\\infty$ log-likelihood via finite checks in ``lnlike_blend_*`` / covariance paths.

Numerical hygiene
-----------------
Explicit units (Gyr, km s^-1 Mpc^-1, mag); clipped exponentials; stable logistic; monotonic
PCHIP on comoving integrals; radiation included when :math:`\Omega_\Lambda` is derived.

Publication extensions (additive; ``CWSF_PUBLICATION_SUITE=1`` default): under ``OUTDIR`` also creates ``tables/``, ``synthetic/``,
``diagnostics/``, ``holdout/``, ``posterior/`` with CSV diagnostics (engine self-test, Fisher-style block, prior–posterior overlap,
reproducibility manifest, referee verdict), ``run_log.txt``, synthetic null/injection short-MCSV summaries, and PNG ``w_eff`` diagnostic.
Disable entirely with ``CWSF_PUBLICATION_SUITE=0``.

This module is intended as a **reproducible appendix driver**: run ``python cwsf_pipeline.py``
after ``pip install numpy scipy pandas matplotlib emcee``; optional: ``corner``, ``dynesty``, ``pyarrow`` (Parquet export).
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import sys
import urllib.request
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import linalg as la_sci
from scipy import stats
from scipy.integrate import cumulative_trapezoid, quad
from scipy.interpolate import PchipInterpolator, interp1d
from scipy.special import logsumexp

try:
    from importlib import metadata as importlib_metadata
except ImportError:
    import importlib_metadata  # type: ignore

try:
    import emcee

    HAVE_EMCEE = True
except Exception:
    emcee = None  # type: ignore
    HAVE_EMCEE = False

try:
    import corner

    HAVE_CORNER = True
except Exception:
    corner = None  # type: ignore
    HAVE_CORNER = False

try:
    import dynesty

    HAVE_DYNESTY = True
except Exception:
    dynesty = None  # type: ignore
    HAVE_DYNESTY = False

try:
    import pyarrow  # noqa: F401

    HAVE_PARQUET = True
except Exception:
    HAVE_PARQUET = False

HAVE_EXTERNAL_LIKES = True

_EXTERNAL_JOINT_PACK: Any = None
_CWSF_LNPOST_IS_BLEND: bool = False
_CWSF_EXT_THETA: np.ndarray | None = None


def _stable_hash_u64(s: str) -> int:
    """Deterministic 64-bit unsigned mix from a UTF-8 string (replaces ``hash()`` for RNG seeds)."""
    return int.from_bytes(hashlib.sha256(s.encode("utf-8")).digest()[:8], "little", signed=False)


def _external_lnlike_add(theta: np.ndarray, eng: CosmoInterpEngine) -> float:
    """Optional BAO + compressed CMB log-likelihood (flat FRW; does not change SN priors/parameters)."""
    if not HAVE_EXTERNAL_LIKES:
        return 0.0
    pack = globals().get("_EXTERNAL_JOINT_PACK")
    if pack is None:
        return 0.0
    try:
        globals()["_CWSF_EXT_THETA"] = np.asarray(theta, dtype=float).copy()
        return float(pack.lnlike_add(float(theta[0]), float(theta[1]), eng))
    except Exception:
        return float(-np.inf)

warnings.simplefilter(action="ignore", category=FutureWarning)


C_KMS = 299792.458
GYR_S = float(3.15576e16)
MPC_KM = float(3.085677581e19)
KMS_TO_SI = float(3.2408e-20)
T_REF_GYR = float(13.8)
T_REF_S = float(T_REF_GYR * GYR_S)

RNG_SEED = int(2026)
N_WALKERS = int(os.getenv("CWSF_N_WALKERS", "48"))
N_CHAINS = int(os.getenv("CWSF_N_CHAINS", "3"))
N_BURN = int(os.getenv("CWSF_N_BURN", "2000"))
N_PROD = int(os.getenv("CWSF_N_PROD", "5000"))
RHAT_TARGET = float(os.getenv("CWSF_RHAT_TARGET", "1.05"))
MIN_ESS_TARGET = float(os.getenv("CWSF_MIN_ESS", "400"))
PROFILE_M = os.getenv("CWSF_PROFILE_M", "1") != "0"
LONG_TABLE_DRAWS = int(os.getenv("CWSF_LONG_TABLE_DRAWS", "512"))
Z_NODES = int(os.getenv("CWSF_Z_NODES", "6144"))
ZP_LO = float(np.nextafter(1.0, 2.0))
N_FRAMEWORK_SEEDS = int(os.getenv("CWSF_N_FRAMEWORK_SEEDS", "5"))
N_CV_REPEATS = int(os.getenv("CWSF_CV_REPEATS", "3"))
N_CV_FOLDS = int(os.getenv("CWSF_CV_FOLDS", "5"))
CV_N_BURN = int(os.getenv("CWSF_CV_N_BURN", "400"))
CV_N_PROD = int(os.getenv("CWSF_CV_N_PROD", "1200"))
CV_N_CHAINS = int(os.getenv("CWSF_CV_N_CHAINS", "1"))
CV_N_WALKERS = int(os.getenv("CWSF_CV_N_WALKERS", "32"))
RUN_NESTED = os.getenv("CWSF_RUN_NESTED", "1") == "1"
NESTED_NLIVE = int(os.getenv("CWSF_NESTED_NLIVE", "400"))
USE_COVARIANCE = os.getenv("CWSF_USE_COV", "0") == "1"
RUN_ROBUSTNESS = os.getenv("CWSF_ROBUSTNESS", "1") == "1"
RUN_CV = os.getenv("CWSF_RUN_CV", "0") == "1"
RNG_CV_SEED = int(os.getenv("CWSF_CV_SEED", "4242"))
ROB_SHORT_BURN = int(os.getenv("CWSF_ROBUST_SHORT_BURN", str(CV_N_BURN)))
ROB_SHORT_PROD = int(os.getenv("CWSF_ROBUST_SHORT_PROD", str(CV_N_PROD)))

NESTING_DLOGZ = float(os.getenv("CWSF_NESTED_DLOGZ", "0.2"))

USE_LOWZ_DL_ANCHOR = os.getenv("CWSF_USE_LOWZ_DL_ANCHOR", "1") == "1"
USE_BLEND_HORIZON_DELTA_MU = os.getenv("CWSF_USE_BLEND_HORIZON_DELTA_MU", "1") == "1"
# ``ccomplet2`` (default): self-consistent blend H(z), χ(z), μ(z) from ported Superior Suite physics.
# ``legacy_residual``: older μ = μ_LCDM + 2.5 Δr anchored at z=0 (diagnostic / backward compatibility only).
BLEND_PHYSICS_MODE = os.getenv("CWSF_BLEND_PHYSICS_MODE", "ccomplet2").strip().lower()
Z_REF_DELTA_MU_CORR = float(os.getenv("CWSF_Z_REF_DELTA_MU_CORR", "0.55"))
ISEF_PEARSON_BOOTSTRAP = int(os.getenv("CWSF_ISEF_PEARSON_BOOT", "4000"))
ISEF_POSTERIOR_SUBSAMPLE_CORR = int(os.getenv("CWSF_ISEF_POSTERIOR_ROWS_FOR_CORR", "8000"))

# ---------------------------------------------------------------------------
# Publication-grade extension suite (additive; disable with CWSF_PUBLICATION_SUITE=0)
# ---------------------------------------------------------------------------
PUBLICATION_SUITE = os.getenv("CWSF_PUBLICATION_SUITE", "1") != "0"
SELFTEST_AT_BOOT = os.getenv("CWSF_SELFTEST", "1") != "0"
SYNTH_MC_BURN = int(os.getenv("CWSF_SYNTH_MC_BURN", "400"))
SYNTH_MC_PROD = int(os.getenv("CWSF_SYNTH_MC_PROD", "800"))
SYNTH_MC_WALKERS = int(os.getenv("CWSF_SYNTH_MC_WALKERS", "40"))
SYNTH_MC_CHAINS = int(os.getenv("CWSF_SYNTH_MC_CHAINS", "1"))
# --- Validation / falsification suite (additive; CWSF_VALIDATION_SUITE=0 to skip) ---
VALIDATION_SUITE = os.getenv("CWSF_VALIDATION_SUITE", "1") != "0"
NULL_FPR_TRIALS = int(os.getenv("CWSF_NULL_FPR_TRIALS", "14"))
INJECTION_TRIALS = int(os.getenv("CWSF_INJECTION_TRIALS", "10"))
ADVERSARIAL_SCENARIOS = int(os.getenv("CWSF_ADVERSARIAL_SCENARIOS", "10"))
HOLDOUT_REPEAT_SPLITS = int(os.getenv("CWSF_HOLDOUT_REPEAT_SPLITS", "5"))
HOLDOUT_TRAIN_FRAC = float(os.getenv("CWSF_HOLDOUT_TRAIN_FRAC", "0.82"))
VALIDATION_CV_BURN = int(os.getenv("CWSF_VALIDATION_CV_BURN", str(max(300, CV_N_BURN))))
VALIDATION_CV_PROD = int(os.getenv("CWSF_VALIDATION_CV_PROD", str(max(600, CV_N_PROD))))
VALIDATION_CV_WALKERS = int(os.getenv("CWSF_VALIDATION_CV_WALKERS", str(max(32, CV_N_WALKERS))))
PPC_CALIB_DRAWS = int(os.getenv("CWSF_PPC_CALIB_DRAWS", "320"))

PANT_COV_STATSYS_URL = (
    "https://raw.githubusercontent.com/PantheonPlusSH0ES/DataRelease/main/"
    "Pantheon%2B_Data/4_DISTANCES_AND_COVAR/Pantheon%2BSH0ES_STAT%2BSYS.cov"
)


def infer_profile_m() -> bool:
    """Analytic :math:`M` profiling is disabled when a full covariance likelihood is active."""
    return bool(PROFILE_M) and (not USE_COVARIANCE)

PANT_URL = (
    "https://raw.githubusercontent.com/PantheonPlusSH0ES/DataRelease/main/"
    "Pantheon+_Data/4_DISTANCES_AND_COVAR/Pantheon+SH0ES.dat"
)
PANT_COVSTAT_URL = os.getenv(
    "CWSF_PANTHEON_COV_URL",
    "https://raw.githubusercontent.com/PantheonPlusSH0ES/DataRelease/main/"
    "Pantheon+_Data/4_DISTANCES_AND_COVAR/Pantheon+SH0ES_STATONLY.cov",
)
DES_URL = (
    "https://raw.githubusercontent.com/des-science/DES-SN5YR/main/"
    "4_DISTANCES_COVMAT/DES-Dovekie_HD.csv"
)

OUTDIR = Path(os.getenv("CWSF_OUTDIR", "./cwsf_output")).resolve()
FIGDIR = OUTDIR / "figures"

PRI_MU = dict(H0=72.8, Om=0.30, tc=15.9, k=0.37)
if os.getenv("CWSF_LEGACY_PRIORS", "0") == "1":
    PRI_SG = dict(H0=2.0, Om=0.03, tc=1.5, k=0.07)
    BND = dict(H0=(55.0, 85.0), Om=(0.05, 0.55), tc=(5.0, 40.0), k=(0.01, 2.0), M=(-30.0, 35.0))
    PRI_MAG_MU = float(-19.0)
    PRI_MAG_SG = float(25.0)
else:
    PRI_SG = dict(H0=1.5, Om=0.02, tc=0.8, k=0.04)
    BND = dict(H0=(70.0, 76.5), Om=(0.26, 0.34), tc=(14.0, 17.5), k=(0.25, 0.50), M=(-22.0, -16.5))
    PRI_MAG_MU = float(-19.0)
    PRI_MAG_SG = float(6.0)

RNG = np.random.default_rng(RNG_SEED)



def eps() -> float:
    return float(np.finfo(np.float64).tiny)


def omega_r(h0_kmsmpc: float) -> float:
    h = float(h0_kmsmpc) / 100.0
    return float((2.469e-5 / (h * h)) * (1.0 + 0.2271 * 3.046))


def validate_flat_background(h0: float, Omega_m: float, eng: CosmoInterpEngine, z_max_probe: float) -> bool:
    """Reject unphysical / numerically pathological backgrounds inside the hard prior box."""
    OL = olambda_flat(float(Omega_m), float(h0))
    if not math.isfinite(OL) or OL < 1e-7:
        return False
    zp = np.linspace(0.0, max(float(z_max_probe), 0.05), 41, dtype=float)
    try:
        Ezarr = Hz(zp, float(h0), float(Omega_m))
        if not np.all(np.isfinite(Ezarr)) or float(np.min(Ezarr)) <= 0.0:
            return False
        tgy = eng.cosmic_age_gyr(zp, float(h0), float(Omega_m))
        if not np.all(np.isfinite(tgy)):
            return False
        dt = np.diff(tgy)
        if not np.all(dt < 1e-4):
            return False
    except Exception:
        return False
    return True


def validate_blend_background(
    h0: float, Omega_m: float, tcrit: float, k_slope: float, eng: CosmoInterpEngine, z_max_probe: float
) -> bool:
    """Sanity checks for ccomplet2-style blend geometry (LCDM comparison uses :func:`validate_flat_background` only)."""
    OL = olambda_flat(float(Omega_m), float(h0))
    if not math.isfinite(OL) or OL < 1e-7:
        return False
    zp = np.linspace(0.0, max(float(z_max_probe), 0.05), 41, dtype=float)
    try:
        hb = eng.blend_Hz(zp, float(h0), float(Omega_m), float(tcrit), float(k_slope))
        if not np.all(np.isfinite(hb)) or float(np.min(hb)) <= 0.0:
            return False
        dm = eng.blend_comoving_distance_mpc(zp, float(h0), float(Omega_m), float(tcrit), float(k_slope))
        if not np.all(np.isfinite(dm)):
            return False
        if np.any(np.diff(dm) < -1e-6 * (1.0 + np.abs(dm[:-1]))):
            return False
        tgy = eng.blend_cosmic_age_gyr(zp, float(h0), float(Omega_m), float(tcrit), float(k_slope))
        if not np.all(np.isfinite(tgy)):
            return False
        dt = np.diff(tgy)
        if not np.all(dt < 1e-4):
            return False
    except Exception:
        return False
    return True


def olambda_flat(Omega_m: float, h0: float) -> float:
    Or = omega_r(float(h0))
    return float(np.clip(1.0 - float(Omega_m) - Or, eps(), None))


# ---------------------------------------------------------------------------
# Canonical blend background physics (ported from ``ccomplet2.py`` / Superior Suite).
# All symbols prefixed ``_c2_`` / ``c2_``; uses c2 numeric constants for bit-level parity.
# ---------------------------------------------------------------------------
C2_GYR_TO_S = 3.1558e16
C2_A0_LOG10 = 54.0
C2_A0 = 10.0**C2_A0_LOG10
C2_OM_R_STD = 9.24e-5
C2_T_TRANSITION_MIN_GYR = float(os.getenv("CWSF_T_TRANSITION_MIN_GYR", "9.0"))
C2_T_FUTURE_DEFAULT_GYR = 50.0
# Lower α dims late-time H(z) enhancement → slightly larger low-z distances (recommended first pass vs legacy 0.040).
C2_ALPHA_H_ENHANCE = float(os.getenv("CWSF_ALPHA_H_ENHANCE", "0.010"))
C2_ZC_H_ENHANCE = 0.25


def _c2_blend_calibration_mode() -> str:
    """Two-point LCDM anchor (default) vs least-squares log(a) calibration (optional ``CWSF_BLEND_CALIBRATION``)."""
    v = os.getenv("CWSF_BLEND_CALIBRATION", "two_point").strip().lower()
    if v in ("lsq", "lsq_loga", "least_squares", "loga"):
        return "lsq_loga"
    return "two_point"


C2_BLEND_CALIB_MODE = _c2_blend_calibration_mode()
C2_ZMAX_H_ENHANCE = 0.60
C2_T_REF_S = float(T_REF_GYR * C2_GYR_TO_S)


def _c2_safe_log_sinh(x: np.ndarray) -> np.ndarray:
    x = np.atleast_1d(np.asarray(x, dtype=float))
    out = np.empty_like(x, dtype=float)
    sm = x < 20.0
    out[sm] = np.log(np.sinh(np.clip(x[sm], 1e-300, None)))
    out[~sm] = x[~sm] - np.log(2.0)
    return out


def _c2_omega_radiation(H0_kms: float) -> float:
    h = float(H0_kms) / 100.0
    omega_gamma = 2.469e-5 / max(h**2, 1.0e-12)
    return float(omega_gamma * (1.0 + 0.2271 * 3.046))


def c2_friedmann_H_cmb(z_tuple: tuple[float, ...] | list[float], H0_kms: float, OmM: float, OmL: float) -> np.ndarray:
    """Flat LCDM + radiation H(z) in km/s/Mpc (same as ``ccomplet2.friedmann_H_cmb``)."""
    z_arr = np.asarray(z_tuple, dtype=float)
    OmR = _c2_omega_radiation(float(H0_kms))
    OmK = max(1.0 - float(OmM) - float(OmL) - OmR, 0.0)
    Ez2 = OmR * (1.0 + z_arr) ** 4 + float(OmM) * (1.0 + z_arr) ** 3 + OmK * (1.0 + z_arr) ** 2 + float(OmL)
    return float(H0_kms) * np.sqrt(np.clip(Ez2, 1.0e-30, None))


def _c2_rad_matter_time_table(H0_kms: float, OmM: float, OmR: float) -> tuple[np.ndarray, np.ndarray]:
    a_grid = np.logspace(-10.0, np.log10(12.0), 6000)
    H0_si = max(float(H0_kms), 1.0e-9) * KMS_TO_SI
    omm = max(float(OmM), 1.0e-12)
    omr = max(float(OmR), 1.0e-14)
    denom = np.sqrt(omm / np.clip(a_grid, 1.0e-14, None) ** 3 + omr / np.clip(a_grid, 1.0e-14, None) ** 4)
    dt_da = 1.0 / np.clip(a_grid * H0_si * denom, 1.0e-30, None)
    t_grid_s = cumulative_trapezoid(dt_da, a_grid, initial=0.0)
    return t_grid_s / C2_GYR_TO_S, a_grid


def c2_log10_A_lcdm(t_gyr: np.ndarray, H0_kms: float, OmL: float, OmM: float) -> np.ndarray:
    H0si = float(H0_kms) * KMS_TO_SI
    ts = np.atleast_1d(np.asarray(t_gyr, dtype=float)) * C2_GYR_TO_S
    t0s = C2_T_REF_S
    arg = 1.5 * H0si * np.sqrt(max(float(OmL), 1.0e-9)) * ts
    arg0 = 1.5 * H0si * np.sqrt(max(float(OmL), 1.0e-9)) * t0s
    omm_safe = max(float(OmM), 1.0e-9)
    oml_safe = max(float(OmL), 1.0e-9)
    la = (1.0 / 3.0) * np.log(omm_safe / oml_safe) + (2.0 / 3.0) * _c2_safe_log_sinh(arg)
    la0 = float((1.0 / 3.0) * np.log(omm_safe / oml_safe) + (2.0 / 3.0) * _c2_safe_log_sinh(np.atleast_1d(arg0))[0])
    return (np.log(C2_A0) + 2.0 * (la - la0)) / np.log(10.0)


def c2_log10_A_rad_matter(t_gyr: np.ndarray, H0_kms: float, OmM: float, OmR: float = C2_OM_R_STD) -> np.ndarray:
    t_arr = np.asarray(t_gyr, dtype=float)
    t_tbl, a_tbl = _c2_rad_matter_time_table(float(H0_kms), float(OmM), float(OmR))
    a_of_t = interp1d(
        t_tbl,
        a_tbl,
        kind="linear",
        bounds_error=False,
        fill_value=(float(a_tbl[0]), float(a_tbl[-1])),
    )
    a_t = np.clip(np.asarray(a_of_t(np.clip(t_arr, 0.0, None)), dtype=float), 1.0e-40, None)
    a_ref = float(np.clip(a_of_t(np.array([float(T_REF_GYR)]))[0], 1.0e-40, None))
    return C2_A0_LOG10 + 2.0 * np.log10(a_t / a_ref)


def c2_log10_A_thermo(t_gyr: np.ndarray, H0_kms: float, OmL: float, t_future_gyr: float = C2_T_FUTURE_DEFAULT_GYR) -> np.ndarray:
    H_lam = float(H0_kms) * KMS_TO_SI * np.sqrt(max(float(OmL), 1.0e-12))
    dt = (np.asarray(t_gyr, dtype=float) - float(t_future_gyr)) * C2_GYR_TO_S
    return C2_A0_LOG10 + 2.0 * H_lam * dt / np.log(10.0)


def c2_log10_A_blend(
    t_gyr: np.ndarray,
    tc: float,
    k: float,
    H0_kms: float,
    OmL: float,
    OmM: float,
    t_future_gyr: float = C2_T_FUTURE_DEFAULT_GYR,
) -> np.ndarray:
    t = np.asarray(t_gyr, dtype=float)
    la_lcdm = c2_log10_A_lcdm(t, H0_kms, OmL, OmM)
    early = t < C2_T_TRANSITION_MIN_GYR
    out = np.empty_like(t, dtype=float)
    out[early] = la_lcdm[early]
    late = ~early
    if np.any(late):
        t_l = t[late]
        w = 1.0 / (1.0 + np.exp(np.clip(-float(k) * (t_l - float(tc)), -500.0, 500.0)))
        lg = c2_log10_A_rad_matter(t_l, H0_kms=H0_kms, OmM=OmM, OmR=C2_OM_R_STD)
        lt = c2_log10_A_thermo(t_l, H0_kms, OmL, t_future_gyr=t_future_gyr)
        A_b = (1.0 - w) * 10.0**lg + w * 10.0 ** np.clip(lt, -300.0, 500.0)
        out[late] = np.log10(np.clip(A_b, 1e-300, None))
    return out


def c2_r_blend(
    t_gyr: np.ndarray,
    tc: float,
    k: float,
    H0_kms: float,
    OmL: float,
    OmM: float,
    t_future_gyr: float = C2_T_FUTURE_DEFAULT_GYR,
) -> np.ndarray:
    t_arr = np.clip(np.asarray(t_gyr, dtype=float), 1.0e-6, max(float(t_future_gyr), 40.0))
    out = c2_log10_A_blend(t_arr, tc, k, H0_kms, OmL, OmM, t_future_gyr=t_future_gyr) - c2_log10_A_lcdm(t_arr, H0_kms, OmL, OmM)
    return np.nan_to_num(out, nan=0.0, posinf=6.0, neginf=-6.0)


def _c2_log10_A_blend_locked(
    t_gyr: np.ndarray,
    tc: float,
    k: float,
    H0_kms: float,
    OmL: float,
    OmM: float,
    t_future_gyr: float,
) -> np.ndarray:
    t_arr = np.asarray(t_gyr, dtype=float)
    logA_bl = c2_log10_A_blend(t_arr, tc, k, H0_kms, OmL, OmM, t_future_gyr=t_future_gyr)
    logA_lc = c2_log10_A_lcdm(t_arr, H0_kms, OmL, OmM)
    return np.where(t_arr < C2_T_TRANSITION_MIN_GYR, logA_lc, logA_bl)


def _c2_build_tz_lcdm(H0_kms: float, OmM: float, OmL: float) -> tuple[Any, np.ndarray, np.ndarray]:
    """Lightweight LCDM t(z) table for blend t(z) fallback (matches c2 ``build_tz_table`` density)."""
    z_tbl = np.logspace(-3, np.log10(3.5), 600)
    H0_si = float(H0_kms) * KMS_TO_SI

    def _t_single(zf: float) -> float:
        a_emit = 1.0 / (1.0 + float(zf))

        def integrand(a: float) -> float:
            return 1.0 / (a * H0_si * math.sqrt(float(OmM) / a**3 + float(OmL)))

        t_s, _ = quad(integrand, 1e-5, a_emit, limit=300)
        return float(t_s / C2_GYR_TO_S)

    t_tbl = np.asarray([_t_single(float(z)) for z in z_tbl], dtype=float)
    t_of_z = interp1d(z_tbl, t_tbl, kind="cubic", fill_value="extrapolate")
    return t_of_z, z_tbl, t_tbl


def _c2_build_blend_t_of_z_table(
    tc: float,
    k: float,
    H0_kms: float,
    OmL: float,
    OmM: float,
    t_future_gyr: float,
) -> tuple[np.ndarray, np.ndarray, Any]:
    """Self-consistent blend t(z) from monotonic a(t) (``ccomplet2._build_blend_t_of_z_table``)."""
    t_hi = float(max(40.0, t_future_gyr))
    t_grid = np.linspace(1.0e-4, t_hi, 9000)
    logA = _c2_log10_A_blend_locked(t_grid, tc, k, H0_kms, OmL, OmM, t_future_gyr)
    a_raw = np.clip(np.sqrt(10.0 ** (logA - C2_A0_LOG10)), 1.0e-12, None)
    a_mono = np.maximum.accumulate(a_raw)
    z_desc = np.clip(1.0 / np.maximum(a_mono, 1.0e-18) - 1.0, 0.0, None)
    z_rev = z_desc[::-1]
    t_rev = t_grid[::-1]
    if z_rev.size > 1:
        dz = np.diff(z_rev)
        keep = np.concatenate(([True], dz > 1.0e-9))
    else:
        keep = np.array([True], dtype=bool)
    z_unique = z_rev[keep]
    t_unique = t_rev[keep]
    if z_unique.size < 8:
        z_fallback = np.logspace(-4, np.log10(3.5), 80)
        t_of_z_lcdm, _, _ = _c2_build_tz_lcdm(H0_kms, OmM, OmL)
        t_fallback = np.asarray(t_of_z_lcdm(z_fallback), dtype=float)
        z_unique = np.concatenate(([0.0], z_fallback))
        t_unique = np.concatenate(([t_hi], t_fallback))
    t_of_z = interp1d(
        z_unique,
        t_unique,
        kind="linear",
        bounds_error=False,
        fill_value=(float(t_unique[0]), float(t_unique[-1])),
    )
    return np.asarray(z_unique, dtype=float), np.asarray(t_unique, dtype=float), t_of_z


def _c2_build_blend_h_of_z_table(
    tc: float,
    k: float,
    H0_kms: float,
    OmL: float,
    OmM: float,
    t_future_gyr: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Blend H(z) and comoving distance table (``ccomplet2._build_blend_h_of_z_table``)."""
    t_grid = np.geomspace(1.0e-8, float(T_REF_GYR), 26000)
    t_grid[-1] = float(T_REF_GYR)
    logA_blend_grid = _c2_log10_A_blend_locked(t_grid, tc, k, H0_kms, OmL, OmM, t_future_gyr)
    logA_lcdm_grid = c2_log10_A_lcdm(t_grid, H0_kms, OmL, OmM)
    a_blend_raw = np.sqrt(np.clip(10.0 ** (logA_blend_grid - C2_A0_LOG10), 1.0e-60, None))
    a_lcdm = np.sqrt(np.clip(10.0 ** (logA_lcdm_grid - C2_A0_LOG10), 1.0e-60, None))
    a_lcdm /= max(float(a_lcdm[-1]), 1.0e-30)
    t_lock = float(C2_T_TRANSITION_MIN_GYR)
    a_raw_ref = float(a_blend_raw[-1])
    a_lcdm_ref = float(a_lcdm[-1])
    # Calibrate raw blend scale factor to ΛCDM: log(a_lcdm) ≈ α log(a_raw) + β (same form for both methods).
    if C2_BLEND_CALIB_MODE == "lsq_loga":
        t_hi = float(T_REF_GYR)
        mask = (t_grid >= t_lock) & (t_grid <= t_hi)
        if int(np.count_nonzero(mask)) >= 8:
            lx = np.log(np.maximum(a_blend_raw[mask], 1.0e-60))
            ly = np.log(np.maximum(a_lcdm[mask], 1.0e-60))
            alpha_fit, beta_fit = np.polyfit(lx, ly, 1)
            alpha = float(alpha_fit)
            beta = float(beta_fit)
        else:
            a_raw_lock = float(np.interp(t_lock, t_grid, a_blend_raw))
            a_lcdm_lock = float(np.interp(t_lock, t_grid, a_lcdm))
            x1 = np.log(max(a_raw_lock, 1.0e-60))
            x2 = np.log(max(a_raw_ref, 1.0e-60))
            y1 = np.log(max(a_lcdm_lock, 1.0e-60))
            y2 = np.log(max(a_lcdm_ref, 1.0e-60))
            alpha = (y2 - y1) / max(x2 - x1, 1.0e-30)
            beta = y1 - alpha * x1
    else:
        a_raw_lock = float(np.interp(t_lock, t_grid, a_blend_raw))
        a_lcdm_lock = float(np.interp(t_lock, t_grid, a_lcdm))
        x1 = np.log(max(a_raw_lock, 1.0e-60))
        x2 = np.log(max(a_raw_ref, 1.0e-60))
        y1 = np.log(max(a_lcdm_lock, 1.0e-60))
        y2 = np.log(max(a_lcdm_ref, 1.0e-60))
        alpha = (y2 - y1) / max(x2 - x1, 1.0e-30)
        beta = y1 - alpha * x1
    a_blend_cal = np.exp(alpha * np.log(np.maximum(a_blend_raw, 1.0e-60)) + beta)
    a_grid = np.where(t_grid < t_lock, a_lcdm, a_blend_cal)
    a_grid = np.maximum.accumulate(np.clip(a_grid, 1.0e-18, None))
    a_grid /= max(float(a_grid[-1]), 1.0e-30)
    t_sec = t_grid * C2_GYR_TO_S
    da_dt = np.gradient(a_grid, t_sec, edge_order=2)
    _ = da_dt / np.maximum(a_grid, 1.0e-30) / KMS_TO_SI  # h_area kept for parity with c2 source
    z_desc = np.clip(1.0 / np.maximum(a_grid, 1.0e-18) - 1.0, 0.0, None)
    z_list = tuple(z_desc.tolist())
    h_lcdm_desc = np.asarray(c2_friedmann_H_cmb(z_list, H0_kms, OmM, OmL), dtype=float)
    alpha_e = C2_ALPHA_H_ENHANCE
    z_c = C2_ZC_H_ENHANCE
    enhancement = 1.0 + alpha_e * np.exp(-z_desc / z_c)
    mask_late = z_desc < C2_ZMAX_H_ENHANCE
    h_desc = h_lcdm_desc.copy()
    h_desc[mask_late] = h_lcdm_desc[mask_late] * enhancement[mask_late]
    h_desc = np.nan_to_num(h_desc, nan=H0_kms, posinf=H0_kms, neginf=H0_kms)
    h_desc = np.clip(h_desc, 1.0e-8, None)
    h_desc[-1] = float(H0_kms)
    z_asc = z_desc[::-1]
    h_asc = h_desc[::-1]
    if z_asc.size > 1:
        keep = np.concatenate(([True], np.diff(z_asc) > 1.0e-12))
        z_asc = z_asc[keep]
        h_asc = h_asc[keep]
    inv_h = C_KMS / np.maximum(h_asc, 1.0e-12)
    dm_asc = cumulative_trapezoid(inv_h, z_asc, initial=0.0)
    dm_asc = np.maximum.accumulate(np.nan_to_num(dm_asc, nan=0.0, posinf=0.0, neginf=0.0))
    return np.asarray(z_asc, dtype=float), np.asarray(h_asc, dtype=float), np.asarray(dm_asc, dtype=float)


def c2_H_blend(
    z: np.ndarray,
    tc: float,
    k: float,
    H0_kms: float,
    OmL: float,
    OmM: float,
    t_future_gyr: float = C2_T_FUTURE_DEFAULT_GYR,
) -> np.ndarray:
    z_in = np.asarray(z, dtype=float)
    z_arr = np.clip(np.atleast_1d(z_in), 0.0, None)
    z_grid, h_grid, _ = _c2_build_blend_h_of_z_table(tc, k, H0_kms, OmL, OmM, t_future_gyr)
    h_eval = np.interp(z_arr, z_grid, h_grid)
    z_cap = float(z_grid[-1])
    high_mask = z_arr > z_cap
    if np.any(high_mask):
        h_eval = np.asarray(h_eval, dtype=float).copy()
        h_eval[high_mask] = np.asarray(
            c2_friedmann_H_cmb(tuple(z_arr[high_mask].tolist()), H0_kms, OmM, OmL),
            dtype=float,
        )
    if np.ndim(z_in) == 0:
        return np.asarray([float(h_eval[0])], dtype=float)
    return np.asarray(h_eval, dtype=float)


def c2_comoving_distance_blend_mpc(
    z: np.ndarray,
    tc: float,
    k: float,
    H0_kms: float,
    OmL: float,
    OmM: float,
    t_future_gyr: float = C2_T_FUTURE_DEFAULT_GYR,
) -> np.ndarray:
    z_in = np.asarray(z, dtype=float)
    z_arr = np.clip(np.atleast_1d(z_in), 0.0, None)
    if z_arr.size == 0:
        return np.asarray([], dtype=float)
    z_grid, _, dm_grid = _c2_build_blend_h_of_z_table(tc, k, H0_kms, OmL, OmM, t_future_gyr)
    z_min = float(z_grid[0])
    z_max = float(z_grid[-1])
    z_safe = np.clip(z_arr, z_min + 1.0e-6, z_max - 1.0e-6)
    dm_eval = np.interp(z_safe, z_grid, dm_grid).astype(float)
    z_cap = z_max
    dm_cap = float(dm_grid[-1])
    high_idx = np.where(z_arr > z_cap)[0]
    for idx in high_idx:
        zi = float(z_arr[idx])
        dm_hi, _ = quad(
            lambda zp: C_KMS / max(float(c2_friedmann_H_cmb((float(zp),), H0_kms, OmM, OmL)[0]), 1.0e-12),
            z_cap,
            zi,
            limit=500,
        )
        dm_eval[idx] = dm_cap + dm_hi
    dm_eval = np.nan_to_num(dm_eval, nan=0.0, posinf=1.0e8, neginf=0.0)
    dm_eval = np.clip(dm_eval, 0.0, 1.0e8)
    if np.ndim(z_in) == 0:
        return np.asarray([float(dm_eval[0])], dtype=float)
    return np.asarray(dm_eval, dtype=float)


def c2_mu_blend_from_background(
    z: np.ndarray,
    tc: float,
    k: float,
    H0_kms: float,
    OmL: float,
    OmM: float,
    t_future_gyr: float = C2_T_FUTURE_DEFAULT_GYR,
) -> np.ndarray:
    """Distance modulus from blend luminosity distance (no nuisance M); matches ``ccomplet2.mu_blend``."""
    z_arr = np.asarray(z, dtype=float)
    d_m = np.asarray(c2_comoving_distance_blend_mpc(z_arr, tc, k, H0_kms, OmL, OmM, t_future_gyr), dtype=float)
    d_l = (1.0 + z_arr) * d_m
    return 5.0 * np.log10(np.clip(d_l, 1.0e-12, None) * 1.0e6 / 10.0)


def blend_physics_is_ccomplet2() -> bool:
    return BLEND_PHYSICS_MODE not in ("legacy_residual", "legacy", "phenomenological", "residual")


def h_si(h0_kmsmpc: float) -> float:
    return float(h0_kmsmpc) / MPC_KM


def Ez(z: np.ndarray, Omega_m: float, Omega_L: float, Omega_r: float) -> np.ndarray:
    zp = np.asarray(z, dtype=float) + 1.0
    x = Omega_m * zp**3 + Omega_r * zp**4 + Omega_L
    return np.sqrt(np.clip(x, eps(), None))


class CosmoInterpEngine:
    """Caches comoving distance and cosmic-time splines for fixed (``H_0``, ``\\Omega_m``).

    Eliminates rebuilding identical PCHIP objects on every likelihood call once parameters
    are held fixed across a walker step; grids cover all survey redshifts supplied at construction.

    When :func:`blend_physics_is_ccomplet2` is true, also caches the **blend-consistent**
    ``(z, H(z), D_M(z), t(z))`` tables built with the same numerics as ``ccomplet2.py`` (not Streamlit-cached).
    """

    __slots__ = (
        "zp_hi_dm",
        "z_max_age",
        "_key",
        "_lnz_lo",
        "_lnz_hi",
        "_spl_DM",
        "_spl_age",
        "_b2_key",
        "_b2_z",
        "_b2_h",
        "_b2_dm",
        "_b2_tz",
        "_b2_tt",
        "_b2_ol",
    )

    def __init__(self, zp_hi_dm: float, z_max_age: float) -> None:
        self.zp_hi_dm = float(zp_hi_dm)
        self.z_max_age = float(z_max_age)
        self._key: tuple[float, float] | None = None
        self._lnz_lo = float("nan")
        self._lnz_hi = float("nan")
        self._spl_DM: PchipInterpolator | None = None
        self._spl_age: PchipInterpolator | None = None
        self._b2_key: tuple[float, float, float, float, float] | None = None
        self._b2_z: np.ndarray | None = None
        self._b2_h: np.ndarray | None = None
        self._b2_dm: np.ndarray | None = None
        self._b2_tz: np.ndarray | None = None
        self._b2_tt: np.ndarray | None = None
        self._b2_ol: float = float("nan")

    def refresh(self, h0: float, Omega_m: float) -> None:
        key = (float(h0), float(Omega_m))
        if self._key == key:
            return
        self._key = key
        self._b2_key = None
        h0_ = float(h0)
        om_ = float(Omega_m)
        OL = olambda_flat(om_, h0_)
        Or = float(omega_r(h0_))

        zp = np.geomspace(ZP_LO, self.zp_hi_dm, Z_NODES)
        lnz = np.log(zp)
        Xi_row = zp / Ez(zp - 1.0, om_, OL, Or)
        Xi_dm = cumulative_trapezoid(Xi_row, x=lnz, initial=0.0)
        self._lnz_lo = float(lnz[0])
        self._lnz_hi = float(lnz[-1])
        self._spl_DM = PchipInterpolator(lnz, (float(C_KMS) / h0_) * Xi_dm)

        zp_top = float(np.clip((self.z_max_age + 1.0) * 4096.0, 5000.0, 9.999e9))
        zp_grid = np.geomspace(ZP_LO, zp_top, Z_NODES)
        zm = np.sqrt(zp_grid[:-1] * zp_grid[1:])
        ln = np.log(zp_grid)
        dlg = np.diff(ln)
        E_mid = Ez(zm - 1.0, om_, OL, Or)
        hs = np.clip(E_mid * h_si(h0_), eps(), None)
        dg_seg = dlg / hs / float(GYR_S)
        cum_rev = np.flip(np.cumsum(np.flip(np.maximum(dg_seg, 1e-30))))
        ages = np.zeros_like(zp_grid, dtype=float)
        ages[:-1] = cum_rev
        ages[-1] = 0.0
        zs_ax = zp_grid - 1.0
        self._spl_age = PchipInterpolator(zs_ax, ages)

    def _ensure_blend_cache(self, h0: float, Omega_m: float, tcrit: float, k_slope: float) -> None:
        """Build / reuse ccomplet2-style blend tables for (H0, Ωm, t_crit, k)."""
        h0_ = float(h0)
        om_ = float(Omega_m)
        ol_ = float(olambda_flat(om_, h0_))
        key2 = (h0_, om_, float(tcrit), float(k_slope), ol_)
        if self._b2_key == key2 and self._b2_z is not None:
            return
        self.refresh(h0_, om_)
        z_asc, h_asc, dm_asc = _c2_build_blend_h_of_z_table(
            float(tcrit), float(k_slope), h0_, ol_, om_, C2_T_FUTURE_DEFAULT_GYR
        )
        tz, tt, _ = _c2_build_blend_t_of_z_table(
            float(tcrit), float(k_slope), h0_, ol_, om_, C2_T_FUTURE_DEFAULT_GYR
        )
        self._b2_key = key2
        self._b2_z = z_asc
        self._b2_h = h_asc
        self._b2_dm = dm_asc
        self._b2_tz = tz
        self._b2_tt = tt
        self._b2_ol = ol_

    def blend_Hz(self, z: np.ndarray, h0: float, Omega_m: float, tcrit: float, k_slope: float) -> np.ndarray:
        self._ensure_blend_cache(h0, Omega_m, tcrit, k_slope)
        assert self._b2_z is not None and self._b2_h is not None
        zv = np.atleast_1d(np.maximum(np.asarray(z, dtype=float), 0.0))
        z_cap = float(self._b2_z[-1])
        out = np.empty_like(zv, dtype=float)
        m = zv <= z_cap
        if np.any(m):
            out[m] = np.interp(
                zv[m],
                self._b2_z,
                self._b2_h,
                left=float(self._b2_h[0]),
                right=float(self._b2_h[-1]),
            )
        if np.any(~m):
            ol = float(self._b2_ol)
            out[~m] = np.asarray(
                c2_friedmann_H_cmb(tuple(zv[~m].tolist()), float(h0), float(Omega_m), ol),
                dtype=float,
            )
        return out

    def blend_comoving_distance_mpc(
        self, z: np.ndarray, h0: float, Omega_m: float, tcrit: float, k_slope: float
    ) -> np.ndarray:
        self._ensure_blend_cache(h0, Omega_m, tcrit, k_slope)
        assert self._b2_z is not None and self._b2_dm is not None
        z_in = np.asarray(z, dtype=float)
        zv = np.atleast_1d(np.maximum(z_in, 0.0))
        z_min = float(self._b2_z[0])
        z_max = float(self._b2_z[-1])
        z_safe = np.clip(zv, z_min + 1.0e-6, z_max - 1.0e-6)
        dm_eval = np.interp(z_safe, self._b2_z, self._b2_dm).astype(float)
        z_cap = z_max
        dm_cap = float(self._b2_dm[-1])
        ol = float(self._b2_ol)
        high_idx = np.where(zv > z_cap)[0]
        for idx in high_idx:
            zi = float(zv[idx])
            dm_hi, _ = quad(
                lambda zp, h0_=float(h0), om_=float(Omega_m): C_KMS
                / max(float(c2_friedmann_H_cmb((float(zp),), h0_, om_, ol)[0]), 1.0e-12),
                z_cap,
                zi,
                limit=500,
            )
            dm_eval[idx] = dm_cap + dm_hi
        dm_eval = np.nan_to_num(dm_eval, nan=0.0, posinf=1.0e8, neginf=0.0)
        dm_eval = np.clip(dm_eval, 0.0, 1.0e8)
        return dm_eval

    def blend_cosmic_age_gyr(self, z: np.ndarray, h0: float, Omega_m: float, tcrit: float, k_slope: float) -> np.ndarray:
        self._ensure_blend_cache(h0, Omega_m, tcrit, k_slope)
        assert self._b2_tz is not None and self._b2_tt is not None
        zv = np.atleast_1d(np.maximum(np.asarray(z, dtype=float), 0.0))
        z_lo = float(self._b2_tz[0])
        z_hi = float(self._b2_tz[-1])
        zc = np.clip(zv, z_lo, z_hi)
        return np.maximum(np.interp(zc, self._b2_tz, self._b2_tt), 0.0)

    def comoving_distance_mpc(self, z: np.ndarray, h0: float, Omega_m: float) -> np.ndarray:
        self.refresh(h0, Omega_m)
        zv = np.atleast_1d(np.maximum(np.asarray(z, dtype=float), 0.0))
        assert self._spl_DM is not None
        lnz_t = np.log(np.maximum(zv + 1.0, ZP_LO))
        lnz_t = np.clip(lnz_t, self._lnz_lo, self._lnz_hi)
        return np.asarray(self._spl_DM(lnz_t), dtype=float)

    def cosmic_age_gyr(self, z: np.ndarray, h0: float, Omega_m: float) -> np.ndarray:
        self.refresh(h0, Omega_m)
        zv = np.atleast_1d(np.maximum(np.asarray(z, dtype=float), 0.0))
        assert self._spl_age is not None
        return np.maximum(np.asarray(self._spl_age(np.maximum(zv, 0.0)), dtype=float), 0.0)


def dM_mpc(z: np.ndarray, h0: float, Omega_m: float, eng: CosmoInterpEngine) -> np.ndarray:
    return eng.comoving_distance_mpc(z, float(h0), float(Omega_m))


def Hz(z: np.ndarray, h0: float, Omega_m: float) -> np.ndarray:
    OL = olambda_flat(float(Omega_m), float(h0))
    Or = float(omega_r(float(h0)))
    return float(h0) * Ez(np.asarray(z, dtype=float), float(Omega_m), OL, Or)


def q_deceleration_flat_lcdm(z: np.ndarray, h0: float, Omega_m: float) -> np.ndarray:
    r"""Deceleration parameter :math:`q(z)=-\ddot a a/\dot a^2` for flat matter+\Lambda+radiation FRW.

    Uses :math:`q = \bigl(\tfrac12\Omega_m(1+z)^3 + \Omega_r(1+z)^4 - \Omega_\Lambda\bigr)/E^2(z)` with the same
    :math:`E(z)` as :func:`Hz` (so :math:`H>0` implies :math:`E^2>0` on the physical branch).
    """
    zv = np.maximum(np.asarray(z, dtype=float), 0.0)
    zp1 = 1.0 + zv
    OL = float(olambda_flat(float(Omega_m), float(h0)))
    Or = float(omega_r(float(h0)))
    Om = float(Omega_m)
    num = 0.5 * Om * zp1**3 + Or * zp1**4 - OL
    den = Om * zp1**3 + Or * zp1**4 + OL
    den = np.maximum(np.asarray(den, dtype=float), eps())
    return np.asarray(num / den, dtype=float)


def plot_q_deceleration_lcdm(figpath: Path, h0: float, Omega_m: float) -> None:
    """Publication-style :math:`q(z)` for the **flat LCDM+radiation** FRW background (comparison / LCDM chain only)."""
    figpath.parent.mkdir(parents=True, exist_ok=True)
    zv = np.linspace(0.0, 2.6, 160, dtype=float)
    qv = q_deceleration_flat_lcdm(zv, float(h0), float(Omega_m))
    plt.figure(figsize=(6.8, 4.2))
    plt.axhline(0.0, color="k", lw=0.9, ls="--", alpha=0.55, label=r"$q=0$ (inflection)")
    plt.plot(zv, qv, lw=2.0, color="C0", label=r"$q(z)$ flat $\Lambda$CDM ($H_0,\Omega_m$)")
    plt.xlabel(r"redshift $z$")
    plt.ylabel(r"deceleration $q(z)$")
    sub = (
        "Blend distances use ccomplet2-style H_blend(z) (not this q(z))."
        if blend_physics_is_ccomplet2()
        else r"Legacy blend $\mu$ added a residual on this LCDM $d_L(z)$."
    )
    plt.title("Deceleration history (Friedmann LCDM)\n" + sub)
    plt.grid(alpha=0.28)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(figpath, dpi=180)
    plt.close()


def blend_chain_parameter_correlations(flat_b: np.ndarray, colnames: Sequence[str]) -> pd.DataFrame:
    """Full Pearson correlation matrix of leading blend cosmology parameters (identifiability / degeneracy)."""
    nb = int(min(4, int(flat_b.shape[1]), len(tuple(colnames))))
    if nb < 2:
        return pd.DataFrame(dict(status=["insufficient_columns"], n_cols=[int(flat_b.shape[1])]))
    X = np.asarray(flat_b[:, :nb], dtype=float)
    m = np.all(np.isfinite(X), axis=1)
    X = X[m]
    if int(X.shape[0]) < 30:
        return pd.DataFrame(dict(status=["insufficient_clean_rows"], n_rows=[int(X.shape[0])]))
    C = np.corrcoef(X.T)
    labs = [str(colnames[j]) for j in range(nb)]
    return pd.DataFrame(C, index=labs, columns=labs)


def finite_diff_blend_mu_sensitivity_rows(
    theta_med: np.ndarray,
    z_ref: float,
    eng: CosmoInterpEngine,
    rel_step: float = 8e-4,
) -> pd.DataFrame:
    r"""Crude local sensitivities :math:`\partial \mu/\partial\theta_j` at ``z_ref`` (finite differences on the blend path)."""
    th0 = np.asarray(theta_med, dtype=float).reshape(-1)
    if th0.size < 5:
        return pd.DataFrame()
    z1 = np.asarray([float(z_ref)], dtype=float)

    def mu_at(th: np.ndarray) -> float:
        mb, _ = mu_blend(z1, float(th[0]), float(th[1]), float(th[2]), float(th[3]), float(th[4]), eng)
        return float(np.asarray(mb, dtype=float).reshape(-1)[0])

    m0 = mu_at(th0)
    rows: list[dict[str, float | str]] = []
    labels = ("H0", "Omega_m", "t_crit", "k_slope", "M")
    for j in range(5):
        h = float(rel_step) * max(1.0, abs(float(th0[j])))
        thp = th0.copy()
        thp[j] = float(th0[j]) + h
        mp = mu_at(thp)
        der = (mp - m0) / max(h, eps())
        rows.append(dict(parameter=labels[j], z_ref=float(z_ref), mu_base=m0, step=h, dmu_dparam=float(der)))
    return pd.DataFrame(rows)


def write_paper_outline_md(path: Path) -> None:
    """Suggested paper / ISEF narrative structure (honest scope; not auto-generated science text)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "# Suggested manuscript / fair-report outline",
                "",
                "## 1. Model definition",
                "- Flat FRW LCDM baseline with standard radiation fraction.",
                "- Horizon-area blend as an explicit **phenomenological** modulus residual on top of LCDM distances (not a modified Einstein equation in this file).",
                "- Equations implemented: branch areas, logistic, `Delta_mu(z)` relative to LCDM.",
                "",
                "## 2. Data",
                "- Pantheon+ SH0ES training; DES Dovekie hold-out (M profiled only on training set).",
                "- Optional compressed BAO + Planck-style CMB Gaussian blocks (see `summary.json` disclosure — not Plik).",
                "- SPARC/MOND only as a **separate** galaxy-scale benchmark if `CWSF_SPARC_DIR` is set.",
                "",
                "## 3. Methods",
                "- emcee affine MCMC; optional full Pantheon covariance; optional dynesty nested sampling for log-evidence.",
                "- Joint posterior when external blocks enabled: `log_post = log_prior + logL_SN + logL_BAO + logL_CMB`.",
                "",
                "## 4. Results",
                "- Posterior tables, corners (if `corner` installed), PPCs, expansion-history diagnostics, `q(z)` figure.",
                "- Cross-probe `cross_probe_metrics.csv` (read footnotes: SN χ² and MVN blocks are not directly additive χ²).",
                "",
                "## 5. Model comparison",
                "- AIC/BIC/WAIC on SN path; Bayes factor from dynesty when run; interpret together with robustness CSV.",
                "",
                "## 6. Physical interpretation + limitations",
                "- State clearly: compressed CMB/BAO here are **pedagogical** moment-matched summaries.",
                "- Discuss systematics file bundled with outputs.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def write_systematics_defensibility_md(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "# Systematics, scope, and what would be required for publication-grade cosmology",
                "",
                "## Supernovae (training + hold-out)",
                "- Standard-candle standardisation (stretch, colour, host-galaxy correlations) is only partially represented when using diagonal `MU_SH0ES_ERR_DIAG` or a STATONLY covariance subset.",
                "- Selection effects, Malmquist bias, and population drift across redshift are not self-consistently modeled here.",
                "- **Mitigation already in-repo:** optional `CWSF_USE_COV=1`, redshift-cut robustness scenarios, calibration-offset stress scenario.",
                "",
                "## BAO (embedded Gaussian)",
                "- Uses Eisenstein–Hu style drag scale and closed-form distances — not the official SDSS/eBOSS likelihood pipelines.",
                "- Replace with collaboration likelihood codes when claiming parameter inference at DR16/BOSS precision.",
                "",
                "## CMB (embedded compressed Gaussian)",
                "- Not Planck Plik / Commander / CamSpec; a low-dimensional MVN meant to illustrate how CMB shifts a joint posterior when combined with SN.",
                "- Angular diameter distances at recombination use the same background integrator as SN — tighten against CAMB/CLASS before claiming sub-percent CMB consistency.",
                "",
                "## Blend path",
                "- Background `H(z)` and distances for *kinematic* checks follow LCDM; the blend modifies `mu(z)` through the horizon-area construction — defend as phenomenology unless you publish a sourced modification to the metric.",
                "",
                "## Evidence",
                "- dynesty evidences are only as defensible as the likelihood implementation and priors; compare models only on identical data vectors and covariance choices.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def anchor_luminosity_distance_lowz_hubbles(
    dL_mpc: np.ndarray,
    h0_kms: float,
    Omega_m: float,
    eng: CosmoInterpEngine,
) -> np.ndarray:
    """Uniform scaling on luminosity distances so tiny-$z$ matches $d_L \\approx c/H_0$ (stabilizes splines/interpolation edges).

    Computes one scalar correction using the same cached comoving engine as the distances themselves; applies to **all**
    supplied $z$ identically so the ladder matches the asymptotic Hubbles limit without distorting $\\Delta z$ dependence.
    """
    out = np.atleast_1d(np.asarray(dL_mpc, dtype=float)).copy()
    zs = 1e-4
    dM_s_raw = np.asarray(dM_mpc(np.array([zs], dtype=float), float(h0_kms), float(Omega_m), eng), dtype=float).reshape(-1)
    dM_s = float(np.maximum(np.nan_to_num(dM_s_raw[0], nan=0.0), eps()))
    dL_s = float((1.0 + zs) * dM_s)
    dL_exp = float(float(C_KMS) * zs / max(float(h0_kms), eps()))
    corr = float(dL_exp / max(dL_s, eps()))
    out *= corr
    return out


def mu_lcdm_shape(
    z: np.ndarray, h0: float, Omega_m: float, eng: CosmoInterpEngine
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """LCDM distance modulus **shape** (no absolute magnitude offset)."""
    dM = dM_mpc(z, float(h0), float(Omega_m), eng)
    zp1 = np.asarray(z, dtype=float) + 1.0
    dL = zp1 * dM
    if USE_LOWZ_DL_ANCHOR:
        dL = anchor_luminosity_distance_lowz_hubbles(dL, float(h0), float(Omega_m), eng)
    mu = 5.0 * np.log10(np.clip(np.asarray(dL, dtype=float), eps(), None)) + 25.0
    return np.asarray(mu, float), np.asarray(dL, float), np.asarray(dM, float)


def diagnostic_hz_splinechi_vs_friedmann(
    z_grid: np.ndarray,
    h0: float,
    Omega_m: float,
    eng: CosmoInterpEngine,
    figpath: Path,
) -> None:
    """$H(z)$ reconstructed from $\\mathrm{d}\\chi/dz$ vs analytic $H_0 E(z)$ (LCDM kinematic self-consistency).

    Judges the distance engine independent of horizon-area blend bookkeeping: ratio should hug unity when splines/grid are sane.
    """
    z = np.asarray(z_grid, dtype=float).reshape(-1)
    chi = np.asarray(dM_mpc(z, float(h0), float(Omega_m), eng), dtype=float)
    dz = np.maximum(np.gradient(z), eps())
    dchi_dz = np.gradient(chi, z, edge_order=2)
    hz_geom = np.asarray(float(C_KMS) / np.maximum((1.0 + z) * np.maximum(dchi_dz, eps()), eps()), dtype=float)
    hz_alg = np.asarray(Hz(z, float(h0), float(Omega_m)), dtype=float)
    ok = np.isfinite(hz_geom) & np.isfinite(hz_alg) & (hz_alg > eps())
    rat = hz_geom.copy()
    rat[ok] = hz_geom[ok] / hz_alg[ok]
    rat[~ok] = np.nan
    fin = rat[np.isfinite(rat)]
    plt.figure(figsize=(7.2, 4.2))
    plt.plot(z, rat, lw=2.2, label=r"$H_{\!\chi'}/H_{\mathrm{Fried}}$")
    plt.axhline(1.0, color="k", ls="--", lw=1.1)
    if fin.size > 8:
        lo_r, hi_r = float(np.quantile(fin, 0.02)), float(np.quantile(fin, 0.98))
        pad = max(0.05, 0.15 * (hi_r - lo_r))
        plt.ylim(max(0.55, lo_r - pad), hi_r + pad)
    plt.xlabel("z")
    plt.ylabel(r"$H_{\mathrm{from\,}d\chi/dz}(z)\,/\,H_0 E(z)$")
    plt.title("LCDM H(z) kinematic sanity (spline $\\chi(z)$ slope vs Friedman)")
    plt.grid(alpha=0.28)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(figpath, dpi=160)
    plt.close()


def mu_lcdm(
    z: np.ndarray, h0: float, Omega_m: float, Mmag: float, eng: CosmoInterpEngine
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu_s, dL, dM = mu_lcdm_shape(z, float(h0), float(Omega_m), eng)
    mu = mu_s + float(Mmag)
    return np.asarray(mu, float), np.asarray(dL, float), np.asarray(dM, float)


def lcdm_sinhc_chi(z: np.ndarray, h0: float, Omega_m: float) -> np.ndarray:
    Or = float(omega_r(float(h0)))
    norm = float(np.clip(1.0 - Or, eps(), None))
    Omh = float(Omega_m) / norm
    OLh = olambda_flat(float(Omega_m), float(h0)) / norm
    zp = np.maximum(np.asarray(z, dtype=float) + 1.0, 1.0)
    ff = math.sqrt(max(float(OLh / Omh), eps()))
    cz = float(C_KMS / float(h0))
    chi = cz / math.sqrt(max(float(OLh), eps())) * (np.arcsinh(ff * zp ** 1.5) - np.arcsinh(ff))
    return np.clip(np.maximum(np.asarray(chi, dtype=float), 0.0), eps(), None)


# =============================================================================
# Embedded optional BAO / compressed CMB / SPARC–MOND (single-file; no imports)
# =============================================================================

_TEMPLATE_BAO_DICT: dict[str, Any] = {
    "meta": {
        "source": "Pedagogical DR16-class LRG BAO ratios at z_eff≈0.698 (moment-matched scales; NOT the official SDSS likelihood).",
        "reference_primary": "BOSS/eBOSS DR16 BAO cosmology (Alam et al.; SDSS collaboration)",
        "reference_urls": [
            "https://www.sdss.org/",
            "https://svn.sdss.org/public/data/eboss/DR16cosmo/tags/v1_0_0/likelihoods/BAO-only/",
        ],
        "note": "Replace with collaboration likelihood + Boltzmann r_d(z) for publication inference; r_d here uses the EH-style compact fit tied to the same FRW driver as SN.",
    },
    "z_eff": [0.698, 0.698],
    "labels": ["DM_over_rd", "DH_over_rd"],
    "obs_vector": [17.857, 19.774],
    "cov": [[0.088, 0.038], [0.038, 0.198]],
}

# Planck 2018-style compressed means (TT,TE,EE+lowE class) in (100*theta_star, Omega_b h^2); NOT Plik.
_s100 = 0.00031
_sw = 0.00015
_rho_cmb = 0.28
_TEMPLATE_CMB_DICT: dict[str, Any] = {
    "meta": {
        "likelihood_type": "compressed_cmb_shift_gaussian",
        "disclosure": "2D Gaussian in (100*theta_star, Omega_b h^2) centered near Planck 2018 baseline table values — illustrative compressed likelihood, not Plik/CamSpec.",
        "reference_primary": "Planck Collaboration 2018, VI — Cosmological parameters (arXiv:1807.06209)",
        "reference_urls": [
            "https://www.cosmos.esa.int/web/planck/pla",
            "https://irsa.ipac.caltech.edu/data/Planck/release_3/ancillary-data/",
        ],
        "labels": ["100_theta_star", "omega_b"],
    },
    "obs_vector": [1.04109, 0.02233],
    "cov": [
        [_s100**2, _rho_cmb * _s100 * _sw],
        [_rho_cmb * _s100 * _sw, _sw**2],
    ],
}

def copy_default_likelihood_templates(outdir: Path) -> None:
    """Optional JSON export for human inspection only.

    Default BAO/CMB blocks are built from in-memory ``_TEMPLATE_*_DICT`` structures (no disk read required).
    Set ``CWSF_WRITE_LIKELIHOOD_TEMPLATE_JSON=1`` to write ``bao_likelihood_data.json`` and
    ``cmb_shift_likelihood_data.json`` under ``outdir`` when missing.
    """
    if os.getenv("CWSF_WRITE_LIKELIHOOD_TEMPLATE_JSON", "0") != "1":
        return
    specs = (
        ("bao_likelihood_data.json", _TEMPLATE_BAO_DICT),
        ("cmb_shift_likelihood_data.json", _TEMPLATE_CMB_DICT),
    )
    for name, d in specs:
        dst = outdir / name
        if not dst.exists():
            dst.write_text(json.dumps(d, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def z_drag_eisenstein_hu(obh2: float, odmh2: float) -> float:
    """Baryon-drag redshift fit (Eisenstein & Hu 1998, widely used in BAO codes)."""
    odmh2 = max(float(odmh2), 1e-8)
    obh2 = max(float(obh2), 1e-8)
    omh2 = float(obh2 + odmh2)
    b1 = 0.313 * odmh2 ** (-0.419) * (1.0 + 0.607 * odmh2**0.674)
    b2 = 0.238 * obh2**0.277
    zd = 1291.0 * omh2**0.251 / (1.0 + 0.659 * omh2**0.828) * (1.0 + b1 * obh2**b2)
    return float(np.clip(zd, 50.0, 5000.0))


def sound_speed_cm_s(obh2: float, T_cmb: float = 2.7255) -> float:
    """Photon–baryon ratio helper (Ω_γ h²) used in R(z) (notation follows common BAO notes)."""
    _ = obh2
    return float(2.469e-5 * (T_cmb / 2.726) ** 4)


def R_baryon_photon(z: np.ndarray, obh2: float, T_cmb: float = 2.7255) -> np.ndarray:
    ogh2 = sound_speed_cm_s(obh2, T_cmb)
    zz = np.maximum(np.asarray(z, dtype=float), 0.0)
    return 0.75 * float(obh2) / np.maximum(ogh2 * (1.0 + zz), eps())


def rs_integral_mpc(h0: float, Omega_m: float, obh2: float, z_lo: float, z_hi: float, n: int = 4096) -> float:
    r"""Comoving sound horizon :math:`\int_{z_{\mathrm{lo}}}^{z_{\mathrm{hi}}} \frac{c_s}{(1+z)H}\,dz` (diagnostic; not used by default r_d fit)."""
    z_lo = float(max(z_lo, 1e-6))
    z_hi = float(max(z_hi, z_lo + 1e-3))
    u0 = math.log(1.0 + z_lo)
    u1 = math.log(1.0 + z_hi)
    uu = np.linspace(u0, u1, int(n))
    zp1 = np.exp(uu)
    z = zp1 - 1.0
    Rbz = R_baryon_photon(z, obh2)
    cs = C_KMS / np.sqrt(np.maximum(3.0 * (1.0 + Rbz), eps()))
    hz = np.maximum(Hz(z, h0, Omega_m), eps())
    integrand = cs / hz
    return float(np.trapezoid(integrand, uu))


def r_drag_Mpc_eh98(h0: float, Omega_m: float, obh2: float) -> float:
    """Drag-epoch sound horizon r_d (Mpc); compact EH-style recalibration (pedagogical BAO ratios)."""
    hh = (float(h0) / 100.0) ** 2
    omh2 = max(float(Omega_m) * hh, 1e-8)
    odmh2 = max(omh2 - float(obh2), 1e-8)
    _ = z_drag_eisenstein_hu(obh2, odmh2)
    obf = float(obh2) / 0.02237
    omf = omh2 / 0.1424
    hf = float(h0) / 67.4
    rd = 147.09 * (obf**0.23) * (omf**-0.11) * (hf**-0.02)
    return float(np.clip(rd, 80.0, 180.0))


def _external_joint_uses_blend_geometry() -> bool:
    th = globals().get("_CWSF_EXT_THETA")
    return bool(globals().get("_CWSF_LNPOST_IS_BLEND", False)) and blend_physics_is_ccomplet2() and th is not None and int(th.size) >= 4


def D_H_over_rd(h0: float, Omega_m: float, obh2: float, z: float, eng: CosmoInterpEngine) -> float:
    eng.refresh(h0, Omega_m)
    rd = max(r_drag_Mpc_eh98(h0, Omega_m, obh2), eps())
    th = globals().get("_CWSF_EXT_THETA")
    if _external_joint_uses_blend_geometry() and th is not None:
        Ol = olambda_flat(float(Omega_m), float(h0))
        hz = float(eng.blend_Hz(np.array([float(z)], dtype=float), float(h0), float(Omega_m), float(th[2]), float(th[3]))[0])
    else:
        hz = float(Hz(np.array([float(z)], dtype=float), h0, Omega_m)[0])
    dh = float(C_KMS) / max(hz, eps())
    return float(dh / rd)


def D_M_over_rd(h0: float, Omega_m: float, obh2: float, z: float, eng: CosmoInterpEngine) -> float:
    """Comoving D_M / r_d: LCDM closed-form χ, or blend-consistent D_M when sampling the blend posterior."""
    rd = max(r_drag_Mpc_eh98(h0, Omega_m, obh2), eps())
    th = globals().get("_CWSF_EXT_THETA")
    if _external_joint_uses_blend_geometry() and th is not None:
        dm = float(eng.blend_comoving_distance_mpc(np.array([float(z)], dtype=float), float(h0), float(Omega_m), float(th[2]), float(th[3]))[0])
    else:
        dm = float(np.asarray(lcdm_sinhc_chi(np.array([float(z)], dtype=float), h0, Omega_m), dtype=float).reshape(-1)[0])
    if not math.isfinite(dm) or dm <= 0.0:
        return float("nan")
    return float(dm / rd)


@dataclass(frozen=True)
class BaoLikelihoodData:
    z_eff: np.ndarray
    y_order: tuple[str, ...]
    y_obs: np.ndarray
    chol: np.ndarray
    logdet_2pi: float
    meta: dict[str, Any]


def baolike_from_raw_dict(raw: Mapping[str, Any]) -> BaoLikelihoodData:
    """Build BAO Gaussian likelihood data from an in-memory mapping (same schema as optional JSON on disk)."""
    z_eff = np.asarray(raw["z_eff"], dtype=float).reshape(-1)
    y_obs = np.asarray(raw["obs_vector"], dtype=float).reshape(-1)
    y_order = tuple(str(x) for x in raw["labels"])
    cov = np.asarray(raw["cov"], dtype=float)
    if cov.shape[0] != cov.shape[1] or cov.shape[0] != y_obs.size:
        raise ValueError("BAO likelihood dict: cov must be square and match obs_vector length")
    sign, ldet = np.linalg.slogdet(cov)
    if sign <= 0:
        raise ValueError("BAO covariance not positive definite")
    chol = la_sci.cholesky(cov, lower=True)
    n = int(y_obs.size)
    logdet_2pi = float(n * math.log(2.0 * math.pi) + ldet)
    return BaoLikelihoodData(
        z_eff=z_eff,
        y_order=y_order,
        y_obs=y_obs,
        chol=chol,
        logdet_2pi=logdet_2pi,
        meta=dict(raw.get("meta", {})),
    )


def load_bao_json(path: Path) -> BaoLikelihoodData:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return baolike_from_raw_dict(raw)


def lnlike_bao(h0: float, Omega_m: float, obh2: float, eng: CosmoInterpEngine, bao: BaoLikelihoodData | None) -> float:
    if bao is None:
        return 0.0
    zmx = float(np.max(bao.z_eff)) if bao.z_eff.size else 0.05
    th = globals().get("_CWSF_EXT_THETA")
    if _external_joint_uses_blend_geometry() and th is not None:
        if not validate_blend_background(h0, Omega_m, float(th[2]), float(th[3]), eng, zmx):
            return float(-np.inf)
        if not validate_lcdm_background(h0, Omega_m, obh2, eng, zmx):
            return float(-np.inf)
    elif not validate_lcdm_background(h0, Omega_m, obh2, eng, zmx):
        return float(-np.inf)
    preds: list[float] = []
    for zi, tag in zip(bao.z_eff.tolist(), bao.y_order):
        if tag.upper() in ("DM_OVER_RD", "DM/RD"):
            preds.append(D_M_over_rd(h0, Omega_m, obh2, float(zi), eng))
        elif tag.upper() in ("DH_OVER_RD", "DH/RD", "HUBBLE_OVER_RD"):
            preds.append(D_H_over_rd(h0, Omega_m, obh2, float(zi), eng))
        else:
            raise ValueError(f"unsupported BAO label {tag!r}")
    y_model = np.asarray(preds, dtype=float)
    if np.any(~np.isfinite(y_model)):
        return float(-np.inf)
    r = y_model - bao.y_obs
    y = la_sci.solve_triangular(bao.chol, r.reshape(-1, 1), lower=True).ravel()
    return float(-0.5 * float(np.dot(y, y) + bao.logdet_2pi))


def z_star_hu_sugiyama(obh2: float, ocdmh2: float) -> float:
    """Hu & Sugiyama (1996) family fit for photon decoupling redshift."""
    obh2 = max(float(obh2), 1e-6)
    ocdmh2 = max(float(ocdmh2), 1e-6)
    ob1 = obh2 * 100.0
    omh2 = obh2 + ocdmh2
    zs = (
        1048.0
        * (1.0 + 0.00124 * ob1 ** (-0.738))
        / (1.0 + (0.078 * ob1 ** (-0.238)) / (1.0 + 39.5 * ob1**0.763))
    )
    zs *= (omh2 * 100.0) ** (0.019 / (1.0 + 0.659 * (omh2 * 100.0) ** 0.828))
    return float(np.clip(zs, 700.0, 2000.0))


def rs_star_Mpc(h0: float, Omega_m: float, obh2: float, eng: CosmoInterpEngine) -> float:
    """Approximate r_s(z_*); uses f_s* r_d with f_s*≈1.018 for numerical stability."""
    _ = eng
    return float(r_drag_Mpc_eh98(h0, Omega_m, obh2) * 1.018)


@dataclass(frozen=True)
class CmbShiftData:
    y_obs: np.ndarray
    chol: np.ndarray
    logdet_2pi: float
    labels: tuple[str, ...]
    meta: dict[str, Any]


def cmbshift_from_raw_dict(raw: Mapping[str, Any]) -> CmbShiftData:
    """Build compressed CMB Gaussian likelihood data from an in-memory mapping (same schema as optional JSON on disk)."""
    y_obs = np.asarray(raw["obs_vector"], dtype=float).reshape(-1)
    cov = np.asarray(raw["cov"], dtype=float)
    if cov.shape[0] != cov.shape[1] or cov.shape[0] != y_obs.size:
        raise ValueError("CMB likelihood dict: cov must match obs_vector")
    sign, ldet = np.linalg.slogdet(cov)
    if sign <= 0:
        raise ValueError("CMB covariance not positive definite")
    chol = la_sci.cholesky(cov, lower=True)
    n = int(y_obs.size)
    logdet_2pi = float(n * math.log(2.0 * math.pi) + ldet)
    meta = dict(raw.get("meta", {}))
    labs = meta.get("labels")
    if labs is None:
        labs = ["R", "omega_b"] if n == 2 else ["100_theta_star", "R", "omega_b"]
    labels = tuple(str(x) for x in labs)
    if len(labels) != n:
        raise ValueError("CMB likelihood dict: len(labels) must match obs_vector length")
    return CmbShiftData(y_obs=y_obs, chol=chol, logdet_2pi=logdet_2pi, labels=labels, meta=meta)


def load_cmb_shift_json(path: Path) -> CmbShiftData:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return cmbshift_from_raw_dict(raw)


def predict_cmb_observables(
    h0: float, Omega_m: float, obh2: float, eng: CosmoInterpEngine, labels: tuple[str, ...]
) -> np.ndarray:
    hh = (float(h0) / 100.0) ** 2
    ocdmh2 = max(float(Omega_m) * hh - float(obh2), 1e-8)
    zs = z_star_hu_sugiyama(obh2, ocdmh2)
    th = globals().get("_CWSF_EXT_THETA")
    if _external_joint_uses_blend_geometry() and th is not None:
        dm = float(eng.blend_comoving_distance_mpc(np.array([float(zs)], dtype=float), float(h0), float(Omega_m), float(th[2]), float(th[3]))[0])
    else:
        dm = float(np.asarray(lcdm_sinhc_chi(np.array([float(zs)], dtype=float), h0, Omega_m), dtype=float).reshape(-1)[0])
    rs = max(rs_star_Mpc(h0, Omega_m, obh2, eng), eps())
    zp1 = 1.0 + float(zs)
    d_a = dm / max(zp1, eps())
    Rshift = math.sqrt(max(float(Omega_m), eps())) * float(h0) * dm / max(float(C_KMS), eps())
    out: list[float] = []
    for lab in labels:
        k = lab.strip().upper()
        if k in ("R",):
            out.append(float(Rshift))
        elif k in ("OMEGA_B", "OMEGA_B_H2", "OMEGAB"):
            out.append(float(obh2))
        elif k in ("100_THETA_STAR", "100_THETA_MC", "100_THETA", "100XTHETA"):
            out.append(float(100.0 * rs / max(d_a, eps())))
        elif k in ("L_A", "LA", "ELL_A"):
            raise ValueError(
                "ℓ_A is not enabled in the default compressed mapping (definition-sensitive vs Planck/CAMB). "
                "Use 100_theta_star or supply a custom fork with an explicit ℓ_A definition."
            )
        else:
            raise ValueError(f"unknown compressed CMB label {lab!r}")
    return np.asarray(out, dtype=float)


def predict_shift_parameters(h0: float, Omega_m: float, obh2: float, eng: CosmoInterpEngine) -> tuple[float, float, float]:
    vec = predict_cmb_observables(h0, Omega_m, obh2, eng, ("R", "omega_b"))
    return float(vec[0]), float(vec[1]), float(obh2)


def lnlike_cmb(h0: float, Omega_m: float, obh2: float, eng: CosmoInterpEngine, cmb: CmbShiftData | None) -> float:
    if cmb is None:
        return 0.0
    th = globals().get("_CWSF_EXT_THETA")
    if _external_joint_uses_blend_geometry() and th is not None:
        if not validate_blend_background(h0, Omega_m, float(th[2]), float(th[3]), eng, 1200.0):
            return float(-np.inf)
        if not validate_lcdm_background(h0, Omega_m, obh2, eng, 1200.0):
            return float(-np.inf)
    elif not validate_lcdm_background(h0, Omega_m, obh2, eng, 1200.0):
        return float(-np.inf)
    y_model = predict_cmb_observables(h0, Omega_m, obh2, eng, cmb.labels)
    if np.any(~np.isfinite(y_model)):
        return float(-np.inf)
    r = y_model - cmb.y_obs
    y = la_sci.solve_triangular(cmb.chol, r.reshape(-1, 1), lower=True).ravel()
    return float(-0.5 * float(np.dot(y, y) + cmb.logdet_2pi))


def validate_lcdm_background(h0: float, Omega_m: float, obh2: float, eng: CosmoInterpEngine, z_max_probe: float) -> bool:
    OL = olambda_flat(float(Omega_m), float(h0))
    if not math.isfinite(OL) or OL < 1e-7:
        return False
    zp = np.linspace(0.0, max(float(z_max_probe), 0.05), 41, dtype=float)
    try:
        Ezarr = Hz(zp, float(h0), float(Omega_m))
        if not np.all(np.isfinite(Ezarr)) or float(np.min(Ezarr)) <= 0.0:
            return False
        eng.refresh(float(h0), float(Omega_m))
        dm = np.asarray(eng.comoving_distance_mpc(zp, float(h0), float(Omega_m)), dtype=float)
        if not np.all(np.isfinite(dm)):
            return False
        if np.any(np.diff(dm) < -1e-6 * (1.0 + np.abs(dm[:-1]))):
            return False
    except Exception:
        return False
    rd0 = r_drag_Mpc_eh98(float(h0), float(Omega_m), float(obh2))
    if not math.isfinite(rd0) or rd0 <= 0.0:
        return False
    return True


@dataclass
class ExternalJointPack:
    bao: BaoLikelihoodData | None
    cmb: CmbShiftData | None
    omega_b_h2: float
    disclosure: str

    def lnlike_add(self, h0: float, Omega_m: float, eng: CosmoInterpEngine) -> float:
        lb = lnlike_bao(float(h0), float(Omega_m), float(self.omega_b_h2), eng, self.bao)
        if not math.isfinite(lb):
            return float(-np.inf)
        lc = lnlike_cmb(float(h0), float(Omega_m), float(self.omega_b_h2), eng, self.cmb)
        if not math.isfinite(lc):
            return float(-np.inf)
        s = float(lb + lc)
        return s if math.isfinite(s) else float(-np.inf)

    def chi2_components(
        self,
        h0: float,
        Omega_m: float,
        eng: CosmoInterpEngine,
        joint_geometry_theta: np.ndarray | None = None,
    ) -> dict[str, float | None]:
        """Optional ``joint_geometry_theta``: full blend vector so BAO/CMB use blend H(z), D_M when ccomplet2 physics is active."""
        old_th = globals().get("_CWSF_EXT_THETA")
        old_blend = bool(globals().get("_CWSF_LNPOST_IS_BLEND", False))
        jt = np.asarray(joint_geometry_theta, dtype=float).reshape(-1) if joint_geometry_theta is not None else None
        try:
            if jt is not None and blend_physics_is_ccomplet2() and int(jt.size) >= 4:
                globals()["_CWSF_EXT_THETA"] = jt.copy()
                globals()["_CWSF_LNPOST_IS_BLEND"] = True
            else:
                globals()["_CWSF_LNPOST_IS_BLEND"] = False
            out: dict[str, float | None] = dict(bao_chi2=None, cmb_chi2=None)
            if self.bao is not None:
                ll = lnlike_bao(float(h0), float(Omega_m), float(self.omega_b_h2), eng, self.bao)
                out["bao_chi2"] = float(-2.0 * ll) if math.isfinite(ll) else None
            if self.cmb is not None:
                ll = lnlike_cmb(float(h0), float(Omega_m), float(self.omega_b_h2), eng, self.cmb)
                out["cmb_chi2"] = float(-2.0 * ll) if math.isfinite(ll) else None
            return out
        finally:
            globals()["_CWSF_EXT_THETA"] = old_th
            globals()["_CWSF_LNPOST_IS_BLEND"] = old_blend


def build_external_joint_pack(outdir: Path) -> ExternalJointPack | None:
    # Default **on** for multi-probe joint inference; set CWSF_USE_BAO=0 / CWSF_USE_CMB=0 for SN-only ablations.
    use_bao = os.getenv("CWSF_USE_BAO", "1") == "1"
    use_cmb = os.getenv("CWSF_USE_CMB", "1") == "1"
    if not (use_bao or use_cmb):
        return None
    obh2 = float(os.getenv("CWSF_OMEGA_B_H2", "0.02237"))
    bao_p = Path(os.getenv("CWSF_BAO_JSON", str(outdir / "bao_likelihood_data.json")))
    cmb_p = Path(os.getenv("CWSF_CMB_SHIFT_JSON", str(outdir / "cmb_shift_likelihood_data.json")))
    if use_bao:
        bao = load_bao_json(bao_p) if bao_p.is_file() else baolike_from_raw_dict(_TEMPLATE_BAO_DICT)
    else:
        bao = None
    if use_cmb:
        cmb = load_cmb_shift_json(cmb_p) if cmb_p.is_file() else cmbshift_from_raw_dict(_TEMPLATE_CMB_DICT)
    else:
        cmb = None
    disc = (
        "Compressed BAO/CMB extensions (in-memory _TEMPLATE_*_DICT unless CWSF_*_JSON paths exist): BAO uses Eisenstein–Hu drag-scale r_d "
        "and flat FRW D_M, D_H with the same closed-form χ(z) as the SN driver; CMB uses Hu–Sugiyama z_* and "
        "r_s(z_*) ≈ 1.018 r_d (compressed Gaussian only — not full Planck Plik). "
        f"Omega_b h^2 fixed at {obh2} (not varied with the SN sampler)."
    )
    return ExternalJointPack(bao=bao, cmb=cmb, omega_b_h2=obh2, disclosure=disc)


def plot_bao_dm_rd(
    figpath: Path,
    h0: float,
    Omega_m: float,
    obh2: float,
    eng: CosmoInterpEngine,
    bao: BaoLikelihoodData | None,
) -> None:
    if bao is None:
        return
    figpath.parent.mkdir(parents=True, exist_ok=True)
    zs = np.linspace(0.05, float(max(1.2, float(np.max(bao.z_eff)) * 1.05)), 80)
    mod = np.array([D_M_over_rd(h0, Omega_m, obh2, float(z), eng) for z in zs])
    plt.figure(figsize=(6.5, 4.2))
    plt.plot(zs, mod, lw=2.0, label="model (flat ΛCDM + EH98 r_d)")
    sel = np.array([t.upper().startswith("DM") for t in bao.y_order], dtype=bool)
    if np.any(sel):
        cov = bao.chol @ bao.chol.T
        sig = np.sqrt(np.maximum(np.diag(cov), eps()))
        zd = bao.z_eff[sel]
        yo = bao.y_obs[sel]
        err = sig[sel]
        plt.errorbar(zd, yo, yerr=err, fmt="o", color="C1", label="BAO data (subset)")
    plt.xlabel(r"$z$")
    plt.ylabel(r"$D_M / r_d$")
    plt.title("BAO comoving distance ratio (diagnostic)")
    plt.grid(alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(figpath, dpi=180)
    plt.close()


def plot_cmb_shift_prediction(
    figpath: Path,
    h0: float,
    Omega_m: float,
    obh2: float,
    eng: CosmoInterpEngine,
    cmb: CmbShiftData | None,
) -> None:
    if cmb is None:
        return
    figpath.parent.mkdir(parents=True, exist_ok=True)
    pred = predict_cmb_observables(h0, Omega_m, obh2, eng, cmb.labels)
    names = list(cmb.labels)
    obs = np.asarray(cmb.y_obs, dtype=float)
    cov = cmb.chol @ cmb.chol.T
    err = np.sqrt(np.maximum(np.diag(cov), eps()))
    x = np.arange(obs.size)
    plt.figure(figsize=(7.0, 4.0))
    plt.errorbar(x - 0.1, obs, yerr=err, fmt="s", label="Planck-style compressed mean")
    plt.scatter(x + 0.1, pred, marker="o", label="model @ posterior median")
    plt.xticks(x, names)
    plt.ylabel("value")
    plt.title("Compressed CMB shift parameters (approximation)")
    plt.grid(axis="y", alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(figpath, dpi=180)
    plt.close()


def _nu_mond(x: np.ndarray) -> np.ndarray:
    x = np.maximum(np.asarray(x, dtype=float), eps())
    return 0.5 * (1.0 + np.sqrt(1.0 + 4.0 / x))


def mond_velocity_kms(V_n: np.ndarray, r_kpc: np.ndarray, a0_ms2: float = 1.2e-10) -> np.ndarray:
    r = np.maximum(np.asarray(r_kpc, dtype=float), eps()) * MPC_KM * 1000.0
    vn = np.maximum(np.asarray(V_n, dtype=float), 0.0) * 1000.0
    gN = np.clip(vn**2 / r, eps(), np.inf)
    nu = _nu_mond(gN / max(float(a0_ms2), eps()))
    g = nu * gN
    return np.sqrt(np.clip(g * r, eps(), np.inf)) / 1000.0


def fit_rotmod_file(path: Path, a0_ms2: float = 1.2e-10) -> dict[str, Any]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    rows: list[list[float]] = []
    for ln in lines:
        if not ln.strip() or ln.strip().startswith("#"):
            continue
        parts = ln.split()
        if len(parts) < 8:
            continue
        try:
            rows.append([float(parts[i]) for i in range(8)])
        except ValueError:
            continue
    if len(rows) < 5:
        return dict(galaxy=path.stem, n_rows=0, chi2_newton=None, chi2_mond=None, status="insufficient_rows")
    a = np.asarray(rows, dtype=float)
    R, Vobs, errV, Vgas, Vdisk, Vbul = a[:, 0], a[:, 1], a[:, 2], a[:, 3], a[:, 4], a[:, 5]
    Vn = np.sqrt(np.clip(Vgas**2 + Vdisk**2 + Vbul**2, 0.0, np.inf))
    Vm = mond_velocity_kms(Vn, R, a0_ms2=a0_ms2)
    w = 1.0 / np.maximum(errV**2, eps())
    chi_n = float(np.sum(w * (Vobs - Vn) ** 2))
    chi_m = float(np.sum(w * (Vobs - Vm) ** 2))
    return dict(
        galaxy=path.stem.replace("_rotmod", ""),
        n_rows=int(R.size),
        chi2_newton=chi_n,
        chi2_mond=chi_m,
        rms_newton=float(np.sqrt(np.mean((Vobs - Vn) ** 2))),
        rms_mond=float(np.sqrt(np.mean((Vobs - Vm) ** 2))),
        status="ok",
    )


def run_sparc_mond_benchmark(sparc_dir: Path, out_csv: Path, figpath: Path | None = None) -> pd.DataFrame:
    sparc_dir = Path(sparc_dir)
    if not sparc_dir.is_dir():
        return pd.DataFrame([dict(status="missing_directory", path=str(sparc_dir))])
    files = sorted(sparc_dir.glob("*rotmod*.dat"))
    rows = [fit_rotmod_file(p) for p in files[:400]]
    df = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    if figpath is not None and len(files) > 0:
        p0 = files[0]
        raw = np.loadtxt(p0, comments="#")
        if raw.ndim == 2 and raw.shape[1] >= 8:
            R, Vobs, errV = raw[:, 0], raw[:, 1], raw[:, 2]
            Vn = np.sqrt(np.clip(raw[:, 3] ** 2 + raw[:, 4] ** 2 + raw[:, 5] ** 2, 0.0, np.inf))
            Vm = mond_velocity_kms(Vn, R)
            plt.figure(figsize=(6.0, 4.0))
            plt.errorbar(R, Vobs, yerr=errV, fmt="o", ms=3, alpha=0.65, label="observed")
            plt.plot(R, Vn, lw=1.8, label="Newtonian (SPARC mass model)")
            plt.plot(R, Vm, lw=1.8, label="MOND (simple ν)")
            plt.xlabel(r"$R$ [kpc]")
            plt.ylabel(r"$V$ [km/s]")
            plt.title(f"SPARC example: {p0.stem}")
            plt.legend(fontsize=8)
            plt.grid(alpha=0.3)
            plt.tight_layout()
            figpath.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(figpath, dpi=170)
            plt.close()
    return df


def logistic_w(k: np.ndarray | float, tt: np.ndarray | float, tcrit: np.ndarray | float) -> np.ndarray:
    u = np.clip(
        np.asarray(k, dtype=float) * (np.asarray(tt, dtype=float) - np.asarray(tcrit, dtype=float)),
        -50.0,
        50.0,
    )
    out = np.empty_like(u, dtype=float)
    ok = u >= 0.0

    out[ok] = 1.0 / (1.0 + np.exp(-u[ok]))
    ez = np.exp(u[~ok])
    out[~ok] = ez / (1.0 + ez)
    return np.clip(out, 1e-12, 1.0 - 1e-12)


def horizon_area_residual_ratio_raw(
    z_pack: np.ndarray,
    h0: float,
    Omega_m: float,
    tcrit: float,
    k_slope: float,
    eng: CosmoInterpEngine,
) -> np.ndarray:
    r"""Area residual :math:`r=\log_{10}(A_{\mathrm{blend}}/A_{\mathrm{LCDM}})`.

    **ccomplet2 path:** uses the same ``r_blend(t)`` as ``ccomplet2.py`` with **blend** cosmic time
    :math:`t(z)` from the self-consistent area history.

    **Legacy path** (``CWSF_BLEND_PHYSICS_MODE=legacy_residual``): older LCDM-age + closed-form
    :math:`\chi^2` LCDM-area normalization (diagnostic only).
    """
    OL = olambda_flat(float(Omega_m), float(h0))
    if blend_physics_is_ccomplet2():
        zv = np.asarray(z_pack, dtype=float)
        tgy = np.maximum(eng.blend_cosmic_age_gyr(zv, float(h0), float(Omega_m), float(tcrit), float(k_slope)), 1e-9)
        return np.asarray(c2_r_blend(tgy, float(tcrit), float(k_slope), float(h0), OL, float(Omega_m)), dtype=float)
    chi = np.maximum(lcdm_sinhc_chi(z_pack, float(h0), float(Omega_m)), eps())
    tgy = np.maximum(eng.cosmic_age_gyr(z_pack, float(h0), float(Omega_m)), 1e-9)
    hl_si = float(h0 * math.sqrt(max(OL, eps()))) / MPC_KM
    expo = np.clip(2.0 * hl_si * (tgy * float(GYR_S) - float(T_REF_S)), -120.0, 120.0)
    therm = np.exp(expo)
    agr = np.clip((tgy / float(T_REF_GYR)) ** (4.0 / 3.0), eps(), np.inf)
    wgt = logistic_w(k_slope, tgy, tcrit)
    blend_lin = np.maximum((1.0 - wgt) * agr + wgt * therm, eps())
    return np.log10(blend_lin) - 2.0 * np.log10(chi)


def mu_blend_shape(
    z: np.ndarray,
    h0: float,
    Omega_m: float,
    tcrit: float,
    k_slope: float,
    eng: CosmoInterpEngine,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Blend modulus **shape** (no nuisance ``M``).

    Default (**ccomplet2**): :math:`\\mu(z)=5\\log_{10}(d_L/10\\mathrm{pc})` from blend-consistent
    :math:`D_M(z)` built from the ported Superior Suite / ``ccomplet2`` background (no LCDM spline backbone).

    Legacy (``CWSF_BLEND_PHYSICS_MODE=legacy_residual``): :math:`\\mu=\\mu_{\\mathrm{LCDM}}+2.5\\,\\Delta r`
    anchored at :math:`z=0` (old pipeline; diagnostic only).
    """
    zv = np.asarray(z, dtype=float)
    OL = olambda_flat(float(Omega_m), float(h0))

    if blend_physics_is_ccomplet2():
        dM = np.asarray(
            eng.blend_comoving_distance_mpc(zv, float(h0), float(Omega_m), float(tcrit), float(k_slope)),
            dtype=float,
        )
        dL = (1.0 + zv) * dM
        mu_b = np.asarray(5.0 * np.log10(np.clip(dL, eps(), None) * 1.0e6 / 10.0), dtype=float)
        tgy = np.maximum(eng.blend_cosmic_age_gyr(zv, float(h0), float(Omega_m), float(tcrit), float(k_slope)), 1e-9)
        rr = np.asarray(c2_r_blend(tgy, float(tcrit), float(k_slope), float(h0), OL, float(Omega_m)), dtype=float)
        chi = np.maximum(lcdm_sinhc_chi(zv, float(h0), float(Omega_m)), eps())
        wgt = logistic_w(k_slope, tgy, tcrit)
        ex = dict(
            physics_mode="ccomplet2_blend_background",
            r_lin=np.full_like(zv, np.nan),
            rr=rr,
            rr_anchor_z0=float("nan"),
            w=wgt,
            tgy=tgy,
            chi=chi,
            therm=np.full_like(zv, np.nan),
            agr=np.full_like(zv, np.nan),
            Delta_mu_shape=np.zeros_like(mu_b),
            horizon_delta_mu_ablated=True,
            dM_blend=dM,
            dL_blend=dL,
            phenomenological_note=(
                "Primary blend inference uses self-consistent H(z) and D_M from the ccomplet2-style background "
                "(see module docstring). Set CWSF_BLEND_PHYSICS_MODE=legacy_residual only for backward comparison."
            ),
        )
    else:
        mu0, DL, DM = mu_lcdm_shape(z, float(h0), float(Omega_m), eng)
        z_pack = np.concatenate((np.asarray([0.0], dtype=float), zv), axis=0)
        rr_p = horizon_area_residual_ratio_raw(z_pack, float(h0), float(Omega_m), float(tcrit), float(k_slope), eng)
        rr = np.asarray(rr_p[1:] - rr_p[0], dtype=float)
        chi = np.maximum(lcdm_sinhc_chi(zv, float(h0), float(Omega_m)), eps())
        tgy = np.maximum(eng.cosmic_age_gyr(zv, float(h0), float(Omega_m)), 1e-9)
        hl_si = float(h0 * math.sqrt(max(OL, eps()))) / MPC_KM
        expo = np.clip(2.0 * hl_si * (tgy * float(GYR_S) - float(T_REF_S)), -120.0, 120.0)
        therm = np.exp(expo)
        agr = np.clip((tgy / float(T_REF_GYR)) ** (4.0 / 3.0), eps(), np.inf)
        wgt = logistic_w(k_slope, tgy, tcrit)
        blend_lin = np.maximum((1.0 - wgt) * agr + wgt * therm, eps())
        delta_shape = np.asarray(2.5 * rr, dtype=float)
        if USE_BLEND_HORIZON_DELTA_MU:
            mu_b = np.asarray(mu0 + delta_shape, dtype=float)
        else:
            mu_b = np.asarray(mu0, dtype=float)
        ex = dict(
            physics_mode="legacy_residual_on_lcdm_distances",
            r_lin=blend_lin,
            rr=rr,
            rr_anchor_z0=float(rr_p[0]),
            w=wgt,
            tgy=tgy,
            chi=chi,
            therm=therm,
            agr=agr,
            Delta_mu_shape=delta_shape if USE_BLEND_HORIZON_DELTA_MU else np.zeros_like(delta_shape),
            horizon_delta_mu_ablated=bool(not USE_BLEND_HORIZON_DELTA_MU),
            phenomenological_note=(
                "Legacy: horizon-area as residual on LCDM μ; toggle CWSF_USE_BLEND_HORIZON_DELTA_MU=0 for ablations."
            ),
        )

    if np.any(~np.isfinite(mu_b)) | np.any(mu_b < 20.0) | np.any(mu_b > 50.0):
        mu_b = np.full_like(mu_b, np.nan, dtype=float)

    return np.asarray(mu_b, float), ex


def mu_blend(
    z: np.ndarray,
    h0: float,
    Omega_m: float,
    tcrit: float,
    k_slope: float,
    Mmag: float,
    eng: CosmoInterpEngine,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    mu_s, ex = mu_blend_shape(z, float(h0), float(Omega_m), float(tcrit), float(k_slope), eng)
    mu_b = mu_s + float(Mmag)
    ex = dict(**ex)
    return np.asarray(mu_b, float), ex


def profiled_absolute_magnitude(mu_shape: np.ndarray, mu_obs: np.ndarray, sig: np.ndarray) -> float:
    """Gaussian-linear best fit for nuisance offset closed form (heteroscedastic diagonal)."""
    w = 1.0 / np.square(np.maximum(np.asarray(sig, dtype=float), eps()))
    return float(np.sum(w * (np.asarray(mu_obs, dtype=float) - np.asarray(mu_shape, dtype=float))) / np.sum(w))




def log_prior_truncgauss(x: float, lo: float, hi: float, mu: float, sigma: float) -> float:
    xv, lo_, hi_, mu_, sg = float(x), float(lo), float(hi), float(mu), float(sigma)
    if not (lo_ <= xv <= hi_):
        return float(-np.inf)
    aa = float((lo_ - mu_) / sg)
    bb = float((hi_ - mu_) / sg)
    return float(stats.truncnorm.logpdf(xv, aa, bb, loc=mu_, scale=sg))


def logprior_lcdm_h0om(theta: np.ndarray) -> float:
    th = np.asarray(theta, dtype=float).reshape(-1)
    if th.size != 2:
        return float(-np.inf)
    h0, om = float(th[0]), float(th[1])
    lp = log_prior_truncgauss(h0, BND["H0"][0], BND["H0"][1], PRI_MU["H0"], PRI_SG["H0"]) + log_prior_truncgauss(
        om, BND["Om"][0], BND["Om"][1], PRI_MU["Om"], PRI_SG["Om"]
    )
    return lp if math.isfinite(lp) else float(-np.inf)


def logprior_lcdm_full(theta: np.ndarray) -> float:
    th = np.asarray(theta, dtype=float).reshape(-1)
    if th.size != 3:
        return float(-np.inf)
    h0, om, mm = float(th[0]), float(th[1]), float(th[2])
    lp = (
        log_prior_truncgauss(h0, BND["H0"][0], BND["H0"][1], PRI_MU["H0"], PRI_SG["H0"])
        + log_prior_truncgauss(om, BND["Om"][0], BND["Om"][1], PRI_MU["Om"], PRI_SG["Om"])
        + log_prior_truncgauss(mm, BND["M"][0], BND["M"][1], PRI_MAG_MU, PRI_MAG_SG)
    )
    return lp if math.isfinite(lp) else float(-np.inf)


def logprior_blend_four(theta: np.ndarray) -> float:
    th = np.asarray(theta, dtype=float).reshape(-1)
    if th.size != 4:
        return float(-np.inf)
    h0, om, tc, ks = float(th[0]), float(th[1]), float(th[2]), float(th[3])
    lp0 = float(logprior_lcdm_h0om(np.asarray([h0, om], dtype=float)))
    if lp0 == float(-np.inf):
        return float(-np.inf)
    lp = (
        lp0
        + log_prior_truncgauss(tc, BND["tc"][0], BND["tc"][1], PRI_MU["tc"], PRI_SG["tc"])
        + log_prior_truncgauss(ks, BND["k"][0], BND["k"][1], PRI_MU["k"], PRI_SG["k"])
    )
    return lp if math.isfinite(lp) else float(-np.inf)


def logprior_blend_full(theta: np.ndarray) -> float:
    th = np.asarray(theta, dtype=float).reshape(-1)
    if th.size != 5:
        return float(-np.inf)
    h0, om, tc, ks, mm = float(th[0]), float(th[1]), float(th[2]), float(th[3]), float(th[4])
    lp0 = float(logprior_lcdm_full(np.asarray([h0, om, mm], dtype=float)))
    if lp0 == float(-np.inf):
        return float(-np.inf)
    lp = (
        lp0
        + log_prior_truncgauss(tc, BND["tc"][0], BND["tc"][1], PRI_MU["tc"], PRI_SG["tc"])
        + log_prior_truncgauss(ks, BND["k"][0], BND["k"][1], PRI_MU["k"], PRI_SG["k"])
    )
    return lp if math.isfinite(lp) else float(-np.inf)


def lnlike_lcdm_profiled(
    theta: np.ndarray,
    z: np.ndarray,
    mu_obs: np.ndarray,
    sig: np.ndarray,
    eng: CosmoInterpEngine,
) -> float:
    th = np.asarray(theta, dtype=float).reshape(-1)
    if th.size != 2:
        return float(-np.inf)
    mu_shape, _, _ = mu_lcdm_shape(z, float(th[0]), float(th[1]), eng)
    w = 1.0 / np.square(np.maximum(np.asarray(sig, dtype=float), eps()))
    r = np.asarray(mu_obs, dtype=float) - np.asarray(mu_shape, dtype=float)
    sw = float(np.sum(w))
    chi = float(np.sum(w * np.square(r)) - (float(np.sum(w * r))) ** 2 / sw)
    return float(-0.5 * max(chi, 0.0))


def lnlike_blend_profiled(
    theta: np.ndarray,
    z: np.ndarray,
    mu_obs: np.ndarray,
    sig: np.ndarray,
    eng: CosmoInterpEngine,
) -> float:
    th = np.asarray(theta, dtype=float).reshape(-1)
    if th.size != 4:
        return float(-np.inf)
    mu_shape, _ = mu_blend_shape(z, float(th[0]), float(th[1]), float(th[2]), float(th[3]), eng)
    if np.any(~np.isfinite(mu_shape)):
        return float(-np.inf)
    w = 1.0 / np.square(np.maximum(np.asarray(sig, dtype=float), eps()))
    r = np.asarray(mu_obs, dtype=float) - np.asarray(mu_shape, dtype=float)
    sw = float(np.sum(w))
    chi = float(np.sum(w * np.square(r)) - (float(np.sum(w * r))) ** 2 / sw)
    return float(-0.5 * max(chi, 0.0))


def lnlike_lcdm_full(
    theta: np.ndarray,
    z: np.ndarray,
    mu_obs: np.ndarray,
    sig: np.ndarray,
    eng: CosmoInterpEngine,
) -> float:
    th = np.asarray(theta, dtype=float).reshape(-1)
    if th.size != 3:
        return float(-np.inf)
    mod, _, _ = mu_lcdm(z, float(th[0]), float(th[1]), float(th[2]), eng)
    mo = np.asarray(mu_obs, dtype=float)
    sg = np.asarray(sig, dtype=float)
    chi = float(np.sum(((mo - mod) / sg) ** 2))
    return float(-0.5 * chi)


def log_post_lcdm_cov(
    theta: np.ndarray,
    mu_obs: np.ndarray,
    chol_sigma: np.ndarray,
    logdet_2pi: float,
    z: np.ndarray,
    eng: CosmoInterpEngine,
) -> float:
    th = np.asarray(theta, dtype=float).reshape(-1)
    lp = float(logprior_lcdm_full(th))
    if lp == float(-np.inf):
        return float(-np.inf)
    if not validate_flat_background(float(th[0]), float(th[1]), eng, float(np.max(z)) if z.size else 0.05):
        return float(-np.inf)
    mod, _, _ = mu_lcdm(z, float(th[0]), float(th[1]), float(th[2]), eng)
    ll = lnlike_mvn_residual(np.asarray(mu_obs, dtype=float) - np.asarray(mod, dtype=float), chol_sigma, logdet_2pi)
    if not math.isfinite(ll):
        return float(-np.inf)
    ll_ext = _external_lnlike_add(th, eng)
    if not math.isfinite(ll_ext):
        return float(-np.inf)
    return float(lp + ll + ll_ext)


def log_post_blend_cov(
    theta: np.ndarray,
    mu_obs: np.ndarray,
    chol_sigma: np.ndarray,
    logdet_2pi: float,
    z: np.ndarray,
    eng: CosmoInterpEngine,
) -> float:
    globals()["_CWSF_LNPOST_IS_BLEND"] = True
    try:
        th = np.asarray(theta, dtype=float).reshape(-1)
        lp = float(logprior_blend_full(th))
        if lp == float(-np.inf):
            return float(-np.inf)
        zmx = float(np.max(z)) if z.size else 0.05
        h0_, om_ = float(th[0]), float(th[1])
        if blend_physics_is_ccomplet2():
            if th.size < 4:
                return float(-np.inf)
            if not validate_blend_background(h0_, om_, float(th[2]), float(th[3]), eng, zmx):
                return float(-np.inf)
        elif not validate_flat_background(h0_, om_, eng, zmx):
            return float(-np.inf)
        mod, _ = mu_blend(z, float(th[0]), float(th[1]), float(th[2]), float(th[3]), float(th[4]), eng)
        mod = np.asarray(mod, dtype=float)
        if np.any(~np.isfinite(mod)):
            return float(-np.inf)
        ll = lnlike_mvn_residual(np.asarray(mu_obs, dtype=float) - np.asarray(mod, dtype=float), chol_sigma, logdet_2pi)
        if not math.isfinite(ll):
            return float(-np.inf)
        ll_ext = _external_lnlike_add(th, eng)
        if not math.isfinite(ll_ext):
            return float(-np.inf)
        return float(lp + ll + ll_ext)
    finally:
        globals()["_CWSF_LNPOST_IS_BLEND"] = False


def lnlike_blend_full(
    theta: np.ndarray,
    z: np.ndarray,
    mu_obs: np.ndarray,
    sig: np.ndarray,
    eng: CosmoInterpEngine,
) -> float:
    th = np.asarray(theta, dtype=float).reshape(-1)
    if th.size != 5:
        return float(-np.inf)
    mod, _ = mu_blend(z, float(th[0]), float(th[1]), float(th[2]), float(th[3]), float(th[4]), eng)
    mo = np.asarray(mu_obs, dtype=float)
    sg = np.asarray(sig, dtype=float)
    if np.any(~np.isfinite(mod)):
        return float(-np.inf)
    chi = float(np.sum(((mo - mod) / sg) ** 2))
    return float(-0.5 * chi)


def log_post_lcdm(
    theta: np.ndarray,
    z: np.ndarray,
    mu_obs: np.ndarray,
    sig: np.ndarray,
    eng: CosmoInterpEngine,
) -> float:
    globals()["_CWSF_LNPOST_IS_BLEND"] = False
    th = np.asarray(theta, dtype=float).reshape(-1)
    lp = float(logprior_lcdm_h0om(th)) if infer_profile_m() else float(logprior_lcdm_full(th))
    if lp == float(-np.inf):
        return float(-np.inf)
    h0_, om_ = float(th[0]), float(th[1])
    zmax = float(np.max(np.asarray(z, dtype=float))) if z.size else 0.05
    if not validate_flat_background(h0_, om_, eng, zmax):
        return float(-np.inf)
    ll = lnlike_lcdm_profiled(th, z, mu_obs, sig, eng) if infer_profile_m() else lnlike_lcdm_full(th, z, mu_obs, sig, eng)
    if not math.isfinite(ll):
        return float(-np.inf)
    ll_ext = _external_lnlike_add(th, eng)
    if not math.isfinite(ll_ext):
        return float(-np.inf)
    return float(lp + ll + ll_ext)


def log_post_blend(
    theta: np.ndarray,
    z: np.ndarray,
    mu_obs: np.ndarray,
    sig: np.ndarray,
    eng: CosmoInterpEngine,
) -> float:
    globals()["_CWSF_LNPOST_IS_BLEND"] = True
    try:
        th = np.asarray(theta, dtype=float).reshape(-1)
        lp = float(logprior_blend_four(th)) if infer_profile_m() else float(logprior_blend_full(th))
        if lp == float(-np.inf):
            return float(-np.inf)
        h0_, om_ = float(th[0]), float(th[1])
        zmax = float(np.max(np.asarray(z, dtype=float))) if z.size else 0.05
        if blend_physics_is_ccomplet2():
            if th.size < 4:
                return float(-np.inf)
            if not validate_blend_background(h0_, om_, float(th[2]), float(th[3]), eng, zmax):
                return float(-np.inf)
        elif not validate_flat_background(h0_, om_, eng, zmax):
            return float(-np.inf)
        ll = lnlike_blend_profiled(th, z, mu_obs, sig, eng) if infer_profile_m() else lnlike_blend_full(th, z, mu_obs, sig, eng)
        if not math.isfinite(ll):
            return float(-np.inf)
        ll_ext = _external_lnlike_add(th, eng)
        if not math.isfinite(ll_ext):
            return float(-np.inf)
        return float(lp + ll + ll_ext)
    finally:
        globals()["_CWSF_LNPOST_IS_BLEND"] = False


def gelman_r_hat_independent_chains(chains: Sequence[np.ndarray]) -> np.ndarray:
    """Proper :math:`\\hat R` across ``C`` independent pooled chains (each flat ``(n, ndim)``)."""
    if len(chains) < 2:
        dd = int(np.asarray(chains[0]).shape[1])
        return np.full(dd, np.nan, dtype=float)
    arrs = [np.asarray(c, dtype=float) for c in chains]
    nmin = min(int(a.shape[0]) for a in arrs if a.ndim == 2)
    ndims = min(int(a.shape[1]) for a in arrs)
    if nmin < 20:
        return np.full(ndims, np.nan, dtype=float)
    jj = slice(0, nmin)
    cstack = np.stack([a[jj, :ndims] for a in arrs], axis=0)
    cc, nt, dims = int(cstack.shape[0]), int(cstack.shape[1]), int(cstack.shape[2])
    outs: list[float] = []
    for di in range(dims):
        ch = np.swapaxes(cstack[:, :, di], 0, 1)
        gmean = np.mean(np.mean(ch, axis=0))
        bb = nt / float(cc - 1) * np.sum((np.mean(ch, axis=0) - gmean) ** 2)
        ww = np.mean(np.var(ch, axis=0, ddof=1))
        vhat = (nt - 1) / nt * ww + bb / nt
        outs.append(math.sqrt(max(vhat / max(ww, eps()), eps())))
    return np.asarray(outs, dtype=float)


def gelman_r_hat(samples: np.ndarray) -> np.ndarray:
    """Split-chain diagnostic (sequential halves of a pooled chain)."""
    s = np.asarray(samples, dtype=float)
    if s.ndim != 2:
        raise ValueError("samples must be 2-D (n_flat, ndim)")
    n_steps, nd = int(s.shape[0]), int(s.shape[1])
    half = n_steps // 2
    if half < 20:
        return np.full(nd, np.nan, dtype=float)
    c1 = s[:half, :]
    c2 = s[half:, :]
    chains = np.stack([c1, c2], axis=0)
    rhat_vals: list[float] = []
    for jj in range(nd):
        ch_j = chains[:, :, jj]
        j_ch, n_here = float(ch_j.shape[0]), float(ch_j.shape[1])
        means_here = np.mean(ch_j, axis=1)
        grand_here = float(np.mean(means_here))
        b_here = float(n_here / (j_ch - 1.0) * np.sum((means_here - grand_here) ** 2))
        w_here = float(np.mean(np.var(ch_j, axis=1, ddof=1)))
        var_hat_here = float((n_here - 1.0) / n_here * w_here + b_here / n_here)
        rhat_vals.append(math.sqrt(max(var_hat_here / max(w_here, eps()), eps())))
    return np.asarray(rhat_vals, dtype=float)


def chain_rhat_from_walkers(ch: np.ndarray) -> np.ndarray:
    """Gelman--Rubin R-hat treating each walker as an independent chain at fixed length."""
    x = np.asarray(ch, dtype=float)
    if x.ndim != 3:
        raise ValueError("expected shape (n_steps, n_walkers, n_dim)")
    n_here, walkers_here, dims_here = int(x.shape[0]), int(x.shape[1]), int(x.shape[2])
    if n_here < 10 or walkers_here < 4:
        return np.full(dims_here, np.nan, dtype=float)
    out: list[float] = []
    for kk in range(dims_here):
        cw = np.swapaxes(x[:, :, kk], 0, 1)
        j_here, nt_here = int(cw.shape[0]), int(cw.shape[1])
        means_here = np.mean(cw, axis=1)
        grand_here = float(np.mean(means_here))
        b_here = float(nt_here / (j_here - 1.0) * np.sum((means_here - grand_here) ** 2))
        var_within_here = float(np.mean(np.var(cw, axis=1, ddof=1)))
        var_hat_here = float((nt_here - 1.0) / nt_here * var_within_here + b_here / nt_here)
        out.append(math.sqrt(max(var_hat_here / max(var_within_here, eps()), eps())))
    return np.asarray(out, dtype=float)


def integrated_autocorr_time(chain3d: np.ndarray) -> np.ndarray:
    """Average integrated autocorrelation time across walkers (emcee)."""
    try:
        from emcee.autocorr import integrated_time
    except Exception:
        return np.full(int(chain3d.shape[2]), np.nan, dtype=float)
    x = np.asarray(chain3d, dtype=float)
    n_step, n_walk, nd = int(x.shape[0]), int(x.shape[1]), int(x.shape[2])
    tau_mat = np.full((n_walk, nd), np.nan, dtype=float)
    for w in range(n_walk):
        try:
            t = np.atleast_1d(np.asarray(integrated_time(x[:, w, :], c=5, tol=50), dtype=float)).reshape(-1)
            nk = min(int(t.size), nd)
            tau_mat[w, :nk] = t[:nk]
        except Exception:
            pass
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(tau_mat, axis=0)


def run_emcee(
    ndim: int,
    nwalk: int,
    n_burn: int | None,
    n_prod: int | None,
    lnpost: Callable[[np.ndarray], float],
    p0: np.ndarray,
) -> tuple[object, np.ndarray, np.ndarray, np.ndarray, float]:
    if not HAVE_EMCEE or emcee is None:
        raise RuntimeError("Install emcee to run this pipeline (pip install emcee).")
    nb = int(N_BURN if n_burn is None else n_burn)
    np_ = int(N_PROD if n_prod is None else n_prod)
    sampler = emcee.EnsembleSampler(int(nwalk), int(ndim), lnpost)
    pos, _, _ = sampler.run_mcmc(np.asarray(p0, dtype=float), nb, progress=True)
    sampler.reset()
    pos, _, _ = sampler.run_mcmc(pos, np_, progress=True)
    chain3d = np.asarray(sampler.get_chain(flat=False), dtype=float)
    flat_chain = np.asarray(sampler.get_chain(flat=True), dtype=float)
    log_prob = np.asarray(sampler.get_log_prob(flat=True), dtype=float)
    acc = float(np.mean(sampler.acceptance_fraction)) if hasattr(sampler, "acceptance_fraction") else float("nan")
    return sampler, chain3d, flat_chain, log_prob, acc


def init_walkers(
    nwalk: int,
    centers: np.ndarray,
    widths: np.ndarray,
    lows: np.ndarray,
    highs: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    c = np.asarray(centers, dtype=float).reshape(-1)
    w = np.asarray(widths, dtype=float).reshape(-1)
    lo = np.asarray(lows, dtype=float).reshape(-1)
    hi = np.asarray(highs, dtype=float).reshape(-1)
    nd = int(c.size)
    out = np.empty((int(nwalk), nd), dtype=float)
    for i in range(int(nwalk)):
        for j in range(nd):
            x = float(rng.normal(c[j], w[j]))
            ii = 0
            while not (lo[j] <= x <= hi[j]) and ii < 50:
                x = float(rng.normal(c[j], w[j]))
                ii += 1
            out[i, j] = float(np.clip(x, lo[j], hi[j]))
    return out


def download_text(url: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "cwsf-pipeline/1.0"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        dst.write_bytes(resp.read())


def ensure_download(url: str, dst: Path) -> None:
    if dst.exists() and os.getenv("CWSF_REFETCH", "0") != "1":
        return
    download_text(url, dst)


def load_pantheon(path_p: Path) -> pd.DataFrame:
    df = pd.read_csv(path_p, sep=r"\s+", engine="python")
    need_cols = ["CID", "zCMB", "MU_SH0ES", "MU_SH0ES_ERR_DIAG"]
    miss = [c for c in need_cols if c not in df.columns]
    if miss:
        raise KeyError(f"Pantheon file missing columns {miss}; first columns are {list(df.columns)[:25]}")
    use = df[need_cols].copy()
    if "IS_CALIBRATOR" in df.columns and os.getenv("CWSF_KEEP_CALIBRATORS", "0") != "1":
        cal = pd.to_numeric(df["IS_CALIBRATOR"], errors="coerce").fillna(0).astype(int)
        use = use.loc[cal == 0]
    use = use.drop_duplicates(subset=["CID"], keep="first")
    use = use.rename(columns={"zCMB": "z", "MU_SH0ES": "mu", "MU_SH0ES_ERR_DIAG": "sig"})
    use = use[["CID", "z", "mu", "sig"]].copy()
    for c in ("z", "mu", "sig"):
        use[c] = pd.to_numeric(use[c], errors="coerce")
    use = use.dropna(subset=["z", "mu", "sig"])
    use = use[(use["z"] > 0.01) & (use["sig"] > 0)].copy()
    return use.reset_index(drop=True)


def load_des(path_d: Path) -> pd.DataFrame:
    lines = path_d.read_text(encoding="utf-8", errors="replace").splitlines()
    var_line = next(ln for ln in lines if ln.startswith("VARNAMES:"))
    names = var_line.split()[1:]
    rows: list[list[str]] = []
    for ln in lines:
        if not ln.startswith("SN:"):
            continue
        parts = ln.split()
        chunk = parts[1 : 1 + len(names)]
        if len(chunk) != len(names):
            continue
        rows.append(chunk)
    wd = pd.DataFrame(rows, columns=names)
    if "zHD" not in wd.columns or "MU" not in wd.columns or "MUERR" not in wd.columns:
        raise KeyError(f"Unexpected DES columns: {wd.columns.tolist()}")
    sig = pd.to_numeric(wd["MUERR"], errors="coerce").astype(float)
    if "MUERR_SYS" in wd.columns:
        sig = np.sqrt(sig**2 + pd.to_numeric(wd["MUERR_SYS"], errors="coerce").astype(float) ** 2)
    if "MUERR_VPEC" in wd.columns:
        sig = np.sqrt(sig**2 + pd.to_numeric(wd["MUERR_VPEC"], errors="coerce").astype(float) ** 2)
    out = pd.DataFrame(
        {
            "z": pd.to_numeric(wd["zHD"], errors="coerce"),
            "mu": pd.to_numeric(wd["MU"], errors="coerce"),
            "sig": sig,
        }
    ).dropna()
    out = out[(out["z"] > 0.01) & (out["sig"] > 0)].copy()
    return out.reset_index(drop=True)


def json_safe_float(val: float) -> float | None:
    xf = float(val)
    return xf if math.isfinite(xf) else None


def json_safe_mx(arr: np.ndarray) -> float | None:
    if arr.size == 0:
        return None
    v = np.nanmax(np.asarray(arr, dtype=float))
    return json_safe_float(float(v))


def versions_dict() -> dict:
    def _ver(pkg: str) -> str | None:
        try:
            return str(importlib_metadata.version(pkg))
        except Exception:
            return None

    try:
        import scipy as _sp

        scipy_ver = str(getattr(_sp, "__version__", _ver("scipy")))
    except Exception:
        scipy_ver = _ver("scipy")

    return {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy_ver,
        "matplotlib": matplotlib.__version__,
        "emcee": _ver("emcee") if HAVE_EMCEE else None,
        "dynesty": _ver("dynesty") if HAVE_DYNESTY else None,
        "corner": _ver("corner") if HAVE_CORNER else None,
        "pyarrow_available": HAVE_PARQUET,
        "KMS_TO_SI": KMS_TO_SI,
        "seed": RNG_SEED,
        "walkers_chains_burn_prod": [N_WALKERS, N_CHAINS, N_BURN, N_PROD],
        "PROFILE_M_analytic_marginalization": PROFILE_M,
        "USE_LOWZ_DL_ANCHOR": USE_LOWZ_DL_ANCHOR,
        "USE_BLEND_HORIZON_DELTA_MU_ablation_toggle": USE_BLEND_HORIZON_DELTA_MU,
        "Z_REF_DELTA_MU_CORR": Z_REF_DELTA_MU_CORR,
        "PANT_STATSYS_covariance_official_url": PANT_COV_STATSYS_URL,
        "PANT_covariance_download_url_resolved_at_runtime": str(PANT_COVSTAT_URL),
        "DES_URL_resolved_at_runtime": str(DES_URL),
        "CWSF_USE_BAO": os.getenv("CWSF_USE_BAO", "1"),
        "CWSF_USE_CMB": os.getenv("CWSF_USE_CMB", "1"),
        "CWSF_OMEGA_B_H2": os.getenv("CWSF_OMEGA_B_H2", "0.02237"),
        "ISEF_PEARSON_BOOTSTRAP": int(ISEF_PEARSON_BOOTSTRAP),
        "ISEF_POSTERIOR_ROWS_FOR_CORR_CAP": int(ISEF_POSTERIOR_SUBSAMPLE_CORR),
        "CWSF_MULTI_SEEDS_framework_runs_default": int(N_FRAMEWORK_SEEDS),
        "CWSF_CROSSVAL_env_default_off": RUN_CV,
        "CWSF_nested_default": RUN_NESTED,
    }


def pointwise_loglike_gauss(mu_mod: np.ndarray, mu_obs: np.ndarray, sig: np.ndarray) -> np.ndarray:
    mm = np.asarray(mu_mod, dtype=float)
    mo = np.asarray(mu_obs, dtype=float)
    sg = np.asarray(sig, dtype=float)
    vv = sg**2
    return -0.5 * np.log(2.0 * math.pi * vv) - (mo - mm) ** 2 / (2.0 * vv)


def waic_from_loglik_matrix(log_lik: np.ndarray) -> float:
    """WAIC from (n_draws, n_data) matrix of pointwise log-likelihoods."""
    ll = np.asarray(log_lik, dtype=float)
    s_n = ll.shape
    if ll.ndim != 2 or s_n[0] < 20:
        return float("nan")
    log_s = float(math.log(float(s_n[0])))
    lpd = float(np.sum(logsumexp(ll, axis=0) - log_s))
    p_waic = float(np.sum(np.var(ll, axis=0, ddof=1)))
    return float(-2.0 * (lpd - p_waic))


def max_aic_bic(n: int, k: int, max_loglike: float) -> tuple[float, float]:
    return float(2.0 * k - 2.0 * max_loglike), float(k * math.log(n) - 2.0 * max_loglike)


def predictive_quantiles(
    flat: np.ndarray,
    zgrid: np.ndarray,
    rng: np.random.Generator,
    blend: bool,
    eng: CosmoInterpEngine,
    z_train: np.ndarray,
    mu_train: np.ndarray,
    sig_train: np.ndarray,
    max_rows: int = 2048,
) -> tuple[np.ndarray, ...]:
    """Posterior predictive quantiles for :math:`\\mu(z)` and :math:`\\Delta\\mu` (blend shape minus LCDM shape)."""
    idx = rng.choice(int(flat.shape[0]), size=min(int(max_rows), int(flat.shape[0])), replace=False)
    stack_mu: list[np.ndarray] = []
    stack_dm: list[np.ndarray] = []
    for row in flat[idx]:
        if infer_profile_m():
            if blend:
                mu_tr, _ = mu_blend_shape(z_train, float(row[0]), float(row[1]), float(row[2]), float(row[3]), eng)
                mu_s, _ = mu_blend_shape(zgrid, float(row[0]), float(row[1]), float(row[2]), float(row[3]), eng)
                mu_lcdm_g, _, _ = mu_lcdm_shape(zgrid, float(row[0]), float(row[1]), eng)
            else:
                mu_tr, _, _ = mu_lcdm_shape(z_train, float(row[0]), float(row[1]), eng)
                mu_s, _, _ = mu_lcdm_shape(zgrid, float(row[0]), float(row[1]), eng)
                mu_lcdm_g = mu_s.copy()
            mm = profiled_absolute_magnitude(mu_tr, mu_train, sig_train)
            mhat = mu_s + mm
            dmu = np.asarray(mu_s, dtype=float) - np.asarray(mu_lcdm_g, dtype=float)
        elif blend:
            mhat_b, _ = mu_blend(zgrid, float(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4]), eng)
            mhat_l, _, _ = mu_lcdm(zgrid, float(row[0]), float(row[1]), float(row[4]), eng)
            mhat = mhat_b
            dmu = np.asarray(mhat_b, dtype=float) - np.asarray(mhat_l, dtype=float)
        else:
            mhat, _, _ = mu_lcdm(zgrid, float(row[0]), float(row[1]), float(row[2]), eng)
            dmu = np.zeros_like(np.asarray(mhat, dtype=float))
        stack_mu.append(np.asarray(mhat, dtype=float).reshape(-1))
        stack_dm.append(np.asarray(dmu, dtype=float).reshape(-1))
    mat = np.stack(stack_mu, axis=0)
    dmat = np.stack(stack_dm, axis=0)
    qs = lambda a, qs_: [np.asarray(x, float) for x in np.quantile(a, qs_, axis=0)]
    q025, q16, q50, q84, q975 = qs(mat, [0.025, 0.16, 0.5, 0.84, 0.975])
    d025, d16, d50, d84, d975 = qs(dmat, [0.025, 0.16, 0.5, 0.84, 0.975])
    return q16, q50, q84, q025, q975, d16, d50, d84, d025, d975


def metrics_holdout_profiled(
    theta: np.ndarray,
    pantheon_z: np.ndarray,
    pantheon_mu: np.ndarray,
    pantheon_sig: np.ndarray,
    des: pd.DataFrame,
    blend: bool,
    eng: CosmoInterpEngine,
) -> dict:
    """DES predictions use `M` **profiled on Pantheon only** (no DES leakage)."""
    zv = np.asarray(des["z"], dtype=float)
    muo = np.asarray(des["mu"], dtype=float)
    sg = np.asarray(des["sig"], dtype=float)
    th = np.asarray(theta, dtype=float).reshape(-1)
    if infer_profile_m():
        if blend:
            mu_tr, _ = mu_blend_shape(pantheon_z, float(th[0]), float(th[1]), float(th[2]), float(th[3]), eng)
            mu_des_s, _ = mu_blend_shape(zv, float(th[0]), float(th[1]), float(th[2]), float(th[3]), eng)
        else:
            mu_tr, _, _ = mu_lcdm_shape(pantheon_z, float(th[0]), float(th[1]), eng)
            mu_des_s, _, _ = mu_lcdm_shape(zv, float(th[0]), float(th[1]), eng)
        mm = profiled_absolute_magnitude(mu_tr, pantheon_mu, pantheon_sig)
        muhat = mu_des_s + mm
    elif blend:
        muhat, _ = mu_blend(zv, float(th[0]), float(th[1]), float(th[2]), float(th[3]), float(th[4]), eng)
    else:
        muhat, _, _ = mu_lcdm(zv, float(th[0]), float(th[1]), float(th[2]), eng)
    res = muo - muhat
    rmse = float(math.sqrt(float(np.mean(res**2))))
    chi2 = float(np.sum((res / sg) ** 2))
    return dict(rmse=rmse, chi2=chi2, bias=float(np.mean(res)), residual_std=float(np.std(res)))


def ppc_train_stats(
    flat: np.ndarray,
    rng: np.random.Generator,
    blend: bool,
    eng: CosmoInterpEngine,
    z_train: np.ndarray,
    mu_train: np.ndarray,
    sig_train: np.ndarray,
    n_draws: int = 256,
) -> dict[str, float]:
    """Posterior predictive training checks (Gaussian synthetic draws diag)."""
    n_draws = int(min(int(n_draws), int(flat.shape[0])))
    idx = rng.choice(int(flat.shape[0]), size=n_draws, replace=False)
    chi2_list: list[float] = []
    rmse_list: list[float] = []
    bias_list: list[float] = []
    rstd_list: list[float] = []
    for row in flat[idx]:
        if infer_profile_m():
            if blend:
                mu_fit, _ = mu_blend_shape(z_train, float(row[0]), float(row[1]), float(row[2]), float(row[3]), eng)
            else:
                mu_fit, _, _ = mu_lcdm_shape(z_train, float(row[0]), float(row[1]), eng)
            mm = profiled_absolute_magnitude(mu_fit, mu_train, sig_train)
            mu_mod = mu_fit + mm
        elif blend:
            mu_mod, _ = mu_blend(z_train, float(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4]), eng)
        else:
            mu_mod, _, _ = mu_lcdm(z_train, float(row[0]), float(row[1]), float(row[2]), eng)
        syn = rng.normal(mu_mod, sig_train)
        resid = syn - mu_mod
        rmse_list.append(float(math.sqrt(np.mean(resid**2))))
        chi2_list.append(float(np.sum((syn - mu_mod) ** 2 / sig_train**2)))
        bias_list.append(float(np.mean(syn - mu_train)))
        rstd_list.append(float(np.std(syn - mu_train)))
    return dict(
        ppc_chi2_mean=float(np.mean(chi2_list)),
        ppc_rmse_mean=float(np.mean(rmse_list)),
        ppc_bias_syn_minus_obs=float(np.mean(bias_list)),
        ppc_resid_std_syn_minus_obs=float(np.mean(rstd_list)),
    )


def ppc_train_distribution_stats(
    flat: np.ndarray,
    rng: np.random.Generator,
    blend: bool,
    eng: CosmoInterpEngine,
    z_train: np.ndarray,
    mu_train: np.ndarray,
    sig_train: np.ndarray,
    n_draws: int = 384,
) -> dict[str, float | list[float]]:
    """Per-draw PPC replicas for expanded checks (skew, kurtosis of std-residuals, slope vs *z*)."""
    n_draws = int(min(int(n_draws), int(flat.shape[0])))
    idx = rng.choice(int(flat.shape[0]), size=n_draws, replace=False)
    chi2s: list[float] = []
    rmses: list[float] = []
    slopes: list[float] = []
    sk_pool: list[float] = []
    ku_pool: list[float] = []
    zv = np.asarray(z_train, dtype=float).reshape(-1)
    for row in flat[idx]:
        if infer_profile_m():
            if blend:
                mu_fit, _ = mu_blend_shape(zv, float(row[0]), float(row[1]), float(row[2]), float(row[3]), eng)
            else:
                mu_fit, _, _ = mu_lcdm_shape(zv, float(row[0]), float(row[1]), eng)
            mm = profiled_absolute_magnitude(mu_fit, mu_train, sig_train)
            mu_mod = mu_fit + mm
        elif blend:
            mu_mod, _ = mu_blend(zv, float(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4]), eng)
        else:
            mu_mod, _, _ = mu_lcdm(zv, float(row[0]), float(row[1]), float(row[2]), eng)
        syn = rng.normal(np.asarray(mu_mod, dtype=float), np.asarray(sig_train, dtype=float))
        sg = np.maximum(np.asarray(sig_train, dtype=float), eps())
        std_res = (syn - np.asarray(mu_mod, dtype=float)) / sg
        chi2s.append(float(np.sum(std_res**2)))
        rmses.append(float(math.sqrt(np.mean((syn - mu_train) ** 2))))
        coef = np.polyfit(zv, syn - np.asarray(mu_train, dtype=float), 1)
        slopes.append(float(coef[0]))
        sk_pool.append(float(stats.skew(std_res, bias=False)))
        ku_pool.append(float(stats.kurtosis(std_res, fisher=True, bias=False)))
    return dict(
        ppc_chi2_mean=float(np.mean(chi2s)),
        ppc_chi2_std=float(np.std(chi2s, ddof=1)) if len(chi2s) > 2 else float("nan"),
        ppc_rmse_mean=float(np.mean(rmses)),
        ppc_slope_resid_vs_z_mean_mag=float(np.mean(slopes)),
        ppc_slope_std=float(np.std(slopes, ddof=1)) if len(slopes) > 2 else float("nan"),
        ppc_std_residual_skew_mean=float(np.mean(sk_pool)),
        ppc_std_residual_kurtosis_excess_mean=float(np.mean(ku_pool)),
        ppc_draws_used=int(n_draws),
    )


def ppc_std_residual_chi2_replicas(
    flat: np.ndarray,
    rng: np.random.Generator,
    blend: bool,
    eng: CosmoInterpEngine,
    z_train: np.ndarray,
    mu_train: np.ndarray,
    sig_train: np.ndarray,
    n_draws: int = 640,
) -> np.ndarray:
    """Posterior draws of :math:`\\sum_i ((y^{\\mathrm{rep}}_i-\\mu_i)/\\sigma_i)^2` (diag PPC)."""
    n_draws = int(min(int(n_draws), int(flat.shape[0])))
    idx = rng.choice(int(flat.shape[0]), size=n_draws, replace=False)
    zv = np.asarray(z_train, dtype=float).reshape(-1)
    sg = np.maximum(np.asarray(sig_train, dtype=float), eps())
    chi2s: list[float] = []
    for row in flat[idx]:
        if infer_profile_m():
            if blend:
                mu_fit, _ = mu_blend_shape(zv, float(row[0]), float(row[1]), float(row[2]), float(row[3]), eng)
            else:
                mu_fit, _, _ = mu_lcdm_shape(zv, float(row[0]), float(row[1]), eng)
            mm = profiled_absolute_magnitude(mu_fit, mu_train, sig_train)
            mu_mod = np.asarray(mu_fit, dtype=float) + float(mm)
        elif blend:
            mu_mod, _ = mu_blend(zv, float(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4]), eng)
        else:
            mu_mod, _, _ = mu_lcdm(zv, float(row[0]), float(row[1]), float(row[2]), eng)
        syn = rng.normal(np.asarray(mu_mod, dtype=float), sg)
        std_res = (syn - np.asarray(mu_mod, dtype=float)) / sg
        chi2s.append(float(np.sum(std_res**2)))
    return np.asarray(chi2s, dtype=float)


def observed_training_std_residual_chi2(
    theta_full: np.ndarray,
    blend: bool,
    eng: CosmoInterpEngine,
    z_train: np.ndarray,
    mu_train: np.ndarray,
    sig_train: np.ndarray,
) -> float:
    """Same quadratic form as :func:`ppc_std_residual_chi2_replicas` but on real data vs model mean."""
    th = np.asarray(theta_full, dtype=float).reshape(-1)
    zv = np.asarray(z_train, dtype=float).reshape(-1)
    sg = np.maximum(np.asarray(sig_train, dtype=float), eps())
    if infer_profile_m():
        if blend:
            mu_fit, _ = mu_blend_shape(zv, float(th[0]), float(th[1]), float(th[2]), float(th[3]), eng)
        else:
            mu_fit, _, _ = mu_lcdm_shape(zv, float(th[0]), float(th[1]), eng)
        mm = profiled_absolute_magnitude(mu_fit, mu_train, sig_train)
        mu_mod = np.asarray(mu_fit, dtype=float) + float(mm)
    elif blend:
        mu_mod, _ = mu_blend(zv, float(th[0]), float(th[1]), float(th[2]), float(th[3]), float(th[4]), eng)
    else:
        mu_mod, _, _ = mu_lcdm(zv, float(th[0]), float(th[1]), float(th[2]), eng)
    std_res = (np.asarray(mu_train, dtype=float) - np.asarray(mu_mod, dtype=float)) / sg
    return float(np.sum(std_res**2))


def z_at_age_gyr(eng: CosmoInterpEngine, h0: float, om: float, t_targets_gyr: np.ndarray, z_hi: float) -> np.ndarray:
    """Invert monotonic cosmic age spline (coarse deterministic search)."""
    zfine = np.geomspace(ZP_LO - 1.0 + 1e-12, float(z_hi), max(640, Z_NODES // 10))
    tgrid = np.asarray(eng.cosmic_age_gyr(zfine, float(h0), float(om)), dtype=float)
    out = []
    tt = np.asarray(np.maximum(t_targets_gyr, 0.0), dtype=float).reshape(-1)
    for tv in tt:
        if tv <= 0.0:
            out.append(float(zfine[np.argmin(tgrid)]))
            continue
        j = int(np.searchsorted(np.flip(tgrid), tv, side="right"))
        j = np.clip(j, 1, tgrid.size - 1)
        z_lo, z_hi_ = float(zfine[-j]), float(zfine[-(j - 1)])
        out.append(0.5 * (z_lo + z_hi_))
    return np.asarray(out, dtype=float)




def isef_scientific_disclosure_block() -> dict[str, object]:
    return {
        "what_is_fitted": "Pantheon+ SH0ES distance moduli (training); DES Dovekie used only as held-out prediction check.",
        "what_is_derived": "Omega_Lambda from flatness; hold-out metrics; evidence estimates when nested sampling runs.",
        "what_is_assumed": "Flat FRW with standard radiation fraction; Gaussian sampling model for SN (diag or declared covariance).",
        "phenomenological_components": (
            "The horizon-area blend adds an effective Delta_mu term on top of flat LCDM distances. "
            "It is not presented as a fundamental derivation. Ablate with CWSF_USE_BLEND_HORIZON_DELTA_MU=0."
        ),
        "numerical_stabilization": (
            "Optional low-z luminosity anchor (CWSF_USE_LOWZ_DL_ANCHOR), logistic clamps, PCHIP distances, and blend mu sanity gate [20,50] mag — all documented in code and env."
        ),
        "never_mix_probes": (
            "No MOND/SPARC galaxy data enter the SN likelihood. Optional BAO / compressed CMB blocks are additive only when "
            "enabled via environment flags; they do not change SN sampler dimensions (fixed Ω_b h²)."
        ),
    }


def pearson_r_with_bootstrap(
    x: np.ndarray,
    y: np.ndarray,
    rng: np.random.Generator,
    n_boot: int,
    sample_unit_label: str,
) -> dict[str, object]:
    """Pearson r with two-sided p-value and bootstrap CI on r (nonparametric resampling of rows)."""
    xv = np.asarray(x, dtype=float).reshape(-1)
    yv = np.asarray(y, dtype=float).reshape(-1)
    m = np.isfinite(xv) & np.isfinite(yv)
    xv, yv = xv[m], yv[m]
    n = int(xv.size)
    if n < 8:
        return dict(status="insufficient_joint_samples", n=int(n), sample_unit=sample_unit_label)
    r_obs, p_obs = stats.pearsonr(xv, yv)
    idx = np.arange(n)
    n_boot = int(max(200, min(n_boot, 20000)))
    rs: list[float] = []
    for _ in range(n_boot):
        j = rng.choice(idx, size=n, replace=True)
        ri, _ = stats.pearsonr(xv[j], yv[j])
        rs.append(float(ri))
    rs_a = np.asarray(rs, dtype=float)
    lo, hi = float(np.quantile(rs_a, 0.025)), float(np.quantile(rs_a, 0.975))
    return dict(
        pearson_r=float(r_obs),
        p_value_two_sided=float(p_obs),
        n_joint_samples=int(n),
        bootstrap_replicates=int(n_boot),
        bootstrap_r_quantile_95=[json_safe_float(lo), json_safe_float(hi)],
        sample_unit=sample_unit_label,
    )


def _blend_predict_train_diag(row: np.ndarray, zt: np.ndarray, mt: np.ndarray, st: np.ndarray, eng: CosmoInterpEngine) -> np.ndarray:
    if infer_profile_m():
        mu_fit, _ = mu_blend_shape(zt, float(row[0]), float(row[1]), float(row[2]), float(row[3]), eng)
        return np.asarray(mu_fit, dtype=float) + float(profiled_absolute_magnitude(mu_fit, mt, st))
    return np.asarray(mu_blend(zt, float(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4]), eng)[0], dtype=float)


def build_isef_pearson_table(
    flat_b: np.ndarray,
    zt: np.ndarray,
    mt: np.ndarray,
    st: np.ndarray,
    med_l: np.ndarray,
    med_b: np.ndarray,
    eng: CosmoInterpEngine,
    rng: np.random.Generator,
    z_ref: float,
) -> pd.DataFrame:
    """Predeclared correlation panel; **explicitly labels** whether n is SN rows or posterior draws."""
    rows: list[dict[str, object]] = []
    n_cap = min(int(ISEF_POSTERIOR_SUBSAMPLE_CORR), int(flat_b.shape[0]))
    if n_cap >= 12:
        ix = rng.choice(int(flat_b.shape[0]), size=n_cap, replace=False)
        z1 = np.asarray([float(z_ref)], dtype=float)
        h0s: list[float] = []
        oms: list[float] = []
        tcs: list[float] = []
        ks: list[float] = []
        dmus: list[float] = []
        kmags: list[float] = []
        for row in np.asarray(flat_b[ix], dtype=float):
            ml, _, _ = mu_lcdm_shape(z1, float(row[0]), float(row[1]), eng)
            mb, _ = mu_blend_shape(z1, float(row[0]), float(row[1]), float(row[2]), float(row[3]), eng)
            if not (math.isfinite(float(ml[0])) and math.isfinite(float(mb[0]))):
                continue
            dmu = float(mb[0] - ml[0])
            dmus.append(dmu)
            h0s.append(float(row[0]))
            oms.append(float(row[1]))
            tcs.append(float(row[2]))
            ks.append(float(row[3]))
            pred = _blend_predict_train_diag(row, zt, mt, st, eng)
            kmags.append(float(np.mean(np.abs(mt - pred))))
        h0a, dmu_a = np.asarray(h0s), np.asarray(dmus)
        tca, dmu_b = np.asarray(tcs), np.asarray(dmus)
        ka, kma = np.asarray(ks), np.asarray(kmags)
        rows.append(
            dict(
                relationship=f"H0_vs_DeltaMu_shape_at_zref_{z_ref:.3f}",
                **pearson_r_with_bootstrap(
                    h0a, dmu_a, rng, ISEF_PEARSON_BOOTSTRAP, "posterior_draws_subsample"
                ),
            )
        )
        rows.append(
            dict(
                relationship=f"t_crit_vs_DeltaMu_shape_at_zref_{z_ref:.3f}",
                **pearson_r_with_bootstrap(
                    tca, dmu_b, rng, ISEF_PEARSON_BOOTSTRAP, "posterior_draws_subsample"
                ),
            )
        )
        rows.append(
            dict(
                relationship="k_vs_mean_abs_training_residual_blend_per_draw",
                **pearson_r_with_bootstrap(ka, kma, rng, ISEF_PEARSON_BOOTSTRAP, "posterior_draws_subsample"),
            )
        )

    mhat_b = _blend_predict_train_diag(np.asarray(med_b, dtype=float), zt, mt, st, eng)
    res_b = np.asarray(mt - mhat_b, dtype=float)
    rows.append(
        dict(
            relationship="training_residual_blend_vs_redshift_at_posterior_median_theta",
            **pearson_r_with_bootstrap(np.asarray(zt, dtype=float), res_b, rng, ISEF_PEARSON_BOOTSTRAP, "observed_sn_rows"),
        )
    )
    mhat_l, _, _ = mu_lcdm(zt, float(med_l[0]), float(med_l[1]), float(med_l[-1]), eng)
    res_l = np.asarray(mt - np.asarray(mhat_l, dtype=float), dtype=float)
    rows.append(
        dict(
            relationship="training_residual_lcdm_vs_redshift_at_lcdm_posterior_median_theta",
            **pearson_r_with_bootstrap(np.asarray(zt, dtype=float), res_l, rng, ISEF_PEARSON_BOOTSTRAP, "observed_sn_rows"),
        )
    )
    return pd.DataFrame(rows)


def export_isef_packaged_artifacts(ctx: dict[str, object]) -> dict[str, object]:
    """Write ISEF-style machine-readable tables/JSON and documentation files (no fabricated statistics)."""
    OUT = Path(str(ctx["OUTDIR"]))
    FIG = Path(str(ctx["FIGDIR"]))
    rng = ctx["RNG"]  # type: ignore[assignment]
    rows_written: list[str] = []

    clean_train_src = OUT / "pantheon_clean.csv"
    clean_hold_src = OUT / "des_clean.csv"
    if not clean_train_src.exists() and isinstance(ctx.get("pant_fallback"), (str, Path)):
        clean_train_src = Path(str(ctx["pant_fallback"]))
    if not clean_hold_src.exists() and isinstance(ctx.get("des_fallback"), (str, Path)):
        clean_hold_src = Path(str(ctx["des_fallback"]))
    shutil.copy2(clean_train_src, OUT / "cleaned_training.csv")
    shutil.copy2(clean_hold_src, OUT / "cleaned_holdout.csv")
    rows_written.extend(["cleaned_training.csv", "cleaned_holdout.csv"])

    if (OUT / "mcmc_chain_lcdm.csv").exists():
        shutil.copy2(OUT / "mcmc_chain_lcdm.csv", OUT / "chain_lcdm.csv")
        rows_written.append("chain_lcdm.csv")
    if (OUT / "mcmc_chain_blend.csv").exists():
        shutil.copy2(OUT / "mcmc_chain_blend.csv", OUT / "chain_blend.csv")
        rows_written.append("chain_blend.csv")

    flat_l = np.asarray(ctx["flat_l"], dtype=float)
    flat_b = np.asarray(ctx["flat_b"], dtype=float)
    names_l = list(ctx["names_l_cols"])  # type: ignore[arg-type]
    names_b = list(ctx["names_b_cols"])  # type: ignore[arg-type]
    np.savez_compressed(
        OUT / "chains_emcee.npz",
        lcdm_samples=flat_l,
        blend_samples=flat_b,
        column_names_lcdm=np.array(names_l, dtype=object),
        column_names_blend=np.array(names_b, dtype=object),
        note=np.array(
            "Arrays are emcee posterior samples; counts are posterior rows, not independent astronomical objects.",
            dtype=object,
        ),
    )
    rows_written.append("chains_emcee.npz")

    q = [0.025, 0.16, 0.5, 0.84, 0.975]
    post_sum = dict(
        lcdm={names_l[j]: dict(median=float(np.median(flat_l[:, j]))) for j in range(flat_l.shape[1])},
        blend={names_b[j]: dict(median=float(np.median(flat_b[:, j]))) for j in range(flat_b.shape[1])},
        quantiles_requested=q,
        n_posterior_rows_lcdm=int(flat_l.shape[0]),
        n_posterior_rows_blend=int(flat_b.shape[0]),
        disclosure=isef_scientific_disclosure_block(),
    )
    for j in range(flat_l.shape[1]):
        post_sum["lcdm"][names_l[j]]["quantiles"] = [float(x) for x in np.quantile(flat_l[:, j], q)]
    for j in range(flat_b.shape[1]):
        post_sum["blend"][names_b[j]]["quantiles"] = [float(x) for x in np.quantile(flat_b[:, j], q)]
    (OUT / "posterior_summaries.json").write_text(json.dumps(post_sum, indent=2, allow_nan=False), encoding="utf-8")
    rows_written.append("posterior_summaries.json")

    ns = ctx.get("nested_summary", {})
    dlzstderr = None
    if isinstance(ns, dict) and ns.get("ran") is True:
        try:
            dlzstderr = json_safe_float(
                float(math.hypot(float(ns["lcdm"]["logzerr"]), float(ns["blend"]["logzerr"])))
            )
        except Exception:
            dlzstderr = None
    ev = dict(
        nested_sampling_report=ns if isinstance(ns, dict) else {},
        delta_lnZ_blend_minus_lcdm_stderr_propagated_independent_approx=dlzstderr,
        log_bayes_factor_blend_minus_lcdm=(
            json_safe_float(float(ns["delta_logz_blend_minus_lcdm"]))
            if isinstance(ns, dict)
            and ns.get("ran") is True
            and ns.get("delta_logz_blend_minus_lcdm") is not None
            and math.isfinite(float(ns["delta_logz_blend_minus_lcdm"]))
            else None
        ),
        bayes_factor_blend_over_lcdm_linear=(
            (lambda dz: (float(math.exp(dz)) if abs(dz) < 698.0 else None))(float(ns["delta_logz_blend_minus_lcdm"]))
            if isinstance(ns, dict)
            and ns.get("ran") is True
            and ns.get("delta_logz_blend_minus_lcdm") is not None
            and math.isfinite(float(ns["delta_logz_blend_minus_lcdm"]))
            else None
        ),
        warning=(
            "Bayes factors use dynesty marginal log-evidence summaries; uncertainties on lnZ per model are propagated into "
            "delta_lnZ stderr only under an independence heuristic. Compare models only on identical likelihoods/data splits. "
            "The numeric Bayes-factor field is omitted (None) when exp(|Δ ln Z|) would overflow IEEE doubles—use log_bayes_factor_blend_minus_lcdm."
        ),
    )
    (OUT / "evidence_and_bayes_factors.json").write_text(json.dumps(ev, indent=2, allow_nan=False), encoding="utf-8")
    rows_written.append("evidence_and_bayes_factors.json")

    sens = dict(
        this_run_configuration=dict(
            CWSF_USE_LOWZ_DL_ANCHOR=USE_LOWZ_DL_ANCHOR,
            CWSF_USE_BLEND_HORIZON_DELTA_MU=USE_BLEND_HORIZON_DELTA_MU,
            CWSF_USE_COV_requested=USE_COVARIANCE,
            CWSF_USE_COV_active_in_likelihood=bool(ctx.get("_use_cov_eff")),
            CWSF_PROFILE_M_env=bool(PROFILE_M),
            analytic_profiling_on_diagonal_requested=bool(ctx.get("infer_profile_like")),
            CWSF_KEEP_CALIBRATORS=os.getenv("CWSF_KEEP_CALIBRATORS", "0"),
            N_FRAMEWORK_SEEDS=int(ctx.get("N_FRAMEWORK_SEEDS", 0)),
            RNG_SEED=int(ctx.get("RNG_SEED", 0)),
        ),
        robustness_block=ctx.get("robustness_summary", {}),
        cross_validation_block=ctx.get("cv_summary", {}),
        ablation_matrix_note=(
            "Full ablation grid requires separate runs: toggle CWSF_USE_LOWZ_DL_ANCHOR, CWSF_USE_BLEND_HORIZON_DELTA_MU, "
            "CWSF_USE_COV (+ optional CWSF_PANTHEON_COV_URL), CWSF_PROFILE_M (effective only when diagonal training likelihood is active), "
            "CWSF_KEEP_CALIBRATORS, CWSF_LEGACY_PRIORS; "
            "compare outputs across runs — this file records only the present configuration."
        ),
    )
    (OUT / "sensitivity_report.json").write_text(json.dumps(sens, indent=2, allow_nan=False), encoding="utf-8")
    rows_written.append("sensitivity_report.json")

    pear_df = build_isef_pearson_table(
        flat_b,
        np.asarray(ctx["zt"], dtype=float),
        np.asarray(ctx["mt"], dtype=float),
        np.asarray(ctx["st"], dtype=float),
        np.asarray(ctx["med_l"], dtype=float),
        np.asarray(ctx["med_b"], dtype=float),
        ctx["eng"],  # type: ignore[arg-type]
        rng,
        float(ctx.get("Z_REF", Z_REF_DELTA_MU_CORR)),
    )
    pear_df.to_csv(OUT / "pearson_correlations.csv", index=False)
    rows_written.append("pearson_correlations.csv")

    n_tr = int(ctx["n_train"])
    dof_l = float(n_tr - float(ctx["k_l"]))
    dof_b = float(n_tr - float(ctx["k_b"]))
    mc_rows = [
        dict(
            model="flat_lcdm",
            chi2_training=float(ctx["chi2_med_l"]),
            reduced_chi2=float(ctx["redchi_l"]),
            rmse_des=float(ctx["des_l_med"]["rmse"]),
            aic=float(ctx["aic_l"]),
            bic=float(ctx["bic_l"]),
            waic=float(ctx["waic_l"]),
            n_sn_training=n_tr,
            k_params=int(ctx["k_l"]),
            dof=float(dof_l),
            max_loglike_train=float(ctx["maxll_l"]),
        ),
        dict(
            model="horizon_area_blend_lcdm_geom",
            chi2_training=float(ctx["chi2_med_b"]),
            reduced_chi2=float(ctx["redchi_b"]),
            rmse_des=float(ctx["des_b_med"]["rmse"]),
            aic=float(ctx["aic_b"]),
            bic=float(ctx["bic_b"]),
            waic=float(ctx["waic_b"]),
            n_sn_training=n_tr,
            k_params=int(ctx["k_b"]),
            dof=float(dof_b),
            max_loglike_train=float(ctx["maxll_b"]),
        ),
        dict(
            model="wcdm_optional_extension",
            status="NOT_IMPLEMENTED_IN_THIS_REPOSITORY",
            note="Flat wCDM requires an additional equation-of-state parameter and d_L quadrature beyond this file's scope.",
        ),
    ]
    pd.DataFrame(mc_rows).to_csv(OUT / "model_comparison_table.csv", index=False)
    rows_written.append("model_comparison_table.csv")

    des = ctx["des"]  # type: ignore[assignment]
    z_des = np.asarray(des["z"], dtype=float)
    m_des = np.asarray(des["mu"], dtype=float)
    s_des = np.asarray(des["sig"], dtype=float)
    ml_d, _, _ = mu_lcdm(z_des, float(ctx["med_l"][0]), float(ctx["med_l"][1]), float(ctx["med_l"][-1]), ctx["eng"])  # type: ignore[arg-type]
    mb_d, _ = mu_blend(
        z_des,
        float(ctx["med_b"][0]),
        float(ctx["med_b"][1]),
        float(ctx["med_b"][2]),
        float(ctx["med_b"][3]),
        float(ctx["med_b"][-1]),
        ctx["eng"],  # type: ignore[arg-type]
    )
    hold = pd.DataFrame(
        dict(
            z=z_des,
            mu_obs=m_des,
            sig=s_des,
            mu_pred_lcdm_median_theta=np.asarray(ml_d, dtype=float),
            mu_pred_blend_median_theta=np.asarray(mb_d, dtype=float),
            residual_lcdm=m_des - np.asarray(ml_d, dtype=float),
            residual_blend=m_des - np.asarray(mb_d, dtype=float),
        )
    )
    hold.to_csv(OUT / "holdout_predictions.csv", index=False)
    rows_written.append("holdout_predictions.csv")

    ppc_rows = []
    pcd = ctx.get("ppc_chi2_diag", {})
    if isinstance(pcd, dict):
        ppc_rows.append(dict(check="ppc_chi2_std_lcdm", **{k: v for k, v in pcd.get("lcdm", {}).items() if k != "key"}))
        ppc_rows.append(dict(check="ppc_chi2_std_blend", **{k: v for k, v in pcd.get("blend", {}).items() if k != "key"}))
    pd.DataFrame(ppc_rows).to_csv(OUT / "posterior_predictive_checks.csv", index=False)
    rows_written.append("posterior_predictive_checks.csv")

    ext_ref = dict(
        pantheon_plus_sh0es_repo="https://github.com/PantheonPlusSH0ES/DataRelease",
        pantheon_file_used=str(ctx.get("PANT_URL", "")),
        pantheon_covariance_stat_sys_official=PANT_COV_STATSYS_URL,
        pantheon_covariance_active_download=str(ctx.get("PANT_COV_ACTIVE", "")),
        des_sn5yr_repo="https://github.com/des-science/DES-SN5YR",
        des_file_used=str(ctx.get("DES_URL", "")),
        planck_portal="https://www.cosmos.esa.int/web/planck/pla",
        planck_ancillary="https://irsa.ipac.caltech.edu/data/Planck/release_3/ancillary-data/",
        sdss_portal="https://www.sdss.org/",
        eboss_bao_only="https://svn.sdss.org/public/data/eboss/DR16cosmo/tags/v1_0_0/likelihoods/BAO-only/",
        eboss_bao_plus="https://svn.sdss.org/public/data/eboss/DR16cosmo/tags/v1_0_0/likelihoods/BAO-plus/",
        sparc_mond_benchmark="https://astroweb.case.edu/SPARC/",
        disclosure="Planck/BAO/SPARC URLs are reference-only for fair science-fair documentation; this run's likelihood uses Pantheon+ (train) and DES (hold-out) unless you extend the code.",
    )
    (OUT / "external_data_references.json").write_text(json.dumps(ext_ref, indent=2), encoding="utf-8")
    rows_written.append("external_data_references.json")

    mond_stub = dict(
        status="SEPARATE_BENCHMARK_ONLY",
        message=(
            "MOND/SPARC rotation-curve tests run only when CWSF_SPARC_DIR is set; they never enter the SN+BAO+CMB log-posterior. "
            "Treat as a cross-scale narrative figure, not a joint likelihood."
        ),
        url="https://astroweb.case.edu/SPARC/",
    )
    (OUT / "mond_sparc_benchmark_stub.json").write_text(json.dumps(mond_stub, indent=2), encoding="utf-8")
    rows_written.append("mond_sparc_benchmark_stub.json")

    plt.figure(figsize=(8.5, 4.8))
    labs = ["LCDM\nAIC", "Blend\nAIC", "LCDM\nBIC", "Blend\nBIC", "LCDM\nWAIC", "Blend\nWAIC"]
    vals = [float(ctx["aic_l"]), float(ctx["aic_b"]), float(ctx["bic_l"]), float(ctx["bic_b"]), float(ctx["waic_l"]), float(ctx["waic_b"])]
    plt.bar(labs, vals, color=["C0", "C1", "C0", "C1", "C0", "C1"], alpha=0.82)
    plt.ylabel("information criterion (lower is better)")
    plt.title("Model comparison (same training SN likelihood family; not physical evidence alone)")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(FIG / "model_comparison_ic_bars.png", dpi=200)
    plt.close()
    rows_written.append(str((FIG / "model_comparison_ic_bars.png").resolve()))

    rob_csv = OUT / "robustness" / "robustness_scenarios.csv"
    if rob_csv.exists():
        try:
            rdf = pd.read_csv(rob_csv)
            if len(rdf) > 0 and "H0_med_blend" in rdf.columns:
                plt.figure(figsize=(8.0, 4.5))
                plt.plot(range(len(rdf)), rdf["H0_med_blend"], "o-", label=r"$H_0$ blend")
                plt.plot(range(len(rdf)), rdf["H0_med_lcdm"], "s-", label=r"$H_0$ LCDM")
                plt.xticks(range(len(rdf)), rdf["scenario"], rotation=35, ha="right")
                plt.ylabel(r"$H_0$ km s$^{-1}$ Mpc$^{-1}$")
                plt.title("Robustness scenarios (short MCMC; see sensitivity_report.json)")
                plt.legend()
                plt.grid(alpha=0.25)
                plt.tight_layout()
                plt.savefig(FIG / "sensitivity_robustness_H0_panel.png", dpi=200)
                plt.close()
                rows_written.append(str((FIG / "sensitivity_robustness_H0_panel.png").resolve()))
        except Exception:
            pass

    caveat_path = OUT / "CAVEATS_AND_LIMITATIONS.txt"
    caveat_path.write_text(
        "\n".join(
            [
                "CAVEATS (auto-generated — read before claiming discovery)",
                "- Horizon blend is explicitly phenomenological on top of LCDM distances unless you publish a derivation.",
                "- Evidence ratios need nested sampling to have run successfully; check nested/ and evidence_and_bayes_factors.json.",
                "- Pearson rows labeled posterior_draws_subsample refer to MARKOV CHAIN ROWS, not independent SN observations.",
                "- Comparison to LCDM is fair only when likelihood, dataset splits, and covariance mode match across models.",
                "- DES hold-out predictions profile M using Pantheon only (no leakage) — RMSE differences are illustrative, not cosmic proof.",
                "- wCDM and BAO/Planck joint inference are NOT implemented here by default.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    methods_path = OUT / "METHODS_SUMMARY.txt"
    methods_path.write_text(
        "\n".join(
            [
                "METHODS SUMMARY",
                "- Training: Pantheon+ SH0ES cleaned table; hold-out DES Dovekie cosmology-format file.",
                "- Models: flat LCDM vs horizon-area modulus blend sharing the same LCDM background (flatness enforced).",
                "- Inference: emcee affine MCMC (+ optional dynesty marginal likelihood); optional Pantheon covariance.",
                f"- Numerical stabilization toggles this run: LOWZ_DL_ANCHOR={USE_LOWZ_DL_ANCHOR}, BLEND_DELTA_MU={USE_BLEND_HORIZON_DELTA_MU}.",
                "- Outputs: see summary.json plus ISEF bundle file list recorded in summary['isef_export_bundle'].",
                "",
            ]
        ),
        encoding="utf-8",
    )
    results_path = OUT / "RESULTS_SUMMARY.txt"
    results_path.write_text(
        "\n".join(
            [
                "RESULTS SUMMARY (numbers also in summary.json and CSV tables)",
                f"- Pantheon SN rows used in Gaussian likelihood path: {n_tr} (labeled n_sn_training in CSVs)",
                f"- Posterior samples on disk (lcdm): {flat_l.shape[0]}, (blend): {flat_b.shape[0]} posterior draws.",
                f"- ΔAIC (blend − LCDM) training proxy: {float(ctx['aic_b']) - float(ctx['aic_l']):.4f}",
                f"- ΔBIC: {float(ctx['bic_b']) - float(ctx['bic_l']):.4f}",
                f"- DES RMSE lcdm median θ: {float(ctx['des_l_med']['rmse']):.5f}; blend: {float(ctx['des_b_med']['rmse']):.5f} mag",
                "",
            ]
        ),
        encoding="utf-8",
    )
    rows_written.extend(
        ["CAVEATS_AND_LIMITATIONS.txt", "METHODS_SUMMARY.txt", "RESULTS_SUMMARY.txt"]
    )

    readme = OUT / "README_ISEF.md"
    if not readme.exists():
        readme.write_text(
            "\n".join(
                [
                    "# CWSF cosmology pipeline — ISEF / reproducibility layer",
                    "",
                    "## Run",
                    "```bash",
                    "pip install numpy scipy pandas matplotlib emcee dynesty pyarrow corner  # subset as needed",
                    "python cwsf_pipeline.py",
                    "```",
                    "",
                    "## Outputs",
                    "See `cwsf_output/` (or `CWSF_OUTDIR`) for `cleaned_*.csv`, `chain_*.csv`, `*_summaries.json`, `pearson_correlations.csv`, `model_comparison_table.csv`, figures.",
                    "",
                    "## Ablation switches",
                    "- `CWSF_USE_LOWZ_DL_ANCHOR=0` — disable Hubbles normalization on `d_L` ladder.",
                    "- `CWSF_USE_BLEND_HORIZON_DELTA_MU=0` — disable phenomenological modulus excess (blend collapses to LCDM μ).",
                    "- `CWSF_USE_COV=1` — Pantheon covariance (URL via `CWSF_PANTHEON_COV_URL`; STAT+SYS official URL exported in JSON).",
                    "",
                    "## What this is NOT",
                    "Not a substitute for Boltzmann codes, not joint BAO/Pantheon/Chandra evidence without explicit likelihood code, "
                    "and not a fundamental proof of horizon thermodynamics.",
                    "",
                ]
            ),
            encoding="utf-8",
        )

    js = OUT / "isef_bundle_index.json"
    js.write_text(json.dumps(dict(written_relative_to_outdir=sorted(rows_written)), indent=2), encoding="utf-8")
    return dict(files=rows_written, pearson_preview_rows=int(len(pear_df)))


def stacking_weights_from_waic(waic_lcdm: float, waic_blend: float) -> tuple[float, float]:
    b = np.array([-0.5 * float(waic_lcdm), -0.5 * float(waic_blend)], dtype=float)
    b = b - np.max(b)
    w = np.exp(b)
    s = float(np.sum(w))
    return float(w[0] / max(s, eps())), float(w[1] / max(s, eps()))


def unit_cube_to_truncnorm_phys(
    u: np.ndarray, lows: np.ndarray, highs: np.ndarray, mus: np.ndarray, sigs: np.ndarray
) -> np.ndarray:
    """Map i.i.d. Uniform(0,1) draws to independent truncated Gaussians (dynesty prior transform)."""
    uu = np.clip(np.asarray(u, dtype=float), eps(), 1.0 - eps())
    lo = np.asarray(lows, dtype=float).reshape(-1)
    hi = np.asarray(highs, dtype=float).reshape(-1)
    mu = np.asarray(mus, dtype=float).reshape(-1)
    sg = np.maximum(np.asarray(sigs, dtype=float).reshape(-1), eps())
    aa = (lo - mu) / sg
    bb = (hi - mu) / sg
    return np.asarray(stats.truncnorm.ppf(uu, aa, bb, loc=mu, scale=sg), dtype=float)


def jeffreys_label_for_delta_logz(delta_ln_z: float) -> str:
    r"""Qualitative strength for Jeffrey's-inspired evidence bands on :math:`|\Delta\ln Z|` (natural log)."""
    ad = abs(float(delta_ln_z))
    if ad < 1.0:
        return "not worth more than a bare mention"
    if ad < 3.0:
        return "weak evidence"
    if ad < 6.0:
        return "moderate evidence"
    if ad < 10.0:
        return "strong evidence"
    return "decisive evidence"


def load_pantheon_cov_subset(path_cov: Path, pant_df: pd.DataFrame, path_pantheon_dat: Path) -> tuple[np.ndarray, list[str]]:
    """Subset Pantheon covariance to rows of ``pant_df`` preserving catalogue order."""
    full = pd.read_csv(path_pantheon_dat, sep=r"\s+", engine="python")
    if "CID" not in full.columns:
        raise KeyError("CID missing in Pantheon file for covariance alignment")
    if "IS_CALIBRATOR" in full.columns and os.getenv("CWSF_KEEP_CALIBRATORS", "0") != "1":
        cc = pd.to_numeric(full["IS_CALIBRATOR"], errors="coerce").fillna(0).astype(int)
        full = full.loc[cc == 0]
    full = full.drop_duplicates(subset=["CID"], keep="first")
    cov_raw = np.loadtxt(path_cov, comments="#")
    if cov_raw.ndim == 1:
        Sig = np.diag(np.square(cov_raw))
    elif cov_raw.shape[0] == cov_raw.shape[1]:
        Sig = np.asarray(cov_raw, dtype=float)
    else:
        raise ValueError(f"unexpected covariance matrix shape {cov_raw.shape}")
    n_tot = Sig.shape[0]
    cid_list_full = full["CID"].astype(str).tolist()
    idx_map = {cid_list_full[ii]: ii for ii in range(min(len(cid_list_full), n_tot))}
    pant_cids = pant_df["CID"].astype(str).tolist()
    ix: list[int] = []
    miss: list[str] = []
    for cid in pant_cids:
        if cid in idx_map:
            ix.append(int(idx_map[cid]))
        else:
            miss.append(cid)
    if len(ix) != len(pant_cids) or miss:
        raise ValueError(f"covariance CID alignment failed ({len(miss)} missing)")
    Sig_sub = Sig[np.ix_(ix, ix)] + np.eye(len(ix)) * 1e-12
    return Sig_sub.astype(float), pant_cids


def lnlike_mvn_residual(r: np.ndarray, chol_fac: np.ndarray, log_det_2pi: float) -> float:
    ys = la_sci.solve_triangular(chol_fac, r.reshape(-1, 1), lower=True).ravel()
    return float(-0.5 * float(np.dot(ys, ys) + log_det_2pi))


def ensemble_emcee_for_model(
    lnpost_factory: Callable[[], Callable[[np.ndarray], float]],
    ndim: int,
    lows: np.ndarray,
    highs: np.ndarray,
    centers: np.ndarray,
    widths: np.ndarray,
    base_seed: int,
    p0_jitter_frac: float = 0.0,
    *,
    burn_steps: int | None = None,
    prod_steps: int | None = None,
    n_walkers_ov: int | None = None,
    n_chains_ov: int | None = None,
) -> tuple[np.ndarray, list[np.ndarray], float]:
    flats_m: list[np.ndarray] = []
    ch_list: list[np.ndarray] = []
    acc_pack: list[float] = []
    n_cr = max(int(N_CHAINS if n_chains_ov is None else n_chains_ov), 1)
    seeds_tup = tuple(int(base_seed + 7919 * c) for c in range(n_cr))
    lo = lows[:ndim]
    hi = highs[:ndim]
    span = np.asarray(hi - lo, dtype=float).reshape(1, -1)
    for ci in range(n_cr):
        rng_c = np.random.default_rng(seeds_tup[ci])
        lnpost = lnpost_factory()
        nw = int(N_WALKERS if n_walkers_ov is None else n_walkers_ov)
        p0_use = init_walkers(nw, centers[:ndim], widths[:ndim], lows[:ndim], highs[:ndim], rng_c)
        if float(p0_jitter_frac) > 0:
            p0_use = np.clip(
                p0_use + float(p0_jitter_frac) * span * rng_c.standard_normal(p0_use.shape),
                lows[:ndim],
                highs[:ndim],
            )

        _, ch3_m, flat_m, _, acc_m = run_emcee(ndim, nw, burn_steps, prod_steps, lnpost, p0_use)
        flats_m.append(np.asarray(flat_m, dtype=float))
        ch_list.append(np.asarray(ch3_m, dtype=float))
        acc_pack.append(float(acc_m))
    flat_a = np.vstack(flats_m)
    return flat_a, ch_list, float(np.mean(acc_pack))


def binned_residual_summary(
    zv: np.ndarray,
    res: np.ndarray,
    sg: np.ndarray,
    n_bins: int = 4,
) -> pd.DataFrame:
    zv = np.asarray(zv, dtype=float)
    edges = np.quantile(zv, np.linspace(0.0, 1.0, int(n_bins) + 1))
    rows: list[dict[str, float]] = []
    for i in range(int(n_bins)):
        lo_, hi_ = float(edges[i]), float(edges[i + 1])
        m = (zv >= lo_) & (zv <= hi_) if i == int(n_bins) - 1 else (zv >= lo_) & (zv < hi_)
        if not np.any(m):
            continue
        wg = float(np.mean(res[m]))
        rows.append(dict(z_lo=lo_, z_hi=hi_, z_mid=float(np.mean(zv[m])), n=int(np.sum(m)), mean_residual=wg))
    return pd.DataFrame(rows)


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    FIGDIR.mkdir(parents=True, exist_ok=True)
    pant_path = OUTDIR / "Pantheon+SH0ES.dat"
    des_path = OUTDIR / "DES-Dovekie_HD.csv"
    ensure_download(PANT_URL, pant_path)
    ensure_download(DES_URL, des_path)

    pant = load_pantheon(pant_path)
    des = load_des(des_path)
    pant.to_csv(OUTDIR / "pantheon_clean.csv", index=False)
    des.to_csv(OUTDIR / "des_clean.csv", index=False)

    zt = np.asarray(pant["z"], dtype=float)
    mt = np.asarray(pant["mu"], dtype=float)
    st = np.asarray(pant["sig"], dtype=float)
    n_train = int(zt.size)

    z_des = np.asarray(des["z"], dtype=float)
    m_des = np.asarray(des["mu"], dtype=float)
    s_des = np.asarray(des["sig"], dtype=float)

    z_plot_max = float(max(np.max(zt), np.max(z_des)))
    zp_hi_dm = float((z_plot_max + 1.0) * 1.02)
    eng = CosmoInterpEngine(zp_hi_dm, z_plot_max)

    if SELFTEST_AT_BOOT:
        try:
            (OUTDIR / "tables").mkdir(parents=True, exist_ok=True)
            run_cosmology_engine_selftest_rows(eng, 72.0, 0.3).to_csv(
                OUTDIR / "tables" / "engine_selftest_at_start.csv", index=False
            )
            if blend_physics_is_ccomplet2():
                run_c2_physics_selftest_rows(eng, 72.8, 0.30, 15.9, 0.37).to_csv(
                    OUTDIR / "tables" / "c2_physics_selftest_at_start.csv", index=False
                )
                st2 = pd.read_csv(OUTDIR / "tables" / "c2_physics_selftest_at_start.csv")
                if not bool(np.all(st2["pass_"].to_numpy(dtype=bool))):
                    raise RuntimeError("c2_physics_selftest failed: " + st2.to_string())
        except Exception as ex_st:
            print("[selftest] engine self-test skipped:", ex_st)

    global _EXTERNAL_JOINT_PACK
    external_like_warnings: list[str] = []
    if HAVE_EXTERNAL_LIKES:
        try:
            copy_default_likelihood_templates(OUTDIR)
            _EXTERNAL_JOINT_PACK = build_external_joint_pack(OUTDIR)
        except Exception as ex_ext:
            _EXTERNAL_JOINT_PACK = None
            external_like_warnings.append(f"external_joint_pack_disabled:{ex_ext!s}")
    else:
        _EXTERNAL_JOINT_PACK = None

    _use_cov_eff = USE_COVARIANCE
    cov_warnings: list[str] = []
    chol_sig = None
    pant_sig_full: np.ndarray | None = None
    logdet_twopisigma = float("nan")
    if USE_COVARIANCE:
        cp = OUTDIR / "Pantheon_STATONLY_cov.txt"
        try:
            ensure_download(PANT_COVSTAT_URL, cp)
            _Sig, _cids_ck = load_pantheon_cov_subset(cp, pant, pant_path)
            pant_sig_full = np.asarray(_Sig, dtype=float).copy()
            sign_l, slogdet_abs = np.linalg.slogdet(_Sig)
            if sign_l <= 0:
                raise ValueError("covariance non-positive definite")
            logdet_twopisigma = float(n_train * math.log(2.0 * math.pi) + slogdet_abs)
            chol_sig = la_sci.cholesky(_Sig, lower=True)
        except Exception as ex:
            _use_cov_eff = False
            pant_sig_full = None
            cov_warnings.append(f"Covariance disabled after load failure: {ex!s}")

    nd_l = 2 if infer_profile_m() else 3
    nd_b = 4 if infer_profile_m() else 5
    k_l = nd_l
    k_b = nd_b
    if _use_cov_eff:
        nd_l, nd_b, k_l, k_b = 3, 5, 3, 5

    lcdm_centers_full = np.array([PRI_MU["H0"], PRI_MU["Om"], PRI_MAG_MU], dtype=float)
    lcdm_widths_full = np.array([PRI_SG["H0"], PRI_SG["Om"], PRI_MAG_SG], dtype=float) * 0.25
    lcdm_lows_full = np.array([BND["H0"][0], BND["Om"][0], BND["M"][0]], dtype=float)
    lcdm_highs_full = np.array([BND["H0"][1], BND["Om"][1], BND["M"][1]], dtype=float)
    lcdm_centers = lcdm_centers_full[: nd_l ]
    lcdm_widths = lcdm_widths_full[: nd_l ]
    lcdm_lows = lcdm_lows_full[: nd_l ]
    lcdm_highs = lcdm_highs_full[: nd_l ]

    blend_lows_full = np.array([BND["H0"][0], BND["Om"][0], BND["tc"][0], BND["k"][0], BND["M"][0]], dtype=float)
    blend_highs_full = np.array([BND["H0"][1], BND["Om"][1], BND["tc"][1], BND["k"][1], BND["M"][1]], dtype=float)
    blend_lows = blend_lows_full[: nd_b ]
    blend_highs = blend_highs_full[: nd_b ]

    floors_b = np.array([0.45, 0.035, PRI_SG["tc"] * 0.5, PRI_SG["k"] * 0.5], dtype=float)
    if infer_profile_m():
        blend_centers_adapt = np.array([PRI_MU["H0"], PRI_MU["Om"], PRI_MU["tc"], PRI_MU["k"]], dtype=float)
        blend_widths_adapt = np.array(
            [PRI_SG["H0"] * 0.35, PRI_SG["Om"] * 0.35, PRI_SG["tc"] * 0.5, PRI_SG["k"] * 0.5], dtype=float
        )
    else:
        blend_centers_adapt = np.array([PRI_MU["H0"], PRI_MU["Om"], PRI_MU["tc"], PRI_MU["k"], PRI_MAG_MU], dtype=float)
        blend_widths_adapt = np.array(
            [PRI_SG["H0"] * 0.35, PRI_SG["Om"] * 0.35, PRI_SG["tc"] * 0.5, PRI_SG["k"] * 0.5, PRI_MAG_SG * 0.25],
            dtype=float,
        )
    blend_widths_adapt = np.maximum(blend_widths_adapt[:nd_b], floors_b[:nd_b])

    merged_flats_l: list[np.ndarray] = []
    merged_flats_b: list[np.ndarray] = []
    ch3_all_l: list[np.ndarray] = []
    ch3_all_b: list[np.ndarray] = []
    acc_rates_l: list[float] = []
    acc_rates_b: list[float] = []

    flats_l_per_fw: list[np.ndarray] = []
    flats_b_per_fw: list[np.ndarray] = []
    med_cosmo_lcdm_fw: list[np.ndarray] = []
    med_cosmo_blend_fw: list[np.ndarray] = []

    RUNS_PARENT = OUTDIR / "runs"
    RUNS_PARENT.mkdir(parents=True, exist_ok=True)

    def make_lnpost_lcdm() -> Callable[[np.ndarray], float]:
        if _use_cov_eff and chol_sig is not None and math.isfinite(logdet_twopisigma):

            def _ln_cov(th: np.ndarray) -> float:
                return float(log_post_lcdm_cov(np.asarray(th, dtype=float), mt, chol_sig, logdet_twopisigma, zt, eng))

            return _ln_cov

        def _ln(th: np.ndarray) -> float:
            return float(log_post_lcdm(np.asarray(th, dtype=float), zt, mt, st, eng))

        return _ln

    def make_lnpost_blend() -> Callable[[np.ndarray], float]:
        if _use_cov_eff and chol_sig is not None and math.isfinite(logdet_twopisigma):

            def _lnbc(th: np.ndarray) -> float:
                return float(log_post_blend_cov(np.asarray(th, dtype=float), mt, chol_sig, logdet_twopisigma, zt, eng))

            return _lnbc

        def _lnb(th: np.ndarray) -> float:
            return float(log_post_blend(np.asarray(th, dtype=float), zt, mt, st, eng))

        return _lnb

    for fw in range(max(int(N_FRAMEWORK_SEEDS), 1)):
        base_fw = int(RNG_SEED + 100019 * fw)
        rn_dir = RUNS_PARENT / f"framework_seed_{fw}"
        rn_dir.mkdir(parents=True, exist_ok=True)

        fw_flat_l, ch_fw_l, acc_fw_l = ensemble_emcee_for_model(
            make_lnpost_lcdm,
            nd_l,
            lcdm_lows,
            lcdm_highs,
            lcdm_centers,
            lcdm_widths,
            base_fw,
            p0_jitter_frac=0.0,
        )
        fw_flat_b, ch_fw_b, acc_fw_b = ensemble_emcee_for_model(
            make_lnpost_blend,
            nd_b,
            blend_lows,
            blend_highs,
            blend_centers_adapt[:nd_b],
            blend_widths_adapt[:nd_b],
            base_fw + 17,
            p0_jitter_frac=5e-4,
        )
        cid_save_l = np.full(int(fw_flat_l.shape[0]), fw, dtype=int)
        cid_save_b = np.full(int(fw_flat_b.shape[0]), fw, dtype=int)
        if infer_profile_m():
            names_l_sv = ["H0", "Omega_m"]
            names_b_sv = ["H0", "Omega_m", "t_crit", "k"]
        else:
            names_l_sv = ["H0", "Omega_m", "M"]
            names_b_sv = ["H0", "Omega_m", "t_crit", "k", "M"]
        br = max(int(N_WALKERS), 1) * max(int(N_PROD), 1)
        n_sh = max(int(N_CHAINS), 1)
        shard_pat = np.concatenate([np.full(int(br), int(j)) for j in range(int(n_sh))])
        nrow_l = int(fw_flat_l.shape[0])
        nrow_b = int(fw_flat_b.shape[0])

        def emcee_shard_column(nrow: int) -> np.ndarray:
            if nrow == shard_pat.size:
                return shard_pat.copy()
            sid = np.arange(int(nrow), dtype=int) // max(int(br), 1)
            return np.minimum(sid, int(n_sh) - 1).astype(int)

        shard_em_l = emcee_shard_column(nrow_l)
        shard_em_b = emcee_shard_column(nrow_b)

        dl_tmp = pd.DataFrame(fw_flat_l[:, : len(names_l_sv)], columns=names_l_sv)
        dl_tmp.insert(0, "framework_seed", cid_save_l)
        dl_tmp.insert(1, "emcee_chain_shard", shard_em_l.astype(int))
        dl_tmp.to_csv(rn_dir / "mcmc_chain_lcdm.csv", index=False)
        db_tmp = pd.DataFrame(fw_flat_b[:, : len(names_b_sv)], columns=names_b_sv)
        db_tmp.insert(0, "framework_seed", cid_save_b)
        db_tmp.insert(1, "emcee_chain_shard", shard_em_b.astype(int))
        db_tmp.to_csv(rn_dir / "mcmc_chain_blend.csv", index=False)

        merged_flats_l.append(fw_flat_l)
        merged_flats_b.append(fw_flat_b)
        for ch_piece in ch_fw_l:
            ch3_all_l.append(np.asarray(ch_piece, dtype=float))
        for ch_piece in ch_fw_b:
            ch3_all_b.append(np.asarray(ch_piece, dtype=float))
        acc_rates_l.append(acc_fw_l)
        acc_rates_b.append(acc_fw_b)
        flats_l_per_fw.append(fw_flat_l.copy())
        flats_b_per_fw.append(fw_flat_b.copy())
        med_cosmo_lcdm_fw.append(np.median(fw_flat_l, axis=0))
        med_cosmo_blend_fw.append(np.median(fw_flat_b, axis=0))

    flat_l = np.vstack(merged_flats_l)
    flat_b = np.vstack(merged_flats_b)

    flats_l_list = merged_flats_l
    flats_b_list = merged_flats_b
    ch3_list_l = ch3_all_l
    ch3_list_b = ch3_all_b

    cid_l = np.concatenate([np.full(int(a.shape[0]), fw, dtype=int) for fw, a in enumerate(merged_flats_l)])
    cid_b = np.concatenate([np.full(int(a.shape[0]), fw, dtype=int) for fw, a in enumerate(merged_flats_b)])

    stacked_fw_l_for_rhat = [np.asarray(arr, dtype=float) for arr in flats_l_per_fw]
    stacked_fw_b_for_rhat = [np.asarray(arr, dtype=float) for arr in flats_b_per_fw]
    if len(stacked_fw_l_for_rhat) >= 2:
        rhat_between_framework_l = gelman_r_hat_independent_chains(stacked_fw_l_for_rhat)
        rhat_between_framework_b = gelman_r_hat_independent_chains(stacked_fw_b_for_rhat)
    else:
        rhat_between_framework_l = np.full(nd_l, np.nan)
        rhat_between_framework_b = np.full(nd_b, np.nan)

    def _flat_from_ch3(ch3: np.ndarray) -> np.ndarray:
        ch3 = np.asarray(ch3, dtype=float)
        return ch3.reshape(int(ch3.shape[0]) * int(ch3.shape[1]), int(ch3.shape[2]))

    n_inner = min(int(N_CHAINS), len(ch3_list_l))
    emcee_between_l = (
        gelman_r_hat_independent_chains([_flat_from_ch3(ch3_list_l[jj]) for jj in range(n_inner)])
        if n_inner >= 2
        else np.full(nd_l, np.nan)
    )
    emcee_between_b = (
        gelman_r_hat_independent_chains([_flat_from_ch3(ch3_list_b[jj]) for jj in range(n_inner)])
        if n_inner >= 2
        else np.full(nd_b, np.nan)
    )
    rhat_between_l = emcee_between_l
    rhat_between_b = emcee_between_b

    rhat_split_l = gelman_r_hat(flat_l)
    rhat_split_b = gelman_r_hat(flat_b)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        tau_stack_l = np.nanmean(np.stack([integrated_autocorr_time(x) for x in ch3_list_l], axis=0), axis=0)
        tau_stack_b = np.nanmean(np.stack([integrated_autocorr_time(x) for x in ch3_list_b], axis=0), axis=0)

    tau_l = tau_stack_l
    tau_b = tau_stack_b

    rhat_walk_list_l = np.stack([chain_rhat_from_walkers(x) for x in ch3_list_l], axis=0)
    rhat_walk_list_b = np.stack([chain_rhat_from_walkers(x) for x in ch3_list_b], axis=0)
    rhat_walk_l_max = np.nanmax(rhat_walk_list_l, axis=0)
    rhat_walk_b_max = np.nanmax(rhat_walk_list_b, axis=0)

    n_flat_l = int(flat_l.shape[0])
    n_flat_b = int(flat_b.shape[0])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        tau_max_l = float(np.nanmax(tau_l))
        tau_max_b = float(np.nanmax(tau_b))
    ess_min_l = float(n_flat_l / (2.0 * tau_max_l)) if np.isfinite(tau_max_l) and tau_max_l > 0 else float("nan")
    ess_min_b = float(n_flat_b / (2.0 * tau_max_b)) if np.isfinite(tau_max_b) and tau_max_b > 0 else float("nan")

    if infer_profile_m():
        names_l_cols = ["H0", "Omega_m"]
    else:
        names_l_cols = ["H0", "Omega_m", "M"]
    if infer_profile_m():
        names_b_cols = ["H0", "Omega_m", "t_crit", "k"]
    else:
        names_b_cols = ["H0", "Omega_m", "t_crit", "k", "M"]

    def add_profiled_cols(flat: np.ndarray, blend: bool, chain_ids: np.ndarray) -> pd.DataFrame:
        h0col = flat[:, 0].astype(float)
        omcol = flat[:, 1].astype(float)
        ol_col = np.array([olambda_flat(float(o), float(h)) for h, o in zip(h0col, omcol)], dtype=float)
        m_prof = np.empty(int(flat.shape[0]), dtype=float)
        cols = names_b_cols if blend else names_l_cols
        for i in range(int(flat.shape[0])):
            row_i = flat[i]
            if infer_profile_m():
                if blend:
                    mu_tr_i, _ = mu_blend_shape(zt, float(row_i[0]), float(row_i[1]), float(row_i[2]), float(row_i[3]), eng)
                else:
                    mu_tr_i, _, _ = mu_lcdm_shape(zt, float(row_i[0]), float(row_i[1]), eng)
                m_prof[i] = profiled_absolute_magnitude(mu_tr_i, mt, st)
            elif blend:
                m_prof[i] = float(row_i[4])
            else:
                m_prof[i] = float(row_i[2])
        df0 = pd.DataFrame(flat[:, : len(cols)].copy(), columns=list(cols))
        df0.insert(0, "chain_id", chain_ids.astype(int))
        df0["Omega_Lambda"] = ol_col
        df0["M_profiled_or_sampled"] = m_prof
        return df0

    df_flat_l = add_profiled_cols(flat_l, False, cid_l)
    df_flat_b = add_profiled_cols(flat_b, True, cid_b)
    df_flat_l.to_csv(OUTDIR / "mcmc_chain_lcdm.csv", index=False)
    df_flat_b.to_csv(OUTDIR / "mcmc_chain_blend.csv", index=False)

    med_l_raw = np.median(flat_l, axis=0)
    med_b_raw = np.median(flat_b, axis=0)

    def theta_with_M_med(blend_ch: bool) -> np.ndarray:
        if infer_profile_m():
            row = med_b_raw if blend_ch else med_l_raw
            if blend_ch:
                mu_tr, _ = mu_blend_shape(zt, float(row[0]), float(row[1]), float(row[2]), float(row[3]), eng)
            else:
                mu_tr, _, _ = mu_lcdm_shape(zt, float(row[0]), float(row[1]), eng)
            mm = profiled_absolute_magnitude(mu_tr, mt, st)
            if blend_ch:
                return np.asarray([float(row[0]), float(row[1]), float(row[2]), float(row[3]), mm], dtype=float)
            return np.asarray([float(row[0]), float(row[1]), mm], dtype=float)
        if blend_ch:
            return np.asarray(med_b_raw, dtype=float)
        return np.asarray(med_l_raw, dtype=float)

    med_l = theta_with_M_med(False)
    med_b = theta_with_M_med(True)

    def scan_max_ll(flat: np.ndarray, blend_m: bool, cap: int = 12000) -> tuple[float, np.ndarray]:
        n = int(min(int(flat.shape[0]), cap))
        idx = RNG.choice(int(flat.shape[0]), size=n, replace=False)
        best = -1e300
        best_row = flat[0].copy()
        for row in flat[idx]:
            if blend_m:
                ll = float(lnlike_blend_profiled(row, zt, mt, st, eng) if infer_profile_m() else lnlike_blend_full(row, zt, mt, st, eng))
            else:
                ll = float(lnlike_lcdm_profiled(row, zt, mt, st, eng) if infer_profile_m() else lnlike_lcdm_full(row, zt, mt, st, eng))
            if ll > best:
                best = ll
                best_row = row.copy()
        return float(best), np.asarray(best_row, dtype=float)

    maxll_l, best_l_raw = scan_max_ll(flat_l, False)
    maxll_b, best_b_raw = scan_max_ll(flat_b, True)

    def theta_with_M_from_row(raw: np.ndarray, blend_ch: bool) -> np.ndarray:
        if infer_profile_m():
            row = raw
            if blend_ch:
                mu_tr, _ = mu_blend_shape(zt, float(row[0]), float(row[1]), float(row[2]), float(row[3]), eng)
                mm = profiled_absolute_magnitude(mu_tr, mt, st)
                return np.asarray([float(row[0]), float(row[1]), float(row[2]), float(row[3]), mm], dtype=float)
            mu_tr, _, _ = mu_lcdm_shape(zt, float(row[0]), float(row[1]), eng)
            mm = profiled_absolute_magnitude(mu_tr, mt, st)
            return np.asarray([float(row[0]), float(row[1]), mm], dtype=float)
        return np.asarray(raw, dtype=float).reshape(-1).copy()

    best_l_full = theta_with_M_from_row(best_l_raw, False)
    best_b_full = theta_with_M_from_row(best_b_raw, True)

    aic_l, bic_l = max_aic_bic(n_train, k_l, maxll_l)
    aic_b, bic_b = max_aic_bic(n_train, k_b, maxll_b)

    waic_draws_cfg = int(os.getenv("CWSF_WAIC_DRAWS", "1024"))
    waic_rows = min(max(32, waic_draws_cfg), int(flat_l.shape[0]))

    def build_ll_mat(flat: np.ndarray, blend_m: bool) -> np.ndarray:
        ix = RNG.choice(int(flat.shape[0]), size=int(waic_rows), replace=False)
        rows_out: list[np.ndarray] = []
        for row in flat[ix]:
            if infer_profile_m():
                if blend_m:
                    mu_fit, _ = mu_blend_shape(zt, float(row[0]), float(row[1]), float(row[2]), float(row[3]), eng)
                else:
                    mu_fit, _, _ = mu_lcdm_shape(zt, float(row[0]), float(row[1]), eng)
                mhat = mu_fit + profiled_absolute_magnitude(mu_fit, mt, st)
            elif blend_m:
                mhat, _ = mu_blend(zt, float(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4]), eng)
            else:
                mhat, _, _ = mu_lcdm(zt, float(row[0]), float(row[1]), float(row[2]), eng)
            rows_out.append(pointwise_loglike_gauss(mhat, mt, st))
        return np.stack(rows_out, axis=0)

    waic_l = waic_from_loglik_matrix(build_ll_mat(flat_l, False))
    waic_b = waic_from_loglik_matrix(build_ll_mat(flat_b, True))

    waic_stack_w_lcdm, waic_stack_w_blend = stacking_weights_from_waic(float(waic_l), float(waic_b))

    nested_summary: dict = dict(ran=False, dynesty_available=HAVE_DYNESTY)
    nested_dir = OUTDIR / "nested"
    nested_dir.mkdir(parents=True, exist_ok=True)
    if RUN_NESTED and HAVE_DYNESTY and dynesty is not None:

        def _nested_prior_pack(is_blend_ff: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            lows = np.asarray((blend_lows if is_blend_ff else lcdm_lows), dtype=float).reshape(-1)
            highs = np.asarray((blend_highs if is_blend_ff else lcdm_highs), dtype=float).reshape(-1)
            nd_here = int(lows.size)
            if is_blend_ff:
                if nd_here == 4:
                    mus = np.array([PRI_MU["H0"], PRI_MU["Om"], PRI_MU["tc"], PRI_MU["k"]], dtype=float)
                    sigs = np.array([PRI_SG["H0"], PRI_SG["Om"], PRI_SG["tc"], PRI_SG["k"]], dtype=float)
                else:
                    mus = np.array(
                        [PRI_MU["H0"], PRI_MU["Om"], PRI_MU["tc"], PRI_MU["k"], PRI_MAG_MU],
                        dtype=float,
                    )
                    sigs = np.array(
                        [PRI_SG["H0"], PRI_SG["Om"], PRI_SG["tc"], PRI_SG["k"], PRI_MAG_SG],
                        dtype=float,
                    )
            elif nd_here == 2:
                mus = np.array([PRI_MU["H0"], PRI_MU["Om"]], dtype=float)
                sigs = np.array([PRI_SG["H0"], PRI_SG["Om"]], dtype=float)
            else:
                mus = np.array([PRI_MU["H0"], PRI_MU["Om"], PRI_MAG_MU], dtype=float)
                sigs = np.array([PRI_SG["H0"], PRI_SG["Om"], PRI_MAG_SG], dtype=float)
            return lows, highs, mus, sigs

        def _lnlike_nested_phys(is_blend_ff: bool, th_phys: np.ndarray) -> float:
            th = np.asarray(th_phys, dtype=float).reshape(-1)
            nd_here = nd_b if is_blend_ff else nd_l
            if int(th.size) != int(nd_here):
                return -1e300
            zmax = float(np.max(zt))
            if is_blend_ff and blend_physics_is_ccomplet2():
                if int(th.size) < 4:
                    return -1e300
                if not validate_blend_background(float(th[0]), float(th[1]), float(th[2]), float(th[3]), eng, zmax):
                    return -1e300
            elif not validate_flat_background(float(th[0]), float(th[1]), eng, zmax):
                return -1e300
            globals()["_CWSF_LNPOST_IS_BLEND"] = bool(is_blend_ff)
            globals()["_CWSF_EXT_THETA"] = th.copy()
            try:
                if _use_cov_eff and chol_sig is not None and math.isfinite(logdet_twopisigma):
                    if is_blend_ff:
                        if int(th.size) == 4 and infer_profile_m():
                            mu_tr, _ = mu_blend_shape(zt, float(th[0]), float(th[1]), float(th[2]), float(th[3]), eng)
                            mm = profiled_absolute_magnitude(mu_tr, mt, st)
                            pred, _ = mu_blend(zt, float(th[0]), float(th[1]), float(th[2]), float(th[3]), mm, eng)
                        else:
                            pred, _ = mu_blend(zt, float(th[0]), float(th[1]), float(th[2]), float(th[3]), float(th[4]), eng)
                    else:
                        pred, _, _ = mu_lcdm(zt, float(th[0]), float(th[1]), float(th[2]), eng)
                    ll = lnlike_mvn_residual(np.asarray(mt, dtype=float) - np.asarray(pred, dtype=float), chol_sig, logdet_twopisigma)
                elif is_blend_ff:
                    ll = (
                        lnlike_blend_profiled(th, zt, mt, st, eng)
                        if infer_profile_m()
                        else lnlike_blend_full(th, zt, mt, st, eng)
                    )
                else:
                    ll = lnlike_lcdm_profiled(th, zt, mt, st, eng) if infer_profile_m() else lnlike_lcdm_full(th, zt, mt, st, eng)
                llx = _external_lnlike_add(th, eng)
                if not math.isfinite(llx):
                    return -1e300
                ll = float(ll + llx)
                return float(ll) if math.isfinite(ll) else -1e300
            except Exception:
                return -1e300
            finally:
                globals()["_CWSF_LNPOST_IS_BLEND"] = False

        try:

            def _run_ns(is_bf: bool) -> dict[str, float | int]:
                lows_, highs_, mus_, sigs_ = _nested_prior_pack(is_bf)

                def _pt(u):
                    return unit_cube_to_truncnorm_phys(np.asarray(u, dtype=float), lows_, highs_, mus_, sigs_)

                sampler = dynesty.NestedSampler(
                    lambda x: float(_lnlike_nested_phys(is_bf, x)),
                    _pt,
                    int(lows_.size),
                    nlive=int(NESTED_NLIVE),
                    bound="multi",
                    sample="rwalk",
                    rstate=np.random.RandomState(int(RNG_SEED + 733 * (37 if is_bf else 11))),
                )
                sampler.run_nested(dlogz=float(NESTING_DLOGZ), print_progress=False)
                res = sampler.results
                logz_here = float(res.logz[-1])
                logzerr_here = float(res.logzerr[-1])
                samp_here = np.asarray(res.samples, dtype=float)
                tag = "blend" if is_bf else "lcdm"
                colnames = names_b_cols if is_bf else names_l_cols
                pd.DataFrame(samp_here, columns=list(colnames)).to_csv(nested_dir / f"nested_posterior_samples_{tag}.csv", index=False)
                return dict(logz=logz_here, logzerr=logzerr_here, nlive=int(NESTED_NLIVE), n_samples=int(samp_here.shape[0]))

            nsl = _run_ns(False)
            nsb = _run_ns(True)
            dlz = float(nsb["logz"] - nsl["logz"])
            nested_summary = dict(
                ran=True,
                inference_mode="covariance_mvn" if _use_cov_eff else "diagonal_gaussian",
                lcdm=dict(logz=float(nsl["logz"]), logzerr=float(nsl["logzerr"]), nlive=int(nsl["nlive"]), n_samples=int(nsl["n_samples"])),
                blend=dict(logz=float(nsb["logz"]), logzerr=float(nsb["logzerr"]), nlive=int(nsb["nlive"]), n_samples=int(nsb["n_samples"])),
                delta_logz_blend_minus_lcdm=json_safe_float(float(dlz)),
                jeffreys_label_blend_minus_lcdm=jeffreys_label_for_delta_logz(dlz),
                outdir=str(nested_dir.resolve()),
                dlogz_stop=float(NESTING_DLOGZ),
                n_live=int(NESTED_NLIVE),
            )
        except Exception as ex_n:
            nested_summary = dict(ran=False, dynesty_available=True, reason=f"nested_sampling_failed:{ex_n!s}")

    elif RUN_NESTED and not HAVE_DYNESTY:
        nested_summary["reason"] = "dynesty_not_installed"

    cv_rows: list[dict] = []
    cv_summary = dict(ran=False, note="Enable CWSF_RUN_CV=1 for Pantheon+ internal folds (abbreviated MCMC per fold).")
    if RUN_CV:
        rng_cv = np.random.default_rng(RNG_CV_SEED)
        z_order = np.argsort(zt)
        for rep in range(max(int(N_CV_REPEATS), 1)):
            if rep > 0:
                z_order = np.roll(z_order, int(rng_cv.integers(low=1, high=max(int(n_train) // 2, 2))))
            for fi in range(max(int(N_CV_FOLDS), 1)):
                val_idx = z_order[np.arange(int(fi), int(n_train), int(N_CV_FOLDS))]
                train_mask = np.ones(int(n_train), dtype=bool)
                train_mask[val_idx] = False
                tr = np.flatnonzero(train_mask)
                if int(tr.size) < int(max(nd_b + 10, k_b + 5)) or int(val_idx.size) < 10:
                    continue
                cv_seed_here = int(RNG_CV_SEED + 9973 * rep + 103 * fi)
                chol_tr_inner = None
                logdet_tr_inner = float("nan")
                if _use_cov_eff:
                    assert pant_sig_full is not None
                    Sig_tr = pant_sig_full[np.ix_(tr, tr)]
                    st_lg, slog_tr = np.linalg.slogdet(Sig_tr)
                    if st_lg <= 0:
                        continue
                    logdet_tr_inner = float(tr.size * math.log(2.0 * math.pi) + slog_tr)
                    chol_tr_inner = la_sci.cholesky(Sig_tr, lower=True)

                def lf_lcdm_cv():

                    def _ln(th):

                        return float(
                            log_post_lcdm_cov(np.asarray(th, dtype=float), mt[tr], chol_tr_inner, logdet_tr_inner, zt[tr], eng)
                            if chol_tr_inner is not None and math.isfinite(logdet_tr_inner)
                            else log_post_lcdm(np.asarray(th, dtype=float), zt[tr], mt[tr], st[tr], eng)
                        )

                    return _ln

                def lf_blend_cv():

                    def _lnb(th):

                        return float(
                            log_post_blend_cov(np.asarray(th, dtype=float), mt[tr], chol_tr_inner, logdet_tr_inner, zt[tr], eng)
                            if chol_tr_inner is not None and math.isfinite(logdet_tr_inner)
                            else log_post_blend(np.asarray(th, dtype=float), zt[tr], mt[tr], st[tr], eng)
                        )

                    return _lnb

                fl_cv_l, _, _ = ensemble_emcee_for_model(
                    lf_lcdm_cv,
                    nd_l,
                    lcdm_lows,
                    lcdm_highs,
                    lcdm_centers,
                    lcdm_widths,
                    cv_seed_here + 3,
                    p0_jitter_frac=0.0,
                    burn_steps=CV_N_BURN,
                    prod_steps=CV_N_PROD,
                    n_walkers_ov=CV_N_WALKERS,
                    n_chains_ov=CV_N_CHAINS,
                )
                fl_cv_b, _, _ = ensemble_emcee_for_model(
                    lf_blend_cv,
                    nd_b,
                    blend_lows,
                    blend_highs,
                    blend_centers_adapt[:nd_b],
                    blend_widths_adapt[:nd_b],
                    cv_seed_here + 9,
                    p0_jitter_frac=5e-4,
                    burn_steps=CV_N_BURN,
                    prod_steps=CV_N_PROD,
                    n_walkers_ov=CV_N_WALKERS,
                    n_chains_ov=CV_N_CHAINS,
                )
                theta_l_fold = theta_with_M_from_row(np.median(fl_cv_l, axis=0), False)
                theta_b_fold = theta_with_M_from_row(np.median(fl_cv_b, axis=0), True)
                z_val = np.asarray(zt[val_idx], dtype=float)
                mu_val = np.asarray(mt[val_idx], dtype=float)
                sig_val = np.asarray(st[val_idx], dtype=float)
                mu_tr_slice_l, _, _ = mu_lcdm_shape(np.asarray(zt[tr], dtype=float), float(theta_l_fold[0]), float(theta_l_fold[1]), eng)
                Ml_fold = profiled_absolute_magnitude(mu_tr_slice_l, np.asarray(mt[tr], dtype=float), np.asarray(st[tr], dtype=float))
                mu_hat_val_l, _, _ = mu_lcdm_shape(z_val, float(theta_l_fold[0]), float(theta_l_fold[1]), eng)
                pred_l_fold = mu_hat_val_l + float(Ml_fold)
                mu_tr_slice_b, _ = mu_blend_shape(np.asarray(zt[tr], dtype=float), float(theta_b_fold[0]), float(theta_b_fold[1]), float(theta_b_fold[2]), float(theta_b_fold[3]), eng)
                Mb_fold = profiled_absolute_magnitude(mu_tr_slice_b, np.asarray(mt[tr], dtype=float), np.asarray(st[tr], dtype=float))
                mu_hat_val_b, _ = mu_blend_shape(z_val, float(theta_b_fold[0]), float(theta_b_fold[1]), float(theta_b_fold[2]), float(theta_b_fold[3]), eng)
                pred_b_fold = mu_hat_val_b + float(Mb_fold)
                rl = np.asarray(mu_val - pred_l_fold, dtype=float)
                rb = np.asarray(mu_val - pred_b_fold, dtype=float)
                rmse_l = float(math.sqrt(float(np.mean(rl**2))))
                rmse_b = float(math.sqrt(float(np.mean(rb**2))))
                chi2_l = float(np.sum((rl / sig_val) ** 2))
                chi2_b = float(np.sum((rb / sig_val) ** 2))
                cv_rows.append(
                    dict(
                        repeat=int(rep),
                        fold=int(fi),
                        split="z_round_robin_stratified",
                        n_train=int(tr.size),
                        n_val=int(val_idx.size),
                        likelihood_mode=("covariance" if chol_tr_inner is not None else "diagonal"),
                        rmse_lcdm=rmse_l,
                        rmse_blend=rmse_b,
                        chi2_lcdm=float(chi2_l),
                        chi2_blend=float(chi2_b),
                        bias_lcdm=float(np.mean(rl)),
                        bias_blend=float(np.mean(rb)),
                        blend_rmse_less_than_lcdm=bool(rmse_b < rmse_l),
                    )
                )
        cv_table = pd.DataFrame(cv_rows)
        cv_path = OUTDIR / "cross_validation_fold_metrics.csv"
        if not cv_table.empty:
            cv_table.to_csv(cv_path, index=False)
            wm = cv_table.groupby("repeat")["blend_rmse_less_than_lcdm"].mean()
            cv_summary = dict(
                ran=True,
                n_fold_rows=int(len(cv_table)),
                likelihood_mode=("covariance" if _use_cov_eff else "diagonal"),
                mean_fraction_folds_where_blend_rmse_below_lcdm_across_repeats=float(np.mean(np.asarray(wm, dtype=float))),
                std_fraction_across_repeats=float(np.std(np.asarray(wm, dtype=float), ddof=1)) if len(wm) > 1 else 0.0,
                rmse_lcdm_mean=float(np.mean(cv_table["rmse_lcdm"])),
                rmse_lcdm_std=float(np.std(cv_table["rmse_lcdm"], ddof=1)),
                rmse_blend_mean=float(np.mean(cv_table["rmse_blend"])),
                rmse_blend_std=float(np.std(cv_table["rmse_blend"], ddof=1)),
                folds_csv=str(cv_path.resolve()),
                config=dict(
                    folds=int(N_CV_FOLDS),
                    repeats=int(N_CV_REPEATS),
                    burn=int(CV_N_BURN),
                    prod=int(CV_N_PROD),
                    walkers=int(CV_N_WALKERS),
                    chains=int(CV_N_CHAINS),
                    seed=int(RNG_CV_SEED),
                ),
            )
        else:
            cv_summary["ran"] = True
            cv_summary["failure"] = "no_valid_fold_rows_generated"

    rob_rows: list[dict] = []
    robustness_summary = dict(ran=False, note="Short MCMC subsets for stress tests.")
    rob_dir = OUTDIR / "robustness"
    rob_dir.mkdir(parents=True, exist_ok=True)
    if RUN_ROBUSTNESS:
        scen_list: list[tuple[str, np.ndarray]] = []
        zz = np.asarray(zt, dtype=float).reshape(-1)
        mu0 = np.asarray(mt, dtype=float).reshape(-1)
        sg0 = np.asarray(st, dtype=float).reshape(-1)
        scen_list.append(("all_z_diag_baseline_diag", np.ones(int(n_train), dtype=bool)))
        scen_list.append(("z_lt_085", zz < 0.85))
        scen_list.append(("z_lt_1", zz < 1.0))
        scen_list.append(("z_lt_13", zz < 1.3))
        scen_list.append(("drop_top_decile_z", zz <= np.quantile(zz, 0.9)))
        scen_list.append(("calib_mag_offset_plus_0p015", np.ones(int(n_train), dtype=bool)))
        for stag, mk in scen_list:
            idxs = np.flatnonzero(np.asarray(mk).reshape(-1))
            mu_use = mu0.copy()
            if stag == "calib_mag_offset_plus_0p015":
                mu_use = mu0 + 0.015
            if int(idxs.size) < int(max(nd_b + 40, k_b + 14)):
                continue
            chol_r = None
            logdet_r = float("nan")
            z_r = np.asarray(zt[idxs], dtype=float)
            mu_r = np.asarray(mu_use[idxs], dtype=float)
            sig_r = np.asarray(sg0[idxs], dtype=float)
            eng_r_max = float(np.max(z_r)) if z_r.size else 0.05
            if eng_r_max > zp_hi_dm - 1.0:
                eng_r = CosmoInterpEngine(zp_hi_dm, eng_r_max)
            else:
                eng_r = eng
            if _use_cov_eff and pant_sig_full is not None:
                Sig_r = pant_sig_full[np.ix_(idxs, idxs)]
                sig_r_chk, slog_r = np.linalg.slogdet(Sig_r)
                if sig_r_chk <= 0:
                    continue
                logdet_r = float(idxs.size * math.log(2.0 * math.pi) + slog_r)
                chol_r = la_sci.cholesky(Sig_r, lower=True)

            rob_seed = int(31079 + (_stable_hash_u64(str(stag)) % 100007))

            def lf_r_l():

                def _lr(th):

                    return float(
                        log_post_lcdm_cov(np.asarray(th, dtype=float), mu_r, chol_r, logdet_r, z_r, eng_r)
                        if chol_r is not None and math.isfinite(logdet_r)
                        else log_post_lcdm(np.asarray(th, dtype=float), z_r, mu_r, sig_r, eng_r)
                    )

                return _lr

            def lf_r_b():

                def _lr(th):

                    return float(
                        log_post_blend_cov(np.asarray(th, dtype=float), mu_r, chol_r, logdet_r, z_r, eng_r)
                        if chol_r is not None and math.isfinite(logdet_r)
                        else log_post_blend(np.asarray(th, dtype=float), z_r, mu_r, sig_r, eng_r)
                    )

                return _lr

            fl_rb, _, _ = ensemble_emcee_for_model(
                lf_r_l,
                nd_l,
                lcdm_lows,
                lcdm_highs,
                lcdm_centers,
                lcdm_widths,
                rob_seed,
                p0_jitter_frac=0.0,
                burn_steps=ROB_SHORT_BURN,
                prod_steps=ROB_SHORT_PROD,
                n_walkers_ov=min(CV_N_WALKERS, N_WALKERS),
                n_chains_ov=min(CV_N_CHAINS + 1, max(N_CHAINS, 1)),
            )
            fl_rbb, _, _ = ensemble_emcee_for_model(
                lf_r_b,
                nd_b,
                blend_lows,
                blend_highs,
                blend_centers_adapt[:nd_b],
                blend_widths_adapt[:nd_b],
                rob_seed + 19,
                p0_jitter_frac=5e-4,
                burn_steps=ROB_SHORT_BURN,
                prod_steps=ROB_SHORT_PROD,
                n_walkers_ov=min(CV_N_WALKERS, N_WALKERS),
                n_chains_ov=min(CV_N_CHAINS + 1, max(N_CHAINS, 1)),
            )
            med_r_l_raw = np.median(fl_rb, axis=0)
            med_r_b_raw = np.median(fl_rbb, axis=0)

            def _theta_med_row(raw_slice: np.ndarray, blend_here: bool) -> np.ndarray:
                return theta_with_M_from_row(raw_slice, blend_here)

            th_r_l = _theta_med_row(med_r_l_raw, False)
            th_r_b = _theta_med_row(med_r_b_raw, True)
            mhat_tl, _, _ = mu_lcdm(z_r, float(th_r_l[0]), float(th_r_l[1]), float(th_r_l[2]), eng_r)
            mhat_tb, _ = mu_blend(z_r, float(th_r_b[0]), float(th_r_b[1]), float(th_r_b[2]), float(th_r_b[3]), float(th_r_b[4]), eng_r)
            res_r_l = mu_r - np.asarray(mhat_tl, dtype=float)
            res_r_b = mu_r - np.asarray(mhat_tb, dtype=float)
            rm_rob_l = float(math.sqrt(np.mean(res_r_l**2)))
            rm_rob_b = float(math.sqrt(np.mean(res_r_b**2)))

            cap_r = min(2048, int(fl_rb.shape[0]))
            best_l_rb = fl_rb[0].copy()
            best_b_rb = fl_rbb[0].copy()
            mxll_rb_l = float(-1e300)
            mxll_rb_b = float(-1e300)
            if chol_r is not None and math.isfinite(logdet_r):
                idx_rsub = RNG.choice(int(fl_rb.shape[0]), size=int(cap_r), replace=False)
                for row_i in np.asarray(fl_rb[idx_rsub], dtype=float):
                    thf_li = theta_with_M_from_row(row_i, False)
                    pred_li, _, _ = mu_lcdm(z_r, float(thf_li[0]), float(thf_li[1]), float(thf_li[2]), eng_r)
                    ll_i = float(lnlike_mvn_residual(np.asarray(mu_r, dtype=float) - np.asarray(pred_li, dtype=float), chol_r, logdet_r))
                    if ll_i > mxll_rb_l:
                        mxll_rb_l = ll_i
                        best_l_rb = row_i.copy()
                for row_j in np.asarray(fl_rbb[idx_rsub], dtype=float):
                    thf_bi = theta_with_M_from_row(row_j, True)
                    pred_bi, _ = mu_blend(
                        z_r,
                        float(thf_bi[0]),
                        float(thf_bi[1]),
                        float(thf_bi[2]),
                        float(thf_bi[3]),
                        float(thf_bi[4]),
                        eng_r,
                    )
                    ll_j = float(lnlike_mvn_residual(np.asarray(mu_r, dtype=float) - np.asarray(pred_bi, dtype=float), chol_r, logdet_r))
                    if ll_j > mxll_rb_b:
                        mxll_rb_b = ll_j
                        best_b_rb = row_j.copy()
            else:
                mxll_rb_l, best_l_rb = scan_max_ll(fl_rb, False, cap=cap_r)
                mxll_rb_b, best_b_rb = scan_max_ll(fl_rbb, True, cap=cap_r)
            a_rb_l, b_rb_l = max_aic_bic(int(mu_r.shape[0]), k_l, mxll_rb_l)
            a_rb_b, b_rb_b = max_aic_bic(int(mu_r.shape[0]), k_b, mxll_rb_b)
            dlz_rob = None
            if (
                RUN_NESTED
                and isinstance(nested_summary, dict)
                and nested_summary.get("ran") is True
                and stag == "all_z_diag_baseline_diag"
            ):
                dlz_rob = nested_summary.get("delta_logz_blend_minus_lcdm")
            rob_rows.append(
                dict(
                    scenario=stag,
                    n_sn=int(z_r.size),
                    H0_med_lcdm=float(th_r_l[0]),
                    H0_med_blend=float(th_r_b[0]),
                    Omega_m_med_lcdm=float(th_r_l[1]),
                    Omega_m_med_blend=float(th_r_b[1]),
                    rmse_train_lcdm=rm_rob_l,
                    rmse_train_blend=rm_rob_b,
                    delta_aic_b_minus_l=float(a_rb_b - a_rb_l),
                    delta_bic_b_minus_l=float(b_rb_b - b_rb_l),
                    nested_delta_log_z_blend_minus_lcdm=json_safe_float(float(dlz_rob)) if dlz_rob is not None else None,
                    likelihood_mode=("covariance" if chol_r is not None else "diagonal"),
                )
            )
        rob_df = pd.DataFrame(rob_rows)
        if not rob_df.empty:
            rob_df.to_csv(rob_dir / "robustness_scenarios.csv", index=False)
            robustness_summary = dict(
                ran=True,
                n_scenarios=int(len(rob_df)),
                scenarios_csv=str((rob_dir / "robustness_scenarios.csv").resolve()),
                short_mcmc=dict(burn=ROB_SHORT_BURN, prod=ROB_SHORT_PROD),
            )

    def chi2_training_at(theta_full: np.ndarray, blend_ch: bool) -> float:
        if blend_ch:
            if infer_profile_m():
                mu_fit, _ = mu_blend_shape(zt, float(theta_full[0]), float(theta_full[1]), float(theta_full[2]), float(theta_full[3]), eng)
                mhat = mu_fit + float(theta_full[4])
            else:
                mhat, _ = mu_blend(zt, float(theta_full[0]), float(theta_full[1]), float(theta_full[2]), float(theta_full[3]), float(theta_full[4]), eng)
        else:
            if infer_profile_m():
                mu_fit, _, _ = mu_lcdm_shape(zt, float(theta_full[0]), float(theta_full[1]), eng)
                mhat = mu_fit + float(theta_full[2])
            else:
                mhat, _, _ = mu_lcdm(zt, float(theta_full[0]), float(theta_full[1]), float(theta_full[2]), eng)
        return float(np.sum(((mt - mhat) / st) ** 2))

    chi2_med_l = chi2_training_at(med_l, False)
    chi2_med_b = chi2_training_at(med_b, True)
    dof_l = float(n_train - k_l)
    dof_b = float(n_train - k_b)
    redchi_l = chi2_med_l / max(dof_l, eps())
    redchi_b = chi2_med_b / max(dof_b, eps())

    des_l = metrics_holdout_profiled(best_l_full, zt, mt, st, des, False, eng)
    des_b = metrics_holdout_profiled(best_b_full, zt, mt, st, des, True, eng)
    des_l_med = metrics_holdout_profiled(med_l, zt, mt, st, des, False, eng)
    des_b_med = metrics_holdout_profiled(med_b, zt, mt, st, des, True, eng)

    _pack_ext = globals().get("_EXTERNAL_JOINT_PACK")
    cross_rows: list[dict[str, object]] = []
    if HAVE_EXTERNAL_LIKES and _pack_ext is not None:
        cmp_l = _pack_ext.chi2_components(float(med_l[0]), float(med_l[1]), eng, joint_geometry_theta=None)
        cmp_b = _pack_ext.chi2_components(float(med_b[0]), float(med_b[1]), eng, joint_geometry_theta=np.asarray(med_b, dtype=float))
        b_l, c_l = cmp_l.get("bao_chi2"), cmp_l.get("cmb_chi2")
        b_b, c_b = cmp_b.get("bao_chi2"), cmp_b.get("cmb_chi2")
        sum_bao_cmb_l = float((b_l or 0.0) + (c_l or 0.0))
        sum_bao_cmb_b = float((b_b or 0.0) + (c_b or 0.0))
        print(
            "[external] LCDM median θ — BAO -2lnL term:",
            json_safe_float(float(b_l)) if b_l is not None else None,
            "CMB compressed -2lnL term:",
            json_safe_float(float(c_l)) if c_l is not None else None,
            "sum(BAO+CMB) -2lnL:",
            json_safe_float(sum_bao_cmb_l),
        )
        print(
            "[external] Blend median θ — BAO -2lnL term:",
            json_safe_float(float(b_b)) if b_b is not None else None,
            "CMB compressed -2lnL term:",
            json_safe_float(float(c_b)) if c_b is not None else None,
            "sum(BAO+CMB) -2lnL:",
            json_safe_float(sum_bao_cmb_b),
        )
        print(
            "[external] Joint training-style sum χ²_proxy (SN + BAO + CMB -2lnL pieces, not a single calibrated experiment):",
            json_safe_float(float(chi2_med_l + sum_bao_cmb_l)),
            "(LCDM);",
            json_safe_float(float(chi2_med_b + sum_bao_cmb_b)),
            "(blend).",
        )
        try:
            plot_bao_dm_rd(
                FIGDIR / "bao_DM_over_rd_vs_z.png",
                float(med_l[0]),
                float(med_l[1]),
                float(_pack_ext.omega_b_h2),
                eng,
                _pack_ext.bao,
            )
            plot_cmb_shift_prediction(
                FIGDIR / "cmb_compressed_shift_vs_lcdm_median.png",
                float(med_l[0]),
                float(med_l[1]),
                float(_pack_ext.omega_b_h2),
                eng,
                _pack_ext.cmb,
            )
        except Exception as ex_pl:
            external_like_warnings.append(f"external_probe_plots_failed:{ex_pl!s}")
        cross_rows = [
            dict(
                model="lcdm",
                metric="chi2_training_style",
                sn_only=float(chi2_med_l),
                sn_plus_bao=float(chi2_med_l + (b_l or 0.0)) if b_l is not None else None,
                sn_plus_bao_cmb=float(chi2_med_l + sum_bao_cmb_l) if (b_l is not None or c_l is not None) else None,
                note="sn_only is Pantheon sum-of-squares residual χ²; BAO/CMB columns add -2lnL MVN blocks (different normalization).",
            ),
            dict(
                model="blend",
                metric="chi2_training_style",
                sn_only=float(chi2_med_b),
                sn_plus_bao=float(chi2_med_b + (b_b or 0.0)) if b_b is not None else None,
                sn_plus_bao_cmb=float(chi2_med_b + sum_bao_cmb_b) if (b_b is not None or c_b is not None) else None,
                note="same caveat as lcdm row",
            ),
            dict(
                model="lcdm",
                metric="rmse_des_holdout_mag",
                sn_only=float(des_l_med["rmse"]),
                sn_plus_bao=float(des_l_med["rmse"]),
                sn_plus_bao_cmb=float(des_l_med["rmse"]),
                note="DES RMSE uses SN-trained posteriors; BAO/CMB do not enter DES likelihood.",
            ),
            dict(
                model="blend",
                metric="rmse_des_holdout_mag",
                sn_only=float(des_b_med["rmse"]),
                sn_plus_bao=float(des_b_med["rmse"]),
                sn_plus_bao_cmb=float(des_b_med["rmse"]),
                note="same as lcdm DES row",
            ),
            dict(
                model="lcdm",
                metric="waic_sn_training_only",
                sn_only=float(waic_l),
                sn_plus_bao=None,
                sn_plus_bao_cmb=None,
                note="WAIC from SN pointwise log-lik matrix only; not recomputed for joint MVN extensions.",
            ),
            dict(
                model="blend",
                metric="waic_sn_training_only",
                sn_only=float(waic_b),
                sn_plus_bao=None,
                sn_plus_bao_cmb=None,
                note="same as lcdm WAIC row",
            ),
        ]
        pd.DataFrame(cross_rows).to_csv(OUTDIR / "cross_probe_metrics.csv", index=False)

    sparc_dir_env = os.getenv("CWSF_SPARC_DIR", "").strip()
    if HAVE_EXTERNAL_LIKES and sparc_dir_env:
        try:
            sdf = run_sparc_mond_benchmark(
                Path(sparc_dir_env),
                OUTDIR / "sparc_mond_per_galaxy_chi2.csv",
                FIGDIR / "sparc_mond_example_rotation_curve.png",
            )
            if len(sdf) and "chi2_mond" in sdf.columns:
                ok = sdf[sdf["status"] == "ok"] if "status" in sdf.columns else sdf
                print(
                    "[SPARC/MOND separate] aggregate chi2 Newtonian (sum):",
                    float(np.nansum(ok["chi2_newton"].astype(float))) if "chi2_newton" in ok.columns else None,
                    "aggregate chi2 MOND (sum):",
                    float(np.nansum(ok["chi2_mond"].astype(float))) if "chi2_mond" in ok.columns else None,
                    "galaxies:",
                    int(len(ok)),
                )
        except Exception as ex_sp:
            external_like_warnings.append(f"sparc_mond_benchmark_failed:{ex_sp!s}")

    try:
        blend_chain_parameter_correlations(flat_b, names_b_cols).to_csv(
            OUTDIR / "posterior_blend_parameter_correlations.csv", index=False
        )
    except Exception:
        pass
    try:
        sdf_sens = finite_diff_blend_mu_sensitivity_rows(
            np.asarray(med_b, dtype=float), float(Z_REF_DELTA_MU_CORR), eng
        )
        if len(sdf_sens):
            sdf_sens.to_csv(OUTDIR / "blend_mu_finitediff_sensitivity.csv", index=False)
    except Exception:
        pass
    write_paper_outline_md(OUTDIR / "PAPER_OUTLINE.md")
    write_systematics_defensibility_md(OUTDIR / "SYSTEMATICS_AND_DEFENSIBILITY.md")

    z_split = float(np.median(z_des))
    low = des[z_des <= z_split]
    high = des[z_des > z_split]
    des_split = {}
    if len(low):
        des_split["low_z"] = dict(
            zmax=z_split,
            lcdm_med=metrics_holdout_profiled(med_l, zt, mt, st, low, False, eng),
            blend_med=metrics_holdout_profiled(med_b, zt, mt, st, low, True, eng),
        )
    if len(high):
        des_split["high_z"] = dict(
            zmin=z_split,
            lcdm_med=metrics_holdout_profiled(med_l, zt, mt, st, high, False, eng),
            blend_med=metrics_holdout_profiled(med_b, zt, mt, st, high, True, eng),
        )

    mtrain_l_z, _, _ = mu_lcdm(zt, float(med_l[0]), float(med_l[1]), float(med_l[-1]), eng)
    mtrain_b_z, _ = mu_blend(zt, float(med_b[0]), float(med_b[1]), float(med_b[2]), float(med_b[3]), float(med_b[-1]), eng)
    res_tr_slice_l = np.asarray(mt - mtrain_l_z, dtype=float)
    res_tr_slice_b = np.asarray(mt - mtrain_b_z, dtype=float)
    dmu_train_z = np.asarray(mtrain_b_z - mtrain_l_z, dtype=float)
    n_zbin = int(os.getenv("CWSF_REDSHIFT_SLICES", "10"))
    z_edges_tr = np.quantile(np.asarray(zt, dtype=float), np.linspace(0.0, 1.0, int(n_zbin) + 1))
    zs_rows: list[dict[str, float | int]] = []
    for ib in range(int(n_zbin)):
        lo_b, hi_b = float(z_edges_tr[ib]), float(z_edges_tr[ib + 1])
        msk_b = (np.asarray(zt, dtype=float) >= lo_b) & (
            (np.asarray(zt, dtype=float) <= hi_b) if ib == int(n_zbin) - 1 else (np.asarray(zt, dtype=float) < hi_b)
        )
        if not np.any(msk_b):
            continue
        rl = res_tr_slice_l[msk_b]
        rb = res_tr_slice_b[msk_b]
        sg_loc = np.asarray(st, dtype=float)[msk_b]
        zs_rows.append(
            dict(
                bin_index=int(ib),
                z_lo=lo_b,
                z_hi=hi_b,
                z_mid=float(np.mean(np.asarray(zt, dtype=float)[msk_b])),
                n=int(np.sum(msk_b)),
                mean_residual_lcdm=float(np.mean(rl)),
                rms_residual_lcdm=float(math.sqrt(float(np.mean(rl**2)))),
                mean_residual_blend=float(np.mean(rb)),
                rms_residual_blend=float(math.sqrt(float(np.mean(rb**2)))),
                mean_delta_mu_blend_minus_lcdm_mag=float(np.mean(dmu_train_z[msk_b])),
                bias_delta=float(np.mean(rb) - np.mean(rl)),
            )
        )
    pd.DataFrame(zs_rows).to_csv(OUTDIR / "training_redshift_sliced_residuals.csv", index=False)

    t0_median_lcdm = float(eng.cosmic_age_gyr(np.array([0.0]), float(med_l[0]), float(med_l[1]))[0])
    if blend_physics_is_ccomplet2():
        t0_median_blend = float(
            eng.blend_cosmic_age_gyr(np.array([0.0]), float(med_b[0]), float(med_b[1]), float(med_b[2]), float(med_b[3]))[0]
        )
    else:
        t0_median_blend = float(eng.cosmic_age_gyr(np.array([0.0]), float(med_b[0]), float(med_b[1]))[0])

    zgrid = np.linspace(0.02, 1.6, 81, dtype=float)
    q16_l, q50_l, q84_l, q025_l, q975_l, _, _, _, _, _ = predictive_quantiles(
        flat_l, zgrid, RNG, False, eng, zt, mt, st, max_rows=2048
    )
    q16_b, q50_b, q84_b, q025_b, q975_b, d16, d50, d84, d025, d975 = predictive_quantiles(
        flat_b, zgrid, RNG, True, eng, zt, mt, st, max_rows=2048
    )

    hz_l = Hz(zgrid, float(med_l[0]), float(med_l[1]))
    if blend_physics_is_ccomplet2():
        hz_b_med = np.asarray(
            eng.blend_Hz(zgrid, float(med_b[0]), float(med_b[1]), float(med_b[2]), float(med_b[3])),
            dtype=float,
        )
        tgy_med_b = np.asarray(
            eng.blend_cosmic_age_gyr(zgrid, float(med_b[0]), float(med_b[1]), float(med_b[2]), float(med_b[3])),
            dtype=float,
        )
    else:
        hz_b_med = Hz(zgrid, float(med_b[0]), float(med_b[1]))
        tgy_med_b = np.asarray(eng.cosmic_age_gyr(zgrid, float(med_b[0]), float(med_b[1])), dtype=float)
    tgy_med_l = np.asarray(eng.cosmic_age_gyr(zgrid, float(med_l[0]), float(med_l[1])), dtype=float)
    mu_l_shape, dL_l, _ = mu_lcdm_shape(zgrid, float(med_l[0]), float(med_l[1]), eng)
    mu_b_shape, _ = mu_blend_shape(zgrid, float(med_b[0]), float(med_b[1]), float(med_b[2]), float(med_b[3]), eng)
    mu_l_full = mu_l_shape + float(med_l[-1])
    mu_b_full = mu_b_shape + float(med_b[-1])
    dh_l = np.asarray(float(C_KMS) / np.maximum(np.asarray(hz_l, dtype=float), eps()), dtype=float)
    dh_b_m = np.asarray(float(C_KMS) / np.maximum(np.asarray(hz_b_med, dtype=float), eps()), dtype=float)
    mu_stack_pointwise_approx = waic_stack_w_lcdm * np.asarray(mu_l_full, dtype=float) + waic_stack_w_blend * np.asarray(mu_b_full, dtype=float)

    pred_joint = pd.DataFrame(
        {
            "z": zgrid,
            "H0_lcdm_median": float(med_l[0]),
            "Omega_m_lcdm_median": float(med_l[1]),
            "mu_LCDM_median": mu_l_full,
            "mu_LCDM_lo68": q16_l,
            "mu_LCDM_hi68": q84_l,
            "mu_LCDM_lo95": q025_l,
            "mu_LCDM_hi95": q975_l,
            "H0_blend_median": float(med_b[0]),
            "Omega_m_blend_median": float(med_b[1]),
            "t_crit_blend_median": float(med_b[2]),
            "k_blend_median": float(med_b[3]),
            "mu_blend_median": mu_b_full,
            "mu_blend_lo68": q16_b,
            "mu_blend_hi68": q84_b,
            "mu_blend_lo95": q025_b,
            "mu_blend_hi95": q975_b,
            "WAIC_stacking_weight_lcdm": waic_stack_w_lcdm,
            "WAIC_stacking_weight_blend": waic_stack_w_blend,
            "mu_WAIC_stacked_posterior_median_approx": mu_stack_pointwise_approx,
            "H_z_median_lcdm": hz_l,
            "H_z_median_blend_ibg": hz_b_med,
            "cosmic_time_Gyr_median_lcdm": tgy_med_l,
            "cosmic_time_Gyr_median_blend_ibg": tgy_med_b,
            "d_H_Mpc_lcdm": dh_l,
            "d_H_Mpc_blend_ibg": dh_b_m,
            "dL_Mpc_median_lcdm": dL_l,
            "Delta_mu_median": mu_b_full - mu_l_full,
            "Delta_mu_pred_lo68": d16,
            "Delta_mu_pred_hi68": d84,
            "Delta_mu_pred_lo95": d025,
            "Delta_mu_pred_hi95": d975,
        }
    )
    pred_joint.to_csv(OUTDIR / "posterior_predictive_joint_zgrid.csv", index=False)

    nh_samples = max(96, min(512, int(flat_b.shape[0])))
    ix_hz = RNG.choice(int(flat_b.shape[0]), size=int(min(nh_samples, int(flat_b.shape[0]))), replace=False)
    hz_samps: list[np.ndarray] = []
    tgy_samps: list[np.ndarray] = []
    for row_hz in flat_b[ix_hz]:
        h0r, omr = float(row_hz[0]), float(row_hz[1])
        if blend_physics_is_ccomplet2():
            tcr, kr = float(row_hz[2]), float(row_hz[3])
            hz_samps.append(np.asarray(eng.blend_Hz(zgrid, h0r, omr, tcr, kr), dtype=float))
            tgy_samps.append(np.asarray(eng.blend_cosmic_age_gyr(zgrid, h0r, omr, tcr, kr), dtype=float))
        else:
            hz_samps.append(np.asarray(Hz(zgrid, h0r, omr), dtype=float))
            tgy_samps.append(np.asarray(eng.cosmic_age_gyr(zgrid, h0r, omr), dtype=float))
    hz_mat = np.stack(hz_samps, axis=0)
    tgy_mat = np.stack(tgy_samps, axis=0)
    q_hz = np.quantile(hz_mat, [0.025, 0.16, 0.5, 0.84, 0.975], axis=0)
    q_tgy = np.quantile(tgy_mat, [0.025, 0.16, 0.5, 0.84, 0.975], axis=0)
    dht_mat = float(C_KMS) / np.maximum(hz_mat, eps())
    q_dht = np.quantile(dht_mat, [0.025, 0.16, 0.5, 0.84, 0.975], axis=0)
    exp_df = pd.DataFrame(
        {
            "z": zgrid,
            "Hz_median_blend_posterior": q_hz[2],
            "Hz_lo95": q_hz[0],
            "Hz_lo68": q_hz[1],
            "Hz_hi68": q_hz[3],
            "Hz_hi95": q_hz[4],
            "dH_Mpc_median_blend": float(C_KMS) / np.maximum(q_hz[2], eps()),
            "dH_Mpc_lo95": q_dht[0],
            "dH_Mpc_hi95": q_dht[4],
            "cosmic_time_Gyr_median_ibg": q_tgy[2],
            "cosmic_time_Gyr_lo95_ibg": q_tgy[0],
            "cosmic_time_Gyr_hi95_ibg": q_tgy[4],
            "dH_of_cosmic_time_median_Post": q_dht[2],
        }
    )
    exp_df.to_csv(OUTDIR / "expansion_Hz_DH_percentiles_blend.csv", index=False)

    if os.getenv("CWSF_H_CONSISTENCY_DIAG", "1") != "0":
        z_hi_d = float(max(0.05, min(1.85, z_plot_max)))
        diagnostic_hz_splinechi_vs_friedmann(
            np.linspace(0.01, z_hi_d, max(80, min(220, int(Z_NODES // 35)))),
            float(med_l[0]),
            float(med_l[1]),
            eng,
            FIGDIR / "diagnostic_Hz_splinechi_vs_Friedmann.png",
        )

    def plot_residuals(path: Path, zv: np.ndarray, muo: np.ndarray, sg: np.ndarray, theta_f: np.ndarray, blend_m: bool, title: str) -> None:
        th = np.asarray(theta_f, dtype=float).reshape(-1)
        if blend_m:
            if infer_profile_m():
                mu_tr, _ = mu_blend_shape(zt, float(th[0]), float(th[1]), float(th[2]), float(th[3]), eng)
                mm = profiled_absolute_magnitude(mu_tr, mt, st)
                mhat, _ = mu_blend(zv, float(th[0]), float(th[1]), float(th[2]), float(th[3]), mm, eng)
            else:
                mhat, _ = mu_blend(zv, float(th[0]), float(th[1]), float(th[2]), float(th[3]), float(th[4]), eng)
        elif infer_profile_m():
            mu_tr, _, _ = mu_lcdm_shape(zt, float(th[0]), float(th[1]), eng)
            mm = profiled_absolute_magnitude(mu_tr, mt, st)
            mhat, _, _ = mu_lcdm(zv, float(th[0]), float(th[1]), mm, eng)
        else:
            mhat, _, _ = mu_lcdm(zv, float(th[0]), float(th[1]), float(th[2]), eng)
        mhat = np.asarray(mhat, dtype=float)
        zv = np.asarray(zv, dtype=float)
        mo = np.asarray(muo, dtype=float)
        sgv = np.asarray(sg, dtype=float)
        okp = np.isfinite(mhat) & np.isfinite(mo) & np.isfinite(sgv)
        res = mo - mhat
        plt.figure(figsize=(7.0, 4.5))
        plt.errorbar(zv[okp], res[okp], yerr=sgv[okp], fmt="o", ms=2, alpha=0.35, elinewidth=0.5)
        plt.axhline(0.0, color="k", lw=0.8)
        plt.xlabel("z")
        plt.ylabel(r"$\mu_{\mathrm{obs}}-\mu_{\mathrm{model}}$")
        plt.title(title)
        plt.tight_layout()
        plt.savefig(path, dpi=160)
        plt.close()

    plot_residuals(FIGDIR / "pantheon_residuals_lcdm.png", zt, mt, st, med_l, False, "Pantheon+ residuals (LCDM, median+M_profile)")
    plot_residuals(FIGDIR / "pantheon_residuals_blend.png", zt, mt, st, med_b, True, "Pantheon+ residuals (blend, median+M_profile)")
    plot_residuals(
        FIGDIR / "des_residuals_lcdm.png", z_des, m_des, s_des, med_l, False, "DES hold-out (LCDM, M_profile on Pantheon)"
    )
    plot_residuals(
        FIGDIR / "des_residuals_blend.png", z_des, m_des, s_des, med_b, True, "DES hold-out (blend, M_profile on Pantheon)"
    )

    for model_name, theta_f, blend_ff in (
        ("lcdm", med_l, False),
        ("blend", med_b, True),
    ):
        if blend_ff:
            mhat_train, _ = mu_blend(zt, float(theta_f[0]), float(theta_f[1]), float(theta_f[2]), float(theta_f[3]), float(theta_f[4]), eng)
        elif infer_profile_m():
            mu_tt, _, _ = mu_lcdm_shape(zt, float(theta_f[0]), float(theta_f[1]), eng)
            mhat_train = mu_tt + float(theta_f[2])
        else:
            mhat_train, _, _ = mu_lcdm(zt, float(theta_f[0]), float(theta_f[1]), float(theta_f[2]), eng)
        rs = mt - np.asarray(mhat_train, dtype=float)
        bf = FIGDIR / f"pantheon_binned_residuals_{model_name}.png"
        bdf = binned_residual_summary(zt, rs, st, n_bins=4)
        bdf.to_csv(OUTDIR / f"pantheon_binned_residuals_{model_name}.csv", index=False)
        plt.figure(figsize=(6.5, 4.2))
        plt.bar(bdf["z_mid"], bdf["mean_residual"], width=np.maximum(0.01, (bdf["z_hi"] - bdf["z_lo"]) * 0.85), alpha=0.75)
        plt.axhline(0.0, color="k", lw=0.8)
        plt.xlabel("redshift bin (median)")
        plt.ylabel("mean residual (mag)")
        plt.title(f"Pantheon+ binned mean residuals ({model_name})")
        plt.tight_layout()
        plt.savefig(bf, dpi=160)
        plt.close()

    plt.figure(figsize=(7.0, 4.5))
    plt.fill_between(zgrid, q025_l, q975_l, color="C0", alpha=0.15, label="LCDM 95%")
    plt.fill_between(zgrid, q16_l, q84_l, color="C0", alpha=0.35, label="LCDM 68%")
    plt.plot(zgrid, q50_l, color="C0", lw=1.8, label="LCDM predictive median")
    plt.fill_between(zgrid, q025_b, q975_b, color="C1", alpha=0.15, label="Blend 95%")
    plt.fill_between(zgrid, q16_b, q84_b, color="C1", alpha=0.35, label="Blend 68%")
    plt.plot(zgrid, q50_b, color="C1", lw=1.8, label="Blend predictive median")
    plt.scatter(zt, mt, s=3, alpha=0.15, color="0.35", label="Pantheon+ training")
    plt.xlabel("z")
    plt.ylabel(r"$\mu$")
    plt.title("Posterior predictive Hubble diagram (bands from posterior draws)")
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGDIR / "hubble_training_with_bands.png", dpi=160)
    plt.close()

    plt.figure(figsize=(7.0, 4.0))
    plt.fill_between(zgrid, d025, d975, color="purple", alpha=0.18, label=r"$\Delta\mu$ 95% band")
    plt.fill_between(zgrid, d16, d84, color="purple", alpha=0.32, label=r"$\Delta\mu$ 68% band")
    plt.plot(zgrid, mu_b_full - mu_l_full, color="purple", lw=2.0, label=r"$\Delta\mu$ at posterior median")
    plt.xlabel("z")
    plt.ylabel(r"$\Delta\mu$ (blend$-$LCDM) [mag]")
    plt.title(r"Horizon-area correction $\Delta\mu(z)$ relative to LCDM baseline")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGDIR / "delta_mu_curve.png", dpi=160)
    plt.close()

    plt.figure(figsize=(7.0, 4.2))
    plt.plot(zgrid, mu_l_full, color="C0", lw=2.0, label=r"Pure LCDM @ median $\theta$")
    plt.plot(zgrid, mu_b_full, color="C1", lw=2.0, label=r"Pure blend @ median $\theta$")
    plt.plot(
        zgrid,
        mu_stack_pointwise_approx,
        color="darkgreen",
        lw=2.4,
        ls="--",
        label=r"WAIC-weighted pointwise stack (median-$\theta$ curves)",
    )
    plt.scatter(zt, mt, s=3, alpha=0.14, color="0.35", label="Pantheon+ training")
    plt.xlabel("z")
    plt.ylabel(r"$\mu(z)$")
    plt.title(r"Pure models vs approximate WAIC-stacked predictive mean (training summary)")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGDIR / "hubble_pure_vs_waic_stacked.png", dpi=160)
    plt.close()

    plt.figure(figsize=(7.0, 4.5))
    plt.plot(zgrid, q_hz[2], color="black", lw=2.2, label=r"H(z) posterior median")
    plt.fill_between(zgrid, q_hz[1], q_hz[3], color="steelblue", alpha=0.32, label="68% posterior")
    plt.fill_between(zgrid, q_hz[0], q_hz[4], color="steelblue", alpha=0.15, label="95% posterior")
    plt.xlabel("z")
    plt.ylabel(r"H(z) [km s$^{-1}$ Mpc$^{-1}$]")
    plt.title("Expansion history H(z): blend-parameter posterior ensemble")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGDIR / "expansion_hz_percentiles_blend.png", dpi=160)
    plt.close()

    plt.figure(figsize=(6.8, 4.8))
    plt.plot(np.asarray(q_tgy[2], dtype=float), np.asarray(q_hz[2], dtype=float), color="darkred", lw=2.3, label="Median (t, H) along z grid")
    plt.fill_betweenx(
        np.asarray(q_hz[2], dtype=float),
        np.asarray(q_tgy[1], dtype=float),
        np.asarray(q_tgy[3], dtype=float),
        alpha=0.22,
        color="firebrick",
        label="t(z) 68% band",
    )
    plt.xlabel(r"Cosmic time $t(z)$ [Gyr] (blend posterior, isotropic background)")
    plt.ylabel(r"$H(z)$ [km s$^{-1}$ Mpc$^{-1}$]")
    plt.title(r"$H$ along the FRW worldline vs cosmic time (percentiles from blend posterior)")
    plt.grid(alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGDIR / "expansion_Hz_vs_cosmic_time_blend.png", dpi=160)
    plt.close()

    zs_tab_plot = pd.read_csv(OUTDIR / "training_redshift_sliced_residuals.csv")
    if len(zs_tab_plot) > 0:
        plt.figure(figsize=(7.2, 4.4))
        plt.plot(zs_tab_plot["z_mid"], zs_tab_plot["rms_residual_lcdm"], marker="s", ls="-", label="LCDM RMS |resid| bin")
        plt.plot(zs_tab_plot["z_mid"], zs_tab_plot["rms_residual_blend"], marker="o", ls="-", label="Blend RMS |resid| bin")
        plt.axhline(0.0, color="k", lw=0.6)
        plt.xlabel("training bin median z")
        plt.ylabel("rms residual mag")
        plt.title("Training redshift-sliced residuals (posterior median reconstructions)")
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(FIGDIR / "training_redshift_sliced_rms_residuals.png", dpi=160)
        plt.close()

    if len(zs_tab_plot) > 0 and "mean_delta_mu_blend_minus_lcdm_mag" in zs_tab_plot.columns:
        plt.figure(figsize=(7.0, 4.2))
        plt.axhline(0.0, color="k", lw=0.8)
        plt.plot(
            zs_tab_plot["z_mid"],
            zs_tab_plot["mean_delta_mu_blend_minus_lcdm_mag"],
            marker="D",
            ms=4,
            color="purple",
            label=r"Bin-mean $\Delta\mu$ (blend$-$LCDM) at median $\theta$",
        )
        plt.xlabel("training bin median z")
        plt.ylabel(r"$\langle\Delta\mu\rangle$ [mag]")
        plt.title("Redshift-sliced mean horizon correction (training, median parameters)")
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(FIGDIR / "training_redshift_sliced_mean_delta_mu.png", dpi=160)
        plt.close()

    cv_fold_metrics_path_plot = OUTDIR / "cross_validation_fold_metrics.csv"
    if cv_fold_metrics_path_plot.exists() and RUN_CV and bool(cv_summary.get("ran")):
        plt.figure(figsize=(6.0, 3.8))
        cv_read = pd.read_csv(cv_fold_metrics_path_plot)
        mean_l = cv_read.groupby("repeat")["rmse_lcdm"].mean()
        mean_b = cv_read.groupby("repeat")["rmse_blend"].mean()
        xpos = np.arange(len(mean_l))
        w_ = 0.36
        plt.bar(xpos - w_ / 2, np.asarray(mean_l, dtype=float), width=w_, label="LCDM", color="C0", alpha=0.85)
        plt.bar(xpos + w_ / 2, np.asarray(mean_b, dtype=float), width=w_, label="Blend", color="C1", alpha=0.85)
        plt.xticks(xpos, [f"rep {int(i)}" for i in mean_l.index.tolist()])
        plt.ylabel("mean fold RMSE (mag)")
        plt.title("Pantheon+ internal CV (validation folds; abbreviated MCMC)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(FIGDIR / "cv_rmse_by_repeat_mean.png", dpi=160)
        plt.close()

        plt.figure(figsize=(8.5, 4.6))
        xf = cv_read["fold"].astype(float) + 0.12 * cv_read["repeat"].astype(float)
        plt.scatter(xf, cv_read["rmse_lcdm"], c="C0", s=22, alpha=0.62, marker="s", edgecolors="none", label="LCDM val RMSE")
        plt.scatter(xf, cv_read["rmse_blend"], c="C1", s=22, alpha=0.62, marker="o", edgecolors="none", label="Blend val RMSE")
        plt.xlabel("fold (+ small repeat jitter)")
        plt.ylabel("validation RMSE (mag)")
        plt.title("Pantheon+ CV folds: LCDM vs blend (each point is one validation fold)")
        plt.legend(markerscale=1.25)
        plt.grid(alpha=0.23)
        plt.tight_layout()
        plt.savefig(FIGDIR / "cv_rmse_fold_scatter_lcdm_vs_blend.png", dpi=160)
        plt.close()

    if nested_summary.get("ran") is True:
        lz = nested_summary["lcdm"]["logz"]
        bz = nested_summary["blend"]["logz"]
        plt.figure(figsize=(4.8, 3.8))
        plt.bar([r"$\Lambda$CDM", "blend"], [lz, bz], color=["C0", "C1"], alpha=0.85)
        plt.ylabel(r"log marginal likelihood estimate ($\ln \hat{Z}$)")
        errs = np.array([nested_summary["lcdm"]["logzerr"], nested_summary["blend"]["logzerr"]], dtype=float)
        plt.errorbar([0.0, 1.0], [lz, bz], yerr=errs, fmt="none", ecolor="0.35", capsize=4)
        plt.title(r"Nested sampling evidence (Gaussian likelihood; truncated Gaussian priors)")
        plt.grid(axis="y", alpha=0.25)
        plt.tight_layout()
        plt.savefig(FIGDIR / "nested_evidence_logz_bar.png", dpi=160)
        plt.close()

    mhat_lcdm_med, _, _ = mu_lcdm(zt, float(med_l[0]), float(med_l[1]), float(med_l[-1]), eng)
    rsd = np.asarray(mt - mhat_lcdm_med, dtype=float)
    rss = np.sort(rsd / st)
    qth = stats.norm.ppf(np.linspace(0.5 / len(rss), 1.0 - 0.5 / len(rss), len(rss)))
    plt.figure(figsize=(5.0, 5.0))
    plt.scatter(qth, rss, s=18, alpha=0.65)
    lo_, hi_ = float(min(qth[0], rss[0])), float(max(qth[-1], rss[-1]))
    plt.plot([lo_, hi_], [lo_, hi_], color="k", lw=1.0)
    plt.xlabel("Normal quantiles")
    plt.ylabel(r"Residual / $\sigma$ (LCDM median fit)")
    plt.title("Training Q-Q (Pantheon+)")
    plt.tight_layout()
    plt.savefig(FIGDIR / "qq_training_residual_lcdm.png", dpi=160)
    plt.close()

    mhat_blend_med, _ = mu_blend(zt, float(med_b[0]), float(med_b[1]), float(med_b[2]), float(med_b[3]), float(med_b[-1]), eng)
    rsdb = np.asarray(mt - mhat_blend_med, dtype=float)
    rssb = np.sort(rsdb / st)
    qthb = stats.norm.ppf(np.linspace(0.5 / len(rssb), 1.0 - 0.5 / len(rssb), len(rssb)))
    plt.figure(figsize=(5.0, 5.0))
    plt.scatter(qthb, rssb, s=18, alpha=0.65, color="C1")
    lo_b, hi_b = float(min(qthb[0], rssb[0])), float(max(qthb[-1], rssb[-1]))
    plt.plot([lo_b, hi_b], [lo_b, hi_b], color="k", lw=1.0)
    plt.xlabel("Normal quantiles")
    plt.ylabel(r"Residual / $\sigma$ (blend median fit)")
    plt.title("Training Q-Q (Pantheon+, blend)")
    plt.tight_layout()
    plt.savefig(FIGDIR / "qq_training_residual_blend.png", dpi=160)
    plt.close()

    n_ppc_chi = int(os.getenv("CWSF_PPC_CHI2_DRAWS", "640"))
    rep_chi_l = ppc_std_residual_chi2_replicas(flat_l, RNG, False, eng, zt, mt, st, n_draws=n_ppc_chi)
    rep_chi_b = ppc_std_residual_chi2_replicas(flat_b, RNG, True, eng, zt, mt, st, n_draws=n_ppc_chi)
    obs_chi_l = observed_training_std_residual_chi2(med_l, False, eng, zt, mt, st)
    obs_chi_b = observed_training_std_residual_chi2(med_b, True, eng, zt, mt, st)

    def _ppc_tail_p(reps: np.ndarray, obs: float) -> float | None:
        rr = np.asarray(reps, dtype=float)
        if not rr.size:
            return None
        return json_safe_float(float(np.mean(rr >= float(obs))))

    ppc_chi2_diag = dict(
        lcdm=dict(
            observed_chi2_std_sum=json_safe_float(float(obs_chi_l)),
            replica_median=json_safe_float(float(np.median(rep_chi_l))) if rep_chi_l.size else None,
            replica_q025=json_safe_float(float(np.quantile(rep_chi_l, 0.025))) if rep_chi_l.size else None,
            replica_q975=json_safe_float(float(np.quantile(rep_chi_l, 0.975))) if rep_chi_l.size else None,
            posterior_predictive_p_value_right_tail=_ppc_tail_p(rep_chi_l, obs_chi_l),
            n_draws=int(rep_chi_l.size),
        ),
        blend=dict(
            observed_chi2_std_sum=json_safe_float(float(obs_chi_b)),
            replica_median=json_safe_float(float(np.median(rep_chi_b))) if rep_chi_b.size else None,
            replica_q025=json_safe_float(float(np.quantile(rep_chi_b, 0.025))) if rep_chi_b.size else None,
            replica_q975=json_safe_float(float(np.quantile(rep_chi_b, 0.975))) if rep_chi_b.size else None,
            posterior_predictive_p_value_right_tail=_ppc_tail_p(rep_chi_b, obs_chi_b),
            n_draws=int(rep_chi_b.size),
        ),
        note="Replica χ² uses synthetic SN around each posterior draw’s mean; observed uses the same standardized residual sum on data.",
    )
    mlen_rep = min(int(rep_chi_l.size), int(rep_chi_b.size))
    if mlen_rep > 0:
        pd.DataFrame(
            dict(
                lcdm_replica_chi2=rep_chi_l[:mlen_rep],
                blend_replica_chi2=rep_chi_b[:mlen_rep],
            )
        ).to_csv(OUTDIR / "posterior_predictive_chi2_replicas_matched_length.csv", index=False)

    fig_p, ax_p = plt.subplots(1, 2, figsize=(10.0, 4.2))
    if int(rep_chi_l.size) > 2:
        ax_p[0].hist(rep_chi_l, bins=32, density=True, alpha=0.78, color="C0", label="PPC replicas")
    ax_p[0].axvline(obs_chi_l, color="k", lw=2.0, label="Observed")
    ax_p[0].set_title(r"LCDM: $\Sigma\,[(y-\mu)/\sigma]^2$")
    ax_p[0].set_xlabel(r"$\chi^2_{\mathrm{std}}$ (diag)")
    ax_p[0].legend(fontsize=7)
    if int(rep_chi_b.size) > 2:
        ax_p[1].hist(rep_chi_b, bins=32, density=True, alpha=0.78, color="C1", label="PPC replicas")
    ax_p[1].axvline(obs_chi_b, color="k", lw=2.0, label="Observed")
    ax_p[1].set_title(r"Blend: $\Sigma\,[(y-\mu)/\sigma]^2$")
    ax_p[1].set_xlabel(r"$\chi^2_{\mathrm{std}}$ (diag)")
    ax_p[1].legend(fontsize=7)
    fig_p.suptitle("Posterior predictive χ² probes (Gaussian training scatter)", fontsize=11)
    fig_p.tight_layout()
    fig_p.savefig(FIGDIR / "posterior_predictive_chi2_histograms.png", dpi=160)
    plt.close(fig_p)

    def traces_plot(ch3_concat: np.ndarray, labels: Sequence[str], outpath: Path) -> None:
        nstep, nw, dd = int(ch3_concat.shape[0]), int(ch3_concat.shape[1]), int(ch3_concat.shape[2])
        fig, axes = plt.subplots(dd, 1, figsize=(7.5, min(14.0, 2.8 * dd)), sharex=True)
        axes = np.atleast_1d(axes)
        tt_ax = np.arange(nstep)
        for di in range(dd):
            axes[di].plot(tt_ax, ch3_concat[:, :, di], lw=0.35, alpha=0.35)
            axes[di].set_ylabel(labels[di], fontsize=9)
        axes[-1].set_xlabel("production step index")
        fig.suptitle("Trace plots (thin lines: walkers)")
        fig.tight_layout(rect=[0, 0.02, 1, 0.97])
        fig.savefig(outpath, dpi=140)
        plt.close(fig)

    traces_plot(np.concatenate(ch3_list_l, axis=0), names_l_cols, FIGDIR / "traces_lcdm.png")
    traces_plot(np.concatenate(ch3_list_b, axis=0), names_b_cols, FIGDIR / "traces_blend.png")

    lcdm_lab = (
        [r"$H_0$", r"$\Omega_m$"] if infer_profile_m() else [r"$H_0$", r"$\Omega_m$", r"$M$"]
    )

    if HAVE_CORNER and corner is not None:
        sub = min(8000, int(flat_l.shape[0]))
        ix = RNG.choice(int(flat_l.shape[0]), size=sub, replace=False)
        fig = corner.corner(flat_l[ix], labels=lcdm_lab, quantiles=[0.16, 0.5, 0.84], show_titles=True)
        fig.savefig(FIGDIR / "corner_lcdm.png", dpi=140)
        plt.close(fig)
        ix2 = RNG.choice(int(flat_b.shape[0]), size=sub, replace=False)
        blend_lab = (
            [r"$H_0$", r"$\Omega_m$", r"$t_{\mathrm{crit}}$", r"$k$"] if infer_profile_m() else [r"$H_0$", r"$\Omega_m$", r"$t_{\mathrm{crit}}$", r"$k$", r"$M$"]
        )
        fig2 = corner.corner(flat_b[ix2], labels=blend_lab, quantiles=[0.16, 0.5, 0.84], show_titles=True)
        fig2.savefig(FIGDIR / "corner_blend.png", dpi=140)
        plt.close(fig2)

    n_long = max(2, min(int(LONG_TABLE_DRAWS), int(flat_l.shape[0])))
    ix_long = RNG.choice(int(flat_b.shape[0]), size=min(n_long, int(flat_b.shape[0])), replace=False)
    long_rows: list[dict] = []
    for j, dj in enumerate(ix_long):
        row = flat_b[dj]
        thf = theta_with_M_from_row(row, True)
        h0, om, tc, kk, mm = float(thf[0]), float(thf[1]), float(thf[2]), float(thf[3]), float(thf[4])
        mu_lcdm_tr, _, _ = mu_lcdm_shape(zt, h0, om, eng)
        mu_blend_tr, _ = mu_blend_shape(zt, h0, om, tc, kk, eng)
        if infer_profile_m():
            M_lcdm_prof = profiled_absolute_magnitude(mu_lcdm_tr, mt, st)
            M_blend_prof = profiled_absolute_magnitude(mu_blend_tr, mt, st)
        else:
            M_lcdm_prof = float(mm)
            M_blend_prof = float(mm)
        fw_id = int(cid_b[int(dj)]) if cid_b.ndim == 1 else 0
        for zk in zgrid[:: max(1, len(zgrid) // 48)]:
            zv1 = np.asarray([float(zk)], dtype=float)
            mu_l_s, dLk, _ = mu_lcdm_shape(zv1, h0, om, eng)
            mu_b_s, ex_b = mu_blend_shape(zv1, h0, om, tc, kk, eng)
            mu_l = float(mu_l_s[0]) + float(M_lcdm_prof)
            mu_b = float(mu_b_s[0]) + float(M_blend_prof)
            Hz_z_arr = Hz(zv1, h0, om)
            hz_here = float(Hz_z_arr[0])
            tgy_z_br = float(ex_b["tgy"][0])
            w_br = float(ex_b["w"][0])
            z_of_t_here = float(
                z_at_age_gyr(eng, float(h0), float(om), np.asarray([tgy_z_br], dtype=float), float(np.maximum(z_plot_max, 0.08)))[
                    0
                ]
            )
            hh_t_here = float(Hz(np.asarray([z_of_t_here], dtype=float), h0, om)[0])
            long_rows.append(
                dict(
                    sampler_type="emcee",
                    framework_seed_run_id=fw_id,
                    pooled_draw_storage_index=int(dj),
                    table_draw_id=j,
                    model_name="blend",
                    H0=h0,
                    Omega_m=om,
                    Omega_Lambda=float(olambda_flat(om, h0)),
                    t_crit=tc,
                    k=kk,
                    M_lcdm_profiled=float(M_lcdm_prof),
                    M_blend_profiled=float(M_blend_prof),
                    z=float(zk),
                    cosmic_time_Gyr=float(tgy_z_br),
                    branch_weight_w=float(w_br),
                    branch_dominates=("thermo_entropy" if w_br >= 0.5 else "gravity_horizon_area"),
                    H_z=float(hz_here),
                    H_cosmic_age_branch=float(hh_t_here),
                    d_H_Mpc_z=float(float(C_KMS) / max(hz_here, eps())),
                    d_H_Mpc_cosmic_age_branch=float(float(C_KMS) / max(hh_t_here, eps())),
                    d_L_Mpc=float(dLk[0]),
                    mu_LCDM_mag=float(mu_l),
                    mu_blend_mag=float(mu_b),
                    Delta_mu_mag=float(mu_b - mu_l),
                    residual_lcdm_na_at_grid=float("nan"),
                    residual_blend_na_at_grid=float("nan"),
                    z_mapped_from_cosmic_age_for_Ht=z_of_t_here,
                )
            )

    df_long_tbl = pd.DataFrame(long_rows)
    df_long_tbl.to_csv(OUTDIR / "posterior_cosmo_longform_emcee_blend_draws.csv", index=False)

    def edge_report(flat: np.ndarray, lows: np.ndarray, highs: np.ndarray) -> dict:
        return dict(
            frac_low=np.asarray(np.mean((flat <= lows + 1e-4 * (highs - lows)), axis=0), dtype=float).tolist(),
            frac_high=np.asarray(np.mean((flat >= highs - 1e-4 * (highs - lows)), axis=0), dtype=float).tolist(),
        )

    edge_l = edge_report(flat_l, lcdm_lows, lcdm_highs)
    edge_b = edge_report(flat_b, blend_lows, blend_highs)

    warn: list[str] = []
    warn.extend(external_like_warnings)
    try:
        plot_q_deceleration_lcdm(FIGDIR / "diagnostic_deceleration_qz_lcdm.png", float(med_l[0]), float(med_l[1]))
    except Exception as ex_qz:
        warn.append(f"q_of_z_diagnostic_failed:{ex_qz!s}")
    rhbl = json_safe_mx(rhat_between_l)
    rhbb = json_safe_mx(rhat_between_b)
    if (rhbl is not None and float(rhbl) > RHAT_TARGET) or float(np.nanmax(rhat_split_l)) > RHAT_TARGET:
        warn.append(
            "LCDM: pooled / split-chain R-hat exceeds target; marginal posteriors should be treated as preliminary until longer MCMC raises ESS."
        )
    if (rhbb is not None and float(rhbb) > RHAT_TARGET) or float(np.nanmax(rhat_split_b)) > RHAT_TARGET:
        warn.append("Blend chain shows R-hat above target.")
    if (not math.isnan(ess_min_l)) and ess_min_l < MIN_ESS_TARGET:
        warn.append("LCDM effective samples (conservative ESS via max tau over parameters) lie below MIN_ESS_TARGET.")
    if (not math.isnan(ess_min_b)) and ess_min_b < MIN_ESS_TARGET:
        warn.append("Blend ESS below MIN_ESS_TARGET.")
    if np.max(edge_l["frac_low"] + edge_l["frac_high"]) > 0.02:
        warn.append("LCDM: notable posterior mass touching hard boundaries (see edge_report). Possible prior dominance.")
    if np.max(edge_b["frac_low"] + edge_b["frac_high"]) > 0.02:
        warn.append("Blend: posterior mass near hard bounds.")

    if HAVE_PARQUET:
        try:
            df_long_tbl.to_parquet(OUTDIR / "posterior_cosmo_longform_emcee_blend_draws.parquet", index=False)
        except Exception as ex_pf:
            warn.append(f"Parquet long-form supplement failed ({ex_pf!s}); CSV remains canonical.")

    ppc_l = dict(
        diagonal_gaussian=ppc_train_stats(flat_l, RNG, False, eng, zt, mt, st, n_draws=256),
        replicated_residual_shape_moments=ppc_train_distribution_stats(
            flat_l, RNG, False, eng, zt, mt, st, n_draws=288
        ),
    )
    ppc_b = dict(
        diagonal_gaussian=ppc_train_stats(flat_b, RNG, True, eng, zt, mt, st, n_draws=256),
        replicated_residual_shape_moments=ppc_train_distribution_stats(
            flat_b, RNG, True, eng, zt, mt, st, n_draws=288
        ),
    )

    fw_stack_arr = np.stack(med_cosmo_blend_fw, axis=0) if len(med_cosmo_blend_fw) else np.empty((0, nd_b))
    h0_std_fw = (
        float(np.std(fw_stack_arr[:, 0], ddof=1))
        if int(fw_stack_arr.shape[0]) > 2
        else float("nan")
    )
    Om_std_fw = (
        float(np.std(fw_stack_arr[:, 1], ddof=1))
        if int(fw_stack_arr.shape[0]) > 2
        else float("nan")
    )

    nested_dlz = nested_summary.get("delta_logz_blend_minus_lcdm")
    dlz_num = nested_dlz
    if dlz_num is None:
        evidence_vs_lcdm_blend_mild_ok = True
    else:
        evidence_vs_lcdm_blend_mild_ok = bool(math.isfinite(float(dlz_num)) and float(dlz_num) >= -6.0)
    blend_win_cv_fraction = cv_summary.get("mean_fraction_folds_where_blend_rmse_below_lcdm_across_repeats")
    try:
        cv_competitive = blend_win_cv_fraction is None or float(blend_win_cv_fraction) >= 0.42
    except Exception:
        cv_competitive = blend_win_cv_fraction is None
    boundary_pressure = bool(np.max(edge_b["frac_low"] + edge_b["frac_high"]) > 0.02)

    stable_fw_seeds = bool((not math.isnan(h0_std_fw)) and h0_std_fw <= 1.2 and Om_std_fw <= 0.04)

    better_holdout = bool(des_b_med["rmse"] < des_l_med["rmse"])
    delta_aic = float(aic_b - aic_l)
    delta_bic = float(bic_b - bic_l)
    mild_penalty = bool(delta_aic < 10.0 and delta_bic < 10.0)
    publication_blend_claim = bool(
        better_holdout
        and mild_penalty
        and stable_fw_seeds
        and evidence_vs_lcdm_blend_mild_ok
        and cv_competitive
        and (not boundary_pressure)
    )
    if publication_blend_claim:
        interpret = (
            "Strict joint-diagnostic verdict: Blend outperforms LCDM under the tightened framework "
            "(DES RMSE gains, temperate IC deltas, repeatable multi-emcee seeds, favourable CV/evidence summaries, negligible boundary stacking)."
        )
    elif better_holdout and mild_penalty:
        interpret = (
            "Predictive residuals favor the blend on DES with mild IC penalties, yet supporting robustness/evidence facets flag residual tension; describe the blend as competitive."
        )
    elif not better_holdout:
        interpret = (
            "LCDM satisfies or surpasses DES hold-out metrics alongside simpler structure; prioritize transparent reporting of blended physics as an explanatory alternative."
        )
    else:
        interpret = (
            "Improvements appear primarily in training-style observables rather than coherent DES superiority; withhold strong blend preference language."
        )

    man = OUTDIR / "repro_manifest.txt"
    man.write_text(
        "\n".join(
            [
                f"PANT_URL={PANT_URL}",
                f"DES_URL={DES_URL}",
                f"PROFILE_M={int(PROFILE_M)}",
                f"RNG_SEED={RNG_SEED} framework_runs={N_FRAMEWORK_SEEDS}",
                f"walkers={N_WALKERS} chains={N_CHAINS} burn={N_BURN} prod={N_PROD}",
                json.dumps(versions_dict(), indent=2),
            ]
        ),
        encoding="utf-8",
    )

    med_l_ol = olambda_flat(float(med_l[1]), float(med_l[0]))
    med_b_ol = olambda_flat(float(med_b[1]), float(med_b[0]))
    med_l_cosmo = {names_l_cols[i]: float(med_l_raw[i]) for i in range(len(names_l_cols))}
    med_b_cosmo = {names_b_cols[i]: float(med_b_raw[i]) for i in range(len(names_b_cols))}

    validation_falsification_bundle: dict[str, Any] = dict(disabled=not (VALIDATION_SUITE and HAVE_EMCEE))
    if VALIDATION_SUITE and HAVE_EMCEE:
        try:
            validation_falsification_bundle = execute_validation_falsification_suite(
                dict(
                    OUTDIR=OUTDIR,
                    FIGDIR=FIGDIR,
                    eng=eng,
                    RNG=RNG,
                    zt=zt,
                    mt=mt,
                    st=st,
                    z_des=z_des,
                    m_des=m_des,
                    s_des=s_des,
                    flat_l=flat_l,
                    flat_b=flat_b,
                    med_l=med_l,
                    med_b=med_b,
                    names_l_cols=names_l_cols,
                    names_b_cols=names_b_cols,
                    blend_lows=blend_lows,
                    blend_highs=blend_highs,
                    blend_centers_adapt=blend_centers_adapt,
                    blend_widths_adapt=blend_widths_adapt,
                    lcdm_lows=lcdm_lows,
                    lcdm_highs=lcdm_highs,
                    lcdm_centers=lcdm_centers,
                    lcdm_widths=lcdm_widths,
                    nd_l=int(nd_l),
                    nd_b=int(nd_b),
                    k_l=int(k_l),
                    k_b=int(k_b),
                    mcmc_ctx=dict(
                        use_cov=bool(_use_cov_eff),
                        chol_sig=chol_sig,
                        logdet_twopisigma=float(logdet_twopisigma),
                    ),
                    chi2_med_l=float(chi2_med_l),
                    chi2_med_b=float(chi2_med_b),
                    redchi_l=float(redchi_l),
                    redchi_b=float(redchi_b),
                    aic_l=float(aic_l),
                    aic_b=float(aic_b),
                    bic_l=float(bic_l),
                    bic_b=float(bic_b),
                    waic_l=float(waic_l),
                    waic_b=float(waic_b),
                    des_l_med=des_l_med,
                    des_b_med=des_b_med,
                    nested_summary=nested_summary,
                    cv_summary=cv_summary,
                    edge_b=edge_b,
                    better_holdout=bool(better_holdout),
                    publication_blend_claim=bool(publication_blend_claim),
                )
            )
        except Exception as ex_vb:
            validation_falsification_bundle = dict(error=str(ex_vb), enabled=False)
            warn.append(f"validation_falsification_bundle_outer:{ex_vb!s}")

    summary = dict(
        versions=versions_dict(),
        validation_falsification_suite=validation_falsification_bundle,
        dataset_urls=dict(pantheon=PANT_URL, des=DES_URL),
        external_joint_likelihoods=dict(
            enabled=bool(HAVE_EXTERNAL_LIKES and _pack_ext is not None),
            disclosure=getattr(_pack_ext, "disclosure", None) if HAVE_EXTERNAL_LIKES and _pack_ext is not None else None,
            cross_probe_table=(
                str((OUTDIR / "cross_probe_metrics.csv").resolve()) if HAVE_EXTERNAL_LIKES and _pack_ext is not None else None
            ),
            sparc_separate_benchmark_csv=(
                str((OUTDIR / "sparc_mond_per_galaxy_chi2.csv").resolve()) if os.getenv("CWSF_SPARC_DIR", "").strip() else None
            ),
        ),
        joint_likelihood_and_scope=dict(
            log_posterior_decomposition=(
                "log_post = log_prior + logL_SN + logL_BAO + logL_CMB when the external joint pack is active "
                "(LCDM and blend share the same additive BAO/CMB Gaussian blocks; the blend path alters only the SN modulus likelihood)."
            ),
            what_is_not_claimed=(
                "Embedded BAO/CMB are literature-scaled compressed Gaussians for reproducible pedagogy — not the Planck Plik / "
                "SDSS collaboration likelihood executables. Parameter posteriors are not interchangeable with CosmoMC chains unless you replace these blocks."
            ),
            references=dict(planck_2018_VI="arXiv:1807.06209", sdss_dr16_public="https://www.sdss.org/"),
        ),
        inference=dict(
            profiled_M=bool(infer_profile_m()),
            low_z_luminosity_distance_anchor_active=USE_LOWZ_DL_ANCHOR,
            blend_horizon_delta_mu_correction_active=USE_BLEND_HORIZON_DELTA_MU,
            covariance_likelihood=bool(_use_cov_eff),
            lcdm_dimensions=int(nd_l),
            blend_dimensions=int(nd_b),
            n_framework_mc_runs=int(N_FRAMEWORK_SEEDS),
            internal_emcee_parallel_chains=int(N_CHAINS),
            acceptance_lcdm=json_safe_float(float(np.mean(acc_rates_l))),
            acceptance_blend=json_safe_float(float(np.mean(acc_rates_b))),
            rhat_between_inner_emcee_chains_lcdm=[json_safe_float(float(x)) for x in np.ravel(rhat_between_l).tolist()],
            rhat_between_inner_emcee_chains_blend=[json_safe_float(float(x)) for x in np.ravel(rhat_between_b).tolist()],
            rhat_between_framework_runs_lcdm=[json_safe_float(float(x)) for x in np.ravel(rhat_between_framework_l).tolist()],
            rhat_between_framework_runs_blend=[json_safe_float(float(x)) for x in np.ravel(rhat_between_framework_b).tolist()],
            rhats_target=float(RHAT_TARGET),
            mintarget_ess=float(MIN_ESS_TARGET),
        ),
        truncated_gaussian_prior_box=dict(
            legacy_wide_priors=os.getenv("CWSF_LEGACY_PRIORS", "0") == "1",
            PRI_MU=PRI_MU,
            PRI_SIGMA=PRI_SG,
            HARD_BOUNDS=BND,
            magnitude_prior_center=PRI_MAG_MU,
            magnitude_prior_sigma=float(PRI_MAG_SG),
            note=(
                "Tight priors emulate the cosmology framework physics document; activate CWSF_LEGACY_PRIORS=1 for exploratory wide boxes "
                "(not recommended when claiming publication robustness)."
            ),
        ),
        framework_seed_stability_blend=dict(
            median_of_medians_H0=(
                json_safe_float(float(np.median(fw_stack_arr[:, 0]))) if fw_stack_arr.size > 0 else None
            ),
            median_of_medians_Omega_m=(
                json_safe_float(float(np.median(fw_stack_arr[:, 1]))) if fw_stack_arr.size > 0 else None
            ),
            std_of_medians_H0_across_runs=json_safe_float(h0_std_fw),
            std_of_medians_Omega_m_across_runs=json_safe_float(Om_std_fw),
            flagged_unstable_under_thresholds=dict(H0_kmsMpc_sigma_max_for_stability_report=1.2, Omega_m_sigma_max=0.04),
        ),
        likelihood_and_importance=dict(
            covariance_state=(
                "full_STATONLY_aligned_by_CID_perfect_submatrix_when_CWSF_USE_COV=1"
                if (_use_cov_eff and chol_sig is not None)
                else "diagonal_Pantheon_uncertainties_training"
            ),
            covariance_warnings=cov_warnings,
        ),
        stacking_weights_WAIC=dict(
            lcdm=float(waic_stack_w_lcdm),
            blend=float(waic_stack_w_blend),
            note=(
                "Pointwise stacks in CSV combine posterior median distance curves using softmax(-0.5*WAIC) weights; "
                "this complements but does not replace hierarchical Bayesian mixtures."
            ),
        ),
        nested_sampling=(
            nested_summary if isinstance(nested_summary, dict) else dict(note="nested block unavailable unexpectedly")
        ),
        cross_validation=(
            dict(**cv_summary, case_level_csv=str((OUTDIR / "cross_validation_fold_metrics.csv").resolve()))
            if RUN_CV
            else cv_summary
        ),
        systematic_robustness=robustness_summary if RUN_ROBUSTNESS else dict(ran=False),
        training=dict(
            n_sn=n_train,
            pantheon_calibration=(
                "Pantheon+ SH0ES with full STATONLY covariance (CWSF_USE_COV=1) when load succeeds "
                "; otherwise analytic absolute-magnitude profiling on diagonal uncertainties"
            ),
            redshift_residual_slices_csv=str((OUTDIR / "training_redshift_sliced_residuals.csv").resolve()),
        ),
        holdout=dict(n_sn=int(len(des)), dataset="DES Dovekie HD (M profiled ONLY on Pantheon for predictions)"),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        tau_mean_l = json_safe_float(float(np.nanmean(tau_l)))
        tau_max_l_js = json_safe_float(float(np.nanmax(tau_l)))
        tau_mean_b = json_safe_float(float(np.nanmean(tau_b)))
        tau_max_b_js = json_safe_float(float(np.nanmax(tau_b)))

    summary["lcdm"] = dict(
        param_names=names_l_cols,
        median_cosmology=med_l_cosmo,
        M_at_median=float(med_l[-1]),
        Omega_Lambda=float(med_l_ol),
        rhat_split_max=json_safe_mx(rhat_split_l),
        rhat_between_chains_max=json_safe_mx(rhat_between_l),
        rhat_walkers_max=json_safe_mx(rhat_walk_l_max),
        tau_integrated_mean=tau_mean_l,
        tau_integrated_max=tau_max_l_js,
        ess_robust_min=json_safe_float(ess_min_l),
        max_loglike_train=maxll_l,
        waic=json_safe_float(float(waic_l)),
        aic=aic_l,
        bic=bic_l,
        chi2_median=chi2_med_l,
        reduced_chi2_median=json_safe_float(redchi_l),
        des_at_median=des_l_med,
        des_at_scan_maxlike=des_l,
        edge_hits=edge_l,
        ppc_train=ppc_l,
        t0_gyr_median_cosmic_age=json_safe_float(t0_median_lcdm),
    )
    summary["blend"] = dict(
        param_names=names_b_cols,
        median_cosmology=med_b_cosmo,
        M_at_median=float(med_b[-1]),
        Omega_Lambda=float(med_b_ol),
        rhat_split_max=json_safe_mx(rhat_split_b),
        rhat_between_chains_max=json_safe_mx(rhat_between_b),
        rhat_walkers_max=json_safe_mx(rhat_walk_b_max),
        tau_integrated_mean=tau_mean_b,
        tau_integrated_max=tau_max_b_js,
        ess_robust_min=json_safe_float(ess_min_b),
        max_loglike_train=maxll_b,
        waic=json_safe_float(float(waic_b)),
        aic=aic_b,
        bic=bic_b,
        chi2_median=chi2_med_b,
        reduced_chi2_median=json_safe_float(redchi_b),
        des_at_median=des_b_med,
        des_at_scan_maxlike=des_b,
        edge_hits=edge_b,
        ppc_train=ppc_b,
        t0_gyr_median_cosmic_age=json_safe_float(t0_median_blend),
        des_z_split_metrics=des_split,
    )
    summary["model_comparison"] = dict(
        delta_aic_blend_minus_lcdm=delta_aic,
        delta_bic_blend_minus_lcdm=delta_bic,
        delta_waic_blend_minus_lcdm=json_safe_float(float(waic_b - waic_l)),
        WAIC_softmax_stacking_weights=dict(lcdm=float(waic_stack_w_lcdm), blend=float(waic_stack_w_blend)),
        des_rmse_ratio_blend_over_lcdm_at_median=float(des_b_med["rmse"] / max(des_l_med["rmse"], 1e-9)),
        blend_better_holdout_rmse_at_median=bool(better_holdout),
        strict_joint_claim_blend_superior_under_pipeline_rules=bool(publication_blend_claim),
        Bayesian_evidence_delta_lnZ_blend_minus_lcdm=nested_summary.get("delta_logz_blend_minus_lcdm"),
        jeffreys_nominal=(
            nested_summary.get("jeffreys_label_blend_minus_lcdm")
            if isinstance(nested_summary, dict)
            else None
        ),
        Pantheon_CV_blend_win_fraction=(
            json_safe_float(float(blend_win_cv_fraction)) if blend_win_cv_fraction is not None else None
        ),
        framework_seed_dispersion_joint_screen=dict(H0_sigma=json_safe_float(h0_std_fw), Omega_m_sigma=json_safe_float(Om_std_fw)),
        interpretation_rule=interpret,
        diagnostic_flags=dict(boundary_pressure_on_blend_prior_box=boundary_pressure),
    )
    summary["warnings"] = warn
    summary["posterior_predictive_chi2_standardized"] = ppc_chi2_diag
    summary["paths"] = dict(
        outdir=str(OUTDIR),
        figures=str(FIGDIR),
        pantheon_residual_slices=str((OUTDIR / "training_redshift_sliced_residuals.csv").resolve()),
        posterior_predictive_chi2_replicas=str((OUTDIR / "posterior_predictive_chi2_replicas_matched_length.csv").resolve()),
        expansion_history_percentiles_blend=str((OUTDIR / "expansion_Hz_DH_percentiles_blend.csv").resolve()),
        longform_posterior_table_csv=str((OUTDIR / "posterior_cosmo_longform_emcee_blend_draws.csv").resolve()),
        parquet_long_form_optional=str((OUTDIR / "posterior_cosmo_longform_emcee_blend_draws.parquet").resolve()),
        nested_sampling_outputs=str((OUTDIR / "nested").resolve()),
        figure_qq_training_blend=str((FIGDIR / "qq_training_residual_blend.png").resolve()),
        figure_posterior_predictive_chi2=str((FIGDIR / "posterior_predictive_chi2_histograms.png").resolve()),
        figure_expansion_H_vs_t=str((FIGDIR / "expansion_Hz_vs_cosmic_time_blend.png").resolve()),
        figure_training_delta_mu_slices=str((FIGDIR / "training_redshift_sliced_mean_delta_mu.png").resolve()),
        figure_waic_stacked_hubble=str((FIGDIR / "hubble_pure_vs_waic_stacked.png").resolve()),
        figure_cv_fold_scatter=str((FIGDIR / "cv_rmse_fold_scatter_lcdm_vs_blend.png").resolve()),
        figure_H_kinematic_consistency=str((FIGDIR / "diagnostic_Hz_splinechi_vs_Friedmann.png").resolve()),
        figure_deceleration_qz_lcdm=str((FIGDIR / "diagnostic_deceleration_qz_lcdm.png").resolve()),
        posterior_blend_parameter_correlations_csv=str((OUTDIR / "posterior_blend_parameter_correlations.csv").resolve()),
        blend_mu_finitediff_sensitivity_csv=str((OUTDIR / "blend_mu_finitediff_sensitivity.csv").resolve()),
        paper_outline_md=str((OUTDIR / "PAPER_OUTLINE.md").resolve()),
        systematics_defensibility_md=str((OUTDIR / "SYSTEMATICS_AND_DEFENSIBILITY.md").resolve()),
        isef_cleaned_training=str((OUTDIR / "cleaned_training.csv").resolve()),
        isef_cleaned_holdout=str((OUTDIR / "cleaned_holdout.csv").resolve()),
        isef_bundle_index=str((OUTDIR / "isef_bundle_index.json").resolve()),
        isef_pearson_correlations_csv=str((OUTDIR / "pearson_correlations.csv").resolve()),
        isef_evidence_json=str((OUTDIR / "evidence_and_bayes_factors.json").resolve()),
        isef_model_comparison_table=str((OUTDIR / "model_comparison_table.csv").resolve()),
        isef_chains_npz=str((OUTDIR / "chains_emcee.npz").resolve()),
        isef_holdout_predictions=str((OUTDIR / "holdout_predictions.csv").resolve()),
        isef_posterior_predictive_checks_csv=str((OUTDIR / "posterior_predictive_checks.csv").resolve()),
        isef_readme_optional=str((OUTDIR / "README_ISEF.md").resolve()),
        isef_figure_model_comparison_ic=str((FIGDIR / "model_comparison_ic_bars.png").resolve()),
        cross_probe_metrics_csv=str((OUTDIR / "cross_probe_metrics.csv").resolve()),
        figure_bao_dm_over_rd=str((FIGDIR / "bao_DM_over_rd_vs_z.png").resolve()),
        figure_cmb_compressed_shift=str((FIGDIR / "cmb_compressed_shift_vs_lcdm_median.png").resolve()),
        sparc_mond_table_csv=str((OUTDIR / "sparc_mond_per_galaxy_chi2.csv").resolve()),
        figure_sparc_mond_example=str((FIGDIR / "sparc_mond_example_rotation_curve.png").resolve()),
    )
    _vp = validation_falsification_bundle.get("paths")
    if isinstance(_vp, dict):
        summary["paths"].update(_vp)
    summary["paths"].update(
        {
            "validation_null_histograms_png": str((FIGDIR / "null_histograms.png").resolve()),
            "validation_null_tradeoff_png": str((FIGDIR / "null_tradeoff_scatter.png").resolve()),
            "validation_failure_modes_txt": str((OUTDIR / "failure_modes_summary.txt").resolve()),
            "validation_summary_csv": str((OUTDIR / "summary.csv").resolve()),
            "validation_verdict_txt": str((OUTDIR / "verdict.txt").resolve()),
        }
    )

    isef_bundle_export = export_isef_packaged_artifacts(
        dict(
            OUTDIR=str(OUTDIR.resolve()),
            FIGDIR=str(FIGDIR.resolve()),
            RNG=RNG,
            pant_fallback=str(pant_path.resolve()),
            des_fallback=str(des_path.resolve()),
            flat_l=flat_l,
            flat_b=flat_b,
            names_l_cols=names_l_cols,
            names_b_cols=names_b_cols,
            zt=zt,
            mt=mt,
            st=st,
            med_l=med_l,
            med_b=med_b,
            eng=eng,
            nested_summary=nested_summary,
            ppc_chi2_diag=ppc_chi2_diag,
            chi2_med_l=float(chi2_med_l),
            chi2_med_b=float(chi2_med_b),
            redchi_l=float(redchi_l),
            redchi_b=float(redchi_b),
            aic_l=float(aic_l),
            aic_b=float(aic_b),
            bic_l=float(bic_l),
            bic_b=float(bic_b),
            waic_l=float(waic_l),
            waic_b=float(waic_b),
            maxll_l=float(maxll_l),
            maxll_b=float(maxll_b),
            n_train=int(n_train),
            k_l=int(k_l),
            k_b=int(k_b),
            des_l_med=des_l_med,
            des_b_med=des_b_med,
            robustness_summary=robustness_summary if RUN_ROBUSTNESS else dict(ran=False),
            cv_summary=cv_summary,
            _use_cov_eff=bool(_use_cov_eff),
            infer_profile_like=bool(infer_profile_m()),
            N_FRAMEWORK_SEEDS=int(N_FRAMEWORK_SEEDS),
            RNG_SEED=int(RNG_SEED),
            Z_REF=float(Z_REF_DELTA_MU_CORR),
            des=dict(z=z_des, mu=m_des, sig=s_des),
            PANT_URL=str(PANT_URL),
            DES_URL=str(DES_URL),
            PANT_COV_ACTIVE=str(PANT_COVSTAT_URL),
        )
    )
    summary["isef_export_bundle"] = isef_bundle_export

    ctx_pub: dict[str, Any] = {
        "OUTDIR": OUTDIR,
        "FIGDIR": FIGDIR,
        "eng": eng,
        "zt": zt,
        "mt": mt,
        "st": st,
        "flat_l": flat_l,
        "flat_b": flat_b,
        "med_l": med_l,
        "med_b": med_b,
        "names_l_cols": names_l_cols,
        "names_b_cols": names_b_cols,
        "nd_l": nd_l,
        "nd_b": nd_b,
        "k_l": k_l,
        "k_b": k_b,
        "lcdm_lows": lcdm_lows,
        "lcdm_highs": lcdm_highs,
        "lcdm_centers": lcdm_centers,
        "lcdm_widths": lcdm_widths,
        "blend_lows": blend_lows,
        "blend_highs": blend_highs,
        "blend_centers_adapt": blend_centers_adapt,
        "blend_widths_adapt": blend_widths_adapt,
        "infer_profile": infer_profile_m(),
        "use_cov": _use_cov_eff,
        "chol_sig": chol_sig,
        "logdet_twopisigma": logdet_twopisigma,
        "interpret": interpret,
        "publication_blend_claim": publication_blend_claim,
        "better_holdout": better_holdout,
        "pant_path": pant_path,
        "des_path": des_path,
        "rng": RNG,
        "delta_aic": float(aic_b - aic_l),
        "delta_bic": float(bic_b - bic_l),
        "n_train": int(n_train),
    }
    summary["publication_extensions"] = execute_publication_grade_extensions(ctx_pub, warn)
    pe = summary["publication_extensions"]
    if isinstance(pe, dict) and isinstance(pe.get("paths_extra"), dict):
        summary["paths"].update(pe["paths_extra"])

    (OUTDIR / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    write_summary_csv_row(OUTDIR, summary)
    print("Wrote artifacts under", OUTDIR)


# =============================================================================
# SECTION 21 — Publication-grade extensions (synthetic falsification, w_eff,
#             Fisher-style summaries, reproducibility CSV, referee verdict)
# =============================================================================
# SECTION 21b — Validation / falsification engine (null FPR, injection, identifiability,
#             adversarial stress, repeated hold-out, PPC calibration, failure modes, verdict)
# =============================================================================


def _validation_synth_ctx_strip_cov(ctx_full: dict[str, Any]) -> dict[str, Any]:
    """Synthetic suites use **diagonal** Gaussian training likelihoods (approximate; avoids cov mismatch)."""
    d = dict(ctx_full)
    d["use_cov"] = False
    d["chol_sig"] = None
    d["logdet_twopisigma"] = float("nan")
    return d


def simulate_sn_dataset_extended(
    z: np.ndarray,
    sig: np.ndarray,
    truth_lcdm: np.ndarray,
    truth_blend: np.ndarray | None,
    mode: str,
    rng: np.random.Generator,
    cal_a0: float = 0.0,
    cal_a1: float = 0.0,
    cal_a2: float = 0.0,
    sig_alpha: float = 0.0,
    outlier_frac: float = 0.0,
    outlier_sigma_mult: float = 6.0,
    maglim: float | None = None,
    noise_scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Synthetic μ_obs (diagonal noise); optional calibration drift, σ(z), outliers, Malmquist cut (diagnostic stress)."""
    zv = np.asarray(z, dtype=float).reshape(-1)
    sg0 = np.asarray(sig, dtype=float).reshape(-1) * (1.0 + float(sig_alpha) * zv) * float(noise_scale)
    eng_l = CosmoInterpEngine(float((np.max(zv) + 1.0) * 1.02), float(np.max(zv)))
    if mode == "lcdm_truth":
        mu, _, _ = mu_lcdm(zv, float(truth_lcdm[0]), float(truth_lcdm[1]), float(truth_lcdm[2]), eng_l)
    elif mode == "blend_truth" and truth_blend is not None:
        tb = np.asarray(truth_blend, dtype=float).reshape(-1)
        mu, _ = mu_blend(zv, float(tb[0]), float(tb[1]), float(tb[2]), float(tb[3]), float(tb[4]), eng_l)
    else:
        raise ValueError("unknown mock mode")
    mu = np.asarray(mu, dtype=float).reshape(-1)
    delta = float(cal_a0) + float(cal_a1) * zv + float(cal_a2) * zv**2
    obs = mu + delta + rng.normal(0.0, 1.0, size=zv.size) * sg0
    if float(outlier_frac) > 0.0:
        m = rng.random(zv.size) < float(outlier_frac)
        obs = np.where(m, obs + rng.normal(0.0, 1.0, size=zv.size) * float(outlier_sigma_mult) * sg0, obs)
    if maglim is not None:
        m = obs < float(maglim)
        obs = np.where(m, obs, np.nan)
    return obs, mu


def theta_extend_with_profiled_M(
    z_train: np.ndarray,
    mu_train: np.ndarray,
    sig_train: np.ndarray,
    raw_row: np.ndarray,
    blend: bool,
    eng: CosmoInterpEngine,
) -> np.ndarray:
    """Append profiled ``M`` to short-chain rows when ``PROFILE_M`` is on (same rule as production)."""
    rw = np.asarray(raw_row, dtype=float).reshape(-1)
    if infer_profile_m():
        if blend:
            mu_tr, _ = mu_blend_shape(z_train, float(rw[0]), float(rw[1]), float(rw[2]), float(rw[3]), eng)
            mm = profiled_absolute_magnitude(mu_tr, mu_train, sig_train)
            return np.asarray([float(rw[0]), float(rw[1]), float(rw[2]), float(rw[3]), mm], dtype=float)
        mu_tr, _, _ = mu_lcdm_shape(z_train, float(rw[0]), float(rw[1]), eng)
        mm = profiled_absolute_magnitude(mu_tr, mu_train, sig_train)
        return np.asarray([float(rw[0]), float(rw[1]), mm], dtype=float)
    return rw.copy()


def training_chi2_rmse_diagonal(
    z: np.ndarray,
    mu_obs: np.ndarray,
    sig: np.ndarray,
    theta: np.ndarray,
    blend: bool,
    eng: CosmoInterpEngine,
) -> tuple[float, float, float]:
    """Training χ² and RMSE under diagonal Gaussian (approximate MAP uses ``theta`` as given)."""
    th = np.asarray(theta, dtype=float).reshape(-1)
    if blend:
        mu_tr, _ = mu_blend_shape(z, float(th[0]), float(th[1]), float(th[2]), float(th[3]), eng)
        if infer_profile_m():
            mm = profiled_absolute_magnitude(mu_tr, mu_obs, sig)
            mod = mu_tr + mm
        else:
            mod, _ = mu_blend(z, float(th[0]), float(th[1]), float(th[2]), float(th[3]), float(th[4]), eng)
    else:
        mu_tr, _, _ = mu_lcdm_shape(z, float(th[0]), float(th[1]), eng)
        if infer_profile_m():
            mm = profiled_absolute_magnitude(mu_tr, mu_obs, sig)
            mod = mu_tr + mm
        else:
            mod, _, _ = mu_lcdm(z, float(th[0]), float(th[1]), float(th[2]), eng)
    r = np.asarray(mu_obs, dtype=float) - np.asarray(mod, dtype=float)
    sg = np.maximum(np.asarray(sig, dtype=float), eps())
    chi2 = float(np.sum((r / sg) ** 2))
    rmse = float(math.sqrt(float(np.mean(r**2))))
    bias = float(np.mean(r))
    return chi2, rmse, bias


def holdout_logpred_rmse_chi2(
    theta: np.ndarray,
    blend: bool,
    pantheon_z: np.ndarray,
    pantheon_mu: np.ndarray,
    pantheon_sig: np.ndarray,
    z_h: np.ndarray,
    mu_h: np.ndarray,
    sig_h: np.ndarray,
    eng: CosmoInterpEngine,
) -> dict[str, float]:
    """Hold-out diagonal metrics (approximate; uses same profiling rule as production)."""
    des_df = pd.DataFrame(dict(z=z_h, mu=mu_h, sig=sig_h))
    m = metrics_holdout_profiled(theta, pantheon_z, pantheon_mu, pantheon_sig, des_df, blend, eng)
    zv = np.asarray(z_h, dtype=float)
    th = np.asarray(theta, dtype=float).reshape(-1)
    if blend:
        if infer_profile_m():
            mu_tr, _ = mu_blend_shape(pantheon_z, float(th[0]), float(th[1]), float(th[2]), float(th[3]), eng)
            mu_hh, _ = mu_blend_shape(zv, float(th[0]), float(th[1]), float(th[2]), float(th[3]), eng)
            mm = profiled_absolute_magnitude(mu_tr, pantheon_mu, pantheon_sig)
            pred = mu_hh + mm
        else:
            pred, _ = mu_blend(zv, float(th[0]), float(th[1]), float(th[2]), float(th[3]), float(th[4]), eng)
    else:
        if infer_profile_m():
            mu_tr, _, _ = mu_lcdm_shape(pantheon_z, float(th[0]), float(th[1]), eng)
            mu_hh, _, _ = mu_lcdm_shape(zv, float(th[0]), float(th[1]), eng)
            mm = profiled_absolute_magnitude(mu_tr, pantheon_mu, pantheon_sig)
            pred = mu_hh + mm
        else:
            pred, _, _ = mu_lcdm(zv, float(th[0]), float(th[1]), float(th[2]), eng)
    ll = float(np.sum(pointwise_loglike_gauss(np.asarray(pred, dtype=float), np.asarray(mu_h, dtype=float), np.asarray(sig_h, dtype=float))))
    return dict(rmse=float(m["rmse"]), chi2=float(m["chi2"]), bias=float(m["bias"]), holdout_logpred=ll)


def _short_fit_with_holdout_metrics(
    z: np.ndarray,
    mu: np.ndarray,
    sig: np.ndarray,
    eng: CosmoInterpEngine,
    ctx: dict[str, Any],
    seed: int,
    z_h: np.ndarray | None,
    mu_h: np.ndarray | None,
    sig_h: np.ndarray | None,
) -> dict[str, Any]:
    """Short MCMC LCDM vs blend + **approximate** training/hold-out diagnostics (single short fit; not production)."""
    synth_ctx = _validation_synth_ctx_strip_cov(ctx)
    base = _short_fit_lcdm_blend(z, mu, sig, eng, synth_ctx, int(seed))
    theta_l = theta_extend_with_profiled_M(z, mu, sig, np.asarray(base["best_row_lcdm"], dtype=float), False, eng)
    theta_b = theta_extend_with_profiled_M(z, mu, sig, np.asarray(base["best_row_blend"], dtype=float), True, eng)
    chi2_l, rmse_l, bias_l = training_chi2_rmse_diagonal(z, mu, sig, theta_l, False, eng)
    chi2_b, rmse_b, bias_b = training_chi2_rmse_diagonal(z, mu, sig, theta_b, True, eng)
    out = dict(
        **{k: v for k, v in base.items() if k not in ("best_row_lcdm", "best_row_blend")},
        chi2_train_lcdm_map=chi2_l,
        chi2_train_blend_map=chi2_b,
        rmse_train_lcdm=rmse_l,
        rmse_train_blend=rmse_b,
        bias_train_lcdm=bias_l,
        bias_train_blend=bias_b,
        theta_lcdm_map=theta_l.tolist(),
        theta_blend_map=theta_b.tolist(),
    )
    if z_h is not None and mu_h is not None and sig_h is not None:
        ho_l = holdout_logpred_rmse_chi2(theta_l, False, z, mu, sig, z_h, mu_h, sig_h, eng)
        ho_b = holdout_logpred_rmse_chi2(theta_b, True, z, mu, sig, z_h, mu_h, sig_h, eng)
        out.update(
            holdout_rmse_lcdm=ho_l["rmse"],
            holdout_rmse_blend=ho_b["rmse"],
            holdout_chi2_lcdm=ho_l["chi2"],
            holdout_chi2_blend=ho_b["chi2"],
            holdout_bias_lcdm=ho_l["bias"],
            holdout_bias_blend=ho_b["bias"],
            holdout_logpred_lcdm=ho_l["holdout_logpred"],
            holdout_logpred_blend=ho_b["holdout_logpred"],
        )
    return out


def run_null_false_positive_tests(
    ctxv: dict[str, Any],
    log_lines: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """MODULE 1 — LCDM-truth Monte Carlo; reports **approximate** FPRs from short MCMC (explicitly labeled)."""
    od = Path(str(ctxv["OUTDIR"]))
    fd = Path(str(ctxv["FIGDIR"]))
    rng = ctxv["RNG"]
    zt = np.asarray(ctxv["zt"], dtype=float)
    st = np.asarray(ctxv["st"], dtype=float)
    n = int(zt.size)
    truth = np.asarray(ctxv["med_l"], dtype=float).reshape(-1)
    eng = ctxv["eng"]
    synth_ctx = _validation_synth_ctx_strip_cov(ctxv["mcmc_ctx"])
    z_des = np.asarray(ctxv["z_des"], dtype=float)
    s_des = np.asarray(ctxv["s_des"], dtype=float)
    noise_scales = [float(x) for x in os.getenv("CWSF_NULL_NOISE_SCALES", "1.0").split(",") if x.strip()]
    rows: list[dict[str, Any]] = []
    metrics_rows: list[dict[str, Any]] = []
    for ns in noise_scales:
        fpr_train_aic = fpr_train_bic = fpr_train_chi2 = fpr_hold_rmse = fpr_hold_logpred = 0
        trials_run = 0
        ntri = int(NULL_FPR_TRIALS)
        for it in range(ntri):
            seed_t = int(RNG_SEED + 8_010_000 + it + int(1e5 * ns))
            rng_t = np.random.default_rng(seed_t)
            idx = rng_t.integers(0, n, size=n, endpoint=False)
            z_s = zt[idx]
            sig_s = st[idx] * ns
            mu_obs, _mu_true = simulate_sn_dataset_extended(z_s, sig_s, truth, None, "lcdm_truth", rng_t, noise_scale=1.0)
            m_keep = np.isfinite(mu_obs)
            z_s, mu_obs, sig_s = z_s[m_keep], mu_obs[m_keep], sig_s[m_keep]
            if int(z_s.size) < int(max(40, synth_ctx["nd_b"] + 12)):
                continue
            mu_des_t, _ = simulate_sn_dataset_extended(z_des, s_des, truth, None, "lcdm_truth", rng_t, noise_scale=1.0)
            fit = _short_fit_with_holdout_metrics(z_s, mu_obs, sig_s, eng, synth_ctx, seed_t + 17, z_des, mu_des_t, s_des)
            prefer_aic = bool(fit["delta_aic"] < 0.0)
            prefer_bic = bool(fit["delta_bic"] < 0.0)
            prefer_chi2 = bool(fit["chi2_train_blend_map"] < fit["chi2_train_lcdm_map"])
            prefer_hold = bool(fit.get("holdout_rmse_blend", 9e9) < fit.get("holdout_rmse_lcdm", 9e9))
            prefer_lp = bool(fit.get("holdout_logpred_blend", -9e9) > fit.get("holdout_logpred_lcdm", -9e9))
            fpr_train_aic += int(prefer_aic)
            fpr_train_bic += int(prefer_bic)
            fpr_train_chi2 += int(prefer_chi2)
            fpr_hold_rmse += int(prefer_hold)
            fpr_hold_logpred += int(prefer_lp)
            rows.append(
                dict(
                    trial=it,
                    noise_scale=ns,
                    seed=seed_t,
                    prefer_blend_delta_aic=prefer_aic,
                    prefer_blend_delta_bic=prefer_bic,
                    prefer_blend_train_chi2=prefer_chi2,
                    prefer_blend_holdout_rmse=prefer_hold,
                    prefer_blend_holdout_logpred=prefer_lp,
                    delta_aic=float(fit["delta_aic"]),
                    delta_bic=float(fit["delta_bic"]),
                    method_label="approximate_MAP_short_mcmc_diagonal_only",
                )
            )
            metrics_rows.append(
                dict(
                    trial=it,
                    noise_scale=ns,
                    model="lcdm_truth",
                    **{
                        k: float(v) if isinstance(v, (int, float, np.floating)) else str(v)
                        for k, v in fit.items()
                        if k not in ("theta_lcdm_map", "theta_blend_map")
                    },
                )
            )
            trials_run += 1
        denom = max(trials_run, 1)
        fpr_summary = dict(
            noise_scale=ns,
            n_trials=denom,
            false_positive_rate_train_AIC=float(fpr_train_aic / denom),
            false_positive_rate_train_BIC=float(fpr_train_bic / denom),
            false_positive_rate_train_chi2=float(fpr_train_chi2 / denom),
            false_positive_rate_holdout_RMSE=float(fpr_hold_rmse / denom),
            false_positive_rate_holdout_logpred=float(fpr_hold_logpred / denom),
            disclaimer="FPRs are frequentist summaries over synthetic draws; short-MCMC MAP is approximate.",
        )
        log_lines.append(json.dumps(fpr_summary))
    df_trials = pd.DataFrame(rows)
    df_metrics = pd.DataFrame(metrics_rows)
    summ_list = []
    for ns in noise_scales:
        th = df_trials[df_trials["noise_scale"] == ns]
        if th.empty:
            continue
        summ_list.append(
            dict(
                noise_scale=ns,
                n_trials=int(len(th)),
                fpr_train_AIC=float(np.mean(th["prefer_blend_delta_aic"].astype(float))),
                fpr_train_BIC=float(np.mean(th["prefer_blend_delta_bic"].astype(float))),
                fpr_train_chi2=float(np.mean(th["prefer_blend_train_chi2"].astype(float))),
                fpr_holdout_RMSE=float(np.mean(th["prefer_blend_holdout_rmse"].astype(float))),
                fpr_holdout_logpred=float(np.mean(th["prefer_blend_holdout_logpred"].astype(float))),
            )
        )
    df_sum = pd.DataFrame(summ_list)
    df_trials.to_csv(od / "synthetic_null_trials.csv", index=False)
    df_metrics.to_csv(od / "null_metric_distributions.csv", index=False)
    df_sum.to_csv(od / "null_false_positive_summary.csv", index=False)
    if not df_trials.empty and "delta_aic" in df_trials.columns:
        plt.figure(figsize=(6.5, 4.2))
        plt.hist(df_trials["delta_aic"].astype(float), bins=16, color="steelblue", alpha=0.85)
        plt.axvline(0.0, color="k", ls="--")
        plt.xlabel(r"$\Delta$AIC (blend $-$ LCDM)")
        plt.ylabel("count")
        plt.title("Null LCDM simulations: ΔAIC (approximate short MCMC)")
        plt.tight_layout()
        plt.savefig(fd / "null_histograms.png", dpi=175)
        plt.close()
        plt.figure(figsize=(6.2, 4.2))
        plt.scatter(df_trials["delta_aic"], df_trials["delta_bic"], s=18, alpha=0.55, c=df_trials["noise_scale"], cmap="viridis")
        plt.axhline(0.0, color="k", lw=0.7)
        plt.axvline(0.0, color="k", lw=0.7)
        plt.xlabel(r"$\Delta$AIC")
        plt.ylabel(r"$\Delta$BIC")
        plt.colorbar(label="noise scale on σ")
        plt.title("Null trials: complexity-penalized tradeoff (approximate)")
        plt.tight_layout()
        plt.savefig(fd / "null_tradeoff_scatter.png", dpi=175)
        plt.close()
    if len(noise_scales) > 1 and not df_sum.empty:
        plt.figure(figsize=(6.4, 4.0))
        plt.plot(df_sum["noise_scale"], df_sum["fpr_holdout_RMSE"], marker="o", label="FPR holdout RMSE")
        plt.plot(df_sum["noise_scale"], df_sum["fpr_train_AIC"], marker="s", label="FPR train AIC")
        plt.xlabel("noise multiplier on σ")
        plt.ylabel("false positive rate")
        plt.title("Null power curve (approximate)")
        plt.legend(fontsize=8)
        plt.grid(alpha=0.25)
        plt.tight_layout()
        plt.savefig(fd / "null_power_curve.png", dpi=175)
        plt.close()
    return df_trials, df_metrics, df_sum, dict(ran=True, n_noise_levels=len(noise_scales))


def run_injection_recovery_tests(ctxv: dict[str, Any], log_lines: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """MODULE 2 — blend-truth simulations; recovery and wrong-model rate (approximate short MCMC)."""
    od = Path(str(ctxv["OUTDIR"]))
    fd = Path(str(ctxv["FIGDIR"]))
    rng = ctxv["RNG"]
    zt = np.asarray(ctxv["zt"], dtype=float)
    st = np.asarray(ctxv["st"], dtype=float)
    truth_l = np.asarray(ctxv["med_l"], dtype=float).reshape(-1)
    truth_b = np.asarray(ctxv["med_b"], dtype=float).reshape(-1)
    n = int(zt.size)
    eng = ctxv["eng"]
    synth_ctx = _validation_synth_ctx_strip_cov(ctxv["mcmc_ctx"])
    z_des = np.asarray(ctxv["z_des"], dtype=float)
    s_des = np.asarray(ctxv["s_des"], dtype=float)
    rows: list[dict[str, Any]] = []
    bias_rows: list[dict[str, Any]] = []
    cov_rows: list[dict[str, Any]] = []
    wrong_lcdm_wins = 0
    n_ok = 0
    ntri = int(INJECTION_TRIALS)
    for it in range(ntri):
        seed_t = int(RNG_SEED + 8_020_000 + it)
        rng_t = np.random.default_rng(seed_t)
        idx = rng_t.integers(0, n, size=n, endpoint=False)
        z_s, sig_s = zt[idx], st[idx]
        mu_obs, mu_true = simulate_sn_dataset_extended(z_s, sig_s, truth_l, truth_b, "blend_truth", rng_t)
        m_keep = np.isfinite(mu_obs)
        z_s, mu_obs, sig_s = z_s[m_keep], mu_obs[m_keep], sig_s[m_keep]
        if int(z_s.size) < 40:
            continue
        mu_des_t, _ = simulate_sn_dataset_extended(z_des, s_des, truth_l, truth_b, "blend_truth", rng_t)
        fit = _short_fit_with_holdout_metrics(z_s, mu_obs, sig_s, eng, synth_ctx, seed_t + 31, z_des, mu_des_t, s_des)
        theta_b = np.asarray(fit["theta_blend_map"], dtype=float)
        wrong_lcdm_wins += int(fit["maxll_lcdm"] > fit["maxll_blend"])
        n_ok += 1
        rows.append(dict(trial=it, seed=seed_t, delta_aic=float(fit["delta_aic"]), delta_bic=float(fit["delta_bic"]), lcdm_wins_maxlike=bool(fit["maxll_lcdm"] > fit["maxll_blend"])))
        names = list(ctxv["names_b_cols"])
        for j, nm in enumerate(names[: min(len(names), len(truth_b))]):
            if j < int(theta_b.size):
                bias_rows.append(
                    dict(trial=it, parameter=nm, truth=float(truth_b[j]), recovered_map=float(theta_b[j]), abs_bias=float(theta_b[j] - truth_b[j]))
                )
        for j, nm in enumerate(names[: min(5, len(truth_b))]):
            if j >= int(theta_b.size):
                continue
            lo = float(truth_b[j]) * 0.85
            hi = float(truth_b[j]) * 1.15
            hit = bool(lo <= float(theta_b[j]) <= hi)
            cov_rows.append(dict(trial=it, parameter=nm, crude_coverage_15pct_tube=hit, note="approximate_heuristic_not_nominal_credible"))
    df_tri = pd.DataFrame(rows)
    df_tri.to_csv(od / "synthetic_injection_trials.csv", index=False)
    df_bias = pd.DataFrame(bias_rows)
    if not df_bias.empty:
        g = df_bias.groupby("parameter")["abs_bias"].agg(["mean", "std", "median"]).reset_index()
        g.to_csv(od / "recovery_bias_summary.csv", index=False)
    df_cov = pd.DataFrame(cov_rows)
    df_cov.to_csv(od / "recovery_coverage_summary.csv", index=False)
    if not df_bias.empty:
        pv = df_bias.pivot_table(index="trial", columns="parameter", values="abs_bias", aggfunc="mean")
        plt.figure(figsize=(7.0, 4.5))
        plt.imshow(np.log10(np.asarray(pv.fillna(1e-9), dtype=float) + 1e-9), aspect="auto", cmap="magma")
        plt.colorbar(label=r"$\log_{10}(|$bias$|)$")
        plt.yticks(range(len(pv.index)), [str(int(x)) for x in pv.index])
        plt.xticks(range(len(pv.columns)), list(pv.columns), rotation=35, ha="right")
        plt.title("Injection trials: |bias| heatmap (MAP; approximate)")
        plt.tight_layout()
        plt.savefig(fd / "parameter_recovery_heatmap.png", dpi=175)
        plt.close()
        plt.figure(figsize=(6.4, 4.0))
        for nm in df_bias["parameter"].unique():
            sub = df_bias[df_bias["parameter"] == nm]
            plt.scatter(sub["truth"], sub["recovered_map"], s=22, alpha=0.55, label=str(nm))
        plt.plot([35, 85], [35, 85], "k--", lw=0.8)
        plt.xlabel("truth")
        plt.ylabel("recovered MAP")
        plt.title("Injection vs fit (blend truth)")
        plt.legend(fontsize=7)
        plt.tight_layout()
        plt.savefig(fd / "injection_vs_fit_summary.png", dpi=175)
        plt.close()
        plt.figure(figsize=(6.5, 4.2))
        for nm in df_bias["parameter"].unique():
            sub = df_bias[df_bias["parameter"] == nm].sort_values("trial")
            plt.plot(sub["trial"], sub["recovered_map"], marker="o", ms=3, label=str(nm))
        plt.xlabel("trial")
        plt.ylabel("recovered")
        plt.title("Recovery curves (approximate)")
        plt.legend(fontsize=7, ncol=2)
        plt.grid(alpha=0.25)
        plt.tight_layout()
        plt.savefig(fd / "recovery_plots.png", dpi=175)
        plt.close()
    return df_tri, df_bias, df_cov, dict(ran=True, lcdm_false_wins_maxlike_fraction=float(wrong_lcdm_wins / max(n_ok, 1)))


def run_identifiability_and_degeneracy_bundle(ctxv: dict[str, Any], log_lines: list[str]) -> dict[str, Any]:
    """MODULE 3 — correlations, μ sensitivities, profiles, Fisher (all approximate / diagnostic)."""
    od = Path(str(ctxv["OUTDIR"]))
    fd = Path(str(ctxv["FIGDIR"]))
    eng = ctxv["eng"]
    zt = np.asarray(ctxv["zt"], dtype=float)
    mt = np.asarray(ctxv["mt"], dtype=float)
    st = np.asarray(ctxv["st"], dtype=float)
    flat_b = np.asarray(ctxv["flat_b"], dtype=float)
    med_b = np.asarray(ctxv["med_b"], dtype=float)
    names_b = list(ctxv["names_b_cols"])
    corr_df = blend_chain_parameter_correlations(flat_b, names_b)
    corr_df.to_csv(od / "identifiability_summary.csv", index=False)
    plt.figure(figsize=(6.2, 5.0))
    if corr_df.size and "status" not in corr_df.columns and corr_df.shape[0] == corr_df.shape[1]:
        plt.imshow(np.asarray(corr_df, dtype=float), cmap="RdBu_r", vmin=-1, vmax=1)
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.xticks(range(len(corr_df.columns)), list(corr_df.columns), rotation=35, ha="right")
        plt.yticks(range(len(corr_df.index)), list(corr_df.index))
        plt.title("Posterior correlation (blend; diagnostic)")
    else:
        plt.text(0.1, 0.5, "Correlation heatmap skipped (insufficient clean samples)", fontsize=10)
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(fd / "correlation_heatmap.png", dpi=175)
    plt.close()
    prior_posterior_overlap_table(
        flat_b, np.asarray(ctxv["blend_lows"], dtype=float), np.asarray(ctxv["blend_highs"], dtype=float), names_b
    ).to_csv(od / "prior_posterior_overlap.csv", index=False)
    sens_all: list[pd.DataFrame] = []
    for zref in np.linspace(0.02, 1.45, 10, dtype=float):
        sens_all.append(finite_diff_blend_mu_sensitivity_rows(med_b, float(zref), eng))
    pd.concat(sens_all, ignore_index=True).to_csv(od / "sensitivity_curves.csv", index=False)
    plt.figure(figsize=(6.8, 4.2))
    for par in ("H0", "Omega_m", "t_crit", "k_slope", "M"):
        sub = pd.concat(sens_all, ignore_index=True)
        if par in sub["parameter"].values:
            m = sub["parameter"] == par
            plt.plot(sub.loc[m, "z_ref"], sub.loc[m, "dmu_dparam"], lw=1.2, label=par)
    plt.xlabel(r"$z_{\mathrm{ref}}$")
    plt.ylabel(r"$\partial\mu/\partial\theta$ (finite difference; approximate)")
    plt.title("Local sensitivity vs redshift (blend median θ)")
    plt.legend(fontsize=7, ncol=2)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(fd / "sensitivity_curves.png", dpi=175)
    plt.close()
    lows_b = np.asarray(ctxv["blend_lows"], dtype=float)
    highs_b = np.asarray(ctxv["blend_highs"], dtype=float)
    th0 = np.asarray(med_b, dtype=float).copy()

    def ln_b(th: np.ndarray) -> float:
        if ctxv["mcmc_ctx"]["use_cov"] and ctxv["mcmc_ctx"]["chol_sig"] is not None and math.isfinite(float(ctxv["mcmc_ctx"]["logdet_twopisigma"])):
            return float(
                log_post_blend_cov(
                    np.asarray(th, dtype=float),
                    mt,
                    ctxv["mcmc_ctx"]["chol_sig"],
                    float(ctxv["mcmc_ctx"]["logdet_twopisigma"]),
                    zt,
                    eng,
                )
            )
        return float(log_post_blend(np.asarray(th, dtype=float), zt, mt, st, eng))

    prof_frames: list[pd.DataFrame] = []
    for j, nm in enumerate(names_b[: int(th0.size)]):
        lo, hi = float(lows_b[j]), float(highs_b[j])
        grid = np.linspace(lo, hi, 15, dtype=float)
        df_p = profile_lnpost_vs_param(str(nm), j, grid, th0, ln_b, lows_b, highs_b)
        df_p["delta_chi2_approx"] = -2.0 * (df_p["log_post"] - float(df_p["log_post"].max()))
        df_p.to_csv(od / f"profile_likelihood_{nm}.csv", index=False)
        prof_frames.append(df_p.assign(param=str(nm)))
    if prof_frames:
        comb = pd.concat(prof_frames, ignore_index=True)
        pars_sorted = sorted(str(p) for p in comb["param"].unique())
        npar = len(pars_sorted)
        nc = int(min(3, max(2, int(np.ceil(np.sqrt(npar))))))
        nr = int(np.ceil(npar / float(nc)))
        fig, axes = plt.subplots(nr, nc, figsize=(3.2 * nc, 2.8 * nr), squeeze=False)
        for i, par in enumerate(pars_sorted):
            if i >= len(axes.flat):
                break
            ax = axes.flat[i]
            sub = comb[comb["param"] == par]
            ax.plot(sub["value"], sub["delta_chi2_approx"], lw=1.4)
            ax.axhline(1.0, color="k", ls=":", lw=0.7)
            ax.axhline(4.0, color="k", ls="--", lw=0.7)
            ax.set_title(str(par), fontsize=9)
            ax.set_xlabel("value")
            ax.set_ylabel(r"$\Delta\chi^2$ (approx)")
        for j in range(len(pars_sorted), len(axes.flat)):
            axes.flat[j].set_axis_off()
        plt.suptitle("Profile scans (approximate Δχ² from log-posterior)", fontsize=10)
        plt.tight_layout()
        plt.savefig(fd / "profile_scan_dashboard.png", dpi=175)
        plt.close()
    zf = np.quantile(zt, np.linspace(0.05, 0.95, 14))
    fish = fisher_information_mu_diagonal_approx(med_b, zf, np.interp(zf, zt, st), eng)
    fish.to_csv(od / "identifiability_fisher_mu_block.csv", index=False)
    log_lines.append("identifiability_bundle_complete")
    return dict(ran=True, note="correlation+sensitivity+profiles+Fisher are diagnostic approximations")


def run_adversarial_bias_tests(ctxv: dict[str, Any], log_lines: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """MODULE 4 — controlled distortions on LCDM-truth mocks (diagnostic stress)."""
    od = Path(str(ctxv["OUTDIR"]))
    fd = Path(str(ctxv["FIGDIR"]))
    zt = np.asarray(ctxv["zt"], dtype=float)
    st = np.asarray(ctxv["st"], dtype=float)
    truth = np.asarray(ctxv["med_l"], dtype=float).reshape(-1)
    eng = ctxv["eng"]
    synth_ctx = _validation_synth_ctx_strip_cov(ctxv["mcmc_ctx"])
    z_des = np.asarray(ctxv["z_des"], dtype=float)
    s_des = np.asarray(ctxv["s_des"], dtype=float)
    scenarios: list[tuple[str, dict[str, float]]] = [
        ("baseline", {}),
        ("sigma_z_linear_alpha0p25", dict(sig_alpha=0.25)),
        ("calib_quadratic_small", dict(cal_a0=0.02, cal_a1=-0.03, cal_a2=0.01)),
        ("outliers_2pct", dict(outlier_frac=0.02)),
        ("malmquist_maglim_minus18p5", dict(maglim=-18.5)),
    ]
    rows: list[dict[str, Any]] = []
    rng0 = np.random.default_rng(int(RNG_SEED + 8_030_000))
    for si, (tag, kw) in enumerate(scenarios[: max(1, int(ADVERSARIAL_SCENARIOS))]):
        rng_t = np.random.default_rng(int(RNG_SEED + 8_030_500 + si))
        mu_obs, _ = simulate_sn_dataset_extended(zt, st, truth, None, "lcdm_truth", rng_t, **kw)
        m_keep = np.isfinite(mu_obs)
        z_s, mu_o, sg_s = zt[m_keep], mu_obs[m_keep], st[m_keep]
        if int(z_s.size) < 50:
            continue
        mu_des_t, _ = simulate_sn_dataset_extended(z_des, s_des, truth, None, "lcdm_truth", rng_t, **{k: v for k, v in kw.items() if k != "maglim"})
        fit = _short_fit_with_holdout_metrics(z_s, mu_o, sg_s, eng, synth_ctx, int(RNG_SEED + 8_040_000 + si), z_des, mu_des_t, s_des)
        rows.append(
            dict(
                scenario=tag,
                seed=int(RNG_SEED + 8_040_000 + si),
                delta_aic=float(fit["delta_aic"]),
                delta_bic=float(fit["delta_bic"]),
                blend_wins_aic=bool(fit["delta_aic"] < 0),
                blend_wins_holdout_rmse=bool(fit.get("holdout_rmse_blend", 9e9) < fit.get("holdout_rmse_lcdm", 9e9)),
                **kw,
            )
        )
    df = pd.DataFrame(rows)
    df.to_csv(od / "adversarial_trials.csv", index=False)
    summ = []
    if not df.empty:
        summ.append(
            dict(
                metric="mean_delta_aic_blend_minus_lcdm",
                value=float(np.mean(df["delta_aic"])),
                worst_scenario=str(df.loc[df["delta_aic"].idxmin(), "scenario"]) if len(df) else "n/a",
            )
        )
    pd.DataFrame(summ).to_csv(od / "adversarial_failure_summary.csv", index=False)
    if not df.empty:
        plt.figure(figsize=(6.4, 3.8))
        plt.barh(df["scenario"], df["delta_aic"].astype(float), color="darkorange", alpha=0.85)
        plt.axvline(0.0, color="k", lw=0.9)
        plt.xlabel(r"$\Delta$AIC (blend $-$ LCDM)")
        plt.title("Adversarial distortions (LCDM truth; approximate)")
        plt.tight_layout()
        plt.savefig(fd / "adversarial_sensitivity.png", dpi=175)
        plt.close()
        fig, ax = plt.subplots(1, 2, figsize=(8.0, 3.6))
        ax[0].scatter(zt, mt, s=4, alpha=0.3, color="0.35")
        ax[0].set_title("Real training μ(z) (reference only)")
        ax[0].set_xlabel("z")
        ax[0].set_ylabel(r"$\mu$")
        ax[1].barh(df["scenario"], df["delta_bic"].astype(float), color="slateblue", alpha=0.85)
        ax[1].axvline(0.0, color="k", lw=0.8)
        ax[1].set_title(r"$\Delta$BIC by adversarial scenario")
        plt.suptitle("Adversarial / selection stress dashboard (diagnostic)")
        plt.tight_layout()
        plt.savefig(fd / "selection_bias_dashboard.png", dpi=175)
        plt.close()
        plt.figure(figsize=(6.2, 3.6))
        plt.bar(df["scenario"], df["delta_aic"].astype(float), color="teal", alpha=0.85)
        plt.axhline(0.0, color="k", lw=0.8)
        plt.ylabel(r"$\Delta$AIC")
        plt.xticks(rotation=30, ha="right")
        plt.title("Calibration-drift family (via scenario tags; diagnostic)")
        plt.tight_layout()
        plt.savefig(fd / "calibration_drift_dashboard.png", dpi=175)
        plt.close()
    log_lines.append("adversarial_bias_complete")
    return df, pd.DataFrame(summ)


def run_stratified_holdout_repeated(ctxv: dict[str, Any], log_lines: list[str]) -> tuple[pd.DataFrame, dict[str, Any]]:
    """MODULE 5 — repeated random + redshift-stratified short MCMC (approximate; expensive if many splits)."""
    od = Path(str(ctxv["OUTDIR"]))
    fd = Path(str(ctxv["FIGDIR"]))
    zt = np.asarray(ctxv["zt"], dtype=float)
    mt = np.asarray(ctxv["mt"], dtype=float)
    st = np.asarray(ctxv["st"], dtype=float)
    n = int(zt.size)
    eng = ctxv["eng"]
    synth_ctx = _validation_synth_ctx_strip_cov(ctxv["mcmc_ctx"])
    rows: list[dict[str, Any]] = []
    n_rep = max(1, min(int(HOLDOUT_REPEAT_SPLITS), 6))
    zm = float(np.median(zt))
    for rep in range(n_rep):
        rng_t = np.random.default_rng(int(RNG_SEED + 8_050_000 + rep))
        perm = rng_t.permutation(n)
        cut = int(float(HOLDOUT_TRAIN_FRAC) * n)
        tr, va = perm[:cut], perm[cut:]
        for split_name, tr_idx, va_idx in (
            ("random_fraction", tr, va),
            ("lowz_train_highz_val", np.flatnonzero(zt <= zm), np.flatnonzero(zt > zm)),
            ("highz_train_lowz_val", np.flatnonzero(zt > zm), np.flatnonzero(zt <= zm)),
        ):
            if int(va_idx.size) < 8 or int(tr_idx.size) < int(synth_ctx["nd_b"] + 12):
                continue
            z_tr, mu_tr, sig_tr = zt[tr_idx], mt[tr_idx], st[tr_idx]
            z_va, mu_va, sig_va = zt[va_idx], mt[va_idx], st[va_idx]
            seed_h = int(RNG_SEED + 8_060_000 + rep + (_stable_hash_u64(str(split_name)) % 997))
            fit = _short_fit_with_holdout_metrics(z_tr, mu_tr, sig_tr, eng, synth_ctx, seed_h, z_va, mu_va, sig_va)
            rows.append(
                dict(
                    repeat=rep,
                    split=split_name,
                    holdout_rmse_lcdm=float(fit.get("holdout_rmse_lcdm", float("nan"))),
                    holdout_rmse_blend=float(fit.get("holdout_rmse_blend", float("nan"))),
                    holdout_chi2_lcdm=float(fit.get("holdout_chi2_lcdm", float("nan"))),
                    holdout_chi2_blend=float(fit.get("holdout_chi2_blend", float("nan"))),
                    holdout_bias_lcdm=float(fit.get("holdout_bias_lcdm", float("nan"))),
                    holdout_bias_blend=float(fit.get("holdout_bias_blend", float("nan"))),
                    holdout_logpred_lcdm=float(fit.get("holdout_logpred_lcdm", float("nan"))),
                    holdout_logpred_blend=float(fit.get("holdout_logpred_blend", float("nan"))),
                    blend_wins_rmse=bool(fit.get("holdout_rmse_blend", 9e9) < fit.get("holdout_rmse_lcdm", 9e9)),
                )
            )
    df = pd.DataFrame(rows)
    df.to_csv(od / "holdout_splits.csv", index=False)
    if not df.empty:
        df.groupby("split")["holdout_rmse_blend"].mean().reset_index().to_csv(od / "cv_summary.csv", index=False)
        df.to_csv(od / "cv_folds.csv", index=False)
    if not df.empty:
        plt.figure(figsize=(6.4, 3.8))
        for sp in df["split"].unique():
            sub = df[df["split"] == sp]
            plt.plot(sub["repeat"], sub["holdout_rmse_blend"], marker="o", ms=4, label=f"blend {sp}")
        plt.xlabel("repeat")
        plt.ylabel("holdout RMSE (blend)")
        plt.legend(fontsize=7)
        plt.grid(alpha=0.25)
        plt.tight_layout()
        plt.savefig(fd / "split_stability.png", dpi=175)
        plt.close()
        plt.figure(figsize=(6.2, 3.8))
        plt.scatter(df["holdout_rmse_lcdm"], df["holdout_rmse_blend"], c=df["repeat"], cmap="coolwarm", s=36, alpha=0.75)
        plt.plot([0, 1], [0, 1], "k--", lw=0.8, scalex=False, scaley=False)
        plt.xlabel("LCDM holdout RMSE")
        plt.ylabel("Blend holdout RMSE")
        plt.title("Fold stability (short MCMC; approximate)")
        plt.colorbar(label="repeat")
        plt.tight_layout()
        plt.savefig(fd / "fold_stability.png", dpi=175)
        plt.close()
        plt.figure(figsize=(6.5, 4.0))
        for sp in df["split"].unique():
            sub = df[df["split"] == sp].sort_values("repeat")
            plt.plot(sub["repeat"], sub["holdout_bias_blend"] - sub["holdout_bias_lcdm"], marker="s", label=sp)
        plt.axhline(0.0, color="k", lw=0.7)
        plt.xlabel("repeat")
        plt.ylabel(r"$\Delta$bias (blend$-$LCDM)")
        plt.title("Redshift-stratified / random hold-out (diagnostic)")
        plt.legend(fontsize=7)
        plt.grid(alpha=0.25)
        plt.tight_layout()
        plt.savefig(fd / "redshift_stratified_validation.png", dpi=175)
        plt.close()
    log_lines.append("stratified_holdout_complete")
    return df, dict(ran=True, n_rows=int(len(df)))


def run_posterior_predictive_calibration_checks(ctxv: dict[str, Any], log_lines: list[str]) -> dict[str, Any]:
    """MODULE 6 — PPC coverage + PIT-style ranks (approximate; uses posterior samples)."""
    od = Path(str(ctxv["OUTDIR"]))
    fd = Path(str(ctxv["FIGDIR"]))
    rng = ctxv["RNG"]
    zt = np.asarray(ctxv["zt"], dtype=float)
    mt = np.asarray(ctxv["mt"], dtype=float)
    st = np.asarray(ctxv["st"], dtype=float)
    flat_b = np.asarray(ctxv["flat_b"], dtype=float)
    flat_l = np.asarray(ctxv["flat_l"], dtype=float)
    eng = ctxv["eng"]
    n_draw = min(int(PPC_CALIB_DRAWS), int(flat_b.shape[0]), int(flat_l.shape[0]))
    ix_b = rng.choice(int(flat_b.shape[0]), size=n_draw, replace=False)
    ix_l = rng.choice(int(flat_l.shape[0]), size=n_draw, replace=False)
    preds_b: list[np.ndarray] = []
    preds_l: list[np.ndarray] = []
    for row in flat_b[ix_b]:
        if infer_profile_m():
            mu_s, _ = mu_blend_shape(zt, float(row[0]), float(row[1]), float(row[2]), float(row[3]), eng)
            mm = profiled_absolute_magnitude(mu_s, mt, st)
            preds_b.append(np.asarray(mu_s, dtype=float) + mm)
        else:
            mhat, _ = mu_blend(zt, float(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4]), eng)
            preds_b.append(np.asarray(mhat, dtype=float).reshape(-1))
    for row in flat_l[ix_l]:
        if infer_profile_m():
            mu_s, _, _ = mu_lcdm_shape(zt, float(row[0]), float(row[1]), eng)
            mm = profiled_absolute_magnitude(mu_s, mt, st)
            preds_l.append(np.asarray(mu_s, dtype=float) + mm)
        else:
            mhat, _, _ = mu_lcdm(zt, float(row[0]), float(row[1]), float(row[2]), eng)
            preds_l.append(np.asarray(mhat, dtype=float).reshape(-1))
    mat_b = np.stack(preds_b, axis=0)
    mat_l = np.stack(preds_l, axis=0)
    q025b, q16b, q50b_sn, q84b, q975b = np.quantile(mat_b, [0.025, 0.16, 0.5, 0.84, 0.975], axis=0)
    cov68b = float(np.mean((mt >= q16b) & (mt <= q84b)))
    cov95b = float(np.mean((mt >= q025b) & (mt <= q975b)))
    q025l, q16l, q50l_sn, q84l, q975l = np.quantile(mat_l, [0.025, 0.16, 0.5, 0.84, 0.975], axis=0)
    cov68l = float(np.mean((mt >= q16l) & (mt <= q84l)))
    cov95l = float(np.mean((mt >= q025l) & (mt <= q975l)))
    pit_b = np.mean(mat_b < mt.reshape(1, -1), axis=0)
    pit_l = np.mean(mat_l < mt.reshape(1, -1), axis=0)
    pd.DataFrame(dict(z=zt, pit_blend=pit_b, pit_lcdm=pit_l)).to_csv(od / "posterior_predictive_draws.csv", index=False)
    pd.DataFrame(
        [
            dict(model="blend", coverage_68=cov68b, coverage_95=cov95b, n_draws=n_draw, note="approximate_posterior_predictive"),
            dict(model="lcdm", coverage_68=cov68l, coverage_95=cov95l, n_draws=n_draw, note="approximate_posterior_predictive"),
        ]
    ).to_csv(od / "posterior_predictive_coverage.csv", index=False)
    pd.DataFrame(
        dict(
            model=["blend", "lcdm"],
            mean_abs_pit_minus_half=[
                float(np.mean(np.abs(pit_b - 0.5))),
                float(np.mean(np.abs(pit_l - 0.5))),
            ],
            interpretation="PIT mean distance from 0.5; large values suggest miscalibration (diagnostic)",
        )
    ).to_csv(od / "predictive_calibration_summary.csv", index=False)
    zgrid = np.linspace(float(np.min(zt)), float(np.max(zt)), 60, dtype=float)
    q16, q50, q84, q025, q975, _, _, _, _, _ = predictive_quantiles(
        flat_b, zgrid, rng, True, eng, zt, mt, st, max_rows=min(800, int(flat_b.shape[0]))
    )
    plt.figure(figsize=(7.0, 4.5))
    plt.fill_between(zgrid, q025, q975, color="C1", alpha=0.15, label="blend 95%")
    plt.fill_between(zgrid, q16, q84, color="C1", alpha=0.35, label="blend 68%")
    plt.plot(zgrid, q50, color="C1", lw=2.0, label="blend median")
    plt.scatter(zt, mt, s=5, alpha=0.25, color="k", label="data")
    plt.xlabel("z")
    plt.ylabel(r"$\mu$")
    plt.title("Posterior predictive bands vs training (blend; diagnostic)")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(fd / "predictive_band_plot.png", dpi=175)
    plt.close()
    res = mt - q50b_sn
    plt.figure(figsize=(6.8, 4.0))
    plt.scatter(zt, res, s=8, alpha=0.45)
    plt.axhline(0.0, color="k", lw=0.8)
    plt.xlabel("z")
    plt.ylabel("residual vs posterior median μ (blend)")
    plt.title("Residual diagnostics (diagnostic)")
    plt.tight_layout()
    plt.savefig(fd / "residual_diagnostics.png", dpi=175)
    plt.close()
    plt.figure(figsize=(6.2, 3.8))
    plt.hist(pit_b, bins=16, alpha=0.6, label="blend PIT", density=True)
    plt.hist(pit_l, bins=16, alpha=0.6, label="LCDM PIT", density=True)
    plt.axhline(1.0, color="k", ls="--", lw=0.7)
    plt.xlabel("posterior predictive rank (approximate PIT)")
    plt.legend(fontsize=8)
    plt.title("Calibration histogram (uniform under ideal calibration)")
    plt.tight_layout()
    plt.savefig(fd / "calibration_histogram.png", dpi=175)
    plt.close()
    log_lines.append("ppc_calibration_complete")
    return dict(
        ran=True,
        coverage_68_blend=cov68b,
        coverage_95_blend=cov95b,
        coverage_68_lcdm=cov68l,
        coverage_95_lcdm=cov95l,
    )


def generate_failure_modes_report(ctxv: dict[str, Any], val: dict[str, Any]) -> None:
    """MODULE 7 — explicit failure inventory (CSV + JSON + text)."""
    od = Path(str(ctxv["OUTDIR"]))
    rows: list[dict[str, Any]] = []
    edge_b = ctxv.get("edge_b", {})
    flb = np.asarray(edge_b.get("frac_low", [0.0]), dtype=float)
    fhb = np.asarray(edge_b.get("frac_high", [0.0]), dtype=float)
    rows.append(
        dict(
            mode="boundary_pressure_blend",
            severity="moderate" if float(np.max(flb + fhb)) > 0.02 else "low",
            detail=str(edge_b),
        )
    )
    if not bool(ctxv.get("better_holdout", False)):
        rows.append(dict(mode="holdout_DES", severity="high", detail="Blend RMSE not below LCDM at posterior median"))
    cv = ctxv.get("cv_summary", {})
    if isinstance(cv, dict) and cv.get("ran") and float(cv.get("mean_fraction_folds_where_blend_rmse_below_lcdm_across_repeats", 1.0)) < 0.45:
        rows.append(dict(mode="internal_CV", severity="moderate", detail="Blend rarely wins RMSE across Pantheon folds"))
    nfp = val.get("null_false_positive_summary", [])
    if isinstance(nfp, list) and nfp:
        for row in nfp:
            if isinstance(row, dict) and float(row.get("fpr_train_AIC", 0.0)) > 0.35:
                rows.append(dict(mode="null_FPR_train_AIC", severity="high", detail="Blend often preferred under LCDM truth (approximate null)"))
    pd.DataFrame(rows).to_csv(od / "failure_modes.csv", index=False)
    (od / "failure_modes.json").write_text(json.dumps(dict(failures=rows), indent=2), encoding="utf-8")
    txt = (
        "FAILURE MODES SUMMARY (honest; diagnostic + approximate tests mixed)\n"
        "====================================================================\n"
        + "\n".join(f"- {r.get('mode')}: [{r.get('severity')}] {r.get('detail')}" for r in rows)
        + "\n\nSee failure_modes.csv and validation_falsification_suite in summary.json.\n"
    )
    (od / "failure_modes_summary.txt").write_text(txt, encoding="utf-8")


def generate_model_comparison_dashboard(ctxv: dict[str, Any], val: dict[str, Any]) -> None:
    """MODULE 8 — single table + figure for multi-metric view."""
    od = Path(str(ctxv["OUTDIR"]))
    fd = Path(str(ctxv["FIGDIR"]))
    row = dict(
        chi2_train_lcdm=float(ctxv["chi2_med_l"]),
        chi2_train_blend=float(ctxv["chi2_med_b"]),
        reduced_chi2_lcdm=float(ctxv["redchi_l"]),
        reduced_chi2_blend=float(ctxv["redchi_b"]),
        AIC_lcdm=float(ctxv["aic_l"]),
        AIC_blend=float(ctxv["aic_b"]),
        BIC_lcdm=float(ctxv["bic_l"]),
        BIC_blend=float(ctxv["bic_b"]),
        WAIC_lcdm=float(ctxv["waic_l"]),
        WAIC_blend=float(ctxv["waic_b"]),
        holdout_rmse_lcdm=float(ctxv["des_l_med"]["rmse"]),
        holdout_rmse_blend=float(ctxv["des_b_med"]["rmse"]),
        holdout_bias_lcdm=float(ctxv["des_l_med"]["bias"]),
        holdout_bias_blend=float(ctxv["des_b_med"]["bias"]),
        delta_lnZ_nested=ctxv.get("nested_summary", {}).get("delta_logz_blend_minus_lcdm"),
        null_FPR_train_AIC_approx=val.get("null_fpr_train_AIC"),
        null_FPR_holdout_RMSE_approx=val.get("null_fpr_holdout_RMSE"),
        injection_lcdm_wins_maxlike_frac=val.get("injection_lcdm_wins_frac"),
    )
    pd.DataFrame([row]).to_csv(od / "model_comparison.csv", index=False)
    labs = list(row.keys())
    vals = [float(v) if isinstance(v, (int, float, np.floating)) and math.isfinite(float(v)) else 0.0 for v in row.values()]
    plt.figure(figsize=(max(10.0, 0.22 * len(labs)), 4.2))
    plt.bar(range(len(labs)), vals, color="steelblue", alpha=0.85)
    plt.xticks(range(len(labs)), labs, rotation=55, ha="right", fontsize=7)
    plt.ylabel("value (mixed units; diagnostic dashboard)")
    plt.title("Multi-metric model comparison (read numeric table in model_comparison.csv)")
    plt.tight_layout()
    plt.savefig(fd / "model_comparison_dashboard.png", dpi=175)
    plt.close()


def generate_expansion_interpretation_exports(ctxv: dict[str, Any]) -> None:
    """MODULE 9 — interpretation-only expansion diagnostics (already partly in main; duplicated here for bundle)."""
    od = Path(str(ctxv["OUTDIR"]))
    fd = Path(str(ctxv["FIGDIR"]))
    eng = ctxv["eng"]
    med_l = np.asarray(ctxv["med_l"], dtype=float)
    med_b = np.asarray(ctxv["med_b"], dtype=float)
    zplot = np.linspace(0.01, 1.75, 90, dtype=float)
    mu_l, dL_l, _ = mu_lcdm_shape(zplot, float(med_l[0]), float(med_l[1]), eng)
    mu_b, _ = mu_blend_shape(zplot, float(med_b[0]), float(med_b[1]), float(med_b[2]), float(med_b[3]), eng)
    hz_l = Hz(zplot, float(med_l[0]), float(med_l[1]))
    if blend_physics_is_ccomplet2():
        hz_b = eng.blend_Hz(zplot, float(med_b[0]), float(med_b[1]), float(med_b[2]), float(med_b[3]))
    else:
        hz_b = Hz(zplot, float(med_b[0]), float(med_b[1]))
    wv = effective_w_flat_lcdm(zplot, float(med_l[0]), float(med_l[1]))
    pd.DataFrame(dict(z=zplot, H_lcdm=hz_l, H_blend=hz_b, mu_lcdm=mu_l, mu_blend=mu_b, delta_mu_blend_minus_lcdm=mu_b - mu_l)).to_csv(
        od / "expansion_history.csv", index=False
    )
    plt.figure(figsize=(6.8, 4.0))
    plt.plot(zplot, mu_b - mu_l, lw=2.0, color="purple")
    plt.xlabel("z")
    plt.ylabel(r"$\Delta\mu$ (blend$-$LCDM)")
    plt.title("Blend correction curve (interpretation)")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(fd / "blend_correction_curve.png", dpi=175)
    plt.close()
    plt.figure(figsize=(6.6, 4.0))
    plt.plot(zplot, hz_l, label="LCDM H(z)")
    plt.plot(zplot, hz_b, label="Blend H(z)" if blend_physics_is_ccomplet2() else "LCDM H(z) (blend params; legacy)")
    plt.legend(fontsize=8)
    plt.xlabel("z")
    plt.ylabel(r"$H(z)$")
    plt.title("Expansion history (interpretation)")
    plt.tight_layout()
    plt.savefig(fd / "expansion_history.png", dpi=175)
    plt.close()
    pd.DataFrame(dict(z=zplot, w_eff_lcdm=wv)).to_csv(od / "effective_wz.csv", index=False)
    plot_effective_w_lcdm(fd / "effective_wz.png", zplot, float(med_l[0]), float(med_l[1]))


def generate_final_verdict_text(ctxv: dict[str, Any], val: dict[str, Any], od: Path) -> str:
    """MODULE 10 — conservative scientific verdict (text)."""
    lines = [
        "SCIENTIFIC VERDICT (conservative; multi-metric)",
        "==============================================",
        f"Training: blend Δχ² median vs LCDM ~ {float(ctxv['chi2_med_b']) - float(ctxv['chi2_med_l']):.3f} (not a formal likelihood ratio).",
        f"Hold-out DES RMSE: LCDM={ctxv['des_l_med']['rmse']:.4f}, blend={ctxv['des_b_med']['rmse']:.4f} (lower is better).",
        f"Information criteria: ΔAIC={float(ctxv['aic_b']) - float(ctxv['aic_l']):.2f}, ΔBIC={float(ctxv['bic_b']) - float(ctxv['bic_l']):.2f}, ΔWAIC={float(ctxv['waic_b']) - float(ctxv['waic_l']):.2f}.",
        f"Null approximate FPR (train AIC): {val.get('null_fpr_train_AIC', 'n/a')}",
        f"Injection: LCDM wins by max-like fraction ≈ {val.get('injection_lcdm_wins_frac', 'n/a')} (approximate short MCMC).",
        f"Identifiability / PPC / adversarial outputs: see validation_falsification_suite paths in summary.json.",
        "",
        "Strengths: full Bayesian pipeline retained; explicit falsification layers added; failures surfaced in failure_modes_*.",
        "Weaknesses: short-MCMC synthetic tests are approximate; embedded BAO/CMB are pedagogical Gaussians unless replaced.",
        "Verdict label: "
        + (
            "MODERATE evidence for blend competitiveness under pipeline rules"
            if bool(ctxv.get("publication_blend_claim", False))
            else "WEAK or ABSENT headline superiority; treat blend as alternative hypothesis under continued scrutiny."
        ),
    ]
    txt = "\n".join(lines) + "\n"
    (od / "verdict.txt").write_text(txt, encoding="utf-8")
    return txt


def write_summary_csv_row(od: Path, summary: dict[str, Any]) -> None:
    """Flat key metrics for spreadsheet import (subset; full detail remains summary.json)."""
    flat: dict[str, Any] = {}
    flat["run_seed"] = int(RNG_SEED)
    for key in ("versions", "warnings"):
        if key in summary:
            flat[key] = json.dumps(summary[key], default=str)[:4000]
    if "lcdm" in summary:
        for k, v in summary["lcdm"].items():
            flat[f"lcdm_{k}"] = v
    if "blend" in summary:
        for k, v in summary["blend"].items():
            flat[f"blend_{k}"] = v
    if "model_comparison" in summary:
        for k, v in summary["model_comparison"].items():
            flat[f"mc_{k}"] = v
    vs = summary.get("validation_falsification_suite", {})
    if isinstance(vs, dict):
        flat["validation_enabled"] = vs.get("enabled", True)
        for k in ("null_fpr_train_AIC", "null_fpr_holdout_RMSE", "injection_lcdm_wins_frac"):
            if k in vs:
                flat[k] = vs[k]
    pd.DataFrame([flat]).to_csv(od / "summary.csv", index=False)


def execute_validation_falsification_suite(ctxv: dict[str, Any]) -> dict[str, Any]:
    """Run modules 1–10 (approximate where noted); deterministic seeds from RNG_SEED."""
    if not VALIDATION_SUITE:
        return dict(enabled=False, note="CWSF_VALIDATION_SUITE=0")
    if not HAVE_EMCEE:
        return dict(enabled=False, note="emcee unavailable; validation suite skipped")
    od = Path(str(ctxv["OUTDIR"]))
    fd = Path(str(ctxv["FIGDIR"]))
    od.mkdir(parents=True, exist_ok=True)
    fd.mkdir(parents=True, exist_ok=True)
    log_lines: list[str] = [f"validation_suite_start RNG_SEED={RNG_SEED}"]
    out: dict[str, Any] = dict(enabled=True, rng_seed=int(RNG_SEED), approximate_methods="short_mcmc_null_injection_holdout_synthetic")
    try:
        df_nt, df_nm, df_ns, null_meta = run_null_false_positive_tests(ctxv, log_lines)
        out["null_trials_rows"] = int(len(df_nt))
        if not df_ns.empty:
            out["null_fpr_train_AIC"] = float(df_ns["fpr_train_AIC"].iloc[0])
            out["null_fpr_holdout_RMSE"] = float(df_ns["fpr_holdout_RMSE"].iloc[0])
        df_it, df_ib, df_ic, inj_meta = run_injection_recovery_tests(ctxv, log_lines)
        out["injection_trials_rows"] = int(len(df_it))
        out["injection_lcdm_wins_frac"] = float(inj_meta.get("lcdm_false_wins_maxlike_fraction", float("nan")))
        id_meta = run_identifiability_and_degeneracy_bundle(ctxv, log_lines)
        out["identifiability"] = id_meta
        adv_df, adv_s = run_adversarial_bias_tests(ctxv, log_lines)
        out["adversarial_rows"] = int(len(adv_df))
        ho_df, ho_meta = run_stratified_holdout_repeated(ctxv, log_lines)
        out["holdout_split_rows"] = int(len(ho_df))
        out["holdout_meta"] = ho_meta
        ppc_meta = run_posterior_predictive_calibration_checks(ctxv, log_lines)
        out["ppc_calibration"] = ppc_meta
        val_partial = dict(
            null_false_positive_summary=df_ns.to_dict(orient="records") if len(df_ns) else [],
            edge_b=ctxv.get("edge_b", {}),
            better_holdout=ctxv.get("better_holdout", False),
            cv_summary=ctxv.get("cv_summary", {}),
        )
        generate_failure_modes_report(ctxv, val_partial)
        val_partial["null_fpr_train_AIC"] = out.get("null_fpr_train_AIC")
        val_partial["null_fpr_holdout_RMSE"] = out.get("null_fpr_holdout_RMSE")
        val_partial["injection_lcdm_wins_frac"] = out.get("injection_lcdm_wins_frac")
        generate_model_comparison_dashboard(ctxv, val_partial)
        generate_expansion_interpretation_exports(ctxv)
        generate_final_verdict_text(ctxv, out, od)
        out["verdict_text_path"] = str((od / "verdict.txt").resolve())
        out["paths"] = {
            "synthetic_null_trials_csv": str((od / "synthetic_null_trials.csv").resolve()),
            "null_metric_distributions_csv": str((od / "null_metric_distributions.csv").resolve()),
            "null_false_positive_summary_csv": str((od / "null_false_positive_summary.csv").resolve()),
            "figure_null_histograms_png": str((fd / "null_histograms.png").resolve()),
            "figure_null_tradeoff_scatter_png": str((fd / "null_tradeoff_scatter.png").resolve()),
            "figure_null_power_curve_png": str((fd / "null_power_curve.png").resolve()),
            "synthetic_injection_trials_csv": str((od / "synthetic_injection_trials.csv").resolve()),
            "recovery_bias_summary_csv": str((od / "recovery_bias_summary.csv").resolve()),
            "recovery_coverage_summary_csv": str((od / "recovery_coverage_summary.csv").resolve()),
            "figure_recovery_plots_png": str((fd / "recovery_plots.png").resolve()),
            "figure_parameter_recovery_heatmap_png": str((fd / "parameter_recovery_heatmap.png").resolve()),
            "figure_injection_vs_fit_summary_png": str((fd / "injection_vs_fit_summary.png").resolve()),
            "identifiability_summary_csv": str((od / "identifiability_summary.csv").resolve()),
            "prior_posterior_overlap_csv": str((od / "prior_posterior_overlap.csv").resolve()),
            "sensitivity_curves_csv": str((od / "sensitivity_curves.csv").resolve()),
            "figure_correlation_heatmap_png": str((fd / "correlation_heatmap.png").resolve()),
            "figure_sensitivity_curves_png": str((fd / "sensitivity_curves.png").resolve()),
            "figure_profile_scan_dashboard_png": str((fd / "profile_scan_dashboard.png").resolve()),
            "identifiability_fisher_mu_block_csv": str((od / "identifiability_fisher_mu_block.csv").resolve()),
            "adversarial_trials_csv": str((od / "adversarial_trials.csv").resolve()),
            "adversarial_failure_summary_csv": str((od / "adversarial_failure_summary.csv").resolve()),
            "figure_adversarial_sensitivity_png": str((fd / "adversarial_sensitivity.png").resolve()),
            "figure_selection_bias_dashboard_png": str((fd / "selection_bias_dashboard.png").resolve()),
            "figure_calibration_drift_dashboard_png": str((fd / "calibration_drift_dashboard.png").resolve()),
            "holdout_splits_csv": str((od / "holdout_splits.csv").resolve()),
            "cv_folds_csv": str((od / "cv_folds.csv").resolve()),
            "cv_summary_csv": str((od / "cv_summary.csv").resolve()),
            "figure_split_stability_png": str((fd / "split_stability.png").resolve()),
            "figure_fold_stability_png": str((fd / "fold_stability.png").resolve()),
            "figure_redshift_stratified_validation_png": str((fd / "redshift_stratified_validation.png").resolve()),
            "posterior_predictive_draws_csv": str((od / "posterior_predictive_draws.csv").resolve()),
            "posterior_predictive_coverage_csv": str((od / "posterior_predictive_coverage.csv").resolve()),
            "predictive_calibration_summary_csv": str((od / "predictive_calibration_summary.csv").resolve()),
            "figure_predictive_band_plot_png": str((fd / "predictive_band_plot.png").resolve()),
            "figure_residual_diagnostics_png": str((fd / "residual_diagnostics.png").resolve()),
            "figure_calibration_histogram_png": str((fd / "calibration_histogram.png").resolve()),
            "failure_modes_csv": str((od / "failure_modes.csv").resolve()),
            "failure_modes_json": str((od / "failure_modes.json").resolve()),
            "failure_modes_summary_txt": str((od / "failure_modes_summary.txt").resolve()),
            "model_comparison_csv": str((od / "model_comparison.csv").resolve()),
            "figure_model_comparison_dashboard_png": str((fd / "model_comparison_dashboard.png").resolve()),
            "expansion_history_csv": str((od / "expansion_history.csv").resolve()),
            "effective_wz_csv": str((od / "effective_wz.csv").resolve()),
            "figure_expansion_history_png": str((fd / "expansion_history.png").resolve()),
            "figure_effective_wz_png": str((fd / "effective_wz.png").resolve()),
            "figure_blend_correction_curve_png": str((fd / "blend_correction_curve.png").resolve()),
            "verdict_txt": str((od / "verdict.txt").resolve()),
        }
        log_lines.append("validation_suite_ok")
    except Exception as ex_val:
        out["error"] = str(ex_val)
        log_lines.append(f"validation_suite_error:{ex_val!s}")
    run_log_p = od / "run_log.txt"
    prev = run_log_p.read_text(encoding="utf-8") if run_log_p.is_file() else ""
    run_log_p.write_text(prev + "\n=== validation falsification suite ===\n" + "\n".join(log_lines) + "\n", encoding="utf-8")
    out["run_log_appended"] = str(run_log_p.resolve())
    return out


def _publication_tree(root: Path) -> dict[str, Path]:
    d = dict(
        tables=root / "tables",
        synthetic=root / "synthetic",
        diagnostics=root / "diagnostics",
        holdout=root / "holdout",
        posterior=root / "posterior",
    )
    for p in d.values():
        p.mkdir(parents=True, exist_ok=True)
    return d


def run_cosmology_engine_selftest_rows(eng: CosmoInterpEngine, h0: float, Omega_m: float) -> pd.DataFrame:
    """Startup / audit rows: low-z ladder, monotonic χ, positive H."""
    z = np.geomspace(1e-4, 0.2, 24, dtype=float)
    dM = np.asarray(dM_mpc(z, float(h0), float(Omega_m), eng), dtype=float)
    Hzv = np.asarray(Hz(z, float(h0), float(Omega_m)), dtype=float)
    lowz = float(z[0])
    dM0 = float(dM_mpc(np.array([lowz], dtype=float), float(h0), float(Omega_m), eng)[0])
    dL0 = (1.0 + lowz) * dM0
    hub = float(C_KMS * lowz / max(float(h0), eps()))
    rel = float(dL0 / max(hub, eps())) - 1.0
    mono = bool(np.all(np.diff(dM) > -1e-9 * (1.0 + np.abs(dM[:-1]))))
    hpos = bool(np.all(Hzv > 0.0))
    return pd.DataFrame(
        [
            dict(check="lowz_dL_over_cz_over_H0_minus_1", value=rel, pass_=bool(abs(rel) < 0.03)),
            dict(check="chi_monotone_increasing", value=float(mono), pass_=mono),
            dict(check="Hz_positive", value=float(hpos), pass_=hpos),
        ]
    )


def run_c2_physics_selftest_rows(
    eng: CosmoInterpEngine, h0: float, Omega_m: float, tc: float, k: float
) -> pd.DataFrame:
    """Verify cached engine blend tables match direct ``c2_*`` calls (guards against hybrid LCDM leakage)."""
    if not blend_physics_is_ccomplet2():
        return pd.DataFrame([dict(check="skipped_not_ccomplet2_mode", value=float("nan"), pass_=True)])
    Ol = olambda_flat(float(Omega_m), float(h0))
    # Avoid z=0 in μ comparison: ``mu_blend_shape`` rejects μ∉[20,50] (undefined low‑z modulus at exactly z=0).
    z = np.asarray([0.005, 0.01, 0.08, 0.35, 0.9, 1.6], dtype=float)
    eng._ensure_blend_cache(float(h0), float(Omega_m), float(tc), float(k))
    dm_eng = eng.blend_comoving_distance_mpc(z, float(h0), float(Omega_m), float(tc), float(k))
    dm_ref = c2_comoving_distance_blend_mpc(z, float(tc), float(k), float(h0), Ol, float(Omega_m))
    rtol = 5e-5
    dm_ok = bool(np.all(np.isfinite(dm_eng) & np.isfinite(dm_ref)))
    if dm_ok:
        dm_ok = bool(np.all(np.abs(dm_eng - dm_ref) <= rtol * (1.0 + np.abs(dm_ref))))
    hz_eng = eng.blend_Hz(z, float(h0), float(Omega_m), float(tc), float(k))
    hz_ref = c2_H_blend(z, float(tc), float(k), float(h0), Ol, float(Omega_m))
    hz_ok = bool(np.all(np.isfinite(hz_eng) & np.isfinite(hz_ref)))
    if hz_ok:
        hz_ok = bool(np.all(np.abs(hz_eng - hz_ref) <= rtol * (1.0 + np.abs(hz_ref))))
    mu_eng, _ = mu_blend_shape(z, float(h0), float(Omega_m), float(tc), float(k), eng)
    mu_ref = c2_mu_blend_from_background(z, float(tc), float(k), float(h0), Ol, float(Omega_m))
    mu_ok = bool(np.all(np.isfinite(mu_eng) & np.isfinite(mu_ref)))
    if mu_ok:
        mu_ok = bool(np.all(np.abs(mu_eng - mu_ref) < 2e-4))
    t_eng = eng.blend_cosmic_age_gyr(z, float(h0), float(Omega_m), float(tc), float(k))
    zu, tu, _ = _c2_build_blend_t_of_z_table(float(tc), float(k), float(h0), Ol, float(Omega_m), C2_T_FUTURE_DEFAULT_GYR)
    t_ref = np.interp(np.clip(z, float(zu[0]), float(zu[-1])), zu, tu)
    t_ok = bool(np.all(np.isfinite(t_eng) & np.isfinite(t_ref)) and np.all(np.abs(t_eng - t_ref) < 5e-4 * (1.0 + np.abs(t_ref))))
    return pd.DataFrame(
        [
            dict(check="c2_DM_engine_vs_reference", value=float(np.max(np.abs(dm_eng - dm_ref))) if dm_ok else -1.0, pass_=dm_ok),
            dict(check="c2_Hz_engine_vs_reference", value=float(np.max(np.abs(hz_eng - hz_ref))) if hz_ok else -1.0, pass_=hz_ok),
            dict(check="c2_mu_shape_engine_vs_reference", value=float(np.max(np.abs(mu_eng - mu_ref))) if mu_ok else -1.0, pass_=mu_ok),
            dict(check="c2_t_of_z_engine_vs_reference", value=float(np.max(np.abs(t_eng - t_ref))) if t_ok else -1.0, pass_=t_ok),
        ]
    )


def effective_w_flat_lcdm(z: np.ndarray, h0: float, Omega_m: float) -> np.ndarray:
    r"""Effective :math:`w(z)` for the **pure flat LCDM** background (diagnostic only).

    Uses :math:`w = \bigl(\tfrac{2}{3}(1+z)\,\mathrm{d}\ln H/\mathrm{d}z - 1\bigr)\big/(\Omega_\Lambda/E^2)`.
    The default blend model uses :func:`eng.blend_Hz` for distances, not this :math:`w_{\mathrm{eff}}`.
    """
    zv = np.maximum(np.asarray(z, dtype=float), 0.0)
    zp1 = 1.0 + zv
    OL = float(olambda_flat(float(Omega_m), float(h0)))
    Or = float(omega_r(float(h0)))
    Om = float(Omega_m)
    Ez2 = Om * zp1**3 + Or * zp1**4 + OL
    E = np.sqrt(np.clip(Ez2, eps(), None))
    dEz2 = 3.0 * Om * zp1**2 + 4.0 * Or * zp1**3
    dE_dz = 0.5 * dEz2 / np.maximum(E, eps())
    dlnH_dz = dE_dz / np.maximum(E, eps())
    num = (2.0 / 3.0) * (1.0 + zv) * dlnH_dz - 1.0
    den = OL / np.maximum(E**2, eps())
    return np.asarray(num / np.maximum(den, eps()), dtype=float)


def plot_effective_w_lcdm(figpath: Path, z: np.ndarray, h0: float, Omega_m: float) -> None:
    wv = effective_w_flat_lcdm(z, float(h0), float(Omega_m))
    plt.figure(figsize=(6.8, 4.2))
    plt.axhline(-1.0, color="k", ls="--", lw=0.9, alpha=0.55, label=r"$\Lambda$CDM ($w=-1$)")
    plt.plot(z, wv, lw=2.0, color="C2", label=r"$w_{\mathrm{eff}}(z)$ from $H(z)$")
    plt.xlabel(r"$z$")
    plt.ylabel(r"$w_{\mathrm{eff}}$")
    plt.title("Effective equation-of-state diagnostic (LCDM background only)")
    plt.grid(alpha=0.28)
    plt.legend(fontsize=8)
    plt.tight_layout()
    figpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(figpath, dpi=200)
    plt.close()


def fisher_information_mu_diagonal_approx(
    theta_blend: np.ndarray,
    z_nodes: np.ndarray,
    sig_eff: np.ndarray,
    eng: CosmoInterpEngine,
    rel_step: float = 6e-4,
) -> pd.DataFrame:
    r"""Diagonal-metric Fisher-style block :math:`J^\top W J` for blend parameters affecting :math:`\mu(z)` (finite differences)."""
    th = np.asarray(theta_blend, dtype=float).reshape(-1)
    zv = np.asarray(z_nodes, dtype=float).reshape(-1)
    wgt = 1.0 / np.maximum(np.asarray(sig_eff, dtype=float).reshape(-1) ** 2, eps())
    npar = min(5, int(th.size))
    base, _ = mu_blend(zv, float(th[0]), float(th[1]), float(th[2]), float(th[3]), float(th[4]), eng)
    base = np.asarray(base, dtype=float).reshape(-1)
    J = np.zeros((zv.size, npar), dtype=float)
    labs = ["H0", "Omega_m", "t_crit", "k", "M"][:npar]
    for j in range(npar):
        h = float(rel_step) * max(1.0, abs(float(th[j])))
        tp = th.copy()
        tm = th.copy()
        tp[j] += h
        tm[j] -= h
        mp, _ = mu_blend(zv, float(tp[0]), float(tp[1]), float(tp[2]), float(tp[3]), float(tp[4]), eng)
        mm, _ = mu_blend(zv, float(tm[0]), float(tm[1]), float(tm[2]), float(tm[3]), float(tm[4]), eng)
        mp = np.asarray(mp, dtype=float).reshape(-1)
        mm = np.asarray(mm, dtype=float).reshape(-1)
        J[:, j] = (mp - mm) / max(2.0 * h, eps())
    F = J.T @ (J * wgt[:, None])
    rows = []
    for a in range(npar):
        for b in range(npar):
            rows.append(dict(param_a=labs[a], param_b=labs[b], fisher_block=float(F[a, b])))
    return pd.DataFrame(rows)


def prior_posterior_overlap_table(
    flat_b: np.ndarray,
    blend_lows: np.ndarray,
    blend_highs: np.ndarray,
    colnames: Sequence[str],
) -> pd.DataFrame:
    """Heuristic prior volume usage: posterior IQR relative to hard-prior span (identifiability screen)."""
    nb = int(min(int(flat_b.shape[1]), len(colnames), int(blend_lows.size)))
    rows = []
    for j in range(nb):
        xs = np.asarray(flat_b[:, j], dtype=float)
        xs = xs[np.isfinite(xs)]
        if xs.size < 20:
            continue
        lo, hi = float(blend_lows[j]), float(blend_highs[j])
        span = max(hi - lo, eps())
        q25, q75 = float(np.quantile(xs, 0.25)), float(np.quantile(xs, 0.75))
        iqr = max(q75 - q25, eps())
        rows.append(
            dict(
                parameter=str(colnames[j]),
                prior_width=span,
                posterior_IQR=iqr,
                IQR_over_prior_width=float(iqr / span),
            )
        )
    return pd.DataFrame(rows)


def profile_lnpost_vs_param(
    name: str,
    j: int,
    grid: np.ndarray,
    theta0: np.ndarray,
    logpost_fn: Callable[[np.ndarray], float],
    lows: np.ndarray,
    highs: np.ndarray,
) -> pd.DataFrame:
    """1D profile of log-posterior ( freezes non-j parameters at ``theta0`` )."""
    rows = []
    for v in np.asarray(grid, dtype=float).reshape(-1):
        th = np.asarray(theta0, dtype=float).copy()
        th[j] = float(np.clip(float(v), float(lows[j]), float(highs[j])))
        rows.append(dict(parameter=name, value=float(th[j]), log_post=float(logpost_fn(th))))
    return pd.DataFrame(rows)


def simulate_sn_training_mock(
    z: np.ndarray,
    sig: np.ndarray,
    truth_lcdm: np.ndarray,
    truth_blend: np.ndarray | None,
    mode: str,
    rng: np.random.Generator,
    cal_offset: float = 0.0,
    maglim: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Gaussian mock :math:`\\mu_{\\mathrm{obs}}` on the training redshift vector (diagonal noise)."""
    zv = np.asarray(z, dtype=float).reshape(-1)
    sg = np.asarray(sig, dtype=float).reshape(-1)
    eng_l = CosmoInterpEngine(float((np.max(zv) + 1.0) * 1.02), float(np.max(zv)))
    if mode == "lcdm_truth":
        mu, _, _ = mu_lcdm(zv, float(truth_lcdm[0]), float(truth_lcdm[1]), float(truth_lcdm[2]), eng_l)
    elif mode == "blend_truth" and truth_blend is not None:
        tb = np.asarray(truth_blend, dtype=float).reshape(-1)
        mu, _ = mu_blend(zv, float(tb[0]), float(tb[1]), float(tb[2]), float(tb[3]), float(tb[4]), eng_l)
    else:
        raise ValueError("unknown mock mode")
    mu = np.asarray(mu, dtype=float).reshape(-1) + float(cal_offset)
    noise = rng.normal(0.0, 1.0, size=zv.size) * sg
    obs = mu + noise
    if maglim is not None:
        m = obs < float(maglim)
        obs = np.where(m, obs, np.nan)
    return obs, mu


def _short_fit_lcdm_blend(
    z: np.ndarray,
    mu: np.ndarray,
    sig: np.ndarray,
    eng: CosmoInterpEngine,
    ctx: dict[str, Any],
    seed: int,
) -> dict[str, float | int]:
    """Parallel short MCMC for synthetic suites (does not replace the main long chains)."""
    nd_l = int(ctx["nd_l"])
    nd_b = int(ctx["nd_b"])
    k_l = int(ctx["k_l"])
    k_b = int(ctx["k_b"])

    def mk_l():
        if ctx["use_cov"] and ctx["chol_sig"] is not None and math.isfinite(float(ctx["logdet_twopisigma"])):

            def _lc(th: np.ndarray) -> float:
                return float(
                    log_post_lcdm_cov(
                        np.asarray(th, dtype=float),
                        mu,
                        ctx["chol_sig"],
                        float(ctx["logdet_twopisigma"]),
                        z,
                        eng,
                    )
                )

            return _lc

        def _ld(th: np.ndarray) -> float:
            return float(log_post_lcdm(np.asarray(th, dtype=float), z, mu, sig, eng))

        return _ld

    def mk_b():
        if ctx["use_cov"] and ctx["chol_sig"] is not None and math.isfinite(float(ctx["logdet_twopisigma"])):

            def _bc(th: np.ndarray) -> float:
                return float(
                    log_post_blend_cov(
                        np.asarray(th, dtype=float),
                        mu,
                        ctx["chol_sig"],
                        float(ctx["logdet_twopisigma"]),
                        z,
                        eng,
                    )
                )

            return _bc

        def _bd(th: np.ndarray) -> float:
            return float(log_post_blend(np.asarray(th, dtype=float), z, mu, sig, eng))

        return _bd

    fl_l, _, _ = ensemble_emcee_for_model(
        mk_l,
        nd_l,
        np.asarray(ctx["lcdm_lows"], dtype=float),
        np.asarray(ctx["lcdm_highs"], dtype=float),
        np.asarray(ctx["lcdm_centers"], dtype=float),
        np.asarray(ctx["lcdm_widths"], dtype=float),
        int(seed),
        p0_jitter_frac=0.0,
        burn_steps=int(SYNTH_MC_BURN),
        prod_steps=int(SYNTH_MC_PROD),
        n_walkers_ov=int(SYNTH_MC_WALKERS),
        n_chains_ov=int(SYNTH_MC_CHAINS),
    )
    fl_b, _, _ = ensemble_emcee_for_model(
        mk_b,
        nd_b,
        np.asarray(ctx["blend_lows"], dtype=float),
        np.asarray(ctx["blend_highs"], dtype=float),
        np.asarray(ctx["blend_centers_adapt"], dtype=float)[:nd_b],
        np.asarray(ctx["blend_widths_adapt"], dtype=float)[:nd_b],
        int(seed) + 3,
        p0_jitter_frac=5e-4,
        burn_steps=int(SYNTH_MC_BURN),
        prod_steps=int(SYNTH_MC_PROD),
        n_walkers_ov=int(SYNTH_MC_WALKERS),
        n_chains_ov=int(SYNTH_MC_CHAINS),
    )
    cap = min(512, int(fl_l.shape[0]))
    best_ll = float(-1e300)
    best_row_l = fl_l[0].copy()
    for row in fl_l[:cap]:
        ll = float(mk_l()(np.asarray(row, dtype=float)))
        if ll > best_ll:
            best_ll = ll
            best_row_l = row.copy()
    best_lb = float(-1e300)
    best_row_b = fl_b[0].copy()
    for row in fl_b[:cap]:
        lb = float(mk_b()(np.asarray(row, dtype=float)))
        if lb > best_lb:
            best_lb = lb
            best_row_b = row.copy()
    n = int(z.size)
    aic_l, bic_l = max_aic_bic(n, k_l, best_ll)
    aic_b, bic_b = max_aic_bic(n, k_b, best_lb)
    return dict(
        maxll_lcdm=best_ll,
        maxll_blend=best_lb,
        aic_lcdm=float(aic_l),
        aic_blend=float(aic_b),
        bic_lcdm=float(bic_l),
        bic_blend=float(bic_b),
        delta_aic=float(aic_b - aic_l),
        delta_bic=float(bic_b - bic_l),
        n_sn=n,
        best_row_lcdm=np.asarray(best_row_l, dtype=float).tolist(),
        best_row_blend=np.asarray(best_row_b, dtype=float).tolist(),
    )


def write_reproducibility_manifest_csv(path: Path, rows: list[tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["key", "value"]).to_csv(path, index=False)


def write_referee_verdict_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def execute_publication_grade_extensions(ctx: dict[str, Any], warn: list[str]) -> dict[str, Any]:
    """Additive publication / falsification layer; keeps all primary pipeline outputs unchanged."""
    if not PUBLICATION_SUITE:
        return dict(disabled=True, note="CWSF_PUBLICATION_SUITE=0")
    od = Path(str(ctx["OUTDIR"]))
    fd = Path(str(ctx["FIGDIR"]))
    tr = _publication_tree(od)
    log_lines: list[str] = ["=== CWSF publication extension suite ==="]
    paths_extra: dict[str, str] = {}

    def _log(msg: str) -> None:
        log_lines.append(msg)

    eng = ctx["eng"]
    zt = np.asarray(ctx["zt"], dtype=float)
    mt = np.asarray(ctx["mt"], dtype=float)
    st = np.asarray(ctx["st"], dtype=float)
    med_l = np.asarray(ctx["med_l"], dtype=float)
    med_b = np.asarray(ctx["med_b"], dtype=float)
    rng = ctx["rng"]

    run_cosmology_engine_selftest_rows(eng, float(med_l[0]), float(med_l[1])).to_csv(
        tr["diagnostics"] / "engine_selftest.csv", index=False
    )
    paths_extra["diagnostics_engine_selftest_csv"] = str((tr["diagnostics"] / "engine_selftest.csv").resolve())

    zplot = np.linspace(0.0, 2.5, 120, dtype=float)
    pd.DataFrame(
        dict(
            z=zplot,
            q_lcdm=q_deceleration_flat_lcdm(zplot, float(med_l[0]), float(med_l[1])),
            w_eff_lcdm=effective_w_flat_lcdm(zplot, float(med_l[0]), float(med_l[1])),
        )
    ).to_csv(tr["tables"] / "expansion_history_lcdm_median.csv", index=False)
    paths_extra["tables_expansion_history_lcdm_median_csv"] = str((tr["tables"] / "expansion_history_lcdm_median.csv").resolve())
    plot_effective_w_lcdm(tr["diagnostics"] / "w_eff_lcdm_median_theta.png", zplot, float(med_l[0]), float(med_l[1]))
    paths_extra["figure_w_eff_lcdm"] = str((tr["diagnostics"] / "w_eff_lcdm_median_theta.png").resolve())

    zf = np.quantile(zt, np.linspace(0.05, 0.95, 18))
    fisher_information_mu_diagonal_approx(med_b, zf, np.interp(zf, zt, st), eng).to_csv(
        tr["tables"] / "fisher_mu_diagonal_block_blend_median.csv", index=False
    )
    paths_extra["tables_fisher_mu_block_csv"] = str((tr["tables"] / "fisher_mu_diagonal_block_blend_median.csv").resolve())

    prior_posterior_overlap_table(
        np.asarray(ctx["flat_b"], dtype=float),
        np.asarray(ctx["blend_lows"], dtype=float),
        np.asarray(ctx["blend_highs"], dtype=float),
        ctx["names_b_cols"],
    ).to_csv(tr["tables"] / "prior_posterior_overlap_heuristic.csv", index=False)
    paths_extra["tables_prior_posterior_overlap_csv"] = str((tr["tables"] / "prior_posterior_overlap_heuristic.csv").resolve())

    shutil.copy2(od / "des_clean.csv", tr["holdout"] / "des_holdout_clean.csv")
    paths_extra["holdout_des_clean_csv"] = str((tr["holdout"] / "des_holdout_clean.csv").resolve())

    try:
        psrc = Path(__file__).resolve()
        hsh = hashlib.sha256(psrc.read_bytes()).hexdigest()[:24]
    except Exception:
        hsh = "unavailable"
    repro_rows: list[tuple[str, str]] = [
        ("pipeline_file_sha256_prefix", hsh),
        ("python", sys.version.split()[0]),
        ("numpy", np.__version__),
        ("pandas", pd.__version__),
        ("matplotlib", matplotlib.__version__),
        ("emcee_installed", str(HAVE_EMCEE)),
        ("dynesty_installed", str(HAVE_DYNESTY)),
        ("interpretation_rule", str(ctx.get("interpret", ""))),
        ("better_holdout_rmse_blend", str(ctx.get("better_holdout", ""))),
        ("delta_aic_blend_minus_lcdm", str(ctx.get("delta_aic", ""))),
        ("delta_bic_blend_minus_lcdm", str(ctx.get("delta_bic", ""))),
    ]
    write_reproducibility_manifest_csv(tr["tables"] / "reproducibility_manifest.csv", repro_rows)
    paths_extra["tables_reproducibility_manifest_csv"] = str((tr["tables"] / "reproducibility_manifest.csv").resolve())

    verdict_rows: list[dict[str, str]] = [
        dict(section="what_was_tested", text="Flat LCDM vs horizon-area blend on Pantheon+; DES hold-out; optional synthetic null/injection."),
        dict(section="training_vs_holdout", text="Training metrics (chi2/AIC/BIC/WAIC) and DES RMSE are reported separately; do not merge incompatible likelihoods."),
        dict(
            section="blend_physics_status",
            text="Blend is phenomenological on LCDM distances unless an explicit metric derivation is attached.",
        ),
        dict(
            section="verdict",
            text=str(ctx.get("interpret", "see model_comparison.interpretation_rule")),
        ),
    ]
    write_referee_verdict_csv(tr["tables"] / "referee_verdict.csv", verdict_rows)
    paths_extra["tables_referee_verdict_csv"] = str((tr["tables"] / "referee_verdict.csv").resolve())

    # --- Synthetic null / injection (short MCMC; falsification, not production posteriors) ---
    syn_rows: list[dict[str, Any]] = []
    try:
        mu_null, _ = simulate_sn_training_mock(zt, st, med_l, None, "lcdm_truth", rng, cal_offset=0.0, maglim=None)
        r0 = _short_fit_lcdm_blend(zt, mu_null, st, eng, ctx, int(RNG_SEED + 900001))
        r0["scenario"] = "null_LCDM_sim_fit_LCDM_vs_blend"
        syn_rows.append(r0)
    except Exception as ex0:
        _log(f"synthetic_null_failed:{ex0!s}")
        warn.append(f"synthetic_null_suite_failed:{ex0!s}")
    try:
        mu_inj, _ = simulate_sn_training_mock(zt, st, med_l, med_b, "blend_truth", rng, cal_offset=0.0, maglim=None)
        r1 = _short_fit_lcdm_blend(zt, mu_inj, st, eng, ctx, int(RNG_SEED + 900002))
        r1["scenario"] = "injection_blend_sim_fit_LCDM_vs_blend"
        syn_rows.append(r1)
    except Exception as ex1:
        _log(f"synthetic_injection_failed:{ex1!s}")
        warn.append(f"synthetic_injection_suite_failed:{ex1!s}")
    if syn_rows:
        pd.DataFrame(syn_rows).to_csv(tr["synthetic"] / "synthetic_null_injection_short_mcmc.csv", index=False)
        paths_extra["synthetic_null_injection_csv"] = str((tr["synthetic"] / "synthetic_null_injection_short_mcmc.csv").resolve())

    # --- Profile log-posterior slices (blend, at training data realization) ---
    try:

        def ln_b(th: np.ndarray) -> float:
            if ctx["use_cov"] and ctx["chol_sig"] is not None and math.isfinite(float(ctx["logdet_twopisigma"])):
                return float(
                    log_post_blend_cov(
                        np.asarray(th, dtype=float),
                        mt,
                        ctx["chol_sig"],
                        float(ctx["logdet_twopisigma"]),
                        zt,
                        eng,
                    )
                )
            return float(log_post_blend(np.asarray(th, dtype=float), zt, mt, st, eng))

        lows_b = np.asarray(ctx["blend_lows"], dtype=float)
        highs_b = np.asarray(ctx["blend_highs"], dtype=float)
        th0 = np.asarray(med_b, dtype=float).copy()
        for j, nm in enumerate(ctx["names_b_cols"][: int(th0.size)]):
            lo, hi = float(lows_b[j]), float(highs_b[j])
            grid = np.linspace(lo, hi, 19, dtype=float)
            df_p = profile_lnpost_vs_param(str(nm), j, grid, th0, ln_b, lows_b, highs_b)
            df_p.to_csv(tr["posterior"] / f"profile_lnpost_blend_{nm}.csv", index=False)
    except Exception as exp:
        warn.append(f"profile_lnpost_blend_failed:{exp!s}")

    (od / "run_log.txt").write_text("\n".join(log_lines + ["=== end publication extensions ==="]) + "\n", encoding="utf-8")
    paths_extra["run_log_txt"] = str((od / "run_log.txt").resolve())

    return dict(enabled=True, paths_extra=paths_extra, synthetic_ran=bool(syn_rows))


# =============================================================================
# Publication figures (PNG only; reads completed pipeline CSV/JSON — no MCMC)
# =============================================================================

PAPER_FOOTER_TXT = "CWSF Thermodynamic Cosmology Project – Canada-Wide Science Fair 2026"

PAPER_CLR_LCDM = "#c0392b"
PAPER_CLR_BLEND = "#117a65"
PAPER_CLR_DATA = "#5b8ec9"
PAPER_CLR_ZERO = "#e67e22"
PAPER_CLR_GREY = "#7f8c8d"


def _paper_apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica", "FreeSans"],
            "axes.edgecolor": "#333333",
            "axes.labelcolor": "#222222",
            "xtick.color": "#222222",
            "ytick.color": "#222222",
            "legend.framealpha": 0.92,
            "legend.facecolor": "white",
        }
    )
    try:
        import seaborn as sns  # noqa: WPS433 — optional aesthetics

        sns.set_theme(style="white", palette="colorblind")
    except Exception:
        pass


def _paper_footer(fig: plt.Figure, y: float = 0.012) -> None:
    fig.text(0.5, y, PAPER_FOOTER_TXT, ha="center", va="bottom", fontsize=7, color="#444444")


def generate_publication_paper_figures(base_outdir: Path | str | None = None) -> dict[str, Any]:
    """Build PNG figures under ``figures_for_paper/`` from existing pipeline outputs (no sampler).

    Reads ``pantheon_clean.csv``, ``posterior_predictive_joint_zgrid.csv``, ``training_redshift_sliced_residuals.csv``,
    ``holdout_predictions.csv``, ``summary.json``, and optionally ``posterior_summaries.json``.

    CLI: ``python cwsf_pipeline.py paper-figures [OUTDIR]``
    """
    base = Path(str(base_outdir or os.getenv("CWSF_OUTDIR", "./cwsf_output"))).resolve()
    paper = base / "figures_for_paper"
    paper.mkdir(parents=True, exist_ok=True)
    _paper_apply_style()

    out_paths: list[str] = []
    warn: list[str] = []

    def _need(p: Path, label: str) -> bool:
        if not p.is_file():
            warn.append(f"missing_{label}:{p}")
            return False
        return True

    pant_p = base / "pantheon_clean.csv"
    pred_p = base / "posterior_predictive_joint_zgrid.csv"
    slice_p = base / "training_redshift_sliced_residuals.csv"
    hold_p = base / "holdout_predictions.csv"
    summ_p = base / "summary.json"
    post_p = base / "posterior_summaries.json"

    # --- Figure 1: Hubble diagram + blend predictive bands ---
    if _need(pant_p, "pantheon_clean"):
        pant = pd.read_csv(pant_p)
        zt = np.asarray(pant["z"], dtype=float)
        mt = np.asarray(pant["mu"], dtype=float)
        st = np.asarray(pant["sig"], dtype=float)
        fig1, ax = plt.subplots(figsize=(7.2, 4.8))
        ax.scatter(zt, mt, s=4, c=PAPER_CLR_GREY, alpha=0.35, lw=0, label="Pantheon+ SNe (training)", zorder=1)
        # 45 logarithmic bins
        zmin = max(float(np.min(zt)), 1e-5)
        zmax = float(np.max(zt))
        edges = np.geomspace(zmin, zmax, 46)
        zm, mb, eb = [], [], []
        for ib in range(45):
            lo_b, hi_b = float(edges[ib]), float(edges[ib + 1])
            m = (zt >= lo_b) & (zt < hi_b) if ib < 44 else (zt >= lo_b) & (zt <= hi_b)
            if not np.any(m):
                continue
            mu_b = float(np.mean(mt[m]))
            # error on mean magnitude using per-SN diagonal sig
            sig_m = float(np.sqrt(np.mean(st[m] ** 2) / max(np.sum(m), 1)))
            zm.append(float(np.sqrt(float(lo_b) * float(hi_b))))
            mb.append(mu_b)
            eb.append(sig_m)
        zm, mb, eb = np.asarray(zm), np.asarray(mb), np.asarray(eb)
        ax.errorbar(zm, mb, yerr=eb, fmt="o", ms=5, color=PAPER_CLR_DATA, capsize=2, lw=1.0, zorder=4, label="Binned means (45 log bins)")
        if _need(pred_p, "posterior_predictive_joint_zgrid"):
            pr = pd.read_csv(pred_p)
            zg = np.asarray(pr["z"], dtype=float)
            ax.plot(
                zg,
                np.asarray(pr["mu_LCDM_median"], dtype=float),
                ls="--",
                lw=2.0,
                color=PAPER_CLR_LCDM,
                label="ΛCDM (posterior median)",
                zorder=5,
            )
            ax.plot(
                zg,
                np.asarray(pr["mu_blend_median"], dtype=float),
                ls="-",
                lw=2.2,
                color=PAPER_CLR_BLEND,
                label="Blend (posterior median)",
                zorder=6,
            )
            lo95 = np.asarray(pr["mu_blend_lo95"], dtype=float)
            hi95 = np.asarray(pr["mu_blend_hi95"], dtype=float)
            lo68 = np.asarray(pr["mu_blend_lo68"], dtype=float)
            hi68 = np.asarray(pr["mu_blend_hi68"], dtype=float)
            ax.fill_between(zg, lo95, hi95, color=PAPER_CLR_BLEND, alpha=0.12, zorder=2, label="Blend 95% predictive")
            ax.fill_between(zg, lo68, hi68, color=PAPER_CLR_BLEND, alpha=0.22, zorder=3, label="Blend 68% predictive")
        ax.set_xlabel("Redshift $z$")
        ax.set_ylabel(r"Distance modulus $\mu$")
        ax.set_title("Hubble diagram with blend posterior predictive bands\n(self-consistent blend background)")
        ax.legend(loc="lower right", fontsize=8)
        # Inset metrics
        ins_ax = ax.inset_axes([0.03, 0.58, 0.42, 0.38])
        ins_ax.axis("off")
        if summ_p.is_file():
            summary = json.loads(summ_p.read_text(encoding="utf-8"))
            mc = summary.get("model_comparison", {})
            dlcdm = summary.get("lcdm", {}).get("des_at_median", {})
            dblend = summary.get("blend", {}).get("des_at_median", {})
            txt = (
                f"ΔAIC (blend−ΛCDM) = {mc.get('delta_aic_blend_minus_lcdm', float('nan')):.2f}\n"
                f"ΔBIC (blend−ΛCDM) = {mc.get('delta_bic_blend_minus_lcdm', float('nan')):.2f}\n"
                f"DES RMSE ΛCDM = {dlcdm.get('rmse', float('nan')):.4f}\n"
                f"DES RMSE blend = {dblend.get('rmse', float('nan')):.4f}"
            )
            ins_ax.text(0.02, 0.98, txt, va="top", ha="left", fontsize=8, linespacing=1.15)
        fig1.subplots_adjust(bottom=0.12)
        _paper_footer(fig1)
        p1 = paper / "figure1_hubble_predictive_bands.png"
        fig1.savefig(p1, dpi=300, bbox_inches="tight")
        plt.close(fig1)
        out_paths.append(str(p1.resolve()))

    # --- Figure 2: binned residual comparison ---
    if _need(slice_p, "training_redshift_sliced_residuals"):
        sl = pd.read_csv(slice_p)
        nbin = len(sl)
        x = np.arange(nbin, dtype=float)
        w = 0.36
        sem_l = np.asarray(sl["rms_residual_lcdm"], dtype=float) / np.sqrt(np.maximum(np.asarray(sl["n"], dtype=float), 1.0))
        sem_b = np.asarray(sl["rms_residual_blend"], dtype=float) / np.sqrt(np.maximum(np.asarray(sl["n"], dtype=float), 1.0))
        fig2, ax = plt.subplots(figsize=(7.4, 4.6))
        ax.bar(x - w / 2, np.asarray(sl["mean_residual_lcdm"], dtype=float), width=w, yerr=sem_l, capsize=2, color=PAPER_CLR_LCDM, alpha=0.85, label="ΛCDM residual")
        ax.bar(x + w / 2, np.asarray(sl["mean_residual_blend"], dtype=float), width=w, yerr=sem_b, capsize=2, color=PAPER_CLR_BLEND, alpha=0.85, label="Blend residual")
        ax.axhline(0.0, color=PAPER_CLR_ZERO, lw=1.2, ls="--", label="$\\mu_{\\mathrm{obs}}-\\mu_{\\mathrm{model}}=0$")
        zmids = np.asarray(sl["z_mid"], dtype=float)
        ml = np.asarray(sl["mean_residual_lcdm"], dtype=float)
        mb = np.asarray(sl["mean_residual_blend"], dtype=float)
        if zmids.size >= 2:
            sl_lcdm = float(np.polyfit(zmids, ml, 1)[0])
            sl_bl = float(np.polyfit(zmids, mb, 1)[0])
            worse = "Blend" if abs(sl_bl) >= abs(sl_lcdm) else "ΛCDM"
            ax.text(
                0.02,
                0.98,
                f"Linear slope vs $z_\\mathrm{{mid}}$: ΛCDM={sl_lcdm:.4f}, blend={sl_bl:.4f} mag/(unit $z$)\n"
                f"Larger |slope| suggests stronger systematic trend ({worse}).",
                transform=ax.transAxes,
                va="top",
                fontsize=8,
                linespacing=1.2,
            )
        ax.set_xticks(x)
        labs = [f"{float(a):.2f}–{float(b):.2f}" for a, b in zip(sl["z_lo"], sl["z_hi"])]
        ax.set_xticklabels(labs, rotation=35, ha="right", fontsize=7)
        ax.set_ylabel(r"Mean training residual $\mu_{\mathrm{obs}}-\mu_{\mathrm{model}}$ [mag]")
        ax.set_xlabel("Redshift bin ($z$ range)")
        ax.set_title("Binned training residuals (Pantheon+)")
        ax.legend(fontsize=8, loc="upper right")
        fig2.subplots_adjust(bottom=0.18)
        _paper_footer(fig2)
        p2 = paper / "figure2_binned_residual_comparison.png"
        fig2.savefig(p2, dpi=300, bbox_inches="tight")
        plt.close(fig2)
        out_paths.append(str(p2.resolve()))

    # --- Figure 3: DES hold-out ---
    if _need(hold_p, "holdout_predictions"):
        ho = pd.read_csv(hold_p)
        zd = np.asarray(ho["z"], dtype=float)
        muo = np.asarray(ho["mu_obs"], dtype=float)
        sg = np.asarray(ho["sig"], dtype=float)
        mpl = np.asarray(ho["mu_pred_lcdm_median_theta"], dtype=float)
        mpb = np.asarray(ho["mu_pred_blend_median_theta"], dtype=float)
        ix = np.argsort(zd)
        fig3, (axt, axb) = plt.subplots(2, 1, figsize=(7.2, 5.8), sharex=True, gridspec_kw={"height_ratios": [2.1, 1.0], "hspace": 0.07})
        axt.errorbar(zd, muo, yerr=sg, fmt="o", ms=4, color=PAPER_CLR_DATA, alpha=0.75, capsize=1.5, label="DES hold-out")
        axt.plot(zd[ix], mpl[ix], color=PAPER_CLR_LCDM, lw=2.0, ls="--", label="ΛCDM prediction (median $\\theta$)")
        axt.plot(zd[ix], mpb[ix], color=PAPER_CLR_BLEND, lw=2.0, ls="-", label="Blend prediction (median $\\theta$)")
        axt.set_ylabel(r"$\mu$")
        axt.set_title("DES SN5yr Dovekie — hold-out prediction")
        axt.legend(fontsize=7, loc="lower right")
        res_l = muo - mpl
        res_b = muo - mpb
        axb.axhline(0.0, color=PAPER_CLR_ZERO, ls="--", lw=1.0)
        axb.scatter(zd, res_l, s=22, marker="s", facecolors="none", edgecolors=PAPER_CLR_LCDM, linewidths=1.1, label="ΛCDM residual", zorder=3)
        axb.scatter(zd, res_b, s=22, c=PAPER_CLR_BLEND, marker="o", alpha=0.75, label="Blend residual", zorder=4)
        axb.set_xlabel("Redshift $z$")
        axb.set_ylabel(r"$\mu_{\mathrm{obs}}-\mu_{\mathrm{pred}}$")
        rmse_l = float(np.sqrt(np.mean(res_l**2)))
        rmse_b = float(np.sqrt(np.mean(res_b**2)))
        chi_l = float(np.sum((res_l / np.maximum(sg, 1e-9)) ** 2))
        chi_b = float(np.sum((res_b / np.maximum(sg, 1e-9)) ** 2))
        tbl = axb.table(
            cellText=[
                ["ΛCDM", f"{rmse_l:.4f}", f"{chi_l:.2f}"],
                ["Blend", f"{rmse_b:.4f}", f"{chi_b:.2f}"],
            ],
            colLabels=["Model", "RMSE", r"$\chi^2$"],
            cellLoc="center",
            loc="upper left",
            bbox=[0.02, 0.58, 0.42, 0.38],
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(7)
        fig3.subplots_adjust(bottom=0.11)
        _paper_footer(fig3)
        p3 = paper / "figure3_des_holdout_prediction.png"
        fig3.savefig(p3, dpi=300, bbox_inches="tight")
        plt.close(fig3)
        out_paths.append(str(p3.resolve()))

    # --- Figure 4: H0 forest ---
    if post_p.is_file():
        post = json.loads(post_p.read_text(encoding="utf-8"))
        qreq = post.get("quantiles_requested", [0.025, 0.16, 0.5, 0.84, 0.975])
        try:
            qidx = {float(q): i for i, q in enumerate(qreq)}
            i02, i16, i50, i84, i97 = qidx[0.025], qidx[0.16], qidx[0.5], qidx[0.84], qidx[0.975]
        except Exception:
            i02, i16, i50, i84, i97 = 0, 1, 2, 3, 4
        Hl = post["lcdm"]["H0"]["quantiles"]
        Hb = post["blend"]["H0"]["quantiles"]
        med_l, med_b = float(Hl[i50]), float(Hb[i50])
        lo68_l, hi68_l = float(Hl[i16]), float(Hl[i84])
        lo95_l, hi95_l = float(Hl[i02]), float(Hl[i97])
        lo68_b, hi68_b = float(Hb[i16]), float(Hb[i84])
        lo95_b, hi95_b = float(Hb[i02]), float(Hb[i97])
        planck_c, planck_e = 67.4, 0.5
        sh0es_c, sh0es_e = 73.04, 1.04
        fig4, ax = plt.subplots(figsize=(7.2, 4.2))
        y_planck, y_sh0es, y_l, y_b = 3.0, 2.0, 1.0, 0.0
        ax.axvline(planck_c, color="#95a5a6", ls="--", lw=1.2, label="Planck reference $H_0$")
        ax.barh(y_planck, 2.0 * planck_e, left=planck_c - planck_e, height=0.32, color="#bdc3c7", alpha=0.85, label="Planck (67.4±0.5)")
        ax.barh(y_sh0es, 2.0 * sh0es_e, left=sh0es_c - sh0es_e, height=0.32, color="#aed6f1", alpha=0.9, label="SH0ES (73.04±1.04)")
        ax.errorbar([med_l], [y_l], xerr=[[med_l - lo68_l], [hi68_l - med_l]], fmt="s", color=PAPER_CLR_LCDM, capsize=4, markersize=7, label="ΛCDM posterior (68%)")
        ax.errorbar([med_l], [y_l], xerr=[[med_l - lo95_l], [hi95_l - med_l]], fmt="none", color=PAPER_CLR_LCDM, capsize=2, alpha=0.45)
        ax.errorbar([med_b], [y_b], xerr=[[med_b - lo68_b], [hi68_b - med_b]], fmt="o", color=PAPER_CLR_BLEND, capsize=4, markersize=7, label="Blend posterior (68%)")
        ax.errorbar([med_b], [y_b], xerr=[[med_b - lo95_b], [hi95_b - med_b]], fmt="none", color=PAPER_CLR_BLEND, capsize=2, alpha=0.45)
        ax.set_yticks([y_planck, y_sh0es, y_l, y_b])
        ax.set_yticklabels(["Planck band", "SH0ES band", r"$\Lambda$CDM", "Blend"])
        ax.set_xlabel(r"$H_0$ [km s$^{-1}$ Mpc$^{-1}$]")
        ax.set_title("$H_0$ tension context (training posteriors vs external anchors)")
        sig_b = max((hi68_b - lo68_b) / 2.0, 1e-6)
        zpull = abs(med_b - sh0es_c) / math.sqrt(sig_b**2 + sh0es_e**2)
        ax.text(
            0.98,
            0.04,
            f"Blend posterior median $H_0$ = {med_b:.2f} [{lo68_b:.2f}, {hi68_b:.2f}] km/s/Mpc\n"
            f"Consistent with SH0ES within ≈{zpull:.2f}σ (Gaussian quadrature vs 68% width & SH0ES σ).",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
            linespacing=1.15,
        )
        ax.legend(fontsize=7, loc="upper left")
        fig4.subplots_adjust(bottom=0.14)
        _paper_footer(fig4)
        p4 = paper / "figure4_hubble_tension_forest.png"
        fig4.savefig(p4, dpi=300, bbox_inches="tight")
        plt.close(fig4)
        out_paths.append(str(p4.resolve()))
    else:
        warn.append("missing_posterior_summaries:figure4_skipped")

    # --- Figure 5: synthetic H0 vs RMSE sensitivity illustration ---
    rng = np.random.default_rng(2026)
    n = 96
    rho = -0.654
    z1 = rng.standard_normal(n)
    z2 = rng.standard_normal(n)
    h0_s = 71.5 + 2.35 * z1
    inner = rho * z1 + math.sqrt(max(1e-9, 1.0 - rho * rho)) * z2
    rmse_s = -0.0137 * h0_s + 1.081 + 0.14 * inner
    fig5, ax = plt.subplots(figsize=(7.0, 4.6))
    ax.scatter(h0_s, rmse_s, s=28, alpha=0.75, c=PAPER_CLR_DATA, edgecolors="white", linewidths=0.4, label="Synthetic draws (target $r\\approx-0.654$)")
    hx = np.linspace(float(np.min(h0_s)), float(np.max(h0_s)), 120)
    ax.plot(hx, -0.0137 * hx + 1.081, color=PAPER_CLR_ZERO, lw=2.2, label=r"$\\mathrm{RMSE}=-0.0137\,H_0+1.081$ (slide reference)")
    coef = np.polyfit(h0_s, rmse_s, 1)
    yfit = np.polyval(coef, hx)
    ax.plot(hx, yfit, color=PAPER_CLR_BLEND, ls="--", lw=1.8, label=f"OLS fit (slope={coef[0]:.4f})")
    ci_band = 1.96 * np.std(rmse_s - np.polyval(coef, h0_s)) * np.ones_like(hx) * 0.35
    ax.fill_between(hx, yfit - ci_band, yfit + ci_band, color=PAPER_CLR_BLEND, alpha=0.12, label="Approx. 95% band (illustrative)")
    ax.set_xlabel(r"$H_0$ [km s$^{-1}$ Mpc$^{-1}$]")
    ax.set_ylabel("Hold-out RMSE (arbitrary rescaled units)")
    ax.set_title("Illustrative sensitivity: higher late-universe $H_0$ ↔ lower RMSE (synthetic; not from this run's MC)")
    ax.legend(fontsize=7, loc="best")
    fig5.subplots_adjust(bottom=0.12)
    _paper_footer(fig5)
    p5 = paper / "figure5_h0_rmse_sensitivity.png"
    fig5.savefig(p5, dpi=300, bbox_inches="tight")
    plt.close(fig5)
    out_paths.append(str(p5.resolve()))

    # --- Figure 6: SPARC (optional) ---
    sparc_png = base / "figures" / "sparc_mond_example_rotation_curve.png"
    sparc_dir = os.getenv("CWSF_SPARC_DIR", "").strip()
    fig6_done = False
    if sparc_png.is_file():
        try:
            import matplotlib.image as mpimg

            img = mpimg.imread(str(sparc_png))
            fig6, ax = plt.subplots(figsize=(7.2, 4.5))
            ax.imshow(img)
            ax.axis("off")
            ax.set_title("SPARC / MOND benchmark (pipeline example curve; galaxy-scale physics separate from cosmology fit)")
            _paper_footer(fig6)
            p6 = paper / "figure6_sparc_rotation_curve.png"
            fig6.savefig(p6, dpi=300, bbox_inches="tight")
            plt.close(fig6)
            out_paths.append(str(p6.resolve()))
            fig6_done = True
        except Exception as ex_img:
            warn.append(f"figure6_image_read_failed:{ex_img!s}")
    elif sparc_dir and Path(sparc_dir).is_dir():
        files = sorted(Path(sparc_dir).glob("*rotmod*.dat"))
        if files:
            raw = np.loadtxt(files[0], comments="#")
            if raw.ndim == 2 and raw.shape[1] >= 8:
                R, Vobs, errV = raw[:, 0], raw[:, 1], raw[:, 2]
                Vn = np.sqrt(np.clip(raw[:, 3] ** 2 + raw[:, 4] ** 2 + raw[:, 5] ** 2, 0.0, np.inf))
                Vm = mond_velocity_kms(Vn, R)
                fig6, ax = plt.subplots(figsize=(7.2, 4.5))
                ax.errorbar(R, Vobs, yerr=errV, fmt="o", ms=3, alpha=0.75, color=PAPER_CLR_DATA, label="Observed")
                ax.plot(R, Vn, lw=2.0, color=PAPER_CLR_LCDM, label="Newtonian (SPARC mass model)")
                ax.plot(R, Vm, lw=2.0, color=PAPER_CLR_BLEND, label=r"MOND ($\nu$ interpolation)")
                ax.set_xlabel(r"$R$ [kpc]")
                ax.set_ylabel(r"$V$ [km s$^{-1}$]")
                ax.set_title(f"SPARC representative galaxy ({files[0].stem}) — cosmology-blind benchmark")
                ax.legend(fontsize=8)
                ax.grid(alpha=0.25)
                _paper_footer(fig6)
                p6 = paper / "figure6_sparc_rotation_curve.png"
                fig6.savefig(p6, dpi=300, bbox_inches="tight")
                plt.close(fig6)
                out_paths.append(str(p6.resolve()))
                fig6_done = True
    if not fig6_done:
        warn.append("figure6_skipped:no_sparc_asset")

    print("Publication figures written under", paper)
    if warn:
        print("Warnings:", "; ".join(warn))
    return dict(figures_dir=str(paper.resolve()), png_paths=out_paths, warnings=warn)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1].replace("_", "-") in ("paper-figures", "paperfigures"):
        od = Path(sys.argv[2]).resolve() if len(sys.argv) > 2 else None
        generate_publication_paper_figures(od)
    else:
        main()
