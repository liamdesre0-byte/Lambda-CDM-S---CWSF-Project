#!/usr/bin/env python3
r"""
ΛCDM+S sigmoid dataset generator (**CSV export only — does not run MCMC**)
==========================================================================

Uses cosmological numerics from ``cwsf_pipeline.py`` (same ``c2_*`` blend as the
embedded ``ccomplet2``/Superior-suite port): logistic :math:`w(t)`, GR vs
thermodynamic branches, and ``CosmoInterpEngine``.

Chain CSVs must carry :math:`H_0,\Omega_m,t_{\rm crit},k` (see ``BLEND_CHAIN_PARAM_ALIASES``
for accepted header spellings such as ``H_0``, ``omega_m``, ``tcrit``, ``kappa``).

**This script never runs ``emcee``, never calls ``cwsf_pipeline.main()``, and never
reruns inference.** It only reads existing chain CSVs produced earlier by
``cwsf_pipeline.py`` (or generates physics tables from closed-form/grid defaults).

Reference posterior centers (your analysis defaults)
----------------------------------------------------
* :math:`t_{\rm crit}=15.1` Gyr (prior center **15.9** Gyr)
* :math:`H_0=72.8` km/s/Mpc (prior **72.5**)
* :math:`k=0.36` Gyr\ :sup:`-1` (prior **0.37**)
* :math:`\Omega_m=0.30` (prior **0.31**)
* :math:`\Omega_\Lambda\approx 0.688` summary (prior **0.685**; **flat** closure in code uses
  :func:`cwsf_pipeline.olambda_flat` from :math:`H_0,\Omega_m`)

Grid defaults bracket these late-time transition parameters (**not** 4–13.5 Gyr).

Modes
-----
* **chain** *(default)* — Read ``mcmc_chain_blend.csv``, write **only**
  ``LambdaCDM_plus_S_posterior_sample_scalars.csv`` (one row per stored sample).
  Uses **chunked** CSV reads so multi-million-row chains need not fit in RAM.
* **grid** — rectangular scan in ``(k,\ t_{\rm crit})`` at fixed
  :math:`(H_0,\Omega_m)` (posterior-centered defaults).
* **posterior-full** — Optional **heavy** mode: full :math:`z`-evolution table per chain row
  (large disk use). Most users should use **chain** only.

Outputs (under ``--outdir``)
----------------------------
* ``LambdaCDM_plus_S_posterior_sample_scalars.csv`` (+ ``.gz``) — from **chain** mode.
* ``LambdaCDM_plus_S_sigmoid_parameter_scan.csv`` — from **grid** or **posterior-full**.

Environment knobs
-----------------
``CWSF_SIGMOID_K_MIN``, ``CWSF_SIGMOID_K_MAX``, ``CWSF_SIGMOID_N_K``,
``CWSF_SIGMOID_TCRIT_MIN``, ``CWSF_SIGMOID_TCRIT_MAX``, ``CWSF_SIGMOID_N_T``,
``CWSF_SIGMOID_N_Z``, ``CWSF_SIGMOID_ZMAX``, ``CWSF_SIGMOID_POSTERIOR_STRIDE``,
``CWSF_SIGMOID_POSTERIOR_MAX_ROWS``, ``CWSF_SIGMOID_CHUNK_ROWS``,
``CWSF_SIGMOID_CHAIN_READ_CHUNKS``, ``CWSF_OUTDIR``, ``CWSF_BLEND_PHYSICS_MODE=ccomplet2``.

Run::

    # Fast: scalars only from existing blend chain (default)
    python sigmoid_dataset.py --outdir ./cwsf_output

    python sigmoid_dataset.py --mode grid --outdir ./cwsf_output

"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

_CWD = Path(__file__).resolve().parent
if str(_CWD) not in sys.path:
    sys.path.insert(0, str(_CWD))


class _LazyCwsfPipeline:
    """Defer importing ``cwsf_pipeline`` (SciPy stack) until first use."""

    _mod: Any = None

    def __getattr__(self, name: str) -> Any:
        if self._mod is None:
            import cwsf_pipeline as _m

            self._mod = _m
        return getattr(self._mod, name)


cw = _LazyCwsfPipeline()


def discover_mcmc_chain_blend(cli_chain: str | None, outdir: Path) -> tuple[Path, list[str]]:
    """Locate ``mcmc_chain_blend.csv``: CLI → outdir → script dir → rglob under script dir & outdir."""
    tried: list[str] = []
    ordered: list[Path] = []

    if cli_chain:
        ordered.append(Path(cli_chain).expanduser().resolve())
    ordered.append((outdir / "mcmc_chain_blend.csv").resolve())
    ordered.append((_CWD / "mcmc_chain_blend.csv").resolve())
    ordered.append((_CWD / "cwsf_output" / "mcmc_chain_blend.csv").resolve())

    seen: set[str] = set()
    for p in ordered:
        ps = str(p)
        if ps in seen:
            continue
        seen.add(ps)
        tried.append(ps)
        if p.is_file():
            return p.resolve(), tried

    for root in (_CWD, outdir):
        if not root.exists():
            continue
        try:
            for hit in root.rglob("mcmc_chain_blend.csv"):
                rp = hit.resolve()
                rs = str(rp)
                if rs in seen:
                    continue
                seen.add(rs)
                tried.append(rs)
                return rp, tried
        except Exception as ex_r:
            tried.append(f"rglob_error:{root}:{ex_r!s}")

    raise FileNotFoundError(
        "Could not find mcmc_chain_blend.csv. Searched:\n  - "
        + "\n  - ".join(tried[:40])
        + ("\n  ..." if len(tried) > 40 else "")
        + "\nPass --chain FULL_PATH or place the file next to this script / under --outdir."
    )


# Blend chain CSVs may use different header spellings; we normalize to these internal names.
BLEND_CHAIN_PARAM_ALIASES: dict[str, tuple[str, ...]] = {
    "H0": (
        "H0",
        "H_0",
        "h0",
        "H0_km_s_Mpc",
        "H0_kms",
        "H0_km_s",
        "h_0",
    ),
    "Omega_m": (
        "Omega_m",
        "omega_m",
        "Om",
        "OmegaM",
        "omegam",
        "Omega_matter",
    ),
    "t_crit": (
        "t_crit",
        "t_crit_Gyr",
        "tcrit",
        "tc",
        "T_crit",
        "t_c",
    ),
    "k": (
        "k",
        "k_Gyr_inv",
        "kappa",
        "slope",
    ),
}


def _norm_chain_col(name: str) -> str:
    return str(name).strip().lower()


def blend_chain_column_rename_map(columns: pd.Index) -> dict[str, str]:
    """Map original CSV column names → canonical ``H0``, ``Omega_m``, ``t_crit``, ``k``."""
    lower_to_orig: dict[str, str] = {}
    for c in columns:
        nk = _norm_chain_col(str(c))
        if nk not in lower_to_orig:
            lower_to_orig[nk] = str(c)

    renames: dict[str, str] = {}
    for canon, aliases in BLEND_CHAIN_PARAM_ALIASES.items():
        for a in aliases:
            nk = _norm_chain_col(a)
            if nk not in lower_to_orig:
                continue
            orig = lower_to_orig[nk]
            if orig != canon:
                renames[orig] = canon
            break

    return renames


def normalize_blend_chain_columns(df: pd.DataFrame) -> pd.DataFrame:
    mp = blend_chain_column_rename_map(df.columns)
    return df.rename(columns=mp, copy=False) if mp else df


# Reference posteriors / priors (documentation; physics uses flat Ω_Λ(H0,Ωm))
REFERENCE_POSTERIOR_CENTERS: dict[str, float] = dict(
    H0_km_s_Mpc=72.8,
    Omega_m=0.30,
    Omega_Lambda_summary=0.688,
    t_crit_Gyr=15.1,
    k_Gyr_inv=0.36,
)
REFERENCE_PRIOR_CENTERS: dict[str, float] = dict(
    H0_km_s_Mpc=72.5,
    Omega_m=0.31,
    Omega_Lambda_summary=0.685,
    t_crit_Gyr=15.9,
    k_Gyr_inv=0.37,
)


# ---------------------------------------------------------------------------
# Defaults (override via CLI / env)
# ---------------------------------------------------------------------------

def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return float(default)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return int(default)


# ---------------------------------------------------------------------------
# Logistic w(t) and explicit branch blend (matches c2_log10_A_blend late branch)
# ---------------------------------------------------------------------------


def w_sigmoid_logistic(t_gyr: np.ndarray, k: float, t_crit: float) -> np.ndarray:
    r""":math:`w(t)=1/(1+\exp(-k(t-t_{\rm crit})))` with stable clipping."""
    t = np.asarray(t_gyr, dtype=float)
    return 1.0 / (1.0 + np.exp(np.clip(-float(k) * (t - float(t_crit)), -500.0, 500.0)))


def A_GR_slide_proxy_m2(t_gyr: np.ndarray, t0_gyr: float | None = None) -> np.ndarray:
    r"""Slide-deck scaling :math:`A_{\rm GR}\propto (t/t_0)^{4/3}` in **linear** m².

    Uses ``C2_A0`` as :math:`A_0`. Reference time defaults to ``T_REF_GYR``.
    """
    t0 = float(t0_gyr) if t0_gyr is not None else float(cw.T_REF_GYR)
    t = np.maximum(np.asarray(t_gyr, dtype=float), 1e-12)
    return np.asarray(cw.C2_A0 * (t / t0) ** (4.0 / 3.0), dtype=float)


def A_thermo_slide_proxy_m2(
    t_gyr: np.ndarray,
    h0_kms: float,
    omega_lambda: float,
    t_future_gyr: float | None = None,
) -> np.ndarray:
    r"""Thermodynamic branch in **linear** m² matching ``c2_log10_A_thermo``:

    :math:`A_{\rm thermo} = 10^{\,\mathrm{log}_{10}A_{\rm thermo}}` with the same
    :math:`H_\Lambda (t-t_{\rm future})` structure as the pipeline.
    """
    tf = float(cw.C2_T_FUTURE_DEFAULT_GYR) if t_future_gyr is None else float(t_future_gyr)
    log10 = cw.c2_log10_A_thermo(
        np.asarray(t_gyr, dtype=float),
        float(h0_kms),
        float(omega_lambda),
        t_future_gyr=tf,
    )
    return np.asarray(10.0 ** np.clip(log10, -300.0, 300.0), dtype=float)


def A_blend_linear_from_branches(
    t_gyr: np.ndarray,
    k: float,
    t_crit: float,
    h0_kms: float,
    omega_lambda: float,
    omega_m: float,
    t_future_gyr: float | None = None,
) -> np.ndarray:
    r""":math:`A_{\Lambda{\rm CDM}+S}(t)=(1-w)A_{\rm GR,branch}+w A_{\rm thermo,branch}`
    using **pipeline** GR / thermo definitions (rad–matter + thermo), identical
    in spirit to :func:`cwsf_pipeline.c2_log10_A_blend` for :math:`t\ge t_{\rm lock}`.
    """
    tf = float(cw.C2_T_FUTURE_DEFAULT_GYR) if t_future_gyr is None else float(t_future_gyr)
    t = np.asarray(t_gyr, dtype=float)
    w = w_sigmoid_logistic(t, k, t_crit)
    lg = cw.c2_log10_A_rad_matter(t, float(h0_kms), float(omega_m), cw.C2_OM_R_STD)
    lt = cw.c2_log10_A_thermo(t, float(h0_kms), float(omega_lambda), t_future_gyr=tf)
    a_gr = 10.0**lg
    a_th = 10.0 ** np.clip(lt, -300.0, 500.0)
    return (1.0 - w) * a_gr + w * a_th


# ---------------------------------------------------------------------------
# Kinematic proxies (same construction as best-fit export pattern)
# ---------------------------------------------------------------------------


def _w_eff_single_fluid_proxy_from_hz(
    z: np.ndarray,
    H_kms_mpc: np.ndarray,
    h0_kms: float,
    omega_m: float,
) -> np.ndarray:
    zv = np.maximum(np.asarray(z, dtype=float), 0.0)
    zp1 = 1.0 + zv
    h0f = float(h0_kms)
    om = float(omega_m)
    orad = float(cw.omega_r(h0f))
    hz = np.asarray(H_kms_mpc, dtype=float)
    ez = hz / max(h0f, cw.eps())
    ez2 = np.clip(ez**2, cw.eps(), None)
    om_z = om * zp1**3 / ez2
    or_z = orad * zp1**4 / ez2
    ol_z = np.clip(1.0 - om_z - or_z, cw.eps(), None)
    dlnh_dz = np.gradient(np.log(np.clip(hz, cw.eps(), None)), zv, edge_order=2)
    num = (2.0 / 3.0) * (1.0 + zv) * dlnh_dz - 1.0
    den = ol_z / np.maximum(ez2, cw.eps())
    return np.asarray(num / np.maximum(den, cw.eps()), dtype=float)


def _q_deceleration_from_hz(z: np.ndarray, H_kms_mpc: np.ndarray) -> np.ndarray:
    zv = np.maximum(np.asarray(z, dtype=float), 0.0)
    hz = np.asarray(H_kms_mpc, dtype=float)
    dhdz = np.gradient(hz, zv, edge_order=2)
    return np.asarray(-1.0 + (1.0 + zv) * dhdz / np.maximum(hz, cw.eps()), dtype=float)


def _fixed_eta_via_cumtrapz(z_asc: np.ndarray, H_kms_mpc: np.ndarray) -> np.ndarray:
    from scipy.integrate import cumulative_trapezoid

    zv = np.asarray(z_asc, dtype=float)
    hv = np.asarray(H_kms_mpc, dtype=float)
    zp1 = 1.0 + zv
    integrand = cw.C_KMS / np.maximum(zp1 * hv, cw.eps())
    return np.asarray(cumulative_trapezoid(integrand, zv, initial=0.0), dtype=float)


def az_mapping_note() -> str:
    """Return documentation string for :math:`A(z)` reconstruction."""
    return (
        "For each parameter set the table lists synchronized columns "
        "(z, t_cosmic_age_Gyr, A_horizon_m2). Mapping A(z) is the tabulated pair "
        "(z, A_horizon_m2); mapping z(A) uses monotone interpolation in z."
    )


# ---------------------------------------------------------------------------
# Redshift grid
# ---------------------------------------------------------------------------


def build_z_grid(n_z: int, z_max: float) -> np.ndarray:
    z_max = float(max(z_max, 10.0))
    n = max(int(n_z), 400)
    z_lo = np.geomspace(1.0e-5, 0.08, max(300, n // 3))
    z_mid = np.geomspace(0.08, min(12.0, z_max), max(350, n // 3))
    z_hi = np.geomspace(min(12.0, z_max) * 1.001, z_max, max(250, n // 3))
    zv = np.unique(np.concatenate([z_lo, z_mid, z_hi]))
    return np.sort(np.clip(zv, 1e-8, z_max)).astype(float)


def refine_z_near_t_crit(
    z_base: np.ndarray,
    eng: cw.CosmoInterpEngine,
    h0: float,
    omega_m: float,
    t_crit: float,
    k_slope: float,
    n_extra: int = 48,
) -> np.ndarray:
    """Insert extra **z** nodes where cosmic age is near ``t_crit`` (blend transition window)."""
    t_tab = np.asarray(eng.blend_cosmic_age_gyr(z_base, h0, omega_m, t_crit, k_slope), dtype=float)
    mask = np.abs(t_tab - float(t_crit)) < 2.5
    if not np.any(mask):
        return z_base
    z_sub = z_base[mask]
    if z_sub.size < 2:
        return z_base
    z_lo = float(np.min(z_sub))
    z_hi = float(np.max(z_sub))
    z_extra = np.geomspace(max(z_lo * 0.98, 1e-8), min(z_hi * 1.02, z_base.max()), n_extra)
    return np.sort(np.unique(np.concatenate([z_base, z_extra]))).astype(float)


# ---------------------------------------------------------------------------
# Single-parameter-set evolution table
# ---------------------------------------------------------------------------


def evolution_table(
    eng: cw.CosmoInterpEngine,
    h0: float,
    omega_m: float,
    t_crit: float,
    k_slope: float,
    zv: np.ndarray,
    *,
    pair_id: int,
    scan_mode: str,
    posterior_weight: float | None = None,
    log_likelihood: float | None = None,
    chi2_train: float | None = None,
) -> pd.DataFrame:
    """Build long-form evolution rows for one *(H0, Ωm, k, t_crit)* combination."""
    h0 = float(h0)
    omega_m = float(omega_m)
    t_crit = float(t_crit)
    k_slope = float(k_slope)
    om_l = float(cw.olambda_flat(omega_m, h0))

    Hp = np.asarray(eng.blend_Hz(zv, h0, omega_m, t_crit, k_slope), dtype=float)
    Hc = np.asarray(cw.Hz(zv, h0, omega_m), dtype=float)
    chi_p = np.asarray(eng.blend_comoving_distance_mpc(zv, h0, omega_m, t_crit, k_slope), dtype=float)
    chi_c = np.asarray(eng.comoving_distance_mpc(zv, h0, omega_m), dtype=float)

    t_p = np.asarray(eng.blend_cosmic_age_gyr(zv, h0, omega_m, t_crit, k_slope), dtype=float)
    t_c = np.asarray(eng.cosmic_age_gyr(zv, h0, omega_m), dtype=float)

    zp1 = 1.0 + zv
    a_arr = 1.0 / np.maximum(zp1, cw.eps())
    d_l = (1.0 + zv) * chi_p
    d_a = chi_p / np.maximum(zp1, cw.eps())

    q_p = _q_deceleration_from_hz(zv, Hp)
    q_c = _q_deceleration_from_hz(zv, Hc)
    w_p = _w_eff_single_fluid_proxy_from_hz(zv, Hp, h0, omega_m)
    w_c = _w_eff_single_fluid_proxy_from_hz(zv, Hc, h0, omega_m)
    eta_p = _fixed_eta_via_cumtrapz(zv, Hp)

    log10_A_p = cw.c2_log10_A_blend(t_p, t_crit, k_slope, h0, om_l, omega_m)
    log10_A_c = cw.c2_log10_A_lcdm(t_c, h0, om_l, omega_m)
    A_p = 10.0 ** np.clip(log10_A_p, -300.0, 300.0)
    A_c = 10.0 ** np.clip(log10_A_c, -300.0, 300.0)

    w_of_t = w_sigmoid_logistic(t_p, k_slope, t_crit)
    A_explicit = A_blend_linear_from_branches(t_p, k_slope, t_crit, h0, om_l, omega_m)

    entropy_proxy = np.asarray(
        cw.c2_r_blend(t_p, t_crit, k_slope, h0, om_l, omega_m),
        dtype=float,
    )
    dA_dt = np.gradient(A_p, np.maximum(t_p, 1.0e-12), edge_order=2)

    delta_H = Hp - Hc
    delta_q = q_p - q_c
    delta_w = w_p - w_c
    delta_chi = chi_p - chi_c
    delta_A = A_p - A_c

    def _pct(num: np.ndarray, den: np.ndarray) -> np.ndarray:
        d = np.maximum(np.abs(den), cw.eps())
        return np.asarray(100.0 * num / d, dtype=float)

    def _pct_s(num: np.ndarray, den: np.ndarray) -> np.ndarray:
        sc = np.maximum(np.abs(den), 1e-8)
        return np.asarray(100.0 * num / sc, dtype=float)

    rows = dict(
        scan_mode=np.full(zv.shape[0], scan_mode),
        pair_id=np.full(zv.shape[0], int(pair_id)),
        k=np.full(zv.shape[0], k_slope),
        t_crit_Gyr=np.full(zv.shape[0], t_crit),
        entropy_amplitude=np.full(zv.shape[0], np.nan),
        transition_width=np.full(zv.shape[0], np.nan),
        saturation_scale=np.full(zv.shape[0], np.nan),
        H_0=np.full(zv.shape[0], h0),
        Omega_M=np.full(zv.shape[0], omega_m),
        Omega_Lambda=np.full(zv.shape[0], om_l),
        z=zv,
        a=a_arr,
        t_cosmic_age_Gyr=t_p,
        H_LambdaCDM_plus_S_km_s_Mpc=Hp,
        E_LambdaCDM_plus_S=Hp / max(h0, cw.eps()),
        chi_comoving_Mpc=chi_p,
        eta_conformal_Mpc=eta_p,
        luminosity_distance_Mpc=d_l,
        angular_diameter_distance_Mpc=d_a,
        q_numeric_proxy=q_p,
        w_eff_single_fluid_proxy=w_p,
        A_horizon_m2=A_p,
        log10_A_horizon_m2=log10_A_p,
        A_blend_explicit_m2=A_explicit,
        w_sigmoid_of_t_cosmic=w_of_t,
        entropy_proxy=entropy_proxy,
        dA_dt_proxy=dA_dt,
        delta_H=delta_H,
        delta_q=delta_q,
        delta_w_eff=delta_w,
        delta_chi=delta_chi,
        delta_A=delta_A,
        pct_H=_pct(delta_H, Hc),
        pct_q=_pct_s(delta_q, q_c),
        pct_w_eff=_pct_s(delta_w, w_c),
        pct_chi=_pct(delta_chi, chi_c),
        pct_A=_pct(delta_A, A_c),
        posterior_weight=np.full(zv.shape[0], np.nan if posterior_weight is None else float(posterior_weight)),
        log_likelihood=np.full(zv.shape[0], np.nan if log_likelihood is None else float(log_likelihood)),
        chi2=np.full(zv.shape[0], np.nan if chi2_train is None else float(chi2_train)),
    )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_chunk(df: pd.DataFrame) -> dict[str, Any]:
    z = df["z"].to_numpy(dtype=float)
    mono_z = bool(np.all(np.diff(np.sort(z)) >= 0)) or z.size <= 1
    fin = bool(np.all(np.isfinite(df.select_dtypes(include=[np.number]).to_numpy())))
    hok = bool(np.all(df["H_LambdaCDM_plus_S_km_s_Mpc"].to_numpy(dtype=float) > 0.0))
    return dict(finite=fin, H_positive=hok, z_sorted_ok=mono_z, n=int(len(df)))


def lcdm_recovery_check(
    eng: cw.CosmoInterpEngine,
    h0: float,
    omega_m: float,
    z_test: np.ndarray,
) -> dict[str, float]:
    """Push ``k→0`` / ``t_crit`` extreme should approximate LCDM distances (diagnostic)."""
    om_l = float(cw.olambda_flat(omega_m, h0))
    tc_big = 1.0e6
    k_tiny = 1.0e-8
    if not cw.validate_blend_background(float(h0), float(omega_m), tc_big, k_tiny, eng, 5.0):
        return dict(ok=0.0, max_rel_delta_chi=float("nan"))
    chi_b = np.asarray(
        eng.blend_comoving_distance_mpc(z_test, float(h0), float(omega_m), tc_big, k_tiny),
        dtype=float,
    )
    chi_l = np.asarray(eng.comoving_distance_mpc(z_test, float(h0), float(omega_m)), dtype=float)
    rel = float(np.max(np.abs(chi_b - chi_l) / np.maximum(np.abs(chi_l), 1.0)))
    return dict(ok=1.0 if rel < 0.05 else 0.0, max_rel_delta_chi=rel)


# ---------------------------------------------------------------------------
# Posterior scalar file (one row per MCMC sample)
# ---------------------------------------------------------------------------


def compute_posterior_sample_scalars_row(
    eng: cw.CosmoInterpEngine,
    row: pd.Series,
    sample_index: int,
) -> dict[str, Any]:
    """Mandatory **w(t)** & explicit **A** branches at :math:`z=0` for one chain row."""
    h0 = float(row["H0"])
    omega_m = float(row["Omega_m"])
    t_crit = float(row["t_crit"])
    k = float(row["k"])
    om_l = float(cw.olambda_flat(omega_m, h0))

    t0 = float(eng.blend_cosmic_age_gyr(np.array([0.0], dtype=float), h0, omega_m, t_crit, k)[0])
    w0 = float(w_sigmoid_logistic(np.array([t0]), k, t_crit)[0])

    lg0 = float(cw.c2_log10_A_rad_matter(np.array([t0]), h0, omega_m, cw.C2_OM_R_STD)[0])
    lt0 = float(cw.c2_log10_A_thermo(np.array([t0]), h0, om_l)[0])
    A_gr = float(10.0**lg0)
    A_th = float(10.0 ** np.clip(lt0, -300.0, 300.0))
    A_lin = float((1.0 - w0) * A_gr + w0 * A_th)

    log10_full = float(cw.c2_log10_A_blend(np.array([t0]), t_crit, k, h0, om_l, omega_m)[0])
    A_full = float(10.0**np.clip(log10_full, -300.0, 300.0))

    chain_id = int(row["chain_id"]) if "chain_id" in row and pd.notna(row["chain_id"]) else -1

    out = dict(
        sample_index=int(sample_index),
        chain_id=chain_id,
        H0=h0,
        Omega_M=omega_m,
        Omega_Lambda=om_l,
        k=k,
        t_crit_Gyr=t_crit,
        t_cosmic_age_Gyr_z0=t0,
        w_sigmoid_at_z0=w0,
        A_GR_branch_m2=A_gr,
        A_Thermo_branch_m2=A_th,
        A_blend_explicit_linear_m2=A_lin,
        A_blend_full_c2_log10_m2=A_full,
        A_GR_slide_proxy_m2=float(A_GR_slide_proxy_m2(np.array([t0]))[0]),
        entropy_amplitude=np.nan,
        transition_width=np.nan,
        saturation_scale=np.nan,
    )
    for col in ("ln_prob", "log_prob", "log_prior", "chi2"):
        if col in row.index:
            try:
                out[col] = float(row[col])
            except Exception:
                out[col] = float("nan")
    return out


def iter_posterior_rows(
    chain_path: Path,
    stride: int,
    max_rows: int | None,
) -> Iterable[tuple[int, pd.Series]]:
    df = normalize_blend_chain_columns(pd.read_csv(chain_path))
    n = len(df)
    step = max(int(stride), 1)
    idxs = list(range(0, n, step))
    if max_rows is not None:
        idxs = idxs[: int(max_rows)]
    for i in idxs:
        yield i, df.iloc[i]


def _flush_concat_chunks(
    dfs: list[pd.DataFrame],
    chunk_rows: int,
) -> tuple[list[pd.DataFrame], pd.DataFrame | None]:
    """Return (remaining_dfs, flushed_batch_or_None)."""
    if not dfs:
        return [], None
    cat = pd.concat(dfs, ignore_index=True)
    if len(cat) < chunk_rows:
        return dfs, None
    batch = cat.iloc[:chunk_rows]
    rest = cat.iloc[chunk_rows:]
    return ([rest] if len(rest) else []), batch


# ---------------------------------------------------------------------------
# Drivers
# ---------------------------------------------------------------------------


def _lcdm_chain_stub_summary(outdir: Path) -> dict[str, Any] | None:
    """Light-touch metadata only (does not load millions of rows)."""
    p = outdir / "mcmc_chain_lcdm.csv"
    if not p.is_file():
        return None
    try:
        df0 = pd.read_csv(p, nrows=2)
        return dict(path=str(p.resolve()), columns=list(df0.columns), note="lcdm_chain_present_header_only")
    except Exception as ex:
        return dict(path=str(p.resolve()), error=str(ex))


def run_chain_scalars_only(
    outdir: Path,
    chain_path: Path,
    *,
    chain_read_chunksize: int,
    max_rows: int | None,
    stride: int,
) -> dict[str, Any]:
    r"""Read **existing** ``mcmc_chain_blend.csv`` and write scalar rows only (**no MCMC**).

    Streams the chain in chunks so very large saved chains remain usable.
    """
    outdir.mkdir(parents=True, exist_ok=True)
    chain_path = Path(chain_path).expanduser().resolve()
    if not chain_path.is_file():
        chain_path, _ = discover_mcmc_chain_blend(None, outdir)
    if not cw.blend_physics_is_ccomplet2():
        raise RuntimeError("Set CWSF_BLEND_PHYSICS_MODE=ccomplet2.")

    zp_hi = float((_env_float("CWSF_SIGMOID_ZMAX", 5000.0) + 1.0) * 1.02)
    eng = cw.CosmoInterpEngine(zp_hi, max(float(cw.T_REF_GYR), 25.0))

    out_csv = outdir / "LambdaCDM_plus_S_posterior_sample_scalars.csv"
    summary: dict[str, Any] = dict(
        mode="chain",
        chain=str(chain_path.resolve()),
        reference_posterior_centers=REFERENCE_POSTERIOR_CENTERS,
        reference_prior_centers=REFERENCE_PRIOR_CENTERS,
        note="No MCMC executed; existing chain CSV read only.",
        lcdm_chain=_lcdm_chain_stub_summary(outdir),
        n_scalar_rows=0,
        n_skipped_validate=0,
        stride=int(stride),
        max_rows=max_rows,
        chain_read_chunksize=int(chain_read_chunksize),
    )

    header = True
    global_idx = 0
    step = max(int(stride), 1)
    first_chunk = True

    for chunk in pd.read_csv(chain_path, chunksize=int(chain_read_chunksize)):
        if first_chunk:
            rmap = blend_chain_column_rename_map(chunk.columns)
            if rmap:
                summary["blend_chain_column_rename"] = rmap
            first_chunk = False
        chunk = normalize_blend_chain_columns(chunk)
        rows_block: list[dict[str, Any]] = []
        for j in range(len(chunk)):
            if max_rows is not None and summary["n_scalar_rows"] >= int(max_rows):
                break
            if global_idx % step != 0:
                global_idx += 1
                continue
            row = chunk.iloc[j]
            try:
                h0 = float(row["H0"])
                om = float(row["Omega_m"])
                tc = float(row["t_crit"])
                k = float(row["k"])
            except Exception:
                global_idx += 1
                continue
            if not cw.validate_blend_background(h0, om, tc, k, eng, 50.0):
                summary["n_skipped_validate"] += 1
                global_idx += 1
                continue
            rows_block.append(compute_posterior_sample_scalars_row(eng, row, global_idx))
            summary["n_scalar_rows"] += 1
            global_idx += 1

        if rows_block:
            pd.DataFrame(rows_block).to_csv(out_csv, mode="a", index=False, header=header)
            header = False

        if max_rows is not None and summary["n_scalar_rows"] >= int(max_rows):
            break

    if summary["n_scalar_rows"] == 0:
        try:
            peek = normalize_blend_chain_columns(pd.read_csv(chain_path, nrows=2))
            req = ("H0", "Omega_m", "t_crit", "k")
            summary["blend_chain_columns_present"] = list(peek.columns)
            summary["blend_chain_missing_after_alias"] = [c for c in req if c not in peek.columns]
        except Exception as ex_peek:
            summary["blend_chain_peek_error"] = str(ex_peek)

    summary["output_csv"] = str(out_csv.resolve())

    gz_path = outdir / "LambdaCDM_plus_S_posterior_sample_scalars.csv.gz"
    try:
        with open(out_csv, "rb") as f_in:
            with gzip.open(gz_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
    except Exception as ex_gz:
        summary["gzip_error"] = str(ex_gz)

    summary["posterior_scalars_gz"] = str(gz_path.resolve())
    (outdir / "LambdaCDM_plus_S_chain_scalars_summary.json").write_text(
        json.dumps(summary, indent=2, default=str),
        encoding="utf-8",
    )
    return summary


def run_grid_scan(
    outdir: Path,
    h0: float,
    omega_m: float,
    *,
    k_min: float,
    k_max: float,
    n_k: int,
    tc_min: float,
    tc_max: float,
    n_tc: int,
    n_z: int,
    z_max: float,
    chunk_rows: int,
) -> dict[str, Any]:
    outdir.mkdir(parents=True, exist_ok=True)
    if not cw.blend_physics_is_ccomplet2():
        raise RuntimeError("Set CWSF_BLEND_PHYSICS_MODE=ccomplet2 for sigmoid_dataset.")

    eng = cw.CosmoInterpEngine(float((z_max + 1.0) * 1.02), max(float(cw.T_REF_GYR), 20.0))

    k_grid = np.linspace(float(k_min), float(k_max), int(n_k))
    tc_grid = np.linspace(float(tc_min), float(tc_max), int(n_tc))

    z_base = build_z_grid(n_z, z_max)
    out_csv = outdir / "LambdaCDM_plus_S_sigmoid_parameter_scan.csv"
    summary: dict[str, Any] = dict(
        mode="grid",
        started_unix=time.time(),
        n_k=int(n_k),
        n_tc=int(n_tc),
        n_z_base=int(z_base.size),
        k_range=[float(k_min), float(k_max)],
        t_crit_range=[float(tc_min), float(tc_max)],
        rows_written=0,
        chunks=0,
        validations=[],
    )

    diag_z = np.array([0.01, 0.1, 0.5, 1.0], dtype=float)
    summary["lcdm_recovery"] = lcdm_recovery_check(eng, h0, omega_m, diag_z)
    summary["A_z_mapping"] = az_mapping_note()

    header_written = False
    pair_id = 0
    acc: list[pd.DataFrame] = []

    for k in k_grid:
        for tc in tc_grid:
            if not cw.validate_blend_background(float(h0), float(omega_m), float(tc), float(k), eng, min(z_max, 200.0)):
                summary.setdefault("rejected_pairs", []).append([float(k), float(tc)])
                continue
            pair_id += 1
            zv = refine_z_near_t_crit(z_base, eng, h0, omega_m, tc, k)
            df = evolution_table(
                eng,
                h0,
                omega_m,
                tc,
                k,
                zv,
                pair_id=pair_id,
                scan_mode="grid",
            )
            acc.append(df)
            while True:
                acc, batch = _flush_concat_chunks(acc, chunk_rows)
                if batch is None:
                    break
                batch.to_csv(
                    out_csv,
                    mode="a",
                    index=False,
                    header=not header_written,
                )
                header_written = True
                summary["rows_written"] += len(batch)
                summary["chunks"] += 1
                summary["validations"].append(validate_chunk(batch))
            if pair_id % 50 == 0:
                print(f"[sigmoid_dataset] grid pair_id={pair_id} k={k:.4f} t_crit={tc:.3f} z_nodes={zv.size}")

    if acc:
        cat = pd.concat(acc, ignore_index=True)
        cat.to_csv(out_csv, mode="a", index=False, header=not header_written)
        summary["rows_written"] += len(cat)
        summary["chunks"] += 1
        summary["validations"].append(validate_chunk(cat))

    summary["output_csv"] = str(out_csv.resolve())
    summary["finished_unix"] = time.time()
    summary["reference_posterior_centers"] = REFERENCE_POSTERIOR_CENTERS
    summary["reference_prior_centers"] = REFERENCE_PRIOR_CENTERS
    (outdir / "LambdaCDM_plus_S_sigmoid_parameter_scan_summary.json").write_text(
        json.dumps(summary, indent=2, default=str),
        encoding="utf-8",
    )
    return summary


def run_posterior_scan(
    outdir: Path,
    chain_path: Path,
    *,
    stride: int,
    max_rows: int | None,
    n_z: int,
    z_max: float,
    chunk_rows: int,
    write_per_sample_csv: bool,
) -> dict[str, Any]:
    outdir.mkdir(parents=True, exist_ok=True)
    chain_path = Path(chain_path).expanduser().resolve()
    if not chain_path.is_file():
        chain_path, _ = discover_mcmc_chain_blend(None, outdir)
    if not cw.blend_physics_is_ccomplet2():
        raise RuntimeError("Set CWSF_BLEND_PHYSICS_MODE=ccomplet2.")

    eng = cw.CosmoInterpEngine(float((z_max + 1.0) * 1.02), max(float(cw.T_REF_GYR), 20.0))
    z_base = build_z_grid(n_z, z_max)

    out_long = outdir / "LambdaCDM_plus_S_sigmoid_parameter_scan.csv"
    out_scalars = outdir / "LambdaCDM_plus_S_posterior_sample_scalars.csv"

    summary: dict[str, Any] = dict(
        mode="posterior-full",
        chain=str(chain_path.resolve()),
        stride=int(stride),
        max_rows=max_rows,
        rows_long=0,
        n_scalar_rows=0,
        A_z_mapping=az_mapping_note(),
        reference_posterior_centers=REFERENCE_POSTERIOR_CENTERS,
        reference_prior_centers=REFERENCE_PRIOR_CENTERS,
        note="posterior-full builds large z×sample tables; use mode=chain for scalars only.",
    )

    header_long = False
    pair_id = 0
    buf: list[pd.DataFrame] = []
    scalar_rows: list[dict[str, Any]] = []

    for idx, row in iter_posterior_rows(chain_path, stride, max_rows):
        h0 = float(row["H0"])
        om = float(row["Omega_m"])
        tc = float(row["t_crit"])
        k = float(row["k"])
        if not cw.validate_blend_background(h0, om, tc, k, eng, min(z_max, 200.0)):
            continue
        pair_id += 1
        zv = refine_z_near_t_crit(z_base, eng, h0, om, tc, k)

        log_p = None
        for col in ("ln_prob", "log_prob"):
            if col in row.index:
                try:
                    log_p = float(row[col])
                except Exception:
                    pass

        df = evolution_table(
            eng,
            h0,
            om,
            tc,
            k,
            zv,
            pair_id=pair_id,
            scan_mode="posterior",
            posterior_weight=1.0,
            log_likelihood=log_p,
            chi2_train=None,
        )
        buf.append(df)

        scalar_rows.append(compute_posterior_sample_scalars_row(eng, row, idx))
        summary["n_scalar_rows"] += 1

        while True:
            buf, batch = _flush_concat_chunks(buf, chunk_rows)
            if batch is None:
                break
            batch.to_csv(out_long, mode="a", index=False, header=not header_long)
            header_long = True
            summary["rows_long"] += len(batch)

        if write_per_sample_csv and pair_id <= _env_int("CWSF_SIGMOID_MAX_SIDE_CSV", 200):
            df.to_csv(outdir / f"evolution_posterior_draw_{pair_id:06d}.csv", index=False)

    if buf:
        cat = pd.concat(buf, ignore_index=True)
        cat.to_csv(out_long, mode="a", index=False, header=not header_long)
        summary["rows_long"] += len(cat)

    pd.DataFrame(scalar_rows).to_csv(out_scalars, index=False)
    summary["posterior_scalars_csv"] = str(out_scalars.resolve())
    summary["output_long_csv"] = str(out_long.resolve())

    gz_ps = outdir / "LambdaCDM_plus_S_posterior_sample_scalars.csv.gz"
    try:
        with open(out_scalars, "rb") as f_in:
            with gzip.open(gz_ps, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
        summary["posterior_scalars_gz"] = str(gz_ps.resolve())
    except Exception as exz:
        summary["posterior_scalars_gz_error"] = str(exz)

    (outdir / "LambdaCDM_plus_S_sigmoid_posterior_summary.json").write_text(
        json.dumps(summary, indent=2, default=str),
        encoding="utf-8",
    )
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(
        description="ΛCDM+S sigmoid dataset generator (CSV only; does not run MCMC). Uses cwsf_pipeline numerics."
    )
    p.add_argument(
        "--mode",
        choices=("chain", "grid", "posterior-full"),
        default=os.getenv("CWSF_SIGMOID_MODE", "chain"),
        help="chain=read existing mcmc_chain_blend.csv → scalars only (default). "
        "grid=k×t_crit scan. posterior-full=heavy per-draw z evolution.",
    )
    p.add_argument("--outdir", type=str, default=os.getenv("CWSF_OUTDIR", "./cwsf_output"))
    p.add_argument(
        "--chain",
        type=str,
        default="",
        help="Path to mcmc_chain_blend.csv (default: <outdir>/mcmc_chain_blend.csv).",
    )
    p.add_argument("--h0", type=float, default=_env_float("CWSF_SIGMOID_GRID_H0", REFERENCE_POSTERIOR_CENTERS["H0_km_s_Mpc"]))
    p.add_argument("--omega-m", type=float, default=_env_float("CWSF_SIGMOID_GRID_OMEGA_M", REFERENCE_POSTERIOR_CENTERS["Omega_m"]))
    args = p.parse_args()

    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    # Posterior-centered grid defaults (bracket late-time t_crit ~15 Gyr, k ~0.36 Gyr^-1)
    k_min = _env_float("CWSF_SIGMOID_K_MIN", 0.18)
    k_max = _env_float("CWSF_SIGMOID_K_MAX", 0.58)
    n_k = _env_int("CWSF_SIGMOID_N_K", 35)
    tc_min = _env_float("CWSF_SIGMOID_TCRIT_MIN", 13.8)
    tc_max = _env_float("CWSF_SIGMOID_TCRIT_MAX", 16.4)
    n_tc = _env_int("CWSF_SIGMOID_N_T", 35)
    n_z = _env_int("CWSF_SIGMOID_N_Z", 1400)
    z_max = _env_float("CWSF_SIGMOID_ZMAX", 5000.0)
    chunk = _env_int("CWSF_SIGMOID_CHUNK_ROWS", 75_000)
    stride = _env_int("CWSF_SIGMOID_POSTERIOR_STRIDE", 1)
    post_max = _env_int("CWSF_SIGMOID_POSTERIOR_MAX_ROWS", 0) or None
    chain_read = _env_int("CWSF_SIGMOID_CHAIN_READ_CHUNKS", 200_000)

    chain_discovered: Path | None = None
    if args.mode in ("chain", "posterior-full"):
        chain_discovered, _ = discover_mcmc_chain_blend(args.chain or None, outdir)

    chain = chain_discovered if chain_discovered is not None else (
        Path(args.chain).expanduser().resolve() if args.chain else (outdir / "mcmc_chain_blend.csv")
    )

    if args.mode == "chain":
        s = run_chain_scalars_only(
            outdir,
            chain,
            chain_read_chunksize=chain_read,
            max_rows=post_max,
            stride=stride,
        )
        print(json.dumps(s, indent=2, default=str))
        return

    if args.mode == "grid":
        s = run_grid_scan(
            outdir,
            float(args.h0),
            float(args.omega_m),
            k_min=k_min,
            k_max=k_max,
            n_k=n_k,
            tc_min=tc_min,
            tc_max=tc_max,
            n_tc=n_tc,
            n_z=n_z,
            z_max=z_max,
            chunk_rows=chunk,
        )
        print(json.dumps(s, indent=2, default=str))
        return

    s = run_posterior_scan(
        outdir,
        chain,
        stride=stride,
        max_rows=post_max,
        n_z=n_z,
        z_max=z_max,
        chunk_rows=chunk,
        write_per_sample_csv=False,
    )
    print(json.dumps(s, indent=2, default=str))


if __name__ == "__main__":
    main()
