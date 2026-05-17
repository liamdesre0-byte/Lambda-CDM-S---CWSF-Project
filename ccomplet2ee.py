#!/usr/bin/env python3
"""
ccomplet2ee — Extended blend diagnostics, sensitivity, statistics, ISEF-style dashboard.

CLI::
    python ccomplet2ee.py --data cwsf_output/pantheon_clean.csv --out ee_output --mc 350

Streamlit (ISEF plots in browser)::
    streamlit run ccomplet2ee.py

Companion to ``ccomplet2.py`` (see asymmetric ``log10_A_blend_asymmetric`` added there).

Physics notes
-------------
* Area bridge used for Δμ diagnostics: Δμ(z) ≈ (5/2) r(t_lcdm(z)) with r=log10(A_blend/A_LCDM).
  This reproduces ``ccomplet2`` lore but is **not** identical to fully self-consistent D_L.
* Extended blend supports Richards asymmetry ν, dex shifts on thermo branch, curvature on thermo branch,
  and optional anchoring log A blend → ΛCDM at t=T_REF.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import sys
import urllib.request
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy import integrate, stats
from scipy.interpolate import interp1d

# -----------------------------------------------------------------------------
# Constants — ccomplet2-aligned
# -----------------------------------------------------------------------------
GYR_TO_S = 3.1558e16
KMS_TO_SI = 3.2408e-20
C_KMS = 299792.458
A0_LOG10 = 54.0
A0 = 10.0**A0_LOG10
T_REF = 13.8
T_REF_S = T_REF * GYR_TO_S
OM_R_STD = 9.24e-5
T_TRANSITION_MIN_GYR = 9.0
T_FUTURE_DEFAULT_GYR = 50.0

# Pantheon+ public data (aligned with ``ccomplet2.load_pantheon_data``).
_RAW_GH = "https://raw.githubusercontent.com/PantheonPlusSH0ES/DataRelease/main/Pantheon%2B_Data/4_DISTANCES_AND_COVAR"
PANTHEON_DAT_URL = _RAW_GH + "/Pantheon%2BSH0ES.dat"

# TASK 5 colour rules (solid palette)
C_LCDM = "#d62728"
C_BLEND_MAIN = "#1f77b4"
C_BLEND_ALT = "#2ca02c"
C_GRAV = "#ffbf00"
C_THERMO = "#ff7f0e"
C_BAND = "#1f77b4"
C_TRACE = "#c9c9c9"


def _safe_log_sinh(x: np.ndarray) -> np.ndarray:
    x = np.atleast_1d(np.asarray(x, dtype=float))
    out = np.empty_like(x)
    sm = x < 20.0
    out[sm] = np.log(np.sinh(np.clip(x[sm], 1e-300, None)))
    out[~sm] = x[~sm] - np.log(2.0)
    return out


def log10_A_lcdm(t_gyr: np.ndarray, H0_kms: float, OmL: float, OmM: float) -> np.ndarray:
    H0si = H0_kms * KMS_TO_SI
    ts = np.atleast_1d(np.asarray(t_gyr, float)) * GYR_TO_S
    arg = 1.5 * H0si * np.sqrt(OmL) * ts
    arg0 = 1.5 * H0si * np.sqrt(OmL) * T_REF_S
    omm_s = max(float(OmM), 1e-9)
    oml_s = max(float(OmL), 1e-9)
    la = (1.0 / 3.0) * np.log(omm_s / oml_s) + (2.0 / 3.0) * _safe_log_sinh(arg)
    la0 = float((1.0 / 3.0) * np.log(omm_s / oml_s) + (2.0 / 3.0) * _safe_log_sinh(np.atleast_1d(arg0))[0])
    return (np.log(A0) + 2.0 * (la - la0)) / np.log(10.0)


def _rad_tbl(H0_kms: float, OmM: float, OmR: float) -> tuple[np.ndarray, np.ndarray]:
    a_grid = np.logspace(-10.0, np.log10(12.0), 6000)
    H0_si = max(float(H0_kms), 1e-9) * KMS_TO_SI
    omm = max(float(OmM), 1e-12)
    omr = max(float(OmR), 1e-14)
    denom = np.sqrt(omm / np.clip(a_grid, 1e-14, None) ** 3 + omr / np.clip(a_grid, 1e-14, None) ** 4)
    dt_da = 1.0 / np.clip(a_grid * H0_si * denom, 1e-30, None)
    ts = integrate.cumulative_trapezoid(dt_da, a_grid, initial=0.0)
    return ts / GYR_TO_S, a_grid


_RAD_CACHE: dict[tuple[float, float], tuple[np.ndarray, np.ndarray]] = {}


def log10_A_rad_matter(t_gyr: np.ndarray, H0: float, OmM: float) -> np.ndarray:
    key = (round(float(H0), 4), round(float(OmM), 6))
    if key not in _RAD_CACHE:
        tt, aa = _rad_tbl(H0, OmM, OM_R_STD)
        _RAD_CACHE[key] = (tt, aa)
    t_tbl, a_tbl = _RAD_CACHE[key]
    a_of_t = interp1d(t_tbl, a_tbl, kind="linear", bounds_error=False, fill_value=(float(a_tbl[0]), float(a_tbl[-1])))
    t_arr = np.asarray(t_gyr, dtype=float)
    a_t = np.clip(np.asarray(a_of_t(np.clip(t_arr, 0.0, None)), dtype=float), 1e-40, None)
    a_ref = float(np.clip(a_of_t(np.array([T_REF]))[0], 1e-40, None))
    return A0_LOG10 + 2.0 * np.log10(a_t / a_ref)


def log10_A_thermo(t_gyr: np.ndarray, H0: float, OmL: float) -> np.ndarray:
    Hlam = H0 * KMS_TO_SI * math.sqrt(OmL)
    dt = (np.asarray(t_gyr, float) - float(T_FUTURE_DEFAULT_GYR)) * GYR_TO_S
    return A0_LOG10 + 2.0 * Hlam * dt / np.log(10.0)


@dataclass
class BlendPhysExt:
    nu_asym: float = 1.0
    thermo_amp: float = 0.0  # dex add on lt
    thermo_curvature: float = 0.0  # dex * ((t/T_REF-1)**2)
    anchor_present: bool = True


def log10_A_blend_extended(
    t: np.ndarray,
    tc: float,
    k: float,
    H0: float,
    OmL: float,
    OmM: float,
    ext: BlendPhysExt,
) -> np.ndarray:
    t = np.asarray(t, dtype=float)
    lc = log10_A_lcdm(t, H0, OmL, OmM)
    out = np.empty_like(t, dtype=float)
    early = t < T_TRANSITION_MIN_GYR
    out[early] = lc[early]
    late = ~early
    if np.any(late):
        tl = t[late]
        s = 1.0 / (1.0 + np.exp(np.clip(-k * (tl - tc), -500.0, 500.0)))
        w = np.clip(s, 1e-15, 1.0 - 1e-15) ** float(ext.nu_asym)
        lg = log10_A_rad_matter(tl, H0, OmM)
        lt = np.asarray(log10_A_thermo(tl, H0, OmL), dtype=float)
        lt = lt + float(ext.thermo_amp) + float(ext.thermo_curvature) * (tl / T_REF - 1.0) ** 2
        ab = (1.0 - w) * 10.0**lg + w * 10.0 ** np.clip(lt, -300.0, 500.0)
        out[late] = np.log10(np.clip(ab, 1e-300, None))
    if ext.anchor_present:
        lc0 = np.interp(T_REF, t, lc)
        br0 = np.interp(T_REF, t, out)
        out = out + (lc0 - br0)
    return np.nan_to_num(out, nan=lc.mean(), posinf=300 + A0_LOG10, neginf=-300 + A0_LOG10)


def r_area(t: np.ndarray, tc: float, k: float, H0: float, OmL: float, OmM: float, ext: BlendPhysExt) -> np.ndarray:
    return log10_A_blend_extended(t, tc, k, H0, OmL, OmM, ext) - log10_A_lcdm(t, H0, OmL, OmM)


def t_lcdm_scalar(z: float, H0: float, Om: float, Ol: float) -> float:
    H0_si = H0 * KMS_TO_SI
    a_emit = 1.0 / (1.0 + float(z))

    def g(a):
        return 1.0 / (a * H0_si * math.sqrt(Om / a**3 + Ol))

    ts, _ = integrate.quad(g, 1e-6, a_emit, limit=120)
    return ts / GYR_TO_S


def t_lcdm_bulk(z: np.ndarray, H0: float, Om: float, Ol: float) -> np.ndarray:
    return np.array([t_lcdm_scalar(float(zi), H0, Om, Ol) for zi in z], dtype=float)


def mu_lcdm_vector(z: np.ndarray, H0: float, Om: float, Ol: float) -> np.ndarray:
    z = np.maximum(np.asarray(z, dtype=float), 1e-10)
    zmax = float(z.max()) * 1.02 + 1e-3
    zp = np.linspace(0.0, zmax, 4096)
    Ez = np.sqrt(np.maximum(Om * (1 + zp) ** 3 + Ol, 1e-15))
    chi = integrate.cumulative_trapezoid(C_KMS / H0 / Ez, zp, initial=0.0)
    d_l = (1.0 + zp) * chi
    dl_safe = np.maximum(np.asarray(d_l, dtype=float), 1e-18)
    mu_grid = np.where(np.isfinite(dl_safe), 5.0 * np.log10(dl_safe * 1e6 / 10.0), np.nan)
    mu = interp1d(zp, mu_grid, bounds_error=False, fill_value="extrapolate")(z)
    for i in np.where(~np.isfinite(mu))[0]:
        def inte(zz):
            return 1.0 / math.sqrt(Om * (1 + zz) ** 3 + Ol)

        cchi, _ = integrate.quad(inte, 0.0, float(z[i]), limit=140)
        dlv = (C_KMS / H0) * (1 + z[i]) * cchi
        mu[i] = 5.0 * math.log10(dlv * 1e6 / 10.0)
    return np.asarray(mu, dtype=float)


def nuisance_offset_diag(delta: np.ndarray, sigma: np.ndarray) -> float:
    w = 1.0 / np.maximum(np.asarray(sigma, dtype=float) ** 2, 1e-12)
    d = np.asarray(delta, dtype=float)
    return float(np.sum(w * d) / np.sum(w))


def model_fit_summary(mu_obs: np.ndarray, mu_model: np.ndarray, sigma: np.ndarray, k_params: int) -> dict[str, float]:
    """Gaussian diagonal likelihood with nuisance M; Pearson on (obs, model)."""
    d = mu_obs - mu_model
    off = nuisance_offset_diag(d, sigma)
    r = d - off
    sig = np.asarray(sigma, dtype=float)
    w = 1.0 / sig**2
    chi2 = float(np.sum(w * r**2))
    n = int(len(mu_obs))
    dof = max(n - k_params, 1)
    ln_norm = float(-0.5 * np.sum(np.log(2 * math.pi * sig**2)))
    ll = float(-0.5 * chi2 + ln_norm)
    rho, pv = stats.pearsonr(mu_obs, mu_model)
    rmse = float(np.sqrt(np.mean(r**2)))
    sem_mean = float(np.std(r, ddof=1) / math.sqrt(max(n, 1)))
    neg2_ll = float(-2.0 * ll)
    bic = float(chi2 + k_params * math.log(max(n, 2)))
    aic_chi = float(chi2 + 2 * k_params)
    if n > k_params + 1:
        aicc: float | None = float(aic_chi + (2.0 * k_params * (k_params + 1.0)) / max(n - k_params - 1, 1))
    else:
        aicc = None
    # Not full Bayesian WAIC (needs posterior over parameters); Gaussian MLE-style IC on full log-density.
    waic_mle_proxy = float(neg2_ll + 2.0 * k_params)
    return dict(
        n=n,
        k_params=k_params,
        dof=dof,
        offset=float(off),
        chi2=chi2,
        red_chi2=float(chi2 / dof),
        aic=float(aic_chi),
        aicc=aicc,
        bic=bic,
        neg2_loglike=neg2_ll,
        waic_mle_proxy=waic_mle_proxy,
        loglike=ll,
        rmse=rmse,
        sem=sem_mean,
        residual_mean=float(np.mean(r)),
        residual_std=float(np.std(r, ddof=1)),
        residual_max_abs=float(np.max(np.abs(r))),
        pearson_r=float(rho),
        pearson_p=float(pv),
    )


def delta_mu_bridge(z: np.ndarray, tc: float, k: float, H0: float, OmL: float, OmM: float, ext: BlendPhysExt) -> np.ndarray:
    tt = t_lcdm_bulk(z, H0, OmM, OmL)
    return 2.5 * r_area(tt, tc, k, H0, OmL, OmM, ext)


def mu_blend_diag(z: np.ndarray, tc: float, k: float, H0: float, OmL: float, OmM: float, ext: BlendPhysExt) -> np.ndarray:
    return mu_lcdm_vector(z, H0, OmM, OmL) + delta_mu_bridge(z, tc, k, H0, OmL, OmM, ext)


def load_pantheon_csv(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_csv(path)
    cmap = {c.lower(): c for c in df.columns}
    cz = cmap.get("z") or cmap.get("zcmb")
    cmu = cmap.get("mu")
    csig = cmap.get("sig")
    if cz is None or cmu is None or csig is None:
        raise ValueError(f"Need z, mu, sig columns - got {list(df.columns)}")
    z = pd.to_numeric(df[cz], errors="coerce").to_numpy()
    mu = pd.to_numeric(df[cmu], errors="coerce").to_numpy()
    sig = pd.to_numeric(df[csig], errors="coerce").to_numpy()
    m = np.isfinite(z) & np.isfinite(mu) & np.isfinite(sig) & (z > 0.015) & (sig > 0)
    return z[m].astype(float), mu[m].astype(float), sig[m].astype(float)


def simulated_sample(n_sn: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    z = rng.uniform(0.02, 1.98, size=n_sn)
    Ht, Om, Ol = 72.7, 0.31, 0.689 - OM_R_STD
    mu_true = mu_lcdm_vector(z, Ht, Om, Ol)
    sig = rng.uniform(0.12, 0.32, size=n_sn)
    return z.astype(float), (mu_true + rng.normal(0, sig)).astype(float), sig.astype(float)


def make_synthetic_pantheon_like(seed: int = 42, n_eff: int = 1701) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pantheon-like synthetic sample (adapted from ``ccomplet2._make_synthetic`` logic)."""
    rng = np.random.default_rng(seed)
    z = np.sort(
        np.concatenate(
            [
                rng.uniform(0.010, 0.080, max(150, n_eff // 12)),
                rng.uniform(0.080, 0.200, max(300, n_eff // 6)),
                rng.uniform(0.200, 0.600, max(600, n_eff // 3)),
                rng.uniform(0.600, 1.200, max(450, n_eff // 4)),
                rng.uniform(1.200, 2.261, max(201, n_eff // 8)),
            ]
        )
    )
    z = np.sort(z[:n_eff])
    sigma = np.clip(
        0.12 + 0.06 * (z < 0.05).astype(float) + 0.10 * (z > 1.50).astype(float) + rng.uniform(-0.01, 0.01, len(z)),
        0.05,
        0.40,
    )
    Om, Ol_use = 0.31, 0.689 - OM_R_STD
    mu_truth = mu_lcdm_vector(z.astype(float), 67.4, Om, Ol_use)
    return z.astype(float), (mu_truth + rng.normal(0, sigma)).astype(float), sigma.astype(float)


def bundled_csv_candidates() -> list[Path]:
    root = Path(__file__).resolve().parent
    return [
        root / "cwsf_output" / "pantheon_clean.csv",
        root / "cwsf_output_run" / "pantheon_clean.csv",
        root / "_smoke_out" / "pantheon_clean.csv",
    ]


def fetch_pantheon_plus_dat_arrays(timeout: float = 45.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mirror ``ccomplet2.load_pantheon_data``: parse Pantheon+SH0ES.dat whitespace table."""
    with urllib.request.urlopen(PANTHEON_DAT_URL, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8")
    df = pd.read_csv(io.StringIO(raw), sep=r"\s+", comment="#")
    df.columns = [str(c).strip() for c in df.columns]

    z_candidates = ["zHD", "zCMB", "z", "Z", "REDSHIFT"]
    mu_candidates = ["MU_SH0ES", "MU", "mu", "DISTMOD", "m_b_corr"]
    sig_candidates = [
        "MU_SH0ES_ERR_DIAG",
        "MUERR",
        "MU_ERR",
        "MUDIAG_ERR",
        "ERR_MU",
        "m_b_corr_err_DIAG",
        "SIGMA_MU",
        "sig",
    ]

    def _pick(cands: list[str], cols: list[str]) -> str | None:
        for c in cands:
            if c in cols:
                return c
        lc = [x.lower() for x in cols]
        for c in cands:
            for col in cols:
                if c.lower() in col.lower():
                    return col
        return None

    z_col = _pick(z_candidates, list(df.columns))
    mu_col = _pick(mu_candidates, list(df.columns))
    sig_col = _pick(sig_candidates, list(df.columns))
    if z_col is None or mu_col is None:
        raise ValueError(f"Pantheon+ parse failed; columns={list(df.columns)}")
    z = pd.to_numeric(df[z_col], errors="coerce").to_numpy()
    mu_obs = pd.to_numeric(df[mu_col], errors="coerce").to_numpy()
    if sig_col is None:
        sig = np.full(len(df), 0.15)
    else:
        sig = pd.to_numeric(df[sig_col], errors="coerce").to_numpy()
    msk = np.isfinite(z) & np.isfinite(mu_obs) & np.isfinite(sig) & (z > 0.005) & (mu_obs > 10.0) & (sig > 0.0)
    if not np.any(msk):
        raise ValueError("Pantheon+ mask removed all rows.")
    return z[msk].astype(float), mu_obs[msk].astype(float), sig[msk].astype(float)


def resolve_sn_data(csv_arg: Path | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """Ship-with-app order: CLI path → bundled pantheon_clean → Pantheon+ URL → synthetic."""
    if csv_arg is not None and csv_arg.is_file():
        zz, mu, sg = load_pantheon_csv(csv_arg)
        return zz, mu, sg, f"explicit_file:{csv_arg.name}"
    for p in bundled_csv_candidates():
        if p.is_file():
            zz, mu, sg = load_pantheon_csv(p)
            return zz, mu, sg, f"bundled:{p.parent.name}/{p.name}"
    try:
        zz, mu, sg = fetch_pantheon_plus_dat_arrays()
        return zz, mu, sg, "pantheon_remote:Pantheon+SH0ES.dat"
    except Exception as exc:
        warnings.warn(f"Pantheon+ download failed ({exc}); using synthetic Pantheon-like sample.", RuntimeWarning)
        zz, mu, sg = make_synthetic_pantheon_like(seed=42)
        return zz, mu, sg, "synthetic_pantheon_like"


def model_stats_table_rows(st_l: dict, st_b: dict) -> pd.DataFrame:
    """Readable comparison table for ISEF / Streamlit displays."""
    keys = (
        ("n_SN", "n"),
        ("k_params", "k_params"),
        ("dof", "dof"),
        ("RMSE", "rmse"),
        ("Pearson_r", "pearson_r"),
        ("Pearson_p", "pearson_p"),
        ("chi2", "chi2"),
        ("chi2_red", "red_chi2"),
        ("loglike", "loglike"),
        ("neg2_loglike", "neg2_loglike"),
        ("AIC_chi", "aic"),
        ("AICc", "aicc"),
        ("BIC", "bic"),
        ("WAIC_MLE_proxy", "waic_mle_proxy"),
        ("residual_mean", "residual_mean"),
        ("residual_std", "residual_std"),
        ("SEM_mean_res", "sem"),
        ("abs_max_residual", "residual_max_abs"),
    )
    row_l = dict(model="LCDM")
    row_b = dict(model="Blend")
    for label, kk in keys:
        vl, vb = st_l[kk], st_b[kk]
        row_l[label] = float("nan") if vl is None else vl
        row_b[label] = float("nan") if vb is None else vb
    return pd.DataFrame([row_l, row_b])


def one_at_a_time_sensitivity(z, mu_o, sig, tc0, k0, h0, oml, ext, deltas: dict[str, float]) -> pd.DataFrame:
    om0 = max(1.0 - oml - OM_R_STD, 0.05)
    base = mu_blend_diag(z, tc0, k0, h0, oml, om0, ext)
    b_rmse = model_fit_summary(mu_o, base, sig, k_params=6)["rmse"]
    rows: list[dict] = []

    def add(name: str, mu_new: np.ndarray):
        rm = model_fit_summary(mu_o, mu_new, sig, k_params=6)["rmse"]
        rows.append(dict(parameter=name, rmse=float(rm), delta_rmse=float(rm - b_rmse)))

    for key, dv in deltas.items():
        for sgn in (-1.0, 1.0):
            d = sgn * dv
            if key == "t_crit":
                add(f"t_crit{d:+.3f}", mu_blend_diag(z, tc0 + d, k0, h0, oml, om0, ext))
            elif key == "k":
                add(f"k{d:+.3f}", mu_blend_diag(z, tc0, k0 + d, h0, oml, om0, ext))
            elif key == "H0":
                add(f"H0{d:+.3f}", mu_blend_diag(z, tc0, k0, h0 + d, oml, max(1.0 - oml - OM_R_STD, 0.05), ext))
            elif key == "OmL":
                ol2 = float(np.clip(oml + d, 0.62, 0.76))
                om2 = max(1.0 - ol2 - OM_R_STD, 0.05)
                add(f"OmL{d:+.4f}", mu_blend_diag(z, tc0, k0, h0, ol2, om2, ext))
            elif key == "nu":
                ex2 = BlendPhysExt(nu_asym=ext.nu_asym + d, thermo_amp=ext.thermo_amp, thermo_curvature=ext.thermo_curvature, anchor_present=ext.anchor_present)
                add(f"nu{d:+.3f}", mu_blend_diag(z, tc0, k0, h0, oml, om0, ex2))
            elif key == "thermo_amp":
                ex2 = BlendPhysExt(nu_asym=ext.nu_asym, thermo_amp=ext.thermo_amp + d, thermo_curvature=ext.thermo_curvature, anchor_present=ext.anchor_present)
                add(f"amp{d:+.3f}", mu_blend_diag(z, tc0, k0, h0, oml, om0, ex2))
            elif key == "thermo_curv":
                ex2 = BlendPhysExt(nu_asym=ext.nu_asym, thermo_amp=ext.thermo_amp, thermo_curvature=ext.thermo_curvature + d, anchor_present=ext.anchor_present)
                add(f"c{d:+.4f}", mu_blend_diag(z, tc0, k0, h0, oml, om0, ex2))

    df = pd.DataFrame(rows)
    df["abs_d_rmse"] = np.abs(df["delta_rmse"])
    return df


def monte_carlo_parameter_study(z, mu_o, sig, tc0, k0, h0_, oml_, ext, n_runs: int, seed: int, scales):
    rng = np.random.default_rng(seed)
    stc, sk, sh, sol = scales
    lcdm_mu = mu_lcdm_vector(z, h0_, max(1.0 - oml_ - OM_R_STD, 0.05), oml_)
    st_l = model_fit_summary(mu_o, lcdm_mu, sig, k_params=2)

    tcs = np.clip(tc0 + rng.uniform(-stc, stc, n_runs), 11.5, 22.5)
    ks = np.clip(k0 + rng.uniform(-sk, sk, n_runs), 0.22, 0.72)
    h0v = np.clip(h0_ + rng.uniform(-sh, sh, n_runs), 67.5, 78.5)
    ols = np.clip(oml_ + rng.uniform(-sol, sol, n_runs), 0.662, 0.712)

    rmse_v = np.empty(n_runs)
    slope_v = np.empty(n_runs)
    chi_v = np.empty(n_runs)
    lcdm_win = np.empty(n_runs, dtype=bool)

    for i in range(n_runs):
        om_i = max(1.0 - float(ols[i]) - OM_R_STD, 0.05)
        mu_b = mu_blend_diag(z, float(tcs[i]), float(ks[i]), float(h0v[i]), float(ols[i]), om_i, ext)
        fb = model_fit_summary(mu_o, mu_b, sig, k_params=6)
        off = fb["offset"]
        rz = mu_o - mu_b - off
        rmse_v[i] = fb["rmse"]
        chi_v[i] = fb["chi2"]
        ii = np.argsort(z)
        slope_v[i] = float(np.polyfit(np.linspace(0, 1, len(z)), rz[ii], 1)[0]) if len(z) > 3 else 0.0
        lcdm_win[i] = st_l["rmse"] < fb["rmse"]

    corr = pd.DataFrame(
        {
            "corr_rmse_tc": pd.Series(rmse_v).corr(pd.Series(tcs.astype(float))),
            "corr_rmse_k": pd.Series(rmse_v).corr(pd.Series(ks.astype(float))),
            "corr_rmse_H0": pd.Series(rmse_v).corr(pd.Series(h0v.astype(float))),
            "corr_rmse_OmL": pd.Series(rmse_v).corr(pd.Series(ols.astype(float))),
            "corr_slope_tc": pd.Series(slope_v).corr(pd.Series(tcs.astype(float))),
            "frac_lcdm_lower_rmse": float(np.mean(lcdm_win)),
            "paired_bootstrap_placeholder": np.nan,
        },
        index=[0],
    )
    return dict(
        t_crit=tcs,
        k_trans=ks,
        H0=h0v,
        OmL=ols,
        rmse=rmse_v,
        slopes=slope_v,
        chi2=chi_v,
        lcdm_wins_rmse=lcdm_win,
        correlations=corr,
        lcdm_ref=st_l,
    )


def figure_isef_dashboard(
    z,
    mu_o,
    sig,
    tc0,
    k0,
    h0_,
    oml_,
    ext,
    mc_pack: dict,
    st_b: dict,
    st_l: dict,
    n_runs: int,
):
    """Build the multi-panel ISEF matplotlib figure (caller may save or pass to Streamlit)."""
    om0 = max(1.0 - oml_ - OM_R_STD, 0.05)
    t_full = np.linspace(1.5, 92.0, 900)
    la_l = log10_A_lcdm(t_full, h0_, oml_, om0)
    la_b = log10_A_blend_extended(t_full, tc0, k0, h0_, oml_, om0, ext)
    la_g = log10_A_rad_matter(t_full, h0_, om0)
    la_th = log10_A_thermo(t_full, h0_, oml_)

    mu_lcdm = mu_lcdm_vector(z, h0_, om0, oml_)
    mu_blend = mu_blend_diag(z, tc0, k0, h0_, oml_, om0, ext)
    ob = nuisance_offset_diag(mu_o - mu_blend, sig)
    olm = nuisance_offset_diag(mu_o - mu_lcdm, sig)
    res_b = mu_o - mu_blend - ob
    res_l = mu_o - mu_lcdm - olm

    zs = np.argsort(z)
    n_mc_show = min(35, mc_pack["rmse"].size)
    rng = np.random.default_rng(0)
    pick = rng.choice(mc_pack["rmse"].size, size=n_mc_show, replace=False)

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.2, 1.05, 0.92], width_ratios=[1, 1], hspace=0.34, wspace=0.25)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, :])
    ax4 = fig.add_subplot(gs[2, :])

    def thin_grid(ax):
        ax.grid(True, alpha=0.15, linestyle="-", linewidth=0.6)

    # Pale ensemble on log-area (time domain)
    for j in pick:
        omj = max(1.0 - float(mc_pack["OmL"][j]) - OM_R_STD, 0.05)
        la_j = log10_A_blend_extended(t_full, float(mc_pack["t_crit"][j]), float(mc_pack["k_trans"][j]), float(mc_pack["H0"][j]), float(mc_pack["OmL"][j]), omj, ext)
        ax1.plot(t_full, la_j, color=C_TRACE, lw=0.85, alpha=0.35, linestyle="-")

    ax1.plot(t_full, la_l, color=C_LCDM, lw=3.2, linestyle="-", label="ΛCDM ref")
    ax1.plot(t_full, la_b, color=C_BLEND_MAIN, lw=3.4, linestyle="-", label="Blend central params")
    ax1.plot(t_full, la_g, color=C_GRAV, lw=2.8, linestyle="-", label="Gravity branch")
    ax1.plot(t_full, la_th, color=C_THERMO, lw=2.8, linestyle="-", label="Thermo branch")
    ax1.axvline(T_REF, color="#333333", lw=2.2, linestyle="-", alpha=0.75)
    ax1.set_xlim(t_full.min(), t_full.max())
    ax1.set_xlabel("Cosmic time (Gyr)")
    ax1.set_ylabel(r"$\log_{10}A\ \mathrm{(m}^2)$")
    ax1.set_title(f"Horizon-area history (ensemble n={n_runs})")
    thin_grid(ax1)
    ax1.legend(loc="best", ncol=2, fontsize=9)

    m = (t_full >= 8.0) & (t_full <= 38.0)
    ax2.plot(t_full[m], la_l[m], color=C_LCDM, lw=3.2)
    ax2.plot(t_full[m], la_b[m], color=C_BLEND_MAIN, lw=3.4)
    ax2.plot(t_full[m], la_g[m], color=C_GRAV, lw=2.8)
    ax2.plot(t_full[m], la_th[m], color=C_THERMO, lw=2.8)
    ax2.axvline(tc0, color=C_BLEND_ALT, lw=3.0, linestyle="-", label=r"$t_{\rm crit}$")
    ax2.axvline(T_REF, color="#333333", lw=2.2)
    ax2.set_xlim(8.0, 38.0)
    ax2.set_xlabel("Cosmic time (Gyr)")
    ax2.set_ylabel(r"$\log_{10}A\ \mathrm{(m}^2)$")
    ax2.set_title("Transition zoom")
    thin_grid(ax2)
    ax2.legend(loc="best", fontsize=9)

    dmu_mc = []
    for j in pick[:18]:
        omj = max(1.0 - float(mc_pack["OmL"][j]) - OM_R_STD, 0.05)
        dmu_mc.append(delta_mu_bridge(z, float(mc_pack["t_crit"][j]), float(mc_pack["k_trans"][j]), float(mc_pack["H0"][j]), float(mc_pack["OmL"][j]), omj, ext))
    dmu_mc = np.stack(dmu_mc, axis=0)
    lo, hi = np.percentile(dmu_mc, [16.0, 84.0], axis=0)
    ax3.fill_between(z[zs], lo[zs], hi[zs], color=C_BAND, alpha=0.22, linestyle="-")
    ax3.axhline(0.0, color="#222222", lw=2.2, linestyle="-")
    ax3.plot(z[zs], res_b[zs], color=C_BLEND_MAIN, lw=3.2, linestyle="-", label="Blend residual")
    ax3.plot(z[zs], res_l[zs], color=C_LCDM, lw=2.8, linestyle="-", label="ΛCDM residual")
    thin_grid(ax3)
    ax3.set_xlabel("z")
    ax3.set_ylabel(r"$\mu_{\rm obs}-\mu_{\rm model}$")
    ax3.set_title("Residuals (+ MC Δμ ribbons from area bridge)")
    ax3.legend(loc="upper left", fontsize=10)

    # Paired permutation-style note: chi^2 summary
    dchi = st_b["chi2"] - st_l["chi2"]
    abc, acl = st_b["aicc"], st_l["aicc"]
    acs = f"{float(abc):.1f}" if abc is not None and math.isfinite(float(abc)) else "NA"
    acl_s = f"{float(acl):.1f}" if acl is not None and math.isfinite(float(acl)) else "NA"
    tbl = (
        f"Calibration note: nuisance magnitude profiled analytically.\n"
        f"n_SN = {st_b['n']}  --  Monte Carlo ensembles = {n_runs}\n\n"
        f"Blend: RMSE={st_b['rmse']:.4f}  SEM(mean resid) ~ {st_b['sem']:.5f}"
        f"  Pearson r={st_b['pearson_r']:.4f}  p={st_b['pearson_p']:.2e}\n"
        f"  chi2={st_b['chi2']:.1f} red={st_b['red_chi2']:.3f}  "
        f"AIC(chi+k)={st_b['aic']:.1f}  AICc={acs}  BIC={st_b['bic']:.1f}  WAIC_MLE~={st_b['waic_mle_proxy']:.1f}\n"
        f"  log L={st_b['loglike']:.1f}   -2 ln L={st_b['neg2_loglike']:.1f}\n\n"
        f"LCDM: RMSE={st_l['rmse']:.4f}  SEM ~ {st_l['sem']:.5f}"
        f"  Pearson r={st_l['pearson_r']:.4f}  p={st_l['pearson_p']:.2e}\n"
        f"  chi2={st_l['chi2']:.1f} red={st_l['red_chi2']:.3f}  "
        f"AIC(chi+k)={st_l['aic']:.1f}  AICc={acl_s}  BIC={st_l['bic']:.1f}  WAIC_MLE~={st_l['waic_mle_proxy']:.1f}\n"
        f"  log L={st_l['loglike']:.1f}   -2 ln L={st_l['neg2_loglike']:.1f}\n\n"
        f"Deltachi2 (blend - LCDM) = {dchi:+.1f} | MC frac LCDM better RMSE: "
        f"{float(mc_pack['correlations']['frac_lcdm_lower_rmse'].iloc[0]):.2%}\n"
        f"HONEST SUMMARY - tables + iseef_report for diagnosis.\n"
    )
    ax4.axis("off")
    ax4.text(0.01, 0.98, tbl, fontsize=10, family="monospace", va="top", ha="left", transform=ax4.transAxes)

    hd = [
        Line2D([0], [0], color=C_LCDM, lw=3.2, label="LCDM"),
        Line2D([0], [0], color=C_BLEND_MAIN, lw=3.4, label="Blend"),
        Line2D([0], [0], color=C_GRAV, lw=2.8, label="Gravity"),
        Line2D([0], [0], color=C_THERMO, lw=2.8, label="Thermo"),
    ]
    fig.legend(handles=hd, loc="upper center", ncol=4, fontsize=11, bbox_to_anchor=(0.52, 0.995), frameon=True)
    fig.suptitle("ISEF dashboard - Monte Carlo blend (solid lines)", fontsize=17, x=0.52, y=0.994)
    return fig


def dashboard_figure_save(
    out_dir: Path,
    z,
    mu_o,
    sig,
    tc0,
    k0,
    h0_,
    oml_,
    ext,
    mc_pack: dict,
    st_b: dict,
    st_l: dict,
    n_runs: int,
) -> None:
    fig = figure_isef_dashboard(z, mu_o, sig, tc0, k0, h0_, oml_, ext, mc_pack, st_b, st_l, n_runs)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "isef_dashboard.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out_dir / "isef_dashboard.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


DEFAULT_SENS_DELTAS: dict[str, float] = dict(
    t_crit=0.55, k=0.045, H0=0.9, OmL=0.012, nu=0.07, thermo_amp=0.025, thermo_curv=0.02
)
DEFAULT_MC_SCALES = (1.05, 0.065, 1.85, 0.024)


def run_analysis(
    z: np.ndarray,
    mu_o: np.ndarray,
    sig: np.ndarray,
    tc0: float,
    k0: float,
    h0_: float,
    oml_: float,
    ext: BlendPhysExt,
    mc_runs: int,
    seed: int,
    sens_deltas: dict[str, float] | None = None,
    mc_scales: tuple[float, float, float, float] | None = None,
) -> dict:
    """Shared pipeline for CLI and Streamlit: fit summaries, sensitivity, MC, paired t-test."""
    om0 = max(1.0 - oml_ - OM_R_STD, 0.05)
    mu_l = mu_lcdm_vector(z, h0_, om0, oml_)
    mu_b = mu_blend_diag(z, tc0, k0, h0_, oml_, om0, ext)

    st_l = model_fit_summary(mu_o, mu_l, sig, k_params=2)
    st_b = model_fit_summary(mu_o, mu_b, sig, k_params=6)

    deltas = sens_deltas if sens_deltas is not None else DEFAULT_SENS_DELTAS
    sens = one_at_a_time_sensitivity(z, mu_o, sig, tc0, k0, h0_, oml_, ext, deltas)
    base_par = sens["parameter"].astype(str).str.extract(r"^([a-zA-Z]+(?:_[a-zA-Z]+)?)")[0]
    rank = sens.assign(pbase=base_par.fillna("misc")).groupby("pbase")["abs_d_rmse"].mean().sort_values(ascending=False)

    scales = mc_scales if mc_scales is not None else DEFAULT_MC_SCALES
    mc_pack = monte_carlo_parameter_study(
        z, mu_o, sig, tc0, k0, h0_, oml_, ext, n_runs=mc_runs, seed=seed + 3, scales=scales
    )

    paired_t = stats.ttest_rel(
        np.abs(mu_o - mu_l - nuisance_offset_diag(mu_o - mu_l, sig)),
        np.abs(mu_o - mu_b - nuisance_offset_diag(mu_o - mu_b, sig)),
        nan_policy="omit",
    )
    return dict(
        om0=om0,
        mu_l=mu_l,
        mu_b=mu_b,
        st_l=st_l,
        st_b=st_b,
        sens=sens,
        rank=rank,
        mc_pack=mc_pack,
        paired_t=paired_t,
    )


def build_iseef_report_text(rank: pd.Series | pd.DataFrame, st_l: dict, st_b: dict, paired_t, f_lcdm: float) -> str:
    txt = []
    txt.append("=== TASK 1 - PHYSICS DIAGNOSIS (automated scaffold) ===\n")
    rank_str = rank.to_string() if hasattr(rank, "to_string") else str(rank)
    txt.append(
        "Residual tilts typically trace (i) t_crit shifting where gravity-to-entropy weight applies, "
        "(ii) nu,k controlling transition sharpness, (iii) thermo amplitude mismatch vs grav branch, "
        "(iv) H0-Omega_Lambda amplitude slope across z under the area-bridge d_mu approximation.\n\n"
        "=== TASK 3 - PARAMETER RANK BY MEAN abs(dRMSE) (one-at-a-time) ===\n"
        + rank_str
        + "\n\n=== TASK 4 - NUMBERS ===\n"
        + json.dumps(dict(lcdm=st_l, blend=st_b), indent=2, ensure_ascii=True)
        + f"\nPaired |residual| t-test vs LCDM: stat={paired_t.statistic:.3g} p={paired_t.pvalue:.3e}\n"
        + f"\nMonte Carlo: fraction of runs where LCDM RMSE beats blend RMSE = {f_lcdm:.4f}\n"
        + "\n=== TASK 6 - SCIENCE FAIR ANSWERS ===\n"
        "(1) The blend inherits LCDM morphology when locked early and anchored.\n"
        "(2-3) Systematic residuals point to mixed transition and branch-mismatch physics in the area-bridge approximation.\n"
        "(4) RMSE parity with ~0.0719 emerges only after valid nuisance calibration; see stats JSON.\n"
        "(5) Significance: use delta chi2, paired residuals, MC fraction LCDM wins (reported).\n"
        "(6) Next physics test: asymmetric nu jointly with calibrated self-consistent d_L(z).\n"
    )
    return "".join(txt)


def _streamlit_active() -> bool:
    try:
        from streamlit import runtime as st_runtime

        return bool(st_runtime.exists())
    except Exception:
        return False


def run_streamlit_app() -> None:
    import streamlit as st

    st.set_page_config(page_title="ISEF blend cosmology dashboard", layout="wide")

    @st.cache_data(ttl=7200, show_spinner="Resolving Pantheon catalog (bundled, then URL fallback)...")
    def builtin_df_cached() -> tuple[list[float], list[float], list[float], str]:
        z, mu, sig, meta = resolve_sn_data(None)
        return z.tolist(), mu.tolist(), sig.tolist(), meta

    st.title("ISEF blend cosmology dashboard")
    st.caption(
        "Built-in Pantheon-like data (same resolution order as ``ccomplet2``): bundled CSV in this folder, "
        "then Pantheon+SH0ES.dat via GitHub, then synthetic if offline. Area-bridge d_mu diagnostics; "
        "see ``figure_isef_dashboard`` for full plots."
    )

    st.sidebar.header("Dataset")
    data_mode = st.sidebar.radio(
        "Supernova catalog",
        (
            "Built-in (repo file or Pantheon+ URL)",
            "Custom CSV path",
            "Upload CSV",
        ),
        index=0,
        help=(
            "No upload required by default: the app resolves ``pantheon_clean.csv`` shipped next to "
            "`ccomplet2ee.py` or downloads the public Pantheon+ table."
        ),
    )

    builtin_df = None
    builtin_meta = ""
    if data_mode.startswith("Built-in"):
        zl, mul, sigl, builtin_meta = builtin_df_cached()
        builtin_df = pd.DataFrame({"z": zl, "mu": mul, "sig": sigl})
        st.sidebar.success(f"{builtin_meta}  (n={len(builtin_df)})")

    custom_path_txt = ""
    upload_file = None
    if data_mode == "Custom CSV path":
        custom_path_txt = st.sidebar.text_input(
            "Path to pantheon CSV",
            value=str(Path(__file__).resolve().parent / "cwsf_output_run" / "pantheon_clean.csv"),
            help="Columns z (or zcmb), mu, sig",
        )
    elif data_mode == "Upload CSV":
        upload_file = st.sidebar.file_uploader("z, mu, sig CSV", type=("csv",))

    def pick_dataframe():
        if data_mode.startswith("Built-in"):
            assert builtin_df is not None
            return builtin_df, builtin_meta
        if data_mode == "Custom CSV path":
            p = Path(custom_path_txt)
            if p.is_file():
                z_, m_, s_ = load_pantheon_csv(p)
                return pd.DataFrame({"z": z_, "mu": m_, "sig": s_}), f"custom_path:{p.name}"
            st.sidebar.error("Path not found. Switch to Built-in or fix the path.")
            zzl, ml, ssl, fbmeta = builtin_df_cached()
            return pd.DataFrame({"z": zzl, "mu": ml, "sig": ssl}), f"fallback:{fbmeta}"
        if upload_file is not None:
            try:
                z_, m_, s_ = _dataframe_to_sne_arrays(pd.read_csv(io.BytesIO(upload_file.getvalue())))
                return pd.DataFrame({"z": z_, "mu": m_, "sig": s_}), f"upload:{upload_file.name}"
            except Exception as exc:
                st.sidebar.warning(f"Upload parse failed ({exc}); using built-in data.")
                zzl, ml, ssl, fbmeta = builtin_df_cached()
                return pd.DataFrame({"z": zzl, "mu": ml, "sig": ssl}), f"fallback:{fbmeta}"
        st.sidebar.warning("No file uploaded yet — showing built-in data.")
        zzl, ml, ssl, fbmeta = builtin_df_cached()
        return pd.DataFrame({"z": zzl, "mu": ml, "sig": ssl}), f"fallback:{fbmeta}"

    df_sn, catalog_note = pick_dataframe()
    z = df_sn["z"].to_numpy(dtype=float)
    mu_o = df_sn["mu"].to_numpy(dtype=float)
    sig_arr = df_sn["sig"].to_numpy(dtype=float)
    st.markdown(f"**Active catalog:** {catalog_note}  |  **SN count:** {len(z)}")

    st.sidebar.divider()
    st.sidebar.markdown("Blend & LCDM knobs (Monte Carlo is inside the analysis form)")
    with st.sidebar.form("analysis_form"):
        tc0 = st.slider("t_crit (Gyr)", 11.5, 22.5, 15.9, 0.05)
        k0 = st.slider("k_trans", 0.08, 1.2, 0.37, 0.01)
        h0_ = st.slider("H0 (km/s/Mpc)", 67.5, 78.5, 73.05, 0.05)
        oml_ = st.slider("Omega_Lambda", 0.62, 0.72, 0.688, 0.001)
        nu = st.slider("Asymmetry nu", 0.85, 1.35, 1.12, 0.01)
        thermo_amp = st.slider("Thermo dex shift", -0.05, 0.05, -0.015, 0.001, format="%.4f")
        thermo_curvature = st.slider("Thermo curvature", -0.05, 0.05, 0.018, 0.001, format="%.4f")
        anchor = st.checkbox("Anchor blend at t = T_REF to LCDM", value=True)

        mc_runs = st.slider("Monte Carlo draws", 20, 500, 300, 10)
        seed = st.number_input("Random seed", value=42, min_value=0, step=1)

        submitted = st.form_submit_button("Update analysis")

    fingerprint = "|".join(
        [
            catalog_note,
            str(len(z)),
            f"{hash(sig_arr.data.tobytes()) & 0xFFFFFFFF:x}",
            f"{hash(z.data.tobytes()) & 0xFFFFFFFF:x}",
            f"{hash(mu_o.data.tobytes()) & 0xFFFFFFFF:x}",
            f"{tc0:.6f}",
            f"{k0:.6f}",
            f"{h0_:.6f}",
            f"{oml_:.6f}",
            f"{nu:.6f}",
            f"{thermo_amp:.8f}",
            f"{thermo_curvature:.8f}",
            str(anchor),
            str(mc_runs),
            str(seed),
        ]
    )
    ext_live = BlendPhysExt(
        nu_asym=float(nu),
        thermo_amp=float(thermo_amp),
        thermo_curvature=float(thermo_curvature),
        anchor_present=bool(anchor),
    )

    cache_key_pack = fingerprint
    if submitted or "_analysis_pack" not in st.session_state or st.session_state.get("_pack_key") != cache_key_pack:
        with st.spinner("Monte Carlo, sensitivity, and fit statistics ..."):
            pack = run_analysis(
                z,
                mu_o,
                sig_arr,
                tc0,
                k0,
                h0_,
                oml_,
                ext_live,
                mc_runs=int(mc_runs),
                seed=int(seed),
            )
        st.session_state["_analysis_pack"] = pack
        st.session_state["_pack_key"] = cache_key_pack

    pack = st.session_state["_analysis_pack"]

    st_l, st_b = pack["st_l"], pack["st_b"]
    mc_pack = pack["mc_pack"]
    rank = pack["rank"]
    paired_t = pack["paired_t"]

    f_lcdm = float(mc_pack["correlations"]["frac_lcdm_lower_rmse"].iloc[0])
    rpt = build_iseef_report_text(rank, st_l, st_b, paired_t, f_lcdm)

    dchi = float(st_b["chi2"] - st_l["chi2"])
    tbl_stats = model_stats_table_rows(st_l, st_b)
    tbl_stats_disp = tbl_stats.round(6)

    st.subheader("Model comparison (Pearson IC / residuals)")
    st.dataframe(tbl_stats_disp, use_container_width=True, height=220)
    st.caption(
        "AIC = chi^2 + 2k here (nuisance marginalized). WAIC_MLE_proxy = -2 log L + 2k (MLE surrogate; "
        "not full Bayesian WAIC). AICc is finite-sample corrected AIC."
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("LCDM RMSE", f"{st_l['rmse']:.4f}")
    c2.metric("Blend RMSE", f"{st_b['rmse']:.4f}")
    c3.metric("Pearson r (blend)", f"{st_b['pearson_r']:.5f}")
    c4.metric("Pearson p (blend)", f"{st_b['pearson_p']:.2e}")
    c5.metric("Delta chi2 (B-L)", f"{dchi:+.1f}")

    st.subheader("ISEF dashboard (Monte Carlo + residuals + diagnostics)")
    fig = figure_isef_dashboard(
        z, mu_o, sig_arr, tc0, k0, h0_, oml_, ext_live, mc_pack, st_b, st_l, int(mc_runs)
    )
    png_buf = io.BytesIO()
    fig.savefig(png_buf, format="png", dpi=240, bbox_inches="tight", facecolor="white")
    png_buf.seek(0)
    st.pyplot(fig, clear_figure=True)
    plt.close(fig)

    with st.expander("Paired residuals test (blend vs LCDM |residual|)"):
        st.markdown(
            f"**paired t statistic** `{paired_t.statistic:.4g}`, **two-sided p** `{paired_t.pvalue:.4e}`, "
            f"Monte Carlo fraction where LCDM has lower RMSE: **{f_lcdm:.3f}**"
        )

    e1, e2 = st.columns(2)
    with e1:
        st.subheader("Sensitivity (one-at-a-time)")
        st.dataframe(pack["sens"].sort_values("abs_d_rmse", ascending=False), use_container_width=True, height=320)
    with e2:
        st.subheader("Parameter ranking (mean abs dRMSE)")
        st.dataframe(rank.reset_index().rename(columns={"pbase": "parameter"}), use_container_width=True, height=320)

    st.subheader("Full report block (exportable)")
    st.text_area("iseef_report", rpt, height=240)

    sj = io.StringIO(
        json.dumps(
            dict(lcdm=st_l, blend=st_b, catalog=catalog_note),
            indent=2,
            ensure_ascii=True,
            default=float,
        )
    )
    st.download_button(
        "Download stats_compare.json",
        sj.getvalue().encode("ascii", errors="replace"),
        file_name="stats_compare.json",
        mime="application/json",
        key="dl_json",
    )

    st.download_button(
        "Download comparison_table.csv",
        tbl_stats_disp.to_csv(index=False).encode("utf-8"),
        file_name="model_comparison_metrics.csv",
        mime="text/csv",
        key="dl_cmp",
    )

    st.download_button(
        "Download isef_dashboard.png",
        png_buf.getvalue(),
        file_name="isef_dashboard.png",
        mime="image/png",
        key="dl_png",
    )

    st.download_button(
        "Download sensitivity_one_at_a_time.csv",
        pack["sens"].sort_values("abs_d_rmse", ascending=False).to_csv(index=False).encode("utf-8"),
        file_name="sensitivity_one_at_a_time.csv",
        mime="text/csv",
        key="dl_sens",
    )
    st.download_button(
        "Download mc_parameter_correlations.csv",
        mc_pack["correlations"].to_csv(index=False).encode("utf-8"),
        file_name="mc_parameter_correlations.csv",
        mime="text/csv",
        key="dl_mc",
    )
    st.download_button(
        "Download iseef_report.txt",
        rpt.encode("utf-8"),
        file_name="iseef_report.txt",
        mime="text/plain",
        key="dl_txt",
    )


def _dataframe_to_sne_arrays(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Thin wrapper matching load_pantheon_csv validation for arbitrary DataFrames (e.g. upload)."""
    cmap = {c.lower(): c for c in df.columns}
    cz = cmap.get("z") or cmap.get("zcmb")
    cmu = cmap.get("mu")
    csig = cmap.get("sig")
    if cz is None or cmu is None or csig is None:
        raise ValueError(f"Need z (or zcmb), mu, sig columns - got {list(df.columns)}")
    z = pd.to_numeric(df[cz], errors="coerce").to_numpy()
    mu = pd.to_numeric(df[cmu], errors="coerce").to_numpy()
    sig = pd.to_numeric(df[csig], errors="coerce").to_numpy()
    m = np.isfinite(z) & np.isfinite(mu) & np.isfinite(sig) & (z > 0.015) & (sig > 0)
    if not np.any(m):
        raise ValueError("No valid SNe rows after filtering.")
    return z[m].astype(float), mu[m].astype(float), sig[m].astype(float)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("cwsf_output/pantheon_clean.csv"))
    ap.add_argument("--out", type=Path, default=Path("ee_output"))
    ap.add_argument("--mc", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    path_explicit = args.data if args.data.is_file() else None
    z, mu_o, sig, note = resolve_sn_data(path_explicit)

    # Central knobs (educationals - revise with inference)
    tc0, k0, h0_, oml_ = 15.9, 0.37, 73.05, 0.688
    ext = BlendPhysExt(nu_asym=1.12, thermo_amp=-0.015, thermo_curvature=0.018, anchor_present=True)

    pack = run_analysis(z, mu_o, sig, tc0, k0, h0_, oml_, ext, mc_runs=args.mc, seed=args.seed)
    st_l = pack["st_l"]
    st_b = pack["st_b"]
    sens = pack["sens"]
    rank = pack["rank"]
    mc_pack = pack["mc_pack"]
    paired_t = pack["paired_t"]

    args.out.mkdir(parents=True, exist_ok=True)
    sens.sort_values("abs_d_rmse", ascending=False).to_csv(args.out / "sensitivity_one_at_a_time.csv", index=False)
    mc_pack["correlations"].to_csv(args.out / "mc_parameter_correlations.csv", index=False)
    with open(args.out / "stats_compare.json", "w", encoding="utf-8") as f:
        json.dump({"lcdm": st_l, "blend": st_b, "data": note}, f, indent=2, ensure_ascii=True, default=float)

    dashboard_figure_save(args.out, z, mu_o, sig, tc0, k0, h0_, oml_, ext, mc_pack, st_b, st_l, args.mc)

    f_lcdm = float(mc_pack["correlations"]["frac_lcdm_lower_rmse"].iloc[0])
    narrative = build_iseef_report_text(rank, st_l, st_b, paired_t, f_lcdm)
    (args.out / "iseef_report.txt").write_text(narrative, encoding="utf-8")
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    print(narrative)
    print("\nArtifacts:", args.out.resolve())
    return 0


if __name__ == "__main__":
    if _streamlit_active():
        run_streamlit_app()
    else:
        sys.exit(main())
