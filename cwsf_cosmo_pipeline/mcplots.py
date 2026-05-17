#!/usr/bin/env python3
"""
ISEF-style Monte Carlo blend plots — fully self-contained.

Run::
    python mcplots.py
    python mcplots.py --outdir figures_mc --n-runs 400 --n-pts 600

Outputs ``isef_mc_area.png`` / ``isef_mc_area.pdf`` in the chosen folder.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy import integrate
from scipy.interpolate import interp1d

# =============================================================================
# Constants (same naming as ccomplet2 / notebook expectations)
# =============================================================================
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

MC_T_MIN = 1.0
MC_T_MAX = 100.0

PRIOR_BOUNDS_MC = {
    "t_crit_gyr": (10.0, 40.0),
    "k_gyr_inv": (0.05, 2.0),
    "H0_kms_mpc": (60.0, 82.0),
    "Omega_Lambda": (0.58, 0.85),
}


def _safe_log_sinh(x: np.ndarray) -> np.ndarray:
    x = np.atleast_1d(np.asarray(x, dtype=float))
    out = np.empty_like(x)
    sm = x < 20.0
    out[sm] = np.log(np.sinh(np.clip(x[sm], 1e-300, None)))
    out[~sm] = x[~sm] - np.log(2.0)
    return out


def _trunc_normal(rng: np.random.Generator, n: int, mu: float, sig: float, lo: float, hi: float) -> np.ndarray:
    x = rng.normal(mu, sig, n)
    return np.clip(x, lo, hi)


def log10_A_lcdm(t_gyr: np.ndarray, H0_kms: float, OmL: float, OmM: float) -> np.ndarray:
    H0si = H0_kms * KMS_TO_SI
    ts = np.atleast_1d(np.asarray(t_gyr, float)) * GYR_TO_S
    t0s = T_REF_S
    arg = 1.5 * H0si * np.sqrt(OmL) * ts
    arg0 = 1.5 * H0si * np.sqrt(OmL) * t0s
    omm_safe = max(float(OmM), 1.0e-9)
    oml_safe = max(float(OmL), 1.0e-9)
    la = (1.0 / 3.0) * np.log(omm_safe / oml_safe) + (2.0 / 3.0) * _safe_log_sinh(arg)
    la0 = float((1.0 / 3.0) * np.log(omm_safe / oml_safe) + (2.0 / 3.0) * _safe_log_sinh(np.atleast_1d(arg0))[0])
    return (np.log(A0) + 2.0 * (la - la0)) / np.log(10.0)


def _rad_matter_time_table_impl(H0_kms: float, OmM: float, OmR: float) -> tuple[np.ndarray, np.ndarray]:
    a_grid = np.logspace(-10.0, np.log10(12.0), 6000)
    H0_si = max(float(H0_kms), 1.0e-9) * KMS_TO_SI
    omm = max(float(OmM), 1.0e-12)
    omr = max(float(OmR), 1.0e-14)
    denom = np.sqrt(omm / np.clip(a_grid, 1.0e-14, None) ** 3 + omr / np.clip(a_grid, 1.0e-14, None) ** 4)
    dt_da = 1.0 / np.clip(a_grid * H0_si * denom, 1.0e-30, None)
    t_grid_s = integrate.cumulative_trapezoid(dt_da, a_grid, initial=0.0)
    return t_grid_s / GYR_TO_S, a_grid


_RAD_TABLE_CACHE: dict[tuple[float, float, float], tuple[np.ndarray, np.ndarray]] = {}


def _rad_matter_lookup(H0_kms: float, OmM: float, OmR: float) -> tuple[np.ndarray, np.ndarray]:
    key = (round(float(H0_kms), 4), round(float(OmM), 6), round(float(OmR), 9))
    if key not in _RAD_TABLE_CACHE:
        _RAD_TABLE_CACHE[key] = _rad_matter_time_table_impl(H0_kms, OmM, OmR)
    return _RAD_TABLE_CACHE[key]


def log10_A_rad_matter(t_gyr: np.ndarray, H0_kms: float, OmM: float, OmR: float = OM_R_STD) -> np.ndarray:
    t_arr = np.asarray(t_gyr, dtype=float)
    t_tbl, a_tbl = _rad_matter_lookup(H0_kms, OmM, OmR)
    a_of_t = interp1d(
        t_tbl,
        a_tbl,
        kind="linear",
        bounds_error=False,
        fill_value=(float(a_tbl[0]), float(a_tbl[-1])),
    )
    a_t = np.clip(np.asarray(a_of_t(np.clip(t_arr, 0.0, None)), dtype=float), 1.0e-40, None)
    a_ref = float(np.clip(a_of_t(np.array([T_REF]))[0], 1.0e-40, None))
    return A0_LOG10 + 2.0 * np.log10(a_t / a_ref)


def log10_A_thermo(t_gyr: np.ndarray, H0_kms: float, OmL: float, t_future_gyr: float = T_FUTURE_DEFAULT_GYR) -> np.ndarray:
    H_lam = H0_kms * KMS_TO_SI * math.sqrt(OmL)
    dt = (np.asarray(t_gyr, float) - float(t_future_gyr)) * GYR_TO_S
    return A0_LOG10 + 2.0 * H_lam * dt / np.log(10.0)


def log10_A_blend(
    t_gyr: np.ndarray,
    tc: float,
    k: float,
    H0_kms: float,
    OmL: float,
    OmM: float,
    t_future_gyr: float = T_FUTURE_DEFAULT_GYR,
) -> np.ndarray:
    t = np.asarray(t_gyr, dtype=float)
    la_lcdm = log10_A_lcdm(t, H0_kms, OmL, OmM)
    early = t < T_TRANSITION_MIN_GYR
    out = np.empty_like(t, dtype=float)
    out[early] = la_lcdm[early]
    late = ~early
    if np.any(late):
        t_l = t[late]
        wloc = 1.0 / (1.0 + np.exp(np.clip(-k * (t_l - tc), -500.0, 500.0)))
        lg = log10_A_rad_matter(t_l, H0_kms, OmM, OM_R_STD)
        lt = log10_A_thermo(t_l, H0_kms, OmL, t_future_gyr=t_future_gyr)
        a_b = (1.0 - wloc) * 10.0**lg + wloc * 10.0 ** np.clip(lt, -300.0, 500.0)
        out[late] = np.log10(np.clip(a_b, 1e-300, None))
    return out


def blend_transition_weight(t_gyr: np.ndarray, tc: float, k: float) -> np.ndarray:
    """Sigmoid weight u(t) toward thermodynamic branch (late-time blend recipe)."""
    t = np.asarray(t_gyr, dtype=float)
    w = np.zeros_like(t, dtype=float)
    late = t >= T_TRANSITION_MIN_GYR
    if np.any(late):
        w[late] = 1.0 / (1.0 + np.exp(np.clip(-k * (t[late] - tc), -500.0, 500.0)))
    return w


def _run_full_mc(
    n_runs_: int = 250,
    n_pts_: int = 500,
    seed_: int = 42,
    tc_mu_: float = 15.9,
    tc_sig_: float = 1.0,
    k_mu_: float = 0.60,
    k_sig_: float = 0.15,
    h0_mu_: float = 73.0,
    h0_sig_: float = 1.0,
    oml_mu_: float = 0.688,
    oml_sig_: float = 0.015,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray], pd.DataFrame]:
    """
    Monte Carlo over blend parameters.

    Returns
    -------
    t : (n_pts,) cosmic time [Gyr]
    ref : (n_pts,) ΛCDM log10(A) using **median** sampled cosmology (display reference).
    la_b : (n_runs, n_pts) blend log10(A) per draw.
    w : (n_pts,) transition weight u(t) at **median** (t_crit, k).
    res : (n_runs, n_pts) la_b - ref
    S : dict of sampled arrays with keys t_crit, k_trans, H0, OmL
    sdf : per-run RMSE (dex) and heuristic accuracy [%]
    """
    rng = np.random.default_rng(seed_)
    b_tc = PRIOR_BOUNDS_MC["t_crit_gyr"]
    b_k = PRIOR_BOUNDS_MC["k_gyr_inv"]
    b_h0 = PRIOR_BOUNDS_MC["H0_kms_mpc"]
    b_oml = PRIOR_BOUNDS_MC["Omega_Lambda"]

    tc_a = _trunc_normal(rng, n_runs_, tc_mu_, tc_sig_, b_tc[0], b_tc[1])
    k_a = _trunc_normal(rng, n_runs_, k_mu_, k_sig_, b_k[0], b_k[1])
    h0_a = _trunc_normal(rng, n_runs_, h0_mu_, h0_sig_, b_h0[0], b_h0[1])
    oml_a = _trunc_normal(rng, n_runs_, oml_mu_, oml_sig_, b_oml[0], b_oml[1])

    t = np.linspace(MC_T_MIN, MC_T_MAX, int(n_pts_), dtype=float)
    la_b = np.zeros((n_runs_, t.size), dtype=float)
    for i in range(n_runs_):
        omm = max(1.0 - float(oml_a[i]), 0.05)
        la_b[i, :] = log10_A_blend(t, float(tc_a[i]), float(k_a[i]), float(h0_a[i]), float(oml_a[i]), OmM=omm)

    med_h0 = float(np.median(h0_a))
    med_oml = float(np.median(oml_a))
    med_omm = max(1.0 - med_oml, 0.05)
    ref = log10_A_lcdm(t, med_h0, med_oml, med_omm)

    res = la_b - ref[np.newaxis, :]

    tc_med = float(np.median(tc_a))
    k_med = float(np.median(k_a))
    w = blend_transition_weight(t, tc_med, k_med)

    rms_rows = np.sqrt(np.mean(res**2, axis=1))
    # Heuristic “accuracy” for annotation (higher when residuals are smaller)
    acc_rows = 100.0 * np.exp(-np.clip(rms_rows * 35.0, 0.0, 12.0))
    sdf = pd.DataFrame({"rmse": rms_rows, "accuracy": acc_rows})

    S = {
        "t_crit": tc_a,
        "k_trans": k_a,
        "H0": h0_a,
        "OmL": oml_a,
    }
    return t, ref, la_b, w, res, S, sdf


def plot_isef_mc_area(out_dir: Path, n_runs: int, n_pts: int, seed: int) -> None:
    """ISEF-level 2×2 figure (user layout), saved PNG + PDF."""
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 12,
            "axes.titlesize": 16,
            "axes.labelsize": 14,
            "legend.fontsize": 10,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "figure.dpi": 160,
            "axes.linewidth": 1.2,
        }
    )

    t, ref, la_b, _w, res, S, sdf = _run_full_mc(
        n_runs_=n_runs,
        n_pts_=n_pts,
        seed_=seed,
        tc_mu_=15.9,
        tc_sig_=1.0,
        k_mu_=0.60,
        k_sig_=0.15,
        h0_mu_=73.0,
        h0_sig_=1.0,
        oml_mu_=0.688,
        oml_sig_=0.015,
    )

    mean_la = la_b.mean(axis=0)
    med_la = np.median(la_b, axis=0)
    p16_la = np.percentile(la_b, 16, axis=0)
    p84_la = np.percentile(la_b, 84, axis=0)
    p05_la = np.percentile(la_b, 5, axis=0)
    p95_la = np.percentile(la_b, 95, axis=0)

    mean_res = res.mean(axis=0)
    med_res = np.median(res, axis=0)
    p16_res = np.percentile(res, 16, axis=0)
    p84_res = np.percentile(res, 84, axis=0)

    tc_med = float(np.median(S["t_crit"]))
    h0_med = float(np.median(S["H0"]))
    oml_med = float(np.median(S["OmL"]))
    omm_med = 1.0 - oml_med

    # Reference curves (display scalings — match notebook / ISEF slide intent)
    logA_gravity = A0_LOG10 + (4.0 / 3.0) * np.log10(np.maximum(t, 1.0e-10) / T_REF)
    Hlam_med = h0_med * KMS_TO_SI * np.sqrt(oml_med)
    logA_thermo = A0_LOG10 + 2.0 * Hlam_med * (t - T_REF) * GYR_TO_S / np.log(10.0)

    fig = plt.figure(figsize=(16, 9), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[2.0, 1.0])

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, :])

    def style_axis(ax):
        ax.grid(True, alpha=0.18, linewidth=0.8)
        for s in ax.spines.values():
            s.set_alpha(0.75)

    sample_idx = np.linspace(0, la_b.shape[0] - 1, min(18, la_b.shape[0]), dtype=int)
    for i in sample_idx:
        ax1.plot(t, la_b[i], color="0.82", lw=0.8, alpha=0.30, zorder=1)

    ax1.fill_between(t, p16_la, p84_la, color="#1f77b4", alpha=0.16, zorder=2)
    ax1.fill_between(t, p05_la, p95_la, color="#1f77b4", alpha=0.08, zorder=1.5)

    ax1.plot(t, mean_la, color="#1f77b4", lw=3.6, ls="-", label="Mean blend", zorder=4)
    ax1.plot(t, med_la, color="#2ca02c", lw=3.0, ls="-", label="Median blend", zorder=4)
    ax1.plot(t, ref, color="#d62728", lw=3.6, ls="-", label="ΛCDM reference", zorder=5)
    ax1.plot(t, logA_gravity, color="#ffbf00", lw=3.0, ls="-", label="Pure gravity", zorder=3)
    ax1.plot(t, logA_thermo, color="#ff7f0e", lw=3.0, ls="-", label="Pure thermodynamic", zorder=3)

    ax1.axvline(13.8, color="black", lw=2.0, ls="--", label="Present day")
    ax1.axvline(tc_med, color="purple", lw=2.0, ls=":", label="Transition time")

    ax1.set_title("MC Ensemble — Full Cosmic History")
    ax1.set_xlabel("Cosmic Time (Gyr)")
    ax1.set_ylabel(r"$\log_{10}(\mathrm{Physical\ Area}/\mathrm{m}^2)$")
    ax1.set_xlim(t.min(), t.max())
    style_axis(ax1)

    acc = float(np.median(sdf["accuracy"]))
    rmse = float(np.median(sdf["rmse"]))
    ax1.text(
        0.02,
        0.98,
        f"Median accuracy: {acc:.2f}%\nMedian RMSE: {rmse:.4f}",
        transform=ax1.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.8", alpha=0.92),
    )

    mask = (t >= 10) & (t <= 40)

    ax2.fill_between(t[mask], p16_la[mask], p84_la[mask], color="#1f77b4", alpha=0.16, zorder=2)
    ax2.fill_between(t[mask], p05_la[mask], p95_la[mask], color="#1f77b4", alpha=0.08, zorder=1.5)

    ax2.plot(t[mask], mean_la[mask], color="#1f77b4", lw=3.6, ls="-", zorder=4)
    ax2.plot(t[mask], med_la[mask], color="#2ca02c", lw=3.0, ls="-", zorder=4)
    ax2.plot(t[mask], ref[mask], color="#d62728", lw=3.6, ls="-", zorder=5)
    ax2.plot(t[mask], logA_gravity[mask], color="#ffbf00", lw=3.0, ls="-", zorder=3)
    ax2.plot(t[mask], logA_thermo[mask], color="#ff7f0e", lw=3.0, ls="-", zorder=3)

    ax2.axvline(13.8, color="black", lw=2.0, ls="--")
    ax2.axvline(tc_med, color="purple", lw=2.0, ls=":")

    ax2.set_title("Transition Region (10–40 Gyr)")
    ax2.set_xlabel("Cosmic Time (Gyr)")
    ax2.set_xlim(10, 40)
    style_axis(ax2)

    ax3.axhline(0.0, color="black", lw=1.3, alpha=0.85)
    ax3.fill_between(t, p16_res, p84_res, color="#2a9d8f", alpha=0.18, zorder=1)
    ax3.plot(t, mean_res, color="#006d77", lw=3.0, ls="-", label="Mean residual", zorder=3)
    ax3.plot(t, med_res, color="#006d77", lw=2.2, ls="--", label="Median residual", zorder=3)

    ax3.axvline(13.8, color="black", lw=2.0, ls="--")
    ax3.axvline(tc_med, color="purple", lw=2.0, ls=":")

    ax3.set_title("Residual Structure Relative to ΛCDM")
    ax3.set_xlabel("Cosmic Time (Gyr)")
    ax3.set_ylabel(r"$\Delta \log_{10} A$")
    ax3.set_xlim(t.min(), t.max())
    style_axis(ax3)

    legend_handles = [
        Line2D([0], [0], color="#1f77b4", lw=3.6, ls="-", label="Mean blend"),
        Line2D([0], [0], color="#2ca02c", lw=3.0, ls="-", label="Median blend"),
        Line2D([0], [0], color="#d62728", lw=3.6, ls="-", label="ΛCDM reference"),
        Line2D([0], [0], color="#ffbf00", lw=3.0, ls="-", label="Pure gravity"),
        Line2D([0], [0], color="#ff7f0e", lw=3.0, ls="-", label="Pure thermodynamic"),
        Line2D([0], [0], color="black", lw=2.0, ls="--", label="Present day"),
        Line2D([0], [0], color="purple", lw=2.0, ls=":", label="Transition time"),
        Line2D([0], [0], color="#2a9d8f", lw=8, alpha=0.18, label="Residual 68% band"),
    ]

    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=4,
        frameon=True,
        framealpha=0.96,
        bbox_to_anchor=(0.5, 1.01),
    )

    fig.suptitle("Monte Carlo Blend Model vs ΛCDM — ISEF Figure", y=1.04, fontsize=18)

    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_p = out_dir / "isef_mc_area.pdf"
    png_p = out_dir / "isef_mc_area.png"
    fig.savefig(pdf_p, bbox_inches="tight")
    fig.savefig(png_p, dpi=320, bbox_inches="tight")
    plt.close(fig)
    print("Wrote:", pdf_p.resolve())
    print("Wrote:", png_p.resolve())


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="ISEF MC blend area figure (self-contained)")
    ap.add_argument("--outdir", type=Path, default=Path("figures_mc"))
    ap.add_argument("--n-runs", type=int, default=250)
    ap.add_argument("--n-pts", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    plot_isef_mc_area(args.outdir, n_runs=args.n_runs, n_pts=args.n_pts, seed=args.seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
