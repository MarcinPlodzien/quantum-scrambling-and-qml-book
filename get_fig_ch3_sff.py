#!/usr/bin/env python3
"""
get_fig_ch3_sff.py  —  Fig 3.1: Spectral Form Factor
=====================================================
3-panel comparison of the three Wigner-Dyson universality classes:

    (a) GOE  (β=1): real symmetric H, time-reversal symmetric.
               Ramp K_c(t) ≈ 2 t/t_H  — steepest slope.
    (b) GUE  (β=2): complex Hermitian H, time-reversal broken.
               Ramp K_c(t) ≈ t/t_H.
    (c) GSE  (β=4): quaternionic self-dual H, Kramers degeneracy.
               Ramp K_c(t) ≈ t/(2 t_H) — shallowest slope.

K(t) = (1/D) |Tr e^{-iHt}|^2,  normalised so K(0)=D, plateau K(∞)=1.

NO KRON PRODUCTS.  All three ensembles are generated as random matrices
directly (O(D^2) memory, O(D^3) diagonalisation only once per sample).
D=400, 500 samples per ensemble.  Runs in ~20 s on a modern laptop.
Cache: data/ch3/sff_goe_gue_gse.npz
"""

import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

ROOT     = Path(__file__).parent
DATA_DIR = ROOT / "data" / "ch3";    DATA_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR  = ROOT.parent / "figures" / "ch3"; FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family" : "serif",
    "axes.labelsize": 10, "axes.titlesize": 10,
    "xtick.labelsize": 9, "ytick.labelsize": 9,
    "legend.fontsize": 8,
})

# ─── Parameters ───────────────────────────────────────────────────────────────
D         = 400    # matrix size per sample (eigenvalues on [-2, 2] semicircle)
N_SAMPLES = 500    # realisations per ensemble
N_TIMES   = 500    # time grid points (log scale)
RNG       = np.random.default_rng(42)

# ─── Random matrix generators (pure RMT, no kron products) ────────────────────

def gen_goe(D, n, rng):
    """
    GOE (β=1): real symmetric H = (A + A^T) / (2√D), A_ij ~ N(0,1).
    Physical systems: spin chains with time-reversal symmetry (e.g. Heisenberg).
    Ramp: K_c(t) = 2t/t_H - (t/t_H)^2  for t < t_H.
    """
    out = []
    for _ in range(n):
        A = rng.standard_normal((D, D))
        out.append(np.linalg.eigvalsh((A + A.T) / (2 * np.sqrt(D))))
    return out

def gen_gue(D, n, rng):
    """
    GUE (β=2): complex Hermitian H = (G + G†) / (2√D), G_ij ~ CN(0,1).
    Physical systems: systems with broken time-reversal (e.g. magnetic field).
    Ramp: K_c(t) = t/t_H  for t < t_H.
    """
    out = []
    for _ in range(n):
        G = (rng.standard_normal((D,D)) + 1j*rng.standard_normal((D,D))) / np.sqrt(2)
        out.append(np.linalg.eigvalsh((G + G.conj().T) / (2 * np.sqrt(D))))
    return out

def gen_gse(D, n, rng):
    """
    GSE (β=4): quaternionic 2D×2D block H = [[A, B]; [-B*, A*]] / (2√(2D)).
    A = DxD Hermitian, B = DxD antisymmetric complex.
    Eigenvalues are Kramers doublets; we keep one from each pair.
    Physical systems: half-integer spin chains with time-reversal (Kramers).
    Ramp: K_c(t) = t/(2 t_H)  for t < t_H.
    """
    out = []
    for _ in range(n):
        A = rng.standard_normal((D,D)) + 1j*rng.standard_normal((D,D))
        A = (A + A.conj().T) / 2                   # Hermitian
        B = rng.standard_normal((D,D)) + 1j*rng.standard_normal((D,D))
        B = (B - B.T) / 2                          # antisymmetric
        H = np.block([[A, B], [-B.conj(), A.conj()]]) / (2 * np.sqrt(2*D))
        evals = np.linalg.eigvalsh(H)
        out.append(evals[::2])                     # one from each Kramers pair
    return out

# ─── SFF computation ───────────────────────────────────────────────────────────

def avg_sff(spectra, times):
    """K(t) = (1/D) mean_k |sum_n exp(-i E_n t)|^2 over ensemble."""
    K = np.zeros(len(times))
    for evals in spectra:
        Z = np.sum(np.exp(-1j * np.outer(evals, times)), axis=0)
        K += np.abs(Z)**2 / len(evals)
    return K / len(spectra)

# ─── Analytical ramp predictions ──────────────────────────────────────────────
# Linear ramp regime K_c(t) = (β/2) * t/t_H  where β=1,2,4
# GOE: slope = 2, GUE: slope = 1, GSE: slope = 1/2

def main():
    cache = DATA_DIR / "sff_goe_gue_gse.npz"

    if cache.exists():
        print(f"Loading cache: {cache}")
        d = np.load(cache)
        times, K_goe, K_gue, K_gse, t_H = d["times"], d["K_goe"], d["K_gue"], d["K_gse"], float(d["t_H"])
    else:
        print(f"Generating {N_SAMPLES} samples per ensemble (D={D}) ...")
        print("  GOE ..."); goe = gen_goe(D, N_SAMPLES, RNG)
        print("  GUE ..."); gue = gen_gue(D, N_SAMPLES, RNG)
        print("  GSE ..."); gse = gen_gse(D, N_SAMPLES, RNG)

        t_H   = np.pi * D / 2                   # Heisenberg time: t_H = 2π/Δ, Δ≈4/D
        times = np.exp(np.linspace(np.log(1e-3), np.log(5*t_H), N_TIMES))

        print("  Computing SFFs ...")
        K_goe = avg_sff(goe, times)
        K_gue = avg_sff(gue, times)
        K_gse = avg_sff(gse, times)
        np.savez(cache, times=times, K_goe=K_goe, K_gue=K_gue, K_gse=K_gse, t_H=t_H)
        print(f"  Saved: {cache}")

    t_norm = times / t_H                        # dimensionless time units

    # Dip/ramp/plateau boundaries (in t/t_H units)
    T_DIP_END  = 0.025
    T_RAMP_END = 0.90

    # ── Single combined panel: all three ensembles overlaid ───────────────────
    fig, ax = plt.subplots(figsize=(6.5, 4.2), constrained_layout=True)

    # Background shading
    ax.axvspan(t_norm[0],  T_DIP_END,  alpha=0.10, color="#fdae61", zorder=0)
    ax.axvspan(T_DIP_END,  T_RAMP_END, alpha=0.10, color="#a1d99b", zorder=0)
    ax.axvspan(T_RAMP_END, t_norm[-1], alpha=0.10, color="#9ecae1", zorder=0)

    # Data curves
    for K, lbl, color in [
        (K_goe, r"GOE ($\beta=1$)", "#2ca02c"),
        (K_gue, r"GUE ($\beta=2$)", "#1f77b4"),
        (K_gse, r"GSE ($\beta=4$)", "#d62728"),
    ]:
        ax.loglog(t_norm, K, lw=1.8, color=color, label=lbl, zorder=3)

    # Ramp slope guides — anchored to the actual K(t) data in the ramp window.
    # In the ramp region K(t) ≈ A * (t/t_H)^1, so loglog slope = 1 for all β.
    # The β-dependence is the PREFACTOR A = β_ramp = 2, 1, 0.5 (GOE, GUE, GSE).
    # We find A by fitting:  A = median(K / (t/t_H)) over the ramp window.
    t_guide = np.linspace(T_DIP_END * 1.5, T_RAMP_END * 0.95, 100)
    ramp_mask = (t_norm >= T_DIP_END * 2) & (t_norm <= T_RAMP_END * 0.6)

    for K, color, slope_label in [
        (K_goe, "#2ca02c", r"$\sim 2\,t/t_H$"),
        (K_gue, "#1f77b4", r"$\sim t/t_H$"),
        (K_gse, "#d62728", r"$\sim t/(2t_H)$"),
    ]:
        # Fit amplitude: A = K / (t/t_H) → median in ramp window
        A = np.median(K[ramp_mask] / t_norm[ramp_mask])
        ax.loglog(t_guide, A * t_guide, "--", color=color, lw=2.2,
                  alpha=1.0, label=slope_label, zorder=4)

    ax.axhline(1., ls="--", color="gray", lw=1.0, zorder=1)
    ax.axvline(1., ls=":",  color="k",   lw=0.9, alpha=0.5, zorder=1)

    # Region labels
    for tx, ty, lbl, c in [(0.04, 0.97, "Dip",     "#b45309"),
                            (0.50, 0.97, "Ramp",    "#31a354"),
                            (0.80, 0.15, "Plateau", "#3182bd")]:
        ax.text(tx, ty, lbl, transform=ax.transAxes, fontsize=9,
                color=c, fontstyle="italic", va="top")

    ax.set_xlabel(r"$t\,/\,t_H$", fontsize=11)
    ax.set_ylabel(r"$K(t)$",       fontsize=11)
    ax.set_xlim(left=1e-3)
    ax.legend(ncol=2, loc="upper right", framealpha=0.90, fontsize=9)

    for ext in (".pdf", ".png"):
        fig.savefig(FIG_DIR / f"fig_ch3_sff{ext}", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"Saved: {FIG_DIR}/fig_ch3_sff.pdf/.png")

if __name__ == "__main__":
    main()
