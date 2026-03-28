#!/usr/bin/env python3
"""
Combined Ch5 figure: MP convergence panels (top) + DKL vs depth (bottom).
Uses cached data from both original scripts.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

BOOK_DIR   = Path(__file__).resolve().parent.parent
DATA_DIR   = Path(__file__).resolve().parent / "data" / "ch5"
OUTPUT_DIR = BOOK_DIR / "figures" / "ch5"
N_Q = 20
HIST_DEPTHS = [5, 10, 15, 20]

plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 13,
    "axes.titlesize": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

def mp_density_xi(xi, c, m):
    x = m * np.exp(-xi)
    xm = (1 - np.sqrt(c))**2
    xp = (1 + np.sqrt(c))**2
    rho = np.zeros_like(x)
    mask = (x > xm) & (x < xp)
    rho[mask] = np.sqrt((xp - x[mask]) * (x[mask] - xm)) / \
                (2 * np.pi * c * x[mask])
    return rho * x


def plot_combined():
    print("  Loading cached data...")
    ry_spec   = np.load(DATA_DIR / f"spectra_ry_N{N_Q}.npz")
    haar_data = np.load(DATA_DIR / f"spectra_haar_N{N_Q}.npz")
    ry_dkl    = np.load(DATA_DIR / f"dkl_ry_N{N_Q}.npz")
    cl_dkl    = np.load(DATA_DIR / f"dkl_cliff_N{N_Q}.npz")

    k = N_Q // 2
    m = 1 << k
    n = (1 << N_Q) >> k
    c = m / n
    xi_grid = np.linspace(1.0, 30.0, 2000)
    mp_curve = mp_density_xi(xi_grid, c, m)

    haar_sp = haar_data["spectra"]
    xi_haar = -np.log(haar_sp[haar_sp > 1e-25])

    # ── Create combined figure ───────────────────────────────
    fig = plt.figure(figsize=(16, 7.5))
    
    # Top row: 4 panels for MP convergence
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 0.85],
                          hspace=0.35, wspace=0.08)
    
    bins = np.linspace(2, 28, 55)
    labels = ["(a)", "(b)", "(c)", "(d)"]
    
    for i, L in enumerate(HIST_DEPTHS):
        ax = fig.add_subplot(gs[0, i])
        
        # Gray Haar histogram
        ax.hist(xi_haar, bins=bins, density=True, alpha=0.35,
                color="gray", edgecolor="white", linewidth=0.4, zorder=1)
        
        # Red circuit histogram
        eigs = ry_spec[f"L{L}"]
        xi = -np.log(eigs[eigs > 1e-25])
        ax.hist(xi, bins=bins, density=True, alpha=0.7,
                color="#E63946", edgecolor="white", linewidth=0.4, zorder=2)
        
        # MP curve
        ax.plot(xi_grid, mp_curve, "k-", lw=2.5, alpha=0.85, zorder=10)
        
        ax.set_xlim(2, 28)
        ax.set_ylim(0, 0.37)
        ax.text(0.97, 0.95, f"{labels[i]}  $L={L}$",
                transform=ax.transAxes, fontsize=12, fontweight="bold",
                va="top", ha="right")
        ax.set_xlabel(r"$\xi = -\ln\,\lambda$")
        if i > 0:
            ax.tick_params(labelleft=False)
    
    fig.axes[0].set_ylabel("Density")
    
    # Bottom row: single wide DKL panel
    ax_dkl = fig.add_subplot(gs[1, :])
    
    depths = ry_dkl["depths"]
    kl_ry = ry_dkl["kl"]
    kl_cl = cl_dkl["kl"]
    
    mask_ry = ~np.isnan(kl_ry)
    mask_cl = ~np.isnan(kl_cl)
    
    ax_dkl.semilogy(depths[mask_ry], kl_ry[mask_ry], "o-", color="#E63946",
                    lw=2.5, markersize=7, zorder=3)
    ax_dkl.semilogy(depths[mask_cl], kl_cl[mask_cl], "s--", color="#457B9D",
                    lw=2, markersize=6, zorder=3)
    
    ax_dkl.text(max(depths) + 0.5, kl_ry[mask_ry][-1],
                "non-Clifford", fontsize=12, color="#E63946",
                va="center", ha="left", fontweight="bold")
    ax_dkl.text(max(depths) + 0.5, kl_cl[mask_cl][-1],
                "Clifford", fontsize=12, color="#457B9D",
                va="center", ha="left", fontweight="bold")
    
    ax_dkl.set_xlabel("Circuit depth $L$")
    ax_dkl.set_ylabel(
        r"$D_{\mathrm{KL}}\!\left(P_{\mathrm{data}} \| P_{\mathrm{MP}}\right)$")
    ax_dkl.set_xlim(0, max(depths) + 4)
    ax_dkl.set_xticks(np.arange(2, max(depths) + 1, 2))
    ax_dkl.text(0.02, 0.95, "(e)", transform=ax_dkl.transAxes,
                fontsize=12, fontweight="bold", va="top", ha="left")
    
    # Save
    outpath = OUTPUT_DIR / "fig_ch5_mp_convergence.pdf"
    fig.savefig(outpath, bbox_inches="tight")
    fig.savefig(outpath.with_suffix(".png"), bbox_inches="tight")
    print(f"    → PDF: {outpath}")
    print(f"    → PNG: {outpath.with_suffix('.png')}")
    plt.close(fig)


if __name__ == "__main__":
    print("=" * 55)
    print("  Ch5 — Combined MP + DKL figure")
    print("=" * 55)
    plot_combined()
    print("Done")
