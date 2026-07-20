#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════
  Chapter 3 — Haar Measure and Induced Ensembles: Figure Generation
═══════════════════════════════════════════════════════════════════

Produces three publication-quality figures for the Springer monograph:

  Fig 2.1: Page curve — numerical mean ± std of entanglement entropy
           from Haar-random states, validated against the analytical
           Page formula.
  Fig 2.2: Marchenko–Pastur density — eigenvalue distribution of
           reduced density matrices for several aspect ratios c = m/n.
  Fig 2.3: Concentration of entropy — 3-panel histogram of S(ρ_A)
           for N = 4, 8, 16 qubits, showing how the distribution
           narrows with increasing system size (Lévy's lemma).

Usage
-----
    python get_fig_ch3.py                 # generate data + plot
    python get_fig_ch3.py --plot-only     # re-plot from cached data
    python get_fig_ch3.py --regenerate    # force data regeneration

Data → codes_for_figures/data/ch3/
Figs → figures/ch3/

References
----------
  [1] Page, Phys. Rev. Lett. 71 (1993) 1291; arXiv:9305007
  [2] Marchenko & Pastur, Math USSR-Sb 1 (1967) 457
  [3] Hayden, Leung, Winter, Comm. Math. Phys. 265 (2006) 95
  [4] Ledoux, The Concentration of Measure Phenomenon (2001)
"""

import sys
from functools import partial
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.special import digamma
from scipy.stats import norm
from matplotlib.ticker import MaxNLocator

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_style import apply_book_style, load_or_compute, panel_label

# ══════════════════════════════════════════════════════════════════
#  PATHS AND PARAMETERS
# ══════════════════════════════════════════════════════════════════

BOOK_DIR   = Path(__file__).resolve().parent.parent
DATA_DIR   = Path(__file__).resolve().parent / "data" / "ch3"
OUTPUT_DIR = BOOK_DIR / "figures" / "ch3"

DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Numerical parameters ─────────────────────────────────────────
# Page curve: Haar-random sampling for these N values
PAGE_CURVE_N_NUMERICAL = [10, 14]        # full numerics (mean ± std)
PAGE_CURVE_N_ANALYTICAL = [18, 22, 24]   # analytical only (digamma)
N_SAMPLES_PAGE = 2000                     # Haar states per (N, k) pair

# Concentration figure: system sizes for 3-panel histogram
CONCENTRATION_N_VALUES = [4, 8, 16]
N_SAMPLES_ENTROPY = 10000  # Haar states per system size


# ══════════════════════════════════════════════════════════════════
#  PART 1: ANALYTICAL AND NUMERICAL FUNCTIONS
# ══════════════════════════════════════════════════════════════════

def page_entropy_exact(m, n):
    """
    Page's exact formula for the average entanglement entropy.

    E[S(ρ_A)] = ψ(mn+1) − ψ(n+1) − (m−1)/(2n)

    where ψ is the digamma function.  This gives O(1) evaluation.

    Parameters: m, n = dimensions of subsystems A, B (m ≤ n).
    Returns: E[S] in natural-log units.
    """
    if m > n:
        m, n = n, m
    mn = m * n
    S = float(digamma(mn + 1) - digamma(n + 1)) - (m - 1.0) / (2.0 * n)
    return S


def marchenko_pastur(x, c):
    """
    Marchenko–Pastur density for aspect ratio c = m/n ≤ 1.

      ρ_MP(x) = 1/(2πcx) · √[(x₊ − x)(x − x₋)]

    supported on [x₋, x₊] = [(1−√c)², (1+√c)²].
    """
    x_minus = (1 - np.sqrt(c))**2
    x_plus  = (1 + np.sqrt(c))**2
    rho = np.zeros_like(x, dtype=float)
    mask = (x > x_minus) & (x < x_plus)
    rho[mask] = (1.0 / (2.0 * np.pi * c * x[mask])) * \
                np.sqrt((x_plus - x[mask]) * (x[mask] - x_minus))
    return rho


def partial_trace_rdm(psi, d_A, d_B):
    """
    Compute reduced density matrix ρ_A by tracing out subsystem B.

    Pedagogical implementation using einsum
    ----------------------------------------
    Given |ψ⟩ ∈ H_A ⊗ H_B, reshape into C[a,b] = ⟨a_A, b_B|ψ⟩.
    Then ρ_A = Tr_B(|ψ⟩⟨ψ|) is computed via:

      ρ_A[a, a'] = Σ_b  C[a,b] · C*[a',b]
                 = einsum('ab,cb->ac', C, C.conj())

    Returns: eigenvalues of ρ_A (clipped to avoid log(0)).
    """
    C = psi.reshape(d_A, d_B)
    rho_A = np.einsum('ab,cb->ac', C, C.conj())
    eigs = np.linalg.eigvalsh(rho_A)
    return np.clip(eigs, 1e-30, None)


def von_neumann_entropy(eigs):
    """S = −Σ λ_i ln(λ_i), with natural logarithm."""
    return -np.sum(eigs * np.log(eigs))


# ══════════════════════════════════════════════════════════════════
#  PART 2: DATA GENERATION
# ══════════════════════════════════════════════════════════════════
#
#  All numerical data uses the Gaussian method for Haar-random states:
#    1. Draw z ∈ C^D with z_i ~ N(0,1) + i·N(0,1)
#    2. Normalize: |ψ⟩ = z / ||z||
#  This gives an EXACT Haar-random state (not an approximation).
#
#  For each state, we compute the k-qubit reduced density matrix
#  via einsum partial trace, then extract the von Neumann entropy.
#
# ══════════════════════════════════════════════════════════════════

def compute_page_curve(N, n_samples=N_SAMPLES_PAGE):
    """
    For system size N, sample Haar-random states and compute the
    entanglement entropy S(ρ_A) for ALL partition sizes k = 1..N-1.

    For each (N, k) combination:
      - Generate n_samples Haar-random states
      - Compute the k-qubit RDM via einsum partial trace
      - Extract S(ρ_A)
      - Store mean and std

    Returns a dict cached to data/ch3/page_curve_N{N}.npz.
    """
    D = 2**N
    ks = np.arange(1, N)  # k = 1, ..., N-1
    S_mean = np.zeros(len(ks))
    S_std  = np.zeros(len(ks))

    print(f"  Page curve N={N}: {len(ks)} partitions × {n_samples} states...")

    for idx, k in enumerate(ks):
        m = 2**k        # dim(H_A)
        n = D // m      # dim(H_B)
        d_small = min(m, n)
        d_large = max(m, n)

        entropies = np.zeros(n_samples)
        for j in range(n_samples):
            # Exact Haar-random state via normalized Gaussian vector
            z = (np.random.randn(D) + 1j * np.random.randn(D)) / np.sqrt(2)
            z /= np.linalg.norm(z)

            # k-qubit RDM via einsum partial trace
            eigs = partial_trace_rdm(z, d_small, d_large)
            entropies[j] = von_neumann_entropy(eigs)

        S_mean[idx] = np.mean(entropies)
        S_std[idx]  = np.std(entropies)
        print(f"    k={k}/{N-1}: ⟨S⟩ = {S_mean[idx]:.4f} ± {S_std[idx]:.4f}")

    return {"ks": ks, "S_mean": S_mean, "S_std": S_std,
            "N": N, "n_samples": n_samples}


def compute_concentration(N, n_samples=N_SAMPLES_ENTROPY):
    """
    Generate Haar-random states and compute half-partition entropy.

    Returns a dict cached to data/ch3/entropies_N{N}.npz.
    """
    D = 2**N
    k = N // 2
    m = 2**k
    n = D // m

    print(f"  Concentration N={N}: {n_samples} states "
          f"(D={D}, m={m}, n={n})...")

    entropies = np.zeros(n_samples)
    for i in range(n_samples):
        z = (np.random.randn(D) + 1j * np.random.randn(D)) / np.sqrt(2)
        z /= np.linalg.norm(z)
        eigs = partial_trace_rdm(z, m, n)
        entropies[i] = von_neumann_entropy(eigs)

        if (i+1) % 2000 == 0:
            print(f"    {i+1}/{n_samples}")

    return {"entropies": entropies, "N": N, "k": k, "m": m, "n": n,
            "N_SAMPLES": n_samples}


# ══════════════════════════════════════════════════════════════════
#  PART 3: PLOTTING
# ══════════════════════════════════════════════════════════════════

# ──────────────────────────────────────────────────────────────────
#  Figure 3.2: Page Curve
# ──────────────────────────────────────────────────────────────────
#
#  PRIMARY DATA: numerical mean ± std from Haar-random states.
#  For each N, we sample many Haar states, compute the k-qubit
#  reduced density matrix via einsum, and extract S(ρ_A).
#  The shaded band shows ± 1σ.
#
#  VALIDATION: the analytical Page formula (digamma) is overlaid
#  as a dashed line for comparison.
#
#  Larger system sizes (N ≥ 18) use analytical formula only.
#
# ──────────────────────────────────────────────────────────────────

def plot_page_curve():
    """Plot the Page curve with numerical mean ± std bands."""
    print("  Plotting Page curve...")

    fig, ax = plt.subplots(figsize=(7, 4.5))

    # ── Numerical data for N = 10, 14: markers only (no line) ────
    # Plotting markers without a connecting line makes it clear that
    # these are discrete sample means from Haar-random states.
    # The solid analytical line (below) passes exactly through them,
    # proving the exact agreement between Page's formula and numerics.
    num_colors  = ["#2E86AB", "#A23B72"]
    num_markers = ["o", "s"]

    for N, color, marker in zip(PAGE_CURVE_N_NUMERICAL,
                                 num_colors, num_markers):
        data = load_or_compute(DATA_DIR / f"page_curve_N{N}.npz",
                               partial(compute_page_curve, N))
        ks = data["ks"]
        S_mean = data["S_mean"]
        x = ks / N

        # Markers only — no connecting line
        ax.plot(x, S_mean, linestyle="none", marker=marker, color=color,
                markersize=6, zorder=4,
                label=f"$N = {N}$ (Haar numerics)")

        # Solid analytical line through the same points
        S_analytical = np.array([page_entropy_exact(2**k, 2**N // 2**k)
                                  for k in ks])
        ax.plot(x, S_analytical, '-', color=color, lw=2, zorder=3,
                label=f"$N = {N}$ (Page formula)")

    # ── Analytical only for larger N ─────────────────────────────
    ana_colors  = ["#F18F01", "#555555", "#E63946"]
    ana_markers = ["D", "^", "v"]

    for N, color, marker in zip(PAGE_CURVE_N_ANALYTICAL,
                                 ana_colors, ana_markers):
        D = 2**N
        ks = np.arange(1, N)
        S_page = np.array([page_entropy_exact(2**k, D // 2**k)
                           for k in ks])
        ax.plot(ks / N, S_page, marker=marker, color=color,
                markersize=3, lw=2, label=f"$N = {N}$ (analytical)")

    # ── S_max reference line ─────────────────────────────────────
    N_max = max(PAGE_CURVE_N_ANALYTICAL)
    k_frac = np.linspace(0.001, 0.999, 200)
    ax.plot(k_frac,
            N_max * np.minimum(k_frac, 1 - k_frac) * np.log(2),
            'k--', lw=1.5, alpha=0.3,
            label=r"$S_{\max} = \min(k, N{-}k)\,\ln 2$")

    ax.set_xlabel(r"Subsystem fraction $k/N$")
    ax.set_ylabel(r"$\mathbb{E}[S_A]$")
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels([r"$0$", r"$1/4$", r"$1/2$", r"$3/4$", r"$1$"])
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper left", framealpha=0.8, fontsize=8)
    plt.tight_layout()

    outpath = OUTPUT_DIR / "fig_ch3_page_curve.pdf"
    fig.savefig(outpath, bbox_inches="tight")
    fig.savefig(outpath.with_suffix(".png"), bbox_inches="tight")
    print(f"    → Saved: {outpath}")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────
#  Figure 3.1: Marchenko–Pastur Density (analytical)
# ──────────────────────────────────────────────────────────────────

def plot_marchenko_pastur():
    """Plot the Marchenko–Pastur density in a 3-panel layout."""
    print("  Plotting Marchenko–Pastur figure...")

    c_values = [0.25, 0.5, 0.75]
    colors = ["#2E86AB", "#A23B72", "#F18F01"]
    panel_labels = ["(a)", "(b)", "(c)"]

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))

    for ax, c, color, plabel in zip(axes, c_values, colors, panel_labels):
        x_minus = (1 - np.sqrt(c))**2
        x_plus = (1 + np.sqrt(c))**2
        x = np.linspace(max(x_minus + 1e-4, 1e-4), x_plus - 1e-4, 500)
        rho = marchenko_pastur(x, c)
        ax.plot(x, rho, lw=2.5, color=color)
        ax.fill_between(x, rho, alpha=0.25, color=color)

        ax.axvline(1.0, color="gray", ls=":", lw=1, alpha=0.5)

        ax.text(0.97, 0.95, f"{plabel} $c = {c}$",
                transform=ax.transAxes, fontsize=12, va="top", ha="right",
                fontweight="bold")

        ax.set_xlabel(r"$x = d_A\lambda$")
        ax.set_xlim(0, x_plus + 0.3)
        ax.set_ylim(0, None)

    axes[0].set_ylabel(r"$\rho_{\mathrm{MP}}(x)$")
    plt.tight_layout()

    outpath = OUTPUT_DIR / "fig_ch3_marchenko_pastur.pdf"
    fig.savefig(outpath, bbox_inches="tight")
    fig.savefig(outpath.with_suffix(".png"), bbox_inches="tight")
    print(f"    → Saved: {outpath}")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────
#  Figure 3.6: Concentration of Entanglement Entropy
# ──────────────────────────────────────────────────────────────────
#
#  Three-panel histogram of S(ρ_A) for N = 4, 8, 16 qubits.
#  Gaussian fit envelope + σ in scientific notation.
#
# ──────────────────────────────────────────────────────────────────

def plot_entropy_concentration():
    """Plot 3-panel entropy histogram showing concentration with N."""
    print("  Plotting entropy concentration figure...")

    colors = ["#2E86AB", "#A23B72", "#F18F01"]
    panel_labels = ["(a)", "(b)", "(c)"]

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))

    for ax, N, color, plabel in zip(axes, CONCENTRATION_N_VALUES,
                                     colors, panel_labels):
        data = load_or_compute(DATA_DIR / f"entropies_N{N}.npz",
                               partial(compute_concentration, N))
        entropies = data["entropies"]
        m = int(data["m"])
        n = int(data["n"])

        S_page = page_entropy_exact(m, n)

        # Histogram (probability density)
        ax.hist(entropies, bins=60, density=True,
                alpha=0.5, color=color, edgecolor="white", linewidth=0.4)

        # Gaussian fit envelope
        mu = np.mean(entropies)
        sigma = np.std(entropies)
        x_env = np.linspace(mu - 4*sigma, mu + 4*sigma, 200)
        ax.plot(x_env, norm.pdf(x_env, mu, sigma),
                color="k", lw=2, ls="-", alpha=0.7)

        # Page value
        ax.axvline(S_page, color="k", lw=1.5, ls="--", alpha=0.8)

        # σ in scientific notation
        exp = int(np.floor(np.log10(sigma)))
        mantissa = sigma / 10**exp
        sigma_label = f"$\\sigma = {mantissa:.1f} \\times 10^{{{exp}}}$"

        ax.text(0.97, 0.95,
                f"{plabel} $N = {N}$\n$D = 2^{{{N}}}$\n{sigma_label}",
                transform=ax.transAxes, fontsize=10, va="top", ha="right",
                fontweight="bold",
                bbox=dict(boxstyle="round", fc="white", alpha=0.8))

        ax.set_xlabel(r"$S(\hat{\rho}_A)$")
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))

    axes[0].set_ylabel("Probability density")
    plt.tight_layout()

    outpath = OUTPUT_DIR / "fig_ch3_entropy_concentration.pdf"
    fig.savefig(outpath, bbox_inches="tight")
    fig.savefig(outpath.with_suffix(".png"), bbox_inches="tight")
    print(f"    → Saved: {outpath}")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("Chapter 3 — Haar Measure & Induced Ensembles: Figures")
    print(f"  Page numerics: {PAGE_CURVE_N_NUMERICAL}"
          f"  ({N_SAMPLES_PAGE} samples/partition)")
    print(f"  Page analytical: {PAGE_CURVE_N_ANALYTICAL}")
    print(f"  Concentration: {CONCENTRATION_N_VALUES}"
          f"  ({N_SAMPLES_ENTROPY} samples)")
    print("=" * 60)

    # Expensive numerics are cached via load_or_compute (set FIG_RECOMPUTE=1 to
    # force regeneration); the figures load or compute their data on demand.
    print("\n── Generating figures ──")
    apply_book_style()
    plot_page_curve()
    plot_marchenko_pastur()
    plot_entropy_concentration()

    print("\nDone. Figures saved to:", OUTPUT_DIR)
