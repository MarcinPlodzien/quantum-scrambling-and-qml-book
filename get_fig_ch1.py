#!/usr/bin/env python3
"""
Generate Random Matrix Theory (RMT) figures for Chapter 1: Mathematical Foundations.

This script separates DATA GENERATION (expensive) from PLOTTING (fast) so that
layout adjustments do not require re-computing eigenvalues.  Data is cached in
.npz files; to regenerate, delete the cache files or pass --regenerate.

Figures produced
────────────────
1. fig_ch1_level_spacing.pdf / .png
   Nearest-neighbour spacing distributions P(s) for the three Dyson ensembles
   (GOE β=1, GUE β=2, GSE β=4).  Each panel shows:
     • a histogram from numerical diagonalization,
     • the Wigner surmise (analytical curve),
     • the Poisson distribution e^{-s} (integrable/uncorrelated reference).

2. fig_ch1_semicircle.pdf / .png
   Eigenvalue density from a large GUE sample compared with the Wigner
   semicircle law.

Mathematical background
───────────────────────
The three Gaussian ensembles are defined by the probability measure

    P(H) dH  ∝  exp[ −(β/4) Tr(H²) ] dH

on the space of D×D matrices H that are:
  • real symmetric        (GOE, Dyson index β = 1),
  • complex Hermitian     (GUE, Dyson index β = 2),
  • quaternion self-dual  (GSE, Dyson index β = 4).

The Dyson index β counts the number of independent real parameters per
off-diagonal matrix element (1 for real, 2 for complex, 4 for quaternion).

Eigenvalue statistics
~~~~~~~~~~~~~~~~~~~~~
The joint eigenvalue density for all three ensembles is

    P(λ₁, …, λ_D) ∝ ∏_{i<j} |λᵢ − λⱼ|^β  ·  exp[ −(β/4) Σᵢ λᵢ² ]

The repulsion factor |λᵢ − λⱼ|^β implies that nearby eigenvalues repel
each other.  The Dyson index β controls the strength of this repulsion:
  β = 1 → linear repulsion   (GOE),
  β = 2 → quadratic repulsion (GUE),
  β = 4 → quartic repulsion   (GSE).

Wigner surmise
~~~~~~~~~~~~~~
For 2×2 matrices drawn from the Gaussian ensemble, the exact
nearest-neighbour spacing distribution is

    P_β(s) = a_β · s^β · exp(−b_β · s²)

where the constants a_β, b_β are fixed by the two normalization conditions:
  (i)   ∫₀^∞ P(s) ds = 1       (probability normalization),
  (ii)  ∫₀^∞ s P(s) ds = 1     (mean spacing = 1 after unfolding).

These integrals yield, using Γ-function identities:

  b_β = [ Γ((β+2)/2) / Γ((β+1)/2) ]²
  a_β = 2 · b_β^{(β+1)/2} / Γ((β+1)/2)

Explicit values:
  β = 1:  b₁ = π/4 ≈ 0.785,       a₁ = π/2 ≈ 1.571
  β = 2:  b₂ = 4/π ≈ 1.273,       a₂ = 32/π² ≈ 3.242
  β = 4:  b₄ = 64/(9π) ≈ 2.264,   a₄ = 2^18/(3^6·π³) ≈ 11.60

Note that P(s) is a probability DENSITY and can exceed 1. In particular,
the GSE Wigner surmise peaks at P(s_peak) ≈ 1.22.

The Wigner surmise is exact for 2×2 matrices and provides an excellent
approximation (within a few percent) to the spacing distribution of
large D matrices.

Poisson distribution
~~~~~~~~~~~~~~~~~~~~
For integrable (non-chaotic) quantum systems, the eigenvalues behave as
uncorrelated random variables. After unfolding to unit mean spacing,
the spacing distribution is

    P_Poisson(s) = e^{-s}

Note that P_Poisson(0) = 1: there is no level repulsion, so small spacings
(near-degeneracies) are common.  This is the key spectral diagnostic
of integrability.

Wigner semicircle law
~~~~~~~~~~~~~~~~~~~~~
In the limit D → ∞, the eigenvalue density of the Gaussian ensembles
converges to the Wigner semicircle.  For the normalization used here
(eigenvalues in [−R, R] with R = 1):

    ρ(λ) = (2 / πR²) √(R² − λ²),   |λ| ≤ R.

This exact result is derived in the text using the Coulomb-gas saddle-point
equation (Exercise 1.5).

GSE implementation note
~~~~~~~~~~~~~~~~~~~~~~~
The GSE is realized as a 2D×2D complex Hermitian matrix with the
symplectic constraint  J H^T J^{-1} = H,  where J = iσ_y ⊗ I_D.
The resulting matrix has the block structure

    H = [[ A,    B  ],
         [-B*,   A* ]]

where A = A† (Hermitian) and B = -B^T (antisymmetric).  The spectrum
consists of D Kramers-degenerate pairs: each eigenvalue appears exactly
twice.  To compute spacing statistics, we extract one eigenvalue from each
pair (i.e., take every second eigenvalue from the sorted spectrum).

Usage:
    python get_fig_ch1.py               # generate data (if not cached) + plot
    python get_fig_ch1.py --regenerate  # force data regeneration
    python get_fig_ch1.py --plot-only   # skip generation, use cached data

Dependencies:
    numpy, scipy, matplotlib
"""

import argparse
import numpy as np
from scipy.special import gamma as gamma_fn
import matplotlib.pyplot as plt
from pathlib import Path

# ── Global plot settings ─────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 13,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "text.usetex": False,  # set True if LaTeX is available
})

SCRIPT_DIR = Path(__file__).parent
OUTPUT_DIR = SCRIPT_DIR.parent / "figures" / "ch1"
DATA_DIR = SCRIPT_DIR / "data" / "ch1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ── RMT sampling parameters ─────────────────────────────────────
D = 200           # matrix dimension (D for GOE/GUE; 2D for GSE internally)
N_SAMPLES = 2000  # number of independent matrix samples per ensemble
N_BINS = 60       # histogram bins for spacing distribution


# ══════════════════════════════════════════════════════════════════
#  PART 1: ENSEMBLE GENERATORS
# ══════════════════════════════════════════════════════════════════
#
#  Each generator returns a Hermitian matrix whose eigenvalues lie
#  approximately in [−1, 1] after the normalization H → H / √(2D).
#  This ensures that the bulk eigenvalue density converges to the
#  Wigner semicircle on [−1, 1] as D → ∞.
#
# ══════════════════════════════════════════════════════════════════

def sample_goe(D: int) -> np.ndarray:
    """
    Sample a D×D matrix from the Gaussian Orthogonal Ensemble (β = 1).

    Construction:
        1. Draw a D×D matrix A with iid N(0,1) entries.
        2. Symmetrize: H = (A + A^T) / 2.
        3. Normalize: H → H / √(2D) so that eigenvalues ∈ [−1, 1].

    The diagonal entries have variance 1, the off-diagonal entries
    have variance 1/2.  This matches the GOE measure
    P(H) ∝ exp[−(D/4) Tr(H²)] after the normalization.
    """
    A = np.random.randn(D, D)
    H = (A + A.T) / 2
    return H / np.sqrt(2 * D)


def sample_gue(D: int) -> np.ndarray:
    """
    Sample a D×D matrix from the Gaussian Unitary Ensemble (β = 2).

    Construction:
        1. Draw a D×D complex matrix A with iid entries
           Re(A_ij), Im(A_ij) ~ N(0, 1/2).
        2. Hermitianize: H = (A + A†) / 2.
        3. Normalize: H → H / √(2D).

    Each off-diagonal element has two independent real parameters,
    giving the Dyson index β = 2.
    """
    A = (np.random.randn(D, D) + 1j * np.random.randn(D, D)) / np.sqrt(2)
    H = (A + A.conj().T) / 2
    return H / np.sqrt(2 * D)


def sample_gse(D: int) -> np.ndarray:
    """
    Sample from the Gaussian Symplectic Ensemble (β = 4).

    The GSE acts on a D-dimensional quaternionic vector space, which
    is realized as a 2D-dimensional complex Hilbert space with the
    symplectic (Kramers) constraint:

        J H^T J^{-1} = H,   where  J = iσ_y ⊗ I_D = [[0, I], [-I, 0]].

    This forces the block structure:
        H = [[ A,    B  ],
             [-B*,   A* ]]
    where:
        A = A†          (D×D Hermitian),
        B = -B^T        (D×D complex antisymmetric).

    The spectrum of H consists of exactly D degenerate pairs
    (Kramers degeneracy).  Each unique eigenvalue appears twice.

    Construction:
        1. Draw A: D×D complex, then Hermitianize.
        2. Draw B: D×D complex, then antisymmetrize.
        3. Assemble the 2D×2D block matrix.
        4. Normalize: H → H / √(4D) to place eigenvalues in [−1, 1].
           (The factor 4D instead of 2D accounts for the doubled
            matrix dimension.)

    Returns the 2D×2D matrix.  The caller must extract one eigenvalue
    from each Kramers pair (see get_spacings_gse).
    """
    # A: D×D Hermitian block
    X = (np.random.randn(D, D) + 1j * np.random.randn(D, D)) / np.sqrt(2)
    A = (X + X.conj().T) / 2

    # B: D×D antisymmetric block
    Y = (np.random.randn(D, D) + 1j * np.random.randn(D, D)) / np.sqrt(2)
    B = (Y - Y.T) / 2

    # Assemble 2D×2D symplectic matrix
    H = np.block([[A, B],
                  [-B.conj(), A.conj()]])

    return H / np.sqrt(4 * D)


# ══════════════════════════════════════════════════════════════════
#  PART 2: SPACING EXTRACTION
# ══════════════════════════════════════════════════════════════════
#
#  "Unfolding" maps the raw eigenvalue sequence to one with uniform
#  mean density.  For the Gaussian ensembles in the bulk, the mean
#  spacing is approximately constant, so dividing by the local mean
#  spacing (computed from the middle 60% of the spectrum to avoid
#  edge effects near the semicircle boundary) provides adequate
#  unfolding for large D.
#
# ══════════════════════════════════════════════════════════════════

def get_spacings(eigenvalues: np.ndarray) -> np.ndarray:
    """
    Compute unfolded nearest-neighbour spacings for GOE / GUE.

    Steps:
        1. Sort eigenvalues in ascending order.
        2. Discard the outer 20% on each side (edge effects near
           the semicircle boundary distort the local density).
        3. Compute consecutive differences Δλᵢ = λᵢ₊₁ − λᵢ.
        4. Normalize to unit mean: sᵢ = Δλᵢ / ⟨Δλ⟩.

    After this simple unfolding, the spacings {sᵢ} should follow
    the Wigner surmise P_β(s).
    """
    eigs = np.sort(eigenvalues)
    n = len(eigs)
    lo, hi = int(0.2 * n), int(0.8 * n)
    bulk = eigs[lo:hi]
    spacings = np.diff(bulk)
    spacings = spacings / np.mean(spacings)
    return spacings


def get_spacings_gse(eigenvalues_2D: np.ndarray) -> np.ndarray:
    """
    Compute unfolded nearest-neighbour spacings for the GSE.

    Because the GSE spectrum has exact Kramers degeneracy, the raw
    2D eigenvalues come in D degenerate pairs:

        λ₁, λ₁, λ₂, λ₂, ..., λ_D, λ_D      (sorted).

    To obtain the physical spacing distribution:
        1. Sort all 2D eigenvalues.
        2. Take every second eigenvalue: λ₁, λ₂, ..., λ_D.
           This extracts one representative from each Kramers pair.
        3. Apply the standard unfolding (bulk extraction + normalization).

    The resulting spacings follow the β = 4 Wigner surmise.
    """
    eigs_all = np.sort(eigenvalues_2D)
    # Extract unique eigenvalues: take every other one from sorted list
    eigs_unique = eigs_all[::2]
    return get_spacings(eigs_unique)


# ══════════════════════════════════════════════════════════════════
#  PART 3: DATA GENERATION (expensive — cached to disk)
# ══════════════════════════════════════════════════════════════════

def generate_spacing_data():
    """
    Diagonalize N_SAMPLES matrices for each ensemble, extract unfolded
    spacings, and save to .npz files.
    """
    ensembles = [
        ("GOE", sample_goe, get_spacings),
        ("GUE", sample_gue, get_spacings),
        ("GSE", sample_gse, get_spacings_gse),
    ]

    for name, sampler, spacer in ensembles:
        print(f"  Generating {name} spacings ({N_SAMPLES} × {D}×{D})...")
        all_spacings = []
        for i in range(N_SAMPLES):
            H = sampler(D)
            eigs = np.linalg.eigvalsh(H)
            spacings = spacer(eigs)
            all_spacings.extend(spacings)

        all_spacings = np.array(all_spacings)
        outfile = DATA_DIR / f"spacings_{name}.npz"
        np.savez(outfile, spacings=all_spacings, D=D, N_SAMPLES=N_SAMPLES)
        print(f"    → {len(all_spacings)} spacings saved to {outfile}")


def generate_semicircle_data():
    """
    Diagonalize N_SAMPLES matrices for each ensemble and save all eigenvalues.
    The semicircle law is universal across GOE/GUE/GSE, so we generate all three
    to visually demonstrate this universality.
    """
    ensembles = [
        ("GOE", sample_goe, False),
        ("GUE", sample_gue, False),
        ("GSE", sample_gse, True),
    ]

    for name, sampler, is_gse in ensembles:
        print(f"  Generating {name} eigenvalues ({N_SAMPLES} × {D}×{D})...")
        all_eigs = []
        for _ in range(N_SAMPLES):
            H = sampler(D)
            eigs = np.linalg.eigvalsh(H)
            if is_gse:
                # GSE: extract unique eigenvalues (one per Kramers pair)
                eigs = np.sort(eigs)[::2]
            all_eigs.extend(eigs)

        all_eigs = np.array(all_eigs)
        outfile = DATA_DIR / f"eigenvalues_{name}.npz"
        np.savez(outfile, eigenvalues=all_eigs, D=D, N_SAMPLES=N_SAMPLES)
        print(f"    → {len(all_eigs)} eigenvalues saved to {outfile}")


def data_exists() -> bool:
    """Check if all cached data files exist."""
    files = [
        DATA_DIR / "spacings_GOE.npz",
        DATA_DIR / "spacings_GUE.npz",
        DATA_DIR / "spacings_GSE.npz",
        DATA_DIR / "eigenvalues_GOE.npz",
        DATA_DIR / "eigenvalues_GUE.npz",
        DATA_DIR / "eigenvalues_GSE.npz",
    ]
    return all(f.exists() for f in files)


def export_csv_summaries():
    """
    Export human-readable CSV files with histogram bin data and analytical
    curves.  These are small (< 10 KB each) and can be loaded in any
    language (Julia, Mathematica, R, gnuplot) for quick inspection or
    alternative plotting without re-running the expensive computation.

    Files produced:
      - spacing_histograms.csv : bin centres, GOE/GUE/GSE histogram
          densities, and Wigner surmise + Poisson analytical values
      - semicircle_histogram.csv : bin centres and eigenvalue density
          histogram, plus analytical semicircle values
    """
    import csv

    # ── Level-spacing histogram summary ──────────────────────────
    s_bins = np.linspace(0, 4, N_BINS + 1)
    s_centres = (s_bins[:-1] + s_bins[1:]) / 2

    rows = []
    for s_c in s_centres:
        row = {"s": f"{s_c:.4f}"}
        row["Poisson"] = f"{np.exp(-s_c):.6f}"
        for beta, name in [(1, "GOE"), (2, "GUE"), (4, "GSE")]:
            row[f"Wigner_{name}"] = f"{wigner_surmise(np.array([s_c]), beta)[0]:.6f}"
        rows.append(row)

    # Add numerical histogram densities
    for name in ["GOE", "GUE", "GSE"]:
        data = np.load(DATA_DIR / f"spacings_{name}.npz")
        hist_vals, _ = np.histogram(data["spacings"], bins=s_bins, density=True)
        for i, row in enumerate(rows):
            row[f"Hist_{name}"] = f"{hist_vals[i]:.6f}"

    outfile = DATA_DIR / "spacing_histograms.csv"
    fieldnames = ["s", "Poisson",
                  "Wigner_GOE", "Hist_GOE",
                  "Wigner_GUE", "Hist_GUE",
                  "Wigner_GSE", "Hist_GSE"]
    with open(outfile, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"    → CSV: {outfile}")

    # ── Semicircle histogram summary ─────────────────────────────
    data = np.load(DATA_DIR / "eigenvalues_GUE.npz")
    eig_bins = np.linspace(-1.2, 1.2, 81)
    eig_centres = (eig_bins[:-1] + eig_bins[1:]) / 2
    hist_vals, _ = np.histogram(data["eigenvalues"], bins=eig_bins, density=True)

    outfile = DATA_DIR / "semicircle_histogram.csv"
    with open(outfile, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["lambda", "Hist_GUE", "Semicircle"])
        for lam, h_val in zip(eig_centres, hist_vals):
            sc_val = semicircle(np.array([lam]), R=1.0)[0]
            writer.writerow([f"{lam:.4f}", f"{h_val:.6f}", f"{sc_val:.6f}"])
    print(f"    → CSV: {outfile}")


# ══════════════════════════════════════════════════════════════════
#  PART 4: ANALYTICAL FUNCTIONS
# ══════════════════════════════════════════════════════════════════
#
#  These are fast (no sampling needed) and are called at plot time.
#
# ══════════════════════════════════════════════════════════════════

def wigner_surmise(s: np.ndarray, beta: int) -> np.ndarray:
    """
    Wigner surmise P_β(s) for Dyson index β ∈ {1, 2, 4}.

    The exact 2×2 spacing distribution:

        P_β(s) = a_β · s^β · exp(−b_β · s²)

    with coefficients derived from ∫P(s)ds = 1 and ∫s·P(s)ds = 1:

        b_β = [ Γ((β+2)/2) / Γ((β+1)/2) ]²
        a_β = 2 · b_β^{(β+1)/2} / Γ((β+1)/2)

    The s^β prefactor encodes eigenvalue repulsion: the probability
    of finding two eigenvalues at normalized distance s vanishes as
    s → 0 with power β.  This is the central spectral signature
    distinguishing integrable (P(0) = 1, no repulsion) from chaotic
    (P(0) = 0, algebraic repulsion) quantum systems.

    Parameters
    ----------
    s : array
        Normalized spacings (s ≥ 0).
    beta : int
        Dyson index (1 = GOE, 2 = GUE, 4 = GSE).

    Returns
    -------
    P(s) : array
        Probability density at each s. Note this is a density, so
        values > 1 are allowed (the GSE peaks at P ≈ 1.22).
    """
    b = (gamma_fn((beta + 2) / 2) / gamma_fn((beta + 1) / 2))**2
    a = 2.0 * b**((beta + 1) / 2) / gamma_fn((beta + 1) / 2)
    return a * s**beta * np.exp(-b * s**2)


def semicircle(x: np.ndarray, R: float = 1.0) -> np.ndarray:
    """
    Wigner semicircle distribution.

    In the D → ∞ limit, the eigenvalue density of any Gaussian
    ensemble converges to:

        ρ(λ) = (2 / πR²) √(R² − λ²),   |λ| ≤ R.

    This is derived from the Coulomb-gas saddle-point equation
    (see Exercise 1.5 in the text).  For our normalization, R = 1.
    """
    rho = np.zeros_like(x)
    mask = np.abs(x) < R
    rho[mask] = (2 / (np.pi * R**2)) * np.sqrt(R**2 - x[mask]**2)
    return rho


# ══════════════════════════════════════════════════════════════════
#  PART 5: PLOTTING (fast — uses cached data)
# ══════════════════════════════════════════════════════════════════

def plot_level_spacing():
    """
    Three-panel figure: P(s) for GOE, GUE, GSE.

    Each panel contains:
      • Histogram: numerical spacings from D×D matrix diagonalization
      • Solid curve: Wigner surmise P_β(s) = a_β s^β exp(−b_β s²)
      • Dashed curve: Poisson distribution P(s) = exp(−s),
        the spacing distribution for uncorrelated (integrable) spectra

    The contrast between Poisson (clustering allowed, P(0) = 1) and
    Wigner (repulsion enforced, P(0) = 0) is the spectral hallmark
    of the integrable-to-chaotic transition.
    """
    print("  Plotting level-spacing figure...")

    panels = [
        ("(a)", r"GOE ($\beta=1$)", "GOE", 1, "#2E86AB"),
        ("(b)", r"GUE ($\beta=2$)", "GUE", 2, "#A23B72"),
        ("(c)", r"GSE ($\beta=4$)", "GSE", 4, "#F18F01"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
    s_theory = np.linspace(0, 4, 300)
    P_poisson = np.exp(-s_theory)

    for ax, (panel_label, display_name, file_key, beta, color) in zip(axes, panels):
        # Load cached data
        data = np.load(DATA_DIR / f"spacings_{file_key}.npz")
        spacings = data["spacings"]

        # ── Histogram (numerical) ────────────────────────────────
        ax.hist(spacings, bins=N_BINS, range=(0, 4), density=True,
                alpha=0.55, color=color, edgecolor="white", linewidth=0.4,
                label="Numerics")

        # ── Wigner surmise (analytical) ──────────────────────────
        P_ws = wigner_surmise(s_theory, beta)
        ax.plot(s_theory, P_ws, color="k", lw=2.0, ls="-",
                label="Wigner surmise")

        # ── Poisson reference (integrable / uncorrelated) ────────
        ax.plot(s_theory, P_poisson, color="gray", lw=1.5, ls="--",
                label=r"Poisson $e^{-s}$")

        # ── Panel label: (a) GOE (β=1)  etc ─────────────────────
        ax.text(0.97, 0.95, f"{panel_label} {display_name}",
                transform=ax.transAxes, fontsize=12, va="top", ha="right",
                fontweight="bold")

        ax.set_xlabel(r"$s / \langle s \rangle$")
        ax.set_xlim(0, 4)
        ax.legend(loc="lower right", framealpha=0.8, fontsize=9)

    axes[0].set_ylabel(r"$P(s)$")
    plt.tight_layout()

    outpath = OUTPUT_DIR / "fig_ch1_level_spacing.pdf"
    fig.savefig(outpath, bbox_inches="tight")
    fig.savefig(outpath.with_suffix(".png"), bbox_inches="tight")
    print(f"    → Saved: {outpath}")
    plt.close(fig)


def plot_semicircle():
    """
    Three-panel eigenvalue density histogram vs. Wigner semicircle.

    The semicircle law is universal: it holds for all three symmetry
    classes.  Showing all three side by side makes this universality
    visually obvious.
    """
    print("  Plotting semicircle figure...")

    panels = [
        ("(a)", r"GOE ($\beta=1$)", "GOE", "#2E86AB"),
        ("(b)", r"GUE ($\beta=2$)", "GUE", "#A23B72"),
        ("(c)", r"GSE ($\beta=4$)", "GSE", "#F18F01"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
    x = np.linspace(-1.2, 1.2, 500)
    rho_sc = semicircle(x, R=1.0)

    for ax, (panel_label, display_name, file_key, color) in zip(axes, panels):
        data = np.load(DATA_DIR / f"eigenvalues_{file_key}.npz")
        all_eigs = data["eigenvalues"]

        # ── Histogram (numerical eigenvalue density) ─────────────
        ax.hist(all_eigs, bins=80, range=(-1.2, 1.2), density=True,
                alpha=0.55, color=color, edgecolor="white", linewidth=0.4,
                label="Numerics")

        # ── Semicircle (analytical) ──────────────────────────────
        ax.plot(x, rho_sc, "k-", lw=2.5,
                label=r"Semicircle $\rho(\lambda)$")

        ax.text(0.97, 0.95, f"{panel_label} {display_name}",
                transform=ax.transAxes, fontsize=12, va="top", ha="right",
                fontweight="bold")

        ax.set_xlabel(r"$\lambda$")
        ax.set_xlim(-1.3, 1.3)
        ax.set_ylim(0, 0.8)
        ax.legend(loc="lower center", framealpha=0.8, fontsize=9)

    axes[0].set_ylabel(r"$\rho(\lambda)$")
    plt.tight_layout()

    outpath = OUTPUT_DIR / "fig_ch1_semicircle.pdf"
    fig.savefig(outpath, bbox_inches="tight")
    fig.savefig(outpath.with_suffix(".png"), bbox_inches="tight")
    print(f"    → Saved: {outpath}")
    plt.close(fig)

# ══════════════════════════════════════════════════════════════════
#  Figure 3: Eigenvalue Tick Marks (Level Repulsion Visualization)
# ══════════════════════════════════════════════════════════════════
#
#  Four horizontal "energy lines" showing eigenvalue positions as
#  vertical tick marks for a single realization of:
#    • Poisson (independent uniform random variables on [0, 1])
#    • GOE (β = 1)
#    • GUE (β = 2)
#    • GSE (β = 4)
#
#  This figure makes eigenvalue repulsion immediately visible:
#  Poisson shows clumping and gaps; increasing β produces
#  progressively more uniform spacing between eigenvalues.
#
# ══════════════════════════════════════════════════════════════════

def plot_eigenvalue_sticks():
    """
    Eigenvalue tick-mark figure: a single realization of each ensemble
    displayed as tick marks on a horizontal energy line.

    Uses the same seed for reproducibility; draws D=60 eigenvalues
    (enough to see the pattern, few enough to resolve individual ticks).
    """
    print("  Plotting eigenvalue tick-mark figure...")
    np.random.seed(42)  # reproducible realization

    D_small = 60  # small enough to see individual ticks

    # ── Generate one realization per ensemble ─────────────────────
    # Poisson: uncorrelated uniform points
    eigs_poisson = np.sort(np.random.uniform(0, 1, D_small))

    # GOE
    H_goe = sample_goe(D_small)
    eigs_goe = np.linalg.eigvalsh(H_goe)

    # GUE
    H_gue = sample_gue(D_small)
    eigs_gue = np.linalg.eigvalsh(H_gue)

    # GSE (extract unique from Kramers pairs)
    H_gse = sample_gse(D_small)
    eigs_gse_all = np.linalg.eigvalsh(H_gse)
    eigs_gse = eigs_gse_all[::2]  # one per Kramers pair

    # ── Rescale all to [0, 1] for uniform visual comparison ──────
    def rescale(e):
        return (e - e.min()) / (e.max() - e.min())

    ensembles = [
        ("Poisson", rescale(eigs_poisson), "gray"),
        (r"GOE ($\beta=1$)", rescale(eigs_goe), "#2E86AB"),
        (r"GUE ($\beta=2$)", rescale(eigs_gue), "#A23B72"),
        (r"GSE ($\beta=4$)", rescale(eigs_gse), "#F18F01"),
    ]

    fig, ax = plt.subplots(figsize=(10, 2.5))

    for i, (label, eigs, color) in enumerate(ensembles):
        y = i  # vertical position
        # Draw energy line
        ax.axhline(y, color="lightgray", lw=0.5, zorder=0)
        # Draw tick marks at eigenvalue positions
        ax.vlines(eigs, y - 0.3, y + 0.3, colors=color, lw=1.2, zorder=2)
        # Label
        ax.text(-0.02, y, label, ha="right", va="center", fontsize=11,
                transform=ax.get_yaxis_transform())

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.7, len(ensembles) - 0.3)
    ax.set_xlabel("Energy (rescaled to [0, 1])")
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    plt.tight_layout()

    outpath = OUTPUT_DIR / "fig_ch1_level_repulsion.pdf"
    fig.savefig(outpath, bbox_inches="tight")
    fig.savefig(outpath.with_suffix(".png"), bbox_inches="tight")
    print(f"    → Saved: {outpath}")
    plt.close(fig)

    np.random.seed(None)  # reset seed


# ══════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Ch1 RMT figures")
    parser.add_argument("--regenerate", action="store_true",
                        help="Force regeneration of cached data")
    parser.add_argument("--plot-only", action="store_true",
                        help="Skip data generation, use cached data")
    args = parser.parse_args()

    print("=" * 60)
    print("Chapter 1 — RMT Figure Generation")
    print(f"  D = {D},  N_SAMPLES = {N_SAMPLES}")
    print("=" * 60)

    # ── Data generation ──────────────────────────────────────────
    if args.plot_only:
        if not data_exists():
            print("ERROR: Cached data not found. Run without --plot-only first.")
            exit(1)
        print("  Using cached data (--plot-only)")
    elif args.regenerate or not data_exists():
        print("\n── Generating data (this takes a few minutes) ──")
        generate_spacing_data()
        generate_semicircle_data()
        export_csv_summaries()
    else:
        print("  Cached data found. Use --regenerate to recompute.")

    # ── Plotting ─────────────────────────────────────────────────
    print("\n── Generating figures ──")
    plot_level_spacing()
    plot_semicircle()
    plot_eigenvalue_sticks()

    print("\nDone. Figures saved to:", OUTPUT_DIR)
