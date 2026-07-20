#!/usr/bin/env python3
"""
get_fig_ch5_dkl_convergence.py
===================================
Chapter 5. How far the entanglement spectrum of a brickwork circuit sits
from the Marchenko–Pastur (MP) law, as a function of circuit depth L, for
two circuits that differ only in whether their gate set is Clifford:

    non-Clifford:  Ry(π/4) + CNOT brickwork   → D_KL falls to ~5·10⁻³
    Clifford:      H + S + CNOT brickwork     → D_KL plateaus at ~1.4

============================================================
WHICH BOOK FIGURE IS THIS?
============================================================
None, directly. This script writes

    figures/ch5/fig_ch5_dkl_convergence.pdf   (+ .png)

a standalone single-panel figure that is NOT included in the final text:
no \\includegraphics and no \\label in any chapter .tex refers to it. It is
a diagnostic view, useful while checking the scan on a full-width canvas.

The numbers it produces DO reach the book, but through another file. The
cached arrays written here,

    data/ch5/dkl_ry_N20.npz
    data/ch5/dkl_cliff_N20.npz

are read back by get_fig_ch5_combined.py and drawn as panel (e) of
figures/ch5/fig_ch5_mp_convergence.pdf, which is Figure 5.2 of Chapter 5
(ch5_designs.tex, label fig:mp_convergence). So the workflow is:

    this script                 → (re)generates the D_KL data
    get_fig_ch5_combined.py     → renders the published Figure 5.2

============================================================
PHYSICS BACKGROUND
============================================================
Cut the N-qubit state in half (d_A = d_B = 2^(N/2)) and diagonalize the
reduced density matrix ρ_A. For a Haar-random state the rescaled
eigenvalues x = d_A·λ follow the Marchenko–Pastur law

    ρ_MP(x) = √[(x₊ − x)(x − x₋)] / (2πcx)        for x ∈ [x₋, x₊]

    c = d_A/d_B,   x∓ = (1 ∓ √c)²

At the equal bipartition used here c = 1, so the support is x ∈ [0, 4].

D_KL(P_data ‖ P_MP) therefore measures how far a circuit's entanglement
spectrum is from the random-state prediction. It is a sharper probe than
the entanglement entropy: the entropy is a single number (the mean of ξ),
while the spectrum constrains the entire eigenvalue distribution.

WHY THESE TWO CIRCUITS
Both circuits share the same brickwork geometry, the same CNOT layers and
the same initial state. They differ only in the single-qubit gate: Ry(π/4)
is non-Clifford, H and S are Clifford. The comparison therefore isolates
exactly one variable, the gate set.

  • Clifford gates map stabilizer states to stabilizer states. The reduced
    density matrix of a stabilizer state is proportional to a projector,
    so its nonzero eigenvalues are ALL EQUAL to 2^(−S): in ξ = −ln λ the
    spectrum is a single delta peak at S·ln 2, at every depth.
  • The Clifford circuit does generate entanglement: its entropy grows
    with depth until it saturates, and the number of nonzero eigenvalues
    grows with it. What never happens is the spread. A delta is not the MP
    bulk, so D_KL cannot approach zero however deep the circuit runs.
  • Swapping H+S for the non-Clifford Ry(π/4) injects magic, and the very
    same geometry now converges: the spectrum becomes full rank and spread
    out, and D_KL falls from O(10) to ~5·10⁻³ by L ~ N.

READER'S TAKEAWAY
Entanglement is not randomness. A Clifford circuit entangles and is still
efficiently classically simulable (Gottesman–Knill); its entanglement
spectrum stays flat and stays far from MP at any depth. Entropy alone
cannot certify Haar-like scrambling. The spectrum, which is sensitive to
higher moments, is what separates stabilizer states from generic ones, and
the surviving O(1) Clifford plateau is the spectral signature of that gap.

============================================================
ALGORITHM
============================================================
1. Initial state (|0…0⟩ + |1…1⟩)/√2, a GHZ state, identical for both
   circuits. Nothing here is random: θ = π/4 is fixed and every gate is
   deterministic, so each depth L is one specific circuit rather than an
   ensemble average.
2. One layer, applied L times:
     Ry+CNOT :  Ry on every qubit → CNOT on even bonds → CNOT on odd bonds
     Clifford:  CNOT on even bonds → H,S on every qubit
                → CNOT on odd bonds → H,S on every qubit
   Gates are applied by viewing the 2^N amplitude vector as a rank-N
   tensor of shape (2,)*N and contracting it with the gate tensor through
   jnp.einsum, so no 2^N × 2^N operator is ever built.
3. Entanglement spectrum: reshape the amplitude vector into the Schmidt
   matrix C and take eigvalsh(C C†). See IMPLEMENTATION NOTES.
4. D_KL: histogram ξ = −ln λ, evaluate the analytic MP density on the same
   bins, ε-smooth both, and sum Σᵢ pᵢ ln(pᵢ/qᵢ). The circuit is compared
   against the ANALYTIC law, never against sampled Haar states, so no step
   here validates the MP law by drawing from the MP law.
5. Steps 1–4 repeat for L = 1…N; the (depths, kl) pair is cached to .npz.

============================================================
IMPLEMENTATION NOTES
============================================================
PARTIAL TRACE BY RESHAPE  (_get_spectrum)
  No explicit partial trace is needed. The amplitude index is the bit
  string s = s_{N−1}…s_0, so reshape(m, n) with m = 1<<k and
  n = (1<<N)>>k simply splits that index at bit k: the row index is the
  bit string of the first k qubits, the column index that of the rest.
  The reshape IS the bipartition, the reshaped array is exactly the
  Schmidt matrix C, and ρ_A = C C† is the partial trace over B. The
  shifts are the bit-level statement of the same split: 1<<k is 2^k and
  (1<<N)>>k is 2^(N−k).
  eigvalsh may return slightly negative eigenvalues, since ρ_A is
  positive semidefinite only up to roundoff. The next step takes −ln λ,
  which would turn those into NaN, hence the clip at 1e-30.

EINSUM SUBSCRIPTS  (_subs_1q / _subs_2q)
  Subscript strings are precomputed once per N rather than rebuilt at
  every gate application. Lowercase letters label the N input axes; the
  axes a gate rewrites receive an uppercase output letter. _AX holds 21
  letters, which caps this construction at N ≤ 21.

WHY ε-SMOOTHING IS UNAVOIDABLE  (kl_from_mp)
  D_KL = Σ p ln(p/q) requires q > 0 wherever p > 0, and both failure modes
  occur here:
    • q = 0 : the MP density vanishes identically outside [x₋, x₊]. At
      small depth a Clifford delta sits entirely outside that window, so
      every populated bin has q = 0 and the exact divergence is +∞.
    • p = 0 : with 30 bins and a spectrum occupying one of them, most bins
      are empty.
  Adding ε = 1e-10 to both histograms before renormalizing keeps every
  logarithm finite. The price is that ε also sets a CEILING on the
  reported divergence at roughly ln(1/ε) ≈ 23.0, and that ceiling is
  visible in the cached data: the Clifford curve sits at 23.07 for L ≤ 9,
  and the non-Clifford curve starts at 22.4. Those points are the
  regularized stand-in for +∞, meaning "no overlap with the MP support at
  all"; their height is set by ε, not by physics. The informative values
  are the ones well below the ceiling: the Clifford plateau at ~1.43 for
  L ≥ 10 (a delta that now lands inside the MP window, but still a delta)
  and the non-Clifford floor at ~5·10⁻³.

WHY x₋ = 0 FORCES A GUARD  (kl_from_mp)
  At equal bipartition c = 1, so x₋ = (1 − √c)² = 0 exactly. The map
  ξ = ln(d_A) − ln(x) sends x → 0 to ξ → +∞, so the MP support is
  unbounded in ξ. The density itself remains normalizable there (ρ_ξ ~ √x
  as x → 0, i.e. a decaying e^(−ξ/2) tail), but the grid endpoint
  ξ₊ = ln(d_A) − ln(x₋) is literally infinite, and the linspace, the bin
  edges and the bin width derived from it would be too. Clamping
  x_minus_safe = max(x₋, 1e-10) truncates that exponential tail at
  ξ₊ ≈ 30 and keeps the grid finite. Note the clamp only fixes the grid
  endpoint: the density is still evaluated against the true unclamped x₋
  in the  x_minus < x < x_plus  test, so the law is truncated, not
  deformed.

JACOBIAN  (kl_from_mp)
  ρ_ξ(ξ) = ρ_MP(x)·|dx/dξ| = ρ_MP(x)·x with x = d_A·e^(−ξ), and the 1/x
  carried by ρ_MP cancels the Jacobian:

      ρ_ξ(ξ) = √[(x₊ − x)(x − x₋)] / (2πc)

  The code keeps the uncancelled form  √arg/(2πc·x) · x  so that both
  factors stay legible.

============================================================
RUNTIME AND CACHING
============================================================
N_Q = 20 gives a 2^20 = 1 048 576 amplitude complex128 state (16 MB),
scanned over 20 depths for two circuits. Each depth restarts from the GHZ
state instead of reusing the depth-(L−1) state, so the total work grows
quadratically rather than linearly in the deepest L.

  --regenerate   recompute even when the .npz cache exists
  --plot-only    skip simulation entirely and plot from cache

Caches to:       data/ch5/dkl_ry_N{N}.npz, data/ch5/dkl_cliff_N{N}.npz
Saves figure to: figures/ch5/fig_ch5_dkl_convergence.pdf (+ .png)

Both DATA_DIR and OUTPUT_DIR are resolved absolutely from __file__, so this
script may be run from any working directory. The two dkl_* caches it writes
are a cross-script contract read back by get_fig_ch5_combined.py, so its
numerics are cached through those files rather than a single load_or_compute
bundle.
"""

import os, sys, time, argparse
from pathlib import Path

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["JAX_ENABLE_X64"] = "1"

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_style import apply_book_style, load_or_compute, panel_label

import jax
import jax.numpy as jnp
from jax import random

CDT = jnp.complex128
FDT = jnp.float64

# ══════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════════════

N_Q = 20                           # number of qubits
KL_DEPTHS = list(range(1, N_Q+1))  # L = 1, 2, ..., N
DATA_DIR = Path(__file__).resolve().parent / "data" / "ch5"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "figures" / "ch5"
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════
#  GATE DEFINITIONS
# ══════════════════════════════════════════════════════════════════

# Ry(θ) = exp(−iθY/2) carries the half angle, so θ = π/4 appears as π/8 here.
# π/4 is not a multiple of π/2: this gate is non-Clifford, and it is the only
# difference between the two circuits below.
_RY = jnp.array([[jnp.cos(jnp.pi/8), -jnp.sin(jnp.pi/8)],
                  [jnp.sin(jnp.pi/8),  jnp.cos(jnp.pi/8)]], dtype=CDT)
_H = jnp.array([[1, 1], [1, -1]], dtype=CDT) / jnp.sqrt(2.0)
_S = jnp.array([[1, 0], [0, 1j]], dtype=CDT)
# Reshaped to (2,2,2,2) = (out_c, out_t, in_c, in_t) so einsum can contract it
# against two axes of the rank-N state tensor without forming a 2^N operator.
CNOT_T = jnp.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]],
                    dtype=CDT).reshape(2,2,2,2)

# ══════════════════════════════════════════════════════════════════
#  EINSUM SUBSCRIPT GENERATION
# ══════════════════════════════════════════════════════════════════

# Lowercase = the N input axes, uppercase = the axis a gate rewrites.
# 21 letters, so this construction supports N ≤ 21.
_AX = "abcdefghijklmnopqrstu"

def _subs_1q(N):
    subs = []
    for j in range(N):
        ax = list(_AX[:N])
        out = list(ax)
        new = chr(ord('A') + j)
        out[j] = new
        subs.append(f"{new}{ax[j]},{''.join(ax)}->{''.join(out)}")
    return subs

def _subs_2q(N):
    subs = {}
    for c in range(N):
        for t in range(N):
            if c == t: continue
            ax = list(_AX[:N])
            out = list(ax)
            C, T = chr(ord('A')+c), chr(ord('A')+t)
            out[c], out[t] = C, T
            subs[(c,t)] = f"{C}{T}{ax[c]}{ax[t]},{''.join(ax)}->{''.join(out)}"
    return subs


# ══════════════════════════════════════════════════════════════════
#  SPECTRUM EXTRACTION
# ══════════════════════════════════════════════════════════════════

def _get_spectrum(psi, N):
    """Equal bipartition → eigenvalues of ρ_A."""
    k = N // 2
    m, n = 1 << k, (1 << N) >> k          # 2^k and 2^(N−k)
    # Splitting the amplitude index at bit k IS the bipartition: rows are the
    # bit string of qubits 0..k−1, columns the rest. So C is the Schmidt
    # matrix and C C† is the partial trace over B; no explicit trace needed.
    C = psi.reshape(m, n)
    rho = jnp.einsum('ab,cb->ac', C, jnp.conj(C))
    # ρ_A is PSD only up to roundoff; clip so the −ln λ downstream never NaNs.
    return np.array(jnp.clip(jnp.linalg.eigvalsh(rho), 1e-30, None))


# ══════════════════════════════════════════════════════════════════
#  SIMULATORS
# ══════════════════════════════════════════════════════════════════

def simulate_ry_cnot(N, L):
    """Ry(π/4)+CNOT brickwork → spectrum. Fixed θ = π/4 (non-Clifford)."""
    D = 1 << N
    sub1, sub2 = _subs_1q(N), _subs_2q(N)
    even = [(j,j+1) for j in range(0,N-1,2)]
    odd  = [(j,j+1) for j in range(1,N-1,2)]
    # (|0…0⟩ + |1…1⟩)/√2: a GHZ state, the same start for both circuits.
    psi = jnp.zeros(D, dtype=CDT).at[0].set(1/jnp.sqrt(2.0)).at[D-1].set(1/jnp.sqrt(2.0))
    for _ in range(L):
        p = psi.reshape((2,)*N)
        for j in range(N):
            p = jnp.einsum(sub1[j], _RY, p)
        for c,t in even:
            p = jnp.einsum(sub2[(c,t)], CNOT_T, p)
        for c,t in odd:
            p = jnp.einsum(sub2[(c,t)], CNOT_T, p)
        psi = p.reshape(D)
    return _get_spectrum(psi, N)


def simulate_clifford(N, L):
    """H+S+CNOT brickwork → spectrum. Clifford-only (stabilizer state)."""
    D = 1 << N
    sub1, sub2 = _subs_1q(N), _subs_2q(N)
    even = [(j,j+1) for j in range(0,N-1,2)]
    odd  = [(j,j+1) for j in range(1,N-1,2)]
    # (|0…0⟩ + |1…1⟩)/√2: a GHZ state, the same start for both circuits.
    psi = jnp.zeros(D, dtype=CDT).at[0].set(1/jnp.sqrt(2.0)).at[D-1].set(1/jnp.sqrt(2.0))
    for _ in range(L):
        p = psi.reshape((2,)*N)
        for c,t in even:
            p = jnp.einsum(sub2[(c,t)], CNOT_T, p)
        for j in range(N):
            p = jnp.einsum(sub1[j], _H, p)
            p = jnp.einsum(sub1[j], _S, p)
        for c,t in odd:
            p = jnp.einsum(sub2[(c,t)], CNOT_T, p)
        for j in range(N):
            p = jnp.einsum(sub1[j], _H, p)
            p = jnp.einsum(sub1[j], _S, p)
        psi = p.reshape(D)
    return _get_spectrum(psi, N)


# ══════════════════════════════════════════════════════════════════
#  MARCHENKO–PASTUR DENSITY (in ξ space)
# ══════════════════════════════════════════════════════════════════

def mp_density_xi(xi, c, m):
    """MP density in ξ space:  ρ_ξ(ξ) = √[(x₊−x)(x−x₋)] / (2πc),  x = m·e^{−ξ}.

    No factor of x survives: the Jacobian ρ_ξ = ρ_MP(x)·|dx/dξ| = ρ_MP(x)·x
    is cancelled by the 1/x inside ρ_MP. The body below keeps the
    uncancelled form √arg/(2πc·x) · x so both factors stay visible.

    UNUSED: nothing in this file calls this function. kl_from_mp() rebuilds
    the same density inline on its own ξ grid. Retained for reference.

    Check the conventions before reusing it. The band edges below are
    divided by n, while x is the RESCALED variable m·e^{−ξ} = m·λ, whose MP
    edges are (1±√c)² with no 1/n. As written the two disagree: the result
    integrates to 1/n rather than 1, and its ξ-support is shifted by ln(n)
    (for N=20, to ξ > 12.5 instead of ξ > 5.5). The inline version in
    kl_from_mp(), and the copy in get_fig_ch5_combined.py that draws the
    published curve, both use the consistent (1±√c)² edges.
    """
    n = m / c
    sq = np.sqrt(c)
    xp = (1 + sq)**2 / n
    xm = (1 - sq)**2 / n
    x = m * np.exp(-xi)
    arg = (xp - x) * (x - xm)
    ok = arg > 0
    rho = np.zeros_like(xi)
    rho[ok] = np.sqrt(arg[ok]) / (2 * np.pi * c * x[ok]) * x[ok]
    return rho


# ══════════════════════════════════════════════════════════════════
#  KL DIVERGENCE (ε-smoothed, histogram-based)
# ══════════════════════════════════════════════════════════════════

def kl_from_mp(eigenvalues, c, m, n_bins=30):
    """
    D_KL(P_data || P_MP)  —  KL divergence from Marchenko–Pastur.

    Algorithm
    ---------
    1. Filter numerical zeros and convert eigenvalues to ξ = -ln λ.
    2. Build the MP reference density on a fine 300-point grid in ξ,
       using ρ_ξ(ξ) = ρ_MP(x)·x  with  x = d_A·exp(-ξ).
       For equal partition (γ = d_A/d_B = 1), the lower MP edge
       is x₋ = 0, so we guard with x₋ → max(x₋, 1e-10) to keep
       ξ_max finite.
    3. Bin both the data histogram and interpolated MP density into
       n_bins uniform bins spanning [min(ξ_data, ξ_MP), max(...)].
    4. Apply ε-smoothing (ε = 1e-10) to both distributions to avoid
       log(0), then renormalize to probability vectors.
    5. Compute the standard KL divergence Σᵢ pᵢ ln(pᵢ/qᵢ).

    Parameters
    ----------
    eigenvalues : array  —  eigenvalues of ρ_A
    c           : float  —  aspect ratio m/n. Accepted but NOT used: step 2
                            recomputes the ratio as γ from m alone, assuming
                            an equal bipartition. Harmless for the callers
                            here (N even, so γ = c = 1), a trap otherwise.
    m           : int    —  d_A = 2^k  (smaller subsystem dimension)
    n_bins      : int    —  number of histogram bins (default 30)

    Returns
    -------
    D_KL ≥ 0.  Returns np.inf for degenerate spectra (< 2 eigenvalues).
    Values near ln(1/ε) ≈ 23 are the ε-regularized stand-in for +∞, i.e.
    a spectrum with no overlap at all with the MP support.
    """
    # ── Step 1: filter and transform ──────────────────────────────
    valid = eigenvalues[eigenvalues > 1e-15]
    if len(valid) < 2:
        return np.inf

    xi_data = -np.log(valid)

    # ── Step 2: MP reference density on a fine ξ grid ─────────────
    n_A = int(np.log2(m))
    n_B = n_A  # equal partition
    d_A = 2**n_A
    d_B = 2**n_B
    if d_A > d_B:
        d_A, d_B = d_B, d_A
    gamma = d_A / d_B  # γ ≤ 1
    x_minus = (1 - np.sqrt(gamma))**2
    x_plus  = (1 + np.sqrt(gamma))**2
    # Guard: at γ = 1 the lower edge is x₋ = 0 exactly, and ξ = ln(d_A) − ln(x)
    # sends x → 0 to ξ → ∞, so the support is unbounded in ξ and ξ₊ below would
    # be infinite (poisoning the linspace and every bin edge derived from it).
    # Clamping truncates the decaying e^(−ξ/2) tail at ξ₊ ≈ 30. Only the grid
    # endpoint uses the clamp; the density test below still uses the true x₋,
    # so the law is truncated rather than deformed.
    x_minus_safe = max(x_minus, 1e-10)
    xi_minus = np.log(d_A) - np.log(x_plus)       # smallest ξ
    xi_plus  = np.log(d_A) - np.log(x_minus_safe)  # largest ξ

    # 300-point ξ grid for interpolation
    xi_mp = np.linspace(xi_minus, xi_plus, 300)
    rho_mp = np.zeros(300)
    for i, xi in enumerate(xi_mp):
        x = d_A * np.exp(-xi)
        if x_minus < x < x_plus:
            rho_x = np.sqrt((x_plus - x) * (x - x_minus)) / (2*np.pi*gamma*x)
            rho_mp[i] = rho_x * x  # Jacobian: |dx/dξ| = x

    # ── Step 3: common bin edges covering both data and MP ────────
    xi_min = min(np.min(xi_data), np.min(xi_mp))
    xi_max = max(np.max(xi_data), np.max(xi_mp))
    bin_edges = np.linspace(xi_min, xi_max, n_bins + 1)
    bw = bin_edges[1] - bin_edges[0]

    # Data histogram (manual binning, normalized density)
    hist_counts = np.zeros(n_bins)
    for xi in xi_data:
        idx = int(np.clip(np.floor((xi - xi_min) / bw), 0, n_bins - 1))
        hist_counts[idx] += 1
    P_data = hist_counts / (np.sum(hist_counts) * bw)

    # MP density at bin centres via linear interpolation
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    P_mp = np.zeros(n_bins)
    for i, xi in enumerate(bin_centers):
        idx = np.searchsorted(xi_mp, xi, side='right') - 1
        if idx < 0:
            P_mp[i] = rho_mp[0]
        elif idx >= len(xi_mp) - 1:
            P_mp[i] = rho_mp[-1]
        else:
            t = (xi - xi_mp[idx]) / (xi_mp[idx+1] - xi_mp[idx])
            P_mp[i] = (1 - t) * rho_mp[idx] + t * rho_mp[idx+1]

    # Normalize MP to integrate to 1 over the bins
    P_mp = np.maximum(P_mp, 0.0)
    if np.sum(P_mp) * bw > 0:
        P_mp = P_mp / (np.sum(P_mp) * bw)

    # ── Step 4–5: ε-smoothed KL divergence ────────────────────────
    # ε rescues two log(0) cases: q = 0 in bins outside the MP support (where a
    # shallow Clifford delta lives, making the exact D_KL = +∞), and p = 0 in
    # the many empty bins. It also caps the reported value near ln(1/ε) ≈ 23.
    eps = 1e-10
    P_data_s = P_data + eps
    P_mp_s = P_mp + eps
    P_data_s = P_data_s / np.sum(P_data_s)
    P_mp_s = P_mp_s / np.sum(P_mp_s)

    KL = 0.0
    for i in range(n_bins):
        if P_data_s[i] > 0 and P_mp_s[i] > 0:
            KL += P_data_s[i] * np.log(P_data_s[i] / P_mp_s[i])

    return max(KL, 0.0)


# ══════════════════════════════════════════════════════════════════
#  DATA GENERATION
# ══════════════════════════════════════════════════════════════════

def generate_data(regenerate=False):
    t0 = time.time()
    k = N_Q // 2
    m = 1 << k
    c = m / ((1 << N_Q) >> k)

    # Non-Clifford KL scan
    kl_file = DATA_DIR / f"dkl_ry_N{N_Q}.npz"
    if kl_file.exists() and not regenerate:
        print(f"  Cached: {kl_file}")
    else:
        print(f"\n  Non-Clifford KL scan N={N_Q}, L=1..{max(KL_DEPTHS)}:")
        kl_vals = []
        for L in KL_DEPTHS:
            t1 = time.time()
            eigs = simulate_ry_cnot(N_Q, L)
            kl = kl_from_mp(eigs, c, m)
            kl_vals.append(kl)
            print(f"    L={L:2d}: D_KL={kl:.4f} ({time.time()-t1:.1f}s)")
        np.savez(kl_file, depths=np.array(KL_DEPTHS), kl=np.array(kl_vals))
        print(f"    → {kl_file}")

    # Clifford KL scan
    cliff_file = DATA_DIR / f"dkl_cliff_N{N_Q}.npz"
    if cliff_file.exists() and not regenerate:
        print(f"  Cached: {cliff_file}")
    else:
        print(f"\n  Clifford KL scan N={N_Q}, L=1..{max(KL_DEPTHS)}:")
        kl_vals = []
        for L in KL_DEPTHS:
            t1 = time.time()
            eigs = simulate_clifford(N_Q, L)
            kl = kl_from_mp(eigs, c, m)
            kl_vals.append(kl)
            print(f"    L={L:2d}: D_KL={kl:.4f} ({time.time()-t1:.1f}s)")
        np.savez(cliff_file, depths=np.array(KL_DEPTHS), kl=np.array(kl_vals))
        print(f"    → {cliff_file}")

    print(f"\n  Total: {time.time()-t0:.1f}s")


# ══════════════════════════════════════════════════════════════════
#  PLOTTING
# ══════════════════════════════════════════════════════════════════

def plot_figure():
    print("\n── Figure ──")
    print("  Plotting figure...")

    ry_data = np.load(DATA_DIR / f"dkl_ry_N{N_Q}.npz")
    cl_data = np.load(DATA_DIR / f"dkl_cliff_N{N_Q}.npz")

    depths = ry_data["depths"]
    kl_ry = ry_data["kl"]
    kl_cl = cl_data["kl"]

    apply_book_style()
    fig, ax = plt.subplots(figsize=(10, 5))

    # Filter NaN
    mask_ry = ~np.isnan(kl_ry)
    mask_cl = ~np.isnan(kl_cl)

    ax.semilogy(depths[mask_ry], kl_ry[mask_ry], "o-", color="#E63946",
                lw=2.5, markersize=7, zorder=3)
    ax.semilogy(depths[mask_cl], kl_cl[mask_cl], "s--", color="#457B9D",
                lw=2, markersize=6, zorder=3)

    # Right-side labels
    ax.text(max(depths) + 0.5, kl_ry[mask_ry][-1],
            "non-Clifford", fontsize=12, color="#E63946",
            va="center", ha="left", fontweight="bold")
    ax.text(max(depths) + 0.5, kl_cl[mask_cl][-1],
            "Clifford", fontsize=12, color="#457B9D",
            va="center", ha="left", fontweight="bold")

    ax.set_xlabel("Circuit depth $L$")
    ax.set_ylabel(
        r"$D_{\mathrm{KL}}\!\left(P_{\mathrm{data}} \| P_{\mathrm{MP}}\right)$")
    ax.set_xlim(0, max(depths) + 4)
    ax.set_xticks(np.arange(2, max(depths) + 1, 2))

    outpath = OUTPUT_DIR / "fig_ch5_dkl_convergence.pdf"
    fig.savefig(outpath)
    fig.savefig(outpath.with_suffix(".png"))
    print(f"    → PDF: {outpath}")
    print(f"    → PNG: {outpath.with_suffix('.png')}")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--regenerate", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()

    print(f"  N = {N_Q}, L = 1..{max(KL_DEPTHS)}")
    print(f"  JAX: {jax.devices()[0].platform}")

    if not args.plot_only:
        generate_data(regenerate=args.regenerate)

    plot_figure()
    print(f"\nDone → {OUTPUT_DIR}")
