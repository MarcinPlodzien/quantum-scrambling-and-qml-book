#!/usr/bin/env python3
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Chapter 5 — D_KL Convergence: Clifford vs non-Clifford circuits       ║
# ║  Figure: D_KL(P_data || P_MP) as a function of circuit depth L         ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
#
# KL divergence from the Marchenko–Pastur distribution as a function of
# brickwork circuit depth L, comparing:
#   • non-Clifford:  Ry(π/4) + CNOT brickwork   → D_KL drops to ~0
#   • Clifford:      H + S + CNOT brickwork      → D_KL stays high
#
# ═══════════════════════════════════════════════════════════════════════════════
#  MATHEMATICAL BACKGROUND
# ═══════════════════════════════════════════════════════════════════════════════
#
# Marchenko–Pastur distribution (in ξ = −ln λ space):
#
#   ρ_MP(x) = 1/(2πcx) · √[(x₊ − x)(x − x₋)]     for x ∈ [x₋, x₊]
#
#   where  c = m/n,  x₊ = (1 + √c)² / n,  x₋ = (1 − √c)² / n
#
#   In ξ = −ln(λ) with x = m·e^{−ξ}:
#     ρ_ξ(ξ) = ρ_MP(x) · x = m·e^{−ξ} / (2πc) · √[(x₊ − x)(x − x₋)]
#
# KL Divergence:
#   D_KL(P || Q) = Σᵢ pᵢ · ln(pᵢ/qᵢ)
#
#   Computed with ε-smoothing to handle sparse histograms:
#     P_smooth = (P + ε) / Σ(P + ε)
#     Q_smooth = (Q + ε) / Σ(Q + ε)
#
# ═══════════════════════════════════════════════════════════════════════════════

import os, sys, time, argparse
from pathlib import Path

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["JAX_ENABLE_X64"] = "1"

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "figure.dpi": 200,
})

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
DATA_DIR = Path("data/ch5")
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "figures" / "ch5"
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════
#  GATE DEFINITIONS
# ══════════════════════════════════════════════════════════════════

_RY = jnp.array([[jnp.cos(jnp.pi/8), -jnp.sin(jnp.pi/8)],
                  [jnp.sin(jnp.pi/8),  jnp.cos(jnp.pi/8)]], dtype=CDT)
_H = jnp.array([[1, 1], [1, -1]], dtype=CDT) / jnp.sqrt(2.0)
_S = jnp.array([[1, 0], [0, 1j]], dtype=CDT)
CNOT_T = jnp.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]],
                    dtype=CDT).reshape(2,2,2,2)

# ══════════════════════════════════════════════════════════════════
#  EINSUM SUBSCRIPT GENERATION
# ══════════════════════════════════════════════════════════════════

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
    m, n = 1 << k, (1 << N) >> k
    C = psi.reshape(m, n)
    rho = jnp.einsum('ab,cb->ac', C, jnp.conj(C))
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
    """MP density ρ_ξ(ξ) = m·e^{-ξ}/(2πc) · √[(x₊−x)(x−x₋)]."""
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
    c           : float  —  aspect ratio m/n
    m           : int    —  d_A = 2^k  (smaller subsystem dimension)
    n_bins      : int    —  number of histogram bins (default 30)

    Returns
    -------
    D_KL ≥ 0.  Returns np.inf for degenerate spectra (< 2 eigenvalues).
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
    # Guard: for γ = 1, x₋ = 0 → ξ → ∞; clamp to keep finite
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
    fig.savefig(outpath, bbox_inches="tight")
    fig.savefig(outpath.with_suffix(".png"), bbox_inches="tight")
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
