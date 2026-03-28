#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  COMBINED 2-PANEL SRE SCALING FIGURE                                        ║
║  Chapter 5 — Circuit Complexity & Quantum Designs                           ║
║                                                                             ║
║  Panel (a): Digital  — M₂ vs circuit depth L   (Ry(π/4)+CNOT brickwork)     ║
║  Panel (b): Analog   — M₂ vs evolution time t  (Trotterized XX model)       ║
║                                                                             ║
║  NARRATIVE CONSISTENCY:                                                      ║
║    The digital circuit is the SAME deterministic Ry(π/4)+CNOT brickwork     ║
║    used in the entanglement scaling figure.  The reader sees:                ║
║      • Entanglement scaling figure: THIS circuit → volume-law entropy       ║
║      • SRE scaling figure:          THIS circuit → M₂ → N−2                ║
║    Same circuit, two diagnostics, same conclusion: quantum chaos.            ║
║                                                                             ║
║  SRE Algorithm References:                                                  ║
║    [1] Huang et al., arXiv:2512.24685 (2024) — XOR-FWHT algorithm          ║
║    [2] Sierant et al., arXiv:2601.07824 (2025) — HadaMAG.jl package        ║
╚══════════════════════════════════════════════════════════════════════════════╝

USAGE:
    python get_fig_ch5_sre_scaling.py              # generate + plot
    python get_fig_ch5_sre_scaling.py --plot-only   # re-plot from cached data

OUTPUT:
    figures/ch5/fig_ch5_sre_scaling.pdf
    figures/ch5/fig_ch5_sre_scaling.png
"""

import os, sys, time, argparse, string
import numpy as np
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
from jax import jit

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ══════════════════════════════════════════════════════════════════
#  AESTHETIC CONFIGURATION — matches entanglement scaling figure
# ══════════════════════════════════════════════════════════════════

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "axes.labelsize": 14,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

CDT = jnp.complex128

# ── Consistent colors and markers for N = 6, 8, 10, 12 ──────────
COLORS  = ["#1f77b4", "#8fbc3b", "#ff7f0e", "#9467bd"]
MARKERS = ["o", "^", "s", "D"]

# ── Paths ────────────────────────────────────────────────────────
BOOK_DIR   = Path(__file__).resolve().parent.parent
DATA_DIR   = Path(__file__).resolve().parent / "data" / "ch5"
OUTPUT_DIR = BOOK_DIR / "figures" / "ch5"
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Configuration ────────────────────────────────────────────────
N_VALUES = [6, 8, 10, 12]

# Digital: circuit depths to scan
DEPTHS = list(range(0, 21))   # L = 0, 1, 2, ..., 20

# Analog: loads pre-computed data from get_fig_ch5_sre_analog.py
# (times = 0, 0.5, 1.0, ..., 10.0)


# ══════════════════════════════════════════════════════════════════
#  GATE DEFINITIONS — SAME as entanglement scaling figure
# ══════════════════════════════════════════════════════════════════
#
#  Ry(π/4) = exp(-i·(π/4)·σ_y/2) = [[cos(π/8), -sin(π/8)],
#                                     [sin(π/8),  cos(π/8)]]
#
#  This is a DETERMINISTIC gate: same angle on every qubit, every layer.
#  No randomness, no averaging — direct narrative link to entanglement fig.
#
# ══════════════════════════════════════════════════════════════════

_RY = jnp.array([[jnp.cos(jnp.pi/8), -jnp.sin(jnp.pi/8)],
                  [jnp.sin(jnp.pi/8),  jnp.cos(jnp.pi/8)]], dtype=CDT)

CNOT_T = jnp.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]],
                    dtype=CDT).reshape(2, 2, 2, 2)


# ══════════════════════════════════════════════════════════════════
#  EINSUM SUBSCRIPT GENERATION
# ══════════════════════════════════════════════════════════════════

_AX = "abcdefghijklmnopqrstu"

def _subs_1q(N):
    """Einsum subscripts for 1-qubit gates on N qubits."""
    subs = []
    for j in range(N):
        ax = list(_AX[:N])
        out = list(ax)
        new = chr(ord('A') + j)
        out[j] = new
        subs.append(f"{new}{ax[j]},{''.join(ax)}->{''.join(out)}")
    return subs

def _subs_2q(N):
    """Einsum subscripts for 2-qubit gates on N qubits."""
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
#  CIRCUIT SIMULATOR — Ry(π/4) + CNOT brickwork
# ══════════════════════════════════════════════════════════════════

def evolve_ry_cnot(N, L):
    """
    Deterministic Ry(π/4)+CNOT brickwork for L layers.

    Same circuit as the entanglement scaling figure.
    Starting from |0⟩^⊗N (stabilizer state → M₂ = 0).
    """
    D = 1 << N
    sub1, sub2 = _subs_1q(N), _subs_2q(N)
    even = [(j, j+1) for j in range(0, N-1, 2)]
    odd  = [(j, j+1) for j in range(1, N-1, 2)]

    # Initial state: |0⟩^⊗N  (stabilizer state)
    psi = jnp.zeros(D, dtype=CDT).at[0].set(1.0 + 0j)

    for _ in range(L):
        p = psi.reshape((2,)*N)
        for j in range(N):
            p = jnp.einsum(sub1[j], _RY, p)
        for c, t in even:
            p = jnp.einsum(sub2[(c, t)], CNOT_T, p)
        for c, t in odd:
            p = jnp.einsum(sub2[(c, t)], CNOT_T, p)
        psi = p.reshape(D)
    return psi


# ══════════════════════════════════════════════════════════════════
#  FWHT-BASED SRE COMPUTATION
# ══════════════════════════════════════════════════════════════════

def _fwht(v, n_qubits):
    """Fast Walsh-Hadamard Transform: H^{⊗N} · v via butterfly."""
    H2 = jnp.array([[1.0, 1.0], [1.0, -1.0]], dtype=v.dtype)
    t = v.reshape((2,) * n_qubits)
    letters = string.ascii_letters
    for q in range(n_qubits):
        idx = list(letters[:n_qubits])
        out_c = letters[n_qubits]
        in_c = idx[q]
        out_idx = idx.copy()
        out_idx[q] = out_c
        t = jnp.einsum(f"{out_c}{in_c},{''.join(idx)}->{''.join(out_idx)}",
                       H2, t)
    return t.reshape(-1)


def compute_sre(psi, n_qubits):
    """Compute M₂ via FWHT. Cost: O(N · 4^N)."""
    D = 1 << n_qubits
    j_indices = jnp.arange(D)
    total = jnp.float64(0.0)
    for x in range(D):
        j_xor_x = jnp.bitwise_xor(j_indices, x)
        C_x = jnp.conj(psi) * psi[j_xor_x]
        F_x = _fwht(C_x, n_qubits)
        total = total + jnp.sum(jnp.abs(F_x)**4)
    return jnp.where(total > 0, -jnp.log2(total / D), jnp.inf)


# ══════════════════════════════════════════════════════════════════
#  DATA GENERATION
# ══════════════════════════════════════════════════════════════════

def generate_digital_data(regenerate=False):
    """Generate M₂ vs depth L for deterministic Ry(π/4)+CNOT circuit."""
    print("\n── Panel (a): Digital Ry(π/4)+CNOT ──")
    t0 = time.time()

    for N in N_VALUES:
        fpath = DATA_DIR / f"sre_deterministic_N{N}.npz"
        if fpath.exists() and not regenerate:
            print(f"  N={N}: cached → {fpath.name}")
            continue

        D = 1 << N
        print(f"\n  N = {N} (D = {D}):  {len(DEPTHS)} depths, single circuit")

        sre_values = np.zeros(len(DEPTHS))
        for di, L in enumerate(DEPTHS):
            t1 = time.time()
            psi = evolve_ry_cnot(N, L)
            sre_values[di] = float(compute_sre(psi, N))
            dt = time.time() - t1
            print(f"    L={L:3d}:  M₂ = {sre_values[di]:.3f}  ({dt:.1f}s)")

        np.savez(fpath, depths=np.array(DEPTHS), sre=sre_values, N=N)
        print(f"    → Saved: {fpath}")

    print(f"\n  Digital total: {time.time()-t0:.1f}s")


# ══════════════════════════════════════════════════════════════════
#  PLOTTING — 2-PANEL FIGURE
# ══════════════════════════════════════════════════════════════════

def plot_combined():
    """Create 2-panel SRE scaling figure."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

    # ── Panel (a): Digital ────────────────────────────────────────
    for i, N in enumerate(N_VALUES):
        fpath = DATA_DIR / f"sre_deterministic_N{N}.npz"
        d = np.load(fpath)
        # Haar-random baseline: M₂ ≈ N − 2
        ax1.axhline(N - 2, color=COLORS[i], ls=":", alpha=0.4, lw=1)
        ax1.plot(d["depths"], d["sre"],
                 color=COLORS[i], marker=MARKERS[i], ms=6, lw=1.8,
                 markeredgecolor="white", markeredgewidth=0.5,
                 label=f"$N = {N}$", zorder=3)

    ax1.set_xlabel(r"Circuit depth $L$")
    ax1.set_ylabel(r"Stabilizer Rényi entropy $M_2$")
    ax1.set_xlim(-0.5, max(DEPTHS) + 0.5)
    ax1.set_xticks(range(0, max(DEPTHS) + 1, 4))
    ax1.set_xticklabels([str(x) for x in range(0, max(DEPTHS) + 1, 4)])
    ax1.set_ylim(-0.5, max(N_VALUES) - 1)
    ax1.legend(loc="lower right", framealpha=0.9, fontsize=11)
    ax1.text(0.05, 0.95, r"$\mathbf{(a)}$",
             transform=ax1.transAxes, ha="left", va="top", fontsize=13)

    # ── Panel (b): Analog ─────────────────────────────────────────
    for i, N in enumerate(N_VALUES):
        fpath = DATA_DIR / f"sre_analog_N{N}.npz"
        d = np.load(fpath)
        ax2.axhline(N - 2, color=COLORS[i], ls=":", alpha=0.4, lw=1)
        ax2.plot(d["times"], d["sre"],
                 color=COLORS[i], marker=MARKERS[i], ms=6, lw=1.8,
                 markeredgecolor="white", markeredgewidth=0.5,
                 label=f"$N = {N}$", zorder=3)

    ax2.set_xlabel(r"Evolution time $t$")
    ax2.set_xlim(-0.3, 10.3)
    ax2.legend(loc="lower right", framealpha=0.9, fontsize=11)
    ax2.text(0.05, 0.95, r"$\mathbf{(b)}$",
             transform=ax2.transAxes, ha="left", va="top", fontsize=13)

    fig.tight_layout(w_pad=1.5)

    pdf_path = OUTPUT_DIR / "fig_ch5_sre_scaling.pdf"
    png_path = OUTPUT_DIR / "fig_ch5_sre_scaling.png"
    fig.savefig(pdf_path)
    fig.savefig(png_path)
    plt.close(fig)
    print(f"  → PDF: {pdf_path}")
    print(f"  → PNG: {png_path}")


# ══════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Chapter 5: SRE Scaling — 2-Panel Figure")
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--regenerate", action="store_true")
    args = parser.parse_args()

    print("=" * 65)
    print("  Chapter 5 — SRE Scaling (2-panel)")
    print("  (a) Digital Ry(π/4)+CNOT  |  (b) Analog Trotterized XX")
    print("  SAME circuit as entanglement scaling — deterministic")
    print("=" * 65)
    print(f"  Digital N: {N_VALUES}, depths: 0..{max(DEPTHS)}")
    print(f"  JAX: {jax.default_backend()}")
    print("=" * 65)

    if not args.plot_only:
        generate_digital_data(regenerate=args.regenerate)

    print("\n── Generating figure ──")
    plot_combined()
    print("\nDone.")
