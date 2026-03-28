#!/usr/bin/env python3
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Chapter 5 — Entanglement Entropy vs System Size                        ║
# ║  Figure: S_A(N/2) / ln2  vs  N   at  fixed circuit depths               ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
#
# ═══════════════════════════════════════════════════════════════════════════════
#  PURPOSE
# ═══════════════════════════════════════════════════════════════════════════════
#
#  This script generates a single-panel publication figure that illustrates
#  the AREA LAW → VOLUME LAW transition as a function of system size.
#
#  We apply the scrambling unitary Û(L) = ∏ Û^(ℓ), where each layer is:
#
#    Û^(ℓ) = ∏_i R̂_y^(i)(π/4) · ∏_{i∈even} CNOT_{i,i+1} · ∏_{i∈odd} CNOT_{i,i+1}
#
#  For several FIXED circuit depths L = 0, 2, 4, 8, 16  and  L = N:
#    • Fixed L:    entropy saturates for large N → area-law-LIKE behaviour
#    • L = N:      entropy grows linearly with N → VOLUME LAW
#
#  The L = 0 curve (GHZ initial state) stays flat at S = ln 2 for all N.
#  The L = N curve (red, dashed) approaches the Page value, confirming
#  that the scrambling circuit produces near-Haar-random entanglement.
#
# ═══════════════════════════════════════════════════════════════════════════════
#  KEY PHYSICAL MESSAGE
# ═══════════════════════════════════════════════════════════════════════════════
#
#  The figure demonstrates that:
#
#    ┌──────────────────────────────────────────────────────────────────────┐
#    │  Only circuits whose depth SCALES with system size (L ∝ N) can     │
#    │  achieve volume-law entanglement.  Fixed-depth circuits produce    │
#    │  at most O(L) entanglement, independent of N.                     │
#    │                                                                    │
#    │  This "light cone" argument explains why scrambling requires       │
#    │  depth L = Ω(N): entanglement must propagate across the chain.    │
#    └──────────────────────────────────────────────────────────────────────┘
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
    "font.size": 12,
    "axes.labelsize": 14,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

import jax
import jax.numpy as jnp
from jax import random

CDT = jnp.complex128
FDT = jnp.float64

# ══════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════════════
#
#  System sizes: N = 6, 8, 10, ..., 20  (even, for clean equal bipartition)
#  Fixed depths: L = 0, 2, 4, 8, 16     (constant depth, area-law regime)
#  Scaling depth: L = N                  (volume-law regime)
#
# ══════════════════════════════════════════════════════════════════

SYSTEM_SIZES = list(range(6, 21, 2))      # N = 6, 8, 10, ..., 20
FIXED_DEPTHS = [0, 2, 4, 8, 16]           # constant circuit depths
DATA_DIR = Path("data/ch5")
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "figures" / "ch5"
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════
#  GATE DEFINITIONS
# ══════════════════════════════════════════════════════════════════

_RY = jnp.array([[jnp.cos(jnp.pi/8), -jnp.sin(jnp.pi/8)],
                  [jnp.sin(jnp.pi/8),  jnp.cos(jnp.pi/8)]], dtype=CDT)
CNOT_T = jnp.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]],
                    dtype=CDT).reshape(2,2,2,2)

# ══════════════════════════════════════════════════════════════════
#  EINSUM SUBSCRIPT GENERATION
# ══════════════════════════════════════════════════════════════════

_AX = "abcdefghijklmnopqrstu"

def _subs_1q(N):
    """Einsum subscripts for 1-qubit gates on each of N qubits."""
    subs = []
    for j in range(N):
        ax = list(_AX[:N])
        out = list(ax)
        new = chr(ord('A') + j)
        out[j] = new
        subs.append(f"{new}{ax[j]},{''.join(ax)}->{''.join(out)}")
    return subs

def _subs_2q(N):
    """Einsum subscripts for 2-qubit gates on all qubit pairs."""
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
#  ENTANGLEMENT ENTROPY
# ══════════════════════════════════════════════════════════════════

def entanglement_entropy(psi, N, k):
    """Von Neumann entropy S_A for subsystem A = first k qubits."""
    k_eff = min(k, N - k)
    m = 1 << k_eff
    n = (1 << N) >> k_eff
    if k <= N - k:
        C = psi.reshape(m, n)
    else:
        C = psi.reshape(n, m).T
    rho = jnp.einsum('ab,cb->ac', C, jnp.conj(C))
    eigs = jnp.clip(jnp.linalg.eigvalsh(rho), 1e-30, None)
    return float(-jnp.sum(eigs * jnp.log(eigs)))


# ══════════════════════════════════════════════════════════════════
#  CIRCUIT SIMULATOR — Ry(π/4) + CNOT brickwork
# ══════════════════════════════════════════════════════════════════
#
#  This is the same scrambling unitary from Chapter 5, Eq. (12):
#
#    Û^(ℓ) = ∏_i R̂_y^(i)(π/4) · ∏_{i∈e} CNOT_{i,i+1} · ∏_{i∈o} CNOT_{i,i+1}
#
#  Starting from the GHZ state (|0...0⟩ + |1...1⟩)/√2.
#
# ══════════════════════════════════════════════════════════════════

def _evolve_ry_cnot(N, L):
    """Ry(π/4)+CNOT brickwork for L layers, starting from GHZ."""
    D = 1 << N
    sub1, sub2 = _subs_1q(N), _subs_2q(N)
    even = [(j, j+1) for j in range(0, N-1, 2)]
    odd  = [(j, j+1) for j in range(1, N-1, 2)]

    # GHZ initial state
    psi = jnp.zeros(D, dtype=CDT)
    psi = psi.at[0].set(1/jnp.sqrt(2.0))
    psi = psi.at[D-1].set(1/jnp.sqrt(2.0))

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
#  PAGE'S FORMULA
# ══════════════════════════════════════════════════════════════════

def page_entropy(N, k):
    """Exact Page entropy for N qubits, subsystem of k qubits."""
    d_A = 1 << min(k, N - k)
    d_B = 1 << max(k, N - k)
    S = sum(1.0/j for j in range(d_B + 1, d_A * d_B + 1))
    S -= (d_A - 1) / (2.0 * d_B)
    return S


# ══════════════════════════════════════════════════════════════════
#  DATA GENERATION — WITH INCREMENTAL SAVES
# ══════════════════════════════════════════════════════════════════
#
#  For each (N, L) pair, we evolve the Ry+CNOT brickwork circuit
#  and compute S_A at the equal bipartition k = N/2.
#
#  The data is stored as a 2D array: rows = system sizes, cols = depths.
#  We save after each (N, L) computation for crash protection.
#
# ══════════════════════════════════════════════════════════════════

def generate_data(regenerate=False):
    t0 = time.time()

    sa_file = DATA_DIR / "entropy_vs_N_depths.npz"

    Ns = np.array(SYSTEM_SIZES)
    # All depths to compute: fixed depths + L=N for each N
    all_depths = sorted(set(FIXED_DEPTHS))  # we handle L=N separately

    # Data matrix: rows = N, cols = fixed depths + L=N
    n_depths = len(all_depths) + 1  # +1 for L=N
    sa = np.full((len(Ns), n_depths), np.nan)
    sa_page = np.full(len(Ns), np.nan)

    # Load partial data if available
    if sa_file.exists():
        old = np.load(sa_file)
        if "sa" in old and old["sa"].shape == sa.shape:
            sa = old["sa"].copy()
            sa_page = old["sa_page"].copy()
            n_done = np.count_nonzero(~np.isnan(sa))
            print(f"  Loaded partial data: {n_done} values computed")
        elif not regenerate:
            print(f"  Cached: {sa_file}")
            return

    print(f"\n  S_A(N/2) vs N at depths L = {all_depths} + [L=N]")
    print(f"  System sizes: {list(Ns)}\n")

    for i, N in enumerate(Ns):
        k = N // 2

        # Page formula
        if np.isnan(sa_page[i]):
            sa_page[i] = page_entropy(N, k)

        # Fixed depths
        for j, L in enumerate(all_depths):
            if not np.isnan(sa[i, j]):
                continue  # already computed

            print(f"    N={N:2d}, L={L:2d}: ", end="", flush=True)
            t1 = time.time()

            if L == 0:
                # GHZ state: S_A = ln 2 for any bipartition
                D = 1 << N
                psi = jnp.zeros(D, dtype=CDT)
                psi = psi.at[0].set(1/jnp.sqrt(2.0))
                psi = psi.at[D-1].set(1/jnp.sqrt(2.0))
            else:
                psi = _evolve_ry_cnot(N, L)

            sa[i, j] = entanglement_entropy(psi, N, k)
            dt = time.time() - t1
            print(f"S/ln2 = {sa[i,j]/np.log(2):.3f}  ({dt:.1f}s)")

            # Incremental save
            np.savez(sa_file, Ns=Ns, depths=np.array(all_depths),
                     sa=sa, sa_page=sa_page)

        # L = N (scaling depth — last column)
        j_N = len(all_depths)  # last column index
        if np.isnan(sa[i, j_N]):
            print(f"    N={N:2d}, L=N={N:2d}: ", end="", flush=True)
            t1 = time.time()
            psi = _evolve_ry_cnot(N, N)
            sa[i, j_N] = entanglement_entropy(psi, N, k)
            dt = time.time() - t1
            print(f"S/ln2 = {sa[i,j_N]/np.log(2):.3f}  ({dt:.1f}s)")

            # Incremental save
            np.savez(sa_file, Ns=Ns, depths=np.array(all_depths),
                     sa=sa, sa_page=sa_page)

    print(f"\n  → {sa_file} (complete)")
    print(f"  Total: {time.time()-t0:.1f}s")


# ══════════════════════════════════════════════════════════════════
#  PLOTTING
# ══════════════════════════════════════════════════════════════════
#
#  Single panel matching the reference style:
#    • L = 0       → open circles, flat at S = 1 (area law)
#    • L = 2,4,8   → increasing curves that saturate
#    • L = 16      → near-volume-law for small N, saturates for large N
#    • L = N       → red X markers with dashed line (volume law)
#    • Page value  → gray dashed reference
#
# ══════════════════════════════════════════════════════════════════

def plot_figure():
    print("  Plotting figure...")

    data = np.load(DATA_DIR / "entropy_vs_N_depths.npz")
    Ns = data["Ns"]
    depths = data["depths"]
    sa = data["sa"]            # shape: (len(Ns), len(depths)+1)
    sa_page = data["sa_page"]

    LN2 = np.log(2)

    # ── Colour palette matching reference ─────────────────────────
    # L=0: light blue (open circles), L=2: blue, L=4: green,
    # L=8: orange, L=16: purple, L=N: red (dashed, X markers)
    colors = {
        0:  "#7FCDBB",   # light teal
        2:  "#2C7FB8",   # blue
        4:  "#7FBC41",   # green
        8:  "#FD8D3C",   # orange
        16: "#9E4A9C",   # purple
    }
    markers = {
        0:  "o",
        2:  "o",
        4:  "^",
        8:  "s",
        16: "p",
    }

    fig, ax = plt.subplots(1, 1, figsize=(6.5, 5))

    # Page value reference (gray dashed)
    ax.plot(Ns, sa_page / LN2, "--", color="gray", lw=1.5, alpha=0.6,
            label="Page value", zorder=1)

    # Fixed-depth curves
    for j, L in enumerate(depths):
        L = int(L)
        c = colors.get(L, "gray")
        m = markers.get(L, "o")
        facecolor = "white" if L == 0 else c
        ax.plot(Ns, sa[:, j] / LN2, f"-{m}", color=c, lw=1.8,
                markersize=7, markerfacecolor=facecolor,
                markeredgecolor=c, markeredgewidth=1.5,
                zorder=3, label=f"$L = {L}$")

    # L = N curve (volume law — red dashed with X markers)
    j_N = len(depths)
    ax.plot(Ns, sa[:, j_N] / LN2, "x--", color="#E63946", lw=2.2,
            markersize=9, markeredgewidth=2.5,
            zorder=5, label=r"$L = N$")

    ax.set_xlabel(r"System size $N$")
    ax.set_ylabel(r"Entanglement entropy $S_{\mathrm{vN}}\,/\,\ln 2$")
    ax.set_xlim(SYSTEM_SIZES[0] - 0.5, SYSTEM_SIZES[-1] + 0.5)
    ax.set_ylim(0, SYSTEM_SIZES[-1] // 2 + 1)
    ax.set_xticks(SYSTEM_SIZES)
    ax.legend(loc="upper left", framealpha=0.9, fontsize=10)

    # ── Save ─────────────────────────────────────────────────────
    outpath = OUTPUT_DIR / "fig_ch5_entanglement_scaling.pdf"
    fig.savefig(outpath, bbox_inches="tight")
    fig.savefig(outpath.with_suffix(".png"), bbox_inches="tight")
    print(f"    → PDF: {outpath}")
    print(f"    → PNG: {outpath.with_suffix('.png')}")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Chapter 5: Entanglement Entropy vs System Size")
    parser.add_argument("--regenerate", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()

    print("=" * 65)
    print("  Chapter 5 — Entanglement Entropy vs System Size")
    print("  S_A(N/2) at fixed depths L = 0, 2, 4, 8, 16, N")
    print("=" * 65)
    print(f"  System sizes: {SYSTEM_SIZES}")
    print(f"  Fixed depths: {FIXED_DEPTHS}")
    print(f"  JAX: {jax.default_backend()}")
    print("=" * 65)

    if args.plot_only:
        sa_file = DATA_DIR / "entropy_vs_N_depths.npz"
        if not sa_file.exists():
            print("ERROR: data missing. Run without --plot-only first.")
            sys.exit(1)
    else:
        generate_data(regenerate=args.regenerate)

    print("\n── Figure ──")
    plot_figure()
    print(f"\nDone → {OUTPUT_DIR}")
