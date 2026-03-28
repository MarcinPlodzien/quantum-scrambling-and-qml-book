#!/usr/bin/env python3
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Chapter 5 — Analog Scrambling: Trotterized XX Model Evolution          ║
# ║  S_A(N/2) vs system size N at fixed evolution times t                   ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
#
# ═══════════════════════════════════════════════════════════════════════════════
#  PURPOSE
# ═══════════════════════════════════════════════════════════════════════════════
#
#  This script generates panel (b) of the entanglement scaling figure,
#  and combines it with panel (a) from get_fig_ch5_entanglement_scaling.py
#  (the digital Ry+CNOT circuit).
#
#  The analog evolution uses a Trotterized simulation of the non-integrable
#  XX model with random transverse field:
#
#    Ĥ = Σ_{i=1}^{N-1} (σ̂_x^i σ̂_x^{i+1} + σ̂_y^i σ̂_y^{i+1}) + Σ_{i=1}^N h_i σ̂_x^i
#
#  where h_i ~ Uniform[-1,1].  The random field breaks integrability,
#  making this a CHAOTIC model that scrambles like the digital circuit.
#
# ═══════════════════════════════════════════════════════════════════════════════
#  TROTTER DECOMPOSITION — KEY IDEA
# ═══════════════════════════════════════════════════════════════════════════════
#
#  Instead of computing the full matrix exponential e^{-iĤt} (which
#  requires O(D²) memory, D = 2^N, prohibitive for N > 14), we decompose
#  each time step dt into a SEQUENCE OF LOCAL GATES:
#
#    e^{-iĤ dt} ≈ ∏_i e^{-i dt h_i σ̂_x^i} · ∏_{i∈even} e^{-i dt Ĥ_{XX}^{i,i+1}}
#               · ∏_{i∈odd}  e^{-i dt Ĥ_{XX}^{i,i+1}}
#
#  This is a first-order Suzuki-Trotter splitting, accurate to O(dt²).
#  For a total evolution time T, we use n_steps = T/dt Trotter steps.
#
#  THE BEAUTIFUL INSIGHT:
#    The Trotter decomposition maps Hamiltonian evolution onto EXACTLY
#    the same brickwork circuit structure as the digital Ry+CNOT circuit!
#    - 1-qubit gates:  exp(-i·dt·h_i·σ_x)  →  Rx rotation
#    - 2-qubit gates:  exp(-i·dt·(σ_x⊗σ_x + σ_y⊗σ_y))  →  XX swap gate
#    We reuse the same einsum-based gate application from the digital
#    circuit simulator, achieving O(N·2^N) per Trotter step.
#
# ═══════════════════════════════════════════════════════════════════════════════
#  TROTTER GATE DERIVATIONS
# ═══════════════════════════════════════════════════════════════════════════════
#
#  1-QUBIT: Transverse field term  h_i σ̂_x^i
#  ─────────────────────────────────────────────
#  exp(-i·dt·h·σ_x) = cos(dt·h)·I − i·sin(dt·h)·σ_x
#
#  In matrix form:
#    ┌                           ┐
#    │  cos(dt·h)   -i·sin(dt·h) │
#    │ -i·sin(dt·h)  cos(dt·h)   │
#    └                           ┘
#  This is simply an Rx(2·dt·h) rotation!
#
#
#  2-QUBIT: XX interaction  σ̂_x^i σ̂_x^{i+1} + σ̂_y^i σ̂_y^{i+1}
#  ───────────────────────────────────────────────────────────────
#  In the computational basis {|00⟩, |01⟩, |10⟩, |11⟩}:
#
#    σ_x⊗σ_x + σ_y⊗σ_y = 2·(|01⟩⟨10| + |10⟩⟨01|)
#
#  This is a "flip-flop" or "partial swap" interaction.
#  The matrix exponential is:
#
#    ┌                                           ┐
#    │  1       0              0           0      │
#    │  0    cos(2·dt)    -i·sin(2·dt)     0      │
#    │  0   -i·sin(2·dt)   cos(2·dt)      0      │
#    │  0       0              0           1      │
#    └                                           ┘
#
#  The |00⟩ and |11⟩ subspaces are unaffected (they have the same
#  spin on both sites, so there is nothing to flip).
#  The |01⟩ ↔ |10⟩ subspace undergoes coherent oscillation.
#
# ═══════════════════════════════════════════════════════════════════════════════

import os, sys, time as timer, argparse
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

CDT = jnp.complex128

# ══════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════════════

SYSTEM_SIZES = list(range(6, 21, 2))      # N = 6, 8, ..., 20
EVOLUTION_TIMES = [0, 2, 4, 6, 8, 10]     # fixed times t
DT = 0.05                                  # Trotter step size
SEED = 42                                  # reproducible random fields
DATA_DIR = Path("data/ch5")
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "figures" / "ch5"
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════
#  EINSUM SUBSCRIPT GENERATION  (same as digital circuit scripts)
# ══════════════════════════════════════════════════════════════════
#
#  The statevector |ψ⟩ is stored as a rank-N tensor of shape (2,)*N.
#  These functions generate the einsum strings for applying 1-qubit
#  and 2-qubit gates via tensor contraction:
#
#    1-qubit:  G[j'] ψ[..., j, ...] → ψ'[..., j', ...]
#    2-qubit:  G[c', t', c, t] ψ[..., c, ..., t, ...] → ψ'[..., c', ..., t', ...]
#
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
#  TROTTER GATE CONSTRUCTION
# ══════════════════════════════════════════════════════════════════
#
#  For a Trotter step of size dt, we build:
#
#    1. N single-qubit gates:  Rx_i = exp(-i·dt·h_i·σ_x)
#       Each depends on the local random field h_i.
#
#    2. One universal 2-qubit gate: XX(dt) = exp(-i·dt·(σ_x⊗σ_x + σ_y⊗σ_y))
#       This is the SAME for all pairs (only the coupling constant matters).
#
# ══════════════════════════════════════════════════════════════════

def _build_rx_gates(h_fields, dt):
    """
    Build N single-qubit Rx gates for the transverse field term.

    exp(-i·dt·h·σ_x) = [[cos(dt·h), -i·sin(dt·h)],
                         [-i·sin(dt·h), cos(dt·h)]]

    Returns a list of N (2,2) JAX arrays.
    """
    gates = []
    for h in h_fields:
        c = jnp.cos(dt * h)
        s = jnp.sin(dt * h)
        gates.append(jnp.array([[c, -1j*s], [-1j*s, c]], dtype=CDT))
    return gates


def _build_xx_gate(dt):
    """
    Build the 2-qubit XX gate for one Trotter step.

    exp(-i·dt·(σ_x⊗σ_x + σ_y⊗σ_y))

    In the {|00⟩, |01⟩, |10⟩, |11⟩} basis:
      |00⟩ → |00⟩                   (unaffected)
      |01⟩ → cos(2dt)|01⟩ - i·sin(2dt)|10⟩  (flip-flop)
      |10⟩ → -i·sin(2dt)|01⟩ + cos(2dt)|10⟩
      |11⟩ → |11⟩                   (unaffected)

    Returns a (2,2,2,2) JAX array for einsum application.
    """
    c = jnp.cos(2.0 * dt)
    s = jnp.sin(2.0 * dt)

    # Build 4×4 matrix, then reshape to (2,2,2,2) tensor
    gate = jnp.array([
        [1,     0,       0,    0],
        [0,     c,    -1j*s,   0],
        [0,  -1j*s,      c,   0],
        [0,     0,       0,    1],
    ], dtype=CDT).reshape(2, 2, 2, 2)
    return gate


# ══════════════════════════════════════════════════════════════════
#  TROTTERIZED TIME EVOLUTION
# ══════════════════════════════════════════════════════════════════
#
#  One Trotter step of the first-order Suzuki–Trotter decomposition:
#
#    e^{-iĤ·dt} ≈ [∏_i Rx_i(dt·h_i)] · [∏_{i∈even} XX_{i,i+1}(dt)]
#                                        · [∏_{i∈odd}  XX_{i,i+1}(dt)]
#
#  This has EXACTLY the same brickwork structure as the digital circuit:
#    Ry gates      →  Rx gates (field-dependent)
#    CNOT gates    →  XX gates (interaction-dependent)
#
#  For total time T, we apply n_steps = round(T/dt) Trotter steps.
#
# ══════════════════════════════════════════════════════════════════

def evolve_xx_trotter(N, t_total, h_fields, dt=DT):
    """
    Evolve GHZ state under XX Hamiltonian via Trotterization.

    Parameters
    ----------
    N : int — number of qubits
    t_total : float — total evolution time
    h_fields : array of shape (N,) — random transverse field strengths
    dt : float — Trotter step size

    Returns
    -------
    psi : array of shape (2^N,) — evolved statevector
    """
    if t_total == 0:
        D = 1 << N
        psi = jnp.zeros(D, dtype=CDT)
        psi = psi.at[0].set(1/jnp.sqrt(2.0))
        psi = psi.at[D-1].set(1/jnp.sqrt(2.0))
        return psi

    # Number of Trotter steps
    n_steps = max(1, round(t_total / dt))

    # Precompute gates and einsum subscripts (done once, reused per step)
    sub1 = _subs_1q(N)
    sub2 = _subs_2q(N)
    rx_gates = _build_rx_gates(h_fields, dt)     # N single-qubit gates
    xx_gate  = _build_xx_gate(dt)                 # one universal 2-qubit gate
    even = [(j, j+1) for j in range(0, N-1, 2)]  # even-odd brickwork
    odd  = [(j, j+1) for j in range(1, N-1, 2)]

    # GHZ initial state
    D = 1 << N
    psi = jnp.zeros(D, dtype=CDT)
    psi = psi.at[0].set(1/jnp.sqrt(2.0))
    psi = psi.at[D-1].set(1/jnp.sqrt(2.0))

    # ── Trotter time-stepping ────────────────────────────────────
    for step in range(n_steps):
        p = psi.reshape((2,)*N)

        # 1. Apply single-qubit field gates: exp(-i·dt·h_i·σ_x)
        for j in range(N):
            p = jnp.einsum(sub1[j], rx_gates[j], p)

        # 2. Apply XX gates on even pairs
        for c, t in even:
            p = jnp.einsum(sub2[(c, t)], xx_gate, p)

        # 3. Apply XX gates on odd pairs
        for c, t in odd:
            p = jnp.einsum(sub2[(c, t)], xx_gate, p)

        psi = p.reshape(D)

    return psi


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

def generate_data(regenerate=False):
    t0 = timer.time()

    sa_file = DATA_DIR / "entropy_vs_N_analog.npz"
    Ns = np.array(SYSTEM_SIZES)
    times = np.array(EVOLUTION_TIMES, dtype=float)
    sa = np.full((len(Ns), len(times)), np.nan)
    sa_page = np.full(len(Ns), np.nan)

    # Load partial data if available
    if sa_file.exists():
        old = np.load(sa_file)
        if "sa" in old and old["sa"].shape == sa.shape and not regenerate:
            sa = old["sa"].copy()
            sa_page = old["sa_page"].copy()
            n_done = np.count_nonzero(~np.isnan(sa))
            if n_done == sa.size:
                print(f"  Cached: {sa_file}")
                return
            print(f"  Loaded partial data: {n_done}/{sa.size} values")

    print(f"\n  S_A(N/2) vs N at times t = {list(times)}")
    print(f"  Trotter step dt = {DT},  system sizes: {list(Ns)}\n")

    for i, N in enumerate(Ns):
        k = N // 2

        # Page formula
        if np.isnan(sa_page[i]):
            sa_page[i] = page_entropy(N, k)

        # Random transverse fields (reproducible per N)
        rng = np.random.default_rng(SEED + N)
        h_fields = rng.uniform(-1, 1, size=N)

        for j, t in enumerate(times):
            if not np.isnan(sa[i, j]):
                continue

            n_steps = max(1, round(t / DT)) if t > 0 else 0
            print(f"    N={N:2d}, t={t:4.1f} ({n_steps:3d} steps): ",
                  end="", flush=True)
            t1 = timer.time()

            psi = evolve_xx_trotter(N, t, h_fields, dt=DT)
            sa[i, j] = entanglement_entropy(psi, N, k)
            dt_wall = timer.time() - t1
            print(f"S/ln2 = {sa[i,j]/np.log(2):.3f}  ({dt_wall:.1f}s)")

            # Incremental save
            np.savez(sa_file, Ns=Ns, times=times, sa=sa, sa_page=sa_page)

    print(f"\n  → {sa_file} (complete)")
    print(f"  Total: {timer.time()-t0:.1f}s")


# ══════════════════════════════════════════════════════════════════
#  PLOTTING — 2-panel: digital (a) + analog (b)
# ══════════════════════════════════════════════════════════════════
#
#  Combines the digital circuit data (from get_fig_ch5_entanglement_
#  scaling.py) with the analog Hamiltonian data into a single
#  2-panel publication figure.
#
# ══════════════════════════════════════════════════════════════════

def plot_figure():
    print("  Plotting combined 2-panel figure...")

    # Load digital data
    dig_file = DATA_DIR / "entropy_vs_N_depths.npz"
    if not dig_file.exists():
        print("  WARNING: digital data not found, run get_fig_ch5_entanglement_scaling.py first")
        print("  Plotting analog-only figure...")
        plot_analog_only()
        return

    data_dig = np.load(dig_file)
    Ns_d     = data_dig["Ns"]
    depths   = data_dig["depths"]
    sa_d     = data_dig["sa"]
    page_d   = data_dig["sa_page"]

    # Load analog data
    data_ana = np.load(DATA_DIR / "entropy_vs_N_analog.npz")
    Ns_a     = data_ana["Ns"]
    times    = data_ana["times"]
    sa_a     = data_ana["sa"]
    page_a   = data_ana["sa_page"]

    LN2 = np.log(2)

    # ── Colour palettes ───────────────────────────────────────────
    colors_d = {0: "#333333", 2: "#2C7FB8", 4: "#7FBC41",
                8: "#FD8D3C", 10: "#D95F02", 16: "#9E4A9C"}
    markers_d = {0: "o", 2: "o", 4: "^", 8: "s", 10: "D", 16: "p"}

    colors_a = {0: "#333333", 2: "#2C7FB8", 4: "#7FBC41",
                6: "#FD8D3C", 8: "#9E4A9C"}
    markers_a = {0: "o", 2: "o", 4: "^", 6: "s", 8: "p"}

    fig, (ax_d, ax_a) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    fig.subplots_adjust(wspace=0.08)

    y_max = max(Ns_d[-1], Ns_a[-1]) // 2 + 1

    # ════════════════════════════════════════════════════════════════
    #  Panel (a): Digital circuit — Ry+CNOT brickwork
    # ════════════════════════════════════════════════════════════════

    ax_d.plot(Ns_d, page_d / LN2, "--", color="gray", lw=1.5, alpha=0.6,
              label="Page value", zorder=1)

    for j, L in enumerate(depths):
        L = int(L)
        c = colors_d.get(L, "gray")
        m = markers_d.get(L, "o")
        fc = "white" if L == 0 else c
        ax_d.plot(Ns_d, sa_d[:, j] / LN2, f"-{m}", color=c, lw=1.8,
                  markersize=7, markerfacecolor=fc,
                  markeredgecolor=c, markeredgewidth=1.5,
                  zorder=3, label=f"$L = {L}$")

    j_N = len(depths)
    ax_d.plot(Ns_d, sa_d[:, j_N] / LN2, "x--", color="#E63946", lw=2.2,
              markersize=9, markeredgewidth=2.5,
              zorder=5, label=r"$L = N$")

    ax_d.set_xlabel(r"System size $N$")
    ax_d.set_ylabel(r"Entanglement entropy $S_{\mathrm{vN}}\,/\,\ln 2$")
    ax_d.set_xlim(SYSTEM_SIZES[0] - 0.5, SYSTEM_SIZES[-1] + 0.5)
    ax_d.set_ylim(0, y_max)
    ax_d.set_xticks(SYSTEM_SIZES)
    ax_d.legend(loc="upper left", framealpha=0.9, fontsize=11)
    ax_d.text(0.97, 0.95, r"$\mathbf{(a)}$", transform=ax_d.transAxes,
              fontsize=14, va="top", ha="right")

    # ════════════════════════════════════════════════════════════════
    #  Panel (b): Analog evolution — XX + random field (Trotterized)
    # ════════════════════════════════════════════════════════════════

    ax_a.plot(Ns_a, page_a / LN2, "--", color="gray", lw=1.5, alpha=0.6,
              label="Page value", zorder=1)

    for j, t in enumerate(times[:-1]):
        t_int = int(t)
        c = colors_a.get(t_int, "gray")
        m = markers_a.get(t_int, "o")
        fc = "white" if t_int == 0 else c
        ax_a.plot(Ns_a, sa_a[:, j] / LN2, f"-{m}", color=c, lw=1.8,
                  markersize=7, markerfacecolor=fc,
                  markeredgecolor=c, markeredgewidth=1.5,
                  zorder=3, label=f"$t = {t_int}$")

    t_last = int(times[-1])
    ax_a.plot(Ns_a, sa_a[:, -1] / LN2, "x--", color="#E63946", lw=2.2,
              markersize=9, markeredgewidth=2.5,
              zorder=5, label=f"$t = {t_last}$")

    ax_a.set_xlabel(r"System size $N$")
    ax_a.set_xlim(SYSTEM_SIZES[0] - 0.5, SYSTEM_SIZES[-1] + 0.5)
    ax_a.set_ylim(0, y_max)
    ax_a.set_xticks(SYSTEM_SIZES)
    ax_a.legend(loc="upper left", framealpha=0.9, fontsize=11)
    ax_a.text(0.97, 0.95, r"$\mathbf{(b)}$", transform=ax_a.transAxes,
              fontsize=14, va="top", ha="right")

    # ── Save ─────────────────────────────────────────────────────
    outpath = OUTPUT_DIR / "fig_ch5_entanglement_scaling.pdf"
    fig.savefig(outpath, bbox_inches="tight")
    fig.savefig(outpath.with_suffix(".png"), bbox_inches="tight")
    print(f"    → PDF: {outpath}")
    print(f"    → PNG: {outpath.with_suffix('.png')}")
    plt.close(fig)


def plot_analog_only():
    """Fallback: plot only the analog panel if digital data is unavailable."""
    data = np.load(DATA_DIR / "entropy_vs_N_analog.npz")
    Ns, times, sa, sa_page = data["Ns"], data["times"], data["sa"], data["sa_page"]
    LN2 = np.log(2)

    colors = {0: "#7FCDBB", 2: "#2C7FB8", 4: "#7FBC41",
              6: "#FD8D3C", 8: "#9E4A9C"}
    markers = {0: "o", 2: "o", 4: "^", 6: "s", 8: "p"}

    fig, ax = plt.subplots(1, 1, figsize=(6.5, 5))
    ax.plot(Ns, sa_page / LN2, "--", color="gray", lw=1.5, alpha=0.6, label="Page value")

    for j, t in enumerate(times[:-1]):
        t_int = int(t)
        c = colors.get(t_int, "gray")
        m = markers.get(t_int, "o")
        fc = "white" if t_int == 0 else c
        ax.plot(Ns, sa[:, j] / LN2, f"-{m}", color=c, lw=1.8,
                markersize=7, markerfacecolor=fc, markeredgecolor=c,
                markeredgewidth=1.5, label=f"$t = {t_int}$")

    t_last = int(times[-1])
    ax.plot(Ns, sa[:, -1] / LN2, "x--", color="#E63946", lw=2.2,
            markersize=9, markeredgewidth=2.5, label=f"$t = {t_last}$")

    ax.set_xlabel(r"System size $N$")
    ax.set_ylabel(r"Entanglement entropy $S_{\mathrm{vN}}\,/\,\ln 2$")
    ax.set_xlim(SYSTEM_SIZES[0] - 0.5, SYSTEM_SIZES[-1] + 0.5)
    ax.set_ylim(0, SYSTEM_SIZES[-1] // 2 + 1)
    ax.set_xticks(SYSTEM_SIZES)
    ax.legend(loc="upper left", framealpha=0.9, fontsize=9)

    outpath = OUTPUT_DIR / "fig_ch5_entanglement_analog.pdf"
    fig.savefig(outpath, bbox_inches="tight")
    fig.savefig(outpath.with_suffix(".png"), bbox_inches="tight")
    print(f"    → PDF: {outpath}")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Chapter 5: Analog XX Model — Trotterized Evolution")
    parser.add_argument("--regenerate", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()

    print("=" * 65)
    print("  Chapter 5 — Analog XX Model: Trotterized Evolution")
    print("  S_A(N/2) at fixed evolution times")
    print("=" * 65)
    print(f"  System sizes: {SYSTEM_SIZES}")
    print(f"  Evolution times: {EVOLUTION_TIMES}")
    print(f"  Trotter dt = {DT} → steps: {[round(t/DT) for t in EVOLUTION_TIMES if t > 0]}")
    print(f"  JAX: {jax.default_backend()}")
    print("=" * 65)

    if args.plot_only:
        if not (DATA_DIR / "entropy_vs_N_analog.npz").exists():
            print("ERROR: data missing. Run without --plot-only first.")
            sys.exit(1)
    else:
        generate_data(regenerate=args.regenerate)

    print("\n── Figure ──")
    plot_figure()
    print(f"\nDone → {OUTPUT_DIR}")
