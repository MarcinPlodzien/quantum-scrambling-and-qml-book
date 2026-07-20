#!/usr/bin/env python3
"""
get_fig_ch7_trainability_scaling.py
===================================
Trainability at scale for Chapter 7 (Quantum Machine Learning): the SAME
Haar / 2-design typicality that flattens variational cost landscapes into
barren plateaus ALSO concentrates quantum-kernel entries.  Both obstructions
are the second-moment overlap scale 2^{-N}, so both variational optimization
and kernel methods degrade exponentially in the qubit number N.

Produces Fig. (ch7) -- figures/ch7/fig_ch7_trainability_scaling.pdf

    Panel (a): BARREN PLATEAU.  Variance of one gradient component of a cost
               function, over random hardware-efficient circuits, versus N.
               A GLOBAL cost (the traceless Z-string O = Z_1 ... Z_N) has a
               variance that collapses exponentially, tracking the 2^{-N} guide:
               the canonical barren plateau.  A LOCAL cost (a single-site O = Z_q,
               same circuits, same differentiated parameter) has a variance that
               decays only polynomially and stays trainable over this range.

    Panel (b): QUANTUM-KERNEL CONCENTRATION.  For an expressive IQP feature map,
               the mean off-diagonal fidelity kernel k(x_i,x_j)=|<phi_i|phi_j>|^2
               tracks 2^{-N}, and its standard deviation shrinks with it: every
               off-diagonal entry is pinned near 2^{-N}, so the Gram matrix
               approaches the identity.  Resolving k_ij above shot noise then
               costs ~O(2^N) measurements.

WHY THE SAME 2^{-N}.  A random parametrized circuit deep enough to form an
approximate unitary 2-design, and an expressive feature map whose data-dependent
unitary looks 2-design-like, both realize the same second-moment (Haar) statistics.
The Haar overlap of two independent pure states in dimension D = 2^N has mean 1/D,
so kernel entries concentrate at 2^{-N} (panel b).  A traceless global observable
O = Z_1 ... Z_N has <O> = 0 under the same statistics, and its second moment
<O>^2 ~ 1/D, so the gradient of the cost C_g = <psi|O|psi> has variance ~2^{-N}
(panel a): the canonical barren-plateau scaling, the SAME exponent as the kernel.
Both panels therefore sit on one 2^{-N} line.  The cure in both cases is the same:
break the typicality with locality (a low-weight cost) or a controlled,
less-than-fully-expressive feature map, so the relevant scale is poly(N) rather
than 2^{-N}.

PANEL (a) ANSATZ.  A hardware-efficient ansatz on N qubits: L = N layers, each a
wall of single-qubit R_y(theta) rotations at independent uniform random angles
followed by a fixed brickwork of CZ gates (even/odd bonds on alternating layers).
Depth L = O(N) is the 1D brickwork 2-design scale.  The exact statevector is real
(R_y and CZ are real), evolved by direct index manipulation; no external
quantum-computing library.  One gradient component is taken by the parameter-shift
rule dC/dtheta = 1/2 [C(theta+pi/2) - C(theta-pi/2)] on the middle-layer rotation of
the middle qubit.  The global and local costs are read from the SAME circuits and
the SAME differentiated parameter, so only the observable differs: global
O = Z_1 ... Z_N versus local O = Z_q on the driven (middle) qubit.  The local
observable reads the same qubit the parameter drives, keeping the gradient inside
the observable's causal light cone (a Z on a far qubit would give a trivially zero,
not merely polynomial, gradient).

PANEL (b) FEATURE MAP.  An expressive IQP-style encoding U(x) = (U_Z(x) H^{oxN})^L
with L = 4 re-uploading layers on N qubits, one feature per qubit, with single-Z and
ZZ phases theta_b = sum_k f_k s^k_b + sum_{k<l} f_k f_l s^k_b s^l_b, s^k_b = (-1)^{bit}.
U_Z is diagonal so its action is an elementwise phase multiply; H^{oxN} is a fast
Walsh-Hadamard transform, fully vectorized over a batch of random data points.

Exact statevector throughout, numpy only, seeded RNG.  All numerics are cached in
data/ch7/fig_ch7_trainability_scaling.npz.
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_style import apply_book_style, load_or_compute, panel_label

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR.parent / "figures" / "ch7"
DATA_DIR = SCRIPT_DIR / "data" / "ch7"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Sweep and sampling (fixed; the RNG seeds make the run reproducible).
# Statevector to N = 12 is comfortable; N = 2..12 keeps the whole figure to a
# couple of minutes.  Samples are chosen for a smooth variance/mean estimate.
# ---------------------------------------------------------------------------
N_MIN, N_MAX = 2, 12
N_CIRCUITS = 800        # random circuits per N for the gradient-variance estimate
N_POINTS = 500          # random data points per N for the kernel statistics
N_PAIRS = 4000          # random off-diagonal pairs drawn from those points
IQP_LAYERS = 4          # depth L of the expressive IQP feature map
IQP_SCALE = 1.5         # data-to-phase scaling; tuned so overlaps hit the Haar 2^{-N}
SEED_BP = 20250718
SEED_KERNEL = 7

C_GLOBAL = "#c1121f"    # global cost -- the barren plateau
C_LOCAL = "#1f6fc4"     # local cost -- stays trainable
C_KERNEL = "#1f6fc4"    # kernel mean
C_KERNEL_STD = "#e8850c"  # kernel std. dev.
C_GUIDE = "0.35"        # 2^{-N} reference


# ===========================================================================
# Panel (a): hardware-efficient ansatz and parameter-shift gradient variance.
# The statevector is real, held as (S, 2^N) over the S sampled circuits at once.
# ===========================================================================
def apply_ry_wall(psi, N, angles):
    """Apply R_y(angles[:, q]) to every qubit q, batched over the S circuits."""
    for q in range(N):
        left, right = 1 << q, 1 << (N - 1 - q)
        v = psi.reshape(-1, left, 2, right)
        c = np.cos(angles[:, q] / 2)[:, None, None]
        s = np.sin(angles[:, q] / 2)[:, None, None]
        a0, a1 = v[:, :, 0, :], v[:, :, 1, :]
        out = np.empty_like(v)
        out[:, :, 0, :] = c * a0 - s * a1
        out[:, :, 1, :] = s * a0 + c * a1
        psi = out.reshape(psi.shape)
    return psi


def cz_mask(N, a, b):
    """Diagonal of CZ on qubits (a, b): -1 where both bits are 1, else +1."""
    idx = np.arange(1 << N)
    both = ((idx >> (N - 1 - a)) & 1) & ((idx >> (N - 1 - b)) & 1)
    return np.where(both, -1.0, 1.0)


def brick_pairs(N, layer):
    """Brickwork CZ bonds: even bonds on even layers, odd bonds on odd layers."""
    start = layer % 2
    return [(i, i + 1) for i in range(start, N - 1, 2)]


def run_ansatz(N, L, angles, diff_layer, diff_qubit, shift, masks):
    """Statevectors of the R_y/CZ ansatz for all S circuits.

    angles has shape (S, L, N).  The single differentiated rotation
    (diff_layer, diff_qubit) is offset by `shift`; every other angle is fixed,
    so the +shift and -shift runs share the same random circuit.
    """
    S = angles.shape[0]
    psi = np.zeros((S, 1 << N))
    psi[:, 0] = 1.0
    for l in range(L):
        a = angles[:, l, :].copy()
        if l == diff_layer:
            a[:, diff_qubit] += shift
        psi = apply_ry_wall(psi, N, a)
        for (p, q) in brick_pairs(N, l):
            psi *= masks[(p, q)][None, :]
    return psi


def zstring_signs(N):
    """Diagonal of the traceless global Z-string O = Z_1...Z_N: (-1)^{popcount(b)}."""
    pc = np.array([bin(b).count("1") for b in range(1 << N)])
    return 1.0 - 2.0 * (pc & 1)


def zsite_signs(N, site):
    """Diagonal of the single-site traceless O = Z_site: (-1)^{bit_site(b)}."""
    idx = np.arange(1 << N)
    return 1.0 - 2.0 * ((idx >> (N - 1 - site)) & 1)


def read_costs(psi, sg, sl):
    """Global (Z-string) and local (single-site Z) Pauli-expectation costs.

    Both observables are diagonal in the computational basis, and psi is real, so
    C = <psi|O|psi> = sum_b sign_O[b] |psi_b|^2.  Both are traceless, so their
    expectation vanishes under 2-design statistics.
    """
    prob = psi ** 2
    return prob @ sg, prob @ sl


def barren_variances(N, rng):
    """Var over random circuits of one parameter-shift gradient component,
    for the global and the local cost, at qubit number N."""
    L = N                                    # brickwork 2-design depth ~ O(N)
    diff_layer, diff_qubit = L // 2, N // 2  # a middle parameter
    local_qubit = N // 2                     # single-site Z on the driven qubit
    sg, sl = zstring_signs(N), zsite_signs(N, local_qubit)
    masks = {(p, q): cz_mask(N, p, q)
             for l in range(L) for (p, q) in brick_pairs(N, l)}
    angles = rng.uniform(0.0, 2.0 * np.pi, size=(N_CIRCUITS, L, N))
    psi_p = run_ansatz(N, L, angles, diff_layer, diff_qubit, +np.pi / 2, masks)
    psi_m = run_ansatz(N, L, angles, diff_layer, diff_qubit, -np.pi / 2, masks)
    Cg_p, Cl_p = read_costs(psi_p, sg, sl)
    Cg_m, Cl_m = read_costs(psi_m, sg, sl)
    grad_g = 0.5 * (Cg_p - Cg_m)             # parameter-shift for R_y (generator Y/2)
    grad_l = 0.5 * (Cl_p - Cl_m)
    return float(np.var(grad_g)), float(np.var(grad_l))


# ===========================================================================
# Panel (b): expressive IQP feature map and off-diagonal kernel statistics.
# ===========================================================================
def fwht(a):
    """Normalized Walsh-Hadamard transform H^{oxN} along the last axis of (P, 2^N)."""
    P, D = a.shape
    a = a.copy()
    h = 1
    while h < D:
        a = a.reshape(P, -1, 2 * h)
        x = a[:, :, :h].copy()
        y = a[:, :, h:].copy()
        a[:, :, :h] = x + y
        a[:, :, h:] = x - y
        a = a.reshape(P, D)
        h *= 2
    return a / np.sqrt(D)


def basis_signs(N):
    """Column k is the Z_k eigenvalue (-1)^{bit_k(b)} on each basis state b -> (2^N, N)."""
    idx = np.arange(1 << N)
    return np.stack([1.0 - 2.0 * ((idx >> (N - 1 - k)) & 1) for k in range(N)], axis=1)


def feature_states(X, N, L, scale):
    """Batched IQP feature states |phi(x)> = (U_Z(x) H^{oxN})^L |0> for rows of X.

    U_Z(x) is diagonal with phase theta_b = sum_k f_k s^k_b + sum_{k<l} f_k f_l s^k_b s^l_b,
    f = scale * x, s^k_b = (-1)^{bit_k(b)}; applied as an elementwise multiply between
    Walsh-Hadamard walls.
    """
    signs = basis_signs(N)
    P, D = X.shape[0], 1 << N
    F = X * scale
    single = F @ signs.T
    pair = np.zeros((P, D))
    for k in range(N):
        for l in range(k + 1, N):
            pair += (F[:, k] * F[:, l])[:, None] * (signs[:, k] * signs[:, l])[None, :]
    phase = np.exp(-1j * (single + pair))
    psi = np.zeros((P, D), dtype=complex)
    psi[:, 0] = 1.0
    for _ in range(L):
        psi = fwht(psi)
        psi *= phase
    return psi


def kernel_stats(N, rng):
    """Mean and std of the off-diagonal fidelity kernel over random data pairs."""
    X = rng.uniform(-np.pi, np.pi, size=(N_POINTS, N))
    psi = feature_states(X, N, IQP_LAYERS, IQP_SCALE)
    i = rng.integers(0, N_POINTS, N_PAIRS)
    j = rng.integers(0, N_POINTS, N_PAIRS)
    keep = i != j
    i, j = i[keep], j[keep]
    k = np.abs(np.sum(psi[i].conj() * psi[j], axis=1)) ** 2
    return float(np.mean(k)), float(np.std(k))


# ===========================================================================
def compute():
    """Both sweeps; cached so re-rendering never recomputes the statevectors."""
    Ns = np.arange(N_MIN, N_MAX + 1)
    var_g, var_l = [], []
    rng_bp = np.random.default_rng(SEED_BP)
    for N in Ns:
        vg, vl = barren_variances(int(N), rng_bp)
        var_g.append(vg)
        var_l.append(vl)
        print(f"  [a] N={N:2d}  Var[grad global]={vg:.3e}  Var[grad local]={vl:.3e}")

    kern_mean, kern_std = [], []
    rng_k = np.random.default_rng(SEED_KERNEL)
    for N in Ns:
        km, ks = kernel_stats(int(N), rng_k)
        kern_mean.append(km)
        kern_std.append(ks)
        print(f"  [b] N={N:2d}  mean k_off={km:.3e}  std k_off={ks:.3e}")

    return {"Ns": Ns,
            "var_g": np.array(var_g), "var_l": np.array(var_l),
            "kern_mean": np.array(kern_mean), "kern_std": np.array(kern_std)}


data = load_or_compute(DATA_DIR / "fig_ch7_trainability_scaling.npz", compute)
Ns = data["Ns"]
var_g, var_l = data["var_g"], data["var_l"]
kern_mean, kern_std = data["kern_mean"], data["kern_std"]
guide = 2.0 ** (-Ns.astype(float))

# fitted slopes (base-2 exponent per qubit) for the report / annotations
slope_g = np.polyfit(Ns, np.log2(var_g), 1)[0]
slope_l = np.polyfit(np.log2(Ns.astype(float)), np.log2(var_l), 1)[0]

# ---------------------------------------------------------------------------
# Figure: two panels, ~5:4 each.
# ---------------------------------------------------------------------------
apply_book_style()
fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(10.5, 4.3), constrained_layout=True)

# ---- Panel (a): barren plateau ----
ax_a.semilogy(Ns, var_g, "o-", color=C_GLOBAL, lw=1.8, ms=6, mec="white", mew=0.8,
              zorder=3, label="global cost")
ax_a.semilogy(Ns, var_l, "s-", color=C_LOCAL, lw=1.8, ms=6, mec="white", mew=0.8,
              zorder=3, label="local cost")
ax_a.semilogy(Ns, guide, "--", color=C_GUIDE, lw=1.4, zorder=1, label=r"$2^{-N}$")
ax_a.set_xlabel(r"number of qubits $N$")
ax_a.set_ylabel(r"$\mathrm{Var}\,[\,\partial_\theta C\,]$")
ax_a.set_xticks(Ns[::2])
ax_a.legend(loc="lower left", framealpha=0.9, handletextpad=0.5, fontsize=9)
ax_a.text(0.97, 0.93,
          "global cost:\nbarren plateau",
          transform=ax_a.transAxes, ha="right", va="top", color=C_GLOBAL, fontsize=9)
ax_a.text(0.97, 0.55, "local cost:\npoly. trainable",
          transform=ax_a.transAxes, ha="right", va="top", color=C_LOCAL, fontsize=9)
panel_label(ax_a, "a", loc="upper left")

# ---- Panel (b): quantum-kernel concentration ----
# Mean and std of the off-diagonal kernel plotted as two curves: both collapse
# onto 2^{-N}, so std tracks the mean and every entry is pinned near 2^{-N}.
ax_b.semilogy(Ns, kern_mean, "o-", color=C_KERNEL, lw=1.8, ms=6, mec="white", mew=0.8,
              zorder=3, label=r"mean $k(\mathbf{x}_i,\mathbf{x}_j)$")
ax_b.semilogy(Ns, kern_std, "s--", color=C_KERNEL_STD, lw=1.6, ms=5, mec="white",
              mew=0.7, zorder=3, label=r"std. dev. of $k$")
ax_b.semilogy(Ns, guide, "--", color=C_GUIDE, lw=1.4, zorder=1, label=r"$2^{-N}$")
ax_b.set_xlabel(r"number of qubits $N$")
ax_b.set_ylabel(r"off-diagonal kernel $|\langle\phi_i|\phi_j\rangle|^2$")
ax_b.set_xticks(Ns[::2])
ax_b.legend(loc="upper right", framealpha=0.9, handletextpad=0.5, fontsize=9)
ax_b.text(0.03, 0.06,
          r"mean $\approx$ std $\approx 2^{-N}$:" "\n"
          r"Gram matrix $\to$ identity," "\n"
          r"resolving $k_{ij}$ costs $\sim\!\mathcal{O}(2^N)$ shots",
          transform=ax_b.transAxes, ha="left", va="bottom", fontsize=8.5)
panel_label(ax_b, "b", loc="upper left")

plt.savefig(OUTPUT_DIR / "fig_ch7_trainability_scaling.pdf")
plt.savefig(OUTPUT_DIR / "fig_ch7_trainability_scaling.png")
print(f"  panel (a): global Var ~ 2^({slope_g:.2f} N)  (Z-string, tracks 2^(-N));"
      f"  local Var ~ N^({slope_l:.2f})")
print(f"  panel (b): mean off-diagonal kernel / 2^(-N) = "
      f"{kern_mean[-1] / guide[-1]:.2f} at N={int(Ns[-1])}")
print(f"  Saved: {OUTPUT_DIR / 'fig_ch7_trainability_scaling.pdf'}")
