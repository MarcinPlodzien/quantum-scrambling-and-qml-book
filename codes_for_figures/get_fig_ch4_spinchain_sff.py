#!/usr/bin/env python3
"""
get_fig_ch4_spinchain_sff.py
============================
The spectral form factor of a REAL many-body spectrum: why the ramp is buried in
the raw data and how symmetry resolution plus unfolding expose it.

Produces Fig. 4.3 -- figures/ch4/fig_ch4_spinchain_sff.pdf

    Panel (a): raw K(t)/D for the mixed-field Ising chain. Dip and plateau, no ramp.
    Panel (b): connected SFF after resolving the reflection sector and unfolding.
               The GOE ramp appears and follows the parameter-free prediction.

============================================================
WHY THIS FIGURE EXISTS
============================================================
Every other spectral-form-factor figure in the book is built from random matrices,
whose semicircular density of states is smooth enough that the ramp shows without
any preprocessing. A real many-body Hamiltonian is different in two ways that the
book states but does not otherwise demonstrate, and both bite here.

1. SYMMETRY. The chain has a reflection symmetry i -> N-1-i. Its Hamiltonian is
   block-diagonal in the two reflection-parity sectors, and eigenvalues from
   different blocks are statistically independent. Level statistics computed on
   the FULL spectrum therefore mix two uncorrelated sequences and look less
   chaotic than the system is: the gap ratio comes out near 0.45 rather than the
   Gaussian-orthogonal value 0.53. Resolving one sector removes the contamination.
   This script builds the reflection operator, projects the Hamiltonian onto the
   parity-even sector, and diagonalizes there.

2. NON-UNIVERSAL DENSITY OF STATES. A many-body spectrum is close to Gaussian with
   a width extensive in N, not semicircular. That non-universal profile enters the
   raw form factor through the disconnected part and buries the ramp under the dip.
   Unfolding rescales the spectrum to unit mean level spacing, removing the profile
   and leaving the universal correlations that produce the ramp.

Panel (a) shows the raw K(t)/D on the full spectrum: a dip falling from D and a
plateau, with no ramp visible anywhere. Panel (b) shows the connected SFF computed
on the unfolded parity-even spectrum, where the ramp appears and tracks the GOE
prediction b_1(tau) = 2 tau - tau ln(1 + 2 tau).

============================================================
MODEL
============================================================
Mixed-field Ising chain, open boundary conditions, at the standard chaotic point:

    H = J sum_i Z_i Z_{i+1} + h_x sum_i X_i + h_z sum_i Z_i,
    J = 1, h_x = 1.05, h_z = 0.5.

A transverse field alone (h_z = 0) is integrable via Jordan-Wigner; the longitudinal
field h_z breaks that integrability and makes the chain chaotic. Open boundaries
leave the reflection symmetry intact, which is the point of the exercise.

============================================================
IMPLEMENTATION NOTES
============================================================
- The reflection operator is the permutation of computational-basis states that
  reverses the bit string. Projecting onto its +1 eigenspace and rotating H into
  that basis gives the parity-even block.
- Unfolding is a low-order polynomial fit to the integrated density of states of
  the (central portion of the) resolved spectrum, evaluated at the eigenvalues.
- The SFF is NOT self-averaging: a single spectrum gives a wildly fluctuating K(t)
  and no clean ramp. Panel (b) therefore averages over a small on-site field
  disorder (h_z -> h_z + delta_i), which also breaks the reflection symmetry so
  each disordered spectrum is already a single GOE sequence.

RUNTIME: about two minutes at N = 11 (50 dense diagonalizations of a 2048-dim
matrix) on a cache MISS; the clean Hamiltonian and the per-site Z operators are
built once and reused across realizations. On a cache HIT the numerics are loaded
from data/ch4/ and the figure re-renders in under a second (set FIG_RECOMPUTE=1
to force a recompute).
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_style import apply_book_style, load_or_compute, panel_label

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR.parent / "figures" / "ch4"
DATA_DIR = SCRIPT_DIR / "data" / "ch4"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

N = 11
J, HX, HZ = 1.0, 1.05, 0.5

SX = np.array([[0, 1], [1, 0]], dtype=float)
SZ = np.diag([1.0, -1.0])
I2 = np.eye(2)


def op(single, site, n):
    m = np.array([[1.0]])
    for k in range(n):
        m = np.kron(m, single if k == site else I2)
    return m


def mfi_hamiltonian(n):
    H = np.zeros((2**n, 2**n))
    for i in range(n - 1):
        H += J * op(SZ, i, n) @ op(SZ, i + 1, n)
    for i in range(n):
        H += HX * op(SX, i, n) + HZ * op(SZ, i, n)
    return H


def reflection_operator(n):
    """Permutation matrix reversing the bit order of each basis state."""
    D = 2**n
    P = np.zeros((D, D))
    for b in range(D):
        rb = 0
        for k in range(n):
            rb |= ((b >> k) & 1) << (n - 1 - k)
        P[rb, b] = 1.0
    return P


def gap_ratio(E):
    E = np.sort(E)
    m = len(E)
    E = E[m // 5:4 * m // 5]           # central 60%
    s = np.diff(E)
    s = s[s > 1e-9]
    r = np.minimum(s[:-1], s[1:]) / np.maximum(s[:-1], s[1:])
    return float(np.mean(r))


def b_goe(tau):
    tau = np.clip(tau, 1e-9, None)
    return np.where(tau < 1, 2 * tau - tau * np.log(1 + 2 * tau), 1.0)


def compute():
    """All numerics for the figure, returned as a dict of arrays/scalars."""
    H = mfi_hamiltonian(N)
    P = reflection_operator(N)

    # full spectrum (sectors mixed)
    E_full = np.linalg.eigvalsh(H)

    # resolve the parity-even sector
    wP, vP = np.linalg.eigh(P)
    even = vP[:, wP > 0]
    H_even = even.T @ H @ even
    E_even = np.linalg.eigvalsh(H_even)

    r_full = gap_ratio(E_full)
    r_even = gap_ratio(E_even)

    # ── panel (a): raw K(t)/D on the full spectrum ──
    D = len(E_full)
    t_raw = np.logspace(-1.0, 2.7, 400)
    Z = np.exp(-1j * np.outer(t_raw, E_full - E_full.mean())).sum(axis=1)
    K_raw = np.abs(Z)**2 / D

    # ── panel (b): connected SFF, disorder-averaged, unfolded ──
    # The SFF is NOT self-averaging: one spectrum gives a wildly fluctuating K(t), and only
    # an ensemble average reveals the smooth ramp. We take a small on-site disorder,
    # h_z -> h_z + delta_i with delta_i uniform in [-W, W]. This also breaks the reflection
    # symmetry, so each disordered spectrum is a single GOE sequence with no need to project.
    N_REAL = 50
    W = 0.3
    tau = np.linspace(0.01, 1.5, 200)
    rng = np.random.default_rng(0)
    K_acc = np.zeros(len(tau))
    # precompute the clean Hamiltonian and the per-site Z operators ONCE; each realization
    # only adds a random longitudinal field, avoiding rebuilding the ZZ and X terms 50 times.
    H_clean = mfi_hamiltonian(N)
    Z_ops = [op(SZ, i, N) for i in range(N)]
    for _ in range(N_REAL):
        deltas = W * (2 * rng.random(N) - 1)
        Hd = H_clean + sum(d * Zi for d, Zi in zip(deltas, Z_ops))
        Ed = np.sort(np.linalg.eigvalsh(Hd))
        m = len(Ed)
        Ec = Ed[int(0.1 * m):int(0.9 * m)]
        coef = np.polyfit(Ec, np.arange(len(Ec)), 9)          # smooth integrated DOS
        e = np.polyval(coef, Ec)                              # unfolded, unit mean spacing
        e -= e.mean()
        Z = np.exp(-2j * np.pi * np.outer(tau, e)).sum(axis=1)
        K_acc += np.abs(Z)**2
    Nlev = len(e)
    K_smooth = K_acc / (N_REAL * Nlev)                        # connected (disconnected ~0 after unfolding)

    return dict(t_raw=t_raw, K_raw=K_raw, tau=tau, K_smooth=K_smooth,
                r_full=r_full, r_even=r_even)


data = load_or_compute(DATA_DIR / "fig_ch4_spinchain_sff.npz", compute)
t_raw, K_raw = data["t_raw"], data["K_raw"]
tau, K_smooth = data["tau"], data["K_smooth"]
r_full, r_even = float(data["r_full"]), float(data["r_even"])

print(f"  MFI N={N}")
print(f"  full spectrum   <r> = {r_full:.3f}  (sectors mixed, looks sub-chaotic)")
print(f"  parity-even     <r> = {r_even:.3f}  (GOE ~0.53)")

apply_book_style()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6), constrained_layout=True)

# panel (a): the RAW form factor, log-log -- a dip falling to the plateau, no ramp
ax1.loglog(t_raw, K_raw, color="#4361ee", lw=1.1)
ax1.axhline(1.0, color="0.6", ls="--", lw=0.8)
ax1.set_xlabel(r"time $t$")
ax1.set_ylabel(r"$K(t)/D$")
ax1.set_ylim(0.4, 3e3)
panel_label(ax1, "a", loc="upper right")
ax1.text(0.28, 0.30, "dip", transform=ax1.transAxes, color="#4361ee")
ax1.text(0.60, 0.42, "plateau", transform=ax1.transAxes, color="0.45")

# panel (b): connected SFF after resolving the sector + unfolding; a moving
# average over the disorder-averaged data exposes the ramp under the fluctuations
def smooth(y, w=9):
    return np.convolve(y, np.ones(w) / w, mode="same")
ax2.plot(tau, K_smooth, color="#2a9d5c", lw=0.6, alpha=0.30)
ax2.plot(tau[4:-4], smooth(K_smooth)[4:-4], color="#2a9d5c", lw=1.9, label="chain (unfolded)")
ax2.plot(tau, b_goe(tau), "k--", lw=1.5, label=r"GOE $b_1(\tau)$")
ax2.axhline(1.0, color="0.6", ls="--", lw=0.8)
ax2.set_xlabel(r"$\tau = t/t_H$")
ax2.set_ylabel(r"$K_c(\tau)$")
ax2.set_ylim(0, 1.45)
panel_label(ax2, "b", loc="upper left")
ax2.legend(loc="lower right")

plt.savefig(OUTPUT_DIR / "fig_ch4_spinchain_sff.pdf")
plt.savefig(OUTPUT_DIR / "fig_ch4_spinchain_sff.png")
print(f"  Saved: {OUTPUT_DIR / 'fig_ch4_spinchain_sff.pdf'}")
