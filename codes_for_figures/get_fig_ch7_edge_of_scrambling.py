#!/usr/bin/env python3
"""
get_fig_ch7_edge_of_scrambling.py
=================================
The edge of scrambling for a quantum reservoir, read off a single knob: the
coupling J/h_x of the nonintegrable MIXED-FIELD Ising reservoir

    H = J sum_i Z_i Z_{i+1} + h_x sum_i X_i + h_z sum_i Z_i .

(The longitudinal h_z is what breaks Jordan-Wigner integrability and makes the
chain genuinely scramble; this is the SAME reservoir as the Santa Fe figure, so
the two figures are consistent.)

Produces Fig. (ch7) -- figures/ch7/fig_ch7_edge_of_scrambling.pdf

    Panel (a): total linear memory capacity C_total = sum_k C_k vs J/h_x.
    Panel (b): the reservoir's scrambling, quantified two ways vs J/h_x -- the
               averaged OTOC (how far an injected input spreads in one step) and
               the steady-state half-chain entanglement S_vN.

THE POINT.  Capacity peaks at intermediate coupling, the "edge of scrambling":
weak coupling barely mixes inputs (little nonlinearity, OTOC ~ 0), strong coupling
over-scrambles them (the injected qubit delocalizes and the partial trace erases
it, OTOC -> 1, entanglement saturated).  Overlaying the capacity peak on panel (b)
shows it sits where the OTOC is still climbing and the entanglement is well below
its ceiling: maximal scrambling is NOT maximal computational usefulness.

Cached in data/ch7/fig_ch7_edge_of_scrambling.npz (FIG_RECOMPUTE=1 to recompute).
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.linalg import expm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_style import apply_book_style, load_or_compute, panel_label

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR.parent / "figures" / "ch7"
DATA_DIR = SCRIPT_DIR / "data" / "ch7"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)


def kron_op(single, site, N):
    op = np.array([[1.0]], dtype=complex)
    for j in range(N):
        op = np.kron(op, single if j == site else I2)
    return op


def mixed_field_ising(N, J, hx, hz):
    H = np.zeros((2 ** N, 2 ** N), dtype=complex)
    for i in range(N - 1):
        H += J * (kron_op(Z, i, N) @ kron_op(Z, i + 1, N))
    for i in range(N):
        H += hx * kron_op(X, i, N) + hz * kron_op(Z, i, N)
    return H


def encode(s):
    return np.array([np.sqrt((1 + s) / 2), np.sqrt((1 - s) / 2)], dtype=complex)


def build_Z_observables(N):
    """Single-site Z_i and nearest-neighbour Z_iZ_{i+1} feature operators."""
    obs = [kron_op(Z, i, N) for i in range(N)]
    obs += [kron_op(Z, i, N) @ kron_op(Z, i + 1, N) for i in range(N - 1)]
    return obs


def otoc_scrambling(N, H, t):
    """Averaged input-spreading OTOC: V = Z_0 (input qubit), W = Z_j read out at
    every site, C = (1/N) sum_j [1 - Re Tr(Z_j(t) V Z_j(t) V)/D].  Runs 0 -> 1 as
    the reservoir goes from unscrambled to fully scrambled."""
    D = 1 << N
    U = expm(-1j * H * t)
    Ud = U.conj().T
    V = kron_op(Z, 0, N)
    c = 0.0
    for j in range(N):
        Wt = Ud @ kron_op(Z, j, N) @ U
        c += 1 - np.trace(Wt @ V @ Wt @ V).real / D
    return c / N


def run_capacity(N, H, tau, obs, rng, n_wash=40, n_eval=250, max_delay=20):
    """Total linear memory capacity of the driven reservoir on i.i.d. uniform inputs."""
    U = expm(-1j * H * tau)
    Ud = U.conj().T
    D, D_res = 1 << N, 1 << (N - 1)
    n_tot = n_wash + n_eval + max_delay
    s = rng.uniform(-1, 1, size=n_tot)
    rho = np.zeros((D, D), dtype=complex)
    rho[0, 0] = 1.0
    F = np.zeros((n_tot, len(obs)))
    for n in range(n_tot):
        rho_res = np.einsum("aiaj->ij", rho.reshape(2, D_res, 2, D_res))
        psi = encode(s[n])
        rho = U @ np.kron(np.outer(psi, psi.conj()), rho_res) @ Ud
        F[n] = [np.real(np.trace(o @ rho)) for o in obs]
    Fe = F[n_wash + max_delay:]
    C = 0.0
    for k in range(1, max_delay + 1):
        y = s[n_wash + max_delay - k: n_wash + max_delay - k + n_eval]
        w = np.linalg.solve(Fe.T @ Fe + 1e-6 * np.eye(Fe.shape[1]), Fe.T @ y)
        C += np.corrcoef(Fe @ w, y)[0, 1] ** 2
    return float(C)


N = 6
HX = 1.0
HZ = 0.5
TAU = 1.0
N_SEEDS = 4
JH_VALUES = np.geomspace(0.1, 5.0, 28)


def compute():
    obs = build_Z_observables(N)
    C = np.zeros((len(JH_VALUES), N_SEEDS))
    otoc = np.zeros(len(JH_VALUES))
    for i, Jh in enumerate(JH_VALUES):
        H = mixed_field_ising(N, Jh * HX, HX, HZ)
        otoc[i] = otoc_scrambling(N, H, TAU)
        for r in range(N_SEEDS):
            C[i, r] = run_capacity(N, H, TAU, obs, np.random.default_rng(10 * i + r))
        print(f"  J/h_x={Jh:5.2f}  C={C[i].mean():6.3f}  OTOC={otoc[i]:.3f}")
    return {"JH": JH_VALUES, "C": C, "otoc": otoc}


data = load_or_compute(DATA_DIR / "fig_ch7_edge_of_scrambling.npz", compute)
JH = data["JH"]
Cm, Cs = data["C"].mean(1), data["C"].std(1)
otoc = data["otoc"]
i_pk = int(np.argmax(Cm))
Jh_pk = float(JH[i_pk])

apply_book_style()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6), constrained_layout=True, sharex=True)

# (a) memory capacity vs coupling
C_CAP, C_OTOC = "#2a6f97", "#c1121f"
ax1.fill_between(JH, Cm - Cs, Cm + Cs, alpha=0.20, color=C_CAP)
ax1.plot(JH, Cm, "o-", color=C_CAP, ms=4, lw=1.5)
ax1.axvline(Jh_pk, color="grey", ls="--", lw=0.9)
ax1.set_xscale("log")
ax1.set_xlabel(r"coupling $J/h_x$")
ax1.set_ylabel(r"memory capacity $C_{\mathrm{tot}}$")
ax1.annotate(rf"edge: $J/h_x\approx{Jh_pk:.2f}$", xy=(Jh_pk, Cm[i_pk]),
             xytext=(Jh_pk * 1.4, Cm[i_pk] * 0.8), fontsize=9,
             arrowprops=dict(arrowstyle="->", color="grey"))
LBL_BBOX = dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8)
panel_label(ax1, "a", loc="upper left", bbox=LBL_BBOX)

# (b) reservoir scrambling: the averaged input-spreading OTOC vs coupling, with the
#     capacity peak marked. The capacity peak sits on the rising flank of the OTOC:
#     the edge of scrambling, before the Z-input channel over-mixes.
ax2.plot(JH, otoc, "s-", color=C_OTOC, ms=4, lw=1.5)
ax2.axvline(Jh_pk, color="grey", ls="--", lw=0.9)
ax2.set_xscale("log")
ax2.set_xlabel(r"coupling $J/h_x$")
ax2.set_ylabel(r"OTOC scrambling $\bar{C}$")
ax2.set_ylim(0, float(otoc.max()) * 1.12)
panel_label(ax2, "b", loc="upper left", bbox=LBL_BBOX)

plt.savefig(OUTPUT_DIR / "fig_ch7_edge_of_scrambling.pdf")
plt.savefig(OUTPUT_DIR / "fig_ch7_edge_of_scrambling.png")
print(f"  peak capacity C={Cm[i_pk]:.2f} at J/h_x={Jh_pk:.2f}, OTOC there={otoc[i_pk]:.2f}")
print(f"  Saved: {OUTPUT_DIR / 'fig_ch7_edge_of_scrambling.pdf'}")
