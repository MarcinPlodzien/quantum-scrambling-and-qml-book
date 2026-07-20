#!/usr/bin/env python3
"""
get_fig_ch7_santafe.py
======================
Quantum reservoir forecasting on the Santa Fe laser benchmark, and the edge of
scrambling in the reservoir's two natural knobs.

Produces Fig. (ch7) -- figures/ch7/fig_ch7_santafe.pdf

    Panel (a): the Santa Fe (set A) far-infrared laser series -- the standard
               chaotic-forecasting benchmark of reservoir computing.
    Panel (b): total forecasting capacity, sum_tau R^2(tau), as a 2D map over the
               two reservoir knobs: coupling J/h and per-step evolution time
               t_qrc.  It is small when the reservoir under-scrambles (weak
               coupling AND short evolution) or over-scrambles (strong coupling
               AND long evolution), and largest on the edge-of-scrambling ridge.
    Panel (c): true vs one-step-ahead predicted intensity on held-out data, at
               the optimal (J/h, t_qrc).

RESERVOIR.  A genuinely scrambling reservoir: the nonintegrable MIXED-FIELD Ising
chain H = J sum Z_i Z_{i+1} + h_x sum X_i + h_z sum Z_i.  (The pure transverse
field h_z=0 is Jordan-Wigner integrable and does not scramble; the longitudinal
h_z breaks integrability.)  The input is written into qubit 0 by an R_y rotation
(<Z>=s), the previous input qubit is traced out each step (the map that gives
fading memory), the state is evolved as a density matrix, and low-weight
observables are collected as features for a linear (ridge) readout.

Cached in data/ch7/fig_ch7_santafe.npz.  Dataset: data/ch7/santafe_laser.npy
(Santa Fe Time Series Competition, set A).
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


def mixed_field_ising(N, J, hx, hz_sites):
    """Nonintegrable mixed-field Ising Hamiltonian (hz_sites: per-site long. field)."""
    H = np.zeros((2 ** N, 2 ** N), dtype=complex)
    for i in range(N - 1):
        H += J * (kron_op(Z, i, N) @ kron_op(Z, i + 1, N))
    for i in range(N):
        H += hx * kron_op(X, i, N) + hz_sites[i] * kron_op(Z, i, N)
    return H


def encode(s):
    # R_y(arccos s)|0>  ->  <Z> = s
    return np.array([np.sqrt((1 + s) / 2), np.sqrt((1 - s) / 2)], dtype=complex)


_POP = np.array([bin(i).count("1") for i in range(1 << 16)])
_PH = np.array([1, 1j, -1, -1j])


def pauli_readout(N):
    """Precompute the readout set {X_i, Y_i, Z_i, X_iX_{i+1}, Y_iY_{i+1},
    Z_iZ_{i+1}} as column-index and weight matrices.  Any Pauli observable
    P = i^p X^x Z^z has <P> = Re sum_b W[b] rho[b, b^x], an O(2^N) dot product
    (verified against brute-force Tr(P rho))."""
    bit = lambda i: 1 << (N - 1 - i)
    specs = []
    for i in range(N):
        specs += [(bit(i), 0, 0), (bit(i), bit(i), 1), (0, bit(i), 0)]          # X, Y, Z
    for i in range(N - 1):
        m = bit(i) | bit(i + 1)
        specs += [(m, 0, 0), (m, m, 2), (0, m, 0)]                              # XX, YY, ZZ
    b = np.arange(1 << N)
    COLS = np.array([b ^ x for (x, z, p) in specs])
    W = np.array([_PH[p] * (1 - 2 * (_POP[z & b] & 1)) for (x, z, p) in specs], dtype=complex)
    return COLS, W


def reservoir_features(series, N, H, tau_ev, COLS, W):
    U = expm(-1j * H * tau_ev)
    Udag = U.conj().T                        # cache once: reused every step below
    D, D_res = 1 << N, 1 << (N - 1)
    rho = np.zeros((D, D), dtype=complex)
    rho[0, 0] = 1.0
    b = np.arange(D)
    F = np.zeros((len(series), COLS.shape[0]))
    for n, s in enumerate(series):
        rho_res = np.einsum("aiaj->ij", rho.reshape(2, D_res, 2, D_res))   # trace out qubit 0
        psi = encode(s)
        rho = U @ np.kron(np.outer(psi, psi.conj()), rho_res) @ Udag
        F[n] = np.real((W * rho[b[None, :], COLS]).sum(axis=1))            # X/Y/Z/XX/YY/ZZ, O(2^N) each
    return F


def forecast_r2(F, series, tau, washout=70, train_frac=0.6, lam=1e-6, return_pred=False):
    Xf = F[washout: len(series) - tau]
    y = series[washout + tau:]
    Xf = np.hstack([Xf, np.ones((len(Xf), 1))])
    ntr = int(train_frac * len(Xf))
    w = np.linalg.solve(Xf[:ntr].T @ Xf[:ntr] + lam * np.eye(Xf.shape[1]), Xf[:ntr].T @ y[:ntr])
    pred, true = Xf[ntr:] @ w, y[ntr:]
    r2 = float(np.corrcoef(pred, true)[0, 1] ** 2)
    return (r2, true, pred) if return_pred else r2


def otoc_scrambling(N, H, t):
    """Reservoir scrambling measure: how far the injected input (at qubit 0) has
    spread after one step of evolution.  With V = Z_0 the input operator and the
    butterfly W = Z_j read out at every site j, average the OTOC

        C = (1/N) sum_j [ 1 - Re Tr[Z_j(t) V Z_j(t) V]/D ] ,   Z_j(t) = U^dag Z_j U,

    the same out-of-time-order diagnostic of Chapter~ch:dynamics but averaged over
    readout sites, so it runs cleanly from 0 (input still localized, unscrambled)
    to 1 (input spread across the chain, over-scrambled) over the reservoir's
    working range of t_qrc."""
    D = 1 << N
    U = expm(-1j * H * t)
    Udag = U.conj().T
    V = kron_op(Z, 0, N)
    c = 0.0
    for j in range(N):
        Wt = Udag @ kron_op(Z, j, N) @ U
        c += 1 - np.trace(Wt @ V @ Wt @ V).real / D
    return c / N


N = 8
HX = 1.0
HZ0 = 0.5
N_SEEDS = 3
N_STEPS = 700
TAUS = list(range(1, 11))
# Dense, wide log grid over the two knobs so the edge-of-scrambling ridge and both
# the under- and over-scrambled flanks are resolved (t_qrc reaches far enough for
# the OTOC to saturate toward 1, so all scrambling contours appear).
JH_VALUES = np.geomspace(0.15, 5.0, 18)
TEV_VALUES = np.geomspace(0.2, 8.0, 16)


def load_series():
    raw = np.load(DATA_DIR / "santafe_laser.npy").astype(float).ravel()[:N_STEPS]
    return 2 * (raw - raw.min()) / (raw.max() - raw.min()) - 1


def compute():
    series = load_series()
    COLS, W = pauli_readout(N)
    cap = np.zeros((len(JH_VALUES), len(TEV_VALUES)))
    otoc = np.zeros((len(JH_VALUES), len(TEV_VALUES)))
    for si in range(N_SEEDS):
        rng = np.random.default_rng(si)
        hz = HZ0 + 0.1 * (2 * rng.random(N) - 1)      # small per-site disorder
        for a, J in enumerate(JH_VALUES):
            H = mixed_field_ising(N, J * HX, HX, hz)
            for b, tev in enumerate(TEV_VALUES):
                F = reservoir_features(series, N, H, tev, COLS, W)
                cap[a, b] += sum(forecast_r2(F, series, t) for t in TAUS) / N_SEEDS
        print(f"  seed {si + 1}/{N_SEEDS} done")
    # OTOC scrambling map on the clean reservoir Hamiltonian (a Hamiltonian property)
    for a, J in enumerate(JH_VALUES):
        H = mixed_field_ising(N, J * HX, HX, HZ0 * np.ones(N))
        for b, tev in enumerate(TEV_VALUES):
            otoc[a, b] = otoc_scrambling(N, H, tev)
    ia, ib = np.unravel_index(np.argmax(cap), cap.shape)
    J_opt, t_opt = float(JH_VALUES[ia]), float(TEV_VALUES[ib])
    H = mixed_field_ising(N, J_opt * HX, HX, HZ0 * np.ones(N))
    _, true, pred = forecast_r2(reservoir_features(series, N, H, t_opt, COLS, W), series, 1, return_pred=True)
    print(f"  optimum: J/h_x={J_opt}, t_qrc={t_opt},  total capacity={cap[ia, ib]:.2f}")
    return {"series": series, "cap": cap, "otoc": otoc, "JH": JH_VALUES, "TEV": TEV_VALUES,
            "J_opt": J_opt, "t_opt": t_opt, "true": true, "pred": pred}


data = load_or_compute(DATA_DIR / "fig_ch7_santafe.npz", compute)
cap, JH, TEV, otoc = data["cap"], data["JH"], data["TEV"], data["otoc"]
J_opt, t_opt = float(data["J_opt"]), float(data["t_opt"])

from matplotlib.ticker import ScalarFormatter, NullFormatter

# Three reservoirs spanning the scrambling axis, read off the maps: under-scrambled
# = least mixing (min OTOC), over-scrambled = most (max OTOC), and the edge = the
# capacity optimum sitting between them. Each is one reservoir run on the cached
# knobs (cheap) -- no need to recompute the heavy capacity map.
series = data["series"]
COLS, W = pauli_readout(N)
iu = np.unravel_index(np.argmin(otoc), otoc.shape)
io = np.unravel_index(np.argmax(otoc), otoc.shape)
C_UNDER, C_EDGE, C_OVER, C_TRUE = "#1f6fc4", "#c1121f", "#e8850c", "#1a1a1a"
REGIMES = [   # (label, J/h_x, t_qrc, colour, marker, linestyle, linewidth, zorder)
    ("under-scrambled",    float(JH[iu[0]]), float(TEV[iu[1]]), C_UNDER, "X", "-",  1.0, 2),
    ("edge of scrambling", J_opt,            t_opt,             C_EDGE,  "*", "--", 1.7, 4),
    ("over-scrambled",     float(JH[io[0]]), float(TEV[io[1]]), C_OVER,  "P", "-",  1.0, 3),
]
PRED_TAUS = [1, 4, 8]


def regime_pred(Jv, tv):
    H = mixed_field_ising(N, Jv * HX, HX, HZ0 * np.ones(N))
    F = reservoir_features(series, N, H, tv, COLS, W)
    return {tau: forecast_r2(F, series, tau, return_pred=True) for tau in PRED_TAUS}


preds = {name: regime_pred(Jv, tv) for (name, Jv, tv, *_ ) in REGIMES}

apply_book_style()
fig = plt.figure(figsize=(11.5, 9.6), constrained_layout=True)
gs = fig.add_gridspec(3, 3)
ax_a = fig.add_subplot(gs[0, 0])
ax_b = fig.add_subplot(gs[0, 1:])
ax_ts = [fig.add_subplot(gs[1, k]) for k in range(3)]
ax_sc = [fig.add_subplot(gs[2, k]) for k in range(3)]

# (a) the benchmark series
ax_a.plot(series[:300], color=C_TRUE, lw=0.9)
ax_a.set_xlabel("time step $n$")
ax_a.set_ylabel("laser intensity (scaled)")
panel_label(ax_a, "a", loc="upper right")

# (b) capacity map over the two knobs, with the three regimes marked (markers
#     un-clipped so corner points stay fully visible). Regimes are chosen from the
#     OTOC map (under = min, over = max) and the capacity map (edge = argmax).
im = ax_b.pcolormesh(TEV, JH, cap, shading="gouraud", cmap="viridis")
ax_b.set_yscale("log"); ax_b.set_xscale("log")
for axis in (ax_b.xaxis, ax_b.yaxis):
    axis.set_major_formatter(ScalarFormatter()); axis.set_minor_formatter(NullFormatter())
ax_b.set_xticks([0.25, 0.5, 1, 2, 4, 8]); ax_b.set_yticks([0.2, 0.5, 1, 2, 5])
ax_b.set_xlabel(r"evolution time $t_{\mathrm{qrc}}$")
ax_b.set_ylabel(r"coupling $J/h_x$")
for (name, Jv, tv, col, mk, *_ ) in REGIMES:
    ax_b.plot(tv, Jv, mk, color=col, ms=15 if mk == "*" else 11, mec="black", mew=0.7,
              label=name, clip_on=False, zorder=6)
ax_b.legend(loc="lower left", fontsize=8, framealpha=0.9)
cb = fig.colorbar(im, ax=ax_b, pad=0.02)
cb.set_label(r"$C_{\mathrm{tot}}$")
panel_label(ax_b, "b", loc="upper left", color="white")

# (c-e) true series vs the three regime forecasts, horizons tau = 1, 4, 8
true1 = preds["edge of scrambling"][1][1]
for k, tau in enumerate(PRED_TAUS):
    ax = ax_ts[k]
    true = preds["edge of scrambling"][tau][1]
    ax.plot(true[:110], color=C_TRUE, lw=1.9, zorder=1, label="true")
    for (name, Jv, tv, col, mk, ls, lw, zo) in REGIMES:
        pred = preds[name][tau][2]
        ax.plot(pred[:110], color=col, ls=ls, lw=lw, zorder=zo, label=name)
    ax.set_xlabel(r"held-out step $n$")
    if k == 0:
        ax.set_ylabel("intensity (scaled)")
        ax.legend(loc="lower right", fontsize=7, framealpha=0.9)
    ax.text(0.03, 0.90, rf"$\tau={tau}$", transform=ax.transAxes, fontsize=10)
    panel_label(ax, "cde"[k], loc="upper right")

# (f-h) predicted vs true scatter at each horizon; on-diagonal = accurate.
#       R^2 per regime is annotated in the regime's colour.
for k, tau in enumerate(PRED_TAUS):
    ax = ax_sc[k]
    ax.plot([-1.1, 1.1], [-1.1, 1.1], ls=":", color="grey", lw=1.0, zorder=1)
    xs = np.array([-1.1, 1.1])
    for r, (name, Jv, tv, col, mk, ls, lw, zo) in enumerate(REGIMES):
        r2, true, pred = preds[name][tau]
        ax.scatter(true, pred, s=7, color=col, alpha=0.40, edgecolors="none", zorder=zo)
        slope, icpt = np.polyfit(true, pred, 1)               # per-regime least-squares fit
        ax.plot(xs, slope * xs + icpt, color=col, lw=2.6, zorder=zo + 3)
        ax.text(0.05, 0.90 - 0.11 * r, rf"$R^2={r2:.2f}$", transform=ax.transAxes,
                color=col, fontsize=8)
    ax.set_xlim(-1.1, 1.1); ax.set_ylim(-1.1, 1.1); ax.set_aspect("equal", "box")
    ax.set_xlabel(r"$s_n$")
    if k == 0:
        ax.set_ylabel(r"predicted $\hat{s}_n$")
    ax.text(0.62, 0.06, rf"$\tau={tau}$", transform=ax.transAxes, fontsize=10)
    panel_label(ax, "fgh"[k], loc="upper right")

plt.savefig(OUTPUT_DIR / "fig_ch7_santafe.pdf")
plt.savefig(OUTPUT_DIR / "fig_ch7_santafe.png")
for (name, Jv, tv, *_ ) in REGIMES:
    print(f"  {name:20s} J/h={Jv:4.2g} t={tv:4.2g}  R2(1,4,8)="
          f"{tuple(round(preds[name][t][0], 2) for t in PRED_TAUS)}")
print(f"  Saved: {OUTPUT_DIR / 'fig_ch7_santafe.pdf'}")
