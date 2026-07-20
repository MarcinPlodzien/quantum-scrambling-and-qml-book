#!/usr/bin/env python3
"""
get_fig_ch7_capacity_horizon.py
===============================
Forecast capacity resolved by horizon: R^2(tau) vs the prediction horizon tau for
the three scrambling regimes (under-scrambled, edge of scrambling, over-scrambled)
of the Santa Fe reservoir.

Produces Fig. (ch7) -- figures/ch7/fig_ch7_capacity_horizon.pdf

THE POINT.  The three regimes are read off the SAME capacity map and OTOC map as
Fig. fig:santafe (this script loads that figure's cache for the series and the
three regime knobs), and it uses the identical reservoir + ridge readout, so the
curves pass exactly through the R^2 values shown in the scatter panels there.

    * At tau = 1 the task is nearly autoregressive: memory alone suffices and all
      three regimes forecast comparably well.
    * As tau grows the regimes separate.  The edge-of-scrambling reservoir decays
      slowest and keeps the most predictive power; the over-scrambled reservoir,
      strong on nonlinearity but poor on retained memory, decays fastest; the
      under-scrambled reservoir has memory but no nonlinearity and stays low.

The area under each curve is the total capacity C_tot = sum_tau R^2(tau), the
quantity plotted as the 2D map in Fig. fig:santafe (shown per regime in the
legend).

Cheap: three reservoir runs on the cached regime knobs; cached in
data/ch7/fig_ch7_capacity_horizon.npz.
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.linalg import expm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_style import apply_book_style, load_or_compute

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR.parent / "figures" / "ch7"
DATA_DIR = SCRIPT_DIR / "data" / "ch7"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---- reservoir physics, identical to get_fig_ch7_santafe.py (kept in step so the
#      curves match that figure's scatter panels exactly) ----------------------
I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)


def kron_op(single, site, N):
    op = np.array([[1.0]], dtype=complex)
    for j in range(N):
        op = np.kron(op, single if j == site else I2)
    return op


def mixed_field_ising(N, J, hx, hz_sites):
    H = np.zeros((2 ** N, 2 ** N), dtype=complex)
    for i in range(N - 1):
        H += J * (kron_op(Z, i, N) @ kron_op(Z, i + 1, N))
    for i in range(N):
        H += hx * kron_op(X, i, N) + hz_sites[i] * kron_op(Z, i, N)
    return H


def encode(s):
    return np.array([np.sqrt((1 + s) / 2), np.sqrt((1 - s) / 2)], dtype=complex)


_POP = np.array([bin(i).count("1") for i in range(1 << 16)])
_PH = np.array([1, 1j, -1, -1j])


def pauli_readout(N):
    bit = lambda i: 1 << (N - 1 - i)
    specs = []
    for i in range(N):
        specs += [(bit(i), 0, 0), (bit(i), bit(i), 1), (0, bit(i), 0)]
    for i in range(N - 1):
        m = bit(i) | bit(i + 1)
        specs += [(m, 0, 0), (m, m, 2), (0, m, 0)]
    b = np.arange(1 << N)
    COLS = np.array([b ^ x for (x, z, p) in specs])
    W = np.array([_PH[p] * (1 - 2 * (_POP[z & b] & 1)) for (x, z, p) in specs], dtype=complex)
    return COLS, W


def reservoir_features(series, N, H, tau_ev, COLS, W):
    U = expm(-1j * H * tau_ev)
    Udag = U.conj().T
    D, D_res = 1 << N, 1 << (N - 1)
    rho = np.zeros((D, D), dtype=complex)
    rho[0, 0] = 1.0
    b = np.arange(D)
    F = np.zeros((len(series), COLS.shape[0]))
    for n, s in enumerate(series):
        rho_res = np.einsum("aiaj->ij", rho.reshape(2, D_res, 2, D_res))
        psi = encode(s)
        rho = U @ np.kron(np.outer(psi, psi.conj()), rho_res) @ Udag
        F[n] = np.real((W * rho[b[None, :], COLS]).sum(axis=1))
    return F


def forecast_r2(F, series, tau, washout=70, train_frac=0.6, lam=1e-6):
    Xf = F[washout: len(series) - tau]
    y = series[washout + tau:]
    Xf = np.hstack([Xf, np.ones((len(Xf), 1))])
    ntr = int(train_frac * len(Xf))
    w = np.linalg.solve(Xf[:ntr].T @ Xf[:ntr] + lam * np.eye(Xf.shape[1]), Xf[:ntr].T @ y[:ntr])
    pred, true = Xf[ntr:] @ w, y[ntr:]
    return float(np.corrcoef(pred, true)[0, 1] ** 2)


N = 8
HX = 1.0
HZ0 = 0.5
TAUS = np.arange(1, 11)                       # matches C_tot = sum_{tau=1}^{10} in fig:santafe
SANTAFE_NPZ = DATA_DIR / "fig_ch7_santafe.npz"
# (label, colour, linestyle, linewidth) per regime -- same palette as fig:santafe
REGIME_STYLE = {
    "under": ("under-scrambled",    "#1f6fc4", "-",  1.8),
    "edge":  ("edge of scrambling", "#c1121f", "--", 2.4),
    "over":  ("over-scrambled",     "#e8850c", "-",  1.8),
}


def compute():
    d = np.load(SANTAFE_NPZ)
    series, cap, otoc, JH, TEV = d["series"], d["cap"], d["otoc"], d["JH"], d["TEV"]
    iu = np.unravel_index(np.argmin(otoc), otoc.shape)   # under-scrambled: least OTOC
    ie = np.unravel_index(np.argmax(cap), cap.shape)     # edge: max total capacity
    io = np.unravel_index(np.argmax(otoc), otoc.shape)   # over-scrambled: most OTOC
    pts = {"under": (JH[iu[0]], TEV[iu[1]]),
           "edge":  (JH[ie[0]], TEV[ie[1]]),
           "over":  (JH[io[0]], TEV[io[1]])}
    COLS, W = pauli_readout(N)
    out = {"taus": TAUS}
    for key, (J, t) in pts.items():
        H = mixed_field_ising(N, J * HX, HX, HZ0 * np.ones(N))
        F = reservoir_features(series, N, H, float(t), COLS, W)
        out[f"R2_{key}"] = np.array([forecast_r2(F, series, int(tau)) for tau in TAUS])
        out[f"pt_{key}"] = np.array([float(J), float(t)])
        print(f"  {key:6s} J/h={float(J):4.2g} t={float(t):4.2g}  "
              f"C_tot={out[f'R2_{key}'].sum():.2f}")
    return out


data = load_or_compute(DATA_DIR / "fig_ch7_capacity_horizon.npz", compute)
taus = data["taus"]

apply_book_style()
fig, ax = plt.subplots(figsize=(6.4, 4.8), constrained_layout=True)

# light guides at the horizons shown in the fig:santafe time-series/scatter panels
for tau_mark in (1, 4, 8):
    ax.axvline(tau_mark, color="0.85", lw=0.8, zorder=0)

for key in ("under", "edge", "over"):
    label, col, ls, lw = REGIME_STYLE[key]
    r2 = data[f"R2_{key}"]
    ax.plot(taus, r2, ls=ls, lw=lw, color=col, marker="o", ms=4,
            label=rf"{label} ($C_{{\mathrm{{tot}}}}={r2.sum():.2f}$)")

ax.set_xlabel(r"forecast horizon $\tau$")
ax.set_ylabel(r"forecast capacity $C(\tau)$")
ax.set_xlim(taus.min() - 0.3, taus.max() + 0.3)
ax.set_ylim(0, 1.0)
ax.set_xticks(taus)
ax.legend(loc="upper right", framealpha=0.9)

plt.savefig(OUTPUT_DIR / "fig_ch7_capacity_horizon.pdf")
plt.savefig(OUTPUT_DIR / "fig_ch7_capacity_horizon.png")
print(f"  Saved: {OUTPUT_DIR / 'fig_ch7_capacity_horizon.pdf'}")
