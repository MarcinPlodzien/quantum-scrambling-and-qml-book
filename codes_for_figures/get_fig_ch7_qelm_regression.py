#!/usr/bin/env python3
"""
get_fig_ch7_qelm_regression.py
==============================
Nonlinear function regression with a QUANTUM EXTREME LEARNING MACHINE (QELM).

Produces Fig. (ch7) -- figures/ch7/fig_ch7_qelm_regression.pdf (+png).

    Panel (a): a handful of reservoir feature functions f_alpha(x) = <O_alpha>(x)
               versus the scalar input x.  A fixed encode->scramble pipeline turns
               a single number x into a bank of nonlinear, multi-frequency
               (Fourier-type) curves.  NONE of these are trained.
    Panel (b): the target f(x) = sin(3x) + 0.5 sin(7x), the training samples, and
               the QELM prediction on a dense held-out grid.  Only a ridge-
               regression readout w on the feature vector is fit.  The test R^2 is
               annotated in-panel.
    Panel (c): test R^2 versus the number of data RE-UPLOADS L, at two reservoir
               evolution times.  Expressivity grows with L: the accessible Fourier
               frequencies reach up to L*N (Schuld-Sweke-Meyer), so a single upload
               (max frequency N < 7) cannot resolve the sin(7x) component, while
               two or more can.  This is the encoding-spectrum mechanism of
               Sec.~data_reuploading, made visible.

PROTOCOL (one input x, exact statevector, no external QC library).
    Start from |0>^N.  Repeat L times: apply a FIXED scrambling unitary
    U = exp(-i H t) with H the nonintegrable MIXED-FIELD Ising chain, then a
    data-encoding layer S(x) = prod_i exp(-i x Z_i / 2) (a diagonal phase whose
    generator has integer-spaced eigenvalue differences).  A final scramble mixes
    the encoded phases into measurable coherences.  Read a fixed bank of low-weight
    Pauli expectation values {<X_i>, <Y_i>, <Z_i>, <Z_i Z_{i+1}>} as the feature
    vector f(x).  Train ONLY a ridge readout w to predict the target.

The reservoir (encoding + scrambling) is fixed; only the linear weights w are
learned.  Cached in data/ch7/fig_ch7_qelm_regression.npz.
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
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)


def kron_op(single, site, N):
    op = np.array([[1.0]], dtype=complex)
    for j in range(N):
        op = np.kron(op, single if j == site else I2)
    return op


def mixed_field_ising(N, J, hx, hz):
    """Nonintegrable mixed-field Ising Hamiltonian H = J sum Z_iZ_{i+1}
    + hx sum X_i + hz sum Z_i.  The longitudinal field hz breaks the
    Jordan-Wigner integrability of the pure transverse-field model, so the
    dynamics scramble and supply a rich feature basis."""
    H = np.zeros((2 ** N, 2 ** N), dtype=complex)
    for i in range(N - 1):
        H += J * (kron_op(Z, i, N) @ kron_op(Z, i + 1, N))
    for i in range(N):
        H += hx * kron_op(X, i, N) + hz * kron_op(Z, i, N)
    return H


def encoding_phases(N):
    """Per-basis-state phase generator g_b for the encoding layer
    S(x) = prod_i exp(-i x Z_i / 2).  Acting on a statevector, S(x) multiplies
    basis state b by exp(-i x g_b), with g_b = sum_i (1 - 2 bit_i)/2 the total
    Z/2 eigenvalue.  Its eigenvalue DIFFERENCES are integers 0..N, so L uploads
    give accessible Fourier frequencies up to L*N."""
    b = np.arange(1 << N)
    pop = np.array([bin(i).count("1") for i in range(1 << N)])
    return (N - 2 * pop) / 2.0          # g_b, shape (2^N,)


_POP = np.array([bin(i).count("1") for i in range(1 << 16)])
_PH = np.array([1, 1j, -1, -1j])


def pauli_readout(N):
    """Precompute the low-weight Pauli readout bank as column-index and weight
    arrays.  The bank collects single-site {X_i, Y_i, Z_i}, nearest-neighbour
    {X_iX_{i+1}, Y_iY_{i+1}, Z_iZ_{i+1}}, and all-pairs {Z_iZ_j}.  For a pure
    state psi a Pauli P = i^p X^x Z^z has <P> = Re sum_b conj(psi[b]) * W[b] *
    psi[b^x], an O(2^N) contraction (checked against brute-force <psi|P|psi>).
    Returns (COLS, W, names)."""
    bit = lambda i: 1 << (N - 1 - i)
    specs, names = [], []
    for i in range(N):
        specs += [(bit(i), 0, 0), (bit(i), bit(i), 1), (0, bit(i), 0)]     # X, Y, Z
        names += [f"X{i}", f"Y{i}", f"Z{i}"]
    for i in range(N - 1):
        m = bit(i) | bit(i + 1)
        specs += [(m, 0, 0), (m, m, 2), (0, m, 0)]                         # XX, YY, ZZ
        names += [f"X{i}X{i + 1}", f"Y{i}Y{i + 1}", f"Z{i}Z{i + 1}"]
    for i in range(N):
        for j in range(i + 2, N):
            m = bit(i) | bit(j)
            specs += [(0, m, 0)]                                           # ZZ, all pairs
            names += [f"Z{i}Z{j}"]
    b = np.arange(1 << N)
    COLS = np.array([b ^ x for (x, z, p) in specs])
    W = np.array([_PH[p] * (1 - 2 * (_POP[z & b] & 1)) for (x, z, p) in specs],
                 dtype=complex)
    return COLS, W, names


def qelm_features(xs, N, U, g, COLS, W, L):
    """Feature matrix F (len(xs) x d): the fixed encode->scramble QELM run at each
    input x.  State: |0>^N, then L times [scramble U, encode S(x)], then a final
    scramble U so the encoded phases become measurable coherences.  Read the Pauli
    bank.  Only U (fixed) and the phase generator g depend on the reservoir; no
    parameter here is trained."""
    b = np.arange(1 << N)
    F = np.zeros((len(xs), COLS.shape[0]))
    for n, x in enumerate(xs):
        psi = np.zeros(1 << N, dtype=complex)
        psi[0] = 1.0
        for _ in range(L):
            psi = U @ psi
            psi *= np.exp(-1j * x * g)          # encoding layer S(x), diagonal
        psi = U @ psi
        F[n] = np.real(np.conj(psi) * W * psi[COLS]).sum(axis=1)
    return F


def target(x):
    return np.sin(3 * x) + 0.5 * np.sin(7 * x)


def ridge_fit(F_tr, y_tr, lam):
    """Standardize features on the training split, fit a ridge readout with bias.
    Returns (w, mu, sd) so the same map applies to held-out inputs."""
    mu, sd = F_tr.mean(0), F_tr.std(0) + 1e-12
    Z_tr = np.hstack([(F_tr - mu) / sd, np.ones((len(F_tr), 1))])
    A = Z_tr.T @ Z_tr + lam * np.eye(Z_tr.shape[1])
    w = np.linalg.solve(A, Z_tr.T @ y_tr)
    return w, mu, sd


def ridge_predict(F, w, mu, sd):
    Z = np.hstack([(F - mu) / sd, np.ones((len(F), 1))])
    return Z @ w


def r2_score(y, yhat):
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return 1.0 - ss_res / ss_tot


# ---------------------------------------------------------------------------
N = 5
J, HX, HZ = 1.0, 0.9, 0.8         # nonintegrable mixed-field Ising reservoir
T_EV = 0.6                        # per-layer scrambling time for the main fit
L_MAIN = 2                        # re-uploads for the main fit (max freq L*N = 10 >= 7)
LAM = 1e-8                        # ridge penalty
N_TRAIN = 80                      # training inputs, evenly spaced in [0, 2pi]
N_TEST = 400                      # dense held-out grid
L_SWEEP = [1, 2, 3, 4, 5, 6, 7, 8]   # re-upload sweep for panel (c): rise (Fourier
#                                       ceiling LN passes 7) then fall (over-scrambling
#                                       concentrates the features toward the Haar value)
T_SWEEP = [0.15, 0.6]            # weak vs strong per-layer scrambling for panel (c)
FEATURE_PANEL = ["X0", "Y2", "Z4", "Z0Z1", "Y1"]   # feature curves shown in (a)


def compute():
    H = mixed_field_ising(N, J, HX, HZ)
    g = encoding_phases(N)
    COLS, W, names = pauli_readout(N)

    x_train = np.linspace(0.0, 2 * np.pi, N_TRAIN, endpoint=False)
    x_test = np.linspace(0.0, 2 * np.pi, N_TEST)
    y_train, y_test = target(x_train), target(x_test)

    # Main fit at (L_MAIN, T_EV).
    U = expm(-1j * H * T_EV)
    F_tr = qelm_features(x_train, N, U, g, COLS, W, L_MAIN)
    F_te = qelm_features(x_test, N, U, g, COLS, W, L_MAIN)
    w, mu, sd = ridge_fit(F_tr, y_train, LAM)
    pred_test = ridge_predict(F_te, w, mu, sd)
    r2 = r2_score(y_test, pred_test)
    mse = float(np.mean((y_test - pred_test) ** 2))
    print(f"  main fit: N={N}, L={L_MAIN}, t={T_EV}  ->  test R^2={r2:.4f}  MSE={mse:.3e}")

    # Feature functions for panel (a): reuse the dense-grid feature matrix.
    idx = [names.index(nm) for nm in FEATURE_PANEL]
    feat_curves = F_te[:, idx].T                      # (n_features, N_TEST)

    # Panel (c): test R^2 vs number of re-uploads, for two scrambling times.
    r2_vs_L = np.zeros((len(T_SWEEP), len(L_SWEEP)))
    for ti, tev in enumerate(T_SWEEP):
        Ut = expm(-1j * H * tev)
        for li, L in enumerate(L_SWEEP):
            Ftr = qelm_features(x_train, N, Ut, g, COLS, W, L)
            Fte = qelm_features(x_test, N, Ut, g, COLS, W, L)
            wl, mul, sdl = ridge_fit(Ftr, y_train, LAM)
            r2_vs_L[ti, li] = r2_score(y_test, ridge_predict(Fte, wl, mul, sdl))
        print(f"  sweep t={tev}: R^2(L={L_SWEEP}) = "
              f"{np.round(r2_vs_L[ti], 3).tolist()}")

    return {
        "x_train": x_train, "y_train": y_train,
        "x_test": x_test, "y_test": y_test, "pred_test": pred_test,
        "r2": r2, "mse": mse,
        "feat_curves": feat_curves, "feat_names": np.array(FEATURE_PANEL),
        "r2_vs_L": r2_vs_L, "L_sweep": np.array(L_SWEEP),
        "t_sweep": np.array(T_SWEEP), "N": N, "L_main": L_MAIN, "t_ev": T_EV,
    }


data = load_or_compute(DATA_DIR / "fig_ch7_qelm_regression.npz", compute)

x_train, y_train = data["x_train"], data["y_train"]
x_test, y_test, pred_test = data["x_test"], data["y_test"], data["pred_test"]
r2, mse = float(data["r2"]), float(data["mse"])
feat_curves, feat_names = data["feat_curves"], data["feat_names"]
r2_vs_L, L_sweep, t_sweep = data["r2_vs_L"], data["L_sweep"], data["t_sweep"]
Nq, L_main = int(data["N"]), int(data["L_main"])

# ---------------------------------------------------------------------------
C_TARGET = "#1a1a1a"
C_PRED = "#c1121f"
C_TRAIN = "#1f6fc4"
FEAT_COLORS = ["#1f6fc4", "#e8850c", "#2a9d8f", "#c1121f", "#7b529b"]
SWEEP_COLORS = ["#8aa0b4", "#c1121f"]

apply_book_style()
fig = plt.figure(figsize=(15, 4.4), constrained_layout=True)
gs = fig.add_gridspec(1, 3)
ax_a = fig.add_subplot(gs[0, 0])
ax_b = fig.add_subplot(gs[0, 1])
ax_c = fig.add_subplot(gs[0, 2])

# (a) untrained reservoir feature functions f_alpha(x) = <O_alpha>(x).
for k in range(feat_curves.shape[0]):
    nm = str(feat_names[k])
    lab = rf"$\langle {nm[0]}_{{{nm[1]}}}\rangle$" if len(nm) == 2 \
        else rf"$\langle Z_{{{nm[1]}}}Z_{{{nm[3]}}}\rangle$"
    ax_a.plot(x_test, feat_curves[k], color=FEAT_COLORS[k % len(FEAT_COLORS)],
              lw=1.4, label=lab)
ax_a.set_xlabel(r"input $x$")
ax_a.set_ylabel(r"feature $\langle \hat{O}_\alpha\rangle(x)$")
ax_a.set_xlim(0, 2 * np.pi)
ax_a.set_xticks([0, np.pi, 2 * np.pi])
ax_a.set_xticklabels(["0", r"$\pi$", r"$2\pi$"])
ax_a.legend(loc="lower left", fontsize=8, ncol=2, framealpha=0.9)
panel_label(ax_a, "a", loc="upper right")

# (b) target, training samples, and QELM prediction; R^2 annotated in-panel.
ax_b.plot(x_test, y_test, color=C_TARGET, lw=2.0, zorder=2, label="target $f(x)$")
ax_b.plot(x_test, pred_test, color=C_PRED, lw=1.7, ls="--", zorder=3,
          label="QELM readout")
ax_b.scatter(x_train, y_train, s=34, color=C_TRAIN, edgecolors="white",
             linewidths=0.5, zorder=4, label="training samples")
ax_b.set_xlabel(r"input $x$")
ax_b.set_ylabel(r"$f(x) = \sin 3x + \frac{1}{2}\, \sin 7x$")
ax_b.set_xlim(0, 2 * np.pi)
ax_b.set_xticks([0, np.pi, 2 * np.pi])
ax_b.set_xticklabels(["0", r"$\pi$", r"$2\pi$"])
_mexp = int(np.floor(np.log10(mse)))
_mman = mse / 10.0 ** _mexp
ax_b.text(0.04, 0.07, rf"test $R^2={r2:.3f}$" + "\n"
          + rf"MSE $={_mman:.1f}\times 10^{{{_mexp}}}$",
          transform=ax_b.transAxes, fontsize=10,
          bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", alpha=0.9))
ax_b.legend(loc="upper right", fontsize=8, framealpha=0.9)
panel_label(ax_b, "b", loc="upper left")

# (c) expressivity grows with re-uploads: test R^2 vs L at two scrambling times.
SWEEP_LABELS = ["weak", "strong"]
for ti in range(len(t_sweep)):
    ax_c.plot(L_sweep, r2_vs_L[ti], "-o", color=SWEEP_COLORS[ti % len(SWEEP_COLORS)],
              lw=1.7, ms=7, mec="white", mew=0.6,
              label=rf"{SWEEP_LABELS[ti]} ($t={float(t_sweep[ti]):.2f}$)")
ax_c.axhline(1.0, color="0.6", ls=":", lw=1.0, zorder=1)
ax_c.axvspan(0.5, 1.5, color="0.85", alpha=0.5, zorder=0)
ax_c.text(1.0, 0.03, rf"$LN={Nq}<7$" + "\n" + "misses $\\sin 7x$",
          transform=ax_c.get_xaxis_transform(), ha="center", va="bottom", fontsize=8)
ax_c.set_xlabel(r"number of re-uploads $L$")
ax_c.set_ylabel(r"test $R^2$")
ax_c.set_xticks(list(L_sweep))
ax_c.set_ylim(-0.05, 1.08)
ax_c.legend(loc="lower right", fontsize=8.5, framealpha=0.9,
            title="per-layer scrambling", title_fontsize=8)
panel_label(ax_c, "c", loc="upper left")

plt.savefig(OUTPUT_DIR / "fig_ch7_qelm_regression.pdf")
plt.savefig(OUTPUT_DIR / "fig_ch7_qelm_regression.png")
print(f"  test R^2 = {r2:.4f}, MSE = {mse:.3e}")
print(f"  R^2 vs L (t={t_sweep.tolist()}):")
for ti in range(len(t_sweep)):
    print(f"    t={float(t_sweep[ti]):.1f}: {np.round(r2_vs_L[ti], 3).tolist()}")
print(f"  Saved: {OUTPUT_DIR / 'fig_ch7_qelm_regression.pdf'}")
