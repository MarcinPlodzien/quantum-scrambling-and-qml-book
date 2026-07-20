#!/usr/bin/env python3
"""
get_fig_ch7_qelm_xor.py
=======================
The XOR problem with a QUANTUM EXTREME LEARNING MACHINE (QELM).

Produces Fig. (ch7) -- figures/ch7/fig_ch7_qelm_xor.pdf (+png).

    Panel (a): the continuous XOR / checkerboard dataset.  Four Gaussian clusters
               centred near (+/-1, +/-1); the label is y = sign(x1 x2), so the two
               classes interlock in a checkerboard.  No straight line separates
               them: XOR is not linearly separable.
    Panel (b): decision regions of the QELM linear readout in the (x1, x2) plane.
               A fixed quantum feature map lifts (x1, x2) into a bank of Pauli
               expectation values in which XOR becomes linearly separable, so the
               trained linear readout recovers the checkerboard.  The straight
               dashed line is the best raw-input linear classifier, which cannot
               bend.  The raw-linear and QELM test accuracies are annotated
               in-panel.

PROTOCOL (one input (x1, x2), exact statevector, no external QC library).
    Angle-encode the two coordinates into an N=4 qubit register: qubits 0,1 carry
    x1 and qubits 2,3 carry x2 through single-qubit rotations R_y(alpha x).  Apply
    ONE fixed, untrained scrambling unitary U = exp(-i H t) with H the nonintegrable
    MIXED-FIELD Ising chain.  Read a fixed bank of weight-one Pauli expectation
    values {<X_i>, <Y_i>, <Z_i>} as the feature vector f(x1, x2).  Train ONLY a
    linear classifier (logistic regression) on f.

ABLATION.  The SAME logistic-regression classifier is fit three ways:
    raw     -- on the raw inputs (x1, x2): ~50% (XOR is not linearly separable);
    bare    -- on weight-one Pauli features of the UNSCRAMBLED encoded state
               (each such feature depends on x1 OR x2 alone, so the features are
               additive across coordinates and still cannot express XOR): ~50%;
    QELM    -- on the SAME features after the fixed scrambling U, which routes the
               two-coordinate correlation into weight-one observables: ~98%.
The fixed dynamics, not the readout, supply the nonlinear feature that solves XOR.

Cached in data/ch7/fig_ch7_qelm_xor.npz.
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.linalg import expm
from sklearn.linear_model import LogisticRegression

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
    dynamics scramble and mix the two encoded coordinates into every qubit."""
    H = np.zeros((2 ** N, 2 ** N), dtype=complex)
    for i in range(N - 1):
        H += J * (kron_op(Z, i, N) @ kron_op(Z, i + 1, N))
    for i in range(N):
        H += hx * kron_op(X, i, N) + hz * kron_op(Z, i, N)
    return H


# qubit -> which coordinate it encodes (0,1 -> x1 ; 2,3 -> x2)
QUBIT_COORD = [0, 0, 1, 1]


def encode(x1, x2, N, alpha):
    """Angle-encode (x1, x2) into a product state on N qubits: R_y(alpha x)|0>
    per qubit, with qubits 0,1 carrying x1 and qubits 2,3 carrying x2.  A single
    R_y gives <Z>=cos(alpha x), <X>=sin(alpha x), <Y>=0, all functions of ONE
    coordinate, so the bare state has no x1-x2 cross term to read out."""
    coords = (x1, x2)
    psi = np.array([1.0], dtype=complex)
    for j in range(N):
        th = alpha * coords[QUBIT_COORD[j]]
        qubit = np.array([np.cos(th / 2), np.sin(th / 2)], dtype=complex)
        psi = np.kron(psi, qubit)
    return psi


_POP = np.array([bin(i).count("1") for i in range(1 << 16)])
_PH = np.array([1, 1j, -1, -1j])


def pauli_readout(N):
    """Precompute the weight-one readout bank {X_i, Y_i, Z_i} as column-index and
    weight arrays.  For a pure state psi a Pauli P = i^p X^x Z^z has expectation
    <P> = Re sum_b conj(psi[b]) W[b] psi[b^x], an O(2^N) contraction (checked
    against brute-force <psi|P|psi>).  Returns (COLS, W, names)."""
    bit = lambda i: 1 << (N - 1 - i)
    specs, names = [], []
    for i in range(N):
        specs += [(bit(i), 0, 0), (bit(i), bit(i), 1), (0, bit(i), 0)]   # X, Y, Z
        names += [f"X{i}", f"Y{i}", f"Z{i}"]
    b = np.arange(1 << N)
    COLS = np.array([b ^ x for (x, z, p) in specs])
    W = np.array([_PH[p] * (1 - 2 * (_POP[z & b] & 1)) for (x, z, p) in specs],
                 dtype=complex)
    return COLS, W, names


def qelm_features(XY, N, U, COLS, W, alpha):
    """Feature matrix F (len(XY) x d): the fixed encode->scramble QELM run at each
    input (x1, x2).  If U is None the bare encoded state is read (no scrambling).
    Only the fixed U enters; no parameter here is trained."""
    F = np.zeros((len(XY), COLS.shape[0]))
    for n, (x1, x2) in enumerate(XY):
        psi = encode(x1, x2, N, alpha)
        if U is not None:
            psi = U @ psi
        F[n] = np.real(np.conj(psi) * W * psi[COLS]).sum(axis=1)
    return F


def make_xor(rng, n_per, noise):
    """Continuous XOR / checkerboard: four Gaussian clusters near (+/-1, +/-1),
    label y = sign(x1 x2) in {0, 1}.  Returns shuffled (XY, y)."""
    centers = [(1.0, 1.0), (1.0, -1.0), (-1.0, 1.0), (-1.0, -1.0)]
    XY, y = [], []
    for (cx, cy) in centers:
        XY.append(rng.normal([cx, cy], noise, size=(n_per, 2)))
        y += [1 if cx * cy > 0 else 0] * n_per
    XY, y = np.vstack(XY), np.array(y)
    idx = rng.permutation(len(y))
    return XY[idx], y[idx]


def fit_score(F_tr, y_tr, F_te, y_te, C=10.0):
    clf = LogisticRegression(max_iter=5000, C=C)
    clf.fit(F_tr, y_tr)
    return clf, float(clf.score(F_te, y_te))


# ---------------------------------------------------------------------------
N = 4
J, HX, HZ = 1.0, 0.9, 0.8       # nonintegrable mixed-field Ising scrambler
T_EV = 2.0                      # fixed scrambling time of the untrained unitary U
ALPHA = 1.0                     # angle-encoding scale R_y(alpha x)
N_PER = 120                     # points per cluster (480 total)
NOISE = 0.35                    # cluster standard deviation
TRAIN_FRAC = 0.6
GRID_N = 220                    # decision-region grid resolution per axis
GRID_LIM = 2.6
SEED = 0


def compute():
    rng = np.random.default_rng(SEED)
    XY, y = make_xor(rng, N_PER, NOISE)
    ntr = int(TRAIN_FRAC * len(y))
    XY_tr, y_tr, XY_te, y_te = XY[:ntr], y[:ntr], XY[ntr:], y[ntr:]

    H = mixed_field_ising(N, J, HX, HZ)
    U = expm(-1j * H * T_EV)
    COLS, W, names = pauli_readout(N)

    # Three fits of the SAME logistic-regression classifier.
    raw_clf, acc_raw = fit_score(XY_tr, y_tr, XY_te, y_te)
    F_tr_b = qelm_features(XY_tr, N, None, COLS, W, ALPHA)
    F_te_b = qelm_features(XY_te, N, None, COLS, W, ALPHA)
    _, acc_bare = fit_score(F_tr_b, y_tr, F_te_b, y_te)
    F_tr_s = qelm_features(XY_tr, N, U, COLS, W, ALPHA)
    F_te_s = qelm_features(XY_te, N, U, COLS, W, ALPHA)
    qelm_clf, acc_qelm = fit_score(F_tr_s, y_tr, F_te_s, y_te)

    print(f"  N={N}, t={T_EV}, alpha={ALPHA}, d={len(names)} features")
    print(f"  raw-linear   test accuracy = {acc_raw:.3f}")
    print(f"  bare-encode  test accuracy = {acc_bare:.3f}")
    print(f"  QELM         test accuracy = {acc_qelm:.3f}")

    # Decision regions over a dense (x1, x2) grid.
    g = np.linspace(-GRID_LIM, GRID_LIM, GRID_N)
    gx, gy = np.meshgrid(g, g)
    grid_pts = np.column_stack([gx.ravel(), gy.ravel()])
    F_grid = qelm_features(grid_pts, N, U, COLS, W, ALPHA)
    prob_qelm = qelm_clf.predict_proba(F_grid)[:, 1].reshape(gx.shape)
    prob_raw = raw_clf.predict_proba(grid_pts)[:, 1].reshape(gx.shape)

    return {
        "XY": XY, "y": y, "ntr": ntr,
        "acc_raw": acc_raw, "acc_bare": acc_bare, "acc_qelm": acc_qelm,
        "grid": g, "prob_qelm": prob_qelm, "prob_raw": prob_raw,
        "N": N, "t_ev": T_EV, "alpha": ALPHA, "n_feat": len(names),
    }


data = load_or_compute(DATA_DIR / "fig_ch7_qelm_xor.npz", compute)

XY, y, ntr = data["XY"], data["y"], int(data["ntr"])
acc_raw, acc_bare, acc_qelm = (float(data["acc_raw"]), float(data["acc_bare"]),
                               float(data["acc_qelm"]))
g = data["grid"]
prob_qelm, prob_raw = data["prob_qelm"], data["prob_raw"]

# ---------------------------------------------------------------------------
C_POS = "#c1121f"     # class y=+1  (sign x1 x2 > 0)
C_NEG = "#1f6fc4"     # class y= 0  (sign x1 x2 < 0)
REGION_CMAP = ListedColormap(["#dbe7f3", "#f6dcdc"])   # light blue / light red

apply_book_style()
fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(10.6, 4.3), constrained_layout=True)

# (a) the XOR / checkerboard dataset.
for lab, col, mk in [(1, C_POS, "o"), (0, C_NEG, "s")]:
    m = y == lab
    ax_a.scatter(XY[m, 0], XY[m, 1], s=32, color=col, marker=mk,
                 edgecolors="white", linewidths=0.4,
                 label=r"$y=+1$" if lab == 1 else r"$y=-1$")
ax_a.axhline(0.0, color="0.6", lw=0.8, ls=":")
ax_a.axvline(0.0, color="0.6", lw=0.8, ls=":")
ax_a.set_xlabel(r"$x_1$")
ax_a.set_ylabel(r"$x_2$")
ax_a.set_xlim(-GRID_LIM, GRID_LIM)
ax_a.set_ylim(-GRID_LIM, GRID_LIM)
ax_a.set_aspect("equal", "box")
ax_a.legend(loc="upper right", fontsize=9, framealpha=0.9,
            title=r"$y=\mathrm{sign}(x_1 x_2)$", title_fontsize=8)
panel_label(ax_a, "a", loc="upper left")

# (b) QELM decision regions; raw-linear boundary overlaid; accuracies in-panel.
ax_b.pcolormesh(g, g, (prob_qelm > 0.5).astype(int), cmap=REGION_CMAP,
                shading="auto", zorder=0)
ax_b.contour(g, g, prob_qelm, levels=[0.5], colors="0.25", linewidths=1.4,
             zorder=2)
ax_b.contour(g, g, prob_raw, levels=[0.5], colors="0.25", linestyles="--",
             linewidths=1.4, zorder=2)
for lab, col, mk in [(1, C_POS, "o"), (0, C_NEG, "s")]:
    m = y == lab
    ax_b.scatter(XY[m, 0], XY[m, 1], s=24, color=col, marker=mk,
                 edgecolors="white", linewidths=0.3, alpha=0.85, zorder=3)
# proxy handles for the two decision boundaries
from matplotlib.lines import Line2D
ax_b.legend(handles=[
        Line2D([0], [0], color="0.25", lw=1.4, label="QELM readout"),
        Line2D([0], [0], color="0.25", lw=1.4, ls="--", label="raw linear")],
    loc="upper right", fontsize=8, framealpha=0.9)
ax_b.text(0.035, 0.045,
          f"raw-linear acc. $= {acc_raw:.2f}$\nQELM acc. $= {acc_qelm:.2f}$",
          transform=ax_b.transAxes, fontsize=10, va="bottom",
          bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.7", alpha=0.92))
ax_b.set_xlabel(r"$x_1$")
ax_b.set_ylabel(r"$x_2$")
ax_b.set_xlim(-GRID_LIM, GRID_LIM)
ax_b.set_ylim(-GRID_LIM, GRID_LIM)
ax_b.set_aspect("equal", "box")
panel_label(ax_b, "b", loc="upper left")

plt.savefig(OUTPUT_DIR / "fig_ch7_qelm_xor.pdf")
plt.savefig(OUTPUT_DIR / "fig_ch7_qelm_xor.png")
print(f"  raw-linear accuracy = {acc_raw:.3f}")
print(f"  bare-encoding accuracy = {acc_bare:.3f}  (no scrambling)")
print(f"  QELM accuracy = {acc_qelm:.3f}")
print(f"  scrambling helps: {acc_qelm > acc_bare + 0.05}")
print(f"  Saved: {OUTPUT_DIR / 'fig_ch7_qelm_xor.pdf'}")
