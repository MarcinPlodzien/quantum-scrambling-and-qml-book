#!/usr/bin/env python3
"""
get_fig_ch7_qelm_phase.py
=========================
Classifying QUANTUM PHASES with a QUANTUM EXTREME LEARNING MACHINE (QELM).

Produces Fig. (ch7) -- figures/ch7/fig_ch7_qelm_phase.pdf (+png).

    Panel (a): a 2D PCA of the QELM feature vectors.  Each point is the ground
               state of the transverse-field Ising model (TFIM) at one field g,
               pushed through a FIXED scramble-then-measure feature map.  The
               clearly-ordered and clearly-disordered training states fall into
               two separated clusters; the dense held-out sweep threads a path
               between them, crossing the gap right at the transition.
    Panel (b): the trained linear readout's predicted paramagnetic-phase
               probability P(disordered | g) versus g across the full sweep.  The
               true critical point g_c = 1 is marked; the decision crossing
               P = 1/2 lands on it, even though the classifier was trained only
               far from criticality and never saw an order parameter.  The
               held-out accuracy is annotated in-panel.

MODEL.  TFIM on an open chain of N spins,
    H(g) = -J sum_i Z_i Z_{i+1} - g sum_i X_i,
with a quantum phase transition at g/J = 1: ferromagnetic (ordered) for g<1,
paramagnetic (disordered) for g>1.  The input to the QELM is the exact ground
state |psi_gs(g)> (dense diagonalization).

PROTOCOL (one input g).  Take |psi_gs(g)>, evolve it under a FIXED, untrained,
nonintegrable scrambling unitary U = exp(-i H_res t) with H_res the MIXED-FIELD
Ising chain H_res = J_r sum Z_iZ_{i+1} + h_x sum X_i + h_z sum Z_i (the
longitudinal h_z breaks the Jordan-Wigner integrability of the pure transverse
model, so the dynamics scramble).  Read a FIXED bank of low-weight Pauli
expectation values f_alpha = <psi_gs| U^dag O_alpha U |psi_gs| as the feature
vector.  Train ONLY a linear (logistic) readout to predict the phase label.

KEY POINT.  The QELM never measures the magnetization order parameter.  A generic
scramble-then-measure feature map plus a linear readout still separates the two
phases, and the readout's decision crosses at the true g_c = 1.  The features are
LINEAR in rho, so a linear readout is well posed.  A nonlinear functional of the
state such as the entanglement entropy could NOT be recovered this way from
single-copy expectation values, no matter how the readout is trained.

Cached in data/ch7/fig_ch7_qelm_phase.npz.
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.linalg import expm, eigh

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


def tfim_hamiltonian(N, J, g):
    """Transverse-field Ising model on an OPEN chain:
    H(g) = -J sum_i Z_i Z_{i+1} - g sum_i X_i.
    Quantum critical point at g/J = 1."""
    H = np.zeros((2 ** N, 2 ** N), dtype=complex)
    for i in range(N - 1):
        H += -J * (kron_op(Z, i, N) @ kron_op(Z, i + 1, N))
    for i in range(N):
        H += -g * kron_op(X, i, N)
    return H


def ground_state(N, J, g):
    """Lowest-energy eigenvector of the TFIM at field g (dense diagonalization).
    The open-chain ground state is nondegenerate for g>0, so this is well defined
    and Z2-symmetric (unbroken parity in finite size)."""
    H = tfim_hamiltonian(N, J, g)
    w, v = eigh(H)
    return v[:, 0].astype(complex)


def mixed_field_ising(N, Jr, hx, hz):
    """Nonintegrable mixed-field Ising reservoir Hamiltonian
    H_res = Jr sum Z_iZ_{i+1} + hx sum X_i + hz sum Z_i.  The longitudinal field
    hz breaks the integrability of the pure transverse model, so exp(-i H_res t)
    genuinely scrambles the input state into its low-weight observables."""
    H = np.zeros((2 ** N, 2 ** N), dtype=complex)
    for i in range(N - 1):
        H += Jr * (kron_op(Z, i, N) @ kron_op(Z, i + 1, N))
    for i in range(N):
        H += hx * kron_op(X, i, N) + hz * kron_op(Z, i, N)
    return H


_POP = np.array([bin(i).count("1") for i in range(1 << 16)])
_PH = np.array([1, 1j, -1, -1j])


def pauli_readout(N):
    """Precompute a FIXED bank of low-weight Pauli observables
    {X_i, Y_i, Z_i} and nearest-neighbour {X_iX_{i+1}, Y_iY_{i+1}, Z_iZ_{i+1}}
    as column-index and weight arrays.  For a pure state psi a Pauli
    P = i^p X^x Z^z has <P> = Re sum_b conj(psi[b]) W[b] psi[b^x], an O(2^N)
    contraction (checked against brute-force <psi|P|psi>).  Returns (COLS, W,
    names)."""
    bit = lambda i: 1 << (N - 1 - i)
    specs, names = [], []
    for i in range(N):
        specs += [(bit(i), 0, 0), (bit(i), bit(i), 1), (0, bit(i), 0)]      # X, Y, Z
        names += [f"X{i}", f"Y{i}", f"Z{i}"]
    for i in range(N - 1):
        m = bit(i) | bit(i + 1)
        specs += [(m, 0, 0), (m, m, 2), (0, m, 0)]                          # XX, YY, ZZ
        names += [f"X{i}X{i+1}", f"Y{i}Y{i+1}", f"Z{i}Z{i+1}"]
    b = np.arange(1 << N)
    COLS = np.array([b ^ x for (x, z, p) in specs])
    W = np.array([_PH[p] * (1 - 2 * (_POP[z & b] & 1)) for (x, z, p) in specs],
                 dtype=complex)
    return COLS, W, names


def qelm_features(psi_list, U, COLS, W):
    """Feature matrix F (n_inputs x d): for each ground state psi, evolve under the
    FIXED scrambling unitary, phi = U psi, then read the fixed Pauli bank
    f_alpha = <phi|O_alpha|phi> = <psi|U^dag O_alpha U|psi>.  Nothing here is
    trained; U and the readout set are fixed once and for all."""
    F = np.zeros((len(psi_list), COLS.shape[0]))
    for n, psi in enumerate(psi_list):
        phi = U @ psi
        F[n] = np.real(np.conj(phi) * W * phi[COLS]).sum(axis=1)
    return F


# ---------------------------------------------------------------------------
# Model / protocol parameters.
N = 8                       # open TFIM chain length (2^N = 256 dim)
J = 1.0                     # Ising coupling; critical point at g_c = J = 1
JR, HX, HZ = 1.0, 1.05, 0.5  # nonintegrable mixed-field Ising reservoir
T_EV = 2.5                  # fixed scrambling time in U = exp(-i H_res t)

# Training fields, drawn ONLY from the clearly-ordered and clearly-disordered
# regions, well away from the critical point.
G_TRAIN_ORD = np.linspace(0.10, 0.65, 10)   # ferromagnetic, label 0
G_TRAIN_DIS = np.linspace(1.35, 2.40, 10)   # paramagnetic,  label 1
# Independent held-out fields, same clear regions, for the reported accuracy.
G_TEST_ORD = np.linspace(0.14, 0.61, 8)
G_TEST_DIS = np.linspace(1.40, 2.35, 8)
# Dense sweep across the transition (includes the critical region) for panel (b).
G_SWEEP = np.linspace(0.02, 2.5, 120)


def compute():
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    COLS, W, names = pauli_readout(N)
    U = expm(-1j * mixed_field_ising(N, JR, HX, HZ) * T_EV)

    def feats(gs):
        return qelm_features([ground_state(N, J, g) for g in gs], U, COLS, W)

    g_train = np.concatenate([G_TRAIN_ORD, G_TRAIN_DIS])
    y_train = np.concatenate([np.zeros(len(G_TRAIN_ORD)), np.ones(len(G_TRAIN_DIS))])
    g_test = np.concatenate([G_TEST_ORD, G_TEST_DIS])
    y_test = np.concatenate([np.zeros(len(G_TEST_ORD)), np.ones(len(G_TEST_DIS))])

    F_train = feats(g_train)
    F_test = feats(g_test)
    F_sweep = feats(G_SWEEP)

    # Standardize on the training features, fit the LINEAR (logistic) readout.
    scaler = StandardScaler().fit(F_train)
    clf = LogisticRegression(C=1.0, max_iter=5000)
    clf.fit(scaler.transform(F_train), y_train)

    acc_test = float(clf.score(scaler.transform(F_test), y_test))
    acc_train = float(clf.score(scaler.transform(F_train), y_train))
    prob_sweep = clf.predict_proba(scaler.transform(F_sweep))[:, 1]  # P(disordered)

    # Decision crossing P = 1/2 by linear interpolation of the sweep.
    cross = np.where(np.diff(np.sign(prob_sweep - 0.5)) != 0)[0]
    if len(cross):
        k = cross[0]
        p0, p1 = prob_sweep[k], prob_sweep[k + 1]
        g_cross = float(G_SWEEP[k] + (0.5 - p0) / (p1 - p0) * (G_SWEEP[k + 1] - G_SWEEP[k]))
    else:
        g_cross = float("nan")

    # 2D PCA of the feature vectors: fit on training, project everything.
    pca = PCA(n_components=2).fit(scaler.transform(F_train))
    P_train = pca.transform(scaler.transform(F_train))
    P_sweep = pca.transform(scaler.transform(F_sweep))

    print(f"  N={N}, reservoir (Jr,hx,hz)=({JR},{HX},{HZ}), t={T_EV}, "
          f"d={F_train.shape[1]} features")
    print(f"  train accuracy={acc_train:.3f}, held-out test accuracy={acc_test:.3f}")
    print(f"  decision crossing P=1/2 at g={g_cross:.4f}  (true g_c=1)")

    return {
        "g_train": g_train, "y_train": y_train,
        "g_sweep": G_SWEEP, "prob_sweep": prob_sweep,
        "P_train": P_train, "P_sweep": P_sweep,
        "acc_test": acc_test, "acc_train": acc_train, "g_cross": g_cross,
        "evr": pca.explained_variance_ratio_,
        "N": N, "t_ev": T_EV,
    }


data = load_or_compute(DATA_DIR / "fig_ch7_qelm_phase.npz", compute)

g_train, y_train = data["g_train"], data["y_train"]
g_sweep, prob_sweep = data["g_sweep"], data["prob_sweep"]
P_train, P_sweep = data["P_train"], data["P_sweep"]
acc_test, g_cross = float(data["acc_test"]), float(data["g_cross"])
evr = data["evr"]

# ---------------------------------------------------------------------------
C_ORD = "#1f6fc4"     # ferromagnetic / ordered
C_DIS = "#c1121f"     # paramagnetic / disordered
C_SWEEP = "#444444"
C_GC = "#2a9d8f"

apply_book_style()
fig = plt.figure(figsize=(10.5, 4.3), constrained_layout=True)
gs = fig.add_gridspec(1, 2)
ax_a = fig.add_subplot(gs[0, 0])
ax_b = fig.add_subplot(gs[0, 1])

# (a) 2D PCA of the feature vectors.  Two training clusters (ordered vs
#     disordered) plus the dense sweep threading between them, coloured by g.
ord_m = y_train < 0.5
sc = ax_a.scatter(P_sweep[:, 0], P_sweep[:, 1], c=g_sweep, cmap="viridis",
                  s=30, alpha=0.85, edgecolors="none", zorder=1)
# locate the critical point on the manifold. The transition is second order
# (continuous), so the arc is smooth here: PCA shows geometry, not the boundary.
i_c = int(np.argmin(np.abs(g_sweep - 1.0)))
ax_a.plot(P_sweep[i_c, 0], P_sweep[i_c, 1], marker="*", color="black", ms=17,
          mec="white", mew=0.8, zorder=5, label=r"critical point $g_c=1$")
ax_a.scatter(P_train[ord_m, 0], P_train[ord_m, 1], s=70, color=C_ORD,
             marker="o", edgecolors="white", linewidths=0.7, zorder=3,
             label="train: ordered $g<1$")
ax_a.scatter(P_train[~ord_m, 0], P_train[~ord_m, 1], s=70, color=C_DIS,
             marker="s", edgecolors="white", linewidths=0.7, zorder=3,
             label="train: disordered $g>1$")
ax_a.set_xlabel(f"PC 1  ({100*evr[0]:.0f}% var.)")
ax_a.set_ylabel(f"PC 2  ({100*evr[1]:.0f}% var.)")
cb = fig.colorbar(sc, ax=ax_a, pad=0.02)
cb.set_label(r"field $g$")
ax_a.legend(loc="lower left", fontsize=8, framealpha=0.92)
panel_label(ax_a, "a", loc="upper right")

# (b) predicted paramagnetic-phase probability vs g, with the true g_c=1 marked
#     and the decision crossing annotated.  Accuracy reported in-panel.
ax_b.axvspan(0.0, 1.0, color=C_ORD, alpha=0.06, zorder=0)
ax_b.axvspan(1.0, 2.5, color=C_DIS, alpha=0.06, zorder=0)
ax_b.axhline(0.5, color="0.6", ls=":", lw=1.0, zorder=1)
ax_b.axvline(1.0, color=C_GC, ls="--", lw=1.5, zorder=2, label=r"true $g_c=1$")
ax_b.plot(g_sweep, prob_sweep, color=C_SWEEP, lw=2.0, zorder=3)
ax_b.plot([g_cross], [0.5], marker="o", color=C_SWEEP, ms=7, mec="white",
          mew=0.8, zorder=4)
# Training fields shown as a rug along the bottom.
ax_b.scatter(g_train[y_train < 0.5], np.full((y_train < 0.5).sum(), 0.02),
             marker="|", s=70, color=C_ORD, zorder=4, label="training fields")
ax_b.scatter(g_train[y_train >= 0.5], np.full((y_train >= 0.5).sum(), 0.02),
             marker="|", s=70, color=C_DIS, zorder=4)
ax_b.set_xlabel(r"transverse field $g/J$")
ax_b.set_ylabel(r"predicted $P(\mathrm{paramagnet}\mid g)$")
ax_b.set_xlim(0.0, 2.5)
ax_b.set_ylim(-0.03, 1.03)
ax_b.text(0.04, 0.90,
          rf"crossing at $g={g_cross:.2f}$" + "\n" + rf"test acc. $={acc_test*100:.0f}\%$",
          transform=ax_b.transAxes, fontsize=10, va="top",
          bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", alpha=0.92))
ax_b.text(0.13, 0.40, "ordered", color=C_ORD, fontsize=9, ha="center",
          transform=ax_b.transAxes)
ax_b.text(0.80, 0.60, "disordered", color=C_DIS, fontsize=9, ha="center",
          transform=ax_b.transAxes)
ax_b.legend(loc="center right", fontsize=8, framealpha=0.92)
panel_label(ax_b, "b", loc="lower right")

plt.savefig(OUTPUT_DIR / "fig_ch7_qelm_phase.pdf")
plt.savefig(OUTPUT_DIR / "fig_ch7_qelm_phase.png")
print(f"  test accuracy = {acc_test:.3f}, decision crossing g = {g_cross:.3f} (g_c=1)")
print(f"  Saved: {OUTPUT_DIR / 'fig_ch7_qelm_phase.pdf'}")
