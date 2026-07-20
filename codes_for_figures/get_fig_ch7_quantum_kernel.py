#!/usr/bin/env python3
"""
get_fig_ch7_quantum_kernel.py
=============================
Quantum-kernel classification of three concentric rings for Chapter 7
(Quantum Machine Learning).

Produces Fig. (ch7) -- figures/ch7/fig_ch7_quantum_kernel.pdf

    Panel (a): the raw three-ring dataset in the (x, y) plane, coloured by class.
               Inner / middle / outer annulus; the classes are concentric and
               therefore NOT linearly separable in their native coordinates.
    Panel (b): the sorted M x M quantum kernel (Gram) matrix K_ij.  With samples
               ordered by class it is visibly block-diagonal: within-class overlaps
               are large, between-class overlaps are small.
    Panel (c): the decision regions of a classical SVM trained on the precomputed
               quantum kernel, drawn by evaluating the kernel of a dense (x, y)
               grid against the training states.  The boundary is a set of closed
               curves that a linear model could never draw in (x, y).
    Panel (d): a 2-D kernel-PCA embedding of the training set.  The quantum feature
               map unfolds the concentric rings into three compact, linearly
               separable clusters.

FEATURE MAP.  An IQP-style data-encoding unitary on N = 3 qubits,

    U(x) = ( U_Z(x) . H^{otimes N} )^L ,

built from a 6-component feature vector f(x, y) = s * (x, y, r, r, r^2, r^2), with
r = sqrt(x^2+y^2).  Each layer is a wall of Hadamards followed by the diagonal
encoding

    U_Z(x) = exp[ -i ( sum_k f_k Z_k + sum_{k<l} f_k f_l Z_k Z_l ) ] ,

so the single-qubit rotations e^{-i f_k Z_k} and the entangling rotations
e^{-i f_k f_l Z_k Z_l} inject the radius into the ZZ phases -- exactly the
nonlinearity the concentric rings need.  The radius is carried by dedicated qubits
at both r and r^2: the quadratic feature breaks the phase aliasing that a purely
linear radial encoding suffers (equally spaced radii r = 1, 2, 3 would collapse
under 2 pi periodicity), so all three rings resolve.  Because U_Z is diagonal in the
computational basis, the feature state |phi(x)> = U(x)|0> is obtained by
alternating an elementwise phase multiply with a Hadamard (Walsh) transform, fully
vectorized over a batch of points; no external quantum-computing library is used.

QUANTUM KERNEL.  The fidelity kernel is the state overlap

    k(x_i, x_j) = |<phi(x_i)|phi(x_j)>|^2 = | <0| U^dag(x_i) U(x_j) |0> |^2 ,

so the whole M x M Gram matrix follows from the M feature states by one matrix
product.  A classical sklearn SVC(kernel='precomputed') then solves the convex SVM
dual; the quantum part only supplies the kernel.

Exact statevector throughout, N = 6 qubits, seeded RNG.  The dataset, kernels, SVM
grid decision map and kernel-PCA embedding are cached in
data/ch7/fig_ch7_quantum_kernel.npz.
"""
import sys
from pathlib import Path
from itertools import product

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.svm import SVC
from sklearn.decomposition import KernelPCA

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_style import apply_book_style, load_or_compute, panel_label

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR.parent / "figures" / "ch7"
DATA_DIR = SCRIPT_DIR / "data" / "ch7"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Model / dataset hyperparameters (all fixed; the RNG seed makes the run exact).
# ---------------------------------------------------------------------------
N_QUBITS = 6            # 6 features (x, y, r, r, r^2, r^2) -> one qubit each
N_LAYERS = 1            # depth L of the IQP feature map
SCALE = 0.2             # data-to-phase scaling s; keeps phases from wrapping (clean
#                         bands, no kernel concentration: off-diagonals stay ~0.2)
RADII = (1.0, 2.0, 3.0)  # inner / middle / outer ring
NOISE = 0.14            # radial Gaussian width of each annulus
N_TRAIN_PER = 120       # training points per class  (M = 360): dense, well-resolved rings
N_TEST_PER = 40         # test points per class      (120)
SVM_C = 10.0            # SVM soft-margin regularization
GRID_N = 240            # resolution of the (x, y) decision-region grid
SEED = 7

CLASS_NAMES = ("inner", "middle", "outer")
CLASS_COLORS = ("#1f6fc4", "#2a9d5c", "#c1121f")


# ---------------------------------------------------------------------------
# Feature map and quantum kernel (exact statevector, numpy only).
# ---------------------------------------------------------------------------
def hadamard_wall(N):
    """The full N-qubit Hadamard H^{otimes N} as a (2^N x 2^N) matrix."""
    H1 = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
    H = np.array([[1.0]], dtype=complex)
    for _ in range(N):
        H = np.kron(H, H1)
    return H


def basis_signs(N):
    """Row b, column k holds (-1)^{bit_k(b)}, i.e. the eigenvalue of Z_k on |b>."""
    return np.array(list(product([1.0, -1.0], repeat=N)))


def feature_vectors(XY):
    """Map plane points (x, y) to the 6-component encoding f = s*(x, y, r, r, r^2, r^2).

    Coordinates on two qubits fix the angular structure; the radius on four qubits
    (two linear, two quadratic) makes the discriminating variable dominant, and the
    r^2 pair breaks the r = 1, 2, 3 phase aliasing of a linear radial encoding.
    """
    x, y = XY[:, 0], XY[:, 1]
    r = np.sqrt(x ** 2 + y ** 2)
    r2 = r ** 2
    return np.column_stack([x, y, r, r, r2, r2]) * SCALE


def feature_states(XY, N, L, H, signs):
    """Batched feature states |phi(x)> = U(x)|0> for every row of XY -> (P, 2^N).

    U(x) = (U_Z(x) H^{otimes N})^L.  U_Z is diagonal, so its action is an
    elementwise phase multiply; H^{otimes N} is the (small) Walsh transform.
    The IQP phase on basis state b is
        theta_b = sum_k f_k s^k_b + sum_{k<l} f_k f_l s^k_b s^l_b ,
    with s^k_b = (-1)^{bit_k(b)}.
    """
    F = feature_vectors(XY)                       # (P, N)
    P, D = F.shape[0], 1 << N
    single = F @ signs.T                          # (P, D): sum_k f_k s^k_b
    pair = np.zeros((P, D))
    for k in range(N):
        for l in range(k + 1, N):
            pair += (F[:, k] * F[:, l])[:, None] * (signs[:, k] * signs[:, l])[None, :]
    phase = np.exp(-1j * (single + pair))         # diagonal of U_Z(x), per point
    psi = np.zeros((P, D), dtype=complex)
    psi[:, 0] = 1.0                               # |0...0>
    for _ in range(L):
        psi = psi @ H.T                           # apply H^{otimes N}
        psi *= phase                              # apply U_Z(x)
    return psi


def fidelity_kernel(A, B):
    """Overlap kernel |<phi_i|phi_j>|^2 for two batches of feature states."""
    return np.abs(A.conj() @ B.T) ** 2


def make_rings(rng, n_per):
    """Three concentric noisy annuli; classes stacked in order (block-sorted)."""
    XY, y = [], []
    for c, R in enumerate(RADII):
        theta = rng.uniform(0.0, 2.0 * np.pi, n_per)
        rad = R + rng.normal(0.0, NOISE, n_per)
        XY.append(np.column_stack([rad * np.cos(theta), rad * np.sin(theta)]))
        y.append(np.full(n_per, c))
    return np.vstack(XY), np.concatenate(y)


def compute():
    """Dataset, quantum kernels, SVM decision map and kernel-PCA embedding.

    All numerics live here so the plotting code below only reads arrays, and the
    (comparatively) expensive kernel and grid evaluations are cached.
    """
    rng = np.random.default_rng(SEED)
    Xtr, ytr = make_rings(rng, N_TRAIN_PER)
    Xte, yte = make_rings(rng, N_TEST_PER)

    H = hadamard_wall(N_QUBITS)
    signs = basis_signs(N_QUBITS)
    psi_tr = feature_states(Xtr, N_QUBITS, N_LAYERS, H, signs)
    psi_te = feature_states(Xte, N_QUBITS, N_LAYERS, H, signs)

    K_tr = fidelity_kernel(psi_tr, psi_tr)        # M x M Gram (block-sorted by class)
    K_te = fidelity_kernel(psi_te, psi_tr)        # test-vs-train kernel

    clf = SVC(kernel="precomputed", C=SVM_C).fit(K_tr, ytr)
    acc_train = float(clf.score(K_tr, ytr))
    acc_test = float(clf.score(K_te, yte))

    # (c) decision regions: kernel of a dense grid against the training states.
    # Kept to the data region: the IQP phases are periodic, so far outside the
    # rings the boundary breaks into the map's characteristic tiling.
    lim = RADII[-1] + 0.5
    gx = np.linspace(-lim, lim, GRID_N)
    gy = np.linspace(-lim, lim, GRID_N)
    GX, GY = np.meshgrid(gx, gy)
    grid_pts = np.column_stack([GX.ravel(), GY.ravel()])
    psi_grid = feature_states(grid_pts, N_QUBITS, N_LAYERS, H, signs)
    K_grid = fidelity_kernel(psi_grid, psi_tr)
    grid_Z = clf.predict(K_grid).reshape(GX.shape).astype(float)

    # (d) 2-D kernel-PCA embedding of the training set from the precomputed kernel.
    proj = KernelPCA(n_components=2, kernel="precomputed").fit_transform(K_tr)

    print(f"  N_qubits={N_QUBITS}, layers={N_LAYERS}, scale={SCALE}, "
          f"M_train={len(ytr)}, M_test={len(yte)}")
    print(f"  SVM train accuracy = {acc_train:.3f}")
    print(f"  SVM test  accuracy = {acc_test:.3f}")
    return {"Xtr": Xtr, "ytr": ytr, "Xte": Xte, "yte": yte,
            "K_tr": K_tr, "acc_train": acc_train, "acc_test": acc_test,
            "gx": gx, "gy": gy, "grid_Z": grid_Z, "proj": proj}


data = load_or_compute(DATA_DIR / "fig_ch7_quantum_kernel.npz", compute)
Xtr, ytr = data["Xtr"], data["ytr"].astype(int)
K_tr = data["K_tr"]
gx, gy, grid_Z = data["gx"], data["gy"], data["grid_Z"]
proj = KernelPCA(n_components=3, kernel="precomputed").fit_transform(K_tr)   # 3-D embedding
acc_train, acc_test = float(data["acc_train"]), float(data["acc_test"])

# ---------------------------------------------------------------------------
# Figure: four panels, ~5:4 each.
# ---------------------------------------------------------------------------
apply_book_style()
region_cmap = ListedColormap([f"{c}" for c in CLASS_COLORS])

fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.6), constrained_layout=True)
ax_a, ax_b, ax_c, ax_d = axes.ravel()
ax_d.remove()                                        # panel (d) is a 3-D KPCA scatter
ax_d = fig.add_subplot(2, 2, 4, projection="3d")

# (a) raw rings in the plane
for c in range(3):
    m = ytr == c
    ax_a.scatter(Xtr[m, 0], Xtr[m, 1], s=30, color=CLASS_COLORS[c],
                 edgecolors="white", linewidths=0.4, label=CLASS_NAMES[c])
ax_a.set_xlabel(r"$x$")
ax_a.set_ylabel(r"$y$")
ax_a.set_aspect("equal", "box")
ax_a.legend(loc="upper right", fontsize=8, framealpha=0.9, handletextpad=0.2)
panel_label(ax_a, "a", loc="upper left")

# (b) sorted quantum kernel matrix -- block-diagonal by class
im = ax_b.imshow(K_tr, cmap="magma", origin="upper", interpolation="nearest")
n = N_TRAIN_PER
for k in (n, 2 * n):
    ax_b.axhline(k - 0.5, color="white", lw=0.7)
    ax_b.axvline(k - 0.5, color="white", lw=0.7)
ticks = [n // 2, n + n // 2, 2 * n + n // 2]
ax_b.set_xticks(ticks); ax_b.set_xticklabels(CLASS_NAMES)
ax_b.set_yticks(ticks); ax_b.set_yticklabels(CLASS_NAMES)
ax_b.set_xlabel(r"sample $j$ (sorted by class)")
ax_b.set_ylabel(r"sample $i$ (sorted by class)")
cb = fig.colorbar(im, ax=ax_b, pad=0.02, fraction=0.046)
cb.set_label(r"$k(\mathbf{x}_i,\mathbf{x}_j)=|\langle\phi_i|\phi_j\rangle|^2$")
panel_label(ax_b, "b", loc="upper left", color="white")

# (c) SVM decision regions in (x, y)
ax_c.contourf(gx, gy, grid_Z, levels=[-0.5, 0.5, 1.5, 2.5],
              colors=CLASS_COLORS, alpha=0.22)
for c in range(3):
    m = ytr == c
    ax_c.scatter(Xtr[m, 0], Xtr[m, 1], s=26, color=CLASS_COLORS[c],
                 edgecolors="white", linewidths=0.4)
ax_c.set_xlabel(r"$x$")
ax_c.set_ylabel(r"$y$")
ax_c.set_aspect("equal", "box")
ax_c.set_xlim(gx[0], gx[-1]); ax_c.set_ylim(gy[0], gy[-1])
panel_label(ax_c, "c", loc="upper left")

# (d) 3-D kernel-PCA embedding: all three rings become separated clusters
for c in range(3):
    m = ytr == c
    ax_d.scatter(proj[m, 0], proj[m, 1], proj[m, 2], s=34, color=CLASS_COLORS[c],
                 edgecolors="white", linewidths=0.4, label=CLASS_NAMES[c], depthshade=False)
ax_d.view_init(elev=22, azim=-58)                    # angle that best splits the clusters
ax_d.set_xlabel("KPCA 1", labelpad=1)
ax_d.set_ylabel("KPCA 2", labelpad=1)
ax_d.set_zlabel("KPCA 3", labelpad=1)
ax_d.tick_params(pad=-1)
ax_d.legend(loc="upper right", fontsize=8, framealpha=0.9, handletextpad=0.2)
ax_d.text2D(0.02, 0.97, "(d)", transform=ax_d.transAxes, fontweight="bold",
            fontsize=plt.rcParams["axes.titlesize"], va="top")

plt.savefig(OUTPUT_DIR / "fig_ch7_quantum_kernel.pdf")
plt.savefig(OUTPUT_DIR / "fig_ch7_quantum_kernel.png")
print(f"  SVM train/test accuracy: {acc_train:.3f} / {acc_test:.3f}")
print(f"  Saved: {OUTPUT_DIR / 'fig_ch7_quantum_kernel.pdf'}")
