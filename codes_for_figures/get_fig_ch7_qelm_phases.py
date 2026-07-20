#!/usr/bin/env python3
"""
get_fig_ch7_qelm_phases.py
==========================
Classifying quantum PHASES with a QUANTUM EXTREME LEARNING MACHINE (QELM), for two
models side by side: a symmetry-BREAKING transition (transverse-field Ising) and a
symmetry-protected TOPOLOGICAL transition (cluster-Ising / SPT).  This is the combined
book figure; the two standalone worked examples live in get_fig_ch7_qelm_phase.py and
get_fig_ch7_qelm_spt.py and in the matching notebooks.

Produces Fig. (ch7) -- figures/ch7/fig_ch7_qelm_phases.pdf (+png), a 2 x 3 grid.

    Row 1, TFIM  H(g) = -J sum Z_iZ_{i+1} - g sum X_i, transition at g_c = 1.
      (a) 2D PCA of the SCRAMBLED feature vectors coloured by g; ordered and
          disordered training clusters, with g_c marked on the manifold.
      (b) trained readout P(paramagnet | g) vs g; crossing near g_c = 1, held-out
          accuracy in-panel.
      (c) CONTRAST: local observables across g.  The nearest-neighbour correlation
          <Z_iZ_{i+1}> is large in the ordered phase and collapses in the disordered
          phase, and the transverse magnetization <X_i> rises through the transition.
          The Ising transition HAS a local order parameter, so local detection works.

    Row 2, cluster-Ising / SPT  H(h) = -J sum Z_{i-1}X_iZ_{i+1} - h sum X_i, h_c = 1.
      (d) 2D PCA of the SCRAMBLED feature vectors coloured by h; SPT and trivial
          training clusters, with h_c marked on the manifold.
      (e) trained readout P(trivial | h) vs h; crossing near h_c = 1, held-out
          accuracy in-panel.
      (f) CONTRAST: the nonlocal string order S(h) drops across h_c while the local
          correlation <Z_iZ_{i+1}> stays identically zero in both phases.  The inset
          bars report the ablation: a linear readout on the BARE features is at chance,
          on the SCRAMBLED features it is high.  The SPT phase has NO local order
          parameter; only the nonlocal string distinguishes the phases, and only the
          symmetry-breaking scramble makes it visible to local measurement.

BOTH ROWS share one fixed, untrained, nonintegrable MIXED-FIELD Ising reservoir
H_res = J_r sum Z_iZ_{i+1} + h_x sum X_i + h_z sum Z_i, U = exp(-i H_res t).  Its
longitudinal h_z breaks Jordan-Wigner integrability (so it scrambles) and, for the SPT
model, breaks the protecting Z2 x Z2 symmetry (so the forbidden observables become
measurable).  Only a linear (logistic) readout is trained, on states far from criticality.

Cached in data/ch7/fig_ch7_qelm_phases.npz.
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


def kron_string(ops_by_site, N):
    op = np.array([[1.0]], dtype=complex)
    for j in range(N):
        op = np.kron(op, ops_by_site.get(j, I2))
    return op


# ---------------------------------------------------------------------------
# Models.
def tfim_hamiltonian(N, J, g):
    """Transverse-field Ising model on an OPEN chain:
    H(g) = -J sum_i Z_i Z_{i+1} - g sum_i X_i.  Critical point at g/J = 1
    (ferromagnetic/ordered for g<1, paramagnetic/disordered for g>1)."""
    H = np.zeros((2 ** N, 2 ** N), dtype=complex)
    for i in range(N - 1):
        H += -J * (kron_op(Z, i, N) @ kron_op(Z, i + 1, N))
    for i in range(N):
        H += -g * kron_op(X, i, N)
    return H


def tfim_ground_state(N, J, g):
    """Lowest eigenvector of the TFIM (dense eigh).  Nondegenerate for g>0, so this
    is unambiguous and Z2-symmetric in finite size."""
    w, v = eigh(tfim_hamiltonian(N, J, g))
    return v[:, 0].astype(complex)


def cluster_ising(N, J, h):
    """Cluster-Ising chain on an OPEN chain:
    H(h) = -J sum_i Z_{i-1} X_i Z_{i+1} - h sum_i X_i.  SPT (cluster) phase for h<1,
    trivial paramagnet for h>1, transition at h/J = 1."""
    H = np.zeros((2 ** N, 2 ** N), dtype=complex)
    for i in range(1, N - 1):
        H += -J * (kron_op(Z, i - 1, N) @ kron_op(X, i, N) @ kron_op(Z, i + 1, N))
    for i in range(N):
        H += -h * kron_op(X, i, N)
    return H


def sublattice_parity(N):
    """Generators of the protecting Z2 x Z2 symmetry of the cluster-Ising chain:
    P_even = prod_{i even} X_i, P_odd = prod_{i odd} X_i."""
    Pe = np.eye(2 ** N, dtype=complex)
    Po = np.eye(2 ** N, dtype=complex)
    for s in range(N):
        if s % 2 == 0:
            Pe = Pe @ kron_op(X, s, N)
        else:
            Po = Po @ kron_op(X, s, N)
    return Pe, Po


def symmetry_projector(N):
    """Projector onto the (P_even, P_odd) = (+1, +1) symmetry sector (the sector that
    also contains the trivial all-|+> state), used to fix a unique symmetric ground
    state for the SPT chain despite its edge modes."""
    Pe, Po = sublattice_parity(N)
    return ((np.eye(2 ** N) + Pe) / 2) @ ((np.eye(2 ** N) + Po) / 2)


def cluster_ground_state(N, J, h, Proj, lam=50.0):
    """Symmetric ground state of the cluster-Ising chain: diagonalize H + lam(I - Proj),
    which lifts every other symmetry sector and leaves the (+1,+1) sector untouched."""
    w, v = eigh(cluster_ising(N, J, h) + lam * (np.eye(2 ** N) - Proj))
    return v[:, 0].astype(complex)


def string_operator(N):
    """Cluster string order S = product of the bulk stabilizers K_i = Z_{i-1}X_iZ_{i+1},
    which telescopes to Z_0 Y_1 X_2 ... X_{N-3} Y_{N-2} Z_{N-1} (Z on the ends, Y just
    inside, X across the bulk).  <S> ~ 1 on the cluster state, ~0 in the trivial phase."""
    ops = {0: Z, N - 1: Z, 1: Y, N - 2: Y}
    for j in range(2, N - 2):
        ops[j] = X
    return kron_string(ops, N)


def mixed_field_ising(N, Jr, hx, hz):
    """Nonintegrable mixed-field Ising reservoir H_res = Jr sum Z_iZ_{i+1} + hx sum X_i
    + hz sum Z_i.  The longitudinal hz breaks integrability (so it scrambles) and the
    cluster Z2 x Z2 symmetry (so the SPT forbidden observables become measurable)."""
    H = np.zeros((2 ** N, 2 ** N), dtype=complex)
    for i in range(N - 1):
        H += Jr * (kron_op(Z, i, N) @ kron_op(Z, i + 1, N))
    for i in range(N):
        H += hx * kron_op(X, i, N) + hz * kron_op(Z, i, N)
    return H


# ---------------------------------------------------------------------------
# Readout banks.  Any Pauli P = i^p X^x Z^z has <P> = Re sum_b conj(psi[b]) W[b]
# psi[b^x], an O(2^N) contraction verified against brute-force <psi|P|psi>.
_POP = np.array([bin(i).count("1") for i in range(1 << 16)])
_PH = np.array([1, 1j, -1, -1j])


def bank_full(N):
    """Generic low-weight bank {X_i, Y_i, Z_i, X_iX_{i+1}, Y_iY_{i+1}, Z_iZ_{i+1}}
    for the TFIM, where local observables already resolve the phases."""
    bit = lambda i: 1 << (N - 1 - i)
    specs = []
    for i in range(N):
        specs += [(bit(i), 0, 0), (bit(i), bit(i), 1), (0, bit(i), 0)]        # X, Y, Z
    for i in range(N - 1):
        m = bit(i) | bit(i + 1)
        specs += [(m, 0, 0), (m, m, 2), (0, m, 0)]                            # XX, YY, ZZ
    return _finish(N, specs)


def bank_forbidden(N):
    """Symmetry-forbidden bank {Y_i, Z_i, Y_iY_{i+1}, Z_iZ_{i+1}} for the SPT model:
    the conventional local order parameters, all forbidden by the protecting Z2 x Z2
    symmetry, hence identically zero in both phases on the bare state."""
    bit = lambda i: 1 << (N - 1 - i)
    specs = []
    for i in range(N):
        specs += [(bit(i), bit(i), 1), (0, bit(i), 0)]                        # Y, Z
    for i in range(N - 1):
        m = bit(i) | bit(i + 1)
        specs += [(m, m, 2), (0, m, 0)]                                       # YY, ZZ
    return _finish(N, specs)


def _finish(N, specs):
    b = np.arange(1 << N)
    COLS = np.array([b ^ x for (x, z, p) in specs])
    W = np.array([_PH[p] * (1 - 2 * (_POP[z & b] & 1)) for (x, z, p) in specs],
                 dtype=complex)
    return COLS, W


def qelm_features(psi_list, U, COLS, W):
    """Feature matrix.  For each state read the bank on the bare state (U is None) or on
    the scrambled state phi = U psi, f_alpha = <psi| U^dag O_alpha U |psi>."""
    F = np.zeros((len(psi_list), COLS.shape[0]))
    for n, psi in enumerate(psi_list):
        phi = U @ psi if U is not None else psi
        F[n] = np.real(np.conj(phi) * W * phi[COLS]).sum(axis=1)
    return F


def add_shot_noise(F, shots, rng):
    """Finite-shot estimate of each Pauli expectation, variance (1 - <O>^2)/shots."""
    var = np.clip(1.0 - F ** 2, 0.0, 1.0) / shots
    return F + rng.normal(size=F.shape) * np.sqrt(var)


def interp_crossing(x, prob):
    """Field where the readout probability crosses 1/2, by linear interpolation."""
    cr = np.where(np.diff(np.sign(prob - 0.5)) != 0)[0]
    if not len(cr):
        return float("nan")
    k = cr[0]
    p0, p1 = prob[k], prob[k + 1]
    return float(x[k] + (0.5 - p0) / (p1 - p0) * (x[k + 1] - x[k]))


# ---------------------------------------------------------------------------
# Model / protocol parameters (shared reservoir, shared readout protocol).
N = 8
J = 1.0
JR, HX, HZ = 1.0, 1.05, 0.5   # nonintegrable, symmetry-breaking reservoir
T_EV = 2.5                    # fixed scrambling time
N_SHOTS = 1024                # finite shots for the SPT bare-vs-scrambled ablation
N_REAL = 60                   # shot-noise realizations averaged

# TFIM training/test/sweep (as in the standalone TFIM example).
G_TRAIN_ORD = np.linspace(0.10, 0.65, 10)
G_TRAIN_DIS = np.linspace(1.35, 2.40, 10)
G_TEST_ORD = np.linspace(0.14, 0.61, 8)
G_TEST_DIS = np.linspace(1.40, 2.35, 8)
G_SWEEP = np.linspace(0.02, 2.5, 120)

# Cluster-Ising / SPT training/test/sweep (as in the standalone SPT example).
H_TRAIN_SPT = np.linspace(0.05, 0.55, 10)
H_TRAIN_TRIV = np.linspace(1.45, 2.50, 10)
H_TEST_SPT = np.linspace(0.10, 0.50, 8)
H_TEST_TRIV = np.linspace(1.50, 2.45, 8)
H_SWEEP = np.linspace(0.02, 2.5, 120)


def compute():
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    U = expm(-1j * mixed_field_ising(N, JR, HX, HZ) * T_EV)
    mid = N // 2
    ZZ_loc = kron_op(Z, mid, N) @ kron_op(Z, mid + 1, N)
    X_loc = kron_op(X, mid, N)

    out = {"N": N, "t_ev": T_EV, "n_shots": N_SHOTS}

    # ---------------- Row 1: TFIM ----------------
    COLS, W = bank_full(N)
    g_train = np.concatenate([G_TRAIN_ORD, G_TRAIN_DIS])
    y_gtrain = np.concatenate([np.zeros(len(G_TRAIN_ORD)), np.ones(len(G_TRAIN_DIS))])
    g_test = np.concatenate([G_TEST_ORD, G_TEST_DIS])
    y_gtest = np.concatenate([np.zeros(len(G_TEST_ORD)), np.ones(len(G_TEST_DIS))])

    psi_gtr = [tfim_ground_state(N, J, g) for g in g_train]
    psi_gte = [tfim_ground_state(N, J, g) for g in g_test]
    psi_gsw = [tfim_ground_state(N, J, g) for g in G_SWEEP]

    Fg_tr = qelm_features(psi_gtr, U, COLS, W)
    Fg_te = qelm_features(psi_gte, U, COLS, W)
    Fg_sw = qelm_features(psi_gsw, U, COLS, W)

    scaler_g = StandardScaler().fit(Fg_tr)
    clf_g = LogisticRegression(C=1.0, max_iter=5000).fit(scaler_g.transform(Fg_tr), y_gtrain)
    acc_g = float(clf_g.score(scaler_g.transform(Fg_te), y_gtest))
    prob_g = clf_g.predict_proba(scaler_g.transform(Fg_sw))[:, 1]     # P(disordered/paramagnet)
    g_cross = interp_crossing(G_SWEEP, prob_g)

    pca_g = PCA(n_components=2).fit(scaler_g.transform(Fg_tr))
    Pg_tr = pca_g.transform(scaler_g.transform(Fg_tr))
    Pg_sw = pca_g.transform(scaler_g.transform(Fg_sw))

    # TFIM contrast observables across the sweep (LOCAL, and they DO distinguish).
    ZZ_g = np.array([np.vdot(p, ZZ_loc @ p).real for p in psi_gsw])
    X_g = np.array([np.vdot(p, X_loc @ p).real for p in psi_gsw])

    out.update({
        "g_train": g_train, "y_gtrain": y_gtrain,
        "g_sweep": G_SWEEP, "prob_g": prob_g,
        "Pg_tr": Pg_tr, "Pg_sw": Pg_sw, "evr_g": pca_g.explained_variance_ratio_,
        "acc_g": acc_g, "g_cross": g_cross, "ZZ_g": ZZ_g, "X_g": X_g,
    })

    # ---------------- Row 2: cluster-Ising / SPT ----------------
    Proj = symmetry_projector(N)
    Sop = string_operator(N)
    COLSf, Wf = bank_forbidden(N)

    h_train = np.concatenate([H_TRAIN_SPT, H_TRAIN_TRIV])
    y_htrain = np.concatenate([np.zeros(len(H_TRAIN_SPT)), np.ones(len(H_TRAIN_TRIV))])
    h_test = np.concatenate([H_TEST_SPT, H_TEST_TRIV])
    y_htest = np.concatenate([np.zeros(len(H_TEST_SPT)), np.ones(len(H_TEST_TRIV))])

    psi_htr = [cluster_ground_state(N, J, h, Proj) for h in h_train]
    psi_hte = [cluster_ground_state(N, J, h, Proj) for h in h_test]
    psi_hsw = [cluster_ground_state(N, J, h, Proj) for h in H_SWEEP]

    Fh_tr = qelm_features(psi_htr, U, COLSf, Wf)      # scrambled
    Fh_te = qelm_features(psi_hte, U, COLSf, Wf)
    Fh_sw = qelm_features(psi_hsw, U, COLSf, Wf)
    Bh_tr = qelm_features(psi_htr, None, COLSf, Wf)   # bare (~0, forbidden)
    Bh_te = qelm_features(psi_hte, None, COLSf, Wf)

    scaler_h = StandardScaler().fit(Fh_tr)
    clf_h = LogisticRegression(C=1.0, max_iter=5000).fit(scaler_h.transform(Fh_tr), y_htrain)
    acc_h = float(clf_h.score(scaler_h.transform(Fh_te), y_htest))
    prob_h = clf_h.predict_proba(scaler_h.transform(Fh_sw))[:, 1]     # P(trivial)
    h_cross = interp_crossing(H_SWEEP, prob_h)

    pca_h = PCA(n_components=2).fit(scaler_h.transform(Fh_tr))
    Ph_tr = pca_h.transform(scaler_h.transform(Fh_tr))
    Ph_sw = pca_h.transform(scaler_h.transform(Fh_sw))

    # Ablation: bare vs scrambled readout accuracy under identical finite-shot noise.
    def noisy_acc(Ftr, Fte):
        rng = np.random.default_rng(0)
        accs = []
        for _ in range(N_REAL):
            Ftr_n = add_shot_noise(Ftr, N_SHOTS, rng)
            Fte_n = add_shot_noise(Fte, N_SHOTS, rng)
            sc = StandardScaler().fit(Ftr_n)
            c = LogisticRegression(C=1.0, max_iter=5000).fit(sc.transform(Ftr_n), y_htrain)
            accs.append(c.score(sc.transform(Fte_n), y_htest))
        return float(np.mean(accs)), float(np.std(accs))

    acc_bare, acc_bare_sd = noisy_acc(Bh_tr, Bh_te)
    acc_scram, acc_scram_sd = noisy_acc(Fh_tr, Fh_te)

    # SPT contrast: nonlocal string order and the (flat, zero) local correlation.
    S_h = np.array([np.vdot(p, Sop @ p).real for p in psi_hsw])
    ZZ_h = np.array([np.vdot(p, ZZ_loc @ p).real for p in psi_hsw])
    bare_mag = float(np.abs(Bh_tr).max())
    S_spt = float(S_h[H_SWEEP < 0.3].mean())
    S_triv = float(S_h[H_SWEEP > 1.1].mean())

    out.update({
        "h_train": h_train, "y_htrain": y_htrain,
        "h_sweep": H_SWEEP, "prob_h": prob_h,
        "Ph_tr": Ph_tr, "Ph_sw": Ph_sw, "evr_h": pca_h.explained_variance_ratio_,
        "acc_h": acc_h, "h_cross": h_cross, "S_h": S_h, "ZZ_h": ZZ_h,
        "acc_bare": acc_bare, "acc_bare_sd": acc_bare_sd,
        "acc_scram": acc_scram, "acc_scram_sd": acc_scram_sd,
        "S_spt": S_spt, "S_triv": S_triv, "bare_mag": bare_mag,
    })

    print(f"  N={N}, reservoir (Jr,hx,hz)=({JR},{HX},{HZ}), t={T_EV}")
    print(f"  [TFIM]  held-out acc={acc_g:.3f}, crossing g={g_cross:.4f} (g_c=1); "
          f"<ZZ> ordered->disordered = {ZZ_g[G_SWEEP<0.3].mean():.2f} -> {ZZ_g[G_SWEEP>2.0].mean():.2f}")
    print(f"  [SPT ]  held-out acc={acc_h:.3f}, crossing h={h_cross:.4f} (h_c=1)")
    print(f"  [SPT ]  string order S(SPT)={S_spt:+.3f}, S(trivial)={S_triv:+.3f}; "
          f"local <ZZ> range [{ZZ_h.min():+.1e},{ZZ_h.max():+.1e}] (flat ~0)")
    print(f"  [SPT ]  ablation: bare={acc_bare:.3f}+/-{acc_bare_sd:.3f} (chance), "
          f"scrambled={acc_scram:.3f}+/-{acc_scram_sd:.3f}; bare |f|max={bare_mag:.1e}")
    return out


data = load_or_compute(DATA_DIR / "fig_ch7_qelm_phases.npz", compute)

# TFIM
g_train, y_gtrain = data["g_train"], data["y_gtrain"]
g_sweep, prob_g = data["g_sweep"], data["prob_g"]
Pg_tr, Pg_sw, evr_g = data["Pg_tr"], data["Pg_sw"], data["evr_g"]
acc_g, g_cross = float(data["acc_g"]), float(data["g_cross"])
ZZ_g, X_g = data["ZZ_g"], data["X_g"]
# SPT
h_train, y_htrain = data["h_train"], data["y_htrain"]
h_sweep, prob_h = data["h_sweep"], data["prob_h"]
Ph_tr, Ph_sw, evr_h = data["Ph_tr"], data["Ph_sw"], data["evr_h"]
acc_h, h_cross = float(data["acc_h"]), float(data["h_cross"])
S_h, ZZ_h = data["S_h"], data["ZZ_h"]
acc_bare, acc_scram = float(data["acc_bare"]), float(data["acc_scram"])

# ---------------------------------------------------------------------------
C_ORD = "#1f6fc4"     # ordered (TFIM) / SPT
C_DIS = "#c1121f"     # disordered (TFIM) / trivial
C_SWEEP = "#444444"
C_GC = "#2a9d8f"

apply_book_style()
fig = plt.figure(figsize=(15.0, 8.5), constrained_layout=True)
gs = fig.add_gridspec(2, 3)
ax_a = fig.add_subplot(gs[0, 0]); ax_b = fig.add_subplot(gs[0, 1]); ax_c = fig.add_subplot(gs[0, 2])
ax_d = fig.add_subplot(gs[1, 0]); ax_e = fig.add_subplot(gs[1, 1]); ax_f = fig.add_subplot(gs[1, 2])

# ===================== Row 1: TFIM =====================
# (a) PCA of scrambled features
ord_m = y_gtrain < 0.5
sc = ax_a.scatter(Pg_sw[:, 0], Pg_sw[:, 1], c=g_sweep, cmap="viridis", s=28,
                  alpha=0.85, edgecolors="none", zorder=1)
i_cg = int(np.argmin(np.abs(g_sweep - 1.0)))
ax_a.plot(Pg_sw[i_cg, 0], Pg_sw[i_cg, 1], marker="*", color="black", ms=16,
          mec="white", mew=0.8, zorder=5, label=r"critical point $g_c=1$")
ax_a.scatter(Pg_tr[ord_m, 0], Pg_tr[ord_m, 1], s=65, color=C_ORD, marker="o",
             edgecolors="white", linewidths=0.7, zorder=3, label=r"train: ordered $g<1$")
ax_a.scatter(Pg_tr[~ord_m, 0], Pg_tr[~ord_m, 1], s=65, color=C_DIS, marker="s",
             edgecolors="white", linewidths=0.7, zorder=3, label=r"train: disordered $g>1$")
ax_a.set_xlabel(f"PC 1  ({100*evr_g[0]:.0f}% var.)")
ax_a.set_ylabel(f"PC 2  ({100*evr_g[1]:.0f}% var.)")
fig.colorbar(sc, ax=ax_a, pad=0.02).set_label(r"field $g$")
ax_a.legend(loc="lower left", fontsize=8, framealpha=0.92)
panel_label(ax_a, "a", loc="upper right")

# (b) TFIM readout
ax_b.axvspan(0.0, 1.0, color=C_ORD, alpha=0.06, zorder=0)
ax_b.axvspan(1.0, 2.5, color=C_DIS, alpha=0.06, zorder=0)
ax_b.axhline(0.5, color="0.6", ls=":", lw=1.0, zorder=1)
ax_b.axvline(1.0, color=C_GC, ls="--", lw=1.5, zorder=2, label=r"true $g_c=1$")
ax_b.plot(g_sweep, prob_g, color=C_SWEEP, lw=2.0, zorder=3)
ax_b.plot([g_cross], [0.5], marker="o", color=C_SWEEP, ms=7, mec="white", mew=0.8, zorder=4)
ax_b.scatter(g_train[ord_m], np.full(ord_m.sum(), 0.02), marker="|", s=70, color=C_ORD,
             zorder=4, label="training fields")
ax_b.scatter(g_train[~ord_m], np.full((~ord_m).sum(), 0.02), marker="|", s=70, color=C_DIS, zorder=4)
ax_b.set_xlabel(r"transverse field $g/J$")
ax_b.set_ylabel(r"predicted $P(\mathrm{paramagnet}\mid g)$")
ax_b.set_xlim(0.0, 2.5); ax_b.set_ylim(-0.03, 1.03)
ax_b.text(0.04, 0.90, rf"crossing at $g={g_cross:.2f}$" + "\n" + rf"test acc. $={acc_g*100:.0f}\%$",
          transform=ax_b.transAxes, fontsize=10, va="top",
          bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", alpha=0.92))
ax_b.text(0.14, 0.42, "ordered", color=C_ORD, fontsize=9, ha="center", transform=ax_b.transAxes)
ax_b.text(0.80, 0.58, "disordered", color=C_DIS, fontsize=9, ha="center", transform=ax_b.transAxes)
ax_b.legend(loc="center right", fontsize=8, framealpha=0.92)
panel_label(ax_b, "b", loc="lower right")

# (c) TFIM contrast: LOCAL observables that DO distinguish the phases
ax_c.axvspan(0.0, 1.0, color=C_ORD, alpha=0.06, zorder=0)
ax_c.axvspan(1.0, 2.5, color=C_DIS, alpha=0.06, zorder=0)
ax_c.axvline(1.0, color=C_GC, ls="--", lw=1.5, zorder=2, label=r"$g_c=1$")
ax_c.plot(g_sweep, ZZ_g, color=C_ORD, lw=2.3, zorder=4,
          label=r"local $\langle Z_iZ_{i+1}\rangle$")
ax_c.plot(g_sweep, X_g, color=C_DIS, lw=2.0, ls="-", marker="o", ms=3, markevery=8,
          zorder=3, label=r"local $\langle X_i\rangle$")
ax_c.set_xlabel(r"transverse field $g/J$")
ax_c.set_ylabel(r"local observable")
ax_c.set_xlim(0.0, 2.5); ax_c.set_ylim(-0.05, 1.05)
ax_c.legend(loc="center right", fontsize=8, framealpha=0.92)
ax_c.text(0.035, 0.30, "local order\nparameter works", color="0.25", fontsize=8.5,
          transform=ax_c.transAxes, va="top")
panel_label(ax_c, "c", loc="upper right")

# ===================== Row 2: cluster-Ising / SPT =====================
# (d) PCA of scrambled features
spt_m = y_htrain < 0.5
sc2 = ax_d.scatter(Ph_sw[:, 0], Ph_sw[:, 1], c=h_sweep, cmap="viridis", s=28,
                   alpha=0.85, edgecolors="none", zorder=1)
i_ch = int(np.argmin(np.abs(h_sweep - 1.0)))
ax_d.plot(Ph_sw[i_ch, 0], Ph_sw[i_ch, 1], marker="*", color="black", ms=16,
          mec="white", mew=0.8, zorder=5, label=r"critical point $h_c=1$")
ax_d.scatter(Ph_tr[spt_m, 0], Ph_tr[spt_m, 1], s=65, color=C_ORD, marker="o",
             edgecolors="white", linewidths=0.7, zorder=3, label=r"train: SPT $h<1$")
ax_d.scatter(Ph_tr[~spt_m, 0], Ph_tr[~spt_m, 1], s=65, color=C_DIS, marker="s",
             edgecolors="white", linewidths=0.7, zorder=3, label=r"train: trivial $h>1$")
ax_d.set_xlabel(f"PC 1  ({100*evr_h[0]:.0f}% var.)")
ax_d.set_ylabel(f"PC 2  ({100*evr_h[1]:.0f}% var.)")
fig.colorbar(sc2, ax=ax_d, pad=0.02).set_label(r"field $h$")
ax_d.legend(loc="lower left", fontsize=8, framealpha=0.92)
panel_label(ax_d, "d", loc="upper right")

# (e) SPT readout
ax_e.axvspan(0.0, 1.0, color=C_ORD, alpha=0.06, zorder=0)
ax_e.axvspan(1.0, 2.5, color=C_DIS, alpha=0.06, zorder=0)
ax_e.axhline(0.5, color="0.6", ls=":", lw=1.0, zorder=1)
ax_e.axvline(1.0, color=C_GC, ls="--", lw=1.5, zorder=2, label=r"true $h_c=1$")
ax_e.plot(h_sweep, prob_h, color=C_SWEEP, lw=2.0, zorder=3)
ax_e.plot([h_cross], [0.5], marker="o", color=C_SWEEP, ms=7, mec="white", mew=0.8, zorder=4)
ax_e.scatter(h_train[spt_m], np.full(spt_m.sum(), 0.02), marker="|", s=70, color=C_ORD,
             zorder=4, label="training fields")
ax_e.scatter(h_train[~spt_m], np.full((~spt_m).sum(), 0.02), marker="|", s=70, color=C_DIS, zorder=4)
ax_e.set_xlabel(r"transverse field $h/J$")
ax_e.set_ylabel(r"predicted $P(\mathrm{trivial}\mid h)$")
ax_e.set_xlim(0.0, 2.5); ax_e.set_ylim(-0.03, 1.03)
ax_e.text(0.04, 0.90, rf"crossing at $h={h_cross:.2f}$" + "\n" + rf"test acc. $={acc_h*100:.0f}\%$",
          transform=ax_e.transAxes, fontsize=10, va="top",
          bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", alpha=0.92))
ax_e.text(0.14, 0.42, "SPT", color=C_ORD, fontsize=9, ha="center", transform=ax_e.transAxes)
ax_e.text(0.80, 0.58, "trivial", color=C_DIS, fontsize=9, ha="center", transform=ax_e.transAxes)
ax_e.legend(loc="center right", fontsize=8, framealpha=0.92)
panel_label(ax_e, "e", loc="lower right")

# (f) SPT contrast: nonlocal string order vs flat local correlation, plus ablation bars
ax_f.axvspan(0.0, 1.0, color=C_ORD, alpha=0.06, zorder=0)
ax_f.axvspan(1.0, 2.5, color=C_DIS, alpha=0.06, zorder=0)
ax_f.axvline(1.0, color=C_GC, ls="--", lw=1.5, zorder=2, label=r"$h_c=1$")
ax_f.plot(h_sweep, S_h, color=C_ORD, lw=2.3, zorder=4, label=r"string order $S(h)$ (nonlocal)")
ax_f.plot(h_sweep, ZZ_h, color=C_DIS, lw=2.0, ls="-", marker="o", ms=3, markevery=8,
          zorder=3, label=r"local $\langle Z_iZ_{i+1}\rangle$")
ax_f.axhline(0.0, color="0.7", lw=0.8, zorder=1)
ax_f.set_xlabel(r"transverse field $h/J$")
ax_f.set_ylabel(r"order parameter")
ax_f.set_xlim(0.0, 2.5); ax_f.set_ylim(-0.12, 1.08)
ax_f.legend(loc="upper right", fontsize=8, framealpha=0.92)
ax_f.text(0.035, 0.55, "no local\norder parameter", color="0.25", fontsize=8.5,
          transform=ax_f.transAxes, va="top")
axin = ax_f.inset_axes([0.40, 0.13, 0.30, 0.42])
axin.bar([0], [acc_bare], width=0.7, color="0.6", edgecolor="black", linewidth=0.6)
axin.bar([1], [acc_scram], width=0.7, color=C_GC, edgecolor="black", linewidth=0.6)
axin.axhline(0.5, color="0.4", ls=":", lw=0.9)
axin.set_xticks([0, 1]); axin.set_xticklabels(["bare", "scram."], fontsize=7.5)
axin.set_ylim(0, 1.05); axin.set_yticks([0.0, 0.5, 1.0]); axin.tick_params(labelsize=7)
axin.set_ylabel("acc.", fontsize=8)
axin.text(0, acc_bare + 0.04, f"{acc_bare*100:.0f}%", ha="center", fontsize=7.5)
axin.text(1, acc_scram + 0.04, f"{acc_scram*100:.0f}%", ha="center", fontsize=7.5)
axin.text(0.5, 0.56, "chance", ha="center", fontsize=6.5, color="0.4", transform=axin.transAxes)
panel_label(ax_f, "f", loc="upper left")

plt.savefig(OUTPUT_DIR / "fig_ch7_qelm_phases.pdf")
plt.savefig(OUTPUT_DIR / "fig_ch7_qelm_phases.png")
print(f"  TFIM: acc={acc_g:.3f}, crossing g={g_cross:.3f} (g_c=1)")
print(f"  SPT : acc={acc_h:.3f}, crossing h={h_cross:.3f} (h_c=1); "
      f"ablation bare={acc_bare:.3f}/scrambled={acc_scram:.3f}")
print(f"  Saved: {OUTPUT_DIR / 'fig_ch7_qelm_phases.pdf'}")
