#!/usr/bin/env python3
"""
get_fig_ch5_syk_qubits.py
=========================
SYK on a quantum computer: entanglement and MAGIC of the Jordan-Wigner qubit-SYK
model, and the lesson that magic alone cannot certify classical hardness.

Produces Fig. (ch5) -- figures/ch5/fig_ch5_syk_qubits.pdf

    Panel (a): half-system entanglement entropy S vs number of qubits L.
    Panel (b): stabilizer Renyi entropy (magic) M_2 vs L.

for three families of states:
    * q=4 SYK   -- chaotic, non-Gaussian: off BOTH free islands (classically hard),
    * q=2 SYK   -- free fermions, Gaussian: matchgate-simulable, yet still magical,
    * Haar      -- reference for the maximal values.

THE POINT.  q=2 (free fermions) carries essentially as much MAGIC as the chaotic
q=4 model -- so magic by itself does NOT separate the easy model from the hard
one.  What makes q=2 classically easy is that it is Gaussian (matchgate-
simulable), an entirely different structure from the stabilizer formalism that
magic measures distance from.  A model is classically hard only when it is off
BOTH free islands (non-stabilizer AND non-Gaussian), which is q=4.

METHOD.  We never diagonalise.  Each Hamiltonian is built as a list of Pauli
strings (qubit_syk.py), a stabilizer initial state |0..0> (magic 0) is evolved to
late times with a Lanczos matrix-exponential, and we read the magic (magic.py,
fast Walsh-Hadamard) and half-cut entanglement of the scrambled state, averaged
over disorder realisations and a window of late-time snapshots.  Starting from
|0..0>, whose SYK energy is ~0 (infinite-temperature-like), keeps us in the
middle of the spectrum where the values are typical.  This runs to L=14 with no
2^L x 2^L matrix ever formed.

The numerics are cached (data/ch5/fig_ch5_syk_qubits.npz); re-rendering is instant
(set FIG_RECOMPUTE=1 to recompute).  On a cold cache the L=12,14 magic dominates
the runtime (M_2 costs O(L 4^L)).
"""
import sys
from pathlib import Path

import numpy as np
from scipy.linalg import eigh as scipy_eigh
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_style import apply_book_style, load_or_compute, panel_label
from qubit_syk import build_terms, compute_dense, lanczos_expm, half_entanglement
from magic import compute_sre

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR.parent / "figures" / "ch5"
DATA_DIR = SCRIPT_DIR / "data" / "ch5"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

L_VALUES = [4, 6, 8, 10, 12, 14]
# disorder realisations per size: uniform 20 across all L so the large-L points
# are as well-averaged as the small ones and the q=4 curve reads monotone.  The
# cost is dominated by L=14, where exact magic is O(L 4^L) and the dense
# mid-spectrum eigenstate is a 2^14 x 2^14 diagonalisation (~4 GB, minutes each).
N_REAL = {L: 20 for L in L_VALUES}
# All sizes use the true mid-spectrum eigenstate (dense diagonalisation).  A
# uniform eigenstate definition matters because the integrable q=2 model does not
# thermalise to its eigenstate value under time evolution (unlike chaotic q=4),
# so mixing eigenstate and time-evolved points across L would be inconsistent.
L_EIGENSTATE_MAX = 14
SNAP_TIMES = (6.0, 7.0, 8.0)


def haar_state(L, rng):
    v = rng.normal(size=1 << L) + 1j * rng.normal(size=1 << L)
    return v / np.linalg.norm(v)


def eigenstate_observables(L, q, seed):
    """(S, M2) of the mid-spectrum eigenstate (dense; L <= 12) or, for larger L,
    of |0..0> evolved to the late-time plateau with the Pauli-string simulator."""
    rng = np.random.default_rng(seed)
    if L <= L_EIGENSTATE_MAX:
        H = compute_dense(build_terms(L, q, rng), L)
        # Only the single mid-spectrum eigenpair (LAPACK zheevr subset selection):
        # exact, and cheaper than a full diagonalisation because we skip building
        # all D eigenvectors.  Note the all-to-all q=4 SYK Hamiltonian is *dense*
        # (~D^2 nonzeros), so sparse/shift-invert methods buy nothing here -- the
        # O(D^3) tridiagonal reduction is unavoidable, but subset selection still
        # trims the eigenvector back-transform.
        mid = (1 << L) // 2
        _, V = scipy_eigh(H, subset_by_index=[mid, mid], overwrite_a=True)
        psi = V[:, 0]                                  # middle eigenstate
        return half_entanglement(psi, L), float(compute_sre(psi, L))
    # large L: late-time-evolved |0..0>, averaged over the snapshot window
    terms = build_terms(L, q, rng)
    psi0 = np.zeros(1 << L, dtype=complex)
    psi0[0] = 1.0
    Ss, Ms = [], []
    for t in SNAP_TIMES:
        psit = lanczos_expm(psi0, terms, L, t, m=60)
        Ss.append(half_entanglement(psit, L))
        Ms.append(float(compute_sre(psit, L)))
    return float(np.mean(Ss)), float(np.mean(Ms))


def compute():
    out = {}
    for L in L_VALUES:
        nr = N_REAL[L]
        for q in (2, 4):
            S = np.zeros(nr)
            M = np.zeros(nr)
            for r in range(nr):
                S[r], M[r] = eigenstate_observables(L, q, seed=1000 * L + 10 * q + r)
            out[f"S_q{q}_L{L}"] = S
            out[f"M_q{q}_L{L}"] = M
            print(f"  L={L:2d} q={q}:  S={S.mean():.3f}+/-{S.std():.3f}  "
                  f"M2={M.mean():.3f}+/-{M.std():.3f}")
        # Haar reference
        Sh = np.zeros(nr)
        Mh = np.zeros(nr)
        for r in range(nr):
            psi = haar_state(L, np.random.default_rng(7000 * L + r))
            Sh[r] = half_entanglement(psi, L)
            Mh[r] = compute_sre(psi, L)
        out[f"S_haar_L{L}"] = Sh
        out[f"M_haar_L{L}"] = Mh
        print(f"  L={L:2d} haar:  S={Sh.mean():.3f}  M2={Mh.mean():.3f}")
    out["L_values"] = np.array(L_VALUES)
    return out


data = load_or_compute(DATA_DIR / "fig_ch5_syk_qubits.npz", compute)
Ls = np.array(L_VALUES)   # plot exactly the configured sizes, not any stale cached list


def series(prefix):
    mean = np.array([data[f"{prefix}_L{L}"].mean() for L in Ls])
    std = np.array([data[f"{prefix}_L{L}"].std() for L in Ls])
    return mean, std


apply_book_style()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6), constrained_layout=True)

styles = {
    "q4": dict(color="#c1121f", marker="o", label="q=4 SYK (chaotic)"),
    "q2": dict(color="#4361ee", marker="s", label="q=2 SYK (free fermions)"),
    "haar": dict(color="#555555", marker="^", ls="--", label="Haar"),
}

for key, kw in styles.items():
    m, s = series(f"S_{key}")
    ax1.errorbar(Ls, m, yerr=s, capsize=2, lw=1.4, **kw)
ax1.plot(Ls, (Ls // 2) * np.log(2) - 0.5, ":", color="black", lw=1.0, label="Page")
ax1.set_xlabel(r"qubits $L$")
ax1.set_ylabel(r"half-cut entanglement $S$")
panel_label(ax1, "a", loc="upper left")
ax1.legend(loc="lower right")

for key, kw in styles.items():
    m, s = series(f"M_{key}")
    # Connect the points with lines to guide the eye. The residual even-odd wobble
    # in the SYK curves is physical (the N_M = 2L mod 8 symmetry class; see caption).
    ax2.errorbar(Ls, m, yerr=s, capsize=2, lw=1.4, ms=7, **kw)
ax2.set_xlabel(r"qubits $L$")
ax2.set_ylabel(r"Stabilizer Rényi Entropy $M_2$")
panel_label(ax2, "b", loc="upper left")
ax2.legend(loc="lower right")

plt.savefig(OUTPUT_DIR / "fig_ch5_syk_qubits.pdf")
plt.savefig(OUTPUT_DIR / "fig_ch5_syk_qubits.png")
print(f"  Saved: {OUTPUT_DIR / 'fig_ch5_syk_qubits.pdf'}")
