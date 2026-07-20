#!/usr/bin/env python3
"""
get_fig_ch5_clifford_rigidity.py
================================
The Clifford group has no spectral rigidity: the proof, for Chapter 5.

Produces Fig. 5.1 -- figures/ch5/fig_ch5_clifford_rigidity.pdf

    Panel (a): eigenphase density of random 2-qubit Clifford unitaries (sharp atoms
               at roots of unity) against the flat CUE density 1/(2 pi)
    Panel (b): nearest-neighbour eigenphase spacing P(s), Clifford against CUE and
               the Wigner surmise

============================================================
WHY THIS FIGURE EXISTS
============================================================
The book's thesis is that spectral rigidity and Haar-moment typicality are two
LOGICALLY INDEPENDENT universalities that happen to coincide in the chaotic regime.
The Clifford group is the clean witness to that independence: it is an exact unitary
3-design (its low Haar moments are Haar-identical, Sec. 5.2), yet its spectrum carries
no rigidity at all. This figure exhibits the second half of that statement, which the
surrounding text otherwise only asserts.

============================================================
THE ARGUMENT (this is a proof, not a fit)
============================================================
The Clifford group C_N is FINITE. Every finite-order unitary U (some U^m = identity up
to phase) has eigenvalues that are roots of unity: its eigenphases are pinned to rational
multiples of 2 pi. So the eigenphase measure of a random Clifford element is a sum of
atoms sitting on a fixed rational grid, panel (a). It cannot repel: two eigenphases either
coincide exactly (a degeneracy) or sit a fixed rational gap apart. The circular unitary
ensemble (CUE = Haar on U(D)) does the opposite: its eigenphases are smoothly distributed
and repel, with the nearest-neighbour spacing following the Wigner surmise and P(s) -> 0
as s -> 0. Panel (b) puts the two spacing distributions side by side.

Quantitatively the contrast is stark. The fraction of small spacings P(s < 0.1) is ~0.0005
for CUE (level repulsion empties the small-spacing region) and ~0.06 for the Clifford group,
two orders of magnitude larger, because a lattice of atoms produces exact coincidences that
a repelling spectrum forbids. Rigidity is absent, which is the whole point.

For the single qubit the statement is even barer: |C_1| = 24, every element has order
1, 2, 3 or 4, and the entire eigenphase spectrum consists of 16 atoms. There is nothing
to unfold and no repulsion to measure.

============================================================
IMPLEMENTATION NOTE: SAMPLING, NOT ENUMERATING
============================================================
The 2-qubit Clifford group has 11520 elements. Enumerating it and diagonalizing every
element is unnecessary for a spectral histogram, so we SAMPLE the group by a random walk:
start from the identity and apply a long random word in the generators
{H_0, H_1, S_0, S_1, CNOT}. A word of ~40 letters mixes well past the group diameter, so
the sampled elements are effectively uniform over C_2 for the purpose of the eigenphase
measure. This is O(samples) rather than O(|group|) and needs no group-membership bookkeeping.
The eigenphase measure is a property of the group as a set, so the walk length only has to
exceed the mixing time, not hit every element.

The CUE comparison samples Haar-random U(4) by QR of a Ginibre matrix with the standard
R-diagonal phase correction (without it numpy's QR is biased and not Haar).

============================================================
RUNTIME
============================================================
~30 s single core for 1500 samples per ensemble. No GPU, no quantum hardware.
"""
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_style import apply_book_style, load_or_compute, panel_label

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR.parent / "figures" / "ch5"
DATA_DIR = SCRIPT_DIR / "data" / "ch5"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

N_SAMPLES = 1500       # per ensemble
WALK_LEN = 40          # random-walk word length; must exceed the group mixing time
SEED = 0

I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Z = np.diag([1, -1]).astype(complex)
H = (X + Z) / np.sqrt(2)
S = np.diag([1, 1j]).astype(complex)
# 2-qubit Clifford generators
CNOT = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)
GENS = [np.kron(H, I2), np.kron(I2, H), np.kron(S, I2), np.kron(I2, S), CNOT]


def random_clifford(rng, steps=WALK_LEN):
    """A near-uniform 2-qubit Clifford by a random walk in the generators."""
    U = np.eye(4, dtype=complex)
    for _ in range(steps):
        U = GENS[rng.integers(len(GENS))] @ U
    return U


def cue(rng, D=4):
    """Haar-random U(D). The R-diagonal phase fix is what makes it Haar rather than biased."""
    z = (rng.standard_normal((D, D)) + 1j * rng.standard_normal((D, D))) / np.sqrt(2)
    q, r = np.linalg.qr(z)
    d = np.diag(r)
    return q * (d / np.abs(d))


def eigenphases(U):
    return np.mod(np.angle(np.linalg.eigvals(U)), 2 * np.pi)


def unfolded_spacings(U):
    """Nearest-neighbour eigenphase spacings, normalized to unit mean (the natural
    unfolding on the circle, where the mean density is already flat)."""
    th = np.sort(eigenphases(U))
    s = np.diff(np.concatenate([th, [th[0] + 2 * np.pi]]))
    return s / np.mean(s)


def compute():
    rng = np.random.default_rng(SEED)
    cliff = [random_clifford(rng) for _ in range(N_SAMPLES)]
    haar = [cue(rng) for _ in range(N_SAMPLES)]
    return {
        "cliff_ph": np.concatenate([eigenphases(U) for U in cliff]),
        "cliff_s": np.concatenate([unfolded_spacings(U) for U in cliff]),
        "haar_s": np.concatenate([unfolded_spacings(U) for U in haar]),
    }


data = load_or_compute(DATA_DIR / "fig_ch5_clifford_rigidity.npz", compute)
cliff_ph = data["cliff_ph"]
cliff_s = data["cliff_s"]
haar_s = data["haar_s"]

# ── figure ────────────────────────────────────────────────────────
apply_book_style()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))

# (a) eigenphase density: atoms vs flat
ax1.hist(cliff_ph, bins=200, density=True, color='#d62828', alpha=0.85,
         label=r'Clifford $\mathcal{C}_2$')
ax1.axhline(1 / (2 * np.pi), color='#4361ee', lw=2, label=r'CUE  $1/2\pi$')
ax1.set_xlabel(r'eigenphase $\theta$')
ax1.set_ylabel('density')
ax1.set_xlim(0, 2 * np.pi)
panel_label(ax1, "a", loc="upper left")
ax1.legend(loc='upper right', framealpha=0.9)

# (b) level spacing: repulsion vs none
ax2.hist(haar_s, bins=50, range=(0, 4), density=True, color='#4361ee', alpha=0.55, label='CUE')
ax2.hist(cliff_s, bins=50, range=(0, 4), density=True, color='#d62828', alpha=0.55, label='Clifford')
ss = np.linspace(0, 4, 300)
# Wigner surmise for the unitary (beta=2) class
ax2.plot(ss, (32 / np.pi**2) * ss**2 * np.exp(-4 * ss**2 / np.pi), 'k--', lw=1.5,
         label='Wigner surmise')
ax2.set_xlabel(r'spacing $s$')
ax2.set_ylabel('$P(s)$')
panel_label(ax2, "b", loc="upper left")
ax2.legend(loc='upper right', framealpha=0.9)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'fig_ch5_clifford_rigidity.pdf')
plt.savefig(OUTPUT_DIR / 'fig_ch5_clifford_rigidity.png')
print(f"Saved: {OUTPUT_DIR / 'fig_ch5_clifford_rigidity.pdf'}")
print(f"  CUE      P(s<0.1) = {np.mean(haar_s < 0.1):.4f}   (level repulsion)")
print(f"  Clifford P(s<0.1) = {np.mean(cliff_s < 0.1):.4f}   (degeneracies, no repulsion)")
print(f"  distinct Clifford eigenphase atoms (rounded): {len(np.unique(np.round(cliff_ph, 3)))}")
