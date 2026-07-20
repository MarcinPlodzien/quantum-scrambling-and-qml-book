#!/usr/bin/env python3
"""
get_fig_ch5_mp_convergence.py
===================================
Entanglement spectrum of the deterministic Ry(pi/4)+CNOT brickwork circuit of
Chapter 5, and its convergence to the Marchenko-Pastur (MP) law.

Computes the data behind Fig. 5.2, panels (a)-(d), and renders a standalone
four-panel version of them.

    Data:   codes_for_figures/data/ch5/spectra_ry_N20.npz
            codes_for_figures/data/ch5/spectra_haar_N20.npz
    Figure: figures/ch5/fig_ch5_mp_convergence.pdf  (and .png)

Usage:
    python get_fig_ch5_mp_convergence.py                 # compute (or load cache) + plot
    FIG_RECOMPUTE=1 python get_fig_ch5_mp_convergence.py  # force the merged cache to rebuild

============================================================
WARNING: THIS SCRIPT ALONE DOES NOT REPRODUCE FIG. 5.2
============================================================
Fig. 5.2 as printed has FIVE panels: the four spectrum histograms (a)-(d)
computed here, plus a wide bottom panel (e) plotting D_KL(P_data || P_MP)
against circuit depth for Clifford and non-Clifford circuits.

The shipped five-panel PDF is assembled by get_fig_ch5_combined.py, which
loads the two .npz caches written here together with dkl_ry_N20.npz and
dkl_cliff_N20.npz written by get_fig_ch5_dkl_convergence.py.

plot_figure() below writes a FOUR-panel figure to exactly the output path the
book includes. Running this script therefore overwrites Fig. 5.2 with a
version that has no panel (e), which then contradicts the printed caption.
Safe refresh order:

    1. python get_fig_ch5_mp_convergence.py    (writes the spectra caches)
    2. python get_fig_ch5_dkl_convergence.py   (writes the D_KL caches)
    3. python get_fig_ch5_combined.py          (writes the real Fig. 5.2, last)

============================================================
PHYSICS BACKGROUND
============================================================
Split the N qubits into A (the first k) and B (the remaining N-k). The Schmidt
decomposition is |psi> = sum_i sqrt(p_i) |i_A> |i_B>, and the entanglement
spectrum is the set of "entanglement energies"

    xi_i = -ln p_i          (the eigenvalues of H_E = -ln rho_A)

The entropy compresses this whole spectrum into one number. The spectrum itself
carries far more, and Chapter 5 uses its SHAPE as the diagnostic. Two extremes
bracket the physics:

  * Stabilizer (Clifford) state: rho_A = (1/chi) * Projector, so every nonzero
    p_i is equal and the spectrum is a DELTA FUNCTION. This holds no matter how
    large the entropy is, which is the point: entropy cannot detect it.

  * Haar-random state: reshaping |psi> to an m x n matrix C makes rho_A = C C+
    a Wishart random matrix. Rescaling x_i = m p_i (so that <x> = 1), the
    eigenvalue density converges, as m, n -> infinity at fixed aspect ratio
    c = m/n, to the Marchenko-Pastur law

        rho_MP(x) = sqrt((x+ - x)(x - x-)) / (2 pi c x),   x in [x-, x+]

    with edges x± = (1 ± sqrt(c))^2.

In the entanglement variable xi = -ln p = k ln2 - ln x, substituting
x = 2^k exp(-xi) with Jacobian |dx/dxi| = x makes the 1/x cancel:

    rho_xi(xi) = rho_MP(x) * x = sqrt((x+ - x)(x - x-)) / (2 pi c)

This is the density plotted as the black curve, and it is exactly the book's
Eq. (eq:mp_xi). It is implemented in mp_density_xi() below.

THE ASPECT RATIO c. With d_A = 2^k and d_B = 2^(N-k), c = d_A/d_B = 2^(2k-N).
It sets how much room the spectrum has to spread:

  * c -> 0 (lopsided cut, d_A << d_B): the support width x+ - x- = 4 sqrt(c)
    shrinks to zero, the p_i pile up at 1/d_A, and rho_A is nearly maximally
    mixed with a nearly flat spectrum.
  * c = 1 (equal cut, k = N/2): x- = 0 and x+ = 4, the maximum spectral spread.
    Because x- = 0, the xi-support is not compact: it has a finite lower edge
    xi_min = k ln2 - ln 4 (= 5.545 for k = 10) and a long tail toward large xi.

This script uses N = 20 and k = 10, so m = n = 1024 and c = 1 exactly. The long
right-hand tail visible in every panel is that c = 1 feature, not an artifact.

WHAT THE READER SHOULD TAKE AWAY. Depth alone does not make a state Haar-like,
and entropy alone cannot tell the difference. As L grows, the non-Clifford
circuit's spectrum broadens and by L ~ N sits on top of BOTH the Haar-sampled
reference and the analytic MP curve. The Ry(pi/4) rotations are what make this
possible: pi/4 lies between the Clifford angles 0 and pi/2, so each layer
injects nonstabilizerness ("magic"). A pure Clifford circuit stays a stabilizer
state forever, and its spectrum stays a delta function however entangled it
gets. That contrast is the "Clifford barrier" plotted in panel (e).

CAVEAT, stated in the book text as well: this is an EMPIRICAL observation for
this one deterministic circuit, not a consequence of the design theorems.
Deterministic layouts with fine-tuned or Clifford-heavy angles can stay trapped
in non-ergodic sectors instead.

============================================================
ALGORITHM
============================================================
1. INITIAL STATE. simulate_ry_cnot() starts from the GHZ state
   (|0...0> + |1...1>)/sqrt(2).
   NOTE: the book text describes the circuit as acting on |0>^(x)N, and states
   that the L = 0 spectrum is a delta function with chi = 1. The code as
   written starts from GHZ, whose L = 0 spectrum is a delta function with
   chi = 2. Recorded here as a text/code discrepancy, not silently reconciled;
   it does not affect the L >~ N/2 panels the figure is about.

2. LAYERS. Each of the L layers applies, in order: Ry(pi/4) on every qubit,
   then CNOT on even pairs (0,1), (2,3), ..., then CNOT on odd pairs (1,2),
   (3,4), .... This matches the book's U_layer. Both sublayers run every layer,
   giving a brickwork light cone that reaches all qubits in O(N) layers. The
   plotted depths L = 5, 10, 15, 20 are N/4, N/2, 3N/4, N.

3. SPECTRUM (_get_spectrum). Reshape the length-2^N amplitude vector into the
   m x n coefficient matrix C, form rho_A = C C+, and diagonalise it with
   eigvalsh. Its eigenvalues ARE the p_i; the figure histograms xi = -ln p.

4. HAAR REFERENCE (simulate_haar). Draw z with iid complex Gaussian entries and
   normalise. A complex Gaussian is unitarily invariant and normalisation
   projects onto the unit sphere, so this is an exact Haar sample on CP^(D-1).
   N_HAAR = 20 samples under a fixed seed; their pooled eigenvalues form the
   gray histogram.

5. HOW CONVERGENCE IS QUANTIFIED. It is NOT quantified in this script. Each
   panel simply overlays gray (pooled Haar reference), red (the circuit at
   depth L), and black (the analytic MP curve); convergence is read off the
   overlap by eye. The quantitative D_KL(L) of panel (e) is computed by
   get_fig_ch5_dkl_convergence.py. The kl_from_mp() function below is an
   unused copy of that machinery (see DEAD CODE).

   NOT CIRCULAR: the gray reference is sampled from complex Gaussians and the
   black curve is evaluated from the closed-form MP density. The two are
   independent constructions, so their agreement is a genuine check of the MP
   law at finite m = n = 1024, not a formula being compared against itself.

============================================================
IMPLEMENTATION NOTES
============================================================
WHY F64, AND WHY THE ENV VAR COMES FIRST. JAX_ENABLE_X64 is set through
os.environ BEFORE `import jax`, because JAX reads it at import time and setting
it afterwards is silently ignored. F64 is not cosmetic here: with c = 1 the MP
tail extends to large xi, i.e. to very small p. Float32 carries eps ~ 1e-7,
which would drown precisely the small eigenvalues that make up the tail this
figure exists to show.

XLA_PYTHON_CLIENT_PREALLOCATE=false stops JAX from reserving most of the GPU on
import. This job is small and has no reason to hold that memory.

MATRIX-FREE EINSUM. |psi> is held as a rank-N tensor of shape (2,)*N, and each
gate is contracted only into the axes it touches: O(2^N) per gate instead of
the O(4^N) of building a 2^N x 2^N matrix. At N = 20 that is a 16 MB vector
versus an 18 TB matrix, so it is the difference between running and not.

PRECOMPUTED SUBSCRIPTS. The einsum strings depend only on N, so _subs_1q and
_subs_2q build them once per call rather than reformatting inside the layer
loop. _subs_2q enumerates all ordered pairs even though only the ~N
nearest-neighbour ones are used; that is a negligible one-off cost.

21-QUBIT CEILING. _AX supplies 21 lowercase input labels and outputs are named
chr(ord('A') + j). N > 21 exhausts the alphabet and the subscripts break. The
N_Q = 20 used here is inside the limit, but this is the constraint to check
first if the figure is ever rescaled.

CNOT INDEX CONVENTION. CNOT_T[c_out, t_out, c_in, t_in] is built by
_CNOT_np[ci, ci ^ ti, ci, ti] = 1.0: the control passes through unchanged and
the target is XORed with the control. It is stored as a (2,2,2,2) tensor rather
than a 4x4 matrix precisely so it can be contracted straight into two axes of
the state tensor without any reshaping.

WHY A BARE RESHAPE GIVES THE RIGHT BIPARTITION. reshape is row-major, so
splitting the length-2^N vector as (m, n) with m = 2^k puts the first k qubits
(the most significant bits of the basis index) into the row index and the rest
into the column index. Subsystem A is then exactly qubits 0..k-1 with no
transpose or axis permutation. This shortcut is valid only because the cut is
contiguous and starts at qubit 0; any other partition would need a transpose
first.

THE CLIP AND THE FILTER ARE ONE GUARD, NOT TWO. rho_A is positive semidefinite
in exact arithmetic, but eigvalsh still returns small NEGATIVE eigenvalues
(order of the roundoff) for its many near-null directions, and -ln of a
negative is nan. _get_spectrum clips at 1e-30 so every entry is positive and
finite; every caller then keeps only eigenvalues > 1e-25. The two constants are
chosen to work together: the clip floor sits BELOW the filter threshold, so the
clipped junk is guaranteed to be discarded rather than resurfacing as a
spurious spike at xi = -ln(1e-30) ~ 69.

DETERMINISTIC CIRCUIT, NO AVERAGING. simulate_ry_cnot() contains no randomness,
so each depth yields one state and exactly m = 1024 eigenvalues. The Haar
reference pools 20 x 1024 = 20480. That sampling asymmetry, not physics, is why
the gray histogram looks smooth while the red one at L = 5 looks ragged.

HAND-TUNED PLOT LIMITS. The histogram bins (2, 28, 55) and the ylim of 0.37 are
tuned for N = 20 at c = 1, where the MP peak sits near xi ~ 6. Changing N_Q
without revisiting them will crop the figure.

============================================================
DEAD CODE (deliberately retained)
============================================================
simulate_clifford() and kl_from_mp() are defined but never called anywhere in
this file, and the KL_DEPTHS constant is likewise unused. They are copies of
the same-named functions in get_fig_ch5_dkl_convergence.py, which is the script
that actually produces panel (e). The _H and _S gate constants exist only to
feed the dead simulate_clifford(). They are left in place because they document
the Clifford negative control that panel (e) plots; removing them would change
no output.

============================================================
RUNTIME / MEMORY
============================================================
N = 20 gives D = 2^20 = 1,048,576 amplitudes, 16 MB as complex128. Each einsum
materialises a fresh tensor that size, so peak memory is a few hundred MB, well
inside a laptop. There is no 2^N x 2^N object anywhere.

Per layer: N = 20 single-qubit contractions plus N-1 = 19 CNOT contractions,
each O(2^N). The full run covers 5 + 10 + 15 + 20 = 50 layers plus 20 Haar
draws. Each spectrum costs m^2 n ~ 1.1e9 complex multiply-adds to build rho_A
plus an O(m^3) eigvalsh on a 1024 x 1024 matrix, done 24 times. This is a short
run on CPU (order seconds per depth); the script prints per-depth and total
timings rather than relying on this estimate.

CACHING. generate_all_data() skips any .npz that already exists unless
--regenerate is passed. --plot-only skips generation entirely and exits with an
error if the caches are missing. Because plotting always runs, see the WARNING
above before invoking any of these.
"""

import argparse
import os
import time
import numpy as np
from pathlib import Path

# ── JAX configuration ────────────────────────────────────────────
# Both flags MUST be set before `import jax` below: JAX reads them once at
# import time, so assigning them later is silently ignored.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("JAX_ENABLE_X64", "1")   # F64: the MP tail lives in the
                                               # small eigenvalues, which F32
                                               # (eps ~ 1e-7) would wipe out

import jax
import jax.numpy as jnp
from jax import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_style import apply_book_style, load_or_compute, panel_label


# ══════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════════════

BOOK_DIR   = Path(__file__).resolve().parent.parent
DATA_DIR   = Path(__file__).resolve().parent / "data" / "ch5"
OUTPUT_DIR = BOOK_DIR / "figures" / "ch5"
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Figure layout ────────────────────────────────────────────────
#   4 panels: (a) L=5  (b) L=10  (c) L=15  (d) L=20
#   Each panel overlays Haar reference (gray) + circuit (red) + MP (black)
#   All for N = 20 qubits, equal bipartition k = 10
N_Q          = 20
HIST_DEPTHS  = [5, 10, 15, 20]
N_HAAR       = 20
KL_DEPTHS    = list(range(1, N_Q + 1))   # UNUSED here; the D_KL(L) scan of
                                         # panel (e) lives in
                                         # get_fig_ch5_dkl_convergence.py

CDT = jnp.complex128   # F64 complex dtype


# ══════════════════════════════════════════════════════════════════
#  GATE PRIMITIVES
# ══════════════════════════════════════════════════════════════════
#
#  All gates stored as small tensors, applied via einsum on the
#  rank-N statevector tensor (2,2,...,2).
#
# ══════════════════════════════════════════════════════════════════

# ── Ry(π/4): maximum magic injection ────────────────────────────
#
#  Ry(θ) = exp(−iθY/2) = [[cos θ/2, −sin θ/2],
#                          [sin θ/2,  cos θ/2]]
#
#  θ = π/4 is non-Clifford → breaks stabilizer structure.
#
THETA_MAGIC = np.pi / 4
_RY = jnp.array([
    [np.cos(THETA_MAGIC / 2), -np.sin(THETA_MAGIC / 2)],
    [np.sin(THETA_MAGIC / 2),  np.cos(THETA_MAGIC / 2)]
], dtype=CDT)

# ── Hadamard H = (1/√2)[[1,1],[1,−1]] ──────────────────────────
_H = jnp.array([[1, 1], [1, -1]], dtype=CDT) / jnp.sqrt(2.0)

# ── Phase S = diag(1, i) ────────────────────────────────────────
_S = jnp.array([[1, 0], [0, 1j]], dtype=CDT)

# ── CNOT as (2,2,2,2) tensor ────────────────────────────────────
#  CNOT[c_out, t_out, c_in, t_in] = δ(c_out, c_in) · δ(t_out, t_in ⊕ c_in)
_CNOT_np = np.zeros((2, 2, 2, 2))
for ci in range(2):
    for ti in range(2):
        _CNOT_np[ci, ci ^ ti, ci, ti] = 1.0
CNOT_T = jnp.array(_CNOT_np, dtype=CDT)


# ══════════════════════════════════════════════════════════════════
#  MATRIX-FREE GATE APPLICATION VIA EINSUM
# ══════════════════════════════════════════════════════════════════
#
#  THE KEY IDEA: Instead of storing |ψ⟩ as a flat vector of length
#  D = 2^N and applying gates as D×D matrices (cost O(4^N)), we:
#
#    1. Reshape |ψ⟩ into a rank-N tensor: shape (2, 2, ..., 2)
#       Each axis corresponds to one qubit.
#
#    2. Apply a gate as a LOCAL tensor contraction using einsum:
#       - Single-qubit gate G on qubit j: contract G (2×2) with axis j
#       - Two-qubit gate (CNOT): contract (2,2,2,2) with axes (c,t)
#
#    3. Cost: O(2^N) per gate, not O(4^N)!
#
#  EXAMPLE for N=4 qubits, applying gate G to qubit 1:
#
#    ψ[a, b, c, d]  ×  G[B, b]  →  ψ'[a, B, c, d]
#
#    In einsum notation: "Bb, abcd -> aBcd"
#
#    This contracts ONLY over the b index (qubit 1), leaving all
#    other qubits untouched. Cost: 2^4 × 2 = 32 multiplies,
#    compared to 2^4 × 2^4 = 256 for full matrix multiplication.
#
#  For CNOT on (control=1, target=2):
#
#    ψ[a, b, c, d]  ×  CNOT[B, C, b, c]  →  ψ'[a, B, C, d]
#
#    Einsum: "BCbc, abcd -> aBCd"
#    Cost: 2^4 × 4 = 64 multiplies
#
# ══════════════════════════════════════════════════════════════════

# 21 lowercase labels for input axes; outputs are named chr(ord('A') + j).
# This caps the simulator at N = 21: beyond that the alphabet runs out and the
# generated subscripts break. Check this first if the figure is ever rescaled.
_AX = "abcdefghijklmnopqrstu"    # axis labels for up to 21 qubits


def _subs_1q(N):
    """
    Pre-build einsum subscript strings for single-qubit gates.

    For N qubits labeled a,b,c,..., applying gate G on qubit j:

      G[NEW_j, old_j] × ψ[a,...,old_j,...] → ψ'[a,...,NEW_j,...]

    We use uppercase A,B,C,... for the output indices.

    Example (N=4, qubit 1):
      Input axes:  "abcd"    (qubit 1 = 'b')
      Output axes: "aBcd"    (qubit 1 contracted to 'B')
      Subscript:   "Bb,abcd->aBcd"

    Returns: list of N subscript strings, one per qubit.
    """
    subs = []
    for j in range(N):
        ax = list(_AX[:N])          # e.g. ['a','b','c','d'] for N=4
        out = list(ax)
        new = chr(ord('A') + j)     # uppercase version: 'A','B','C',...
        out[j] = new                # replace old axis with new
        subs.append(f"{new}{ax[j]},{''.join(ax)}->{''.join(out)}")
    return subs


def _subs_2q(N):
    """
    Pre-build einsum subscript strings for two-qubit CNOT gates.

    CNOT is stored as a (2,2,2,2) tensor:
      CNOT[c_out, t_out, c_in, t_in]

    For CNOT on (control=c, target=t):

      CNOT[C_new, T_new, c_old, t_old] × ψ[...,c_old,...,t_old,...]
        → ψ'[...,C_new,...,T_new,...]

    Example (N=4, control=0, target=1):
      Subscript: "ABab,abcd->ABcd"

    Returns: dict mapping (control, target) → subscript string
    """
    subs = {}
    for c in range(N):
        for t in range(N):
            if c == t:
                continue
            ax = list(_AX[:N])
            out = list(ax)
            C = chr(ord('A') + c)   # new control axis
            T = chr(ord('A') + t)   # new target axis
            out[c], out[t] = C, T
            subs[(c, t)] = f"{C}{T}{ax[c]}{ax[t]},{''.join(ax)}->{''.join(out)}"
    return subs


# ══════════════════════════════════════════════════════════════════
#  SIMULATION ENGINE
# ══════════════════════════════════════════════════════════════════
#
#  The full simulation pipeline:
#
#  1. Prepare GHZ state: ψ = (|0...0⟩ + |1...1⟩)/√2
#     → flat vector of length D = 2^N
#
#  2. For each layer ℓ = 1, ..., L:
#     a) Reshape ψ to (2,2,...,2) tensor
#     b) Apply Ry(π/4) on each qubit via einsum contraction
#     c) Apply CNOT on even pairs (0,1), (2,3), ... via einsum
#     d) Apply CNOT on odd pairs  (1,2), (3,4), ... via einsum
#     e) Flatten back to vector
#
#  3. Extract entanglement spectrum:
#     a) Reshape ψ as (m × n) coefficient matrix C, m = n = 2^{N/2}
#     b) Build ρ_A = C · C†  (m × m reduced density matrix)
#     c) Diagonalise: eigs = eigvalsh(ρ_A)
#
#  Total cost per layer: O(N · 2^N) — linear in N, exponential in 2^N.
#  This is MUCH cheaper than the naive O(4^N) matrix approach.
#
# ══════════════════════════════════════════════════════════════════

def _get_spectrum(psi, N):
    """
    Extract entanglement spectrum from statevector |ψ⟩.

    PROCEDURE:
      1. Equal bipartition: A = first k = N/2 qubits,
                            B = remaining N−k qubits.
         Dimensions: m = dim(H_A) = 2^k,  n = dim(H_B) = 2^{N−k}
         For equal partition: m = n = 2^{N/2}

      2. Reshape ψ (length D = 2^N) into coefficient matrix C (m × n):
           C[i,j] = ⟨iₐ, jᵦ | ψ⟩

      3. Compute reduced density matrix:
           ρ_A = C · C† = Σⱼ C[:,j] C[:,j]†
         This is the m×m matrix whose eigenvalues are the
         squared Schmidt coefficients: {p₁, p₂, ..., pₘ}

      4. Diagonalise via eigvalsh (Hermitian eigenvalue solver).

    Cost: O(m²n) for the contraction + O(m³) for diagonalisation.
    For N=18: m = n = 512, so ~134M multiply-adds + 512³ ≈ 134M eigensolver.
    """
    k = N // 2
    m = 1 << k                      # dim(H_A) = 2^k
    n = (1 << N) >> k               # dim(H_B) = 2^{N-k}
    # A bare reshape suffices because reshape is row-major: the first k qubits
    # are the most significant bits of the basis index, so they land in the row
    # index and B in the column index. Valid ONLY for a contiguous cut starting
    # at qubit 0; any other partition would need a transpose first.
    C = psi.reshape(m, n)           # coefficient matrix
    # ρ_A = C · C†  via einsum (ab = row indices, cb = contract over B)
    rho = jnp.einsum('ab,cb->ac', C, jnp.conj(C))
    # ρ_A is PSD in exact arithmetic, but eigvalsh still returns small NEGATIVE
    # eigenvalues (~roundoff) for its many near-null directions, and −ln of a
    # negative is nan. Clip to keep everything positive and finite. The floor
    # 1e-30 is deliberately BELOW the > 1e-25 filter every caller applies, so
    # the clipped junk is dropped rather than resurfacing as a fake spike at
    # ξ = −ln(1e-30) ≈ 69.
    return np.array(jnp.clip(jnp.linalg.eigvalsh(rho), 1e-30, None))


def simulate_ry_cnot(N, L):
    """
    Ry(π/4)+CNOT brickwork circuit → entanglement spectrum.

    CIRCUIT STRUCTURE (Ry+CNOT brickwork):

      |ψ₀⟩ = GHZ = (|0...0⟩ + |1...1⟩)/√2

      For each layer ℓ = 1, ..., L:
        1. Ry(π/4) on ALL N qubits     ← magic injection (non-Clifford!)
        2. CNOT on even pairs: (0,1), (2,3), (4,5), ...
        3. CNOT on odd  pairs: (1,2), (3,4), (5,6), ...

    NOTE: Both even AND odd CNOT pairs are applied EVERY layer.
    This creates a brickwork entanglement pattern that reaches
    all qubits within O(N) layers.

    The magic injection angle θ = π/4 gives the maximum
    non-Cliffordness per gate. At depth L ~ N, the entanglement
    spectrum converges to the Marchenko-Pastur distribution.

    IMPLEMENTATION:
      - State stored as rank-N tensor (2,2,...,2)
      - Gates applied via pre-computed einsum subscripts
      - Cost per layer: O(N · 2^N) for N single-qubit + (N-1) CNOT gates

    Returns: eigenvalues of ρ_A, shape (2^{N/2},)
    """
    D = 1 << N          # Hilbert space dimension = 2^N
    sub1 = _subs_1q(N)  # pre-build einsum strings for 1-qubit gates
    sub2 = _subs_2q(N)  # pre-build einsum strings for 2-qubit gates

    # Nearest-neighbour CNOT pairs (brickwork pattern)
    even = [(j, j+1) for j in range(0, N-1, 2)]  # (0,1),(2,3),...
    odd  = [(j, j+1) for j in range(1, N-1, 2)]  # (1,2),(3,4),...

    # ── GHZ initial state ────────────────────────────────────────
    # |GHZ⟩ = (|00...0⟩ + |11...1⟩)/√2
    # Index 0 = |00...0⟩,  index D-1 = |11...1⟩
    psi = jnp.zeros(D, dtype=CDT)
    psi = psi.at[0].set(1/jnp.sqrt(2.0))
    psi = psi.at[D-1].set(1/jnp.sqrt(2.0))

    # ── Apply L brickwork layers ─────────────────────────────────
    for _ in range(L):
        # Reshape flat vector → rank-N tensor for einsum contractions
        p = psi.reshape((2,)*N)

        # Step 1: Ry(π/4) on ALL qubits (magic injection)
        # Each einsum contracts _RY (2×2) with one tensor axis
        for j in range(N):
            p = jnp.einsum(sub1[j], _RY, p)

        # Step 2: CNOT on even pairs (0,1), (2,3), ...
        # Each einsum contracts CNOT_T (2,2,2,2) with two tensor axes
        for (c, t) in even:
            p = jnp.einsum(sub2[(c, t)], CNOT_T, p)

        # Step 3: CNOT on odd pairs (1,2), (3,4), ...
        for (c, t) in odd:
            p = jnp.einsum(sub2[(c, t)], CNOT_T, p)

        # Flatten back for next iteration
        psi = p.reshape(D)

    return _get_spectrum(psi, N)


def simulate_haar(N, key):
    """
    Generate Haar-random pure state → entanglement spectrum.

    ALGORITHM (Gaussian method):
      1. Draw z_j = x_j + i·y_j where x_j, y_j ~ N(0,1) iid
      2. Normalise: |ψ⟩ = z / ||z||

    WHY THIS WORKS:
      The multivariate complex Gaussian CN(0,I) is rotationally
      invariant under any unitary U (since U preserves the norm).
      After normalisation to the unit sphere, this gives an EXACT
      Haar-uniform sample on CP^{D-1} (complex projective space).

    This provides the theoretical reference distribution for the
    entanglement spectrum, which should follow Marchenko-Pastur.

    Returns: eigenvalues of ρ_A, shape (2^{N/2},)
    """
    D = 1 << N
    k1, k2 = random.split(key)
    z = (random.normal(k1, (D,)) + 1j * random.normal(k2, (D,))).astype(CDT)
    z = z / jnp.linalg.norm(z)
    return _get_spectrum(z, N)


def simulate_clifford(N, L):
    """
    Clifford brickwork circuit → entanglement spectrum.

    NOT CALLED by this script (nor are _H and _S, which exist only to feed it).
    This is a copy of the function in get_fig_ch5_dkl_convergence.py, which is
    the script that actually computes the Clifford curve of Fig. 5.2(e).
    Retained because it documents that negative control; deleting it would
    change no output here.

    CIRCUIT STRUCTURE (Clifford_brickwork):

      |ψ₀⟩ = GHZ = (|0...0⟩ + |1...1⟩)/√2

      For each layer ℓ = 1, ..., L:
        1. CNOT on even pairs: (0,1), (2,3), ...
        2. H + S on all qubits       ← Clifford single-qubit gates
        3. CNOT on odd  pairs: (1,2), (3,4), ...
        4. H + S on all qubits

    KEY DIFFERENCE FROM Ry+CNOT:
      Clifford circuits use ONLY Clifford gates (H, S, CNOT).
      The resulting state is ALWAYS a stabilizer state, meaning:
        • Reduced density matrix ρ_A = (1/χ)Π (projector)
        • ALL non-zero eigenvalues are EQUAL → flat spectrum
        • Spectrum is a DELTA FUNCTION, never approaches MP!

    This provides the negative control: Clifford D_KL stays HIGH,
    proving that MAGIC is needed for MP convergence.

    Returns: eigenvalues of ρ_A, shape (2^{N/2},)
    """
    D = 1 << N
    sub1 = _subs_1q(N)
    sub2 = _subs_2q(N)
    even = [(j, j+1) for j in range(0, N-1, 2)]
    odd  = [(j, j+1) for j in range(1, N-1, 2)]

    # GHZ initial state
    psi = jnp.zeros(D, dtype=CDT)
    psi = psi.at[0].set(1/jnp.sqrt(2.0))
    psi = psi.at[D-1].set(1/jnp.sqrt(2.0))

    for _ in range(L):
        p = psi.reshape((2,)*N)
        # CNOT even
        for (c, t) in even:
            p = jnp.einsum(sub2[(c, t)], CNOT_T, p)
        # H + S on all qubits
        for j in range(N):
            p = jnp.einsum(sub1[j], _H, p)
            p = jnp.einsum(sub1[j], _S, p)
        # CNOT odd
        for (c, t) in odd:
            p = jnp.einsum(sub2[(c, t)], CNOT_T, p)
        # H + S on all qubits
        for j in range(N):
            p = jnp.einsum(sub1[j], _H, p)
            p = jnp.einsum(sub1[j], _S, p)
        psi = p.reshape(D)

    return _get_spectrum(psi, N)


# ══════════════════════════════════════════════════════════════════
#  MARCHENKO–PASTUR DENSITY (in ξ = −ln λ space)
# ══════════════════════════════════════════════════════════════════
#
#  ρ_MP(x) = (1/2πcx) · √[(x₊−x)(x−x₋)]
#
#  In ξ = −ln(λ) space with x = m·e^{−ξ}:
#    ρ_ξ(ξ) = ρ_MP(x) · |dx/dξ| = ρ_MP(x) · x
#
# ══════════════════════════════════════════════════════════════════

def mp_density_xi(xi, c, m):
    """MP density ρ_ξ(ξ) with Jacobian."""
    x = m * np.exp(-xi)
    xm = (1 - np.sqrt(c))**2
    xp = (1 + np.sqrt(c))**2
    rho = np.zeros_like(x)
    mask = (x > xm) & (x < xp)
    rho[mask] = np.sqrt((xp - x[mask]) * (x[mask] - xm)) / \
                (2 * np.pi * c * x[mask])
    return rho * x


# ══════════════════════════════════════════════════════════════════
#  KL DIVERGENCE: D_KL(P_data || P_MP)
# ══════════════════════════════════════════════════════════════════

def kl_from_mp(eigenvalues, c, m, n_bins=50):
    """
    KL divergence between empirical ξ-histogram and MP density.

    NOT CALLED by this script; the four panels produced here compare the
    spectra visually rather than metrically. This is a copy of the function in
    get_fig_ch5_dkl_convergence.py (which uses n_bins=30, not 50), the script
    that actually computes Fig. 5.2(e). Retained for reference only: do not
    assume this copy is the one that produced any published number.

    SPECIAL CASE: For stabilizer states, the spectrum is a delta
    function (all eigenvalues equal). In this case the histogram
    has zero width → no overlap with MP → D_KL is effectively ∞.
    We return a large finite value (100) as a sentinel.
    """
    valid = eigenvalues[eigenvalues > 1e-25]
    if len(valid) < 2:
        return 100.0   # too few eigenvalues → maximally non-MP
    xi = -np.log(valid)

    # If spectrum is degenerate (e.g. stabilizer: all λ equal),
    # the spread of ξ is ~0 → clearly not MP
    spread = np.ptp(xi)
    if spread < 1e-6:
        return 100.0

    lo, hi = np.percentile(xi, [1, 99])
    bins = np.linspace(max(lo, 0.5), min(hi, 30.0), n_bins + 1)
    w = np.diff(bins)
    ctr = 0.5 * (bins[:-1] + bins[1:])
    counts, _ = np.histogram(xi, bins=bins)
    total = counts.sum()
    if total == 0:
        return 100.0
    p = counts / (total * w)
    q = mp_density_xi(ctr, c, m)
    ok = (p > 1e-12) & (q > 1e-12)
    if ok.sum() < 1:
        return 100.0
    return max(np.sum(p[ok] * np.log(p[ok] / q[ok]) * w[ok]), 1e-4)


# ══════════════════════════════════════════════════════════════════
#  DATA GENERATION
# ══════════════════════════════════════════════════════════════════

def generate_all_data(regenerate=False):
    """Generate Ry+CNOT spectra and Haar reference for histograms."""
    key = random.PRNGKey(42)
    t0 = time.time()

    # ── Ry+CNOT spectra at histogram depths ──────────────────────
    ry_file = DATA_DIR / f"spectra_ry_N{N_Q}.npz"
    if ry_file.exists() and not regenerate:
        print(f"  Cached: {ry_file}")
    else:
        print(f"\n  Ry+CNOT N={N_Q} (deterministic):")
        d = {}
        for L in HIST_DEPTHS:
            t1 = time.time()
            eigs = simulate_ry_cnot(N_Q, L)
            d[f"L{L}"] = eigs
            xi = -np.log(eigs[eigs > 1e-25])
            print(f"    L={L:2d}: {len(xi)} eigs, "
                  f"ξ∈[{xi.min():.1f},{xi.max():.1f}] ({time.time()-t1:.1f}s)")
        np.savez(ry_file, **d)
        print(f"    → {ry_file}")

    # ── Haar-random reference ────────────────────────────────────
    haar_file = DATA_DIR / f"spectra_haar_N{N_Q}.npz"
    if haar_file.exists() and not regenerate:
        print(f"  Cached: {haar_file}")
    else:
        print(f"  Haar N={N_Q} ({N_HAAR} samples):")
        sp = []
        for i in range(N_HAAR):
            key, sk = random.split(key)
            sp.append(simulate_haar(N_Q, sk))
        np.savez(haar_file, spectra=np.stack(sp))
        print(f"    → {haar_file}")

    print(f"\n  Total: {time.time()-t0:.1f}s")


def data_exists():
    return all(f.exists() for f in [
        DATA_DIR / f"spectra_ry_N{N_Q}.npz",
        DATA_DIR / f"spectra_haar_N{N_Q}.npz",
    ])


# ══════════════════════════════════════════════════════════════════
#  PLOTTING
# ══════════════════════════════════════════════════════════════════
#
#  Layout: 4 panels (a-d) at L = 5, 10, 15, 20
#  Each panel: gray Haar histogram (background) + red circuit histogram
#              + black MP analytical curve
#
# ══════════════════════════════════════════════════════════════════

def compute():
    """Ensure the spectrum caches exist, then merge them into one dict.

    The expensive statevector simulation + Haar sampling live in
    generate_all_data(), which writes/loads spectra_ry_N{N}.npz and
    spectra_haar_N{N}.npz -- the SAME caches get_fig_ch5_combined.py reads to
    assemble the book's five-panel Fig. 5.2, so those are preserved untouched.
    """
    generate_all_data(regenerate=False)
    ry = np.load(DATA_DIR / f"spectra_ry_N{N_Q}.npz")
    haar = np.load(DATA_DIR / f"spectra_haar_N{N_Q}.npz")
    out = {f"ry_L{L}": ry[f"L{L}"] for L in HIST_DEPTHS}
    out["haar_spectra"] = haar["spectra"]
    return out


def plot_figure(data):
    """
    4-panel figure: each panel overlays Haar (gray) + circuit (red).
    Panels (a-d): L = 5, 10, 15, 20 for N = 20 qubits.
    """
    print("  Plotting figure...")

    k = N_Q // 2
    m = 1 << k
    n = (1 << N_Q) >> k
    c = m / n
    xi_grid = np.linspace(1.0, 30.0, 2000)
    mp_curve = mp_density_xi(xi_grid, c, m)

    # Pooled Haar spectrum for background histogram
    haar_sp = data["haar_spectra"]
    xi_haar = -np.log(haar_sp[haar_sp > 1e-25])

    apply_book_style()
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.8), sharey=True)
    fig.subplots_adjust(wspace=0.08)

    bins = np.linspace(2, 28, 55)

    for i, L in enumerate(HIST_DEPTHS):
        ax = axes[i]

        # Gray Haar histogram (background)
        ax.hist(xi_haar, bins=bins, density=True, alpha=0.35,
                color="gray", edgecolor="white", linewidth=0.4,
                zorder=1, label="Haar")

        # Red circuit histogram (foreground, opaque)
        eigs = data[f"ry_L{L}"]
        xi = -np.log(eigs[eigs > 1e-25])
        ax.hist(xi, bins=bins, density=True, alpha=0.7,
                color="#E63946", edgecolor="white", linewidth=0.4,
                zorder=2, label=f"$L={L}$")

        # MP analytical curve
        ax.plot(xi_grid, mp_curve, "k-", lw=2.5, alpha=0.85, zorder=10)

        ax.set_xlim(2, 28)
        ax.set_ylim(0, 0.37)
        panel_label(ax, "abcd"[i], loc="upper left")
        # circuit depth of this panel (physics info, kept as a small annotation)
        ax.text(0.97, 0.95, rf"$L={L}$", transform=ax.transAxes,
                fontsize=12, va="top", ha="right")
        ax.set_xlabel(r"$\xi = -\ln\,\lambda$")

    axes[0].set_ylabel("Density")

    # NOT fig_ch5_mp_convergence.pdf: that name belongs to the book's Fig. 5.2, which is the
    # FIVE-panel figure assembled by get_fig_ch5_combined.py from this script's caches plus the
    # D_KL caches. This script draws only the four spectrum panels, so writing it under the
    # book's name silently replaced Fig. 5.2 with a version its caption no longer described.
    outpath = OUTPUT_DIR / "fig_ch5_mp_spectra.pdf"
    fig.savefig(outpath)
    fig.savefig(outpath.with_suffix(".png"))
    print(f"    → PDF: {outpath}")
    print(f"    → PNG: {outpath.with_suffix('.png')}")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════

print("=" * 65)
print("  Chapter 5 — Entanglement Spectrum → Marchenko–Pastur")
print("  Matrix-free JAX statevector simulator (F64)")
print("=" * 65)
print(f"  N = {N_Q} qubits, k = {N_Q//2} (equal bipartition)")
print(f"  Histograms: L = {HIST_DEPTHS}")
print(f"  Haar:       {N_HAAR} samples")
print(f"  JAX:        {jax.default_backend()}")
print("=" * 65)

data = load_or_compute(DATA_DIR / "fig_ch5_mp_spectra.npz", compute)

print("\n── Figure ──")
plot_figure(data)
print(f"\nDone → {OUTPUT_DIR}")
