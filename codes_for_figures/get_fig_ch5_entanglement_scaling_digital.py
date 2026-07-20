#!/usr/bin/env python3
"""
get_fig_ch5_entanglement_scaling_digital.py
===========================================
Entanglement entropy of a DIGITAL (gate-circuit) scrambler versus system size.

Computes the half-chain von Neumann entropy S_A(N/2) of an Ry(pi/4) + CNOT
brickwork circuit for system sizes N = 6, 8, ..., 20 at fixed circuit depths
L = 0, 2, 4, 8, 16 and at the scaling depth L = N, together with the Page
value as a Haar-random benchmark.

============================================================
WHICH FIGURE THIS FEEDS  (read before running)
============================================================
Book figure:  Figure 5.3, PANEL (a)   (ch5_designs.tex, \\label{fig:entanglement_scaling})

Figure 5.3 in the book is a TWO-panel figure:
    panel (a) = this script's digital brickwork circuit, depths L = 0,2,4,8,16,N
    panel (b) = get_fig_ch5_entanglement_scaling_analog.py, Trotterized
                Hamiltonian evolution at fixed times t = 0,2,4,6,8,10

The two-panel PDF that the book actually includes is assembled by the ANALOG
script, which reads this script's cached array (data/ch5/entropy_vs_N_depths.npz)
to draw panel (a) and computes panel (b) itself.

FILENAME COLLISION (known, deliberate, do not "fix" casually):
This script also writes a standalone SINGLE-panel figures/ch5/fig_ch5_entanglement_scaling.pdf
using the same output name as the analog script's two-panel version. Whichever
script runs last wins that filename. The correct order is therefore:

    1. python3 get_fig_ch5_entanglement_scaling_digital.py    (produces panel-(a) data)
    2. python3 get_fig_ch5_entanglement_scaling_analog.py     (writes the book's 2-panel PDF)

Running the digital script alone, or last, leaves a single-panel PDF on disk that
does NOT match the book's Figure 5.3 caption. This script's own plot_figure() is
best understood as a preview/diagnostic of panel (a), not as the book figure.

============================================================
PHYSICS BACKGROUND
============================================================
"DIGITAL" here means scrambling by a discrete, layered GATE CIRCUIT, as opposed
to the continuous-time Hamiltonian evolution of the analog sibling script. Time
is not a parameter: the control knob is the integer circuit DEPTH L, the number
of brickwork layers applied. Each layer is

    U^(l) = prod_i Ry^(i)(pi/4) . prod_{i even} CNOT_{i,i+1} . prod_{i odd} CNOT_{i,i+1}

so the circuit is built from a fixed, deterministic gate set: a single-qubit
Ry rotation by pi/4 on every site, then CNOTs on even bonds, then CNOTs on odd
bonds. The initial state is the GHZ state (|0...0> + |1...1>)/sqrt(2).

WHAT THE SCALING WITH N SHOWS
For a pure state cut into A (first N/2 qubits) and B (the rest), S_A measures
the entanglement across the cut. Plotting S_A(N/2) against N separates two
regimes:

  Fixed depth L:  entanglement can only spread within the circuit's light cone.
                  Each brickwork layer couples nearest neighbours, so after L
                  layers only sites within O(L) of the cut are correlated across
                  it. S_A therefore saturates at an L-dependent plateau once N
                  exceeds roughly the light-cone width, and stops growing with N.
                  This is AREA-LAW-LIKE behaviour: the entropy tracks the size of
                  the boundary (one cut) rather than the volume of A.

  Depth L = N:    the light cone spans the whole chain, and S_A grows linearly
                  in N. This is the VOLUME LAW: S_A proportional to |A| = N/2.

THE PAGE VALUE
A Haar-random pure state of N qubits has, on average, half-chain entropy given
by Page's formula, which is very close to but slightly below the maximum
(N/2) ln 2. The deficit is the O(1) Page correction. The Page curve is drawn as
the gray dashed reference: it is the benchmark for "as entangled as a random
state". Note the Page value is the Haar AVERAGE, so an individual state may sit
slightly above it; this is not an error.

READER'S TAKEAWAY
Only circuits whose depth scales with system size (L proportional to N) reach
volume-law, near-Page entanglement. Fixed-depth circuits are stuck at O(L)
entanglement no matter how large N grows. This light-cone argument is why
scrambling to a Haar-like state requires depth L = Omega(N).

The larger point of Figure 5.3, made by reading panels (a) and (b) together, is
a NEGATIVE result: both the digital circuit and the analog Hamiltonian reach the
Page scale, so a scalar entropy near its Page value does NOT certify Haar-like
randomness or classical hardness. High entanglement is necessary but far from
sufficient, which is what motivates the finer diagnostics (entanglement spectrum
shape, nonstabilizerness) later in Chapter 5.

WHAT THE COMPUTED DATA ACTUALLY LOOKS LIKE (units of ln 2, N = 6 ... 20)
    L = 0  : exactly 1.00 at every N. The GHZ state has S_A = ln 2 across any cut.
    L = 2  : saturates near 1.7 to 2.2
    L = 4  : saturates near 2.8 to 3.5
    L = 8  : still rising at N = 20 (light cone has not yet been outrun)
    L = 16 : still tracks Page closely up to N = 20
    L = N  : lands within about 0.03 of the Page value for N >= 8
Only L = 2 and L = 4 visibly saturate within N <= 20; L = 8 and L = 16 would need
larger N to show their plateaus. The small-depth curves also alternate with
N mod 4, a finite-size artefact of the deterministic (non-random) gate pattern.

============================================================
ALGORITHM
============================================================
STATE EVOLUTION (_evolve_ry_cnot)
The state vector of length D = 2^N is reshaped to an N-index tensor of shape
(2,)*N, one axis per qubit. Gates are then applied as tensor contractions with
jnp.einsum, contracting the gate's input leg against the target qubit's axis:

    1-qubit: "Aa,...a...->...A..."           (Ry on qubit j)
    2-qubit: "ABab,...a...b...->...A...B..." (CNOT on bond (c,t))

The einsum subscript strings are precomputed once per N by _subs_1q / _subs_2q,
because building them inside the layer loop would rebuild identical strings
L * N times. Lowercase letters index the contracted (input) axes, uppercase the
new (output) axes, which keeps every qubit's axis in its original position.
CNOT is stored pre-reshaped to (2,2,2,2) so no reshape happens per gate.

Each of the L layers applies: Ry on all N qubits, then CNOTs on even bonds, then
CNOTs on odd bonds. The tensor is flattened back to length D at the end.

ENTROPY (entanglement_entropy)
For subsystem A = the first k qubits, the state vector is reshaped into the
Schmidt matrix C of shape (2^k, 2^(N-k)), so that row index = A configuration and
column index = B configuration. This works with no transpose because the reshape
of (2,)*N in C order already puts the first k qubits in the leading (row) index.
Then rho_A = C C^dagger via einsum('ab,cb->ac'), eigenvalues from eigvalsh, and
S = -sum(p log p).

NO AVERAGING OVER REALIZATIONS
There is deliberately none. The gate set (Ry angle pi/4, CNOT pattern) and the
GHZ initial state are entirely deterministic, so each (N, L) point is one exact
number rather than a sample mean. There is no randomness in this script at all,
hence no seed and no error bars. (The analog sibling DOES draw random fields
h_i, so it is the one that needs seeding.) This is why the curves are perfectly
smooth apart from the genuine N mod 4 structure noted above.

============================================================
IMPLEMENTATION NOTES  (the "why")
============================================================
WHY THE k > N-k BRANCH IN entanglement_entropy
The function sets k_eff = min(k, N-k) and, when k > N-k, builds C transposed so
that rho is the SMALLER of the two reduced density matrices. For a pure state
S(rho_A) = S(rho_B), so this returns the identical number while diagonalising a
2^(N-k) matrix instead of a 2^k one. In this script k = N//2 always, so this
branch never actually fires; it is there to keep the helper correct if reused
with an unequal cut.

WHY EIGENVALUES ARE CLIPPED
jnp.clip(eigvalsh(rho), 1e-30, None) guards the p log p sum. Exactly-zero Schmidt
coefficients are generic here (the GHZ state has only two), and floating-point
roundoff can push a true zero eigenvalue slightly NEGATIVE. Either case would
make log() return -inf or NaN and poison the sum. Clipping to 1e-30 sends those
terms to a harmless ~1e-30 * (-69) ~ 0 contribution. The clip floor must stay far
below any physical eigenvalue, which 1e-30 comfortably is.

WHY JAX, AND WHY NO jit
JAX is used for fast einsum over the (2,)*N tensor and for float64 (enabled via
JAX_ENABLE_X64=1 at the top of the file, before the jax import: JAX defaults to
float32, which is not enough precision for entropies near the Page value). It
runs eagerly, with no jit/vmap. Each (N, L) point is computed once and the einsum
calls already dispatch to BLAS-backed kernels, so tracing/compiling would mostly
add warm-up cost per distinct N. XLA_PYTHON_CLIENT_PREALLOCATE=false stops JAX
from grabbing most of the GPU up front, which matters only if a GPU is present.

WHY EVEN SYSTEM SIZES ONLY
SYSTEM_SIZES steps by 2 so that N/2 is an integer and the bipartition is exactly
equal at every N. An odd N would make the cut asymmetric and the comparison
against the Page value at k = N/2 ill-defined.

_AX HAS 21 LETTERS
"abcdefghijklmnopqrstu" allows N <= 21 einsum axes (uppercase counterparts give
the outputs). SYSTEM_SIZES tops out at N = 20, so this fits with one to spare.
Raising the maximum N past 21 requires extending _AX. Note also that _subs_2q
eagerly builds subscripts for all N*(N-1) ordered pairs while only the ~N
brickwork bonds are used; the strings are tiny, so this is harmless.

CACHING IS INCREMENTAL, AND THE CACHE MATRIX IS SHAPE-KEYED
The (len(Ns), len(depths)+1) array is saved to NPZ after EVERY (N, L) point, so a
crash or interrupt during the expensive large-N runs loses at most one point. On
restart, entries that are not NaN are skipped, so the run resumes where it left
off. NaN is the "not yet computed" sentinel, which is why the array is allocated
with np.full(..., np.nan). The cache is only reused if its shape matches the
current SYSTEM_SIZES/FIXED_DEPTHS; editing those config lists changes the shape
and forces a clean recompute.

DATA_DIR AND OUTPUT_DIR ARE BOTH ABSOLUTE (resolved from __file__)
Both paths are derived from __file__, so this script may be run from any
working directory and always reads/writes the same data/ch5 cache. The analog
script resolves DATA_DIR the same way, so it finds this script's panel-(a)
cache (entropy_vs_N_depths.npz) regardless of where either is launched.

============================================================
RUNTIME AND MEMORY
============================================================
Cost is dominated by the largest sizes: the state vector holds D = 2^N complex128
amplitudes (N = 20: 2^20 * 16 B = 16 MB per copy, and einsum needs a second copy
for its output). The reduced density matrix at the half cut is 2^(N/2) x 2^(N/2)
(N = 20: 1024 x 1024 = 16 MB), and its eigendecomposition is negligible next to
the circuit evolution. Peak memory stays well under 1 GB, so this is CPU-time
bound rather than memory bound.

Work per (N, L) point scales as O(L * N * 2^N). The L = N points at N = 18, 20
dominate the total. A full cold run over all sizes and depths is minutes, not
hours, on a modern CPU; a warm (fully cached) run is seconds.

FLAGS
    (no flags)     compute any missing (N, L) points, then plot
    --plot-only    skip computation and plot straight from the NPZ cache;
                   errors out if the cache is missing
    --regenerate   intended to force a recompute from scratch. CAVEAT: it only
                   takes effect when the cached array's SHAPE differs from the
                   current config. If a complete, shape-matching cache exists,
                   every point is skipped as already-computed and --regenerate
                   is a no-op. To truly recompute, delete
                   data/ch5/entropy_vs_N_depths.npz.

============================================================
FILES
============================================================
Reads/writes cache:  data/ch5/entropy_vs_N_depths.npz   (absolute, from __file__)
                     keys: Ns, depths, sa (len(Ns) x len(depths)+1; last column
                     is L=N), sa_page
Writes figure:       ../figures/ch5/fig_ch5_entanglement_scaling.pdf and .png
                     (single-panel preview; OVERWRITTEN by the analog script's
                     two-panel version, which is the one the book includes)
"""

import os, sys, time, argparse
from pathlib import Path

# Both must be set BEFORE jax is imported below; JAX reads them at import time.
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"  # don't grab the whole GPU up front
os.environ["JAX_ENABLE_X64"] = "1"                     # float64: float32 is too coarse
                                                       # to resolve entropies near Page

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_style import apply_book_style, load_or_compute, panel_label

import jax
import jax.numpy as jnp
from jax import random

CDT = jnp.complex128
FDT = jnp.float64

# ══════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════════════
#
#  System sizes: N = 6, 8, 10, ..., 20  (even, for clean equal bipartition)
#  Fixed depths: L = 0, 2, 4, 8, 16     (constant depth, area-law regime)
#  Scaling depth: L = N                  (volume-law regime)
#
# ══════════════════════════════════════════════════════════════════

SYSTEM_SIZES = list(range(6, 21, 2))      # N = 6, 8, 10, ..., 20
FIXED_DEPTHS = [0, 2, 4, 8, 16]           # constant circuit depths
DATA_DIR = Path(__file__).resolve().parent / "data" / "ch5"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "figures" / "ch5"
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════
#  GATE DEFINITIONS
# ══════════════════════════════════════════════════════════════════

# Ry(theta) = [[cos(theta/2), -sin(theta/2)], [sin(theta/2), cos(theta/2)]].
# The pi/8 entries are theta/2 for theta = pi/4, i.e. this IS Ry(pi/4).
_RY = jnp.array([[jnp.cos(jnp.pi/8), -jnp.sin(jnp.pi/8)],
                  [jnp.sin(jnp.pi/8),  jnp.cos(jnp.pi/8)]], dtype=CDT)
# Stored pre-reshaped to (control_out, target_out, control_in, target_in) so the
# per-gate einsum in the layer loop needs no reshape.
CNOT_T = jnp.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]],
                    dtype=CDT).reshape(2,2,2,2)

# ══════════════════════════════════════════════════════════════════
#  EINSUM SUBSCRIPT GENERATION
# ══════════════════════════════════════════════════════════════════

# One lowercase letter per qubit axis; uppercase 'A'+j are the matching output
# legs. 21 letters => supports N <= 21 (SYSTEM_SIZES stops at 20).
_AX = "abcdefghijklmnopqrstu"

def _subs_1q(N):
    """Einsum subscripts for 1-qubit gates on each of N qubits.

    Built once per N and reused across all L layers: the strings are identical
    every layer, so generating them in the loop would rebuild them L*N times.
    Only axis j is relabelled, which keeps every qubit in its original position.
    """
    subs = []
    for j in range(N):
        ax = list(_AX[:N])
        out = list(ax)
        new = chr(ord('A') + j)
        out[j] = new
        subs.append(f"{new}{ax[j]},{''.join(ax)}->{''.join(out)}")
    return subs

def _subs_2q(N):
    """Einsum subscripts for 2-qubit gates on all qubit pairs.

    Enumerates all N*(N-1) ordered pairs though only the ~N brickwork bonds are
    ever looked up. The strings are tiny, so the waste is negligible.
    """
    subs = {}
    for c in range(N):
        for t in range(N):
            if c == t: continue
            ax = list(_AX[:N])
            out = list(ax)
            C, T = chr(ord('A')+c), chr(ord('A')+t)
            out[c], out[t] = C, T
            subs[(c,t)] = f"{C}{T}{ax[c]}{ax[t]},{''.join(ax)}->{''.join(out)}"
    return subs


# ══════════════════════════════════════════════════════════════════
#  ENTANGLEMENT ENTROPY
# ══════════════════════════════════════════════════════════════════

def entanglement_entropy(psi, N, k):
    """Von Neumann entropy S_A for subsystem A = first k qubits.

    Returns S in natural-log units (the caller divides by ln 2 to plot in bits).
    """
    # Diagonalise whichever reduced density matrix is SMALLER: for a pure state
    # S(rho_A) = S(rho_B), so this is the same number for less work.
    # Callers here always pass k = N//2, so the k > N-k branch never fires.
    k_eff = min(k, N - k)
    m = 1 << k_eff
    n = (1 << N) >> k_eff
    if k <= N - k:
        # C-order reshape already puts the first k qubits in the row index,
        # so this is the Schmidt matrix (A configuration, B configuration).
        C = psi.reshape(m, n)
    else:
        C = psi.reshape(n, m).T
    rho = jnp.einsum('ab,cb->ac', C, jnp.conj(C))
    # Clip guards the p*log(p) sum: zero Schmidt coefficients are generic (GHZ
    # has only two nonzero), and roundoff can push a true zero slightly negative,
    # either of which gives -inf/NaN. 1e-30 is far below any physical eigenvalue.
    eigs = jnp.clip(jnp.linalg.eigvalsh(rho), 1e-30, None)
    return float(-jnp.sum(eigs * jnp.log(eigs)))


# ══════════════════════════════════════════════════════════════════
#  CIRCUIT SIMULATOR — Ry(π/4) + CNOT brickwork
# ══════════════════════════════════════════════════════════════════
#
#  This is the same scrambling unitary from Chapter 5, Eq. (12):
#
#    Û^(ℓ) = ∏_i R̂_y^(i)(π/4) · ∏_{i∈e} CNOT_{i,i+1} · ∏_{i∈o} CNOT_{i,i+1}
#
#  Starting from the GHZ state (|0...0⟩ + |1...1⟩)/√2.
#
# ══════════════════════════════════════════════════════════════════

def _evolve_ry_cnot(N, L):
    """Ry(π/4)+CNOT brickwork for L layers, starting from GHZ."""
    D = 1 << N
    sub1, sub2 = _subs_1q(N), _subs_2q(N)
    even = [(j, j+1) for j in range(0, N-1, 2)]
    odd  = [(j, j+1) for j in range(1, N-1, 2)]

    # GHZ initial state
    psi = jnp.zeros(D, dtype=CDT)
    psi = psi.at[0].set(1/jnp.sqrt(2.0))
    psi = psi.at[D-1].set(1/jnp.sqrt(2.0))

    # One brickwork layer = Ry on every site, then even bonds, then odd bonds.
    # Gates act as tensor contractions on the (2,)*N view, one axis per qubit.
    for _ in range(L):
        p = psi.reshape((2,)*N)
        for j in range(N):
            p = jnp.einsum(sub1[j], _RY, p)
        for c, t in even:
            p = jnp.einsum(sub2[(c, t)], CNOT_T, p)
        for c, t in odd:
            p = jnp.einsum(sub2[(c, t)], CNOT_T, p)
        psi = p.reshape(D)
    return psi  # L = 0 returns the GHZ state unchanged


# ══════════════════════════════════════════════════════════════════
#  PAGE'S FORMULA
# ══════════════════════════════════════════════════════════════════

def page_entropy(N, k):
    """Exact Page entropy for N qubits, subsystem of k qubits.

    Mean half-chain entropy of a Haar-random pure state, in natural-log units:
        S_page = sum_{j=d_B+1}^{d_A*d_B} 1/j  -  (d_A - 1)/(2*d_B),   d_A <= d_B
    This is an INDEPENDENT analytic benchmark: the plotted circuit entropies come
    from actual state evolution, never from this formula, so the agreement at
    L = N is a real check rather than a tautology.
    The convention d_A <= d_B is required by the formula, hence the min/max.
    """
    d_A = 1 << min(k, N - k)
    d_B = 1 << max(k, N - k)
    S = sum(1.0/j for j in range(d_B + 1, d_A * d_B + 1))
    S -= (d_A - 1) / (2.0 * d_B)
    return S


# ══════════════════════════════════════════════════════════════════
#  DATA GENERATION — WITH INCREMENTAL SAVES
# ══════════════════════════════════════════════════════════════════
#
#  For each (N, L) pair, we evolve the Ry+CNOT brickwork circuit
#  and compute S_A at the equal bipartition k = N/2.
#
#  The data is stored as a 2D array: rows = system sizes, cols = depths.
#  We save after each (N, L) computation for crash protection.
#
# ══════════════════════════════════════════════════════════════════

def generate_data(regenerate=False):
    t0 = time.time()

    sa_file = DATA_DIR / "entropy_vs_N_depths.npz"

    Ns = np.array(SYSTEM_SIZES)
    # All depths to compute: fixed depths + L=N for each N
    all_depths = sorted(set(FIXED_DEPTHS))  # we handle L=N separately

    # Data matrix: rows = N, cols = fixed depths + L=N.
    # NaN is the "not yet computed" sentinel that drives the resume logic below.
    n_depths = len(all_depths) + 1  # +1 for L=N
    sa = np.full((len(Ns), n_depths), np.nan)
    sa_page = np.full(len(Ns), np.nan)

    # Load partial data if available. The cache is keyed only by SHAPE, so
    # editing SYSTEM_SIZES/FIXED_DEPTHS changes the shape and forces a recompute.
    if sa_file.exists():
        old = np.load(sa_file)
        if "sa" in old and old["sa"].shape == sa.shape:
            # NOTE: `regenerate` is deliberately not consulted here, so a
            # complete shape-matching cache makes --regenerate a no-op (every
            # point below is skipped as already-computed). Delete the NPZ to
            # force a true recompute.
            sa = old["sa"].copy()
            sa_page = old["sa_page"].copy()
            n_done = np.count_nonzero(~np.isnan(sa))
            print(f"  Loaded partial data: {n_done} values computed")
        elif not regenerate:
            print(f"  Cached: {sa_file}")
            return

    print(f"\n  S_A(N/2) vs N at depths L = {all_depths} + [L=N]")
    print(f"  System sizes: {list(Ns)}\n")

    for i, N in enumerate(Ns):
        k = N // 2

        # Page formula
        if np.isnan(sa_page[i]):
            sa_page[i] = page_entropy(N, k)

        # Fixed depths
        for j, L in enumerate(all_depths):
            if not np.isnan(sa[i, j]):
                continue  # already computed

            print(f"    N={N:2d}, L={L:2d}: ", end="", flush=True)
            t1 = time.time()

            if L == 0:
                # GHZ state: S_A = ln 2 for any bipartition
                D = 1 << N
                psi = jnp.zeros(D, dtype=CDT)
                psi = psi.at[0].set(1/jnp.sqrt(2.0))
                psi = psi.at[D-1].set(1/jnp.sqrt(2.0))
            else:
                psi = _evolve_ry_cnot(N, L)

            sa[i, j] = entanglement_entropy(psi, N, k)
            dt = time.time() - t1
            print(f"S/ln2 = {sa[i,j]/np.log(2):.3f}  ({dt:.1f}s)")

            # Save after every point: an interrupt during the expensive large-N
            # runs then costs at most this one value, not the whole sweep.
            np.savez(sa_file, Ns=Ns, depths=np.array(all_depths),
                     sa=sa, sa_page=sa_page)

        # L = N (scaling depth). Kept in the last column rather than in
        # all_depths because its value differs per row, so it cannot be a
        # shared column label like the fixed depths.
        j_N = len(all_depths)  # last column index
        if np.isnan(sa[i, j_N]):
            print(f"    N={N:2d}, L=N={N:2d}: ", end="", flush=True)
            t1 = time.time()
            psi = _evolve_ry_cnot(N, N)
            sa[i, j_N] = entanglement_entropy(psi, N, k)
            dt = time.time() - t1
            print(f"S/ln2 = {sa[i,j_N]/np.log(2):.3f}  ({dt:.1f}s)")

            # Incremental save
            np.savez(sa_file, Ns=Ns, depths=np.array(all_depths),
                     sa=sa, sa_page=sa_page)

    print(f"\n  → {sa_file} (complete)")
    print(f"  Total: {time.time()-t0:.1f}s")


# ══════════════════════════════════════════════════════════════════
#  PLOTTING
# ══════════════════════════════════════════════════════════════════
#
#  Single panel matching the reference style:
#    • L = 0       → open circles, flat at S = 1 (area law)
#    • L = 2,4,8   → increasing curves that saturate
#    • L = 16      → near-volume-law for small N, saturates for large N
#    • L = N       → red X markers with dashed line (volume law)
#    • Page value  → gray dashed reference
#
# ══════════════════════════════════════════════════════════════════

def plot_figure():
    print("  Plotting figure...")

    data = np.load(DATA_DIR / "entropy_vs_N_depths.npz")
    Ns = data["Ns"]
    depths = data["depths"]
    sa = data["sa"]            # shape: (len(Ns), len(depths)+1)
    sa_page = data["sa_page"]

    LN2 = np.log(2)

    # ── Colour palette matching reference ─────────────────────────
    # L=0: light blue (open circles), L=2: blue, L=4: green,
    # L=8: orange, L=16: purple, L=N: red (dashed, X markers)
    colors = {
        0:  "#7FCDBB",   # light teal
        2:  "#2C7FB8",   # blue
        4:  "#7FBC41",   # green
        8:  "#FD8D3C",   # orange
        16: "#9E4A9C",   # purple
    }
    markers = {
        0:  "o",
        2:  "o",
        4:  "^",
        8:  "s",
        16: "p",
    }

    apply_book_style()
    fig, ax = plt.subplots(1, 1, figsize=(6.5, 5))

    # Page value reference (gray dashed)
    ax.plot(Ns, sa_page / LN2, "--", color="gray", lw=1.5, alpha=0.6,
            label="Page value", zorder=1)

    # Fixed-depth curves
    for j, L in enumerate(depths):
        L = int(L)
        c = colors.get(L, "gray")
        m = markers.get(L, "o")
        facecolor = "white" if L == 0 else c
        ax.plot(Ns, sa[:, j] / LN2, f"-{m}", color=c, lw=1.8,
                markersize=7, markerfacecolor=facecolor,
                markeredgecolor=c, markeredgewidth=1.5,
                zorder=3, label=f"$L = {L}$")

    # L = N curve (volume law — red dashed with X markers)
    j_N = len(depths)
    ax.plot(Ns, sa[:, j_N] / LN2, "x--", color="#E63946", lw=2.2,
            markersize=9, markeredgewidth=2.5,
            zorder=5, label=r"$L = N$")

    ax.set_xlabel(r"System size $N$")
    ax.set_ylabel(r"Entanglement entropy $S_{\mathrm{vN}}\,/\,\ln 2$")
    ax.set_xlim(SYSTEM_SIZES[0] - 0.5, SYSTEM_SIZES[-1] + 0.5)
    ax.set_ylim(0, SYSTEM_SIZES[-1] // 2 + 1)
    ax.set_xticks(SYSTEM_SIZES)
    ax.legend(loc="upper left", framealpha=0.9, fontsize=10)

    # ── Save ─────────────────────────────────────────────────────
    # NAME COLLISION: get_fig_ch5_entanglement_scaling_analog.py writes this same
    # path with the two-panel figure the book actually includes (Fig. 5.3). Last
    # writer wins, so run the analog script AFTER this one. See module docstring.
    # NOT fig_ch5_entanglement_scaling.pdf: that name belongs to the book's Fig. 5.3, the
    # TWO-panel figure written by get_fig_ch5_entanglement_scaling_analog.py, which reads this
    # script's cache for its panel (a). This single-panel preview is a diagnostic, and writing it
    # under the book's name silently replaced Fig. 5.3 with a figure its caption did not describe.
    outpath = OUTPUT_DIR / "fig_ch5_entanglement_scaling_digital.pdf"
    fig.savefig(outpath)
    fig.savefig(outpath.with_suffix(".png"))
    print(f"    → PDF: {outpath}")
    print(f"    → PNG: {outpath.with_suffix('.png')}")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Chapter 5: Entanglement Entropy vs System Size")
    parser.add_argument("--regenerate", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()

    print("=" * 65)
    print("  Chapter 5 — Entanglement Entropy vs System Size")
    print("  S_A(N/2) at fixed depths L = 0, 2, 4, 8, 16, N")
    print("=" * 65)
    print(f"  System sizes: {SYSTEM_SIZES}")
    print(f"  Fixed depths: {FIXED_DEPTHS}")
    print(f"  JAX: {jax.default_backend()}")
    print("=" * 65)

    if args.plot_only:
        sa_file = DATA_DIR / "entropy_vs_N_depths.npz"
        if not sa_file.exists():
            print("ERROR: data missing. Run without --plot-only first.")
            sys.exit(1)
    else:
        generate_data(regenerate=args.regenerate)

    print("\n── Figure ──")
    plot_figure()
    print(f"\nDone → {OUTPUT_DIR}")
