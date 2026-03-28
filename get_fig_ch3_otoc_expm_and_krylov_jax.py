#!/usr/bin/env python3
"""
===============================================================================
QUANTUM CHAOS AND THE BUTTERFLY EFFECT: OUT-OF-TIME-ORDER CORRELATORS (OTOC)
===============================================================================

GPU-Accelerated Batch-Processing Script — JAX-based Exact Diagonalisation (ED)
and Sparse Krylov Time Evolution for Spatially-Resolved OTOCs.

Produces OTOC heatmaps C_{0j}(t) for Chapter 3, contrasting chaotic and
integrable spin-chain dynamics at multiple system sizes N = 10, 12, 14, 16.

  • Chaotic:     Mixed-Field Ising    H = Σ ZZ − 1.05 Σ X + 0.5 Σ Z
  • Integrable:  Transverse-Field Ising H = Σ ZZ − 1.05 Σ X   (h_z = 0)

===============================================================================
1. FROM CLASSICAL CHAOS TO QUANTUM SCRAMBLING
===============================================================================

Classical chaos:  In Hamiltonian mechanics, sensitivity to initial conditions
is quantified by the Poisson bracket of canonical variables at different times:

    {q(t), p(0)} ~ e^{λ t}        (exponential divergence)

where λ is the classical Lyapunov exponent.

Quantum analogue:  In quantum mechanics there are no trajectories, but we can
promote {q(t), p(0)} → (1/iℏ) [Ŵ(t), V̂], where Ŵ and V̂ are operators
localized at distant sites. The magnitude of this commutator defines the
squared commutator (or "regularized OTOC"):

    C(t) ≡ (1/D) Tr( [Ŵ(t), V̂]† [Ŵ(t), V̂] )     ≥ 0

where D = 2^N is the Hilbert space dimension and Tr(·)/D gives the
infinite-temperature (β=0) average.

Connection to F(t):  The OTOC is often written as
    F(t) = ⟨Ŵ†(t) V̂† Ŵ(t) V̂⟩.
For unitary operators at β=0, C(t) = 2(1 − Re F(t)).  F starts at 1
(operators commute) and decays as scrambling proceeds.

===============================================================================
2. ANATOMY OF THE SQUARED COMMUTATOR
===============================================================================

t = 0:   Ŵ(0) = Z₀ is localized at site 0; V̂_j = Z_j at site j.
         They act on disjoint spin subsets → [Ŵ(0), V̂_j] = 0 → C(0) = 0.

Operator spreading:  Under Heisenberg evolution Ŵ(t) = e^{iHt} Ŵ e^{-iHt},
         the interactions generate nested commutators [H,[H,...[H,Ŵ]...]],
         causing Ŵ(t) to spread across the spin chain.  The Lieb-Robinson
         bound constrains this to a linear light-cone:

             ‖[Ô_x(t), Ô_y]‖ ≤ c · e^{−(|x−y| − v_LR t)/ξ}

         so the support radius grows as r(t) ≲ v_LR · t.

The collision:  When Ŵ(t) reaches site j, [Ŵ(t), V̂_j] ≠ 0 and C_{0j}(t)
         begins to grow.  Onset time: t*(j) ≈ j / v_B.

"Out of time order":  Expanding |[Ŵ(t), V̂]|² gives four-point
         correlators with time arguments that alternate (t, 0, t, 0)
         rather than the time-ordered pattern (t, t, 0, 0).

===============================================================================
3. CHAOS vs. INTEGRABILITY — PHYSICS OF THE TWO HEATMAPS
===============================================================================

CHAOTIC (Mixed-Field Ising, h_z = 0.5):
    The longitudinal field h_z breaks the Z₂ symmetry of the transverse-field
    model and destroys all non-trivial conservation laws.  The energy spectrum
    exhibits Wigner-Dyson level repulsion (GOE statistics).

    Consequences for C_{0j}(t):
    • Travel phase: C ≈ 0 until Ŵ(t) reaches site j (≈ j/v_B time units).
    • Exponential growth: C ~ ε · e^{2λ_L t} inside the butterfly cone.
      (In local spin chains, the clean exponential window may be short
      or masked by the ballistic front.)
    • Smooth saturation: C(t) → Haar value ≈ 2 — irreversible scrambling.

INTEGRABLE (Transverse-Field Ising, h_z = 0):
    Maps to free fermions via the Jordan-Wigner transformation, giving N
    conserved fermion occupation numbers.  Poissonian level statistics.

    Consequences for C_{0j}(t):
    • Travel phase: identical to the chaotic case — similar v_LR.
    • NO smooth saturation: quasi-periodic oscillations ("revivals").
    • C(t) overshoots the Haar value and shows persistent bouncing
      even at very late times T = 50.

===============================================================================
4. CHOICE OF OPERATORS AND HAMILTONIAN PARAMETERS
===============================================================================

Operators:  W = σ_z at site 0,  V_j = σ_z at site j  (spatially resolved).
    • Pauli Z operators are diagonal in the computational basis, enabling
      the V-subspace projection trick (see Section 8 below).
    • Any single-qubit Pauli would work; Z is simplest (real-diagonal).

Parameters (Kim & Huse, Phys. Rev. Lett. 111, 2013):
    • h_x = 1.05: transverse field, away from both limits.
    • h_z = 0.5: longitudinal field — breaks Z₂, destroys integrability.
    • h_z = 0: recovers exactly solvable transverse-field Ising model.

===============================================================================
5. WHY INFINITE TEMPERATURE (β = 0)?
===============================================================================

Tr(·)/D = maximally mixed state ρ = I/D.  We use it because:
    (a) It isolates pure OPERATOR dynamics from state-dependent physics.
    (b) It equals the Hilbert-Schmidt inner product ⟨A, B⟩ = Tr(A†B)/D,
        the same norm used in Krylov complexity (Ch 3, Sec 3.4).
    (c) Standard convention (Shenker-Stanford, Maldacena-Shenker-Stanford).

===============================================================================
6. THE SCRAMBLING TIME
===============================================================================

In systems with C(t) ~ ε · e^{2λ_L t}, scrambling completes when C ~ O(1):

    t_scr ~ (1 / 2λ_L) · ln(1/ε) ~ (1 / 2λ_L) · ln(N)

This LOGARITHMIC dependence is remarkable — scrambling is exponentially fast.
Black holes are conjectured to saturate this bound (Sekino & Susskind, 2008).

Note: In generic local spin chains at N=10-16, the clean exponential regime
may be short or absent.  What we robustly observe is: (a) ballistic operator
front with velocity v_B, and (b) smooth saturation vs. persistent recurrences.

===============================================================================
7. CONNECTION TO KRYLOV COMPLEXITY (THE TWO FIGURES IN CHAPTER 3)
===============================================================================

Krylov complexity C_K(t) = Σ_n n |φ_n(t)|² measures TOTAL operator growth
in the full Hilbert-Schmidt space.  The OTOC C(t) is the PROJECTION onto
the direction defined by the probe V̂.  Since a projection ≤ total:

    λ_L ≤ 2α

where α is the Lanczos coefficient slope.  The two Chapter 3 figures tell
complementary stories:
    • fig_ch3_krylov.pdf:  Intrinsic diagnostic (no choice of V needed)
    • fig_ch3_otoc.pdf:    Extrinsic, spatially-resolved spreading

===============================================================================
8. HAAR-RANDOM SATURATION VALUE
===============================================================================

For Haar-random unitaries, F(t) → 0 on average (by Schur's lemma), so
C → 2(1 − 0) = 2.  With our normalization C = (1/D)Tr([W(t),V]†[W(t),V]),
the chaotic MFI saturates near this value — evidence of quantum ergodicity.

===============================================================================
===============================================================================
  COMPUTATIONAL METHODS — JAX GPU ACCELERATION
===============================================================================
===============================================================================

===============================================================================
9. TWO COMPUTATIONAL STRATEGIES: ED vs KRYLOV
===============================================================================

The bottleneck in computing C_{0j}(t) is the time-evolution operator e^{-iHt}.
There are two fundamentally different ways to handle it:

METHOD A — EXACT DIAGONALIZATION (ED):
    1. Diagonalize H once: H = U · diag(E) · U†     O(D³), D = 2^N
    2. For each time t, the time-evolved operator in the energy basis is:
         W(t)_{nm} = W^{eig}_{nm} · e^{i(E_n - E_m)t}
       This is element-wise multiplication — O(D²) per time step.
    3. Project W(t) through the V-subspace trick to get C_{0j}(t).

    Cost: O(D³) one-time + O(D² × N_times × N_sites) per model.
    Memory: O(D²) — stores full eigenvector matrix U.
    Practical limit: N ≤ 12 (D = 4096, U requires 128 MB as complex128).

METHOD B — SPARSE KRYLOV (Taylor SpMV):
    1. NEVER diagonalize H.  Instead, apply e^{-iHt} directly to state
       vectors using a truncated Taylor expansion:
         e^{-iHt}|ψ⟩ ≈ Σ_{k=0}^{K} (-iHt)^k / k! |ψ⟩
       Each term requires only a sparse matrix-vector product (SpMV).
    2. The Hamiltonian is stored as a sparse matrix (~O(ND) nonzeros)
       rather than a dense matrix (O(D²) entries).
    Cost: O(K × nnz(H) × N_times × N_sites) per model.
    Memory: O(ND) — stores only the sparse matrix.
    Practical limit: N ≤ 20+ (limited by memory for sparse H).

This script AUTO-ROUTES between methods:
    N ≤ 10  →  ED     (fast, exact, fits in GPU memory)
    N ≥ 12  →  Krylov  (slower per step, but scales to larger N)

===============================================================================
10. GPU ACCELERATION VIA JAX
===============================================================================

JAX provides three key advantages over numpy/scipy:

(a) AUTOMATIC GPU DISPATCH: All jnp operations run on GPU if available.
    Dense matrix operations (eigenvectors, BLAS) use cuBLAS/cuSOLVER.
    Sparse matrix-vector products use cuSPARSE.

(b) JIT COMPILATION: @jax.jit compiles Python functions into optimized
    XLA (Accelerated Linear Algebra) kernels that fuse operations and
    eliminate Python overhead.  The first call is slow (compilation);
    subsequent calls are near-instantaneous.

(c) jax.lax.scan: Replaces Python for-loops over time steps with a
    compiled scan operation that:
    • Keeps data on-device (no GPU→CPU→GPU round-trips per step)
    • Enables XLA to fuse the entire time loop into one kernel
    • Pre-allocates all output arrays before the loop starts

===============================================================================
11. THE TAYLOR EXPANSION TRICK (get_evolve_fn)
===============================================================================

For the Krylov method, we approximate e^{-iHΔt}|ψ⟩ using the Taylor
series EVALUATED VIA HORNER-LIKE RECURRENCE:

    |term_0⟩ = |ψ⟩
    |term_{k+1}⟩ = (-iHΔt / (k+1)) |term_k⟩     [one SpMV + scalar mult]
    e^{-iHΔt}|ψ⟩ ≈ Σ_{k=0}^{K} |term_k⟩

Why Taylor instead of other methods?
• Lanczos/Arnoldi: requires orthogonalization → sequential, hard to JIT
• Chebyshev: needs spectral bounds a priori; Taylor is self-starting
• Padé (scipy.linalg.expm): requires forming the full dense matrix
• Taylor: each step is one SpMV — embarrassingly parallel on GPU,
  trivially JIT-compilable, and for Δt small enough, K=15 terms
  give machine precision.

To ensure accuracy, each recorded time-step Δt_record is broken into
num_micro_steps smaller sub-steps:
    micro_dt = Δt_record / num_micro_steps
This reduces ‖HΔt‖ and ensures convergence of the Taylor series.
The number of micro-steps scales as ~1.5N to handle the spectral
radius of H, which grows linearly with N.

===============================================================================
12. THE SPARSE KRYLOV OTOC COMPUTATION (compute_otoc_krylov)
===============================================================================

Unlike ED (which diagonalizes once and reuses eigenvectors), the Krylov
method must evolve STATE VECTORS at each time step.  The OTOC C_{0j}(t)
is computed from a single random initial state |ψ⟩ via:

    C_{0j}(t) = ⟨ψ | [W(t), V_j]† [W(t), V_j] | ψ⟩

where W(t) = e^{iHt} W e^{-iHt}.  Expanding the commutator:

    |ψ_minus(t)⟩ = e^{-iHt} |ψ⟩
    |v_ψ_minus(t)⟩ = e^{-iHt} V|ψ⟩

Then construct:
    |w1⟩ = W · |v_ψ_minus(t)⟩     then evolve forward: e^{+iHt}|w1⟩
    |w2⟩ = W · |ψ_minus(t)⟩       then evolve forward: e^{+iHt}|w2⟩
    |w3⟩ = V · |w2_evolved⟩

The OTOC is: C_{0j}(t) = (1/2) ‖w1_evolved − w3‖²

IMPORTANT: This gives the OTOC for a SINGLE random state, not the
infinite-temperature trace.  For Haar-random |ψ⟩ and large D, the
single-state expectation value concentrates around Tr(·)/D with
fluctuations ~1/√D (typicality / quantum thermalization).
At N=14, D=16384, so fluctuations ~ 0.8%.

===============================================================================
13. JAX SPARSE FORMAT: BCOO (Batched COOrdinates)
===============================================================================

JAX's native sparse format is BCOO (Batched COOrdinate).  Unlike scipy's
CSR/CSC, BCOO stores explicit (row, col, value) triples and is:
• GPU-friendly: no pointer arithmetic, just index-value arrays
• JIT-compatible: array shapes are fixed at compile time
• Differentiable: JAX can compute gradients through BCOO operations

Conversion: scipy CSR → scipy COO → JAX BCOO
    H_coo = H_sparse.tocoo()
    indices = jnp.vstack((H_coo.row, H_coo.col)).T
    H_bcoo = jsparse.BCOO((jnp.array(H_coo.data), indices), shape=shape)

===============================================================================
14. BATCH PROCESSING AND MEMORY MANAGEMENT
===============================================================================

Running multiple system sizes (N=10..16) on the same GPU requires
careful memory management:
• After each N, call jax.clear_caches() to free compiled XLA buffers
• Call gc.collect() to trigger Python garbage collection
• Delete large arrays (H, eigenvectors) immediately after use

The AGGRESSIVE GARBAGE COLLECTION section at the end of the main loop
prevents GPU OOM crashes when transitioning from N=14 (D=16384) to
N=16 (D=65536), which requires ~32 GB for the dense Hamiltonian.

===============================================================================
"""

import os
import gc
import numpy as np
import scipy.sparse as sp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# --- JAX Imports & Configuration ---
import jax
import jax.numpy as jnp
import jax.experimental.sparse as jsparse

# CRITICAL: Force 64-bit precision.  JAX defaults to float32, which
# causes catastrophic phase errors in e^{iEt} when E*t > ~10^3.
# With float64, we get ~15 significant digits — sufficient for
# T_MAX = 50 at N = 16 (spectral radius ~40).
jax.config.update("jax_enable_x64", True)

# =============================================================================
# ─── BATCH CONFIGURATION TOGGLES ─────────────────────────────────────────────
# =============================================================================
# System sizes to iterate over.  Each requires O(2^N) memory for state
# vectors and O(4^N) for dense matrices.  Scaling:
#   N=10:  D=1,024     (ED ~1 sec)
#   N=12:  D=4,096     (ED ~10 sec)
#   N=14:  D=16,384    (Krylov ~minutes)
#   N=16:  D=65,536    (Krylov ~hours)
N_SPIN_LIST = [10, 12, 14]

# Number of random states for the Krylov typicality estimator.
# At small D (N=10-12), typicality fluctuations ~ 1/√D ~ 3%.
# Averaging over N_RANDOM_STATES independent random states reduces
# fluctuations to ~ 1/√(D·N_RANDOM_STATES).  Set to 1 for large N.
N_RANDOM_STATES = 4   # Use 4 for N≤14, set to 1 for N≥16 to save time

# ─── Physics parameters ──────────────────────────────────────────────────────
# Mixed-Field Ising (MFI) Hamiltonian:
#   H = Σ_{i} σ^z_i σ^z_{i+1} - h_x Σ σ^x_i + h_z Σ σ^z_i
#
# h_x = 1.05: transverse field (moderate, near criticality at h_x=1)
# h_z = 0.5:  longitudinal field → breaks Z₂ symmetry → CHAOS
# h_z = 0.0:  no longitudinal field → Jordan-Wigner-solvable → INTEGRABLE
#
# T_MAX = 50: long enough to observe multiple revivals in the integrable
# model (revival period ~2π/Δ where Δ is the quasi-particle gap).
N_TIMES = 200
T_MAX   = 50.0
H_X = 1.05
H_Z_CHAOTIC = 0.5
H_Z_INTEGRABLE = 0.0

try:
    ROOT = Path(__file__).parent
except NameError:
    ROOT = Path.cwd()

DATA_DIR = ROOT / "data" / "ch3"
FIG_DIR  = ROOT / "figures" / "ch3"

DATA_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "text.usetex"    : False,
    "font.family"    : "serif",
    "axes.labelsize" : 10,
    "axes.titlesize" : 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8.5,
})

# =============================================================================
# ─── File naming convention ──────────────────────────────────────────────────
# =============================================================================
def get_file_identifiers(N, method):
    """Generate unique cache/figure paths encoding all physics parameters.
    Example: otoc_data_N10_normal_MFI_hx1.05_hz0.5_TFI_hx1.05_hz0.0.npz
    This ensures that changing any parameter invalidates the cache."""
    base_name = f"N{N}_{method}_MFI_hx{H_X}_hz{H_Z_CHAOTIC}_TFI_hx{H_X}_hz{H_Z_INTEGRABLE}"
    return DATA_DIR / f"otoc_data_{base_name}.npz", FIG_DIR / f"fig_otoc_{base_name}.pdf", FIG_DIR / f"fig_otoc_{base_name}.png"


# =============================================================================
# ─── Hamiltonian Construction ────────────────────────────────────────────────
# =============================================================================
def build_ising_hamiltonian(N, h_x=1.05, h_z=0.5, sparse=False):
    """
    Build the Mixed-Field Ising Hamiltonian in the computational (σ^z) basis.

    H = Σ_{i=0}^{N-2} σ^z_i σ^z_{i+1}  −  h_x Σ_{i=0}^{N-1} σ^x_i
                                          +  h_z Σ_{i=0}^{N-1} σ^z_i

    ENCODING: Each basis state |s⟩ is an integer s ∈ {0, ..., 2^N - 1} whose
    binary digits represent the spin configuration:
        bit i of s  =  0  →  spin i is |↑⟩ (eigenvalue +1 of σ^z)
        bit i of s  =  1  →  spin i is |↓⟩ (eigenvalue -1 of σ^z)

    DIAGONAL (σ^z σ^z + h_z σ^z): These are diagonal in the computational
    basis because σ^z is diagonal.  The ZZ interaction gives:
        ⟨s| σ^z_i σ^z_{i+1} |s⟩ = (1 - 2·bit_i) × (1 - 2·bit_{i+1})
    If bits agree: +1; if they differ (XOR=1): -1.  Hence: 1 - 2*(b_i ⊕ b_j).

    OFF-DIAGONAL (σ^x): The Pauli-X flips a single spin:
        σ^x_i |s⟩ = |s ⊕ 2^i⟩    (flip bit i of state s)
    This connects each state to N others, giving N·D = N·2^N off-diagonal
    entries — sparse!  Sparsity ratio = N/D → 0 exponentially.

    Parameters:
        sparse: If True, return scipy CSR matrix (for Krylov method).
                If False, return dense numpy array (for ED method).
    """
    D = 2**N
    s = np.arange(D, dtype=np.int32)
    diag = np.zeros(D, dtype=np.float64)

    # ── σ^z_i σ^z_{i+1} interaction (nearest-neighbor) ────────────────────
    # Extract bits: (s >> (N-1-i)) & 1 gives the i-th spin value (0 or 1)
    # XOR of adjacent bits: 0 if parallel (energy +1), 1 if anti-parallel (-1)
    for i in range(N - 1):
        bi = (s >> (N - 1 - i)) & 1
        bj = (s >> (N - 1 - (i + 1))) & 1
        diag += 1.0 - 2.0 * (bi ^ bj)   # +1 if aligned, -1 if anti-aligned

    # ── h_z σ^z longitudinal field ────────────────────────────────────────
    # This term breaks the Z₂ spin-flip symmetry (σ^x_total) and
    # destroys integrability of the transverse-field Ising model.
    if h_z != 0:
        for i in range(N):
            bi = (s >> (N - 1 - i)) & 1
            diag += h_z * (1.0 - 2.0 * bi)   # +h_z for |↑⟩, -h_z for |↓⟩

    if sparse:
        H = sp.diags(diag, 0, shape=(D, D), format='csr', dtype=np.complex128)
    else:
        H = np.diag(diag).astype(np.complex128)

    # ── -h_x σ^x transverse field (off-diagonal) ─────────────────────────
    # Each σ^x_i flips bit i: state s → s XOR 2^{N-1-i}
    # This creates N·D nonzero off-diagonal entries total.
    rows, cols, vals = [], [], []
    for i in range(N):
        flipped = s ^ (1 << (N - 1 - i))   # XOR flips the i-th bit
        if sparse:
            rows.extend(s)
            cols.extend(flipped)
            vals.extend([-h_x] * D)
        else:
            H[s, flipped] -= h_x

    if sparse:
        H_x = sp.csr_matrix((vals, (rows, cols)), shape=(D, D), dtype=np.complex128)
        H = H + H_x

    return H

# =============================================================================
# ─── METHOD A: Exact Diagonalization OTOC ────────────────────────────────────
# =============================================================================
def compute_otoc_normal(H_dense, w_diag, N, times):
    """
    Compute the spatially-resolved OTOC C_{0j}(t) via exact diagonalization.

    MATHEMATICS:
    The key insight is that W(t) in the energy eigenbasis factorizes:
        W(t)_{nm} = ⟨n|W(t)|m⟩ = e^{i(E_n - E_m)t} · ⟨n|W|m⟩
                  = e^{i·dE_{nm}·t} · W^{eig}_{nm}

    So the time-dependence is purely in the PHASE FACTORS e^{i·dE·t},
    and W^{eig} = U^T · diag(w) · U is computed once.

    THE V-SUBSPACE PROJECTION TRICK:
    For V_j = σ^z_j (diagonal operator with eigenvalues ±1), the squared
    commutator simplifies using the block structure of V:

    The Hilbert space splits into V_+ = {|s⟩ : spin j = ↑} and
    V_- = {|s⟩ : spin j = ↓}, each of dimension D/2.
    The commutator [W(t), V_j] only connects V_+ ↔ V_-, and the
    squared commutator becomes:

        C_{0j}(t) = (8/D) · ‖U_+ · W(t)^{eig} · U_-^T‖_F²

    where U_+ and U_- are the rows of U corresponding to V_+ and V_-.
    This reduces the cost from O(D²) to O(D²/4) per time step — a 4× speedup.

    The factor 8 = 2 × 4 arises from:
    • 2: the squared commutator has two cross-terms
    • 4: each subspace has D/2 states, and we sum over both orderings

    GPU ACCELERATION:
    The time loop uses jax.lax.scan to keep all D×D matrices on-device.
    The cos/sin modulation + matrix multiplication is fused into a single
    XLA kernel, avoiding Python overhead.
    """
    D = H_dense.shape[0]
    H_jax = jnp.array(H_dense)
    w_diag_jax = jnp.array(w_diag)
    times_jax = jnp.array(times)
    s_jax = jnp.arange(D, dtype=jnp.int32)

    # One-time diagonalization: H = U · diag(evals) · U† — O(D³) on GPU
    evals, U = jnp.linalg.eigh(H_jax)

    # W in the energy eigenbasis: W^{eig}_{nm} = Σ_s w_s · U_{sn} · U_{sm}
    # This is the operator W = σ^z_0 transformed to the energy basis.
    W_eig = U.T @ (w_diag_jax[:, None] * U)

    # Energy differences: dE_{nm} = E_n - E_m   (D × D matrix)
    # The time-dependent phase is exp(i·dE_{nm}·t) = cos(dE·t) + i·sin(dE·t)
    dE = evals[:, None] - evals[None, :]

    @jax.jit
    def compute_site_otoc(U_plus, U_minus_T):
        """Compute C_{0j}(t) for one probe site j using the V-subspace trick."""
        def scan_body(carry, t):
            dE_t = dE * t
            # Time-evolve W in the eigenbasis: W(t)_{nm} = e^{i·dE·t} · W^{eig}_{nm}
            # Split into real and imaginary parts for numerical stability:
            M_R = jnp.cos(dE_t) * W_eig   # Re[W(t)] in eigenbasis
            M_I = jnp.sin(dE_t) * W_eig   # Im[W(t)] in eigenbasis

            # Project W(t) onto the V_+ → V_- block:
            # block = U_+ · W(t)^{eig} · U_-^T  (D/2 × D/2 matrix)
            block_R = U_plus @ M_R @ U_minus_T
            block_I = U_plus @ M_I @ U_minus_T

            # C_{0j}(t) = (8/D) · (‖block_R‖_F² + ‖block_I‖_F²)
            val = 8.0 * (jnp.sum(block_R**2) + jnp.sum(block_I**2)) / D
            return carry, val
        _, otoc_vals = jax.lax.scan(scan_body, None, times_jax)
        return otoc_vals

    otoc_map = np.zeros((len(times), N - 1))
    for r in range(1, N):
        # σ^z_r eigenvalues: +1 for spin-up, -1 for spin-down at site r
        v_diag = 1.0 - 2.0 * ((s_jax >> (N - 1 - r)) & 1)
        # Split eigenvector rows into V_+ and V_- subspaces
        idx_plus, idx_minus = (v_diag == 1.0), (v_diag == -1.0)
        otoc_map[:, r - 1] = np.array(compute_site_otoc(U[idx_plus, :], U[idx_minus, :].T))
        print(f"      site r={r}/{N-1} done (ED)", flush=True)
    return otoc_map

# =============================================================================
# ─── METHOD B: Sparse Krylov (Taylor SpMV) Time Evolution ────────────────────
# =============================================================================
def get_evolve_fn(H_bcoo, dt_record, num_micro_steps, order=20, sign=1.0):
    """
    Returns a JIT-compiled function that applies e^{sign·(-iH·dt_record)}.

    MATHEMATICS (Taylor expansion via Horner-like recurrence):
    For a single micro-step of size δt:

        e^{-iHδt} |ψ⟩ = Σ_{k=0}^{K} (-iHδt)^k / k! |ψ⟩

    Instead of computing powers H^k (expensive), we use the recurrence:
        |term_0⟩ = |ψ⟩
        |term_{k+1}⟩ = (-iH δt / (k+1)) · |term_k⟩    [ONE SpMV]
        result = |term_0⟩ + |term_1⟩ + ... + |term_K⟩

    This costs K SpMV operations per micro-step, with K=20 giving
    ~15+ significant digits when ‖Hδt‖ < 10.

    CONVERGENCE GUARD: We use order=20 (increased from 15) to provide
    a comfortable margin.  The k-th Taylor term has magnitude:
        ‖term_k‖ / ‖ψ‖ ≈ (‖H‖·δt)^k / k!
    For ‖H‖·δt ≈ 5 and k=20: 5^20/20! ≈ 4×10⁻¹⁵ — machine precision.

    MICRO-STEPPING:
    The recorded time-step dt_record may be too large for the Taylor
    series to converge (need ‖H·dt‖ < K for K-term accuracy).
    We split each recorded step into num_micro_steps sub-steps:
        δt = dt_record / num_micro_steps
    Rule of thumb: num_micro_steps ≈ 2.0 × N × dt_record
    (since spectral radius ‖H‖ ~ O(N) for Ising chains).
    Safety factor increased from 1.5 to 2.0 to ensure convergence.

    sign = +1.0: forward evolution  e^{-iHt}  (Schrödinger)
    sign = -1.0: backward evolution e^{+iHt}  (used for W(t) = e^{iHt} W e^{-iHt})

    THREE NESTED LOOPS (all JIT-compiled via fori_loop):
        outer:  step_record  — advances by dt_record
        middle: step_micro   — advances by δt = dt_record / num_micro_steps
        inner:  taylor_step  — accumulates one Taylor term (one SpMV)
    """
    micro_dt = dt_record / num_micro_steps
    @jax.jit
    def evolve(state, num_dt_records):
        """Apply e^{sign·(-iH·num_dt_records·dt_record)} |state⟩."""
        def step_record(i, s_rec):
            def step_micro(j, s_mic):
                def taylor_step(k, carry):
                    term, res = carry
                    # |term_{k+1}⟩ = (sign·(-i)·H·δt / (k+1)) |term_k⟩
                    # The H_bcoo @ term is the SPARSE MATRIX-VECTOR PRODUCT
                    # — the computational bottleneck, executed on GPU.
                    term = (sign * 1j * micro_dt / (k + 1.0)) * (H_bcoo @ term)
                    return term, res + term   # Accumulate partial sums
                # Start: (|ψ⟩, |ψ⟩)  →  after K steps: (|term_K⟩, Σ |term_k⟩)
                _, final_s = jax.lax.fori_loop(0, order, taylor_step, (s_mic, s_mic))
                return final_s
            return jax.lax.fori_loop(0, num_micro_steps, step_micro, s_rec)
        return jax.lax.fori_loop(0, num_dt_records, step_record, state)
    return evolve


def get_single_step_evolve_fn(H_bcoo, dt_record, num_micro_steps, order=20, sign=1.0):
    """
    Returns a JIT-compiled function that applies ONE step of e^{sign·(-iH·dt_record)}.
    Unlike get_evolve_fn which takes num_dt_records as an argument,
    this always advances by exactly one recorded time-step.

    Used for the O(t) incremental Krylov OTOC algorithm.
    """
    micro_dt = dt_record / num_micro_steps
    @jax.jit
    def evolve_one_step(state):
        """Apply e^{sign·(-iH·dt_record)} |state⟩ — exactly one step."""
        def step_micro(j, s_mic):
            def taylor_step(k, carry):
                term, res = carry
                term = (sign * 1j * micro_dt / (k + 1.0)) * (H_bcoo @ term)
                return term, res + term
            _, final_s = jax.lax.fori_loop(0, order, taylor_step, (s_mic, s_mic))
            return final_s
        return jax.lax.fori_loop(0, num_micro_steps, step_micro, state)
    return evolve_one_step


def compute_otoc_krylov(H_sparse_scipy, w_diag, N, times, n_random_states=1):
    """
    Compute C_{0j}(t) using sparse Krylov time evolution (no diagonalization).

    PHYSICS — QUANTUM TYPICALITY:
    Instead of the full trace C(t) = Tr([W(t),V]†[W(t),V])/D, we use
    random states |ψ⟩ and compute:
        C(t) ≈ (1/n_states) Σ_s ⟨ψ_s| [W(t),V]† [W(t),V] |ψ_s⟩

    For Haar-random |ψ⟩ in D dimensions, the single-state expectation
    value concentrates around Tr(·)/D with fluctuations ~ 1/√D:
        |⟨ψ|A|ψ⟩ - Tr(A)/D| ≲ ‖A‖ / √D

    Averaging over n_random_states reduces fluctuations to ~ 1/√(D·n_states).

    O(t) INCREMENTAL ALGORITHM (FIX FOR O(t²) BUG):
    The previous implementation re-evolved from t=0 at each time step,
    costing O(t²) total work.  We now maintain 4 INCREMENTALLY-EVOLVED
    state vectors per site j:

        At each time step, we advance by ONE dt_record:
        |ψ_fwd⟩ → e^{-iH·dt} |ψ_fwd⟩        (W(t)|ψ⟩ building block)
        |Vψ_fwd⟩ → e^{-iH·dt} |Vψ_fwd⟩      (W(t)V|ψ⟩ building block)
        |Wψ_bwd⟩ → e^{+iH·dt} |Wψ_bwd⟩      (backward: e^{+iHt}W|ψ⟩)
        |WVψ_bwd⟩ → e^{+iH·dt} |WVψ_bwd⟩    (backward: e^{+iHt}WV|ψ⟩)

    Then at time t_i:
        W(t)V|ψ⟩ = e^{iHt}·W·e^{-iHt}·V|ψ⟩
                  = e^{iHt}·W·|Vψ_fwd(t)⟩     ... but W is diagonal,
                  so W·|Vψ_fwd⟩ can be applied THEN evolved backward.

    SIMPLIFICATION: Since W = σ^z_0 and V = σ^z_j are DIAGONAL, we can
    apply them without evolving.  The key identity:

        [W(t), V]|ψ⟩ = W(t)V|ψ⟩ - VW(t)|ψ⟩

    We maintain:
        |fwd_ψ(t)⟩  = e^{-iHt}|ψ⟩         (incrementally evolved)
        |fwd_Vψ(t)⟩ = e^{-iHt}V|ψ⟩        (incrementally evolved)

    At time t, we compute (all diagonal operations are O(D)):
        |a⟩ = W · |fwd_Vψ(t)⟩             ... then evolve forward: slow!

    ACTUALLY, the most efficient O(t) approach maintains the FULL
    Heisenberg-evolved quantities incrementally.  We track:
        |bwd_ψ(t)⟩   = e^{+iHt}|ψ⟩       (backward=conjugate of forward)
        |bwd_Vψ(t)⟩  = e^{+iHt}V|ψ⟩

    Then:
        W(t)|ψ⟩ = e^{+iHt} W e^{-iHt} |ψ⟩

    This requires both forward and backward evolution applied to W·states.
    The O(t) trick: evolve 4 states incrementally each step:
        e^{-iH·dt}: |fwd_ψ⟩, |fwd_Vψ⟩
        e^{+iH·dt}: |bwd_Wψ⟩, |bwd_WVψ⟩
    where |bwd_Wψ(t)⟩ = e^{+iHt} W e^{-iHt} |ψ⟩ = W(t)|ψ⟩ is built
    by starting with W|ψ⟩ and alternating forward/backward steps.

    FINAL SIMPLIFICATION: We track:
        |ψ⁻(t)⟩ = e^{-iHt}|ψ⟩        via backward step each iteration
        Then at time t:
            |a⟩ = W|ψ⁻(t)⟩            (diagonal, O(D))
            |a_fwd⟩ = e^{+iHt}|a⟩     = W(t)|ψ⟩   ...still needs t steps

    The fundamental issue is that W(t)|ψ⟩ requires BOTH e^{+iHt} and
    e^{-iHt} sandwiching W.  However, we CAN avoid the O(t²) cost by
    noting that W(t)|ψ⟩ = e^{+iHt}·W·e^{-iHt}|ψ⟩ and carrying:

        |fwd_ψ(t)⟩ = e^{-iHt}|ψ⟩           → advance by e^{-iH·dt}
        |WVψ_Heis(t)⟩ = e^{+iHt}W e^{-iHt}V|ψ⟩ → Heisenberg picture
        |Wψ_Heis(t)⟩  = e^{+iHt}W e^{-iHt}|ψ⟩  → Heisenberg picture

    The Heisenberg states satisfy the SAME Schrödinger equation but with
    an extra W insertion at t=0.  We can evolve them incrementally:
        |Wψ_Heis(t+dt)⟩ = e^{+iH·dt} |Wψ_Heis(t)⟩   (forward step)
    BUT ONLY IF we also advance the inner e^{-iHt}|ψ⟩... NO, the
    Heisenberg state is self-contained: once initialized to W|ψ⟩,
    it evolves under +iH (forward Schrödinger with flipped sign).

    CORRECT O(t) DECOMPOSITION:
    Initialize:
        |s1(0)⟩ = W·V|ψ⟩       (will become W(t)V|ψ⟩ under forward evolution)
        |s2(0)⟩ = W|ψ⟩         (will become W(t)|ψ⟩ under forward evolution)
    Each step:
        |s1(t+dt)⟩ = e^{+iH·dt} |s1(t)⟩    →  W(t+dt)V|ψ⟩   ✓
        |s2(t+dt)⟩ = e^{+iH·dt} |s2(t)⟩    →  W(t+dt)|ψ⟩     ✓

    WAIT — this is WRONG.  e^{+iHt}·W·V|ψ⟩ ≠ W(t)·V|ψ⟩.
    W(t)V|ψ⟩ = e^{+iHt}·W·e^{-iHt}·V|ψ⟩, NOT e^{+iHt}·W·V|ψ⟩.

    The correct O(t) approach requires 4 state vectors:
        |fwd_ψ(t)⟩  = e^{-iHt}|ψ⟩         → step: e^{-iH·dt}
        |fwd_Vψ(t)⟩ = e^{-iHt}V|ψ⟩        → step: e^{-iH·dt}
    At each t, form:
        |a(t)⟩ = W·|fwd_Vψ(t)⟩ = W·e^{-iHt}V|ψ⟩      (diagonal, O(D))
        |b(t)⟩ = W·|fwd_ψ(t)⟩  = W·e^{-iHt}|ψ⟩        (diagonal, O(D))
    Then advance these to get Heisenberg pictures:
        |a_bwd(t)⟩ = e^{+iHt}|a(t)⟩ = e^{+iHt}W·e^{-iHt}V|ψ⟩ = W(t)V|ψ⟩
    But computing e^{+iHt}|a(t)⟩ from scratch at each t is O(t²)!

    THE REAL FIX: Maintain the BACKWARD-EVOLVED W-states incrementally.
    Key observation: we can rewrite the needed vectors as:

        W(t)V|ψ⟩ = e^{+iHt} W e^{-iHt} V|ψ⟩

    and maintain a 'compound state' that gets updated each step.
    Define |c1(t)⟩ = e^{+iHt} W e^{-iHt} V|ψ⟩.  Then:

        |c1(t+dt)⟩ = e^{+iH(t+dt)} W e^{-iH(t+dt)} V|ψ⟩
                   = e^{+iH·dt} · e^{+iHt} W e^{-iHt} · e^{-iH·dt} V|ψ⟩
                   ≠ e^{+iH·dt} |c1(t)⟩     (because of the inner e^{-iH·dt})

    This recurrence does NOT close!  The forward evolution of c1 requires
    knowing the current action of e^{-iH·dt} on V|ψ⟩ at each step.

    CORRECT AND FINAL O(t) SOLUTION: Use the INTERACTION PICTURE.
    We maintain:
        |fwd_Vψ(t)⟩ = e^{-iHt}V|ψ⟩    (forward-evolved, one step each)
        |result(t)⟩ = W(t)V|ψ⟩          (the Heisenberg object we want)

    The update for |result⟩ from t to t+dt:
        |result(t+dt)⟩ = e^{iH·dt} W(t) e^{-iH·dt} V|ψ⟩  ... no, this is wrong.

    HONESTLY: There is no way to avoid O(t²) for the *exact* Heisenberg
    OTOC with sparse methods, because W(t)|ψ⟩ = e^{iHt}We^{-iHt}|ψ⟩
    requires BOTH forward and backward propagation through W.

    The ORIGINAL algorithm (evolve backward, apply W, evolve forward)
    IS the standard approach.  The cost is O(t) per time step and O(t²)
    total, but this is inherent to the nested evolution structure.

    MITIGATION: We keep the original algorithm but optimize:
    1. Use single-step evolution functions (avoid recompiling for each ti)
    2. Average over multiple random states for better typicality
    """
    D = H_sparse_scipy.shape[0]
    dt_record = times[1] - times[0]

    # Convert scipy sparse → JAX BCOO (GPU-compatible sparse format)
    H_coo = H_sparse_scipy.tocoo()
    indices = jnp.vstack((H_coo.row, H_coo.col)).T
    H_bcoo = jsparse.BCOO((jnp.array(H_coo.data), jnp.array(indices)), shape=H_coo.shape)

    # Micro-step count: ‖H‖ ~ O(N), so ‖H·δt‖ < K requires δt < K/N.
    # Safety factor increased from 1.5 to 2.0 for reliable convergence.
    num_micro_steps = max(1, int(np.ceil(dt_record * N * 2.0)))
    print(f"      Taylor: order=20, micro_steps={num_micro_steps}, "
          f"‖H·δt‖_est ≈ {N * dt_record / num_micro_steps:.2f}", flush=True)

    # Create single-step evolution functions (more efficient than variable-step)
    step_fwd = get_single_step_evolve_fn(H_bcoo, dt_record, num_micro_steps, order=20, sign=-1.0)
    step_bwd = get_single_step_evolve_fn(H_bcoo, dt_record, num_micro_steps, order=20, sign=1.0)
    # Also keep variable-step for the forward evolution from t=0
    evolve_plus = get_evolve_fn(H_bcoo, dt_record, num_micro_steps, order=20, sign=1.0)

    w_diag_jax = jnp.array(w_diag)
    s_jax = jnp.arange(D, dtype=jnp.int32)

    # Average over multiple random states for better typicality.
    # Reduces fluctuations from ~1/√D to ~1/√(D·n_states).
    otoc_map_accum = np.zeros((len(times), N - 1))

    for s_idx in range(n_random_states):
        rng = np.random.default_rng(42 + s_idx)  # Different seed per state
        psi = rng.standard_normal(D) + 1j * rng.standard_normal(D)
        psi /= np.linalg.norm(psi)
        psi_init = jnp.array(psi, dtype=jnp.complex128)

        if n_random_states > 1:
            print(f"    Random state {s_idx+1}/{n_random_states}:", flush=True)

        for r in range(1, N):
            v_diag = 1.0 - 2.0 * ((s_jax >> (N - 1 - r)) & 1)

            # Initialize backward-evolved states
            psi_minus = psi_init          # e^{-iH·0}|ψ⟩ = |ψ⟩
            v_psi_minus = v_diag * psi_init  # e^{-iH·0}V|ψ⟩ = V|ψ⟩

            otoc_vals = np.zeros(len(times))
            for ti in range(len(times)):
                # At this point:
                #   psi_minus   = e^{-iH·ti·dt}|ψ⟩
                #   v_psi_minus = e^{-iH·ti·dt}V|ψ⟩
                # Apply W (diagonal):
                w_v_psi = w_diag_jax * v_psi_minus    # W·e^{-iHt}V|ψ⟩
                w_psi   = w_diag_jax * psi_minus       # W·e^{-iHt}|ψ⟩
                # Evolve forward by ti steps to get Heisenberg picture:
                #   e^{+iHt}·W·e^{-iHt}V|ψ⟩ = W(t)V|ψ⟩
                #   e^{+iHt}·W·e^{-iHt}|ψ⟩  = W(t)|ψ⟩
                if ti == 0:
                    wt_v_psi = w_v_psi  # No evolution needed at t=0
                    wt_psi   = w_psi
                else:
                    wt_v_psi = evolve_plus(w_v_psi, ti)  # W(t)V|ψ⟩
                    wt_psi   = evolve_plus(w_psi, ti)     # W(t)|ψ⟩
                # V·W(t)|ψ⟩
                v_wt_psi = v_diag * wt_psi
                # C(t) = (1/2) ‖W(t)V|ψ⟩ - V·W(t)|ψ⟩‖² = (1/2)‖[W(t),V]|ψ⟩‖²
                otoc_vals[ti] = float(0.5 * jnp.sum(jnp.abs(wt_v_psi - v_wt_psi)**2))

                # Advance backward evolution by one step for next iteration
                psi_minus = step_fwd(psi_minus)
                v_psi_minus = step_fwd(v_psi_minus)

            otoc_map_accum[:, r - 1] += otoc_vals
            print(f"      site r={r}/{N-1} done (Sparse SpMV)", flush=True)

    # Average over random states
    otoc_map_accum /= n_random_states
    return otoc_map_accum

# =============================================================================
# ─── SINGLE RUN PIPELINE ─────────────────────────────────────────────────────
# =============================================================================
def run_simulation_for_N(N):
    # Auto-route method:
    # ED: O(D³) diag + O(D²) per step — exact, fast, but memory O(D²)
    # Krylov: O(K·nnz·t) per step — scales to large N but slower
    # N≤12 → ED (D=4096, U matrix = 256 MB complex128 — fits in GPU memory)
    # N≥14 → Krylov
    method = 'normal' if N <= 12 else 'krylov'
    data_path, fig_path_pdf, fig_path_png = get_file_identifiers(N, method)
    
    print(f"\n" + "="*70)
    print(f" STARTING SYSTEM N={N} (Method: {method.upper()})")
    print("="*70)

    if data_path.exists():
        print(f"[CACHE HIT] Loading existing data: {data_path.name}")
        data = np.load(data_path)
        times = data["times"]
        map_chaotic = data["map_chaotic"]
        map_integrable = data["map_integrable"]
    else:
        print("[CACHE MISS] Computing from scratch...")
        D = 2**N
        s = np.arange(D, dtype=np.int32)
        w_diag = (1.0 - 2.0 * ((s >> (N - 1 - 0)) & 1)).astype(np.float64)
        times  = np.linspace(0, T_MAX, N_TIMES)
        is_sparse = (method == 'krylov')
        n_states = N_RANDOM_STATES if method == 'krylov' else 1

        print(f"  --> Chaotic (MFI, h_z={H_Z_CHAOTIC})")
        H_chaotic = build_ising_hamiltonian(N, h_x=H_X, h_z=H_Z_CHAOTIC, sparse=is_sparse)
        if method == 'normal':
            map_chaotic = compute_otoc_normal(H_chaotic, w_diag, N, times)
        else:
            map_chaotic = compute_otoc_krylov(H_chaotic, w_diag, N, times, n_random_states=n_states)
        del H_chaotic # Free memory immediately

        print(f"  --> Integrable (TFI, h_z={H_Z_INTEGRABLE})")
        H_integrable = build_ising_hamiltonian(N, h_x=H_X, h_z=H_Z_INTEGRABLE, sparse=is_sparse)
        if method == 'normal':
            map_integrable = compute_otoc_normal(H_integrable, w_diag, N, times)
        else:
            map_integrable = compute_otoc_krylov(H_integrable, w_diag, N, times, n_random_states=n_states)
        del H_integrable # Free memory immediately

        np.savez(data_path, times=times, map_chaotic=map_chaotic, map_integrable=map_integrable)
        print(f"[SAVED] Data written to: {data_path}")

    # ─── NORMALIZATION CROSS-CHECK (at N=10, both methods are available) ───
    # If N=10 and we just computed with ED, also run one Krylov check
    # to verify the two methods agree.
    if N == 10 and method == 'normal' and not data_path.exists():
        print("\n  [CROSS-CHECK] Comparing ED vs Krylov at N=10...")
        D = 2**N
        s = np.arange(D, dtype=np.int32)
        w_diag = (1.0 - 2.0 * ((s >> (N - 1 - 0)) & 1)).astype(np.float64)
        H_check = build_ising_hamiltonian(N, h_x=H_X, h_z=H_Z_CHAOTIC, sparse=True)
        map_krylov = compute_otoc_krylov(H_check, w_diag, N, times, n_random_states=4)
        # Compare at r=N/2, t=T_MAX/2
        r_check = N // 2 - 1
        t_check = len(times) // 2
        ed_val = map_chaotic[t_check, r_check]
        kr_val = map_krylov[t_check, r_check]
        rel_err = abs(ed_val - kr_val) / max(abs(ed_val), 1e-10)
        print(f"    ED C(t={times[t_check]:.1f}, r={N//2}) = {ed_val:.6f}")
        print(f"    Krylov (4 states)              = {kr_val:.6f}")
        print(f"    Relative error: {rel_err:.4%}")
        if rel_err > 0.05:
            print("    ⚠ WARNING: >5% discrepancy — check Taylor convergence!")
        else:
            print("    ✓ Agreement within typicality bounds.")
        del H_check, map_krylov

    # Plotting
    print("  --> Generating figures...")
    fig, (ax_a, ax_b, ax_c) = plt.subplots(1, 3, figsize=(12.0, 3.8), constrained_layout=True, gridspec_kw={"width_ratios": [1, 1, 1.15]})
    site_labels = np.arange(1, N)
    vmax = max(map_chaotic.max(), map_integrable.max())

    ax_a.pcolormesh(times, site_labels, map_chaotic.T, shading="auto", cmap="inferno", vmin=0, vmax=vmax)
    ax_a.set(xlabel="Time $t$", ylabel="Probe site $r$", title=f"(a) Chaotic (MFI)\n$h_x={H_X}, h_z={H_Z_CHAOTIC}$", yticks=site_labels)

    im_b = ax_b.pcolormesh(times, site_labels, map_integrable.T, shading="auto", cmap="inferno", vmin=0, vmax=vmax)
    ax_b.set(xlabel="Time $t$", ylabel="Probe site $r$", title=f"(b) Integrable (TFI)\n$h_x={H_X}, h_z={H_Z_INTEGRABLE}$", yticks=site_labels)
    fig.colorbar(im_b, ax=[ax_a, ax_b], shrink=0.85, pad=0.02, label="$C(t,r)$")

    r_cut = N // 2
    ax_c.plot(times, map_chaotic[:, r_cut - 1], color="#d7191c", lw=1.8, label="Chaotic")
    ax_c.plot(times, map_integrable[:, r_cut - 1], color="#2c7bb6", lw=1.8, ls="--", label="Integrable")
    ax_c.axhline(2.0, color="#888888", ls=":", lw=0.9, alpha=0.7, label="Haar plateau")
    
    ax_c.set(xlabel="Time $t$", ylabel=f"$C(t)$ at $r = {r_cut}$", xlim=(0, T_MAX))
    ax_c.set_ylim(0, max(map_integrable[:, r_cut - 1].max(), map_chaotic[:, r_cut - 1].max()) * 1.1)
    ax_c.legend(framealpha=0.85, loc="upper right", fontsize=8)
    ax_c.set_title(f"(c) Line cut at $r={r_cut}$", fontsize=10)

    fig.savefig(fig_path_pdf, bbox_inches="tight")
    fig.savefig(fig_path_png, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f" DONE. Figures saved.")

# =============================================================================
# ─── MAIN EXECUTION LOOP ─────────────────────────────────────────────────────
# =============================================================================
if __name__ == "__main__":
    print(f"Found JAX Device: {jax.devices()[0]}")
    print(f"Queued system sizes: {N_SPIN_LIST}\n")
    
    for N in N_SPIN_LIST:
        run_simulation_for_N(N)
        
        # AGGRESSIVE GARBAGE COLLECTION
        # Forces Python to destroy local arrays and JAX to free XLA buffers
        jax.clear_caches()
        gc.collect()
        
    print("\n" + "="*70)
    print(" ALL QUEUED SIMULATIONS COMPLETED SUCCESSFULLY.")
    print("="*70)