#!/usr/bin/env python3
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║     CHAPTER 5 — ENTANGLEMENT SPECTRUM AND MARCHENKO-PASTUR DISTRIBUTION  ║
# ║     Figure: Convergence of ξ = −ln(λ) histogram to MP law               ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
#
# This script generates a publication figure showing that the entanglement
# spectrum of Ry+CNOT quantum circuits converges to the Marchenko–Pastur
# distribution as the circuit depth L increases.
#
# ═══════════════════════════════════════════════════════════════════════════════
# PART 0: MATHEMATICAL BACKGROUND
# ═══════════════════════════════════════════════════════════════════════════════
#
# SINGLE-QUBIT PAULI MATRICES:
# ─────────────────────────────
# The Pauli matrices are the fundamental operators in quantum computing:
#
#     I = |1  0|    X = |0  1|    Y = |0 -i|    Z = |1  0|
#         |0  1|        |1  0|        |i  0|        |0 -1|
#
# Key properties:
#   • Each Pauli is Hermitian (P† = P) and unitary (P†P = I)
#   • Each Pauli squares to identity: X² = Y² = Z² = I
#   • They anticommute: XY = -YX, YZ = -ZY, ZX = -XZ
#   • Any 2×2 matrix can be written as a linear combination of {I, X, Y, Z}
#
# THE CLIFFORD GROUP C_N:
# ────────────────────────
# The Clifford group maps Pauli strings to Pauli strings under conjugation:
#
#     C_N = {U : U P U† ∈ P_N for all P ∈ P_N}
#
# Generators:
#
# ┌─────────────────────────────────────────────────────────────────────────────┐
# │ HADAMARD (H):           Swaps X ↔ Z                                       │
# │     H = (1/√2)|1  1|    H X H† = Z,  H Z H† = X,  H Y H† = -Y           │
# │              |1 -1|                                                       │
# ├─────────────────────────────────────────────────────────────────────────────┤
# │ PHASE (S):              Maps X → Y                                        │
# │     S = |1  0|          S X S† = Y,  S Z S† = Z,  S Y S† = -X            │
# │         |0  i|                                                            │
# ├─────────────────────────────────────────────────────────────────────────────┤
# │ CNOT (CX):              Entangling two-qubit gate                         │
# │     CNOT = |1 0 0 0|    CNOT (X⊗I) CNOT† = X⊗X                           │
# │            |0 1 0 0|    CNOT (I⊗X) CNOT† = I⊗X                           │
# │            |0 0 0 1|    CNOT (Z⊗I) CNOT† = Z⊗I                           │
# │            |0 0 1 0|    CNOT (I⊗Z) CNOT† = Z⊗Z                           │
# └─────────────────────────────────────────────────────────────────────────────┘
#
# ═══════════════════════════════════════════════════════════════════════════════
# PART 1: STABILIZER STATES AND FLAT ENTANGLEMENT SPECTRUM
# ═══════════════════════════════════════════════════════════════════════════════
#
# A stabilizer state |ψ⟩ is uniquely defined by N commuting Pauli stabilizers
# {g₁, ..., gₙ} such that gᵢ|ψ⟩ = +1|ψ⟩ for all i.
#
# KEY THEOREM: Any state |ψ⟩ = U|0⟩^⊗N where U is Clifford is a stabilizer
# state. Its reduced density matrix is always proportional to a PROJECTOR:
#
#     ρ_A = (1/χ) Π    where Π is rank-χ projector, χ = 2^r
#
# Therefore ALL non-zero eigenvalues are EQUAL: λᵢ = 1/χ for i = 1,...,χ
#
# ┌─────────────────────────────────────────────────────────────────────────────┐
# │ CONSEQUENCE FOR ENTANGLEMENT SPECTRUM:                                     │
# │                                                                            │
# │ The entanglement spectrum ξᵢ = −ln(pᵢ) for stabilizer states is a         │
# │ DELTA FUNCTION at ξ = r × ln(2):                                          │
# │                                                                            │
# │   ξᵢ = −ln(1/χ) = ln(χ) = r × ln(2)   for ALL i = 1,...,χ                │
# │                                                                            │
# │ This is the fingerprint of classical simulability!                         │
# └─────────────────────────────────────────────────────────────────────────────┘
#
# ═══════════════════════════════════════════════════════════════════════════════
# PART 2: Ry(θ) MAGIC INJECTION AND BREAKING STABILIZER STRUCTURE
# ═══════════════════════════════════════════════════════════════════════════════
#
# Ry(θ) = exp(−iθY/2) = [cos(θ/2)  −sin(θ/2)]
#                        [sin(θ/2)   cos(θ/2)]
#
# • θ = 0, π, 2π:     Identity or Pauli → Clifford (SRE = 0)
# • θ = π/2, 3π/2:    H-like rotation → Clifford (SRE = 0)
# • θ = π/4:          MAXIMUM magic per gate!
#
# We use θ_magic = π/4 for maximum non-Cliffordness.
#
# As circuit depth L increases:
#   L ≪ N:  Spectrum still structured, far from MP    → LOW scrambling
#   L ~ N:  Spectrum approaches MP distribution       → FULL scrambling
#
# ═══════════════════════════════════════════════════════════════════════════════
# PART 3: BIPARTITE CUT AND SCHMIDT DECOMPOSITION
# ═══════════════════════════════════════════════════════════════════════════════
#
# For N qubits split into A (first k) and B (remaining N−k):
#
#     |ψ⟩ = Σᵢ λᵢ |iₐ⟩ ⊗ |iᵦ⟩     (Schmidt decomposition)
#
# Procedure:
#   1. Reshape |ψ⟩ as matrix C of shape (m × n), m = 2^k, n = 2^{N−k}
#   2. Compute ρ_A = C · C†
#   3. Eigenvalues {pᵢ} = {λᵢ²} give the entanglement spectrum
#   4. ξᵢ = −ln(pᵢ) is the entanglement spectrum variable
#
# ═══════════════════════════════════════════════════════════════════════════════
# PART 4: MARCHENKO–PASTUR DISTRIBUTION
# ═══════════════════════════════════════════════════════════════════════════════
#
# For Haar-random states with equal bipartition m = n = 2^{N/2}:
#
#     ρ_MP(x) = (1/2πcx) · √[(x₊−x)(x−x₋)]     for x ∈ [x₋, x₊]
#
# where c = m/n = 1, x = m·λ, x₊ = (1+√c)² = 4, x₋ = (1−√c)² = 0.
#
# In the entanglement spectrum variable ξ = −ln(λ):
#
#     x = m·e^{−ξ},   ρ_ξ(ξ) = ρ_MP(x) · |dx/dξ| = ρ_MP(x) · x
#
# KEY INSIGHT: The transition from FLAT (Clifford) to MP (Haar-random) is
# driven by MAGIC INJECTION. Adding non-Clifford gates progressively breaks
# the uniformity of the stabilizer structure!
#
# ═══════════════════════════════════════════════════════════════════════════════
# PART 5: QUANTIFYING CONVERGENCE VIA KL DIVERGENCE
# ═══════════════════════════════════════════════════════════════════════════════
#
# The Kullback–Leibler divergence measures how close the empirical spectrum
# is to the MP distribution:
#
#     D_KL(P_data || P_MP) = Σ_k p_k · ln(p_k / q_k) · Δξ
#
# where p_k is the histogram density and q_k is the MP density at bin k.
#
# D_KL → 0:  circuit has scrambled to Haar-like entanglement
# D_KL ≫ 0:  spectrum still far from MP (not fully scrambled)
#
# ═══════════════════════════════════════════════════════════════════════════════
# CIRCUIT STRUCTURE
# ═══════════════════════════════════════════════════════════════════════════════
#
# Ry+CNOT brickwork layer:
#   1. Ry(π/4) on ALL N qubits (deterministic magic injection)
#   2. CNOT on even pairs: (0,1), (2,3), ...
#   3. CNOT on odd  pairs: (1,2), (3,4), ...
#
# Initial state: GHZ = (|0...0⟩ + |1...1⟩)/√2
#
# ═══════════════════════════════════════════════════════════════════════════════
# SIMULATION METHOD
# ═══════════════════════════════════════════════════════════════════════════════
#
# Matrix-free approach: statevector stored as rank-N tensor (2,2,...,2).
# Gate application via einsum contraction: O(2^N) per gate.
# F64 precision for numerical accuracy.
#
# Usage:
#     python get_fig_ch5_mp_convergence.py              # generate + plot
#     python get_fig_ch5_mp_convergence.py --plot-only   # re-plot cached
#     python get_fig_ch5_mp_convergence.py --regenerate  # force regen
#
# Data → codes_for_figures/data/ch5/
# Figs → figures/ch5/

import argparse
import os
import time
import numpy as np
from pathlib import Path

# ── JAX configuration ────────────────────────────────────────────
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("JAX_ENABLE_X64", "1")   # F64 for accuracy

import jax
import jax.numpy as jnp
from jax import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


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
KL_DEPTHS    = list(range(1, N_Q + 1))

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
    C = psi.reshape(m, n)           # coefficient matrix
    # ρ_A = C · C†  via einsum (ab = row indices, cb = contract over B)
    rho = jnp.einsum('ab,cb->ac', C, jnp.conj(C))
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

plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 13,
    "legend.fontsize": 10,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def plot_figure():
    """
    4-panel figure: each panel overlays Haar (gray) + circuit (red).
    Panels (a-d): L = 5, 10, 15, 20 for N = 20 qubits.
    """
    print("  Plotting figure...")

    ry_data   = np.load(DATA_DIR / f"spectra_ry_N{N_Q}.npz")
    haar_data = np.load(DATA_DIR / f"spectra_haar_N{N_Q}.npz")

    k = N_Q // 2
    m = 1 << k
    n = (1 << N_Q) >> k
    c = m / n
    xi_grid = np.linspace(1.0, 30.0, 2000)
    mp_curve = mp_density_xi(xi_grid, c, m)

    # Pooled Haar spectrum for background histogram
    haar_sp = haar_data["spectra"]
    xi_haar = -np.log(haar_sp[haar_sp > 1e-25])

    fig, axes = plt.subplots(1, 4, figsize=(16, 3.8), sharey=True)
    fig.subplots_adjust(wspace=0.08)

    bins = np.linspace(2, 28, 55)
    labels = ["(a)", "(b)", "(c)", "(d)"]

    for i, L in enumerate(HIST_DEPTHS):
        ax = axes[i]

        # Gray Haar histogram (background)
        ax.hist(xi_haar, bins=bins, density=True, alpha=0.35,
                color="gray", edgecolor="white", linewidth=0.4,
                zorder=1, label="Haar")

        # Red circuit histogram (foreground, opaque)
        eigs = ry_data[f"L{L}"]
        xi = -np.log(eigs[eigs > 1e-25])
        ax.hist(xi, bins=bins, density=True, alpha=0.7,
                color="#E63946", edgecolor="white", linewidth=0.4,
                zorder=2, label=f"$L={L}$")

        # MP analytical curve
        ax.plot(xi_grid, mp_curve, "k-", lw=2.5, alpha=0.85, zorder=10)

        ax.set_xlim(2, 28)
        ax.set_ylim(0, 0.37)
        ax.text(0.97, 0.95, f"{labels[i]}  $L={L}$",
                transform=ax.transAxes, fontsize=12, fontweight="bold",
                va="top", ha="right")
        ax.set_xlabel(r"$\xi = -\ln\,\lambda$")

    axes[0].set_ylabel("Density")

    outpath = OUTPUT_DIR / "fig_ch5_mp_convergence.pdf"
    fig.savefig(outpath, bbox_inches="tight")
    fig.savefig(outpath.with_suffix(".png"), bbox_inches="tight")
    print(f"    → PDF: {outpath}")
    print(f"    → PNG: {outpath.with_suffix('.png')}")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Chapter 5: Entanglement spectrum → MP convergence")
    parser.add_argument("--regenerate", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()

    print("=" * 65)
    print("  Chapter 5 — Entanglement Spectrum → Marchenko–Pastur")
    print("  Matrix-free JAX statevector simulator (F64)")
    print("=" * 65)
    print(f"  N = {N_Q} qubits, k = {N_Q//2} (equal bipartition)")
    print(f"  Histograms: L = {HIST_DEPTHS}")
    print(f"  Haar:       {N_HAAR} samples")
    print(f"  JAX:        {jax.default_backend()}")
    print("=" * 65)

    if args.plot_only:
        if not data_exists():
            print("ERROR: data missing. Run without --plot-only first.")
            exit(1)
    else:
        generate_all_data(regenerate=args.regenerate)

    print("\n── Figure ──")
    plot_figure()
    print(f"\nDone → {OUTPUT_DIR}")
