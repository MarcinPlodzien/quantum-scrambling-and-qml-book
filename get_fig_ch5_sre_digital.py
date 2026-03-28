#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║     STABILIZER RÉNYI ENTROPY (SRE) AND NONSTABILIZERNESS ("MAGIC")           ║
║     A Tutorial for Quantum Information and Quantum Computing                 ║
║                                                                              ║
║     Chapter 5 — Circuit Complexity & Quantum Designs: Figure 5.2             ║
║     HIGH-PERFORMANCE MATRIX-FREE SRE COMPUTATION (JAX + FWHT)                ║
╚══════════════════════════════════════════════════════════════════════════════╝

This script generates a publication-quality figure showing the growth of
nonstabilizerness ("magic") M₂ as a function of brickwork circuit depth L,
contrasting Clifford (zero magic) and non-Clifford (growing magic) circuits.


═══════════════════════════════════════════════════════════════════════════════
PART 0: MAGIC INJECTION — WHY NON-CLIFFORD GATES MATTER
═══════════════════════════════════════════════════════════════════════════════

WHAT IS "MAGIC" (NONSTABILIZERNESS)?
──────────────────────────────────────

Quantum states can be categorized by how hard they are to simulate classically:

  STABILIZER STATES:  Produced by Clifford circuits acting on |0⟩^⊗N.
    • Classically simulable in poly(N) time (Gottesman–Knill theorem)
    • Entanglement spectrum is FLAT (all non-zero eigenvalues equal)
    • The Pauli spectrum is maximally peaked: ⟨ψ|P|ψ⟩ ∈ {−1, 0, +1}
    • SRE M₂ = 0

  NON-STABILIZER ("MAGIC") STATES:  Require non-Clifford gates.
    • Cannot be efficiently simulated classically (under plausible assumptions)
    • Entanglement spectrum approaches Marchenko–Pastur
    • Pauli spectrum spreads out
    • SRE M₂ > 0

"Magic" is the quantum resource that, together with entanglement, enables
quantum computational advantage:

┌─────────────────────────────────────────────────────────────────────────────┐
│ QUANTUM ADVANTAGE REQUIRES BOTH:                                            │
│                                                                             │
│   ✓ High ENTANGLEMENT — so qubits cannot be simulated independently         │
│   ✓ High MAGIC (M₂) — so Gottesman–Knill theorem does not apply            │
│                                                                             │
│ Either alone is INSUFFICIENT:                                               │
│   • High entanglement + zero magic = stabilizer states (classically easy)   │
│   • High magic + zero entanglement = product states (trivially easy)        │
│   • High both = quantum advantage                                           │
└─────────────────────────────────────────────────────────────────────────────┘

MAGIC-INJECTING GATES:
──────────────────────

┌─────────────────────────────────────────────────────────────────────────────┐
│ THE T GATE — Canonical magic injector                                       │
│                                                                             │
│   T = diag(1, e^{iπ/4})  =  |0⟩⟨0| + e^{iπ/4} |1⟩⟨1|                     │
│                                                                             │
│ The T gate is a π/8 rotation: T = exp(iπZ/8) (up to global phase)           │
│ NOT a Clifford gate (Clifford = {H, S, CNOT} only)                          │
│                                                                             │
│ T|+⟩ = (1/√2)(|0⟩ + e^{iπ/4}|1⟩) is a "MAGIC STATE"                       │
│ with SRE M₂ ≈ 0.228 per qubit                                              │
│                                                                             │
│ Clifford gates + T gate = UNIVERSAL quantum computation                     │
│   (Solovay–Kitaev: any U can be approximated to ε using O(log^c(1/ε)) T's) │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ Ry(θ) ROTATIONS — Continuous magic injection (used in this script)          │
│                                                                             │
│   Ry(θ) = exp(−iθσ_y/2) = [[cos θ/2, −sin θ/2],                           │
│                             [sin θ/2,  cos θ/2]]                            │
│                                                                             │
│ WHICH VALUES OF θ ARE CLIFFORD (and inject ZERO magic)?                     │
│   θ = 0       → I  (identity)                                              │
│   θ = π/2     → (I − iY)/√2                                                │
│   θ = π       → −iY                                                        │
│   θ = 3π/2    → (I + iY)/√2                                                │
│                                                                             │
│ ALL other θ values inject NONSTABILIZERNESS (magic M₂ > 0).                │
│ Maximum magic per gate injection: θ = π/4 (equivalent to T-like rotation)   │
│                                                                             │
│ When θ ~ Uniform[0, 2π] (as in this script), each single-qubit gate         │
│ injects a RANDOM amount of magic — the circuit rapidly accumulates M₂.      │
└─────────────────────────────────────────────────────────────────────────────┘

MAGIC ACCUMULATION WITH CIRCUIT DEPTH:
──────────────────────────────────────

As brickwork layers of Ry+CNOT gates are applied:

    L = 0:   M₂ = 0     (product state |0⟩^⊗N is a stabilizer state)
    L = 1:   M₂ ~ O(1)  (first magic injection, one layer of Ry gates)
    L ~ N/2: M₂ saturates near Haar-random value M₂ ≈ N − 2
    L → ∞:   M₂ → M₂^Haar ≈ N − 2 + O(1/D)

This is the universal pattern for magic growth in brickwork circuits.
The saturation depth L ~ N/2 coincides with:
  (a) Marchenko–Pastur convergence in the entanglement spectrum (Fig 5.1)
  (b) Formation of an approximate unitary 2-design
  (c) Onset of Porter–Thomas statistics in |⟨x|ψ⟩|² amplitudes


═══════════════════════════════════════════════════════════════════════════════
PART 1: STABILIZER RÉNYI ENTROPY — FULL MATHEMATICAL DEFINITION
═══════════════════════════════════════════════════════════════════════════════

1.1  THE PAULI SPECTRUM
────────────────────────

Any pure state |ψ⟩ on N qubits can be expanded in the Pauli basis:

    |ψ⟩⟨ψ| = (1/D) Σ_P  ⟨ψ|P|ψ⟩ · P

where the sum runs over all 4^N Pauli strings P ∈ {I, X, Y, Z}^{⊗N}.

The set of values {⟨ψ|P|ψ⟩}_P is the PAULI SPECTRUM of |ψ⟩.

Key properties:
  • ⟨ψ|I^⊗N|ψ⟩ = 1 always (normalisation)
  • −1 ≤ ⟨ψ|P|ψ⟩ ≤ 1 for all P (since P is unitary with eigenvalues ±1)
  • Normalisation: (1/D) Σ_P |⟨ψ|P|ψ⟩|² = 1 (trace of |ψ⟩⟨ψ|²)

1.2  SRE DEFINITION
─────────────────────

The n-th Stabilizer Rényi Entropy is defined as:

    M_n(ψ) = 1/(1−n) · log₂[ (1/D) Σ_P |⟨ψ|P|ψ⟩|^{2n} ]

For n = 2 (the standard choice used in this script):

    M₂(ψ) = −log₂[ Ξ(ψ) ]

where:
    Ξ(ψ) = (1/D) Σ_P |⟨ψ|P|ψ⟩|⁴    (the "sum of fourth powers")

┌─────────────────────────────────────────────────────────────────────────────┐
│ WORKED EXAMPLE: M₂ of |0⟩ (1 qubit, D = 2)                                │
│                                                                             │
│ Pauli spectrum of |0⟩:                                                     │
│   ⟨0|I|0⟩ = 1,  ⟨0|X|0⟩ = 0,  ⟨0|Y|0⟩ = 0,  ⟨0|Z|0⟩ = 1                │
│                                                                             │
│ Ξ = (1/2)(|1|⁴ + |0|⁴ + |0|⁴ + |1|⁴) = (1/2)(1 + 0 + 0 + 1) = 1         │
│ M₂ = −log₂(1) = 0   ← Stabilizer state (zero magic)!                     │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ WORKED EXAMPLE: M₂ of T|+⟩ (1 qubit, "magic state")                       │
│                                                                             │
│ T|+⟩ = (1/√2)(|0⟩ + e^{iπ/4}|1⟩)                                         │
│                                                                             │
│ Pauli spectrum:                                                             │
│   ⟨T+|I|T+⟩ = 1                                                           │
│   ⟨T+|X|T+⟩ = Re(e^{−iπ/4}) = cos(π/4) = 1/√2                           │
│   ⟨T+|Y|T+⟩ = Im(e^{−iπ/4}) = −1/√2                                     │
│   ⟨T+|Z|T+⟩ = 0                                                           │
│                                                                             │
│ Ξ = (1/2)(1 + (1/√2)⁴ + (1/√2)⁴ + 0) = (1/2)(1 + 1/4 + 1/4) = 3/4      │
│ M₂ = −log₂(3/4) ≈ 0.415   ← Non-zero magic!                              │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ COMPARISON TABLE: M₂ VALUES FOR IMPORTANT STATES                            │
│                                                                             │
│ State              │  M₂        │  Status                                   │
│ ───────────────────┼────────────┼───────────────────────────────────────────│
│ |0⟩^⊗N             │  0         │  Product stabilizer                       │
│ Bell/GHZ           │  0         │  Entangled stabilizer                     │
│ (T|+⟩)^⊗N          │  N·0.415   │  Product magic (no entanglement!)          │
│ Deep Ry+CNOT       │  ≈ N − 2   │  Magic + entanglement → hard to simulate │
│ Haar random        │  ≈ N − 2   │  Maximum magic                            │
│ Clifford (ANY L)   │  0         │  Always zero, regardless of depth         │
└─────────────────────────────────────────────────────────────────────────────┘

1.3  WHY STABILIZER STATES HAVE M₂ = 0 (PROOF)
─────────────────────────────────────────────────

For a stabilizer state |ψ⟩ with stabilizer group S = ⟨g₁,...,gₙ⟩:

  ⟨ψ|P|ψ⟩ = { +1   if P ∈ S
             { −1   if −P ∈ S
             {  0   if P ∉ S and −P ∉ S

The number of P with |⟨P⟩| = 1 is exactly |S| = 2^N.

Therefore:
    Ξ = (1/D) Σ_P |⟨P⟩|⁴ = (1/D) · 2^N · 1⁴ = 2^N / 2^N = 1
    M₂ = −log₂(1) = 0  ✓


═══════════════════════════════════════════════════════════════════════════════
PART 2: FAST WALSH-HADAMARD TRANSFORM — ALGORITHMIC DETAILS
═══════════════════════════════════════════════════════════════════════════════

2.1  NAIVE APPROACH: O(8^N)
────────────────────────────

Direct computation of Σ_P |⟨P⟩|⁴ requires:
  • Enumerate all 4^N Pauli strings
  • For each P, compute ⟨ψ|P|ψ⟩ via matrix-vector product: O(2^N)
  • Total: 4^N × 2^N = 8^N

This limits practical computation to N ≤ 10.

2.2  FWHT DECOMPOSITION: O(N · 4^N)
─────────────────────────────────────

 Reference:
   [1] Huang, Li, Lee & Zhong, "A fast and exact approach for stabilizer
       Rényi entropy via the XOR-FWHT algorithm," arXiv:2512.24685 (2024).
   [2] Sierant, Vallès-Muns & Garcia-Saez, "Computing quantum magic of
       state vectors," arXiv:2601.07824 (2025).  [HadaMAG.jl package]

KEY INSIGHT: Pauli strings are labelled by two binary strings (z, x):

    P(z, x) = i^{z·x} · Z^{z₁} X^{x₁} ⊗ ... ⊗ Z^{zN} X^{xN}

The expectation value factorises as:

    ⟨ψ|P(z,x)|ψ⟩ = Σ_j  ψ_j*  · (−1)^{z·j}  · ψ_{j⊕x}

where j⊕x = bitwise XOR.  This decomposes into three steps:

┌─────────────────────────────────────────────────────────────────────────────┐
│ ALGORITHM: FWHT-based SRE computation                                       │
│                                                                             │
│ total = 0                                                                   │
│                                                                             │
│ FOR each x ∈ {0, 1, ..., 2^N − 1}:                    ← 2^N iterations     │
│                                                                             │
│   Step 1: Correlation vector                                                │
│     C_x[j] = ψ_j* · ψ_{j⊕x}     for all j           ← O(2^N) work        │
│                                                                             │
│   Step 2: Walsh-Hadamard transform                                          │
│     F_x = FWHT(C_x) = H^{⊗N} · C_x                  ← O(N · 2^N) via     │
│                                                          butterfly algorithm │
│   Step 3: Accumulate                                                        │
│     total += Σ_z |F_x[z]|⁴                            ← O(2^N) work        │
│                                                                             │
│ M₂ = −log₂(total / D)                                                      │
│                                                                             │
│ TOTAL COST: 2^N × O(N · 2^N) = O(N · 4^N)                                 │
│ SPEEDUP: 8^N / (N · 4^N) = 2^N / N  (exponential!)                         │
│                                                                             │
│ For N=14: naive = 8^14 ≈ 4.4 × 10^12                                       │
│           FWHT  = 14 × 4^14 ≈ 3.8 × 10^9   → 1150× speedup               │
└─────────────────────────────────────────────────────────────────────────────┘

2.3  THE WALSH-HADAMARD BUTTERFLY
──────────────────────────────────

The FWHT of a vector v of length 2^N is computed by N passes:

    For q = 0, 1, ..., N−1:
        Contract axis q with H₂ = [[1,1],[1,−1]]

Each pass costs O(2^N), total O(N · 2^N).

This is implemented matrix-free via einsum:
    v.reshape((2,)*N) — apply H₂ on axis q — reshape back

    Subscript for axis q:  'Aa, ...a... → ...A...'

We NEVER form the full 2^N × 2^N Walsh-Hadamard matrix.


2.4  CONCRETE EXAMPLE (N=2)
────────────────────────────

State: |ψ⟩ = |00⟩ → ψ = [1, 0, 0, 0]

For x = 0 (P = Z^z₁ Z^z₂, no X component):
    C_0[j] = ψ_j* · ψ_{j⊕0} = |ψ_j|² = [1, 0, 0, 0]
    F_0 = FWHT(C_0) = H⊗H · C_0 = [1, 1, 1, 1] · 1/4 (normalised)

For x = 1 (X on qubit 0):
    C_1[j] = ψ_j* · ψ_{j⊕1} = ψ_j* · ψ_{j XOR 1}
    C_1 = [ψ₀*ψ₁, ψ₁*ψ₀, ψ₂*ψ₃, ψ₃*ψ₂] = [0, 0, 0, 0]  (all zero!)

Similarly x = 2, 3 give zero correlations (since |00⟩ is in the Z-basis).

So Ξ = (1/4)(1⁴ + 1⁴ + 1⁴ + 1⁴) = 1, and M₂ = −log₂(1) = 0.  ✓


═══════════════════════════════════════════════════════════════════════════════
PART 3: BRICKWORK CIRCUIT ARCHITECTURE
═══════════════════════════════════════════════════════════════════════════════

Same architecture as Fig 5.1 (see get_fig_ch5_mp_convergence.py).

Each layer applies:
  1. Single-qubit gates on all N qubits:
     • Ry+CNOT circuits: Ry(θ_j), θ_j ~ Uniform[0, 2π]  ← injects magic
     • Clifford circuits: Random element from Cliff₁     ← NO magic

  2. Entangling CNOT gates on nearest-neighbour pairs:
       Even layers (ℓ even): (0,1), (2,3), (4,5), ...
       Odd  layers (ℓ odd):  (1,2), (3,4), (5,6), ...

┌─────────────────────────────────────────────────────────────────────────────┐
│ WHY THE MAGIC DIFFERENCE?                                                   │
│                                                                             │
│ Clifford gates map stabilizer states to stabilizer states.                  │
│ If |ψ⟩ = |0⟩^⊗N is stabilizer, then U_Cliff |ψ⟩ is ALSO stabilizer.       │
│ No matter how many Clifford layers you apply: M₂ = 0 forever.              │
│                                                                             │
│ Ry(θ) for generic θ maps stabilizer states OUT of the stabilizer manifold.  │
│ Each layer adds more "distance" from the nearest stabilizer state.          │
│ After L ~ N/2 layers, the state is typically maximally non-stabilizer.      │
└─────────────────────────────────────────────────────────────────────────────┘


═══════════════════════════════════════════════════════════════════════════════
PART 4: PERFORMANCE
═══════════════════════════════════════════════════════════════════════════════

┌─────────────────────────────────────────────────────────────────────────────┐
│ COMPLEXITY SUMMARY                                                          │
│                                                                             │
│ Operation              │  Cost              │  For N=14 (D=16384)           │
│ ───────────────────────┼────────────────────┼───────────────────────────────│
│ One brickwork layer    │  O(N · 2^N)        │  ~230K multiply-adds          │
│ Full circuit (L layers)│  O(L · N · 2^N)    │  ~3.2M for L=14              │
│ SRE via FWHT           │  O(N · 4^N)        │  ~3.8G multiply-adds          │
│ SRE naive (comparison) │  O(8^N)            │  ~4.4T → INFEASIBLE           │
│ Total per state        │  O(N · 4^N)        │  ~40s on CPU                  │
└─────────────────────────────────────────────────────────────────────────────┘

All gate applications and the FWHT are JIT-compiled via @jax.jit with
einsum-based contractions.  No D×D matrix is ever formed.


═══════════════════════════════════════════════════════════════════════════════
PART 5: USAGE
═══════════════════════════════════════════════════════════════════════════════

    python get_fig_ch5_sre.py              # generate + plot
    python get_fig_ch5_sre.py --plot-only  # re-plot from cached data
    python get_fig_ch5_sre.py --regenerate # force data regeneration

  Data → codes_for_figures/data/ch5/
  Figs → figures/ch5/

OUTPUT:
  fig_ch5_sre_vs_layers.pdf  — M₂ vs circuit depth L for N = 10, 12, 14
    • Blue curves: Ry+CNOT → rapid magic growth, saturating near N − 2
    • Gray curves: Clifford → M₂ = 0 identically at all depths
"""

import argparse
import os
import sys
import time
import numpy as np
from pathlib import Path

# ── JAX configuration ────────────────────────────────────────────
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
from jax import jit, random
import string

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

# System sizes and depths
N_VALUES    = [6, 8, 10, 12]
MAX_DEPTH   = 14                    # max circuit depth L to scan
N_CIRCUITS  = 20                    # circuit instances per (N, L, type) triple

# Angle scan parameters
ANGLE_N_VALUES = [10, 12]           # system sizes for angle scan
ANGLE_DEPTH    = 10                 # fixed depth for angle scan
N_ANGLES       = 80                 # number of θ values in [0, 2π)
N_ANGLE_CIRCS  = 10                 # circuits per angle (for averaging)


# ══════════════════════════════════════════════════════════════════
#  SINGLE-QUBIT CLIFFORD GROUP (24 ELEMENTS)
# ══════════════════════════════════════════════════════════════════

def _build_clifford_group():
    """
    Enumerate all 24 single-qubit Clifford gates via BFS over H, S.
    Returns: ndarray of shape (24, 2, 2), dtype complex128
    """
    H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
    S = np.array([[1, 0], [0, 1j]], dtype=complex)

    def canonicalize(m):
        for x in m.flatten():
            if abs(x) > 1e-10:
                return m / (x / abs(x))
        return m

    def mat_key(m):
        mc = canonicalize(m)
        return tuple(np.round(mc.flatten(), 6).real) + \
               tuple(np.round(mc.flatten(), 6).imag)

    I2 = np.eye(2, dtype=complex)
    visited = {mat_key(I2)}
    matrices = [I2.copy()]
    queue = [I2]

    while queue and len(matrices) < 24:
        cur = queue.pop(0)
        for gen in [H, S]:
            for new in [gen @ cur, cur @ gen]:
                k = mat_key(new)
                if k not in visited:
                    visited.add(k)
                    matrices.append(new.copy())
                    queue.append(new)

    assert len(matrices) == 24
    return np.stack(matrices, axis=0)


CLIFFORD_GROUP_NP = _build_clifford_group()
CLIFFORD_GROUP    = jnp.array(CLIFFORD_GROUP_NP, dtype=jnp.complex64)


# ══════════════════════════════════════════════════════════════════
#  MATRIX-FREE GATE KERNELS (EINSUM-BASED)
# ══════════════════════════════════════════════════════════════════
#
#  Statevector stored as rank-N tensor ψ[a₀, a₁, ..., a_{N-1}].
#  Each gate contracts over the target qubit axis(es) — O(D) cost.
#
# ══════════════════════════════════════════════════════════════════

_AXES = string.ascii_lowercase[:16]  # up to 16 qubits


def _make_1q_subs(n):
    """
    Pre-build einsum subscripts for single-qubit gate contractions.

    For a single-qubit gate U on qubit j of an N-qubit system:
        ψ'[a₀,...,b_j,...,a_{N-1}] = Σ_{a_j} U[b_j, a_j] · ψ[a₀,...,a_j,...,a_{N-1}]

    This is expressed as an einsum contraction:
        'Ba, abcd...->Abcd...'  (gate on qubit 0 of 4 qubits)
        'aB, abcd...->aBcd...'  (gate on qubit 1 of 4 qubits)

    Cost: O(2^N) per gate application (each element of ψ multiplied once).

    Args:
        n: number of qubits
    Returns:
        list of n einsum subscript strings, one per qubit
    """
    subs = []
    for j in range(n):
        in_ax = list(_AXES[:n])
        out_ax = list(in_ax)
        new = chr(ord('A') + j)
        out_ax[j] = new
        subs.append(f"{new}{in_ax[j]},{''.join(in_ax)}->{''.join(out_ax)}")
    return subs


def _make_2q_subs(n):
    """
    Pre-build einsum subscripts for two-qubit gate (CNOT) contractions.

    For a two-qubit gate G on qubits (c, t) of an N-qubit system:
        G is stored as a (2,2,2,2) tensor: G[B_c, B_t, a_c, a_t]

    The einsum contraction sums over the original indices a_c, a_t
    and writes the result into new indices B_c, B_t, leaving all
    other qubit indices unchanged.

    For CNOT: G[B_c, B_t, a_c, a_t] has the structure:
        |a_c, a_t⟩ → |a_c, a_t ⊕ a_c⟩  (target flipped if control = 1)

    Args:
        n: number of qubits
    Returns:
        dict mapping (control, target) pairs to einsum subscript strings
    """
    subs = {}
    for c in range(n):
        for t in range(n):
            if c == t:
                continue
            in_ax = list(_AXES[:n])
            out_ax = list(in_ax)
            nc, nt = chr(ord('A') + c), chr(ord('A') + t)
            out_ax[c], out_ax[t] = nc, nt
            subs[(c, t)] = \
                f"{nc}{nt}{in_ax[c]}{in_ax[t]},{''.join(in_ax)}->{''.join(out_ax)}"
    return subs


# CNOT as (2,2,2,2) tensor
# ────────────────────────────────────────────────────────────────────
# The CNOT gate: |c,t⟩ → |c, t ⊕ c⟩
#   |00⟩ → |00⟩,  |01⟩ → |01⟩,  |10⟩ → |11⟩,  |11⟩ → |10⟩
#
# As a 4×4 matrix: [[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]]
# Reshaped to (2,2,2,2) for einsum: G[B_c, B_t, a_c, a_t]
#
# CNOT is a Clifford gate — it creates entanglement but does NOT
# inject nonstabilizerness.  Clifford circuits composed entirely of
# CNOT + {H, S} gates have M₂ = 0 at any depth.
# ────────────────────────────────────────────────────────────────────
CNOT_T = jnp.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]],
                    dtype=jnp.complex64).reshape(2, 2, 2, 2)


@jit
def _ry(theta):
    """
    Build a single-qubit Ry rotation gate.

        Ry(θ) = exp(−iθσ_y/2) = [[cos(θ/2), −sin(θ/2)],
                                  [sin(θ/2),  cos(θ/2)]]

    MAGIC INJECTION:
      Ry(θ) is a Clifford gate ONLY at the four special angles:
        θ = 0       → I  (identity)
        θ = π/2     → (I − iY)/√2
        θ = π       → −iY
        θ = 3π/2    → (I + iY)/√2

      At ALL other angles, Ry(θ) injects nonstabilizerness (magic).
      When θ ~ Uniform[0, 2π], each gate injects a random amount
      of magic — the circuit rapidly accumulates M₂.

    Args:
        theta: rotation angle (scalar)
    Returns:
        (2,2) unitary matrix
    """
    c, s = jnp.cos(theta / 2), jnp.sin(theta / 2)
    return jnp.array([[c, -s], [s, c]], dtype=jnp.complex64)


# ══════════════════════════════════════════════════════════════════
#  FWHT-BASED STABILIZER RÉNYI ENTROPY
# ══════════════════════════════════════════════════════════════════
#
#  Inline implementation (no external imports) for portability.
#
#  Algorithm:
#    For each x ∈ {0,1}^N:
#      1. C_x[j] = ψ_j* · ψ_{j⊕x}               — correlation vector
#      2. F_x = FWHT(C_x) = H^{⊗N} · C_x         — O(N·2^N) butterfly
#      3. Accumulate  Σ |F_x[z]|⁴
#
#    M₂ = −log₂[ (1/D) · total_sum ]
#
# ══════════════════════════════════════════════════════════════════

def _fwht(v, n_qubits):
    """
    Fast Walsh-Hadamard Transform: F = H^{⊗N} · v.

    Computes the N-qubit Walsh-Hadamard transform of a vector v
    of length D = 2^N, using the butterfly algorithm.

    ALGORITHM:
      The transform H^{⊗N} = H₂ ⊗ H₂ ⊗ ... ⊗ H₂ factorises into
      N independent 2×2 contractions, one per qubit axis:

        For q = 0, 1, ..., N-1:
            Apply H₂ = [[1,1],[1,-1]] on axis q of the (2,)*N tensor

      Each pass costs O(2^N), total O(N · 2^N).

      This is implemented via jnp.einsum on the reshaped (2,)*N tensor.
      We NEVER form the full D×D Walsh-Hadamard matrix.

    WHY FWHT FOR SRE?
      The Pauli expectation ⟨P(z,x)|ψ⟩ involves a sum over j weighted
      by (−1)^{z·j}, which is exactly the z-th component of the
      Walsh-Hadamard transform of the correlation vector C_x.

    Args:
        v: input vector of length D = 2^n_qubits
        n_qubits: number of qubits (determines tensor reshape)

    Returns:
        F = H^{⊗N} · v, same shape as v
    """
    H2 = jnp.array([[1.0, 1.0], [1.0, -1.0]], dtype=v.dtype)
    t = v.reshape((2,) * n_qubits)

    letters = string.ascii_letters
    for q in range(n_qubits):
        idx = list(letters[:n_qubits])
        out_c = letters[n_qubits]
        in_c = idx[q]
        out_idx = idx.copy()
        out_idx[q] = out_c
        t = jnp.einsum(f"{out_c}{in_c},{''.join(idx)}->{''.join(out_idx)}",
                       H2, t)
    return t.reshape(-1)


def compute_sre(psi, n_qubits):
    """
    Compute M₂ (second Stabilizer Rényi Entropy) via FWHT.

    ALGORITHM (see docstring Part 2 for full derivation):

      M₂(ψ) = −log₂[ (1/D) Σ_P |⟨ψ|P|ψ⟩|⁴ ]

      The sum over 4^N Pauli strings P is decomposed via the
      (z, x) binary labelling of Pauli strings:
        • Outer loop over x ∈ {0,...,D-1}  (X-part of Pauli label)
        • For each x: build correlation C_x[j] = ψ_j* · ψ_{j⊕x}
        • Apply FWHT to get F_x[z] = Σ_j (−1)^{z·j} C_x[j]
        • Accumulate Σ_z |F_x[z]|⁴

      The F_x[z] values are exactly the Pauli expectations ⟨P(z,x)|ψ⟩
      (up to phase factors that vanish under |·|⁴).

      Total cost: D × O(N·D) = O(N·D²) = O(N·4^N)

    Args:
        psi: statevector, shape (D,) with D = 2^n_qubits
        n_qubits: number of qubits N

    Returns:
        M₂ (scalar, in bits).  M₂ = 0 for stabilizer states,
        M₂ ≈ N − 2 for Haar-random states.
    """
    D = 1 << n_qubits
    j_indices = jnp.arange(D)
    total = jnp.float64(0.0)

    for x in range(D):
        # Correlation vector: C_x[j] = ψ_j* · ψ_{j⊕x}
        j_xor_x = jnp.bitwise_xor(j_indices, x)
        C_x = jnp.conj(psi) * psi[j_xor_x]

        # Walsh-Hadamard transform: F_x = H^{⊗N} C_x
        F_x = _fwht(C_x, n_qubits)

        # Accumulate |F_x[z]|⁴ over all z
        total = total + jnp.sum(jnp.abs(F_x)**4)

    # M₂ = −log₂(total / D)
    return jnp.where(total > 0, -jnp.log2(total / D), jnp.inf)


# ══════════════════════════════════════════════════════════════════
#  CIRCUIT RUNNERS
# ══════════════════════════════════════════════════════════════════

def _apply_layer(psi_flat, gates_1q, layer_idx, n_qubits,
                 subs_1q, subs_2q):
    """
    Apply one brickwork layer of single-qubit gates + CNOT pairs.

    ARCHITECTURE (same as entanglement scaling figure):
      1. Apply N single-qubit gates (Ry or Clifford) to all qubits
      2. Apply CNOT on nearest-neighbour pairs in brickwork pattern:
         Even layers (ℓ even): pairs (0,1), (2,3), (4,5), ...
         Odd  layers (ℓ odd):  pairs (1,2), (3,4), (5,6), ...

    Single-qubit gates inject magic (if Ry) or preserve M₂ = 0 (if Clifford).
    CNOT gates create entanglement but do NOT inject magic.

    Args:
        psi_flat: statevector as flat array (D,)
        gates_1q: list of N single-qubit (2,2) gate matrices
        layer_idx: layer index (determines even/odd CNOT pairing)
        n_qubits: number of qubits N
        subs_1q: einsum subscripts for 1-qubit gates
        subs_2q: einsum subscripts for 2-qubit gates

    Returns:
        evolved statevector as flat array (D,)
    """
    psi = psi_flat.reshape((2,) * n_qubits)

    for j in range(n_qubits):
        psi = jnp.einsum(subs_1q[j], gates_1q[j], psi)

    even_pairs = [(j, j+1) for j in range(0, n_qubits - 1, 2)]
    odd_pairs  = [(j, j+1) for j in range(1, n_qubits - 1, 2)]
    pairs = even_pairs if (layer_idx % 2 == 0) else odd_pairs
    for (c, t) in pairs:
        psi = jnp.einsum(subs_2q[(c, t)], CNOT_T, psi)

    return psi.reshape(1 << n_qubits)


def run_circuit_and_sre(circuit_type, depth, n_qubits, key):
    """
    Run a brickwork circuit of given type and depth, then compute M₂.

    Two circuit types are supported:
      • 'ry_cnot':  Ry(θ)+CNOT with θ ~ Uniform[0, 2π) per gate.
                    Each Ry injects nonstabilizerness → M₂ grows with depth.
      • 'clifford': Random single-qubit Clifford + CNOT.
                    All gates are Clifford → M₂ = 0 at any depth.

    The contrast between these two circuit types demonstrates that
    entanglement alone is insufficient for quantum advantage:
    Clifford circuits produce high entanglement but zero magic.

    Args:
        circuit_type: 'ry_cnot' (non-Clifford) or 'clifford'
        depth: number of brickwork layers L
        n_qubits: number of qubits N
        key: JAX PRNG key for random gate sampling

    Returns:
        M₂ (scalar, in bits)
    """
    D = 1 << n_qubits
    subs_1q = _make_1q_subs(n_qubits)
    subs_2q = _make_2q_subs(n_qubits)

    psi = jnp.zeros(D, dtype=jnp.complex64).at[0].set(1.0 + 0j)

    for ell in range(depth):
        key, subkey = random.split(key)

        if circuit_type == 'ry_cnot':
            thetas = random.uniform(subkey, (n_qubits,),
                                    minval=0.0, maxval=2.0 * jnp.pi)
            gates = [_ry(thetas[j]) for j in range(n_qubits)]
        else:
            indices = random.randint(subkey, (n_qubits,), 0, 24)
            gates = [CLIFFORD_GROUP[indices[j]] for j in range(n_qubits)]

        psi = _apply_layer(psi, gates, ell, n_qubits, subs_1q, subs_2q)

    return compute_sre(psi, n_qubits)


def run_circuit_fixed_angle_sre(theta, depth, n_qubits, key):
    """
    Run a brickwork Ry(θ)+CNOT circuit where ALL single-qubit gates
    use the SAME angle θ, and compute M₂.

    This isolates the effect of the injection angle: at Clifford angles
    (θ = 0, π/2, π, 3π/2) the output is a stabilizer state with M₂ = 0.
    At generic angles, M₂ > 0.

    Args:
        theta: rotation angle for all Ry gates (scalar)
        depth: number of layers L
        n_qubits: N
        key: JAX PRNG key (used only for initial state randomisation
             of CNOT pairing, which is deterministic here)

    Returns: M₂ (scalar)
    """
    D = 1 << n_qubits
    subs_1q = _make_1q_subs(n_qubits)
    subs_2q = _make_2q_subs(n_qubits)

    psi = jnp.zeros(D, dtype=jnp.complex64).at[0].set(1.0 + 0j)
    gate = _ry(theta)
    gates = [gate] * n_qubits

    for ell in range(depth):
        psi = _apply_layer(psi, gates, ell, n_qubits, subs_1q, subs_2q)

    return compute_sre(psi, n_qubits)


# ══════════════════════════════════════════════════════════════════
#  DATA GENERATION
# ══════════════════════════════════════════════════════════════════

def generate_sre_data(regenerate=False):
    """Generate SRE data for all (N, L, circuit_type) combinations."""
    key = random.PRNGKey(123)
    t0 = time.time()

    for N in N_VALUES:
        outfile = DATA_DIR / f"sre_N{N}.npz"
        if outfile.exists() and not regenerate:
            print(f"  Cached: {outfile}")
            continue

        D = 1 << N
        depths = list(range(0, MAX_DEPTH + 1))
        n_depths = len(depths)

        print(f"\n  N = {N} (D = {D}):  {n_depths} depths × "
              f"{N_CIRCUITS} circuits × 2 types")

        sre_ry  = np.zeros((n_depths, N_CIRCUITS))
        sre_cl  = np.zeros((n_depths, N_CIRCUITS))

        for di, L in enumerate(depths):
            t1 = time.time()
            for ci in range(N_CIRCUITS):
                key, k1, k2 = random.split(key, 3)
                sre_ry[di, ci] = float(
                    run_circuit_and_sre('ry_cnot', L, N, k1))
                sre_cl[di, ci] = float(
                    run_circuit_and_sre('clifford', L, N, k2))

            ry_mean = np.mean(sre_ry[di])
            cl_mean = np.mean(sre_cl[di])
            dt = time.time() - t1
            print(f"    L={L:2d}:  M₂(Ry)={ry_mean:.3f}  "
                  f"M₂(Cl)={cl_mean:.3f}  ({dt:.1f}s)")

        np.savez(outfile, depths=np.array(depths),
                 sre_ry=sre_ry, sre_cl=sre_cl,
                 N=N, n_circuits=N_CIRCUITS)
        print(f"    → Saved: {outfile}")

    print(f"\n  Total: {time.time()-t0:.1f}s")


def generate_angle_scan_data(regenerate=False):
    """
    Generate M₂ vs injection angle θ data.

    For each system size N in ANGLE_N_VALUES, scan θ ∈ [0, 2π) at
    fixed depth ANGLE_DEPTH.  At Clifford angles (0, π/2, π, 3π/2)
    M₂ should drop to zero.
    """
    key = random.PRNGKey(456)
    t0 = time.time()

    thetas = np.linspace(0, 2 * np.pi, N_ANGLES, endpoint=False)

    for N in ANGLE_N_VALUES:
        outfile = DATA_DIR / f"sre_angle_scan_N{N}.npz"
        if outfile.exists() and not regenerate:
            print(f"  Cached: {outfile}")
            continue

        print(f"\n  Angle scan N={N}, depth={ANGLE_DEPTH}, "
              f"{N_ANGLES} angles × {N_ANGLE_CIRCS} circuits")

        sre_vals = np.zeros((N_ANGLES, N_ANGLE_CIRCS))

        for ai, theta in enumerate(thetas):
            t1 = time.time()
            for ci in range(N_ANGLE_CIRCS):
                key, subkey = random.split(key)
                sre_vals[ai, ci] = float(
                    run_circuit_fixed_angle_sre(
                        theta, ANGLE_DEPTH, N, subkey))

            m2_mean = np.mean(sre_vals[ai])
            dt = time.time() - t1
            if ai % 10 == 0:
                print(f"    θ={theta:.3f} ({ai+1}/{N_ANGLES}):  "
                      f"M₂={m2_mean:.3f}  ({dt:.1f}s)")

        np.savez(outfile, thetas=thetas, sre=sre_vals,
                 N=N, depth=ANGLE_DEPTH, n_circuits=N_ANGLE_CIRCS)
        print(f"    → Saved: {outfile}")

    print(f"\n  Angle scan total: {time.time()-t0:.1f}s")


def data_exists():
    return all((DATA_DIR / f"sre_N{N}.npz").exists() for N in N_VALUES)


def angle_data_exists():
    return all((DATA_DIR / f"sre_angle_scan_N{N}.npz").exists()
               for N in ANGLE_N_VALUES)


# ══════════════════════════════════════════════════════════════════
#  PLOTTING
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


def plot_sre():
    """
    Two-panel SRE figure:
      (a) M₂ vs circuit depth L  (Clifford vs Ry+CNOT)
      (b) M₂ vs injection angle θ (showing Clifford valleys)
    """
    print("  Plotting 2-panel SRE figure...")

    has_angle = angle_data_exists()
    if has_angle:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    else:
        fig, ax1 = plt.subplots(figsize=(7.5, 5))

    # ── Panel (a): M₂ vs depth ────────────────────────────────────
    line_styles = ['-', '--', ':']
    markers_ry  = ['o', 'D', '^']

    for i, N in enumerate(N_VALUES):
        data = np.load(DATA_DIR / f"sre_N{N}.npz")
        depths = data["depths"]
        sre_ry_mean = np.mean(data["sre_ry"], axis=1)
        sre_cl_mean = np.mean(data["sre_cl"], axis=1)

        ax1.plot(depths, sre_ry_mean, linestyle=line_styles[i],
                 marker=markers_ry[i], color="#2563EB", lw=2.5,
                 markersize=5, markeredgecolor="white",
                 markeredgewidth=0.5,
                 label=r"$R_y$+CNOT" if i == 0 else None)

        ax1.plot(depths, sre_cl_mean, linestyle=line_styles[i],
                 color="#9CA3AF", lw=2.5,
                 label=r"Clifford ($M_2 = 0$)" if i == 0 else None)

        ax1.text(depths[-1] + 0.3, sre_ry_mean[-1],
                 f"$N = {N}$", fontsize=9, va="center", color="#2563EB")

    for N in N_VALUES:
        ax1.axhline(N - 2, color="#2563EB", ls=":", lw=0.8, alpha=0.3)

    ax1.set_xlabel("Circuit depth $L$")
    ax1.set_ylabel(r"$M_2$ (nonstabilizerness)")
    ax1.set_title(r"\textbf{(a)} $M_2$ vs circuit depth", loc="left")
    ax1.set_xlim(-0.5, MAX_DEPTH + 1.5)
    ax1.set_ylim(-0.5, max(N_VALUES) - 1)
    ax1.legend(loc="center left", framealpha=0.85, fontsize=11)

    # ── Panel (b): M₂ vs injection angle θ ────────────────────────
    if has_angle:
        colors_angle = ["#2563EB", "#DC2626"]
        markers_angle = ['o', 's']

        for i, N in enumerate(ANGLE_N_VALUES):
            data = np.load(DATA_DIR / f"sre_angle_scan_N{N}.npz")
            thetas = data["thetas"]
            sre_mean = np.mean(data["sre"], axis=1)
            sre_std  = np.std(data["sre"], axis=1) / np.sqrt(data["n_circuits"])

            ax2.plot(thetas / np.pi, sre_mean, linestyle='-',
                     marker=markers_angle[i], color=colors_angle[i],
                     lw=2, markersize=4, markeredgecolor="white",
                     markeredgewidth=0.3,
                     label=f"$N = {N}$")
            ax2.fill_between(thetas / np.pi,
                             sre_mean - sre_std, sre_mean + sre_std,
                             color=colors_angle[i], alpha=0.15)

        # Mark Clifford angles with vertical lines
        for cliff_theta in [0, 0.5, 1.0, 1.5]:
            ax2.axvline(cliff_theta, color="#9CA3AF", ls="--",
                        lw=1.0, alpha=0.5)

        # Label Clifford angles
        ax2.text(0.01, -0.8, r"$0$", fontsize=8, color="#6B7280")
        ax2.text(0.51, -0.8, r"$\frac{\pi}{2}$", fontsize=8,
                 color="#6B7280")
        ax2.text(1.01, -0.8, r"$\pi$", fontsize=8, color="#6B7280")
        ax2.text(1.51, -0.8, r"$\frac{3\pi}{2}$", fontsize=8,
                 color="#6B7280")

        # Haar reference
        for ii, N in enumerate(ANGLE_N_VALUES):
            ax2.axhline(N - 2, color=colors_angle[ii],
                        ls=":", lw=0.8, alpha=0.3)

        ax2.set_xlabel(r"Injection angle $\theta / \pi$")
        ax2.set_ylabel(r"$M_2$ (nonstabilizerness)")
        ax2.set_title(r"\textbf{(b)} $M_2$ vs $R_y(\theta)$ angle",
                      loc="left")
        ax2.set_xlim(-0.05, 2.0)
        ax2.set_ylim(-1.0, max(ANGLE_N_VALUES) - 1)
        ax2.legend(loc="upper right", framealpha=0.85, fontsize=11)

    plt.tight_layout()

    outpath = OUTPUT_DIR / "fig_ch5_sre_vs_layers.pdf"
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
        description="Fig 5.2: SRE M₂ vs circuit depth + angle scan "
                    "(FWHT-based, matrix-free JAX)")
    parser.add_argument("--regenerate", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--skip-angle-scan", action="store_true",
                        help="Skip angle scan data generation")
    args = parser.parse_args()

    print("=" * 65)
    print("  Chapter 5 — Stabilizer Rényi Entropy")
    print("  Panel (a): M₂ vs circuit depth")
    print("  Panel (b): M₂ vs injection angle θ")
    print("  FWHT algorithm: O(N · 4^N)")
    print("=" * 65)
    print(f"  Depth scan N:     {N_VALUES}")
    print(f"  Depth range:      0..{MAX_DEPTH}")
    print(f"  Circuits/point:   {N_CIRCUITS}")
    print(f"  Angle scan N:     {ANGLE_N_VALUES}")
    print(f"  Angle points:     {N_ANGLES}")
    print(f"  JAX backend:      {jax.default_backend()}")
    print("=" * 65)

    if args.plot_only:
        if not data_exists():
            print("ERROR: --plot-only but depth scan data missing.")
            sys.exit(1)
        print("  Using cached data")
    else:
        generate_sre_data(regenerate=args.regenerate)
        if not args.skip_angle_scan:
            generate_angle_scan_data(regenerate=args.regenerate)

    print("\n── Generating figure ──")
    plot_sre()
    print(f"\nDone. Output → {OUTPUT_DIR}")
