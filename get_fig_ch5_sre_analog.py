#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║     STABILIZER RÉNYI ENTROPY (SRE) — ANALOG XX MODEL EVOLUTION             ║
║     A Tutorial for Quantum Information and Quantum Computing                ║
║                                                                             ║
║     Chapter 5 — Circuit Complexity & Quantum Designs                        ║
║     HIGH-PERFORMANCE MATRIX-FREE SRE COMPUTATION (JAX + FWHT)               ║
║     TROTTERIZED HAMILTONIAN TIME EVOLUTION                                  ║
╚══════════════════════════════════════════════════════════════════════════════╝

This script computes the growth of nonstabilizerness ("magic") M₂ under
continuous-time Hamiltonian evolution of the non-integrable XX model,
complementing the digital circuit results in get_fig_ch5_sre_digital.py.

Together, these two scripts demonstrate that magic growth is a UNIVERSAL
feature of quantum chaotic dynamics, arising in both:
  • Digital circuits (random Ry+CNOT brickwork layers)
  • Analog Hamiltonians (XX model with random transverse fields)


═══════════════════════════════════════════════════════════════════════════════
PART 0: THE XX HAMILTONIAN — PHYSICS AND INTEGRABILITY BREAKING
═══════════════════════════════════════════════════════════════════════════════

THE MODEL:
──────────

    Ĥ = Σ_{i=1}^{N-1} (σ̂_x^i σ̂_x^{i+1} + σ̂_y^i σ̂_y^{i+1}) + Σ_{i=1}^N h_i σ̂_x^i

where h_i ~ Uniform[-1,1] are quenched random transverse fields.

 • The first sum is the isotropic XX interaction (also called the
   flip-flop or hopping interaction), which conserves total spin.
 • The second sum is a random transverse field that BREAKS integrability.

WHY THIS MODEL?
────────────────

 1. Without the random field (h_i = 0), the XX model is integrable:
    it maps to free fermions via the Jordan-Wigner transformation.
    Integrable systems thermalize slowly and may not fully scramble.

 2. With random fields (h_i ≠ 0), the XX model becomes NON-INTEGRABLE
    (quantum chaotic):
    • Level statistics follow Wigner-Dyson (GOE/GUE) instead of Poisson
    • The system thermalizes and scrambles information
    • States evolve toward Haar-random-like properties

 3. The random field also breaks the U(1) symmetry of total magnetization,
    ensuring the system explores the full Hilbert space.

MAGIC GROWTH UNDER HAMILTONIAN EVOLUTION:
──────────────────────────────────────────

 As |0⟩^⊗N evolves under e^{-iĤt}:

     t = 0:     M₂ = 0     (|0⟩^⊗N is a stabilizer state)
     t ~ O(1):  M₂ ~ O(N)  (rapid magic injection from non-Clifford evolution)
     t ≫ 1:     M₂ → N − 2 (Haar-random saturation value)

 The saturation at M₂ ≈ N − 2 is the SAME as for deep random circuits.
 This universality reflects the underlying randomization of the Pauli
 spectrum by chaotic dynamics.


═══════════════════════════════════════════════════════════════════════════════
PART 1: STABILIZER RÉNYI ENTROPY — DEFINITION AND PROPERTIES
═══════════════════════════════════════════════════════════════════════════════

1.1  THE PAULI SPECTRUM
────────────────────────

Any pure state |ψ⟩ on N qubits can be expanded in the Pauli basis:

    |ψ⟩⟨ψ| = (1/D) Σ_P  ⟨ψ|P|ψ⟩ · P

where D = 2^N and the sum runs over all 4^N Pauli strings P ∈ {I, X, Y, Z}^⊗N.
The values {⟨ψ|P|ψ⟩}_P form the PAULI SPECTRUM.

Key properties:
  • ⟨ψ|I^⊗N|ψ⟩ = 1 always (normalisation)
  • −1 ≤ ⟨ψ|P|ψ⟩ ≤ 1 for all P
  • Normalisation: (1/D) Σ_P |⟨ψ|P|ψ⟩|² = 1

1.2  SRE DEFINITION
─────────────────────

The second Stabilizer Rényi Entropy:

    M₂(ψ) = −log₂[ (1/D) Σ_P |⟨ψ|P|ψ⟩|⁴ ]  =  −log₂[ Ξ(ψ) ]

where Ξ(ψ) = (1/D) Σ_P |⟨P⟩|⁴ is the SUM OF FOURTH POWERS.

Properties:
  • M₂ = 0 ↔ |ψ⟩ is a stabilizer state
  • M₂ is invariant under Clifford unitaries
  • M₂ ≤ N  (saturates for Haar-random states at ≈ N − 2)
  • Additive for product states: M₂(ψ⊗φ) = M₂(ψ) + M₂(φ)


═══════════════════════════════════════════════════════════════════════════════
PART 2: FAST WALSH-HADAMARD TRANSFORM (FWHT) FOR SRE
═══════════════════════════════════════════════════════════════════════════════

2.1  NAIVE APPROACH: O(8^N)
────────────────────────────

Direct computation of Σ_P |⟨P⟩|⁴ requires:
  • Enumerate all 4^N Pauli strings P
  • For each P, compute ⟨ψ|P|ψ⟩ via matrix-vector product: O(2^N)
  • Total: 4^N × 2^N = 8^N operations

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

where j⊕x = bitwise XOR.  Crucially, the z-sum is a WALSH-HADAMARD
TRANSFORM of the correlation vector C_x[j] = ψ_j* · ψ_{j⊕x}.

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
│ For N=12: naive = 8^12 ≈ 6.9 × 10^10                                       │
│           FWHT  = 12 × 4^12 ≈ 2.0 × 10^8   → 345× speedup                │
└─────────────────────────────────────────────────────────────────────────────┘

2.3  THE WALSH-HADAMARD BUTTERFLY
──────────────────────────────────

The FWHT of a vector v of length 2^N is computed by N passes:

    For q = 0, 1, ..., N−1:
        Contract axis q with H₂ = [[1,1],[1,−1]]

Each pass costs O(2^N), total O(N · 2^N).

This is implemented matrix-free via einsum:
    v.reshape((2,)*N) — apply H₂ on axis q — reshape back

We NEVER form the full 2^N × 2^N Walsh-Hadamard matrix.


═══════════════════════════════════════════════════════════════════════════════
PART 3: TROTTER DECOMPOSITION — BRICKWORK STRUCTURE
═══════════════════════════════════════════════════════════════════════════════

3.1  FIRST-ORDER SUZUKI-TROTTER
────────────────────────────────

The time evolution operator e^{-iĤt} is decomposed into small steps:

    e^{-iĤ dt} ≈ [∏_i exp(-i·dt·h_i·σ̂_x^i)]        ← field term
               · [∏_{i∈even} exp(-i·dt·Ĥ_{XX}^{i,i+1})]  ← even bonds
               · [∏_{i∈odd}  exp(-i·dt·Ĥ_{XX}^{i,i+1})]  ← odd bonds

This is the SAME brickwork structure as the digital circuit, with:
  • Single-qubit gates: Rx(2·dt·h_i) rotations         ← from field
  • Two-qubit gates: XX flip-flop gates                  ← from interaction

3.2  GATE MATRICES
───────────────────

SINGLE-QUBIT FIELD GATE:
    exp(-i·dt·h·σ_x) = cos(dt·h)·I − i·sin(dt·h)·σ_x
                      = [[cos(dt·h),     -i·sin(dt·h)],
                         [-i·sin(dt·h),   cos(dt·h)]]

    This is an Rx rotation about the x-axis by angle 2·dt·h.
    For generic h, this is a NON-CLIFFORD gate that injects magic.

    The Rx gate is Clifford ONLY when dt·h is a multiple of π/2:
      dt·h = 0     → Identity
      dt·h = π/2   → −i·σ_x  (a Clifford)
    For random h, this is generically non-Clifford → injects magic.

TWO-QUBIT XX GATE:
    exp(-i·dt·(σ_x⊗σ_x + σ_y⊗σ_y))

    In the computational basis {|00⟩, |01⟩, |10⟩, |11⟩}:
      |00⟩ → |00⟩                                   (unchanged)
      |01⟩ → cos(2dt)|01⟩ − i·sin(2dt)|10⟩          (flip-flop)
      |10⟩ → −i·sin(2dt)|01⟩ + cos(2dt)|10⟩         (flip-flop)
      |11⟩ → |11⟩                                   (unchanged)

    This is a partial SWAP: at dt = π/4, it becomes −i·SWAP.
    The XX gate is ENTANGLING but number-conserving.

3.3  MATRIX-FREE EINSUM IMPLEMENTATION
───────────────────────────────────────

Gate application is performed via tensor contraction:
  ψ[a₀, a₁, ..., a_{N-1}] is a rank-N tensor.

Single-qubit gate U on qubit j:
    ψ'[..., b_j, ...] = Σ_{a_j} U[b_j, a_j] · ψ[..., a_j, ...]

Two-qubit gate G on qubits (c, t):
    ψ'[..., B_c, B_t, ...] = Σ_{a_c, a_t} G[B_c, B_t, a_c, a_t]
                             · ψ[..., a_c, a_t, ...]

Both are implemented via jnp.einsum — no matrices larger than
4×4 are ever formed, regardless of system size.


═══════════════════════════════════════════════════════════════════════════════
PART 4: PERFORMANCE
═══════════════════════════════════════════════════════════════════════════════

┌─────────────────────────────────────────────────────────────────────────────┐
│ COMPLEXITY SUMMARY                                                          │
│                                                                             │
│ Operation              │  Cost              │  For N=12 (D=4096)            │
│ ───────────────────────┼────────────────────┼──────────────────────────────│
│ One Trotter step       │  O(N · 2^N)        │  ~50K multiply-adds           │
│ Full evolution (200 st)│  O(200·N·2^N)      │  ~10M multiply-adds           │
│ SRE via FWHT           │  O(N · 4^N)        │  ~2.0 × 10^8 ops             │
│   (bottleneck!)        │                    │  ~30s on CPU                  │
│ SRE naive (comparison) │  O(8^N)            │  ~6.9 × 10^10 → INFEASIBLE   │
│ Total per time point   │  O(N · 4^N)        │  ~30s on CPU                  │
│                                                                             │
│ For N=16: FWHT = 16 × 4^16 ≈ 6.9 × 10^10 → HOURS per point               │
│ → N=12 is the practical limit for exact SRE on a single CPU.               │
└─────────────────────────────────────────────────────────────────────────────┘


═══════════════════════════════════════════════════════════════════════════════
PART 5: USAGE
═══════════════════════════════════════════════════════════════════════════════

    python get_fig_ch5_sre_analog.py              # generate + plot
    python get_fig_ch5_sre_analog.py --plot-only   # re-plot from cached data
    python get_fig_ch5_sre_analog.py --regenerate  # force data regeneration

  Data → codes_for_figures/data/ch5/
  Figs → figures/ch5/

OUTPUT:
  fig_ch5_sre_vs_time.pdf  — M₂ vs evolution time t for N = 6, 8, 10, 12
    • At t = 0, M₂ = 0 (stabilizer initial state)
    • As t grows, M₂ rises and saturates at ≈ N − 2 (Haar-random value)
    • The saturation is FASTER for larger N (more rapid scrambling)
"""


import argparse
import os
import sys
import time as timer
import string
import numpy as np
from pathlib import Path

# ── JAX configuration ────────────────────────────────────────────
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
from jax import jit

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "axes.labelsize": 14,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

CDT = jnp.complex128

# ══════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════════════

BOOK_DIR   = Path(__file__).resolve().parent.parent
DATA_DIR   = Path(__file__).resolve().parent / "data" / "ch5"
OUTPUT_DIR = BOOK_DIR / "figures" / "ch5"
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# System sizes and time scan
N_VALUES    = [6, 8, 10, 12]
TIME_MAX    = 10.0              # max evolution time
TIME_STEP   = 0.5               # time between SRE measurements
DT_TROTTER  = 0.05              # Trotter step size
SEED        = 42                # reproducible random fields


# ══════════════════════════════════════════════════════════════════
#  EINSUM SUBSCRIPT GENERATION
# ══════════════════════════════════════════════════════════════════
#
#  Statevector stored as rank-N tensor ψ[a₀, a₁, ..., a_{N-1}].
#  Gate application via tensor contraction — O(2^N) per gate.
#
# ══════════════════════════════════════════════════════════════════

_AXES = string.ascii_lowercase[:20]

def _make_1q_subs(n):
    """
    Pre-build einsum subscripts for single-qubit gate contractions.

    For a single-qubit gate U on qubit j of an N-qubit system:
        ψ'[a₀,...,b_j,...,a_{N-1}] = Σ_{a_j} U[b_j, a_j] · ψ[a₀,...,a_j,...,a_{N-1}]

    This is expressed as an einsum contraction:
        'Ba, abcd...->Abcd...'  (gate on qubit 0 of 4 qubits)
        'aB, abcd...->aBcd...'  (gate on qubit 1 of 4 qubits)

    Args:
        n: number of qubits
    Returns:
        list of n einsum subscript strings, one per qubit
    """
    subs = []
    for j in range(n):
        ax = list(_AXES[:n])
        out = list(ax)
        new = chr(ord('A') + j)
        out[j] = new
        subs.append(f"{new}{ax[j]},{''.join(ax)}->{''.join(out)}")
    return subs

def _make_2q_subs(n):
    """
    Pre-build einsum subscripts for two-qubit gate contractions.

    For a two-qubit gate G on qubits (c, t) of an N-qubit system:
        G is stored as a (2,2,2,2) tensor: G[B_c, B_t, a_c, a_t]

    The einsum contraction sums over the original indices a_c, a_t
    and writes the result into new indices B_c, B_t, leaving all
    other qubit indices unchanged.

    Args:
        n: number of qubits
    Returns:
        dict mapping (control, target) pairs to einsum subscript strings
    """
    subs = {}
    for c in range(n):
        for t in range(n):
            if c == t: continue
            ax = list(_AXES[:n])
            out = list(ax)
            C, T = chr(ord('A')+c), chr(ord('A')+t)
            out[c], out[t] = C, T
            subs[(c,t)] = f"{C}{T}{ax[c]}{ax[t]},{''.join(ax)}->{''.join(out)}"
    return subs


# ══════════════════════════════════════════════════════════════════
#  TROTTER GATE CONSTRUCTION
# ══════════════════════════════════════════════════════════════════

def _build_rx_gates(h_fields, dt):
    """
    Build the single-qubit field rotation gates for all qubits.

    Each gate implements the transverse field term of the Hamiltonian:
        exp(-i · dt · h_i · σ_x)

    Using the identity exp(-iθA) = cos(θ)I − i·sin(θ)A for A² = I:

        exp(-i·dt·h·σ_x) = cos(dt·h)·I − i·sin(dt·h)·σ_x

    In the computational basis:
        = [[cos(dt·h),     -i·sin(dt·h)],
           [-i·sin(dt·h),   cos(dt·h)]]

    This is equivalent to an Rx rotation: Rx(2·dt·h) = exp(-i·dt·h·σ_x).

    MAGIC INJECTION:
      For generic h values, dt·h ∉ {0, π/2, π, 3π/2, ...}, so these
      rotations are NOT Clifford gates and therefore INJECT MAGIC.
      This is why the chaotic Hamiltonian generates nonstabilizerness.

    Args:
        h_fields: array of N field strengths h_i
        dt: Trotter time step
    Returns:
        list of N (2,2) gate matrices, one per qubit
    """
    gates = []
    for h in h_fields:
        c = jnp.cos(dt * h)
        s = jnp.sin(dt * h)
        gates.append(jnp.array([[c, -1j*s], [-1j*s, c]], dtype=CDT))
    return gates


def _build_xx_gate(dt):
    """
    Build the two-qubit XX interaction gate.

    Implements the nearest-neighbour XX coupling:
        exp(-i · dt · (σ_x⊗σ_x + σ_y⊗σ_y))

    DERIVATION:
      The operator σ_x⊗σ_x + σ_y⊗σ_y can be written as 2(|01⟩⟨10| + |10⟩⟨01|),
      which is the flip-flop (hopping) operator:
      it swaps |01⟩ ↔ |10⟩ while leaving |00⟩ and |11⟩ invariant.

      Since (|01⟩⟨10| + |10⟩⟨01|)² = |01⟩⟨01| + |10⟩⟨10| (projector onto
      the {|01⟩, |10⟩} subspace), we can exponentiate analytically:

      exp(-i·2dt·(|01⟩⟨10| + |10⟩⟨01|)) acts as:
        |00⟩ →  |00⟩                                     (unchanged)
        |01⟩ →  cos(2dt)|01⟩ − i·sin(2dt)|10⟩             (flip-flop)
        |10⟩ → −i·sin(2dt)|01⟩ + cos(2dt)|10⟩             (flip-flop)
        |11⟩ →  |11⟩                                     (unchanged)

      At dt = π/4: this becomes −i·SWAP (full state exchange).

    The gate is stored as a (2,2,2,2) tensor for einsum contraction:
        G[b_c, b_t, a_c, a_t]  where c = control qubit, t = target qubit.

    Args:
        dt: Trotter time step
    Returns:
        (2,2,2,2) gate tensor
    """
    c = jnp.cos(2.0 * dt)
    s = jnp.sin(2.0 * dt)
    return jnp.array([
        [1,     0,       0,    0],
        [0,     c,    -1j*s,   0],
        [0,  -1j*s,      c,   0],
        [0,     0,       0,    1],
    ], dtype=CDT).reshape(2, 2, 2, 2)


# ══════════════════════════════════════════════════════════════════
#  TROTTERIZED TIME EVOLUTION
# ══════════════════════════════════════════════════════════════════

def evolve_one_step(psi_flat, N, rx_gates, xx_gate, subs_1q, subs_2q, even, odd):
    """
    Apply one first-order Trotter step to the statevector.

    The step implements:
      |ψ(t+dt)⟩ ≈ [∏_i Rx_i] · [∏_{even} XX] · [∏_{odd} XX] · |ψ(t)⟩

    Gate application order:
      1. All N single-qubit field rotations Rx(2·dt·h_i)
      2. XX gates on even-index pairs: (0,1), (2,3), ...
      3. XX gates on odd-index pairs:  (1,2), (3,4), ...

    This brickwork layout ensures all gates within each sublayer commute,
    so the ordering within each sublayer doesn't matter.

    Args:
        psi_flat: statevector as flat array (D,)
        N: number of qubits
        rx_gates: list of N single-qubit (2,2) gate matrices
        xx_gate: (2,2,2,2) two-qubit gate tensor
        subs_1q: einsum subscript strings for 1-qubit gates
        subs_2q: einsum subscript strings for 2-qubit gates
        even: list of even-index qubit pairs [(0,1), (2,3), ...]
        odd: list of odd-index qubit pairs [(1,2), (3,4), ...]

    Returns:
        evolved statevector as flat array (D,)
    """
    p = psi_flat.reshape((2,)*N)

    # 1. Single-qubit field gates
    for j in range(N):
        p = jnp.einsum(subs_1q[j], rx_gates[j], p)

    # 2. XX gates on even pairs
    for c, t in even:
        p = jnp.einsum(subs_2q[(c, t)], xx_gate, p)

    # 3. XX gates on odd pairs
    for c, t in odd:
        p = jnp.einsum(subs_2q[(c, t)], xx_gate, p)

    return p.reshape(1 << N)


def evolve_xx_trotter(N, n_steps, h_fields, dt=DT_TROTTER):
    """
    Evolve |0⟩^⊗N under XX Hamiltonian for n_steps Trotter steps.

    Returns the final statevector.
    """
    D = 1 << N
    subs_1q = _make_1q_subs(N)
    subs_2q = _make_2q_subs(N)
    rx_gates = _build_rx_gates(h_fields, dt)
    xx_gate  = _build_xx_gate(dt)
    even = [(j, j+1) for j in range(0, N-1, 2)]
    odd  = [(j, j+1) for j in range(1, N-1, 2)]

    # Initial state: |0⟩^⊗N  (stabilizer state → M₂ = 0)
    psi = jnp.zeros(D, dtype=CDT).at[0].set(1.0 + 0j)

    for step in range(n_steps):
        psi = evolve_one_step(psi, N, rx_gates, xx_gate,
                              subs_1q, subs_2q, even, odd)
    return psi


# ══════════════════════════════════════════════════════════════════
#  FWHT-BASED STABILIZER RÉNYI ENTROPY
# ══════════════════════════════════════════════════════════════════
#
#  M₂(ψ) = −log₂[ (1/D) Σ_P |⟨ψ|P|ψ⟩|⁴ ]
#
#  Algorithm:
#    FOR x ∈ {0,...,D-1}:
#      C_x[j] = ψ_j* · ψ_{j⊕x}      (correlation)
#      F_x = FWHT(C_x)                  (O(N·2^N))
#      total += Σ_z |F_x[z]|⁴
#    M₂ = −log₂(total / D)
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
      by (-1)^{z·j}, which is exactly the z-th component of the
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

      The sum over 4^N Pauli strings P is decomposed into:
        • Outer loop over x ∈ {0,...,D-1}  (X-part of Pauli label)
        • For each x: build correlation C_x[j] = ψ_j* · ψ_{j⊕x}
        • Apply FWHT to get F_x[z] = Σ_j (-1)^{z·j} C_x[j]
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
        j_xor_x = jnp.bitwise_xor(j_indices, x)
        C_x = jnp.conj(psi) * psi[j_xor_x]
        F_x = _fwht(C_x, n_qubits)
        total = total + jnp.sum(jnp.abs(F_x)**4)

    return jnp.where(total > 0, -jnp.log2(total / D), jnp.inf)


# ══════════════════════════════════════════════════════════════════
#  DATA GENERATION — M₂ vs evolution time
# ══════════════════════════════════════════════════════════════════

def generate_data(regenerate=False):
    """Generate SRE data: M₂ vs time for each N."""
    t0 = timer.time()

    time_points = np.arange(0, TIME_MAX + 1e-9, TIME_STEP)

    for N in N_VALUES:
        outfile = DATA_DIR / f"sre_analog_N{N}.npz"
        if outfile.exists() and not regenerate:
            print(f"  Cached: {outfile}")
            continue

        D = 1 << N
        n_times = len(time_points)
        sre_vals = np.full(n_times, np.nan)

        # Random transverse fields (deterministic per N)
        rng = np.random.default_rng(SEED + N)
        h_fields = rng.uniform(-1, 1, size=N)

        print(f"\n  N = {N} (D = {D}):  scanning M₂ at {n_times} time points")
        print(f"    Trotter dt = {DT_TROTTER}, field h ~ U[-1,1]")

        # Precompute gates (reused across times)
        subs_1q = _make_1q_subs(N)
        subs_2q = _make_2q_subs(N)
        rx_gates = _build_rx_gates(h_fields, DT_TROTTER)
        xx_gate  = _build_xx_gate(DT_TROTTER)
        even = [(j, j+1) for j in range(0, N-1, 2)]
        odd  = [(j, j+1) for j in range(1, N-1, 2)]

        # Initial state: |0⟩^⊗N
        psi = jnp.zeros(D, dtype=CDT).at[0].set(1.0 + 0j)
        steps_done = 0

        for ti, t in enumerate(time_points):
            t1 = timer.time()

            # Evolve from previous time point to this one
            steps_to_do = round(t / DT_TROTTER) - steps_done
            for _ in range(steps_to_do):
                psi = evolve_one_step(psi, N, rx_gates, xx_gate,
                                     subs_1q, subs_2q, even, odd)
            steps_done += steps_to_do

            # Compute SRE
            sre_vals[ti] = float(compute_sre(psi, N))
            dt_wall = timer.time() - t1
            print(f"    t={t:5.1f} (steps={steps_done:4d}):  "
                  f"M₂ = {sre_vals[ti]:.3f}  ({dt_wall:.1f}s)")

            # Incremental save
            np.savez(outfile, times=time_points, sre=sre_vals,
                     N=N, dt=DT_TROTTER, h_fields=h_fields)

        print(f"    → Saved: {outfile}")

    print(f"\n  Total: {timer.time()-t0:.1f}s")


# ══════════════════════════════════════════════════════════════════
#  PLOTTING — M₂ vs evolution time
# ══════════════════════════════════════════════════════════════════

def plot_sre():
    """Single-panel SRE figure: M₂ vs evolution time t."""
    print("  Plotting SRE vs time figure...")

    colors = ["#2C7FB8", "#7FBC41", "#FD8D3C", "#9E4A9C"]
    markers = ["o", "^", "s", "D"]

    fig, ax = plt.subplots(figsize=(7, 5))

    for i, N in enumerate(N_VALUES):
        infile = DATA_DIR / f"sre_analog_N{N}.npz"
        if not infile.exists():
            print(f"  WARNING: {infile} not found, skipping N={N}")
            continue

        data = np.load(infile)
        times = data["times"]
        sre = data["sre"]

        ax.plot(times, sre, f"-{markers[i]}", color=colors[i], lw=2,
                markersize=6, markeredgecolor="white", markeredgewidth=0.5,
                label=f"$N = {N}$")

        # Haar reference: M₂ ≈ N − 2
        ax.axhline(N - 2, color=colors[i], ls=":", lw=0.8, alpha=0.4)

    ax.set_xlabel(r"Evolution time $t$")
    ax.set_ylabel(r"$M_2$ (nonstabilizerness)")
    ax.set_xlim(-0.2, TIME_MAX + 0.2)
    ax.set_ylim(-0.5, max(N_VALUES) - 1)
    ax.legend(loc="lower right", framealpha=0.9, fontsize=11)

    outpath = OUTPUT_DIR / "fig_ch5_sre_vs_time.pdf"
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
        description="Fig: SRE M₂ vs time (Trotterized XX Hamiltonian)")
    parser.add_argument("--regenerate", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()

    print("=" * 65)
    print("  Chapter 5 — SRE (Magic) vs Time: Analog XX Model")
    print("  Trotterized evolution + FWHT SRE computation")
    print("=" * 65)
    print(f"  System sizes: {N_VALUES}")
    print(f"  Time range:   0..{TIME_MAX} (step {TIME_STEP})")
    print(f"  Trotter dt:   {DT_TROTTER}")
    print(f"  JAX backend:  {jax.default_backend()}")
    print("=" * 65)

    if args.plot_only:
        missing = [N for N in N_VALUES
                   if not (DATA_DIR / f"sre_analog_N{N}.npz").exists()]
        if missing:
            print(f"ERROR: missing data for N = {missing}")
            sys.exit(1)
        print("  Using cached data")
    else:
        generate_data(regenerate=args.regenerate)

    print("\n── Generating figure ──")
    plot_sre()
    print(f"\nDone. Output → {OUTPUT_DIR}")
