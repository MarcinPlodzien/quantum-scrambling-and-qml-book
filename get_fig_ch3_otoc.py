#!/usr/bin/env python3
"""
===============================================================================
QUANTUM CHAOS AND THE BUTTERFLY EFFECT: OUT-OF-TIME-ORDER CORRELATORS (OTOC)
===============================================================================

Produces the OTOC figure for Chapter 3, contrasting chaotic and integrable
spin-chain dynamics:

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

    C(t) ≡ (1 / 2D) Tr( [Ŵ(t), V̂]† [Ŵ(t), V̂] )     ≥ 0         ...(*)

where D = 2^N is the Hilbert space dimension and the trace gives the
infinite-temperature (β=0) average ⟨·⟩_{β=0} = Tr(·)/D.  The factor 1/2
is conventional so that C → 1 for Haar-random W(t).

Connection to F(t):  The OTOC is often written as
    F(t) = ⟨Ŵ†(t) V̂† Ŵ(t) V̂⟩.
For unitary operators at β=0, C(t) = 2(1 − Re F(t)).  F starts at 1
(operators commute) and decays as scrambling proceeds.

===============================================================================
2. ANATOMY OF THE SQUARED COMMUTATOR
===============================================================================

t = 0:   Ŵ(0) = Z₀ is localized at site 0; V̂ = Z_{N/2} at site N/2.
         They act on disjoint spin subsets → [Ŵ(0), V̂] = 0 → C(0) = 0.

Operator spreading:  Under Heisenberg evolution Ŵ(t) = e^{iHt} Ŵ e^{-iHt},
         the interactions generate nested commutators [H,[H,...[H,Ŵ]...]],
         causing Ŵ(t) to spread across the spin chain.  The Lieb-Robinson
         bound constrains this to a linear light-cone:

             ‖[Ô_x(t), Ô_y]‖ ≤ c · e^{−(|x−y| − v_LR t)/ξ}

         so the support radius grows as r(t) ≲ v_LR · t.

The collision:  When Ŵ(t) reaches site N/2, the commutator [Ŵ(t), V̂] ≠ 0
         and C(t) begins to grow.  Onset time: t* ≈ (N/2) / v_B.

"Out of time order":  Expanding |[Ŵ(t), V̂]|² gives four-point
         correlators with time arguments that alternate (t, 0, t, 0)
         rather than the time-ordered pattern (t, t, 0, 0).

===============================================================================
3. CHAOS vs. INTEGRABILITY — PHYSICS OF THE TWO CURVES
===============================================================================

CHAOTIC (Mixed-Field Ising, h_z = 0.5):
    The longitudinal field h_z breaks the Z₂ symmetry of the transverse-field
    model and destroys all non-trivial conservation laws.  The energy spectrum
    exhibits Wigner-Dyson level repulsion (GOE statistics).

    Consequences for C(t):
    • Travel phase: C(t) ≈ 0 until W(t) reaches V (≈ N/(2v_B) time units).
    • Exponential growth: C(t) ~ ε · e^{2λ_L t} inside the butterfly cone.
      (In local spin chains, the clean exponential window may be short
      or masked by the ballistic front; the MSS bound λ_L ≤ 2π/β is
      unbounded at β = 0.)
    • Smooth saturation: C(t) → plateau ≈ 1 (Haar-random prediction).
      Local information is irreversibly scrambled — the hallmark of
      quantum thermalization.

INTEGRABLE (Transverse-Field Ising, h_z = 0):
    This model maps to free fermions via the Jordan-Wigner transformation,
    giving N conserved fermion occupation numbers.  The spectrum has
    Poissonian level statistics (no level repulsion).

    Consequences for C(t):
    • Travel phase: identical to the chaotic case — both models have
      similar Lieb-Robinson velocities.
    • NO smooth saturation: C(t) exhibits large quasi-periodic oscillations
      ("quantum recurrences").  The conserved charges prevent ergodic
      mixing; information bounces coherently between chain boundaries.
    • C(t) overshoots the Haar value (up to ~1.9) and shows persistent
      revivals even at very late times.

===============================================================================
4. THE COMMUTATOR SUBSPACE TRICK — FULL DERIVATION
===============================================================================

NAIVE COST:  Computing C(t) at each time step requires:
    (i)   W(t) = U e^{iΛt} U† W U e^{-iΛt} U†    → two D×D matmuls, O(D³)
    (ii)  comm = W(t) V − V W(t)                    → O(D²)
    (iii) C(t) = ‖comm‖²_F / (2D)                   → O(D²)
    Bottleneck: (i), O(D³) per time step, unfeasible for 400 steps.

KEY IDENTITY:  Since V = σ_z is diagonal with eigenvalues v_k ∈ {+1, −1}:

    [W(t), V]_{km} = W(t)_{km} · (v_m − v_k)

    This is ZERO when v_k = v_m, and ±2 · W(t)_{km} when v_k ≠ v_m.
    Partition the computational basis into V₊ = {k : v_k = +1} and
    V₋ = {m : v_m = −1}, each of size D/2:

    C(t) = (1/2D) Σ_{km} |[W(t),V]_{km}|²
         = (1/2D) · 4 · Σ_{k∈V₊, m∈V₋} |W(t)_{km}|²    (the +/− block)
         + (1/2D) · 4 · Σ_{k∈V₋, m∈V₊} |W(t)_{km}|²    (the −/+ block)

    Since W(t) is Hermitian, the (−/+) block is the conjugate transpose
    of the (+/−) block, so both contribute equally:

    C(t) = (4/D) · ‖W(t)_{+−}‖²_F

    where W(t)_{+−} is the (D/2 × D/2) cross-subspace block.

EIGENBASIS TIME EVOLUTION:  In the energy eigenbasis {|a⟩}:

    W(t)^{eig}_{ab} = W^{eig}_{ab} · e^{i(E_a − E_b)t}

Using Euler's formula to stay in real arithmetic (W_eig is real-symmetric):

    M_R = W^{eig} ⊙ cos(ΔE · t)     (element-wise Hadamard product)
    M_I = W^{eig} ⊙ sin(ΔE · t)

where ΔE_{ab} = E_a − E_b.  The cross-subspace block in the Z-basis is:

    block_R = U₊ · M_R · U₋ᵀ        (D/2 × D) · (D × D) · (D × D/2)
    block_I = U₊ · M_I · U₋ᵀ

and:

    C(t) = (4/D) · (‖block_R‖²_F + ‖block_I‖²_F)

    where ‖A‖²_F = Tr(AᵀA) = vdot(A, A) avoids forming the product.

HARDWARE OPTIMIZATIONS:
    • np.ascontiguousarray(U₊, U₋ᵀ, W_eig): ensures C-contiguous memory
      layout for optimal BLAS dgemm cache utilization.
    • Pre-allocated buffers (dE_t, M_R, M_I, temp_R, temp_I, block_R/I):
      eliminates per-step malloc/free overhead in the Python heap.
    • np.multiply/cos/sin with out= parameter: avoids temporary array
      allocation for element-wise operations.
    • np.vdot(A, A): computes ‖A‖²_F without forming AᵀA.
    • np.dot with out= parameter: writes BLAS dgemm output directly
      into pre-allocated buffers.

MEMORY LAYOUT (N=12, D=4096):
    • W_eig, dE, M_R, M_I:    D × D × 8 bytes = 128 MB each
    • U_plus, U_minus_T:       D/2 × D × 8 bytes = 64 MB each
    • temp_R, temp_I:          D/2 × D × 8 bytes = 64 MB each
    • block_R, block_I:        D/2 × D/2 × 8 bytes = 32 MB each
    • Total working set: ~850 MB

===============================================================================
5. WHY INFINITE TEMPERATURE (β = 0)?
===============================================================================

The trace Tr(·)/D is NOT a "thermal Gibbs state" in the usual sense.  It is
the maximally mixed state ρ = I/D, which weights ALL energy eigenstates
equally.  We use it because:

    (a) It isolates pure OPERATOR dynamics from state-dependent physics.
        No state preparation is needed — the diagnostic is entirely about
        how operators spread under the Hamiltonian.
    (b) It equals the Hilbert-Schmidt inner product ⟨A, B⟩ = Tr(A†B)/D,
        which is the same norm used in operator Krylov complexity
        (Chapter 3, Sec. 3.4).  This makes the comparison direct.
    (c) It is the standard convention in the scrambling literature
        (Shenker-Stanford, Maldacena-Shenker-Stanford, Sekino-Susskind).
    (d) It avoids computing thermal partition functions Z = Tr(e^{−βH}),
        which would add an expensive O(D) eigenvalue computation per
        temperature point.

===============================================================================
6. CHOICE OF OPERATORS AND HAMILTONIAN PARAMETERS
===============================================================================

Operators:  W = σ_z at site 0,  V = σ_z at site N/2.
    • Pauli Z operators are diagonal in the computational basis, which
      enables the commutator subspace trick (Section 4).
    • W and V are placed at maximum separation (N/2 = 6 sites apart) to
      give the clearest Lieb-Robinson travel time before scrambling onset.
    • Any single-qubit Pauli would work (X, Y, Z), but Z is simplest
      because it is real-diagonal (no complex arithmetic needed for V).

Parameters (Kim & Huse, Phys. Rev. Lett. 111, 2013):
    • h_x = 1.05: transverse field, chosen to be away from both the
      paramagnetic (h_x → ∞) and ferromagnetic (h_x → 0) limits.
    • h_z = 0.5: longitudinal field.  This breaks the Z₂ symmetry
      (σ_x^i → σ_x^i, σ_z^i → −σ_z^i) of the pure transverse-field model
      and destroys ALL integrability.  Setting h_z = 0 recovers the
      exactly solvable (integrable) transverse-field Ising model.

===============================================================================
7. THE SCRAMBLING TIME
===============================================================================

In systems with a clean exponential growth window C(t) ~ ε · e^{2λ_L t},
scrambling completes when C(t) ~ O(1), i.e., when the commutator becomes
order unity.  This gives the scrambling time:

    t_scr ~ (1 / 2λ_L) · ln(1/ε)

Since ε ~ 1/N for local operators in N-body systems, we get:

    t_scr ~ (1 / 2λ_L) · ln(N)

This LOGARITHMIC dependence on system size is remarkable — scrambling is
exponentially FAST.  Black holes are conjectured to be "fast scramblers"
that saturate this lower bound (Sekino & Susskind, 2008).

Note: In generic local spin chains at N=12, the clean exponential regime
may be short or absent.  What we robustly observe is: (a) ballistic
operator front with velocity v_B, and (b) smooth saturation (chaotic)
versus persistent recurrences (integrable).

===============================================================================
8. CONNECTION TO KRYLOV COMPLEXITY (THE TWO FIGURES IN CHAPTER 3)
===============================================================================

The Krylov complexity C_K(t) = Σ_n n |φ_n(t)|² measures the TOTAL operator
growth in the full Hilbert-Schmidt space, using the intrinsic Krylov basis.

The OTOC C(t) measures operator growth PROJECTED onto a specific direction
defined by the probe operator V̂.

Since a projection cannot exceed the total:

    λ_L ≤ 2α

where α is the asymptotic slope b_n ≈ α·n of the Lanczos coefficients
(from the Krylov figure, α ≈ 0.32 at N=14).  This is the Krylov-space
counterpart of the MSS bound λ_L ≤ 2π T.

The two figures in Chapter 3 tell a complementary story:
    • fig_ch3_krylov.pdf:  Intrinsic diagnostic (no choice of V needed)
    • fig_ch3_otoc.pdf:    Extrinsic diagnostic (depends on V), but
      directly probes the spatially-resolved spreading of operators.

===============================================================================
9. HAAR-RANDOM SATURATION VALUE
===============================================================================

For a Haar-random unitary U, the squared commutator saturates to:

    C_Haar = 2(1 − 1/D) ≈ 2    (for D ≫ 1)

with our normalization C(t) = (1/2D) · Tr(|[W(t),V]|²).  However, the
chaotic model saturates near C ≈ 1, not 2, because W and V are Hermitian
(not unitary) operators with Tr(W²) = D, ||W||² = 1.  The precise Haar
prediction for traceless Hermitian operators with ||W||² = ||V||² = 1 is:

    C_Haar = 2 · (1 − 1/D²) · (D² / (D²−1)) ≈ 2    (for W,V unitary)
    C_Haar ≈ 1                                        (for normalized σ_z)

The fact that the chaotic MFI saturates near this value is evidence of
quantum ergodicity: the time evolution "looks random" at late times.

===============================================================================
10. BITWISE HAMILTONIAN CONSTRUCTION — A COMPLETE GUIDE
===============================================================================

The Hamiltonian is built using bitwise operations on integer state labels,
avoiding expensive Kronecker products (⊗) entirely.  This section explains
the technique in full detail with worked examples, and shows how to adapt
it to construct ANY nearest-neighbor spin-1/2 Hamiltonian.

─── 10.1  THE STATE-INTEGER CORRESPONDENCE ───

Each computational basis state |s⟩ of N spin-1/2 particles is mapped to a
unique integer s ∈ {0, 1, ..., D−1} where D = 2^N:

    |s⟩ = |s_{N−1} s_{N−2} ... s_1 s_0⟩     (big-endian binary)

Convention:  s_i = 0 → spin-up (↑),  Z eigenvalue = +1
             s_i = 1 → spin-down (↓), Z eigenvalue = −1

WORKED EXAMPLE (N=3, D=8):
    s = 0 = 0b000 → |↑↑↑⟩       s = 4 = 0b100 → |↓↑↑⟩
    s = 1 = 0b001 → |↑↑↓⟩       s = 5 = 0b101 → |↓↑↓⟩
    s = 2 = 0b010 → |↑↓↑⟩       s = 6 = 0b110 → |↓↓↑⟩
    s = 3 = 0b011 → |↑↓↓⟩       s = 7 = 0b111 → |↓↓↓⟩

─── 10.2  EXTRACTING SPIN VALUES ───

To read the spin at site i from state s, use bit-shifting and masking:

    bit_i = (s >> (N − 1 − i)) & 1

The shift (N−1−i) accounts for big-endian ordering (site 0 = leftmost bit).
The mask "& 1" isolates just that one bit.

WORKED EXAMPLE (N=4, s = 10 = 0b1010):
    Site 0: (10 >> 3) & 1 = (0b0001) & 1 = 1 → spin-down  (↓)
    Site 1: (10 >> 2) & 1 = (0b0010) & 1 = 0 → spin-up    (↑)
    Site 2: (10 >> 1) & 1 = (0b0101) & 1 = 1 → spin-down  (↓)
    Site 3: (10 >> 0) & 1 = (0b1010) & 1 = 0 → spin-up    (↑)

VECTORIZATION:  NumPy processes ALL D states simultaneously:
    s = np.arange(D, dtype=int32)    # shape (D,)
    bit_i = (s >> (N-1-i)) & 1      # shape (D,), one bit per state

─── 10.3  SINGLE-SITE OPERATORS ───

Z_i (diagonal — does NOT change the state):
    Z_i |s⟩ = (−1)^{s_i} |s⟩ = (1 − 2·bit_i) · |s⟩

    Code:  diag += coupling * (1.0 - 2.0 * bit_i)

    bit_i = 0 (↑) → Z eigenvalue = +1
    bit_i = 1 (↓) → Z eigenvalue = −1

X_i (off-diagonal — FLIPS spin i):
    X_i |s⟩ = |s ⊕ 2^{N−1−i}⟩        (XOR flips the i-th bit)

    Code:  flipped = s ^ (1 << (N-1-i))
           H[s, flipped] += coupling

    Example (N=3, site 1):  mask = 1 << 1 = 0b010
        |↑↑↑⟩ (0b000) → 0b000 ⊕ 0b010 = 0b010 → |↑↓↑⟩  ✓
        |↓↑↓⟩ (0b101) → 0b101 ⊕ 0b010 = 0b111 → |↓↓↓⟩  ✓

Y_i (off-diagonal — FLIPS spin i with phase):
    Y_i = −i (|↑⟩⟨↓| − |↓⟩⟨↑|),  so Y_i|s⟩ = ±i · |s ⊕ 2^{N−1−i}⟩

    Code:  flipped = s ^ (1 << (N-1-i))
           sign = 1.0 - 2.0 * bit_i     # +1 if ↑→↓ (raising), −1 if ↓→↑
           H[s, flipped] += coupling * (1j * sign)   # note: complex!

─── 10.4  TWO-SITE INTERACTION OPERATORS ───

Z_i Z_{i+1} (diagonal — no spin flips):
    Z_i Z_{i+1} |s⟩ = (−1)^{s_i} · (−1)^{s_{i+1}} · |s⟩
                     = (−1)^{s_i ⊕ s_{i+1}} · |s⟩

    If spins are ALIGNED (same):      eigenvalue = +1
    If spins are ANTI-ALIGNED (diff): eigenvalue = −1

    Code:  diag += J * (1.0 - 2.0 * (bit_i ^ bit_j))

    The XOR (^) returns 1 when spins differ, 0 when same — exactly what
    we need!

X_i X_{i+1} (off-diagonal — flips BOTH spins i and i+1):
    X_i X_{i+1} |s⟩ = |s ⊕ 2^{N−1−i} ⊕ 2^{N−1−(i+1)}⟩

    Code:  mask = (1 << (N-1-i)) | (1 << (N-1-(i+1)))
           flipped = s ^ mask
           H[s, flipped] += coupling

    This flips EXACTLY two bits simultaneously using a combined mask.

Y_i Y_{i+1} (off-diagonal — flips BOTH spins with phase):
    Y_i Y_{i+1} |s⟩ = −(−1)^{s_i ⊕ s_{i+1}} · |s ⊕ mask⟩

    Code:  mask = (1 << (N-1-i)) | (1 << (N-1-(i+1)))
           flipped = s ^ mask
           phase = -(1.0 - 2.0 * (bit_i ^ bit_j))
           H[s, flipped] += coupling * phase

    Key insight: Y_i Y_{i+1} is REAL despite Y being imaginary, because
    the two factors of i cancel: (i)·(i) = −1, and the sign structure
    from Y = −i(|↑⟩⟨↓| − |↓⟩⟨↑|) gives a real result.

─── 10.5  RECIPE BOOK: COMMON HAMILTONIANS ───

All nearest-neighbor spin-1/2 Hamiltonians can be built from the above
building blocks.  Here are the most common ones:

TRANSVERSE-FIELD ISING (TFIM) — integrable, maps to free fermions:
    H = J Σ Z_i Z_{i+1} − h Σ X_i
    Diagonal:      J · (1 − 2·(bit_i ⊕ bit_j))       [ZZ]
    Off-diagonal:  −h at positions (s, s ⊕ flip_mask)  [X]

XXZ MODEL — integrable via Bethe ansatz:
    H = J_xy (Σ X_i X_{i+1} + Y_i Y_{i+1}) + J_z Σ Z_i Z_{i+1}
    Diagonal:      J_z · (1 − 2·(bit_i ⊕ bit_j))     [ZZ]
    Off-diagonal:  J_xy at (s, s ⊕ 2-bit-flip)        [XX + YY]

    KEY SIMPLIFICATION:  X_i X_{i+1} + Y_i Y_{i+1} = 2(S⁺_i S⁻_{i+1} + S⁻_i S⁺_{i+1})
    This is a SPIN-EXCHANGE operator: it flips ↑↓ → ↓↑ and ↓↑ → ↑↓,
    but does NOTHING to ↑↑ or ↓↓.  In bitwise terms:

    Code:  mask = (1 << (N-1-i)) | (1 << (N-1-(i+1)))
           flipped = s ^ mask
           # Only connect states where bits i and i+1 DIFFER:
           active = (bit_i != bit_j)
           H[s[active], flipped[active]] += 2 * J_xy

HEISENBERG MODEL (J_xy = J_z = J, special case of XXZ):
    H = J Σ (X_i X_{i+1} + Y_i Y_{i+1} + Z_i Z_{i+1})
    = J Σ (2 S⁺_i S⁻_{i+1} + 2 S⁻_i S⁺_{i+1} + Z_i Z_{i+1})

MIXED-FIELD ISING (this code, chaotic):
    H = Σ Z_i Z_{i+1} − 1.05 Σ X_i + 0.5 Σ Z_i
    Diagonal:      (1 − 2·(bit_i ⊕ bit_j)) + 0.5·(1 − 2·bit_i)
    Off-diagonal:  −1.05 at (s, s ⊕ flip_mask)

─── 10.6  COMPLEXITY COMPARISON ───

KRONECKER PRODUCT METHOD (textbook):
    Build each term as I ⊗ ... ⊗ σ_α ⊗ ... ⊗ I using np.kron().
    Cost: O(N · D²) = O(N · 4^N) — each kron() produces a D×D matrix.
    For N=12: D=4096, each kron chain cost = 4096² = 16M operations.
    Total: 12 × 16M = 192M operations.  PLUS memory for intermediate
    Kronecker products.

BITWISE METHOD (this code):
    Loop over N sites, compute diagonal/off-diagonal in O(D) each.
    Cost: O(N · D) = O(N · 2^N).
    For N=12: 12 × 4096 = 49K operations.  About 4000× FASTER.
    Memory: only the final D×D matrix + O(D) temporary arrays.

This speedup is the difference between "builds in milliseconds" and
"builds in minutes" for N=14 (D=16384).

This builds the entire D×D Hamiltonian in O(N·D) time with O(D·(N+1))
non-zero entries, compared to O(N·D²) for the Kronecker product approach.

===============================================================================
FILES
===============================================================================
    Caches data to:   data/ch3/otoc_data_t15.npz
    Saves figure to:  figures/ch3/fig_ch3_otoc.pdf  (+.png at 300 dpi)
===============================================================================
"""

import numpy as np
import scipy.linalg as la
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# =============================================================================
# ─── Safe Pathing ────────────────────────────────────────────────────────────
# =============================================================================
try:
    ROOT = Path(__file__).parent
except NameError:
    ROOT = Path.cwd()

DATA_DIR = ROOT / "data" / "ch3"
FIG_DIR  = ROOT.parent / "figures" / "ch3" 

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
# ─── Parameters ──────────────────────────────────────────────────────────────
# =============================================================================
N_SPIN  = 10       # Hilbert space size D = 2^10 = 1024 states
N_TIMES = 800      # Time resolution (higher for T=40)
T_MAX   = 50.0     # Long time to capture integrable revivals / bouncing

# =============================================================================
# ─── Dense Hamiltonian builder ───────────────────────────────────────────────
# =============================================================================
def build_ising_hamiltonian(N, h_x=1.05, h_z=0.5):
    """
    Constructs the Hamiltonian using bitwise operations (see Section 10):
    H = Sum(Z_i Z_{i+1}) - h_x * Sum(X_i) + h_z * Sum(Z_i)
    """
    D = 2**N
    s = np.arange(D, dtype=np.int32)

    diag = np.zeros(D, dtype=np.float64)
    # 1. ZZ Interactions (Diagonal): eigenvalue = 1 - 2*(bit_i XOR bit_j)
    for i in range(N - 1):
        bi = (s >> (N - 1 - i)) & 1
        bj = (s >> (N - 1 - (i + 1))) & 1
        diag += 1.0 - 2.0 * (bi ^ bj) 
        
    # 2. Longitudinal Z Field (Diagonal): eigenvalue = h_z * (1 - 2*bit_i)
    if h_z != 0:
        for i in range(N):
            bi = (s >> (N - 1 - i)) & 1
            diag += h_z * (1.0 - 2.0 * bi)

    H = np.diag(diag)

    # 3. Transverse X Field (Off-Diagonal): flips spin i via XOR
    for i in range(N):
        flipped = s ^ (1 << (N - 1 - i))
        H[s, flipped] -= h_x

    return H

# =============================================================================
# ─── Spatially-Resolved OTOC Heatmap ────────────────────────────────────────
# =============================================================================
def compute_otoc_heatmap(H_dense, w_diag, N, times):
    """
    Compute C(t, r) for ALL probe sites r = 1, ..., N-1.

    The key optimization: we diagonalize H only ONCE.  The eigensystem
    (evals, U) and the energy-basis W matrix (W_eig, dE) are SHARED
    across all probe sites r.  Only the V-subspace projection (U_plus,
    U_minus_T) changes for each r.

    Returns:
        otoc_map : ndarray of shape (len(times), N-1)
            otoc_map[:, r-1] = C(t, r) for probe site r.
    """
    D = H_dense.shape[0]
    s = np.arange(D, dtype=np.int32)

    # 1. DIAGONALIZE ONCE (shared for all r)
    evals, U = la.eigh(H_dense)
    print(f"  Eigenvalue range: [{evals[0]:.2f}, {evals[-1]:.2f}]")

    # 2. W IN ENERGY BASIS (shared for all r)
    W_eig = np.ascontiguousarray(U.T @ (w_diag[:, None] * U))
    dE = evals[:, None] - evals[None, :]

    # 3. PRE-ALLOCATE SHARED BUFFERS (reused across sites and time steps)
    dE_t = np.empty_like(dE)
    M_R  = np.empty_like(W_eig)
    M_I  = np.empty_like(W_eig)

    otoc_map = np.zeros((len(times), N - 1))

    # 4. LOOP OVER PROBE SITES r = 1, ..., N-1
    for r in range(1, N):
        # V = sigma_z at site r (diagonal, eigenvalues ±1)
        v_diag = (1.0 - 2.0 * ((s >> (N - 1 - r)) & 1)).astype(np.float64)

        # V-subspace projection for THIS site
        idx_plus  = (v_diag == 1.0)
        idx_minus = (v_diag == -1.0)
        U_plus    = np.ascontiguousarray(U[idx_plus, :])
        U_minus_T = np.ascontiguousarray(U[idx_minus, :].T)

        # Per-site buffers
        temp_R  = np.empty((U_plus.shape[0], W_eig.shape[1]), dtype=np.float64)
        temp_I  = np.empty((U_plus.shape[0], W_eig.shape[1]), dtype=np.float64)
        block_R = np.empty((U_plus.shape[0], U_minus_T.shape[1]), dtype=np.float64)
        block_I = np.empty((U_plus.shape[0], U_minus_T.shape[1]), dtype=np.float64)

        # 5. TIME LOOP for this site
        for ti, t in enumerate(times):
            np.multiply(dE, t, out=dE_t)

            np.cos(dE_t, out=M_R)
            M_R *= W_eig
            np.dot(U_plus, M_R, out=temp_R)
            np.dot(temp_R, U_minus_T, out=block_R)

            np.sin(dE_t, out=M_I)
            M_I *= W_eig
            np.dot(U_plus, M_I, out=temp_I)
            np.dot(temp_I, U_minus_T, out=block_I)

            # C(t) = Tr([W(t),V]†[W(t),V]) / D = 8 ‖block‖²_F / D
            val_R = np.vdot(block_R, block_R)
            val_I = np.vdot(block_I, block_I)
            otoc_map[ti, r - 1] = 8.0 * (val_R + val_I) / D

        print(f"    site r={r}/{N-1} done", flush=True)

    return otoc_map

# =============================================================================
# ─── Main ────────────────────────────────────────────────────────────────────
# =============================================================================
def main():
    cache_path = DATA_DIR / f"otoc_heatmap_N{N_SPIN}_t{int(T_MAX)}.npz"
    D = 2**N_SPIN
    s = np.arange(D, dtype=np.int32)

    # W = sigma_z at site 0
    site_W = 0
    w_diag = (1.0 - 2.0 * ((s >> (N_SPIN - 1 - site_W)) & 1)).astype(np.float64)
    times  = np.linspace(0, T_MAX, N_TIMES)

    # Also check for JAX-generated cache files (different naming convention)
    jax_cache_path = DATA_DIR / f"otoc_data_N{N_SPIN}_normal_MFI_hx1.05_hz0.5_TFI_hx1.05_hz0.0.npz"

    if cache_path.exists():
        print(f"Loading cached OTOC heatmap from {cache_path}")
        data = np.load(cache_path)
        times           = data["times"]
        map_chaotic     = data["map_chaotic"]
        map_integrable  = data["map_integrable"]
        N_cached = int(data["N_spin"]) if "N_spin" in data else N_SPIN
        if N_cached != N_SPIN:
            print(f"  Cache N={N_cached} != {N_SPIN}. Re-running.")
            cache_path.unlink()
            return main()
    elif jax_cache_path.exists():
        print(f"Loading JAX-generated OTOC data from {jax_cache_path}")
        data = np.load(jax_cache_path)
        times           = data["times"]
        map_chaotic     = data["map_chaotic"]
        map_integrable  = data["map_integrable"]
    else:
        print(f"\n=== Chaotic (Mixed-Field Ising, h_z=0.5) ===")
        H_chaotic = build_ising_hamiltonian(N_SPIN, h_x=1.05, h_z=0.5)
        map_chaotic = compute_otoc_heatmap(H_chaotic, w_diag, N_SPIN, times)
        del H_chaotic

        print(f"\n=== Integrable (Transverse-Field Ising, h_z=0) ===")
        H_integrable = build_ising_hamiltonian(N_SPIN, h_x=1.05, h_z=0.0)
        map_integrable = compute_otoc_heatmap(H_integrable, w_diag, N_SPIN, times)
        del H_integrable

        np.savez(cache_path, times=times,
                 map_chaotic=map_chaotic,
                 map_integrable=map_integrable,
                 N_spin=N_SPIN)
        print(f"\nSaved: {cache_path}")

    # =========================================================================
    # PLOT — Three-panel: (a) chaotic heatmap, (b) integrable heatmap, (c) line
    # =========================================================================
    fig, (ax_a, ax_b, ax_c) = plt.subplots(
        1, 3, figsize=(12.0, 3.8), constrained_layout=True,
        gridspec_kw={"width_ratios": [1, 1, 1.15]}
    )

    site_labels = np.arange(1, N_SPIN)
    vmax = max(map_chaotic.max(), map_integrable.max())

    # ── Panel (a): Chaotic heatmap ────────────────────────────────────────
    im_a = ax_a.pcolormesh(times, site_labels, map_chaotic.T,
                           shading="auto", cmap="inferno",
                           vmin=0, vmax=vmax)
    ax_a.set_xlabel(r"Time $t$")
    ax_a.set_ylabel(r"Site $j$")
    ax_a.text(0.04, 0.96, r"$\mathbf{(a)}$  Chaotic (MFI)",
              transform=ax_a.transAxes, fontsize=10, va="top", ha="left",
              color="white", fontweight="bold")
    ax_a.set_yticks(site_labels)

    # ── Panel (b): Integrable heatmap ─────────────────────────────────────
    im_b = ax_b.pcolormesh(times, site_labels, map_integrable.T,
                           shading="auto", cmap="inferno",
                           vmin=0, vmax=vmax)
    ax_b.set_xlabel(r"Time $t$")
    ax_b.set_ylabel(r"Site $j$")
    ax_b.text(0.04, 0.96, r"$\mathbf{(b)}$  Integrable (TFI)",
              transform=ax_b.transAxes, fontsize=10, va="top", ha="left",
              color="white", fontweight="bold")
    ax_b.set_yticks(site_labels)

    # Shared colorbar between panels (a) and (b)
    cbar = fig.colorbar(im_b, ax=[ax_a, ax_b], shrink=0.85,
                        pad=0.02, label=r"$C_{0j}(t)$")

    # ── Panel (c): Line cuts at r = N/2 ──────────────────────────────────
    r_cut = N_SPIN // 2
    ax_c.plot(times, map_chaotic[:, r_cut - 1],
              color="#d7191c", lw=1.8, label="Chaotic (MFI)")
    ax_c.plot(times, map_integrable[:, r_cut - 1],
              color="#2c7bb6", lw=1.8, ls="--", label="Integrable (TFI)")
    ax_c.axhline(2.0, color="#888888", ls=":", lw=0.9, alpha=0.7,
                 label="Haar plateau")
    ax_c.set_xlabel(r"Time $t$")
    ax_c.set_ylabel(r"$C_{0j}(t)$  at  $j = N/2$")
    ax_c.set_xlim(0, T_MAX)
    ax_c.set_ylim(0, max(map_integrable[:, r_cut - 1].max(),
                         map_chaotic[:, r_cut - 1].max()) * 1.1)
    ax_c.legend(framealpha=0.85, loc="upper right", fontsize=8)
    ax_c.text(0.04, 0.96, r"$\mathbf{(c)}$",
              transform=ax_c.transAxes, fontsize=11, va="top", ha="left")

    out_path = FIG_DIR / "fig_ch3_otoc.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(out_path.with_suffix('.png'), bbox_inches="tight", dpi=300)
    print(f"\nSaved: {out_path}")
    plt.close(fig)

if __name__ == "__main__":
    main()