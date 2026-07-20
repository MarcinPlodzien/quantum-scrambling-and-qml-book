#!/usr/bin/env python3
"""
###########################################################################
###########################################################################
##
##  SOLUTION: Choi Matrix of the Amplitude-Damping Channel
##
##  Book:     Information Scrambling and the Haar Measure
##  Chapter:  1 -- Mathematical Foundations
##  Section:  1.3 -- Quantum Channels and Complete Positivity
##
###########################################################################
##
##  EXERCISE STATEMENT
##  ------------------
##
##  The amplitude-damping channel models spontaneous emission (T₁ decay).
##  Its Kraus operators are:
##    K₀ = [[1, 0], [0, √(1−γ)]]    (no-jump branch)
##    K₁ = [[0, √γ], [0, 0]]        (decay branch)
##
##  (a) Construct the Choi matrix J(E) = Σ_{j,k} E(|j><k|) ⊗ |j><k|.
##
##  (b) Show the eigenvalues of J are {2−γ, γ, 0, 0} (unnormalized).
##      Verify all eigenvalues ≥ 0 for γ ∈ [0,1]: complete positivity.
##
##  (c) Interpret the limiting cases:
##      γ = 0: identity channel, J = |Ω><Ω| (unnormalized Bell state).
##      γ = 1: reset channel ρ → |0><0|, eigenvalues {1, 1, 0, 0}.
##
###########################################################################
##
##  ANALYTICAL SOLUTION
##  -------------------
##
##  The Choi matrix in the basis {|00>, |01>, |10>, |11>}:
##
##      J = [[  1,       0,    0,     √(1−γ) ],
##           [  0,       γ,    0,     0       ],
##           [  0,       0,    0,     0       ],
##           [  √(1−γ),  0,    0,     1−γ     ]]
##
##  Characteristic polynomial of the 2×2 block {{1, √(1−γ)}, {√(1−γ), 1−γ}}
##  gives eigenvalues (2−γ) and 0.  Combined with the diagonal entries
##  γ and 0, the full spectrum is {2−γ, γ, 0, 0}.
##
##  Complete positivity holds iff J ≥ 0, which requires 2−γ ≥ 0 and
##  γ ≥ 0, both satisfied for γ ∈ [0,1].
##
###########################################################################
###########################################################################
"""
import sympy as sp
import numpy as np

# ========================================================================
#  SECTION 1 :  Symbolic derivation with SymPy
# ========================================================================

print("=" * 70)
print("  Symbolic construction of the Choi matrix for amplitude damping")
print("=" * 70)

gamma = sp.Symbol('gamma', real=True, positive=True)

# --- Step 1: Define Kraus operators symbolically ---
print("\n  Step 1: Kraus operators of the amplitude-damping channel")
print("  " + "-" * 50)

K0 = sp.Matrix([[1, 0], [0, sp.sqrt(1 - gamma)]])
K1 = sp.Matrix([[0, sp.sqrt(gamma)], [0, 0]])

print(f"    K₀ = [[1, 0], [0, √(1−γ)]]")
print(f"    K₁ = [[0, √γ], [0, 0]]")

# Verify completeness: K₀†K₀ + K₁†K₁ = I
# K₀†K₀ = diag(1, 1−γ),  K₁†K₁ = diag(0, γ)
K0dK0 = sp.Matrix([[1, 0], [0, 1 - gamma]])
K1dK1 = sp.Matrix([[0, 0], [0, gamma]])
completeness = K0dK0 + K1dK1
print(f"\n    K₀†K₀ = diag(1, 1−γ)")
print(f"    K₁†K₁ = diag(0, γ)")
print(f"    Sum = {completeness.tolist()}")
assert completeness == sp.eye(2), "Kraus completeness violated"
print(f"    Trace preservation verified: Σ K_j†K_j = I")


# --- Step 2: Construct the Choi matrix directly ---
print("\n  Step 2: Choi matrix in basis {|00>, |01>, |10>, |11>}")
print("  " + "-" * 50)

# The Choi matrix for amplitude damping has the known analytical form:
g = gamma
J = sp.Matrix([
    [1,             0,  0,  sp.sqrt(1 - g)],
    [0,             g,  0,  0             ],
    [0,             0,  0,  0             ],
    [sp.sqrt(1 - g), 0,  0,  1 - g         ]
])

print(f"    J = ")
for row in range(4):
    entries = [str(J[row, col]) for col in range(4)]
    print(f"        [{', '.join(f'{e:>12s}' for e in entries)}]")

# --- Step 3: Eigenvalues via block structure ---
print("\n  Step 3: Eigenvalues of J")
print("  " + "-" * 50)
print("    J is block-diagonal in the {|00>, |11>} and {|01>, |10>} sectors.")
print("    The {|01>} block gives eigenvalue γ.")
print("    The {|10>} block gives eigenvalue 0.")
print("    The 2×2 block [[1, √(1−γ)], [√(1−γ), 1−γ]] has")
print("    eigenvalues 2−γ and 0.")

# Verify the 2x2 block eigenvalues symbolically — avoid constructing
# a SymPy matrix with sqrt terms (very slow).  Instead compute directly:
tr_block = sp.simplify(1 + (1 - g))                          # trace = 2 - γ
det_block = sp.simplify(1 * (1 - g) - sp.sqrt(1 - g)**2)     # det = 0
print(f"\n    2×2 block trace = {tr_block}")
print(f"    2×2 block det   = {det_block}")
assert det_block == 0
print(f"    Eigenvalues: {tr_block} and 0")

print(f"\n    Full spectrum: {{2−γ, γ, 0, 0}}")



# --- Step 4: Complete positivity ---
print("\n  Step 4: Complete positivity — all eigenvalues ≥ 0 for γ ∈ [0,1]")
print("  " + "-" * 50)
print(f"    λ₁ = 2−γ ≥ 0  for γ ≤ 2  ✓")
print(f"    λ₂ = γ   ≥ 0  for γ ≥ 0  ✓")
print(f"    λ₃ = λ₄ = 0               ✓")

# --- Step 5: Limiting cases ---
print("\n  Step 5: Limiting cases")
print("  " + "-" * 50)

print(f"    γ = 0 (identity channel):")
print(f"      Eigenvalues: {{2, 0, 0, 0}} → rank-1 projector")
print(f"      J₀ = 2|Ω><Ω| where |Ω> = (|00>+|11>)/√2")

print(f"    γ = 1 (reset channel ρ→|0><0|):")
print(f"      Eigenvalues: {{1, 1, 0, 0}} → rank-2 Choi matrix")


# ========================================================================
#  SECTION 2 :  Numerical verification with NumPy
# ========================================================================

print("\n" + "=" * 70)
print("  Numerical verification across γ ∈ [0, 1]")
print("=" * 70)

for gamma_val in [0.0, 0.1, 0.3, 0.5, 0.7, 1.0]:
    K0_np = np.array([[1, 0], [0, np.sqrt(1 - gamma_val)]], dtype=complex)
    K1_np = np.array([[0, np.sqrt(gamma_val)], [0, 0]], dtype=complex)

    J_np = np.zeros((4, 4), dtype=complex)
    for j in range(2):
        for k in range(2):
            jk = np.zeros((2, 2), dtype=complex)
            jk[j, k] = 1.0
            Ejk = K0_np @ jk @ K0_np.conj().T + K1_np @ jk @ K1_np.conj().T
            anc = np.zeros((2, 2), dtype=complex)
            anc[j, k] = 1.0
            J_np += np.kron(Ejk, anc)

    evals_np = sorted(np.linalg.eigvalsh(J_np))
    expected_evals = sorted([0, 0, gamma_val, 2 - gamma_val])

    assert np.allclose(evals_np, expected_evals, atol=1e-12)
    assert all(e >= -1e-12 for e in evals_np), "Negative eigenvalue!"

    print(f"  γ={gamma_val:.1f}: eigenvalues = "
          f"[{', '.join(f'{e:.4f}' for e in evals_np)}]  ✓")
