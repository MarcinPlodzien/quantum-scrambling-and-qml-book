#!/usr/bin/env python3
"""
###########################################################################
###########################################################################
##
##  SOLUTION: The Adjoint Representation and DLA Closure
##
##  Book:     Information Scrambling and the Haar Measure
##  Chapter:  1 -- Mathematical Foundations
##  Section:  1.4 -- Lie Algebras and the Exponential Map
##
###########################################################################
##
##  EXERCISE STATEMENT
##  ------------------
##
##  Consider the 2-qubit transverse-field Ising generators:
##    H₁ = σ_z ⊗ I,   H₂ = I ⊗ σ_z,   H₃ = σ_x ⊗ σ_x
##
##  (a) Compute the commutator [-iH₁, -iH₃] and express it in terms
##      of Pauli tensor products.
##
##  (b) Starting from {-iH₁, -iH₂, -iH₃}, iterate all nested
##      commutators until the algebra closes.  Show dim(DLA) = 6.
##
##  (c) Verify that σ_z ⊗ σ_z commutes with all three generators.
##      This Z₂ parity symmetry restricts the DLA to a proper
##      subalgebra of su(4).
##
###########################################################################
##
##  ANALYTICAL SOLUTION
##  -------------------
##
##  (a) [-iH₁, -iH₃] = -[H₁, H₃]  (since (-i)² = -1)
##      [H₁, H₃] = [σ_z⊗I, σ_x⊗σ_x] = [σ_z, σ_x] ⊗ σ_x = 2iσ_y ⊗ σ_x
##      Therefore [-iH₁, -iH₃] = -2i σ_y ⊗ σ_x.
##
##  (b) The DLA closes at dimension 6.  The six independent generators
##      span the Lie algebra of the symmetry-restricted subgroup.
##      Since dim su(4) = 15, the circuit cannot explore the full
##      unitary group — only a 6-parameter submanifold.
##
##  (c) The parity operator P = σ_z ⊗ σ_z satisfies [P, H_j] = 0
##      for all generators.  By Schur's lemma, this conserved quantity
##      decomposes the Hilbert space into parity sectors, preventing
##      the DLA from reaching su(4).  The gradient variance scales
##      as Ω(1/6) rather than Ω(1/15): symmetry mitigates barren
##      plateaus.
##
###########################################################################
###########################################################################
"""
import sympy as sp
import numpy as np
from numpy import kron

# ========================================================================
#  SECTION 1 :  Symbolic derivation with SymPy
# ========================================================================

print("=" * 70)
print("  Symbolic derivation of the DLA for the transverse-field Ising model")
print("=" * 70)

# --- Define Pauli matrices symbolically ---
I2s = sp.eye(2)
sx_s = sp.Matrix([[0, 1], [1, 0]])
sy_s = sp.Matrix([[0, -sp.I], [sp.I, 0]])
sz_s = sp.Matrix([[1, 0], [0, -1]])

# --- Step 1: Verify the fundamental Pauli commutation [σ_z, σ_x] = 2iσ_y ---
print("\n  Step 1: Pauli commutation relation [σ_z, σ_x] = 2i σ_y")
print("  " + "-" * 50)

comm_zx = sz_s * sx_s - sx_s * sz_s
expected_comm = 2 * sp.I * sy_s
diff = sp.simplify(comm_zx - expected_comm)
print(f"    [σ_z, σ_x] = {comm_zx.tolist()}")
print(f"    2i σ_y      = {expected_comm.tolist()}")
print(f"    Match: {diff.equals(sp.zeros(2))}")
assert diff.equals(sp.zeros(2)), "[σ_z, σ_x] ≠ 2i σ_y"

# --- Step 2: Compute [-iH₁, -iH₃] symbolically ---
print("\n  Step 2: Commutator [-iH₁, -iH₃] = -2i σ_y ⊗ σ_x")
print("  " + "-" * 50)
print("    H₁ = σ_z ⊗ I,  H₃ = σ_x ⊗ σ_x")
print("    [-iH₁, -iH₃] = (-i)²[H₁, H₃] = -[H₁, H₃]")
print("    [H₁, H₃] = [σ_z, σ_x] ⊗ [I·σ_x] = 2iσ_y ⊗ σ_x")
print("    Therefore [-iH₁, -iH₃] = -2i σ_y ⊗ σ_x")

H1_s = sp.kronecker_product(sz_s, I2s)
H3_s = sp.kronecker_product(sx_s, sx_s)
comm_13 = (-sp.I * H1_s) * (-sp.I * H3_s) - (-sp.I * H3_s) * (-sp.I * H1_s)
target = -2 * sp.I * sp.kronecker_product(sy_s, sx_s)
diff13 = sp.simplify(comm_13 - target)
print(f"    Symbolic verification: {diff13.equals(sp.zeros(4))}")
assert diff13.equals(sp.zeros(4)), "[-iH₁, -iH₃] ≠ -2i σ_y⊗σ_x"

# --- Step 3: Verify Z₂ symmetry [σ_z⊗σ_z, H_j] = 0 ---
print("\n  Step 3: Z₂ parity symmetry — [σ_z⊗σ_z, H_j] = 0")
print("  " + "-" * 50)

H2_s = sp.kronecker_product(I2s, sz_s)
ZZ_s = sp.kronecker_product(sz_s, sz_s)

for name, H in [("H₁ = σ_z⊗I", H1_s), ("H₂ = I⊗σ_z", H2_s),
                ("H₃ = σ_x⊗σ_x", H3_s)]:
    comm = sp.simplify(ZZ_s * H - H * ZZ_s)
    print(f"    [σ_z⊗σ_z, {name}] = 0 : {comm.equals(sp.zeros(4))}")
    assert comm.equals(sp.zeros(4)), f"[ZZ, {name}] ≠ 0"

print("    The conserved parity operator restricts the DLA to a")
print("    proper subalgebra of su(4).")

# ========================================================================
#  SECTION 2 :  Numerical verification with NumPy
# ========================================================================

print("\n" + "=" * 70)
print("  Numerical verification: iterative DLA closure")
print("=" * 70)

I2 = np.eye(2, dtype=complex)
sx = np.array([[0, 1], [1, 0]], dtype=complex)
sy = np.array([[0, -1j], [1j, 0]])
sz = np.array([[1, 0], [0, -1]], dtype=complex)

H1 = kron(sz, I2)
H2 = kron(I2, sz)
H3 = kron(sx, sx)

def is_linearly_independent(op, algebra, tol=1e-10):
    """Check if op is linearly independent from the current algebra basis."""
    if np.linalg.norm(op) < tol:
        return False
    M = np.array([a.flatten() for a in algebra])
    r_before = np.linalg.matrix_rank(M, tol=tol)
    M_aug = np.vstack([M, op.flatten()])
    r_after = np.linalg.matrix_rank(M_aug, tol=tol)
    return r_after > r_before

algebra = [-1j * H1, -1j * H2, -1j * H3]

for iteration in range(20):
    n = len(algebra)
    changed = False
    for i in range(n):
        for j in range(i + 1, n):
            comm = algebra[i] @ algebra[j] - algebra[j] @ algebra[i]
            if is_linearly_independent(comm, algebra):
                algebra.append(comm)
                changed = True
    if not changed:
        break

dim_DLA = len(algebra)
print(f"\n  DLA closed after {iteration + 1} iterations.")
print(f"  dim(DLA) = {dim_DLA}  (expected 6)")
print(f"  dim su(4) = 15")
print(f"  Ratio: {dim_DLA}/15 = {dim_DLA/15:.2f}")
assert dim_DLA == 6, f"DLA dimension = {dim_DLA}, expected 6"

# Verify the commutator [-iH₁, -iH₃] = -2i σ_y ⊗ σ_x
comm_13_np = (-1j*H1) @ (-1j*H3) - (-1j*H3) @ (-1j*H1)
assert np.allclose(comm_13_np, -2j * kron(sy, sx))
print(f"\n  [-iH₁, -iH₃] = -2i σ_y⊗σ_x  ✓")

# Verify Z₂ symmetry numerically
ZZ = kron(sz, sz)
for name, gen in [("H₁", H1), ("H₂", H2), ("H₃", H3)]:
    comm_check = ZZ @ gen - gen @ ZZ
    assert np.allclose(comm_check, 0)
    print(f"  [σ_z⊗σ_z, {name}] = 0  ✓")

print(f"\n  The Z₂ symmetry constrains the DLA to dim = 6 ⊂ su(4).")
print(f"  Gradient variance scales as Ω(1/6) instead of Ω(1/15).")
