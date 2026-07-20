#!/usr/bin/env python3
"""
###########################################################################
###########################################################################
##
##  SOLUTION: Sectional Curvature of SU(2)
##
##  Book:     Information Scrambling and the Haar Measure
##  Chapter:  1 -- Mathematical Foundations
##  Section:  1.7 -- Riemannian Geometry of Unitary Groups
##
###########################################################################
##
##  EXERCISE STATEMENT
##  ------------------
##
##  For the bi-invariant Hilbert-Schmidt metric on U(D), the sectional
##  curvature of the plane spanned by orthonormal X, Y ∈ u(D) is:
##    K(X,Y) = (1/4) ‖[X,Y]‖²
##
##  (a) Verify that X = −iσ_x/√2 and Y = −iσ_y/√2 are orthonormal
##      in the HS metric:  g(A,B) = −Tr(AB).
##
##  (b) Compute [X,Y], ‖[X,Y]‖², and K(X,Y).
##
##  (c) SU(2) is isometric to S³ with radius R = √2.
##      Verify K = 1/R² = 1/2.
##
###########################################################################
##
##  ANALYTICAL SOLUTION
##  -------------------
##
##  The HS inner product on u(D): g(X,Y) = −Tr(XY) for skew-Hermitian X,Y.
##  The sign ensures positivity: −Tr(X²) = Tr(X†X) ≥ 0.
##
##  (a) ‖X‖² = −Tr(X²) = −Tr(−σ_x²/2) = Tr(I)/2 = 1.  Same for Y.
##      g(X,Y) = −Tr(XY) = (1/2)Tr(σ_x σ_y) = (i/2)Tr(σ_z) = 0.
##
##  (b) [X,Y] = [−iσ_x/√2, −iσ_y/√2] = −(1/2)[σ_x, σ_y]
##            = −(1/2)(2iσ_z) = −iσ_z.
##      ‖[X,Y]‖² = −Tr((−iσ_z)²) = −Tr(−σ_z²) = Tr(I) = 2.
##      K(X,Y) = (1/4)·2 = 1/2.
##
##  (c) S³(R=√2) has constant sectional curvature K = 1/R² = 1/2.  ✓
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
print("  Symbolic computation of sectional curvature on SU(2)")
print("=" * 70)

# Pauli matrices
sx = sp.Matrix([[0, 1], [1, 0]])
sy = sp.Matrix([[0, -sp.I], [sp.I, 0]])
sz = sp.Matrix([[1, 0], [0, -1]])

# --- Step 1: Orthonormal generators ---
print("\n  Step 1: Orthonormality in the Hilbert-Schmidt metric")
print("  " + "-" * 50)
print("    X = −iσ_x/√2,  Y = −iσ_y/√2")
print("    g(A,B) = −Tr(AB)  for skew-Hermitian A, B")

X = -sp.I * sx / sp.sqrt(2)
Y = -sp.I * sy / sp.sqrt(2)

norm_X_sq = sp.simplify(-sp.trace(X * X))
norm_Y_sq = sp.simplify(-sp.trace(Y * Y))
inner_XY = sp.simplify(-sp.trace(X * Y))

print(f"\n    ‖X‖² = −Tr(X²)  = {norm_X_sq}")
print(f"    ‖Y‖² = −Tr(Y²)  = {norm_Y_sq}")
print(f"    g(X,Y) = −Tr(XY) = {inner_XY}")

assert norm_X_sq == 1, f"‖X‖² = {norm_X_sq}"
assert norm_Y_sq == 1, f"‖Y‖² = {norm_Y_sq}"
assert inner_XY == 0, f"g(X,Y) = {inner_XY}"

# --- Step 2: Commutator [X,Y] ---
print("\n  Step 2: Commutator [X,Y] = −iσ_z")
print("  " + "-" * 50)

comm = sp.simplify(X * Y - Y * X)
target = -sp.I * sz
diff = sp.simplify(comm - target)
print(f"    [X,Y] = {comm.tolist()}")
print(f"    −iσ_z = {target.tolist()}")
print(f"    Match: {diff.equals(sp.zeros(2))}")
assert diff.equals(sp.zeros(2))

# Step via Pauli commutation relation
print("\n    Derivation:")
print("      [X,Y] = (−i/√2)²[σ_x, σ_y] = −(1/2)·2iσ_z = −iσ_z")
comm_xy = sp.simplify(sx * sy - sy * sx)
print(f"      [σ_x, σ_y] = {comm_xy.tolist()} = 2iσ_z  ✓")

# --- Step 3: Sectional curvature ---
print("\n  Step 3: Sectional curvature K = (1/4)‖[X,Y]‖²")
print("  " + "-" * 50)

norm_comm_sq = sp.simplify(-sp.trace(comm * comm))
K = sp.Rational(1, 4) * norm_comm_sq

print(f"    ‖[X,Y]‖² = −Tr([X,Y]²) = {norm_comm_sq}")
print(f"    K(X,Y) = (1/4) × {norm_comm_sq} = {K}")
assert K == sp.Rational(1, 2)

# --- Step 4: Consistency with S³(R=√2) ---
print("\n  Step 4: Consistency with S³ of radius R = √2")
print("  " + "-" * 50)

R = sp.sqrt(2)
K_sphere = 1 / R**2
print(f"    K_{{S³}} = 1/R² = 1/{R**2} = {K_sphere}")
print(f"    K_{{SU(2)}} = {K}")
assert K == K_sphere
print(f"    Match: SU(2) ≅ S³(√2) as Riemannian manifolds.  ✓")

# ========================================================================
#  SECTION 2 :  Numerical verification with NumPy
# ========================================================================

print("\n" + "=" * 70)
print("  Numerical verification")
print("=" * 70)

sx_np = np.array([[0, 1], [1, 0]], dtype=complex)
sy_np = np.array([[0, -1j], [1j, 0]])
sz_np = np.array([[1, 0], [0, -1]], dtype=complex)

X_np = -1j * sx_np / np.sqrt(2)
Y_np = -1j * sy_np / np.sqrt(2)

norm_X = -np.trace(X_np @ X_np).real
norm_Y = -np.trace(Y_np @ Y_np).real
inner = -np.trace(X_np @ Y_np).real

comm_np = X_np @ Y_np - Y_np @ X_np
norm_comm = -np.trace(comm_np @ comm_np).real
K_np = norm_comm / 4

print(f"\n  ‖X‖² = {norm_X:.6f},  ‖Y‖² = {norm_Y:.6f},  g(X,Y) = {inner:.6f}")
print(f"  ‖[X,Y]‖² = {norm_comm:.6f}")
print(f"  K = {K_np:.6f}  (expected 0.5)")

assert np.isclose(K_np, 0.5)
assert np.allclose(comm_np, -1j * sz_np)

print(f"\n  SU(2) has positive curvature K = 1/2.")
print(f"  Two Hamiltonian evolutions starting in different directions")
print(f"  reconverge, like great circles on a sphere.")
print(f"  The curvature is set by the Pauli commutation relations.")
