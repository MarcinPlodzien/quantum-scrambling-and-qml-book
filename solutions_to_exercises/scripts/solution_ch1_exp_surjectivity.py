#!/usr/bin/env python3
"""
###########################################################################
###########################################################################
##
##  SOLUTION: Surjectivity of the Exponential Map on U(D)
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
##  Show that every U ∈ U(D) can be written as U = exp(X) for some
##  X ∈ u(D).  (Hint: use the spectral theorem.)
##
###########################################################################
##
##  ANALYTICAL SOLUTION
##  -------------------
##
##  Since U is unitary and hence normal, the spectral theorem gives
##    U = Σ_j exp(iθ_j) |v_j><v_j|.
##
##  Define the skew-Hermitian generator
##    X = i Σ_j θ_j |v_j><v_j|.
##
##  Then:
##    X† = −i Σ_j θ_j |v_j><v_j| = −X        (skew-Hermitian)
##    exp(X) = Σ_j exp(iθ_j) |v_j><v_j| = U   (spectral mapping)
##
##  Surjectivity is special to compact groups.  It fails for
##  non-compact groups (e.g., GL(n,R) with det < 0).
##
###########################################################################
###########################################################################
"""
import sympy as sp
import numpy as np
from scipy.linalg import expm
from scipy.stats import unitary_group

# ========================================================================
#  SECTION 1 :  Symbolic derivation with SymPy
# ========================================================================

print("=" * 70)
print("  Symbolic proof of surjectivity for U(D)")
print("=" * 70)

# --- Step 1: Spectral theorem construction ---
print("\n  Step 1: Spectral decomposition U = V diag(e^{iθ}) V†")
print("  " + "-" * 50)
print("    Given U unitary, diagonalize: U = V Λ V†")
print("    with Λ_jj = e^{iθ_j}.")
print("    Define X = V · diag(iθ₁, ..., iθ_D) · V†")
print("    Then X† = V · diag(−iθ₁, ..., −iθ_D) · V† = −X  (skew-Hermitian)")
print("    and exp(X) = V · diag(e^{iθ₁}, ..., e^{iθ_D}) · V† = U.")

# Symbolic construction for D=2
theta1, theta2 = sp.symbols('theta_1 theta_2', real=True)

Lambda = sp.Matrix([[sp.exp(sp.I * theta1), 0],
                     [0, sp.exp(sp.I * theta2)]])
X_diag = sp.Matrix([[sp.I * theta1, 0],
                     [0, sp.I * theta2]])

# Verify exp(X_diag) = Lambda
print("\n  Step 2: Explicit verification for a diagonal 2×2 unitary")
print("  " + "-" * 50)

exp_X = sp.Matrix([[sp.exp(X_diag[0, 0]), 0],
                    [0, sp.exp(X_diag[1, 1])]])
diff = sp.simplify(exp_X - Lambda)
print(f"    X = diag(iθ₁, iθ₂)")
print(f"    exp(X) = diag(e^{{iθ₁}}, e^{{iθ₂}})")
print(f"    exp(X) − Λ = {diff.tolist()}")
assert diff.equals(sp.zeros(2))

# Verify skew-Hermiticity
print("\n  Step 3: Skew-Hermiticity X† = −X")
print("  " + "-" * 50)
X_dag = X_diag.H
sum_check = sp.simplify(X_diag + X_dag)
print(f"    X + X† = {sum_check.tolist()}")
assert sum_check.equals(sp.zeros(2))
print(f"    X ∈ u(2) confirmed.")

# --- Step 4: Counter-example for non-compact groups ---
print("\n  Step 4: Counter-example — surjectivity fails for GL(n,R)")
print("  " + "-" * 50)
print("    A = diag(−1, 1) ∈ GL(2,R) has det(A) = −1.")
print("    If A = exp(X) with X real, then det(A) = exp(Tr(X)) > 0.")
print("    Contradiction: matrices with det < 0 have no real logarithm.")
print("    Surjectivity holds only for compact groups (like U(D)).")

# ========================================================================
#  SECTION 2 :  Numerical verification with NumPy
# ========================================================================

print("\n" + "=" * 70)
print("  Numerical verification: logarithm construction for random unitaries")
print("=" * 70)

np.random.seed(42)

for d_val in [2, 3, 4, 8]:
    print(f"\n  D = {d_val}:")
    for trial in range(5):
        U = unitary_group.rvs(d_val)

        # Diagonalize: U = V diag(e^{iθ}) V†
        eigenvalues, V = np.linalg.eig(U)
        phases = np.angle(eigenvalues)

        # Construct X = V diag(iθ) V⁻¹
        X = V @ np.diag(1j * phases) @ np.linalg.inv(V)

        # Check skew-Hermiticity
        skew_err = np.linalg.norm(X + X.conj().T)
        assert skew_err < 1e-10

        # Check exp(X) = U
        U_recon = expm(X)
        recon_err = np.linalg.norm(U_recon - U)
        assert recon_err < 1e-10

        print(f"    trial {trial}: ‖X+X†‖ = {skew_err:.2e}, "
              f"‖exp(X)−U‖ = {recon_err:.2e}  ✓")
