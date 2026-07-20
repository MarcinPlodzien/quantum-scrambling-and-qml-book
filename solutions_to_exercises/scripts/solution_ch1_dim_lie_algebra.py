#!/usr/bin/env python3
"""
###########################################################################
###########################################################################
##
##  SOLUTION: Dimension of the Lie Algebra u(D) and su(D)
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
##  (a) Count the real degrees of freedom in a D×D skew-Hermitian matrix
##      and show that dim_R u(D) = D².
##
##  (b) The tracelessness constraint Tr(X) = 0 defines su(D).
##      Show dim_R su(D) = D² − 1.
##
##  (c) For N qubits (D = 2^N), how does dim su(D) scale with N?
##      Relate to the barren plateau phenomenon.
##
###########################################################################
##
##  ANALYTICAL SOLUTION
##  -------------------
##
##  (a) A skew-Hermitian matrix X = −X† has:
##      • Diagonal entries: purely imaginary → D real parameters.
##      • Upper triangle: D(D−1)/2 complex entries → D(D−1) real params.
##        Lower triangle is fixed by X_{jk} = −conj(X_{kj}).
##      Total: D + D(D−1) = D².
##
##  (b) Tr(X) = sum of D imaginary diagonal entries.
##      Setting this sum to zero imposes 1 real constraint.
##      dim_R su(D) = D² − 1.
##
##  (c) For D = 2^N:  dim su(2^N) = 4^N − 1.
##      The gradient variance in a 2-design circuit scales as
##      Ω(1/dim(g)).  For su(2^N) this gives Ω(1/(4^N − 1)):
##      exponential suppression with qubit count.
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
print("  Symbolic derivation of dim u(D) and dim su(D)")
print("=" * 70)

D = sp.Symbol('D', positive=True, integer=True)
N = sp.Symbol('N', positive=True, integer=True)

# --- Step 1: Parameter counting for skew-Hermitian matrices ---
print("\n  Step 1: Real degrees of freedom in u(D)")
print("  " + "-" * 50)

diag_params = D                          # D purely imaginary entries
offdiag_params = D * (D - 1)             # D(D-1)/2 complex = D(D-1) real
total = sp.expand(diag_params + offdiag_params)

print(f"    Diagonal (imaginary):     {diag_params}")
print(f"    Off-diagonal (complex):   {offdiag_params}")
print(f"    Total = {diag_params} + {offdiag_params} = {total}")
assert total == D**2, f"Expected D², got {total}"
print(f"    dim_R u(D) = D²  ✓")

# --- Step 2: Tracelessness constraint ---
print("\n  Step 2: Tracelessness removes one real parameter")
print("  " + "-" * 50)

dim_su = D**2 - 1
print(f"    Tr(X) = Σ_j X_jj is purely imaginary (1 real constraint)")
print(f"    dim_R su(D) = D² − 1 = {dim_su}")

# --- Step 3: Exponential scaling for N qubits ---
print("\n  Step 3: Exponential scaling — dim su(2^N) = 4^N − 1")
print("  " + "-" * 50)

dim_qubits = sp.simplify((2**N)**2 - 1)
dim_alt = 4**N - 1
print(f"    dim su(2^N) = (2^N)² − 1 = 4^N − 1")
print(f"    Symbolic: {dim_qubits}")

# Barren plateau scaling
print(f"\n    Gradient variance ~ 1/dim(g) ~ 1/(4^N − 1) ~ 4^{{−N}}")

# Tabulate for specific N values
print(f"\n    {'N':>4s}  {'D=2^N':>8s}  {'dim su(D)':>12s}  {'Var ~ 1/dim':>12s}")
print(f"    {'─'*4:>4s}  {'─'*8:>8s}  {'─'*12:>12s}  {'─'*12:>12s}")
for n_val in range(1, 11):
    d_val = 2**n_val
    dim_val = d_val**2 - 1
    var_val = 1.0 / dim_val
    print(f"    {n_val:4d}  {d_val:8d}  {dim_val:12,d}  {var_val:12.2e}")

# ========================================================================
#  SECTION 2 :  Numerical verification with NumPy
# ========================================================================

print("\n" + "=" * 70)
print("  Numerical verification: constructing random elements of u(D), su(D)")
print("=" * 70)

np.random.seed(42)

for d_val in [2, 3, 4, 8]:
    # Random skew-Hermitian matrix: X = A − A†
    A = np.random.randn(d_val, d_val) + 1j * np.random.randn(d_val, d_val)
    X = A - A.conj().T

    # Verify skew-Hermiticity
    assert np.allclose(X + X.conj().T, 0)

    # Verify diagonal is purely imaginary
    diag_real = np.abs(X.diagonal().real).max()
    assert diag_real < 1e-12

    # Parameter count
    param_count = d_val + d_val * (d_val - 1)
    assert param_count == d_val**2

    # Traceless projection → su(D)
    X_su = X - np.trace(X) / d_val * np.eye(d_val)
    assert np.abs(np.trace(X_su)) < 1e-12

    print(f"  D={d_val}: dim u({d_val}) = {d_val**2}, "
          f"dim su({d_val}) = {d_val**2 - 1}  ✓")

print(f"\n  At N=10 qubits: Var(∂C/∂θ) ~ 1/{4**10-1:,} ≈ {1/(4**10-1):.2e}")
print(f"  This exponential suppression is the barren plateau phenomenon.")
