#!/usr/bin/env python3
"""
###########################################################################
###########################################################################
##
##  SOLUTION: Schur–Weyl Decomposition for k = 2
##
##  Book:     Information Scrambling and the Haar Measure
##  Chapter:  1 -- Mathematical Foundations
##  Section:  1.5 -- Representation Theory and the SWAP Operator
##
###########################################################################
##
##  EXERCISE STATEMENT
##  ------------------
##
##  The SWAP operator F on C^D ⊗ C^D exchanges tensor factors:
##    F|a,b> = |b,a>.
##
##  (a) Prove Tr(F) = D and Tr(F²) = D².
##
##  (b) Show dim(Sym²) = D(D+1)/2 and dim(∧²) = D(D−1)/2,
##      verifying dim(Sym²) + dim(∧²) = D².
##
##  (c) For ρ = I/D (maximally mixed), verify the SWAP trick:
##      Tr(ρ²) = Tr(F · ρ⊗ρ) = 1/D.
##
###########################################################################
##
##  ANALYTICAL SOLUTION
##  -------------------
##
##  (a) F_{(a,b),(c,d)} = δ_{a,d} δ_{b,c}.
##      Tr(F) = Σ_{a,b} F_{(a,b),(a,b)} = Σ_{a,b} δ_{a,b}² = D.
##      F² = I (SWAP is an involution), hence Tr(F²) = Tr(I) = D².
##
##  (b) F has eigenvalues +1 (symmetric subspace) and −1 (antisymmetric).
##      dim(Sym²) = D(D+1)/2,  dim(∧²) = D(D−1)/2.
##      Sum = D(D+1)/2 + D(D−1)/2 = D².
##
##  (c) The SWAP trick identity: Tr(AB) = Tr[(A⊗B) · F].
##      Setting A = B = ρ = I/D:
##      Tr(ρ²) = Tr[(ρ⊗ρ)F] = (1/D²) Tr(F) = D/D² = 1/D.
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
print("  Symbolic verification of the SWAP operator and Schur–Weyl duality")
print("=" * 70)

D = sp.Symbol('D', positive=True, integer=True)

# --- Step 1: Trace identities for the SWAP operator ---
print("\n  Step 1: Trace identities Tr(F) = D, Tr(F²) = D²")
print("  " + "-" * 50)
print("    F_{(a,b),(c,d)} = δ_{a,d} δ_{b,c}")
print("    Tr(F) = Σ_{a,b} δ_{a,b} δ_{b,a} = Σ_a 1 = D")
print("    F² = I  (involution), so Tr(F²) = Tr(I_{D²}) = D²")

# Symbolic verification for small D
for d_val in [2, 3, 4]:
    D2 = d_val**2
    F = sp.zeros(D2)
    for a in range(d_val):
        for b in range(d_val):
            row = b * d_val + a
            col = a * d_val + b
            F[row, col] = 1
    tr_F = sp.trace(F)
    F2 = F * F
    tr_F2 = sp.trace(F2)
    is_involution = F2.equals(sp.eye(D2))
    print(f"    D={d_val}: Tr(F) = {tr_F}, Tr(F²) = {tr_F2}, F²=I: {is_involution}")
    assert tr_F == d_val
    assert tr_F2 == d_val**2
    assert is_involution

# --- Step 2: Subspace dimensions ---
print("\n  Step 2: Symmetric and antisymmetric subspace dimensions")
print("  " + "-" * 50)

dim_sym = D * (D + 1) / 2
dim_anti = D * (D - 1) / 2
total = sp.simplify(dim_sym + dim_anti)
print(f"    dim(Sym²)  = D(D+1)/2")
print(f"    dim(∧²)    = D(D−1)/2")
print(f"    Sum        = {total}")
assert total == D**2

# --- Step 3: SWAP trick identity ---
print("\n  Step 3: SWAP trick — Tr(ρ²) = Tr[(ρ⊗ρ)·F] = 1/D")
print("  " + "-" * 50)
print("    General identity: Tr(AB) = Tr[(A⊗B)·F]")
print("    For ρ = I/D:  Tr(ρ²) = Tr(I/D²) = 1/D")
print("    Via SWAP:  Tr[(ρ⊗ρ)F] = (1/D²)Tr(F) = D/D² = 1/D  ✓")

purity_direct = sp.Rational(1, 1) / D
purity_swap = sp.Rational(1, 1) * D / D**2
assert sp.simplify(purity_direct - purity_swap) == 0
print(f"    Both expressions equal 1/D.  Identity verified symbolically.")

# ========================================================================
#  SECTION 2 :  Numerical verification with NumPy
# ========================================================================

print("\n" + "=" * 70)
print("  Numerical verification for D = 2, 3, 4, 5")
print("=" * 70)

for d_val in [2, 3, 4, 5]:
    SWAP = np.zeros((d_val**2, d_val**2))
    for a in range(d_val):
        for b in range(d_val):
            row = b * d_val + a
            col = a * d_val + b
            SWAP[row, col] = 1

    # Part (a): Trace identities
    tr_F = np.trace(SWAP)
    tr_F2 = np.trace(SWAP @ SWAP)
    assert np.isclose(tr_F, d_val)
    assert np.isclose(tr_F2, d_val**2)

    # Part (b): Eigenvalue decomposition
    eigvals = np.linalg.eigvalsh(SWAP)
    n_plus = np.sum(np.isclose(eigvals, 1))
    n_minus = np.sum(np.isclose(eigvals, -1))
    dim_sym_val = d_val * (d_val + 1) // 2
    dim_anti_val = d_val * (d_val - 1) // 2
    assert n_plus == dim_sym_val
    assert n_minus == dim_anti_val

    # Part (c): SWAP trick purity
    rho = np.eye(d_val) / d_val
    purity_direct = np.trace(rho @ rho).real
    purity_swap = np.trace(SWAP @ np.kron(rho, rho)).real
    assert np.isclose(purity_direct, 1/d_val)
    assert np.isclose(purity_swap, purity_direct)

    print(f"\n  D={d_val}: Tr(F)={int(tr_F)}, Tr(F²)={int(tr_F2)}, "
          f"Sym²={dim_sym_val}, ∧²={dim_anti_val}, "
          f"Tr(ρ²)=1/{d_val}  ✓")
