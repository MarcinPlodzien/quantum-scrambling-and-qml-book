#!/usr/bin/env python3
"""
###########################################################################
###########################################################################
##
##  SOLUTION: The Depolarizing Channel — Kraus Decomposition,
##            Unitality, and Entanglement Fidelity
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
##  The depolarizing channel replaces a state with the maximally mixed
##  state with probability p:
##    E_p(rho) = (1-p) rho + (p/D) I
##
##  For a single qubit (D=2), an equivalent Kraus representation is:
##    E_p(rho) = (1-3p/4) rho + (p/4)(X rho X + Y rho Y + Z rho Z)
##
##  (a) Show that E_p preserves the identity: E_p(I/2) = I/2.
##      This makes the channel UNITAL.
##
##  (b) Verify the Kraus completeness relation: sum_k K_k^dag K_k = I.
##
##  (c) Compute the entanglement fidelity:
##      F_e = <Phi+| (E_p ⊗ I)|Phi+><Phi+| |Phi+>
##      and show F_e = 1 - 3p/4.
##
###########################################################################
##
##  ANALYTICAL SOLUTION (verified symbolically below)
##  -------------------------------------------------
##
##  (a) Unitality:
##      E_p(I/2)  = (1-3p/4)·(I/2) + (p/4)·(X(I/2)X + Y(I/2)Y + Z(I/2)Z)
##      Each σ_j·(I/2)·σ_j = I/2, because σ_j is unitary.
##      So E_p(I/2) = (1-3p/4)·(I/2) + 3·(p/4)·(I/2) = (I/2).
##      Alternatively: E_p(I/D) = (1-p)·I/D + (p/D)·I = I/D.
##
##  (b) Kraus completeness:
##      K_0 = sqrt(1-3p/4)·I,  K_j = sqrt(p/4)·σ_j   (j=1,2,3)
##      sum_k K_k†K_k = (1-3p/4)·I + 3·(p/4)·I = I.
##
##  (c) Entanglement fidelity via the maximally entangled state
##      |Φ+> = (|00>+|11>)/√2:
##        F_e = <Φ+| (E_p⊗I)|Φ+><Φ+| |Φ+>
##      The identity branch contributes (1-3p/4)·|<Φ+|Φ+>|² = 1-3p/4.
##      Each Pauli branch contributes (p/4)·|<Φ+|(σ_j⊗I)|Φ+>|².
##      Now <Φ+|(σ_j⊗I)|Φ+> = (1/2)Tr(σ_j) = 0 for all j.
##      Therefore F_e = 1 - 3p/4.
##
###########################################################################
##
##  STRUCTURE OF THIS SCRIPT
##  ------------------------
##
##  SECTION 1:  SymPy symbolic proof of all three parts
##              (Pauli algebra, Kraus completeness, fidelity formula)
##
##  SECTION 2:  NumPy numerical verification across a sweep of p values
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
print("  SECTION 1: Symbolic derivation — Depolarizing channel properties")
print("=" * 70)

p = sp.Symbol('p', real=True, positive=True)

# ------ Define 2x2 Pauli matrices symbolically ------
I2 = sp.eye(2)
sx = sp.Matrix([[0, 1], [1, 0]])
sy = sp.Matrix([[0, -sp.I], [sp.I, 0]])
sz = sp.Matrix([[1, 0], [0, -1]])
paulis = [sx, sy, sz]

# ========================================================================
#  Step 1: Verify sigma_j (I/2) sigma_j = I/2
# ========================================================================
print("\n  Step 1: Unitality — σ_j · (I/2) · σ_j = I/2 for all Paulis")
print("  " + "-" * 60)
print("    The depolarizing channel preserves the maximally mixed state")
print("    because each Pauli rotation is a unitary involution.  We verify")
print("    this algebraically for each Pauli matrix.")

rho_max = I2 / 2
for name, sigma in [("σ_x", sx), ("σ_y", sy), ("σ_z", sz)]:
    result = sp.simplify(sigma * rho_max * sigma)
    match = result == rho_max
    print(f"    {name} · (I/2) · {name} = I/2 : {match}")
    assert match, f"{name} * (I/2) * {name} != I/2"

# Symbolic proof of E_p(I/2) = I/2
E_I2 = (1 - 3*p/4) * rho_max
for sigma in paulis:
    E_I2 += (p/4) * sigma * rho_max * sigma
E_I2 = sp.simplify(E_I2)
print(f"\n    E_p(I/2) = {E_I2}")
assert E_I2 == rho_max, "E_p(I/2) != I/2"
print(f"    Confirmed: E_p(I/2) = I/2 for all p.  Channel is UNITAL.")

# ========================================================================
#  Step 2: Kraus completeness — sum_k K_k† K_k = I
# ========================================================================
print("\n  Step 2: Kraus completeness relation — trace preservation")
print("  " + "-" * 60)
print("    The Kraus operators are:")
print("      K_0 = √(1 - 3p/4) · I")
print("      K_j = √(p/4) · σ_j    for j = 1,2,3")
print("    We verify K_0†K_0 + K_1†K_1 + K_2†K_2 + K_3†K_3 = I.")

completeness = (1 - 3*p/4) * I2  # K_0† K_0 = (1-3p/4) I
for sigma in paulis:
    # K_j† K_j = (p/4) σ_j† σ_j = (p/4) I  (since σ_j² = I)
    KdK = sp.simplify((p/4) * sigma.H * sigma)
    completeness += KdK

completeness = sp.simplify(completeness)
print(f"\n    Σ_k K_k† K_k = {completeness}")
assert completeness == I2, f"Completeness failed: {completeness}"
print(f"    Confirmed: Σ_k K_k† K_k = I.  Channel is trace-preserving.")

# ========================================================================
#  Step 3: Pauli algebra — σ_j² = I and Tr(σ_j) = 0
# ========================================================================
print("\n  Step 3: Key Pauli identities used in the derivation")
print("  " + "-" * 60)

for name, sigma in [("σ_x", sx), ("σ_y", sy), ("σ_z", sz)]:
    sq = sp.simplify(sigma * sigma)
    tr = sp.trace(sigma)
    print(f"    {name}² = I: {sq == I2},    Tr({name}) = {tr}")
    assert sq == I2, f"{name}^2 != I"
    assert tr == 0, f"Tr({name}) != 0"

# ========================================================================
#  Step 4: Entanglement fidelity F_e = 1 - 3p/4
# ========================================================================
print("\n  Step 4: Entanglement fidelity F_e = 1 - 3p/4")
print("  " + "-" * 60)
print("    |Φ+> = (|00> + |11>)/√2.   The fidelity is:")
print("    F_e = <Φ+| (E_p ⊗ I) |Φ+><Φ+| |Φ+>")
print()
print("    Identity branch: (1-3p/4) · |<Φ+|Φ+>|² = 1 - 3p/4")
print("    Pauli branches:  (p/4) · |<Φ+| (σ_j ⊗ I) |Φ+>|²")
print("    Since <Φ+|(σ_j ⊗ I)|Φ+> = (1/2)Tr(σ_j) = 0 for all j,")
print("    the Pauli branches vanish identically.")

# Symbolic verification: <Φ+|(σ_j ⊗ I)|Φ+>
# |Φ+> = (1/√2)(|00> + |11>)  represented as a 4-vector
bell = sp.Matrix([1, 0, 0, 1]) / sp.sqrt(2)

for name, sigma in [("σ_x", sx), ("σ_y", sy), ("σ_z", sz)]:
    S_full = sp.kronecker_product(sigma, I2)
    overlap = (bell.H * S_full * bell)[0, 0]
    overlap = sp.simplify(overlap)
    print(f"    <Φ+|(${name}$⊗I)|Φ+> = {overlap}  (= Tr({name})/2)")
    assert overlap == 0, f"Overlap for {name} != 0"

F_e_sym = 1 - 3*p/4
print(f"\n    RESULT:  F_e = 1 - 3p/4")
print(f"    At p=0: F_e = 1 (perfect channel)")
print(f"    At p=1: F_e = 1/4 = 1/D² (fully depolarizing)")

# ========================================================================
#  SECTION 2 :  Numerical verification with NumPy
# ========================================================================

print("\n" + "=" * 70)
print("  SECTION 2: Numerical verification across p ∈ [0, 1]")
print("=" * 70)

I2_np = np.eye(2, dtype=complex)
paulis_np = [
    np.array([[0, 1], [1, 0]], dtype=complex),       # X
    np.array([[0, -1j], [1j, 0]], dtype=complex),     # Y
    np.array([[1, 0], [0, -1]], dtype=complex),       # Z
]

def depol_channel(rho, p_val):
    """Apply the depolarizing channel E_p(rho) via Kraus representation."""
    result = (1 - 3*p_val/4) * rho
    for sigma in paulis_np:
        result += (p_val/4) * sigma @ rho @ sigma
    return result

for p_val in [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]:
    print(f"\n  p = {p_val:.2f}:")

    # Part (a): Unitality -- E_p(I/2) = I/2
    rho_max_np = I2_np / 2
    out = depol_channel(rho_max_np, p_val)
    err_unital = np.linalg.norm(out - rho_max_np)
    print(f"    (a) ||E_p(I/2) - I/2|| = {err_unital:.2e}  [unital]")
    assert err_unital < 1e-14

    # Part (b): Kraus completeness -- sum K_k^dag K_k = I
    completeness_np = (1 - 3*p_val/4) * I2_np
    for sigma in paulis_np:
        completeness_np += (p_val/4) * sigma @ sigma
    err_complete = np.linalg.norm(completeness_np - I2_np)
    print(f"    (b) ||Σ K_k†K_k - I|| = {err_complete:.2e}  [trace-preserving]")
    assert err_complete < 1e-14

    # Part (c): Entanglement fidelity F_e = 1 - 3p/4
    bell_np = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
    rho_bell = np.outer(bell_np, bell_np.conj())

    rho_out = np.zeros((4, 4), dtype=complex)
    rho_out += (1 - 3*p_val/4) * rho_bell
    for sigma in paulis_np:
        S_full = np.kron(sigma, I2_np)
        rho_out += (p_val/4) * S_full @ rho_bell @ S_full

    F_e = (bell_np.conj() @ rho_out @ bell_np).real
    F_e_pred = 1 - 3*p_val/4

    print(f"    (c) F_e = {F_e:.6f}  (predicted 1-3p/4 = {F_e_pred:.6f})")
    assert abs(F_e - F_e_pred) < 1e-12

print(f"\n  At p=0 (no noise): F_e = 1, perfect channel.")
print(f"  At p=1 (full depolarization): F_e = 1/4 = 1/D²,")
print(f"  the minimum possible for a D=2 channel.")
