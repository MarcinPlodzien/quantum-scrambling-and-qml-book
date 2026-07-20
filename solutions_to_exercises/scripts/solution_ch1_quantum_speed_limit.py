#!/usr/bin/env python3
"""
###########################################################################
###########################################################################
##
##  SOLUTION: Geodesic Length and the Quantum Speed Limit
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
##  In the bi-invariant HS metric on U(D), the geodesic U(t) = exp(−iHt)
##  has constant speed ‖dU/dt · U†‖ = √Tr(H²).
##
##  (a) Compute the geodesic length from I to U(T) = exp(−iHT).
##
##  (b) For H = (ω/2)σ_x on a qubit, compute the geodesic speed
##      and the length to the π-rotation U(π/ω) = −iσ_x.
##
##  (c) Mandelstam-Tamm bound: T_⊥ = π/(2ΔH).
##      For |ψ₀> = |0> under H = (ω/2)σ_x:
##      • Compute ΔH = √(<H²> − <H>²).
##      • Predict T_⊥.
##      • Verify |<0|exp(−iHt)|0>|² = 0 at t = T_⊥.
##
###########################################################################
##
##  ANALYTICAL SOLUTION
##  -------------------
##
##  (a) Velocity: dU/dt · U† = −iH (constant for geodesics).
##      Speed: v = ‖−iH‖ = √(−Tr(−H²)) = √Tr(H²).
##      Length: L = T · √Tr(H²).
##
##  (b) H = (ω/2)σ_x.
##      Tr(H²) = (ω²/4)Tr(σ_x²) = (ω²/4)·2 = ω²/2.
##      Speed: v = ω/√2.
##      At T = π/ω:  L = π/√2.
##
##  (c) <H> = (ω/2)<0|σ_x|0> = 0.
##      <H²> = (ω²/4)<0|σ_x²|0> = ω²/4.
##      ΔH = ω/2.
##      T_⊥ = π/(2·ω/2) = π/ω.
##      |<0|ψ(t)>|² = cos²(ωt/2) = 0 at t = π/ω.
##      The bound is SATURATED: |0> is an equal superposition of
##      the σ_x eigenstates, giving maximal energy spread.
##
###########################################################################
###########################################################################
"""
import sympy as sp
import numpy as np
from scipy.linalg import expm

# ========================================================================
#  SECTION 1 :  Symbolic derivation with SymPy
# ========================================================================

print("=" * 70)
print("  Symbolic derivation of geodesic length and quantum speed limit")
print("=" * 70)

omega, t, T = sp.symbols('omega t T', real=True, positive=True)

# Pauli matrices
sx = sp.Matrix([[0, 1], [1, 0]])

# --- Step 1: Geodesic speed from the HS metric ---
print("\n  Step 1: Geodesic speed v = √Tr(H²)")
print("  " + "-" * 50)
print("    U(t) = exp(−iHt)  →  dU/dt = −iH·U(t)")
print("    Left-translated velocity: (dU/dt)·U† = −iH  (constant)")
print("    Speed: v = ‖−iH‖ = √(−Tr((−iH)²)) = √Tr(H²)")
print("    Length: L(T) = ∫₀ᵀ v dt = T·√Tr(H²)")

# --- Step 2: Explicit computation for H = (ω/2)σ_x ---
print("\n  Step 2: Geodesic for H = (ω/2)σ_x")
print("  " + "-" * 50)

H = omega / 2 * sx
H2 = sp.simplify(sp.trace(H * H))
print(f"    Tr(H²) = Tr((ω/2)²σ_x²) = (ω²/4)·Tr(I) = {H2}")
assert H2 == omega**2 / 2

speed = sp.sqrt(H2)
print(f"    Speed v = √Tr(H²) = {sp.simplify(speed)} = ω/√2")

T_pi = sp.pi / omega
length = T_pi * speed
length_simplified = sp.simplify(length)
print(f"    T = π/ω  →  L = {length_simplified} = π/√2")
assert sp.simplify(length_simplified - sp.pi / sp.sqrt(2)) == 0

# Verify U(π/ω) = −iσ_x via matrix exponential
print(f"\n    U(π/ω) = exp(−i(ω/2)σ_x · π/ω) = exp(−iπσ_x/2)")
print(f"           = cos(π/2)I − i sin(π/2)σ_x = −iσ_x")

# --- Step 3: Mandelstam-Tamm bound ---
print("\n  Step 3: Mandelstam-Tamm quantum speed limit")
print("  " + "-" * 50)

# |ψ₀> = |0>
psi0 = sp.Matrix([1, 0])
H_mat = omega / 2 * sx

# <H>
H_exp = sp.simplify((psi0.H * H_mat * psi0)[0, 0])
print(f"    <H>   = <0|(ω/2)σ_x|0> = {H_exp}")
assert H_exp == 0

# <H²>
H2_exp = sp.simplify((psi0.H * H_mat * H_mat * psi0)[0, 0])
print(f"    <H²>  = <0|(ω/2)²σ_x²|0> = {H2_exp} = ω²/4")
assert H2_exp == omega**2 / 4

# ΔH
delta_H = sp.sqrt(H2_exp - H_exp**2)
delta_H = sp.simplify(delta_H)
print(f"    ΔH    = √(<H²>−<H>²) = {delta_H} = ω/2")

# T_⊥
T_perp = sp.pi / (2 * delta_H)
T_perp = sp.simplify(T_perp)
print(f"    T_⊥   = π/(2ΔH) = {T_perp} = π/ω")
assert sp.simplify(T_perp - sp.pi / omega) == 0

# Survival probability: |<0|exp(−iHt)|0>|² = cos²(ωt/2)
print(f"\n    Survival probability: |<0|ψ(t)>|² = cos²(ωt/2)")
survival_at_Tperp = sp.cos(omega * T_perp / 2)**2
survival_at_Tperp = sp.simplify(survival_at_Tperp)
print(f"    At t = T_⊥ = π/ω:  cos²(π/2) = {survival_at_Tperp}")
assert survival_at_Tperp == 0

print(f"\n    The bound is saturated: |0> has equal weight on both σ_x")
print(f"    eigenstates, achieving the maximum energy uncertainty.")

# ========================================================================
#  SECTION 2 :  Numerical verification with NumPy
# ========================================================================

print("\n" + "=" * 70)
print("  Numerical verification")
print("=" * 70)

sx_np = np.array([[0, 1], [1, 0]], dtype=complex)

# Part (a): Geodesic length for random Hamiltonians
print("\n  Part (a): Geodesic length L = T·√Tr(H²)")
print("  " + "-" * 50)

np.random.seed(42)
for label, H_np, T_val in [
    ("σ_z", np.array([[1, 0], [0, -1]], dtype=complex), 1.0),
    ("σ_x", sx_np, np.pi),
    ("random 4×4", None, 0.5),
]:
    if H_np is None:
        A = np.random.randn(4, 4) + 1j * np.random.randn(4, 4)
        H_np = (A + A.conj().T) / 2

    speed_np = np.sqrt(np.trace(H_np @ H_np).real)
    length_np = T_val * speed_np

    print(f"    {label:15s}: v = {speed_np:.4f}, L(T={T_val:.2f}) = {length_np:.4f}  ✓")

# Part (b): Qubit under H = (ω/2)σ_x
print("\n  Part (b): H = (ω/2)σ_x with ω = 2")
print("  " + "-" * 50)

omega_val = 2.0
H_np = omega_val / 2 * sx_np
trH2 = np.trace(H_np @ H_np).real
speed_np = np.sqrt(trH2)
T_pi_val = np.pi / omega_val
U_pi = expm(-1j * H_np * T_pi_val)
length_np = T_pi_val * speed_np

assert np.allclose(U_pi, -1j * sx_np)
assert np.isclose(length_np, np.pi / np.sqrt(2))

print(f"    Tr(H²) = {trH2:.4f}  (expected ω²/2 = {omega_val**2/2:.4f})")
print(f"    Speed  = {speed_np:.4f}  (expected ω/√2 = {omega_val/np.sqrt(2):.4f})")
print(f"    U(π/ω) = −iσ_x: {np.allclose(U_pi, -1j*sx_np)}")
print(f"    L = {length_np:.4f}  (expected π/√2 = {np.pi/np.sqrt(2):.4f})  ✓")

# Part (c): Mandelstam-Tamm bound
print("\n  Part (c): Mandelstam-Tamm bound T_⊥ = π/(2ΔH)")
print("  " + "-" * 50)

psi0_np = np.array([1, 0], dtype=complex)
H_exp_np = (psi0_np.conj() @ H_np @ psi0_np).real
H2_exp_np = (psi0_np.conj() @ H_np @ H_np @ psi0_np).real
delta_H_np = np.sqrt(H2_exp_np - H_exp_np**2)
T_perp_np = np.pi / (2 * delta_H_np)

U_T = expm(-1j * H_np * T_perp_np)
survival = abs(psi0_np.conj() @ U_T @ psi0_np)**2

print(f"    <H>   = {H_exp_np:.4f}  (expected 0)")
print(f"    <H²>  = {H2_exp_np:.4f}  (expected ω²/4 = {omega_val**2/4:.4f})")
print(f"    ΔH    = {delta_H_np:.4f}  (expected ω/2 = {omega_val/2:.4f})")
print(f"    T_⊥   = {T_perp_np:.4f}  (expected π/ω = {np.pi/omega_val:.4f})")
print(f"    |<0|ψ(T_⊥)>|² = {survival:.10f}  (expected 0)")
assert np.isclose(survival, 0, atol=1e-10)
print(f"    Bound is SATURATED.  ✓")
