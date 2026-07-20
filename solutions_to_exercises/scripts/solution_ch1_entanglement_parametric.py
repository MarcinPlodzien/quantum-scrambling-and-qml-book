#!/usr/bin/env python3
"""
###########################################################################
###########################################################################
##
##  SOLUTION: Entanglement of a Parametric Two-Qubit State
##
##  Book:     Information Scrambling and the Haar Measure
##  Chapter:  1 -- Mathematical Foundations
##  Section:  1.1 -- Bipartite Entanglement
##
###########################################################################
##
##  EXERCISE STATEMENT
##  ------------------
##
##  Consider the two-qubit state:
##    |psi(theta)> = cos(theta)|00> + sin(theta)|11>
##
##  (a)  Compute the reduced density matrix rho_A = Tr_B(|psi><psi|).
##
##  (b)  Compute the purity gamma = Tr(rho_A^2) and the von Neumann
##       entropy S = -Tr(rho_A ln rho_A) as functions of theta.
##
##  (c)  Show the state is maximally entangled at theta = pi/4
##       (gamma = 1/2, S = ln 2) and separable at theta = 0, pi/2.
##
###########################################################################
##
##  ANALYTICAL SOLUTION (verified symbolically below)
##  -------------------------------------------------
##
##  (a)  The full density matrix is:
##         |psi><psi| = cos^2(theta)|00><00| + sin^2(theta)|11><11|
##                    + cos(theta)sin(theta)[|00><11| + |11><00|]
##       Tracing over B:
##         rho_A = cos^2(theta)|0><0| + sin^2(theta)|1><1|
##       This is diagonal -- off-diagonal terms vanish because
##       <0_B|1_B> = 0.
##
##  (b)  Purity:  gamma = cos^4(theta) + sin^4(theta).
##       Entropy: S = -cos^2(theta) ln(cos^2(theta))
##                    -sin^2(theta) ln(sin^2(theta)).
##
##  (c)  At theta = pi/4: cos^2 = sin^2 = 1/2.
##         gamma = 1/4 + 1/4 = 1/2  (maximally mixed rho_A = I/2).
##         S = ln(2)  (one bit of entanglement).
##       At theta = 0: product state |00>, gamma = 1, S = 0.
##       At theta = pi/2: product state |11>, gamma = 1, S = 0.
##
###########################################################################
##
##  STRUCTURE OF THIS SCRIPT
##  ------------------------
##
##  SECTION 1:  SymPy symbolic derivation  (every algebraic step
##              printed to the terminal)
##
##  SECTION 2:  NumPy numerical verification at specific angles
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
print("  SECTION 1: Symbolic derivation of purity and entropy")
print("=" * 70)

theta = sp.Symbol('theta', real=True, positive=True)

# --- Step 1: Build the state |psi(theta)> = cos(theta)|00> + sin(theta)|11> ---
print("\n  Step 1: The parametric two-qubit state")
print("  " + "-" * 50)

c = sp.cos(theta)
s = sp.sin(theta)
print(f"    |psi(theta)> = cos(theta)|00> + sin(theta)|11>")

# --- Step 2: Reduced density matrix rho_A = Tr_B(|psi><psi|) ---
print("\n  Step 2: Reduced density matrix rho_A = Tr_B(|psi><psi|)")
print("  " + "-" * 50)

# |psi><psi| has four terms.  Tracing over B kills the off-diagonals
# because <0_B|1_B> = 0.  Only the diagonal survives:
rho_A_00 = sp.trigsimp(c**2)
rho_A_11 = sp.trigsimp(s**2)

print(f"    rho_A = cos^2(theta) |0><0| + sin^2(theta) |1><1|")
print(f"    rho_A_00 = {rho_A_00}")
print(f"    rho_A_11 = {rho_A_11}")

# Verify trace = 1
trace = sp.trigsimp(rho_A_00 + rho_A_11)
print(f"    Tr(rho_A) = {trace}")
assert trace == 1, f"Trace = {trace}, expected 1"

# --- Step 3: Purity gamma = Tr(rho_A^2) ---
print("\n  Step 3: Purity gamma = Tr(rho_A^2)")
print("  " + "-" * 50)

gamma = sp.trigsimp(rho_A_00**2 + rho_A_11**2)
print(f"    gamma = cos^4(theta) + sin^4(theta)")
print(f"          = {gamma}")

# Simplify using cos^4 + sin^4 = 1 - 2 sin^2 cos^2 = 1 - sin^2(2t)/2
gamma_alt = sp.trigsimp(sp.expand(gamma))
print(f"    Simplified: {gamma_alt}")

# Check special values
gamma_0 = gamma.subs(theta, 0)
gamma_pi4 = sp.trigsimp(gamma.subs(theta, sp.pi/4))
gamma_pi2 = sp.trigsimp(gamma.subs(theta, sp.pi/2))
print(f"\n    gamma(0)    = {gamma_0}  (separable: pure state)")
print(f"    gamma(pi/4) = {gamma_pi4}  (maximally entangled)")
print(f"    gamma(pi/2) = {gamma_pi2}  (separable: pure state)")

assert gamma_0 == 1, f"gamma(0) = {gamma_0}"
assert gamma_pi4 == sp.Rational(1, 2), f"gamma(pi/4) = {gamma_pi4}"
assert gamma_pi2 == 1, f"gamma(pi/2) = {gamma_pi2}"

# --- Step 4: Von Neumann entropy S = -Tr(rho_A ln rho_A) ---
print("\n  Step 4: Von Neumann entropy S = -Tr(rho_A ln rho_A)")
print("  " + "-" * 50)

# S = -cos^2(t) ln(cos^2(t)) - sin^2(t) ln(sin^2(t))
S_sym = -c**2 * sp.log(c**2) - s**2 * sp.log(s**2)
print(f"    S(theta) = -cos^2(theta) ln(cos^2(theta))")
print(f"              - sin^2(theta) ln(sin^2(theta))")

# Evaluate at theta = pi/4
S_pi4 = sp.trigsimp(S_sym.subs(theta, sp.pi/4))
print(f"\n    S(pi/4) = {S_pi4}")
assert S_pi4 == sp.log(2), f"S(pi/4) = {S_pi4}, expected ln(2)"
print(f"            = ln(2)  ✓  (one bit of entanglement)")

# At the boundary theta -> 0+, we use L'Hopital: x ln(x) -> 0 as x -> 0+
S_0_limit = sp.limit(S_sym, theta, 0, '+')
print(f"    S(0+)   = {S_0_limit}  (separable)")

# --- Step 5: Summary ---
print("\n  Step 5: Summary of symbolic results")
print("  " + "-" * 50)
print(f"    rho_A is always diagonal with eigenvalues (cos^2, sin^2).")
print(f"    Purity:  gamma = cos^4 + sin^4 in [1/2, 1].")
print(f"    Entropy: S = H_2(cos^2) in [0, ln 2].")
print(f"    Maximum entanglement at theta = pi/4.")

# ========================================================================
#  SECTION 2 :  Numerical verification with NumPy
# ========================================================================

print("\n" + "=" * 70)
print("  SECTION 2: Numerical verification at specific angles")
print("=" * 70)

print(f"\n  {'theta/pi':>10s}  {'gamma':>10s}  {'S':>10s}  {'S/ln2':>10s}")
print(f"  {'-'*10:>10s}  {'-'*10:>10s}  {'-'*10:>10s}  {'-'*10:>10s}")

for t in np.linspace(0, np.pi/2, 9):
    c2 = np.cos(t)**2
    s2 = np.sin(t)**2

    # Purity
    purity = c2**2 + s2**2

    # Von Neumann entropy (handling edge cases where c2 or s2 = 0)
    S = 0.0
    if c2 > 1e-15:
        S -= c2 * np.log(c2)
    if s2 > 1e-15:
        S -= s2 * np.log(s2)

    print(f"  {t/np.pi:10.4f}  {purity:10.6f}  {S:10.6f}  {S/np.log(2):10.6f}")

    # Verify specific cases
    if np.isclose(t, 0):
        assert np.isclose(purity, 1.0), f"theta=0: purity should be 1"
        assert np.isclose(S, 0.0), f"theta=0: entropy should be 0"
    elif np.isclose(t, np.pi/4):
        assert np.isclose(purity, 0.5), f"theta=pi/4: purity should be 1/2"
        assert np.isclose(S, np.log(2)), f"theta=pi/4: S should be ln(2)"
    elif np.isclose(t, np.pi/2):
        assert np.isclose(purity, 1.0), f"theta=pi/2: purity should be 1"
        assert np.isclose(S, 0.0), f"theta=pi/2: entropy should be 0"

print(f"\n  The purity interpolates between 1 (separable) and 1/2 (maximally")
print(f"  entangled).  The entropy peaks at ln(2) = {np.log(2):.6f} nats")
print(f"  at theta = pi/4, where rho_A = I/2.")
