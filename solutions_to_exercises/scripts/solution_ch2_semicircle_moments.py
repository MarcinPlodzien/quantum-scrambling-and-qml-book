#!/usr/bin/env python3
"""
###########################################################################
#  EXERCISE 2.2 — Moments of the Wigner Semicircle
#
#  Chapter 2, Random Matrix Theory
#  Topic: Wigner semicircle law, Catalan numbers, kurtosis
#
#  ---------- EXERCISE STATEMENT ----------
#
#  The Wigner semicircle law:
#    rho_sc(lambda) = (1/2pi) * sqrt(4 - lambda^2)   for |lambda| <= 2.
#
#  (a) Show all odd moments vanish: m_{2k+1} = 0.
#  (b) Compute m_2 and verify m_2 = 1.
#  (c) Compute m_4 and show m_4 = 2.  (Hint: lambda = 2*sin(theta).)
#  (d) Interpret m_4/m_2^2 = 2 < 3 (platykurtic: lighter tails than
#      Gaussian, because spectral density has compact support [-2,2]).
#
#  ---------- ANALYTICAL SOLUTION ----------
#
#  (a) rho_sc is even, so lambda^{2k+1} * rho_sc is odd on [-2,2].
#      Integral of an odd function on a symmetric interval = 0.
#
#  (b) Substitute lambda = 2*sin(theta):
#      m_2 = (8/pi) * int_{-pi/2}^{pi/2} sin^2(theta) cos^2(theta) dtheta
#           = (8/pi) * pi/8 = 1.
#
#  (c) m_4 = (32/pi) * int sin^4(theta) cos^2(theta) dtheta
#           = (32/pi) * pi/16 = 2.
#      The semicircle moments m_{2k} are the Catalan numbers:
#      C_0=1, C_1=1, C_2=2, C_3=5, ...
#
#  (d) m_4/m_2^2 = 2 < 3 (Gaussian kurtosis).
#      The semicircle is platykurtic: flatter, with hard edges at +/-2.
#
#  ---------- NUMERICAL VERIFICATION ----------
###########################################################################
"""
import numpy as np
from scipy.integrate import quad

def rho_sc(lam):
    """Wigner semicircle density on [-2, 2]."""
    if abs(lam) > 2:
        return 0.0
    return np.sqrt(4 - lam**2) / (2 * np.pi)

# ========================================================================
# Part (a): Odd moments vanish by symmetry
# ========================================================================
print("Part (a): Odd moments vanish by symmetry of the semicircle law")
print("=" * 60)

for k in [1, 3, 5, 7]:
    m_odd, _ = quad(lambda x: x**k * rho_sc(x), -2, 2)
    print(f"  m_{k} = {m_odd:.2e}  (expected 0)")
    assert abs(m_odd) < 1e-12, f"Odd moment m_{k} = {m_odd} is nonzero!"

# ========================================================================
# Part (b): Second moment m_2 = 1
# ========================================================================
print(f"\nPart (b): Second moment m_2 = 1")
print("=" * 60)

m2, _ = quad(lambda x: x**2 * rho_sc(x), -2, 2)
print(f"  m_2 = {m2:.10f}  (expected 1)")
assert abs(m2 - 1) < 1e-10, f"m_2 = {m2}, expected 1"

# Verify the intermediate integral: int_{-pi/2}^{pi/2} sin^2 cos^2 dtheta = pi/8
I_22, _ = quad(lambda t: np.sin(t)**2 * np.cos(t)**2, -np.pi/2, np.pi/2)
print(f"  Intermediate: int sin^2 cos^2 = {I_22:.10f}  (expected pi/8 = {np.pi/8:.10f})")
assert abs(I_22 - np.pi/8) < 1e-10

# ========================================================================
# Part (c): Fourth moment m_4 = 2  (= Catalan number C_2)
# ========================================================================
print(f"\nPart (c): Fourth moment m_4 = 2  (Catalan number C_2)")
print("=" * 60)

m4, _ = quad(lambda x: x**4 * rho_sc(x), -2, 2)
print(f"  m_4 = {m4:.10f}  (expected 2)")
assert abs(m4 - 2) < 1e-10, f"m_4 = {m4}, expected 2"

# Verify the intermediate integral: int sin^4 cos^2 = pi/16
I_42, _ = quad(lambda t: np.sin(t)**4 * np.cos(t)**2, -np.pi/2, np.pi/2)
print(f"  Intermediate: int sin^4 cos^2 = {I_42:.10f}  (expected pi/16 = {np.pi/16:.10f})")
assert abs(I_42 - np.pi/16) < 1e-10

# Also verify the Catalan number pattern: m_{2k} = C_k
# C_0=1, C_1=1, C_2=2, C_3=5, C_4=14
catalan = [1, 1, 2, 5, 14]
print(f"\n  Catalan number check for moments m_{{2k}} = C_k:")
for k in range(5):
    mk, _ = quad(lambda x: x**(2*k) * rho_sc(x), -2, 2)
    print(f"    m_{2*k} = {mk:.6f}  (expected C_{k} = {catalan[k]})")
    assert abs(mk - catalan[k]) < 1e-6, f"m_{2*k} = {mk}, expected {catalan[k]}"

# ========================================================================
# Part (d): Kurtosis ratio m_4/m_2^2 = 2 (platykurtic)
# ========================================================================
print(f"\nPart (d): Kurtosis ratio m_4/m_2^2 = 2 < 3 (platykurtic)")
print("=" * 60)

ratio = m4 / m2**2
print(f"  m_4/m_2^2 = {ratio:.6f}  (expected 2)")
print(f"  Gaussian:    m_4/m_2^2 = 3.0  (mesokurtic)")
print(f"  Semicircle:  m_4/m_2^2 = 2.0  (platykurtic)")
print(f"  The semicircle has lighter tails because its support is")
print(f"  compact: eigenvalues are strictly bounded to [-2, 2].")
assert abs(ratio - 2) < 1e-10

