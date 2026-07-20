#!/usr/bin/env python3
"""
###########################################################################
###########################################################################
##
##  SOLUTION: Bloch Sphere Geodesics and the Fubini-Study Metric
##
##  Book:     Information Scrambling and the Haar Measure
##  Chapter:  1 -- Mathematical Foundations
##  Section:  1.2 -- Geometry of Quantum State Space
##
###########################################################################
##
##  EXERCISE STATEMENT
##  ------------------
##
##  A general single-qubit pure state in spherical coordinates reads
##
##      |ψ(θ, φ)> = cos(θ/2) |0> + e^{iφ} sin(θ/2) |1>.
##
##  (a)  Starting from the Fubini-Study infinitesimal line element
##
##          ds²_FS  =  <dψ | dψ>  -  |<ψ | dψ>|²,
##
##       show that
##
##          ds²_FS  =  (1/4)( dθ²  +  sin²(θ) dφ² ).
##
##       Interpret the result: the Bloch sphere carries a round metric
##       of radius R = 1/2.
##
##  (b)  Compute the geodesic distance between the north pole |0> and
##       the equatorial state |+> = (|0> + |1>)/√2.
##
###########################################################################
##
##  ANALYTICAL SOLUTION
##  -------------------
##
##  (a)  Differentiating |ψ>:
##
##       |dψ> = -(1/2) sin(θ/2) dθ |0>
##            + e^{iφ} [ (1/2) cos(θ/2) dθ + i sin(θ/2) dφ ] |1>
##
##       Term 1:  <dψ|dψ>
##         = (1/4) sin²(θ/2) dθ² + (1/4) cos²(θ/2) dθ² + sin²(θ/2) dφ²
##         = (1/4) dθ²  +  sin²(θ/2) dφ²
##
##       Term 2:  <ψ|dψ>
##         The real part (dθ terms) cancels:
##           cos(θ/2)·[-(1/2)sin(θ/2)] + sin(θ/2)·[(1/2)cos(θ/2)] = 0.
##         Only the imaginary part survives:
##           <ψ|dψ> = i sin²(θ/2) dφ.
##         Hence |<ψ|dψ>|² = sin⁴(θ/2) dφ².
##
##       Subtraction:
##         ds²_FS = (1/4) dθ² + [sin²(θ/2) - sin⁴(θ/2)] dφ²
##                = (1/4) dθ² + sin²(θ/2) cos²(θ/2) dφ²
##                = (1/4) dθ² + (1/4) sin²(θ) dφ²
##
##       where we used 4 sin²(θ/2) cos²(θ/2) = sin²(θ).
##       This is the round metric on S² with radius R = 1/2.
##
##  (b)  d_FS(|0>, |+>) = arccos |<0|+>| = arccos(1/√2) = π/4.
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
print("  Symbolic derivation of the Fubini-Study metric on the Bloch sphere")
print("=" * 70)

theta = sp.Symbol('theta', real=True, positive=True)
dt, dp = sp.symbols('dtheta dphi', real=True)

# --- Step 1: Components of |dψ> ---
print("\n  Step 1: Differential of the state vector |dψ>")
print("  " + "-" * 50)

dpsi_0 = -sp.Rational(1, 2) * sp.sin(theta / 2) * dt
dpsi_1_re = sp.Rational(1, 2) * sp.cos(theta / 2) * dt
dpsi_1_im = sp.sin(theta / 2) * dp

print(f"    dψ_0         = {dpsi_0}")
print(f"    Re(dψ_1)     = {dpsi_1_re}")
print(f"    Im(dψ_1)     = {dpsi_1_im}")

# --- Step 2: <dψ|dψ> ---
print("\n  Step 2: First term  <dψ|dψ> = |dψ_0|² + |dψ_1|²")
print("  " + "-" * 50)

term1 = sp.expand(dpsi_0**2 + dpsi_1_re**2 + dpsi_1_im**2)
term1 = sp.trigsimp(term1)
print(f"    <dψ|dψ> = {term1}")

coeff_dt2_t1 = sp.trigsimp(term1.coeff(dt**2))
coeff_dp2_t1 = sp.trigsimp(term1.coeff(dp**2))
print(f"    Coefficient of dθ²:  {coeff_dt2_t1}")
print(f"    Coefficient of dφ²:  {coeff_dp2_t1}")

# --- Step 3: <ψ|dψ> and |<ψ|dψ>|² ---
print("\n  Step 3: Second term  |<ψ|dψ>|²")
print("  " + "-" * 50)

inner_re = sp.expand(sp.cos(theta/2) * dpsi_0 + sp.sin(theta/2) * dpsi_1_re)
inner_re = sp.trigsimp(inner_re)
inner_im = sp.sin(theta/2) * dpsi_1_im

print(f"    Re(<ψ|dψ>) = {inner_re}")
print(f"    Im(<ψ|dψ>) = {inner_im}")
print(f"    The dθ terms cancel: <ψ|dψ> is purely imaginary.")

term2 = sp.expand(inner_re**2 + inner_im**2)
term2 = sp.trigsimp(term2)
print(f"    |<ψ|dψ>|²  = {term2}")

# --- Step 4: Subtraction → ds²_FS ---
print("\n  Step 4: Fubini-Study metric  ds² = <dψ|dψ> − |<ψ|dψ>|²")
print("  " + "-" * 50)

ds2 = sp.trigsimp(term1 - term2)
print(f"    ds²_FS = {ds2}")

coeff_dt2 = sp.trigsimp(ds2.coeff(dt**2))
coeff_dp2 = sp.trigsimp(ds2.coeff(dp**2))
print(f"    Coefficient of dθ²:  {coeff_dt2}")
print(f"    Coefficient of dφ²:  {coeff_dp2}")

# Verify dφ² coefficient equals (1/4)sin²(θ)
target_dp2 = sp.Rational(1, 4) * sp.sin(theta)**2
diff = sp.trigsimp(coeff_dp2 - target_dp2)
print(f"\n    Verification: coeff(dφ²) − (1/4)sin²(θ) = {diff}")
assert diff == 0, f"Symbolic mismatch: diff = {diff}"

print(f"\n    RESULT:  ds²_FS = (1/4) dθ² + (1/4) sin²(θ) dφ²")
print(f"                    = (1/4) [ dθ² + sin²(θ) dφ² ]")
print(f"    Round metric on S² with radius R = 1/2.")

# --- Step 5: Geodesic distance ---
print("\n  Step 5: Geodesic distance d_FS(|0>, |+>)")
print("  " + "-" * 50)

overlap = sp.sqrt(sp.Rational(1, 2))
d_FS_sym = sp.acos(overlap)
print(f"    |<0|+>| = 1/√2")
print(f"    d_FS    = arccos(1/√2) = {sp.simplify(d_FS_sym)}")

# ========================================================================
#  SECTION 2 :  Numerical verification with NumPy
# ========================================================================

print("\n" + "=" * 70)
print("  Numerical verification at specific angles")
print("=" * 70)

print("\n  Verifying the dφ² coefficient at selected angles:")
print(f"  sin²(θ/2) − sin⁴(θ/2) = sin²(θ/2)cos²(θ/2) = (1/4)sin²(θ)")
print(f"  {'θ':>10s}  {'sin²−sin⁴':>12s}  {'sin²cos²':>12s}  {'(1/4)sin²θ':>12s}")

for t in [np.pi/6, np.pi/4, np.pi/3, np.pi/2, 2*np.pi/3, np.pi]:
    s2 = np.sin(t/2)**2
    val1 = s2 - s2**2
    val2 = s2 * np.cos(t/2)**2
    val3 = 0.25 * np.sin(t)**2
    print(f"  {t:10.4f}  {val1:12.8f}  {val2:12.8f}  {val3:12.8f}")
    assert abs(val1 - val2) < 1e-14 and abs(val2 - val3) < 1e-14

d_FS_num = np.arccos(1/np.sqrt(2))
print(f"\n  d_FS(|0>,|+>) = {d_FS_num:.10f}")
print(f"  π/4           = {np.pi/4:.10f}")
assert abs(d_FS_num - np.pi/4) < 1e-14

print(f"\n  The Bloch sphere has radius R = 1/2.")
print(f"  Total geodesic circumference = 2πR = π.")
print(f"  The distance π/4 between |0> and |+> spans one-eighth of")
print(f"  this circumference (90° angular separation on the Bloch sphere).")
