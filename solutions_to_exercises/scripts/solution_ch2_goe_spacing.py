#!/usr/bin/env python3
"""
###########################################################################
#  EXERCISE 2.1 — The 2x2 GOE and Linear Level Repulsion
#
#  Chapter 2, Random Matrix Theory
#  Topic: Gaussian Orthogonal Ensemble, Wigner surmise, level repulsion
#
#  ---------- EXERCISE STATEMENT ----------
#
#  Consider a 2x2 GOE matrix H = [[a, c], [c, b]], where a,b,c are
#  real random variables distributed under the GOE measure
#  P(H) ~ exp(-Tr(H^2)/2).  This gives a,b ~ N(0,1) and c ~ N(0,1/2).
#
#  (a) Write down the joint distribution P(a,b,c).
#  (b) Find the energy spacing s = lambda_+ - lambda_- in terms of a,b,c.
#  (c) Change variables to t=a+b, x=a-b, y=2c.  Integrate out the
#      trace and angle to prove P(s) ~ s * exp(-s^2/4).
#
#  ---------- ANALYTICAL SOLUTION ----------
#
#  (a) P(a,b,c) ~ exp(-(a^2 + b^2 + 2c^2)/2).
#      The factor 2 in front of c^2 comes from the off-diagonal
#      variance being halved: Var(c) = 1/2, so the exponent
#      (-c^2/(2*1/2)) = -c^2.  Combined with a^2/2 + b^2/2,
#      the total is -(a^2 + b^2 + 2c^2)/2.
#
#  (b) Eigenvalues: lambda_pm = (a+b)/2 +/- sqrt((a-b)^2/4 + c^2).
#      Spacing: s = sqrt((a-b)^2 + 4c^2).
#
#  (c) Let t = a+b, x = a-b, y = 2c.  Then:
#      a = (t+x)/2, b = (t-x)/2, c = y/2.
#      a^2 + b^2 + 2c^2 = (t^2 + x^2 + y^2)/2.
#      Jacobian |J| = 1/4 (the volume element scales by |da db dc / dt dx dy|).
#      P(t,x,y) ~ exp(-t^2/4) * exp(-(x^2+y^2)/4).
#      The trace variable t decouples and integrates to a constant.
#      The spacing is s = sqrt(x^2+y^2).
#      In polar coordinates: dx dy = s ds dtheta, and integrating theta:
#      P(s) ~ s * exp(-s^2/4).
#      The linear vanishing P(s) ~ s as s->0 reflects beta=1.
#
#  ---------- NUMERICAL VERIFICATION ----------
###########################################################################
"""
import numpy as np
from scipy.integrate import quad

# ========================================================================
# Part (a) & (b): Sample from the GOE and verify spacing distribution
# ========================================================================
print("Part (a)+(b): Sampling 2x2 GOE matrices")
print("=" * 60)

np.random.seed(42)
n_samples = 500000
spacings = np.zeros(n_samples)

for i in range(n_samples):
    # Draw from the GOE measure: a,b ~ N(0,1), c ~ N(0, 1/2)
    a = np.random.randn()
    b = np.random.randn()
    c = np.random.randn() * np.sqrt(0.5)

    # Eigenvalues of [[a, c], [c, b]]
    # spacing s = sqrt((a-b)^2 + 4c^2)
    spacings[i] = np.sqrt((a - b)**2 + 4 * c**2)

print(f"  Generated {n_samples} spacings from 2x2 GOE matrices.\n")

# ========================================================================
# Part (c): Compare with the Wigner surmise P(s) = (pi*s/2)*exp(-pi*s^2/4)
# Wait -- the exercise derives P(s) ~ s*exp(-s^2/4),
# which is the UNNORMALIZED version.  Let us normalize it.
# ========================================================================
print("Part (c): Verifying the spacing distribution P(s) ~ s*exp(-s^2/4)")
print("=" * 60)

# Normalization: int_0^inf s * exp(-s^2/4) ds = [-2*exp(-s^2/4)]_0^inf = 2
# So the normalized density is P(s) = (s/2) * exp(-s^2/4).
norm, _ = quad(lambda s: s * np.exp(-s**2 / 4), 0, np.inf)
print(f"  Normalization integral: {norm:.6f}  (expected 2)")
assert abs(norm - 2) < 1e-6, f"Normalization = {norm}, expected 2"

# Compare histogram moments with analytical predictions
# <s>   = int s * P_norm(s) ds = (1/2) int s^2 exp(-s^2/4) ds
#       = (1/2) * sqrt(pi) * 2 = sqrt(pi)  [Gaussian integral]
mean_s_analytical = np.sqrt(np.pi)
mean_s_numerical = np.mean(spacings)
print(f"\n  <s> numerical  = {mean_s_numerical:.4f}")
print(f"  <s> analytical = {mean_s_analytical:.4f}  (= sqrt(pi))")
assert abs(mean_s_numerical - mean_s_analytical) < 0.02, \
    f"Mean spacing mismatch: {mean_s_numerical} vs {mean_s_analytical}"

# <s^2> = (1/2) int s^3 exp(-s^2/4) ds = (1/2) * 2 * 4 = 4
# (using int_0^inf s^3 exp(-s^2/4) ds = 2*(4/2)^(4/2-1)*Gamma(2) ... 
#  or substitution u=s^2/4: int = 4*int u*exp(-u)*2du = 8)
mean_s2_analytical = 4.0
mean_s2_numerical = np.mean(spacings**2)
print(f"\n  <s^2> numerical  = {mean_s2_numerical:.4f}")
print(f"  <s^2> analytical = {mean_s2_analytical:.4f}")
assert abs(mean_s2_numerical - mean_s2_analytical) < 0.05, \
    f"Second moment mismatch: {mean_s2_numerical} vs {mean_s2_analytical}"

# Verify the Jacobian: a^2 + b^2 + 2c^2 = (t^2 + x^2 + y^2)/2
print("\n  Jacobian check: a^2+b^2+2c^2 = (t^2+x^2+y^2)/2")
for _ in range(100):
    a, b = np.random.randn(2)
    c = np.random.randn() * np.sqrt(0.5)
    t, x, y = a + b, a - b, 2 * c
    lhs = a**2 + b**2 + 2 * c**2
    rhs = (t**2 + x**2 + y**2) / 2
    assert abs(lhs - rhs) < 1e-12, f"Exponent transformation failed"
print("  100 random checks passed.")

# Check the Jacobian determinant = 1/4
# a = (t+x)/2, b = (t-x)/2, c = y/2
# J = [[da/dt, da/dx, da/dy], ...] = [[1/2, 1/2, 0], [1/2, -1/2, 0], [0, 0, 1/2]]
# det(J) = (1/2)(-1/4 - 0) - (1/2)(1/4 - 0) + 0 = -1/4
J = np.array([[0.5, 0.5, 0], [0.5, -0.5, 0], [0, 0, 0.5]])
det_J = np.linalg.det(J)
print(f"  |det(J)| = {abs(det_J):.4f}  (expected 1/4 = 0.25)")
assert abs(abs(det_J) - 0.25) < 1e-12

print(f"\n  The linear vanishing P(s) ~ s near s=0 reflects the Dyson")
print(f"  index beta=1: only 1 real parameter (the off-diagonal c)")
print(f"  connects the two levels, so the avoidance space is 1D.")

