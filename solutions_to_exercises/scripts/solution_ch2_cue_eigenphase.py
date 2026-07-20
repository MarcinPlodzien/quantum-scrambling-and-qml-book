#!/usr/bin/env python3
"""
###########################################################################
#  EXERCISE 2.3 — Eigenphase Repulsion in the 2x2 CUE
#
#  Chapter 2, Random Matrix Theory
#  Topic: Circular Unitary Ensemble, level repulsion, eigenphase statistics
#
#  ---------- EXERCISE STATEMENT ----------
#
#  A 2x2 CUE matrix (Haar measure on U(2)) has eigenvalues
#  exp(i*theta_1), exp(i*theta_2) with joint density:
#    P(theta_1, theta_2) ~ |exp(i*theta_1) - exp(i*theta_2)|^2
#
#  (a) Show |exp(i*t1) - exp(i*t2)|^2 = 2 - 2*cos(t1 - t2).
#  (b) With phi = t1 - t2, show P(phi) = (1 - cos(phi))/(2*pi).
#  (c) Verify P(0) = 0 (eigenphases cannot coincide).
#      Find the most probable spacing.
#
#  ---------- ANALYTICAL SOLUTION ----------
#
#  (a) |e^{it1} - e^{it2}|^2 = (e^{it1}-e^{it2})(e^{-it1}-e^{-it2})
#                              = 2 - e^{i(t1-t2)} - e^{-i(t1-t2)}
#                              = 2 - 2*cos(t1-t2).
#
#  (b) P depends only on phi = t1-t2.  Integrating over the
#      center-of-mass Theta = (t1+t2)/2 gives factor 2*pi.
#      Normalization: Z = int_0^{2pi} (2-2cos(phi)) dphi = 4*pi.
#      P(phi) = (2-2cos(phi))/(4*pi) = (1-cos(phi))/(2*pi).
#
#  (c) P(0) = 0: level repulsion forbids coinciding eigenphases.
#      Maximum at cos(phi) = -1, i.e. phi = pi.
#      Eigenphases prefer to sit diametrically opposite on the
#      unit circle -- the maximally separated configuration.
#
#  ---------- NUMERICAL VERIFICATION ----------
###########################################################################
"""
import numpy as np
from scipy.integrate import quad
from scipy.stats import unitary_group

# ========================================================================
# Part (a): Algebraic identity
# ========================================================================
print("Part (a): |e^{it1} - e^{it2}|^2 = 2 - 2*cos(t1-t2)")
print("=" * 60)

for t1, t2 in [(0.5, 1.3), (np.pi, 0), (2.7, 4.1)]:
    lhs = abs(np.exp(1j*t1) - np.exp(1j*t2))**2
    rhs = 2 - 2*np.cos(t1 - t2)
    print(f"  t1={t1:.2f}, t2={t2:.2f}: LHS={lhs:.6f}, RHS={rhs:.6f}")
    assert abs(lhs - rhs) < 1e-12, "Identity failed!"

# ========================================================================
# Part (b): Marginal distribution P(phi) = (1-cos(phi))/(2*pi)
# ========================================================================
print(f"\nPart (b): Normalization and marginal density")
print("=" * 60)

# Verify normalization
norm, _ = quad(lambda phi: (1 - np.cos(phi)) / (2*np.pi), 0, 2*np.pi)
print(f"  int P(phi) dphi = {norm:.10f}  (expected 1)")
assert abs(norm - 1) < 1e-10, f"Normalization = {norm}"

# Verify normalization of the pre-factor
Z, _ = quad(lambda phi: 2 - 2*np.cos(phi), 0, 2*np.pi)
print(f"  Z = int (2-2cos) dphi = {Z:.6f}  (expected 4*pi = {4*np.pi:.6f})")
assert abs(Z - 4*np.pi) < 1e-6

# Compare with Monte Carlo sampling from U(2)
print(f"\n  Monte Carlo comparison with 50000 Haar-random U(2) matrices:")
np.random.seed(42)
phase_diffs = []
for _ in range(50000):
    U = unitary_group.rvs(2)
    phases = np.angle(np.linalg.eigvals(U))
    diff = (phases[1] - phases[0]) % (2*np.pi)
    phase_diffs.append(diff)

phase_diffs = np.array(phase_diffs)

# Compare mean of phi with analytical prediction
# <phi> = int phi * (1-cos(phi))/(2pi) dphi from 0 to 2pi
mean_analytical, _ = quad(lambda phi: phi * (1 - np.cos(phi)) / (2*np.pi), 0, 2*np.pi)
mean_numerical = np.mean(phase_diffs)
print(f"  <phi> numerical  = {mean_numerical:.4f}")
print(f"  <phi> analytical = {mean_analytical:.4f}")
assert abs(mean_numerical - mean_analytical) < 0.1

# ========================================================================
# Part (c): P(0) = 0 and most probable spacing = pi
# ========================================================================
print(f"\nPart (c): Level repulsion at phi=0 and peak at phi=pi")
print("=" * 60)

P_0 = (1 - np.cos(0)) / (2*np.pi)
P_pi = (1 - np.cos(np.pi)) / (2*np.pi)
print(f"  P(0)  = {P_0:.6f}  (expected 0: eigenphases cannot coincide)")
print(f"  P(pi) = {P_pi:.6f}  (expected 1/pi = {1/np.pi:.6f})")
assert abs(P_0) < 1e-15, "P(0) should be exactly 0"
assert abs(P_pi - 1/np.pi) < 1e-10

# Verify that the mode (most probable spacing) is near pi in the MC data
hist, bin_edges = np.histogram(phase_diffs, bins=50)
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
mode_idx = np.argmax(hist)
mode_numerical = bin_centers[mode_idx]
print(f"  Mode (numerical) = {mode_numerical:.4f}  (expected pi = {np.pi:.4f})")
assert abs(mode_numerical - np.pi) < 0.3, f"Mode = {mode_numerical}, expected pi"

print(f"\n  Physical interpretation: eigenphases prefer diametrically")
print(f"  opposite positions on the unit circle (phi = pi).")
print(f"  This is the CUE analogue of level repulsion in the GUE.")

