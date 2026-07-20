#!/usr/bin/env python3
"""
###########################################################################
#  EXERCISE 3.3 — The Fourth Moment of the Trace
#
#  Chapter 3, Haar Ensembles
#  Topic: Weingarten calculus at k=2, fourth moment of |Tr(U)|
#
#  ---------- EXERCISE STATEMENT ----------
#
#  Using the k=2 Weingarten functions:
#    Wg(id, D) = 1/(D^2-1),  Wg((12), D) = -1/(D(D^2-1))
#  Compute E_U[|Tr(U)|^4] for Haar-random U in U(D).
#
#  ---------- ANALYTICAL SOLUTION ----------
#
#  |Tr(U)|^4 = sum_{i1,i2,l1,l2} U_{i1,i1} U_{i2,i2} U*_{l1,l1} U*_{l2,l2}
#
#  Setting j_a = i_a and m_a = l_a in the Weingarten formula:
#
#  Four (pi, sigma) contributions from S_2 x S_2:
#
#    pi=id,  sigma=id:    delta_{i1,l1}*delta_{i2,l2} -> sum = D^2.  Wg(id).
#    pi=(12), sigma=(12): delta_{i1,l2}*delta_{i2,l1} -> sum = D^2.  Wg(id).
#    pi=id,  sigma=(12):  all indices collapse -> sum = D.  Wg((12)).
#    pi=(12), sigma=id:   all indices collapse -> sum = D.  Wg((12)).
#
#  Total = 2*D^2/(D^2-1) + 2*D*(-1)/(D*(D^2-1))
#        = 2*D^2/(D^2-1) - 2/(D^2-1)
#        = 2*(D^2-1)/(D^2-1) = 2.
#
#  The answer is exactly 2 for all D >= 2.  Remarkable!
#
#  ---------- NUMERICAL VERIFICATION ----------
###########################################################################
"""
import numpy as np
from scipy.stats import unitary_group

np.random.seed(42)

print("E[|Tr(U)|^4] for Haar-random U(D)")
print("=" * 50)

for D in [2, 3, 4, 5, 8, 16]:
    n_samples = 50000
    fourth_moments = np.zeros(n_samples)

    for i in range(n_samples):
        U = unitary_group.rvs(D)
        fourth_moments[i] = abs(np.trace(U))**4

    mc_estimate = np.mean(fourth_moments)

    # Also verify the Weingarten functions explicitly
    Wg_id = 1.0 / (D**2 - 1)
    Wg_12 = -1.0 / (D * (D**2 - 1))
    analytical = 2 * D**2 * Wg_id + 2 * D * Wg_12

    print(f"  D={D:2d}: MC = {mc_estimate:.4f}, "
          f"analytical = {analytical:.4f}  (expected 2)")
    assert abs(analytical - 2) < 1e-12, \
        f"Analytical formula gives {analytical}, not 2"
    assert abs(mc_estimate - 2) < 0.1, \
        f"MC estimate {mc_estimate} too far from 2"

print(f"\n  The fourth moment is exactly 2 for all D >= 2.")
print(f"  This universal result shows that |Tr(U)| concentrates")
print(f"  around ~1 regardless of dimension.")

