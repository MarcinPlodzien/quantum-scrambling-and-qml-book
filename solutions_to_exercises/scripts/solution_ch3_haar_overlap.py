#!/usr/bin/env python3
"""
###########################################################################
#  EXERCISE 3.1 — Overlap of Haar-Random States and Measure Concentration
#
#  Chapter 3, Haar Ensembles
#  Topic: Beta distribution, state overlaps, concentration of measure
#
#  ---------- EXERCISE STATEMENT ----------
#
#  The squared overlap chi = |<psi|phi>|^2 between two independent
#  Haar-random pure states in D dimensions follows:
#    P(chi) = (D-1)(1 - chi)^{D-2}   for chi in [0,1].
#
#  (a) Compute E[chi] and verify E[chi] = 1/D.
#  (b) Compute Var(chi).
#
#  ---------- ANALYTICAL SOLUTION ----------
#
#  (a) Using substitution u = 1-chi:
#      E[chi] = (D-1) int_0^1 (1-u) u^{D-2} du
#             = (D-1)[1/(D-1) - 1/D] = 1/D.
#
#  (b) E[chi^2] = (D-1) int_0^1 (1-u)^2 u^{D-2} du
#               = (D-1)[1/(D-1) - 2/D + 1/(D+1)]
#               = 2/(D(D+1)).
#      Var = 2/(D(D+1)) - 1/D^2 = (D-1)/(D^2(D+1)).
#      For large D: Var ~ 1/D^2.
#
#  ---------- NUMERICAL VERIFICATION ----------
###########################################################################
"""
import numpy as np
from scipy.stats import unitary_group

np.random.seed(42)

for D in [2, 4, 8, 16, 64]:
    n_samples = min(50000, max(5000, 200000 // D))
    overlaps = np.zeros(n_samples)

    for i in range(n_samples):
        # Two independent Haar-random states (first column of Haar-random U)
        psi = unitary_group.rvs(D)[:, 0]
        phi = unitary_group.rvs(D)[:, 0]
        overlaps[i] = abs(np.dot(psi.conj(), phi))**2

    mean_num = np.mean(overlaps)
    mean_ana = 1.0 / D
    var_num = np.var(overlaps)
    var_ana = (D - 1) / (D**2 * (D + 1))

    print(f"D={D:3d}: E[chi] = {mean_num:.5f} (expected {mean_ana:.5f}), "
          f"Var[chi] = {var_num:.2e} (expected {var_ana:.2e})")
    assert abs(mean_num - mean_ana) < 0.01, f"Mean mismatch at D={D}"
    assert abs(var_num - var_ana) / var_ana < 0.1, f"Var mismatch at D={D}"

print(f"\n  As D grows, both E[chi] and Var[chi] shrink: random states")
print(f"  in high dimensions are nearly orthogonal with high probability.")
print(f"  This is the concentration of measure phenomenon.")

