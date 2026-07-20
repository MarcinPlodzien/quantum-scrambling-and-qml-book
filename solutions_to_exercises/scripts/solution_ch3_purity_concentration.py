#!/usr/bin/env python3
"""
###########################################################################
#  EXERCISE 3.6 — Concentration of Purity
#
#  Chapter 3, Haar Ensembles
#  Topic: Levy's lemma, Lipschitz functions, concentration of measure
#
#  ---------- EXERCISE STATEMENT ----------
#
#  The purity gamma = Tr(rho_A^2) has mean E[gamma] = (m+n)/(mn+1).
#  The function f(psi) = Tr(rho_A^2) is Lipschitz with eta_f = 4.
#
#  (a) Using Levy's lemma, write Pr(|gamma - E[gamma]| > eps).
#  (b) For m=n=2^{N/2}: find N such that Pr(gamma > E[gamma]+0.01) < 10^{-6}.
#
#  ---------- ANALYTICAL SOLUTION ----------
#
#  (a) Pr(|gamma - E[gamma]| > eps) <= 2 exp(-c D eps^2 / 16)
#      where c > 0 is an order-unity constant and D = m*n.
#
#  (b) For equal bipartition: D = 2^N, E[gamma] ~ 2/m = 2^{1-N/2}.
#      Require 2 exp(-c 2^N * 10^{-4} / 16) < 10^{-6}.
#      With c~1: 2^N > (16/10^{-4}) * ln(2e6) ~ 2.3e6.
#      Since 2^{21} ~ 2.1e6 and 2^{22} ~ 4.2e6, need N >= 22.
#
#  ---------- NUMERICAL VERIFICATION ----------
#
#  We cannot easily verify Levy's lemma numerically at N=22 (Hilbert
#  space too large), but we can verify concentration at small N by
#  sampling many Haar-random states and checking the purity distribution.
###########################################################################
"""
import numpy as np
from scipy.stats import unitary_group

np.random.seed(42)

print("Concentration of purity for Haar-random states")
print("=" * 60)

for N in [4, 6, 8]:
    m = 2**(N // 2)
    n = m
    D = m * n
    n_samples = min(5000, max(200, 50000 // D))

    gamma_mean_theory = (m + n) / (m * n + 1)

    purities = []
    for _ in range(n_samples):
        psi = unitary_group.rvs(D)[:, 0]
        psi_mat = psi.reshape(m, n)
        rho_A = psi_mat @ psi_mat.conj().T
        gamma = np.trace(rho_A @ rho_A).real
        purities.append(gamma)

    purities = np.array(purities)
    gamma_mean_num = np.mean(purities)
    gamma_std_num = np.std(purities)

    # Fraction exceeding E[gamma] + 0.01
    n_exceed = np.sum(purities > gamma_mean_theory + 0.01)
    frac_exceed = n_exceed / n_samples

    print(f"\n  N={N:2d} (D={D:5d}, m=n={m:3d}):")
    print(f"    E[gamma] theory = {gamma_mean_theory:.6f}")
    print(f"    E[gamma] MC     = {gamma_mean_num:.6f}")
    print(f"    std(gamma)      = {gamma_std_num:.6f}")
    print(f"    Pr(gamma > E+0.01) = {frac_exceed:.4f}  ({n_exceed}/{n_samples})")

# Verify the Levy bound parameters
print(f"\n" + "=" * 60)
print(f"Part (b): Minimum N for Pr(gamma > E+0.01) < 10^{{-6}}")
print(f"=" * 60)
print(f"  Levy's lemma: Pr(|gamma-E| > eps) <= 2 exp(-c D eps^2/eta_f^2)")
print(f"  eta_f = 4 (Lipschitz constant), eps = 0.01, c ~ 1")
print(f"  Require 2 exp(-c 2^N * 10^{{-4}} / 16) < 10^{{-6}}")
print(f"  => c * 2^N / 1.6e5 > ln(2e6) ~ 14.5")
print(f"  => 2^N > 2.3e6  (with c=1)")
print(f"  2^21 = {2**21:,}  (too small)")
print(f"  2^22 = {2**22:,}  (sufficient)")
print(f"  => N >= 22 qubits")

