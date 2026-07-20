#!/usr/bin/env python3
"""
###########################################################################
#  EXERCISE 3.4 — The Asymptotic Page Formula via Marchenko-Pastur
#
#  Chapter 3, Haar Ensembles
#  Topic: Page's theorem, Marchenko-Pastur law, entanglement entropy
#
#  ---------- EXERCISE STATEMENT ----------
#
#  Derive the asymptotic Page formula: E[S(rho_A)] ~ ln(m) - m/(2n)
#  by treating eigenvalues as following the Marchenko-Pastur law.
#
#  ---------- ANALYTICAL SOLUTION ----------
#
#  Step 1: Using lambda = x/m and the MP law with mean x = 1, variance c = m/n:
#    S = ln(m) - int x ln(x) rho_MP(x) dx.
#
#  Step 2: Taylor expand f(x) = x ln(x) around x=1:
#    f(1) = 0, f'(1) = 1, f''(1) = 1.
#    f(x) ~ (x-1) + (1/2)(x-1)^2.
#
#  Step 3: Integrate:
#    int x ln(x) rho_MP dx ~ <x-1> + (1/2)<(x-1)^2>
#                           = 0 + (1/2)*Var(x) = m/(2n).
#    S ~ ln(m) - m/(2n).
#
#  ---------- NUMERICAL VERIFICATION ----------
###########################################################################
"""
import numpy as np
from scipy.stats import unitary_group

np.random.seed(42)

print("Asymptotic Page formula: E[S] ~ ln(m) - m/(2n)")
print("=" * 60)

for m, n in [(2, 2), (4, 4), (8, 8), (16, 16), (4, 16), (8, 64)]:
    D = m * n  # total Hilbert space dimension
    n_samples = min(5000, max(500, 100000 // D))

    entropies = []
    for _ in range(n_samples):
        # Generate Haar-random state on C^D
        psi = unitary_group.rvs(D)[:, 0]
        # Reduced density matrix: rho_A = Tr_B(|psi><psi|)
        psi_mat = psi.reshape(m, n)
        rho_A = psi_mat @ psi_mat.conj().T
        # Von Neumann entropy
        eigs = np.linalg.eigvalsh(rho_A)
        eigs = eigs[eigs > 1e-15]
        S = -np.sum(eigs * np.log(eigs))
        entropies.append(S)

    S_numerical = np.mean(entropies)
    S_page = np.log(m) - m / (2 * n)

    print(f"  m={m:3d}, n={n:3d}: E[S] = {S_numerical:.4f}  "
          f"(Page: ln({m})-{m}/{2*n} = {S_page:.4f})")

print(f"\n  The approximation improves as m,n grow (continuum limit).")
print(f"  For m=n, the Page correction m/(2n)=1/2 is independent of m,")
print(f"  giving the asymptotic 'half-nat' entropy deficit below ln(m).")

