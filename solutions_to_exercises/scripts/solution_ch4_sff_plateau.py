#!/usr/bin/env python3
"""
###########################################################################
#  EXERCISE 4.3 — SFF Plateau Value for the GUE
#
#  Chapter 4, Scrambling Dynamics
#  Topic: Spectral form factor, energy level statistics, dephasing
#
#  ---------- EXERCISE STATEMENT ----------
#
#  For a DxD GUE Hamiltonian with eigenvalues {E_n}:
#    K(t) = |Z(it)|^2 = sum_{m,n} exp(-it(E_m - E_n))
#
#  (a) Show the long-time average K_bar = D.
#  (b) Explain why K_plateau = D (not D^2).
#
#  ---------- ANALYTICAL SOLUTION ----------
#
#  (a) Time-average term by term:
#      - Diagonal (m=n): integrand = 1, contributes D terms.
#      - Off-diagonal (m!=n): (1/T) int_0^T exp(-i*omega*t) dt
#        = (1-exp(-i*omega*T))/(i*omega*T) -> 0 as T -> inf.
#      (GUE level repulsion ensures omega_{mn} != 0 for m != n.)
#      K_bar = D.
#
#  (b) At t=0: all D^2 terms contribute coherently -> K(0) = D^2.
#      At late times: off-diagonal dephase, only D diagonal survive.
#      K_plateau = D counts the number of energy levels.
#      The ratio K(0)/K_bar = D quantifies the coherent enhancement.
#
#  ---------- NUMERICAL VERIFICATION ----------
###########################################################################
"""
import numpy as np

np.random.seed(42)

print("Spectral form factor plateau for GUE Hamiltonians")
print("=" * 60)

for D in [4, 8, 16, 32]:
    # Generate a GUE Hamiltonian
    A = np.random.randn(D, D) + 1j * np.random.randn(D, D)
    H = (A + A.conj().T) / (2 * np.sqrt(D))
    energies = np.linalg.eigvalsh(H)

    # Compute K(t) at many time points
    # The SFF is K(t) = sum_{m,n} exp(-it(E_m - E_n))
    t_values = np.linspace(0, 500, 10000)
    K_values = np.zeros(len(t_values))

    for idx, t in enumerate(t_values):
        phases = np.exp(-1j * energies * t)
        Z = np.sum(phases)
        K_values[idx] = abs(Z)**2

    K_0 = K_values[0]
    K_bar = np.mean(K_values[1000:])  # skip early-time transient

    print(f"\n  D = {D}:")
    print(f"    K(0) = {K_0:.1f}  (expected D^2 = {D**2})")
    print(f"    K_bar (late-time avg) = {K_bar:.2f}  (expected D = {D})")
    assert abs(K_0 - D**2) < 0.5, f"K(0) = {K_0}, expected {D**2}"
    assert abs(K_bar - D) < D * 0.3, f"K_bar = {K_bar}, expected {D}"

    # Check all off-diagonal gaps are nonzero (level repulsion)
    gaps = np.diff(energies)
    min_gap = np.min(gaps)
    print(f"    min level spacing = {min_gap:.6f}  (nonzero: level repulsion)")
    assert min_gap > 0

print(f"\n  The plateau K = D counts the number of energy levels.")
print(f"  Off-diagonal terms dephase at late times, leaving only the")
print(f"  D diagonal contributions.  The early-time 'ramp' connecting")
print(f"  K(0) = D^2 to K_plateau = D encodes correlations between levels.")

