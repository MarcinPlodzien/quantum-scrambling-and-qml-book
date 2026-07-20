#!/usr/bin/env python3
"""
###########################################################################
#  EXERCISE 7.2 — Local vs Global Cost Functions in Deep Circuits
#
#  Chapter 7, Quantum Machine Learning
#  Topic: Trainability, cost function design, gradient landscape
#
#  ---------- EXERCISE STATEMENT ----------
#
#  For a 2-design circuit on N qubits, compare two traceless cost observables:
#    H_G = |0><0|^{xN} - I/D   (global projector)
#    H_L = (1/N) sum_j Z_j     (normalized local observable)
#
#  (a) Compute Tr(H_G^2) and Tr(H_L^2).
#  (b) Determine asymptotic gradient variance scaling for both.
#  (c) Find ratio Var_L/Var_G and interpret.
#
#  ---------- ANALYTICAL SOLUTION ----------
#
#  (a) H_G^2 ~ |0><0|^{xN} for D >> 1.   Tr(H_G^2) ~ 1.
#      H_L^2 = (1/N^2) sum_{i,j} Z_i Z_j.
#      Cross terms: Tr(Z_i Z_j) = 0 for i != j.
#      Diagonal: Tr(Z_j^2) = D.  So Tr(H_L^2) = D/N.
#
#  (b) Var ~ Tr(H^2)/(2D^2).
#      Var_G ~ 1/(2D^2) = O(2^{-2N}).
#      Var_L ~ (D/N)/(2D^2) = 1/(2ND) = O(1/(N*2^N)).
#
#  (c) Var_L/Var_G ~ D/N = 2^N/N >> 1.
#      Local cost has exponentially larger gradient -- decisive advantage.
#
#  ---------- NUMERICAL VERIFICATION ----------
###########################################################################
"""
import numpy as np
from scipy.stats import unitary_group

np.random.seed(42)

print("Local vs Global cost function gradient scaling")
print("=" * 60)

for N in [2, 3, 4, 5, 6]:
    D = 2**N

    # Construct H_G = |0><0|^{xN} - I/D
    ket0 = np.zeros(D, dtype=complex)
    ket0[0] = 1
    H_G = np.outer(ket0, ket0) - np.eye(D) / D

    # Construct H_L = (1/N) sum_j Z_j
    H_L = np.zeros((D, D), dtype=complex)
    for j in range(N):
        Zj = np.eye(1, dtype=complex)
        for k in range(N):
            if k == j:
                Zj = np.kron(Zj, np.array([[1,0],[0,-1]], dtype=complex))
            else:
                Zj = np.kron(Zj, np.eye(2, dtype=complex))
        H_L += Zj / N

    trHG2 = np.trace(H_G @ H_G).real
    trHL2 = np.trace(H_L @ H_L).real

    # Monte Carlo variance estimation
    n_samples = 5000
    costs_G, costs_L = [], []
    for _ in range(n_samples):
        U = unitary_group.rvs(D)
        psi = U @ ket0
        costs_G.append((psi.conj() @ H_G @ psi).real)
        costs_L.append((psi.conj() @ H_L @ psi).real)

    var_G = np.var(costs_G)
    var_L = np.var(costs_L)

    pred_G = trHG2 / D**2   # ~ Tr(H^2)/(D^2-1) for 2-design
    pred_L = trHL2 / D**2

    print(f"\n  N={N} (D={D}):")
    print(f"    Tr(H_G^2) = {trHG2:.4f} (expected ~1)")
    print(f"    Tr(H_L^2) = {trHL2:.4f} (expected D/N = {D/N:.4f})")
    print(f"    Var_G = {var_G:.2e}  (pred ~ {pred_G:.2e})")
    print(f"    Var_L = {var_L:.2e}  (pred ~ {pred_L:.2e})")
    if var_G > 0:
        print(f"    Var_L/Var_G = {var_L/var_G:.1f}  (expected ~ D/N = {D/N:.1f})")

print(f"\n  The local cost provides an exponentially larger gradient signal.")
print(f"  While both vanish in the 2-design limit, the local cost has")
print(f"  half the exponent -- a decisive practical advantage.")

