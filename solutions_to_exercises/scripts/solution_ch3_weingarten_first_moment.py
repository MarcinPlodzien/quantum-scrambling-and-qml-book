#!/usr/bin/env python3
"""
###########################################################################
#  EXERCISE 3.2 — Operator Averages using Weingarten Calculus
#
#  Chapter 3, Haar Ensembles
#  Topic: First-moment Haar average, Schur's lemma
#
#  ---------- EXERCISE STATEMENT ----------
#
#  The first moment formula: E_U[U A U^dag] = Tr(A)/D * I_D.
#  Let A, B be arbitrary DxD complex matrices.
#  Compute E_U[Tr(U A U^dag B)].
#
#  ---------- ANALYTICAL SOLUTION ----------
#
#  E_U[Tr(UAUB)] = Tr(E_U[UAU^dag] B) = Tr((Tr(A)/D)*I*B)
#                 = (Tr(A)/D)*Tr(B) = Tr(A)*Tr(B)/D.
#
#  The expectation of the product of traces factorizes:
#  The Haar average erases all correlations between A and B,
#  leaving only the trace information.
#
#  ---------- NUMERICAL VERIFICATION ----------
###########################################################################
"""
import numpy as np
from scipy.stats import unitary_group

np.random.seed(42)

for D in [2, 3, 4, 8]:
    n_samples = 20000

    # Random test matrices A and B
    A = np.random.randn(D, D) + 1j * np.random.randn(D, D)
    B = np.random.randn(D, D) + 1j * np.random.randn(D, D)

    # Analytical prediction
    prediction = np.trace(A) * np.trace(B) / D

    # Monte Carlo estimate
    mc_sum = 0.0
    for _ in range(n_samples):
        U = unitary_group.rvs(D)
        val = np.trace(U @ A @ U.conj().T @ B)
        mc_sum += val
    mc_estimate = mc_sum / n_samples

    print(f"D={D}: E[Tr(UAU^dag B)] = {mc_estimate:.4f}  "
          f"(predicted Tr(A)Tr(B)/D = {prediction:.4f})")
    assert abs(mc_estimate - prediction) < 0.15 * abs(prediction) + 0.1, \
        f"Mismatch at D={D}"

