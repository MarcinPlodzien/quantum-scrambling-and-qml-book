#!/usr/bin/env python3
"""
###########################################################################
#  EXERCISE 7.4 — QELM Feature Concentration under Haar Scrambling
#
#  Chapter 7, Quantum Machine Learning
#  Topic: Quantum extreme learning machines, feature maps, concentration
#
#  ---------- EXERCISE STATEMENT ----------
#
#  A QELM on N_in + N_res qubits (D = 2^{N_in+N_res}) processes input
#  rho_in(x) via a Haar-random U, then measures computationally.
#
#  (a) Show E_U[E_b] = I_in/D, so E[f_b(x)] = 1/D (input-independent).
#  (b) Show Var_U[f_b] = (D-1)/(D^2(D+1)).
#  (c) Shot cost: nu > D to resolve features from shot noise.
#
#  ---------- ANALYTICAL SOLUTION ----------
#
#  (a) E[U^dag Pi_b U] = Tr(Pi_b)/D * I = I/D.
#      Sandwiching with <0_res| and |0_res> gives E[E_b] = I_in/D.
#      E[f_b] = Tr(I_in * rho_in)/D = 1/D.
#
#  (b) E[sum_b f_b^2] = (Tr(sigma^2)+1)/(D+1) = 2/(D+1) for pure input.
#      By Haar symmetry: E[f_b^2] = 2/(D(D+1)).
#      Var = 2/(D(D+1)) - 1/D^2 = (D-1)/(D^2(D+1)).
#
#  (c) Shot noise ~ 1/sqrt(D*nu).  Feature fluctuation ~ 1/D.
#      Require 1/sqrt(D*nu) < 1/D  =>  nu > D.
#
#  ---------- NUMERICAL VERIFICATION ----------
###########################################################################
"""
import numpy as np
from scipy.stats import unitary_group

np.random.seed(42)

print("QELM feature concentration under Haar scrambling")
print("=" * 60)

for N_in, N_res in [(1, 1), (1, 2), (2, 2)]:
    N_total = N_in + N_res
    D = 2**N_total
    D_in = 2**N_in
    n_samples = 5000

    # Two different input states to test input-independence
    psi_0 = np.zeros(D_in, dtype=complex); psi_0[0] = 1
    psi_1 = np.ones(D_in, dtype=complex) / np.sqrt(D_in)

    res_state = np.zeros(2**N_res, dtype=complex); res_state[0] = 1

    for label, psi_in in [("psi=|0>", psi_0), ("psi=|+>", psi_1)]:
        sigma = np.kron(np.outer(psi_in, psi_in.conj()),
                       np.outer(res_state, res_state.conj()))

        features = []
        for _ in range(n_samples):
            U = unitary_group.rvs(D)
            rho_out = U @ sigma @ U.conj().T
            # Born probabilities = diagonal of rho_out
            probs = np.diag(rho_out).real
            features.append(probs)

        features = np.array(features)  # shape (n_samples, D)

        # Part (a): E[f_b] should be 1/D for all b
        mean_features = np.mean(features, axis=0)
        mean_pred = 1.0 / D

        # Part (b): Var[f_b] should be (D-1)/(D^2*(D+1))
        var_features = np.var(features, axis=0)
        var_pred = (D - 1) / (D**2 * (D + 1))

        print(f"\n  N_in={N_in}, N_res={N_res} (D={D}), {label}:")
        print(f"    E[f_b] = {np.mean(mean_features):.6f}  "
              f"(predicted 1/D = {mean_pred:.6f})")
        print(f"    Var[f_b] = {np.mean(var_features):.2e}  "
              f"(predicted (D-1)/(D^2(D+1)) = {var_pred:.2e})")

        assert abs(np.mean(mean_features) - mean_pred) < 0.01
        assert abs(np.mean(var_features) - var_pred) / var_pred < 0.2

# Part (c): Shot cost scaling
print(f"\n  Shot cost analysis:")
print(f"  {'N_total':>8s}  {'D':>8s}  {'nu_min':>10s}")
for N_total in range(2, 16):
    D = 2**N_total
    print(f"  {N_total:8d}  {D:8d}  {D:10d}")

print(f"\n  An exponential number of shots is required to resolve any")
print(f"  input-dependent signal from a Haar-scrambled QELM.")
print(f"  This motivates operating below full scrambling.")

