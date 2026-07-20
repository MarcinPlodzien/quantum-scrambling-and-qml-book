#!/usr/bin/env python3
"""
###########################################################################
#  EXERCISE 7.1 — Barren Plateau Scaling from the Weingarten Formula
#
#  Chapter 7, Quantum Machine Learning
#  Topic: Variational circuits, gradient variance, 2-designs, shot cost
#
#  ---------- EXERCISE STATEMENT ----------
#
#  A parametrized circuit forms a unitary 2-design on N qubits.
#  Cost: C(theta) = <0|U^dag O U|0>,  O = sum_j Z_j  (traceless, Tr(O^2)=ND).
#
#  (a) Compute Var[dC/dtheta_i] and confirm exponential suppression.
#  (b) Determine the shot cost nu to resolve the gradient.
#  (c) For per-gate error p=10^{-3}, estimate critical depth L_crit.
#
#  ---------- ANALYTICAL SOLUTION ----------
#
#  (a) Var[dC/dtheta_i] ~ Tr(O^2)/(2*D^2) = ND/(2*D^2) = N/(2*D)
#                        = N * 2^{-(N+1)}.
#      Exponentially small in N.
#
#  (b) Shot variance: Var_shot[g_i] = N/(2*nu).
#      Require: N/(2*nu) < N/(2*D)  =>  nu > D = 2^N.
#      The N cancels exactly: shot cost scales as 2^N.
#
#  (c) L_crit ~ 1/p = 10^3.
#      Light-cone diameter in 1D brickwork: ~L_crit = 1000 qubits.
#
#  ---------- NUMERICAL VERIFICATION ----------
###########################################################################
"""
import numpy as np
from scipy.stats import unitary_group

np.random.seed(42)

print("Barren plateau gradient variance for 2-design circuits")
print("=" * 60)

for N in [2, 3, 4, 5, 6]:
    D = 2**N

    # Observable: O = sum_j Z_j
    O = np.zeros((D, D), dtype=complex)
    for j in range(N):
        Zj = np.eye(1, dtype=complex)
        for k in range(N):
            if k == j:
                Zj = np.kron(Zj, np.array([[1,0],[0,-1]], dtype=complex))
            else:
                Zj = np.kron(Zj, np.eye(2, dtype=complex))
        O += Zj

    trO2 = np.trace(O @ O).real
    prediction = trO2 / (2 * D**2)

    # Monte Carlo: sample Haar-random unitaries (exact 2-design)
    # and compute cost function. Variance estimated from samples.
    n_samples = 5000
    psi0 = np.zeros(D, dtype=complex)
    psi0[0] = 1.0

    costs = []
    for _ in range(n_samples):
        U = unitary_group.rvs(D)
        psi = U @ psi0
        cost = (psi.conj() @ O @ psi).real
        costs.append(cost)

    var_cost = np.var(costs)

    # The cost variance is related to but not identical to the gradient
    # variance. For 2-designs, Var[C] ~ Tr(O^2)/(D^2-1) ~ Tr(O^2)/D^2
    # (the gradient variance has an additional factor ~1/2 from the
    # parameter-shift structure, but scales identically).

    print(f"  N={N}: Tr(O^2)={trO2:.0f}, Var[C] = {var_cost:.6f}  "
          f"(~ Tr(O^2)/D^2 = {trO2/D**2:.6f})")

# Show the exponential scaling
print(f"\n  Gradient variance scaling: Var ~ N * 2^{{-(N+1)}}")
print(f"  {'N':>4s}  {'Var (predicted)':>16s}")
for N in range(2, 16):
    D = 2**N
    var_pred = N / (2 * D)
    print(f"  {N:4d}  {var_pred:16.2e}")

print(f"\n  Shot cost scaling:")
print(f"  Require nu > D = 2^N shots per gradient component.")
print(f"  For N=20: nu > {2**20:,} ~ 10^6 shots per parameter.")

print(f"\n  Critical depth (p = 10^{{-3}}):")
print(f"  L_crit ~ 1/p = 1000 layers.")
print(f"  Light-cone diameter in 1D brickwork: ~1000 qubits.")

