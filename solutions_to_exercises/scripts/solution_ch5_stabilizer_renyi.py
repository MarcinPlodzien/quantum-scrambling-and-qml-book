#!/usr/bin/env python3
"""
###########################################################################
#  EXERCISE 5.2 — Stabilizer Renyi Entropy of Haar-Random States
#
#  Chapter 5, Unitary Designs
#  Topic: Magic (nonstabilizerness), SRE, Pauli spectrum, Gottesman-Knill
#
#  ---------- EXERCISE STATEMENT ----------
#
#  The second SRE is M_2(psi) = -log2[ (1/D) sum_P <P>^4 ].
#
#  (a) Show E_Haar[<A>^4] = 3/((D+1)(D+3)) for traceless A^2=I.
#  (b) Show E[Xi_P] = 4/(D+3) where Xi_P = (1/D) sum_P <P>^4.
#  (c) -log2(E[Xi_P]) = N-2 + O(2^{-N}).
#  (d) Stabilizer state: Xi_P = 1, M_2 = 0.
#
#  ---------- ANALYTICAL SOLUTION ----------
#
#  (a) Fourth moment via symmetric subspace projector:
#      Expanding over S_4, only fixed-point-free permutations contribute
#      (since Tr(A)=0).  Two classes survive:
#      - 3 products of transpositions: each gives Tr(A^2)^2 = D^2
#      - 6 four-cycles: each gives Tr(A^4) = Tr(I) = D
#      Numerator = (3D^2+6D)/24.  Denominator = D(D+1)(D+2)(D+3)/24.
#      Result = 3/((D+1)(D+3)).  The (D+2) cancels.
#
#  (b) Xi_P = (1/D)[1 + (D^2-1)*3/((D+1)(D+3))]
#           = (1/D)[(D+3+3D-3)/(D+3)] = 4/(D+3).
#
#  (c) -log2(4/(D+3)) = log2((D+3)/4) ~ N-2 for D=2^N >> 1.
#
#  (d) Stabilizer state: exactly D Paulis have <P>=+/-1, rest are 0.
#      Xi_P = D/D = 1.  M_2 = 0.  Zero magic.
#
#  ---------- NUMERICAL VERIFICATION ----------
###########################################################################
"""
import numpy as np
from scipy.stats import unitary_group
from itertools import product

np.random.seed(42)

# --- Pauli matrices ---
I2 = np.eye(2, dtype=complex)
sx = np.array([[0,1],[1,0]], dtype=complex)
sy = np.array([[0,-1j],[1j,0]])
sz = np.array([[1,0],[0,-1]], dtype=complex)
paulis = [I2, sx, sy, sz]

def all_pauli_strings(N):
    """Generate all 4^N N-qubit Pauli strings."""
    ops = []
    for indices in product(range(4), repeat=N):
        P = paulis[indices[0]]
        for k in range(1, N):
            P = np.kron(P, paulis[indices[k]])
        ops.append(P)
    return ops

# ========================================================================
# Part (a): Fourth moment E[<A>^4] = 3/((D+1)(D+3))
# ========================================================================
print("Part (a): E_Haar[<A>^4] = 3/((D+1)(D+3)) for traceless A^2=I")
print("=" * 60)

for N in [1, 2, 3]:
    D = 2**N
    prediction = 3.0 / ((D+1) * (D+3))

    # Pick a non-identity Pauli as the test operator A
    A = paulis[1]  # sigma_x
    for _ in range(N-1):
        A = np.kron(A, I2)  # X on first qubit, I on rest

    # Monte Carlo: sample Haar-random states and compute <A>^4
    n_samples = 20000
    fourth_moments = []
    for _ in range(n_samples):
        psi = unitary_group.rvs(D)[:, 0]
        exp_A = (psi.conj() @ A @ psi).real
        fourth_moments.append(exp_A**4)

    mc_mean = np.mean(fourth_moments)
    print(f"  N={N} (D={D}): MC = {mc_mean:.6f}, predicted = {prediction:.6f}")
    assert abs(mc_mean - prediction) < 0.005, f"Mismatch at N={N}"

# ========================================================================
# Part (b): E[Xi_P] = 4/(D+3)
# ========================================================================
print(f"\nPart (b): E[Xi_P] = 4/(D+3)")
print("=" * 60)

for N in [1, 2]:
    D = 2**N
    P_strings = all_pauli_strings(N)
    prediction = 4.0 / (D + 3)

    n_samples = 10000
    xi_values = []
    for _ in range(n_samples):
        psi = unitary_group.rvs(D)[:, 0]
        xi = 0.0
        for P in P_strings:
            exp_P = (psi.conj() @ P @ psi).real
            xi += exp_P**4
        xi /= D
        xi_values.append(xi)

    mc_mean = np.mean(xi_values)
    print(f"  N={N} (D={D}): MC = {mc_mean:.6f}, predicted = {prediction:.6f}")
    assert abs(mc_mean - prediction) < 0.02, f"Mismatch at N={N}"

# ========================================================================
# Part (c): -log2(E[Xi_P]) ~ N-2
# ========================================================================
print(f"\nPart (c): -log2(E[Xi_P]) = N-2 + O(2^{{-N}})")
print("=" * 60)

for N in range(1, 11):
    D = 2**N
    xi_mean = 4.0 / (D + 3)
    M2_approx = -np.log2(xi_mean)
    print(f"  N={N:2d}: -log2(4/(D+3)) = {M2_approx:.4f}  (N-2 = {N-2})")

# ========================================================================
# Part (d): Stabilizer state has Xi_P = 1, M_2 = 0
# ========================================================================
print(f"\nPart (d): Stabilizer state verification for N=2")
print("=" * 60)

# |00> is a stabilizer state, stabilized by {II, ZI, IZ, ZZ}
psi_stab = np.array([1, 0, 0, 0], dtype=complex)
P_strings = all_pauli_strings(2)

n_nonzero = 0
xi = 0.0
for P in P_strings:
    exp_P = (psi_stab.conj() @ P @ psi_stab).real
    if abs(exp_P) > 1e-10:
        n_nonzero += 1
    xi += exp_P**4
xi /= 4  # D=4

print(f"  State: |00>  (computational basis = stabilizer state)")
print(f"  Non-zero Pauli expectations: {n_nonzero}  (expected D=4)")
print(f"  Xi_P = {xi:.6f}  (expected 1)")
print(f"  M_2 = -log2({xi:.6f}) = {-np.log2(xi):.6f}  (expected 0)")
assert n_nonzero == 4, f"Found {n_nonzero} nonzero Paulis, expected 4"
assert abs(xi - 1) < 1e-10, f"Xi_P = {xi}, expected 1"

print(f"\n  Stabilizer states concentrate all Pauli weight on D strings,")
print(f"  giving M_2=0 (zero magic).  Haar-random states spread weight")
print(f"  uniformly, giving M_2 ~ N-2 (near-maximal magic).")

