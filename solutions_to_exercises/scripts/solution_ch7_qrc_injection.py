#!/usr/bin/env python3
"""
###########################################################################
#  EXERCISE 7.5 — Injection Contractivity and Restarting Overhead in QRC
#
#  Chapter 7, Quantum Machine Learning
#  Topic: Quantum reservoir computing, fading memory, trace distance
#
#  ---------- EXERCISE STATEMENT ----------
#
#  A quantum reservoir injects data by tracing out qubit 1 and
#  re-preparing it: E_x(rho) = Tr_1(rho) tensor |psi(x)><psi(x)|_1.
#
#  (a) Write an explicit Kraus decomposition for E_x.
#  (b) Show E_x is contractive under the trace distance.
#  (c) Show the restarting readout requires nu*T*(T+1)/2 circuit runs.
#
#  ---------- ANALYTICAL SOLUTION ----------
#
#  (a) Kraus operators K_a = (I_bar1 tensor |psi(x)>_1)(I_bar1 tensor <a|_1)
#      for a = 0, 1.  Completeness: sum_a K_a^dag K_a = I.
#
#  (b) E_x(rho) - E_x(sigma) = Tr_1(rho-sigma) tensor |psi><psi|.
#      ||...||_1 = ||Tr_1(rho-sigma)||_1 * 1.
#      Partial trace is CPTP, hence contractive: ||Tr_1(Delta)||_1 <= ||Delta||_1.
#
#  (c) At step n, replay n injection-evolution cycles from scratch.
#      Total = nu * sum_{n=1}^T n = nu * T(T+1)/2.
#
#  ---------- NUMERICAL VERIFICATION ----------
###########################################################################
"""
import numpy as np

np.random.seed(42)

# ========================================================================
# Part (a): Kraus decomposition verification
# ========================================================================
print("Part (a): Kraus decomposition of the injection map")
print("=" * 60)

N = 3  # number of qubits
D = 2**N
D_bar = 2**(N - 1)  # dimension of qubits 2..N

# Data-dependent input state for qubit 1
theta_x = 0.7
psi_x = np.array([np.cos(theta_x / 2), np.sin(theta_x / 2)], dtype=complex)

# Build Kraus operators
# K_a = (I_bar tensor |psi_x>) (I_bar tensor <a|)
# This maps rho -> sum_a <a|_1 rho |a>_1 tensor |psi_x><psi_x|
K_ops = []
for a in range(2):
    bra_a = np.zeros((1, 2), dtype=complex)
    bra_a[0, a] = 1
    ket_psi = psi_x.reshape(2, 1)

    # K_a acts on the full D-dimensional space
    # Left part: I_bar tensor bra_a (projects qubit 1 to |a>)
    proj = np.kron(np.eye(D_bar), bra_a)  # shape (D_bar, D)
    # Right part: I_bar tensor ket_psi (replaces qubit 1 with |psi_x>)
    inject = np.kron(np.eye(D_bar), ket_psi)  # shape (D, D_bar)

    K_a = inject @ proj  # shape (D, D)
    K_ops.append(K_a)

# Verify completeness: sum K_a^dag K_a = I
completeness = sum(K.conj().T @ K for K in K_ops)
print(f"  sum K_a^dag K_a = I: {np.allclose(completeness, np.eye(D))}")
assert np.allclose(completeness, np.eye(D))

# Verify action: E_x(rho) = Tr_1(rho) tensor |psi_x><psi_x|
rho_test = np.random.randn(D, D) + 1j * np.random.randn(D, D)
rho_test = rho_test @ rho_test.conj().T
rho_test /= np.trace(rho_test)

# Apply via Kraus
rho_out_kraus = sum(K @ rho_test @ K.conj().T for K in K_ops)

# Apply directly: Tr_1(rho) tensor |psi><psi|
rho_reshaped = rho_test.reshape(D_bar, 2, D_bar, 2)
rho_bar = np.trace(rho_reshaped, axis1=1, axis2=3)  # partial trace over qubit 1
rho_out_direct = np.kron(rho_bar, np.outer(psi_x, psi_x.conj()))

err = np.linalg.norm(rho_out_kraus - rho_out_direct)
print(f"  ||E_x(rho)_Kraus - E_x(rho)_direct|| = {err:.2e}")
assert err < 1e-10

# ========================================================================
# Part (b): Contractivity under trace distance
# ========================================================================
print(f"\nPart (b): Contractivity of the injection map")
print("=" * 60)

n_tests = 100
for _ in range(n_tests):
    # Two random density matrices
    A = np.random.randn(D, D) + 1j * np.random.randn(D, D)
    rho = A @ A.conj().T; rho /= np.trace(rho)
    B = np.random.randn(D, D) + 1j * np.random.randn(D, D)
    sigma = B @ B.conj().T; sigma /= np.trace(sigma)

    # Trace distance: (1/2)||rho - sigma||_1
    delta = rho - sigma
    td_in = 0.5 * np.sum(np.abs(np.linalg.eigvalsh(delta)))

    # Apply injection map
    rho_out = sum(K @ rho @ K.conj().T for K in K_ops)
    sigma_out = sum(K @ sigma @ K.conj().T for K in K_ops)
    delta_out = rho_out - sigma_out
    td_out = 0.5 * np.sum(np.abs(np.linalg.eigvalsh(delta_out)))

    assert td_out <= td_in + 1e-10, \
        f"Contractivity violated: {td_out} > {td_in}"

print(f"  Tested {n_tests} random state pairs: contractivity holds in all cases.")
print(f"  The injection map erases information stored in qubit 1,")
print(f"  providing the physical mechanism for fading memory.")

# ========================================================================
# Part (c): Restarting overhead
# ========================================================================
print(f"\nPart (c): Restarting readout circuit count")
print("=" * 60)

for T in [10, 100, 1000]:
    for nu in [100]:
        total = nu * T * (T + 1) // 2
        print(f"  T={T:5d}, nu={nu:4d}: total runs = {total:>12,}")

print(f"\n  The quadratic scaling nu*T^2/2 in sequence length T is a")
print(f"  severe overhead that motivates the weak-monitoring protocols.")

