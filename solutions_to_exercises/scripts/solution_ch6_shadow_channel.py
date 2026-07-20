#!/usr/bin/env python3
"""
###########################################################################
#  EXERCISE 6.1 — Depolarizing Channel and Classical Shadows
#
#  Chapter 6, Applications
#  Topic: Classical shadows, channel inversion, single-qubit Clifford
#
#  ---------- EXERCISE STATEMENT ----------
#
#  Averaging the measurement map over the single-qubit Clifford group
#  yields the depolarizing channel M(sigma) = (sigma + Tr(sigma)*I)/3.
#  Verify that M^{-1}(tau) = 3*tau - Tr(tau)*I is the inverse,
#  and for tau = U^dag|x><x|U (with Tr(tau)=1):
#    rho^{sh} = 3*U^dag|x><x|U - I.
#
#  ---------- ANALYTICAL SOLUTION ----------
#
#  Apply M^{-1} to M(sigma):
#    M^{-1}(M(sigma)) = 3*(sigma + Tr(sigma)*I)/3
#                      - Tr((sigma + Tr(sigma)*I)/3)*I
#                     = sigma + Tr(sigma)*I
#                      - (Tr(sigma) + 2*Tr(sigma))/3 * I
#                     = sigma + Tr(sigma)*I - Tr(sigma)*I
#                     = sigma.
#  For tau with Tr(tau)=1: rho^{sh} = 3*tau - I.
#
#  ---------- NUMERICAL VERIFICATION ----------
###########################################################################
"""
import numpy as np

I2 = np.eye(2, dtype=complex)

def M(sigma):
    """Depolarizing measurement channel for single-qubit Clifford shadows."""
    return (sigma + np.trace(sigma) * I2) / 3

def M_inv(tau):
    """Inverse of the depolarizing measurement channel."""
    return 3 * tau - np.trace(tau) * I2

# We test on a range of 2x2 matrices (not just density matrices)
# to confirm that M^{-1}(M(sigma)) = sigma for all sigma.
print("Verifying M^{-1}(M(sigma)) = sigma for random 2x2 matrices")
print("=" * 60)
np.random.seed(42)

for trial in range(10):
    sigma = np.random.randn(2, 2) + 1j * np.random.randn(2, 2)
    recovered = M_inv(M(sigma))
    err = np.linalg.norm(recovered - sigma)
    print(f"  trial {trial}: ||M^{{-1}}(M(sigma)) - sigma|| = {err:.2e}")
    assert err < 1e-12, f"Inverse failed at trial {trial}"

# Now verify the shadow snapshot formula for a physical measurement outcome.
# Suppose we applied a random Clifford U and measured outcome |x>.
# The post-measurement state is tau = U^dag|x><x|U.
print("\nVerifying the shadow snapshot formula rho^{sh} = 3*U^dag|x><x|U - I")
print("=" * 60)

# Take U = Hadamard, measurement outcome |0>
H_gate = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
ket0 = np.array([[1], [0]], dtype=complex)
tau = H_gate.conj().T @ (ket0 @ ket0.conj().T) @ H_gate

# The shadow snapshot
rho_sh = 3 * tau - I2
rho_sh_direct = M_inv(tau)

print(f"  U = Hadamard, measurement outcome |0>")
print(f"  tau = H^dag|0><0|H = |+><+|")
print(f"  rho^{{sh}} = 3*|+><+| - I = ")
print(f"    {rho_sh}")
print(f"  Via M^{{-1}}: {rho_sh_direct}")
assert np.allclose(rho_sh, rho_sh_direct), "Shadow formula mismatch"

# Verify that E[rho^{sh}] = rho for a known state rho = |0><0|
# by averaging over all 6 single-qubit Clifford measurement outcomes
print("\nVerifying E[rho^{sh}] = rho for rho = |0><0|")
print("=" * 60)

# The single-qubit Clifford group has 24 elements, but for
# classical shadows we only need the 3 measurement bases {X, Y, Z}
# with 2 outcomes each = 6 shadow snapshots.
rho_true = np.array([[1, 0], [0, 0]], dtype=complex)  # |0><0|

# Measurement bases: eigenstates of X, Y, Z
bases = {
    'Z': [np.array([1, 0], dtype=complex), np.array([0, 1], dtype=complex)],
    'X': [np.array([1, 1], dtype=complex)/np.sqrt(2),
          np.array([1, -1], dtype=complex)/np.sqrt(2)],
    'Y': [np.array([1, 1j], dtype=complex)/np.sqrt(2),
          np.array([1, -1j], dtype=complex)/np.sqrt(2)],
}

avg_shadow = np.zeros((2, 2), dtype=complex)
for basis_name, eigenstates in bases.items():
    for ket in eigenstates:
        # Born probability of outcome |ket>: p = <ket|rho|ket>
        prob = (ket.conj() @ rho_true @ ket).real

        # Shadow snapshot: 3|ket><ket| - I
        snapshot = 3 * np.outer(ket, ket.conj()) - I2

        # Weight by probability
        avg_shadow += prob * snapshot / 3  # 3 bases, equal probability

print(f"  E[rho^{{sh}}] = \n    {avg_shadow}")
print(f"  rho_true   = \n    {rho_true}")
err = np.linalg.norm(avg_shadow - rho_true)
print(f"  ||E[rho^{{sh}}] - rho|| = {err:.2e}")
assert err < 1e-10, f"Shadow average deviates from true state"

