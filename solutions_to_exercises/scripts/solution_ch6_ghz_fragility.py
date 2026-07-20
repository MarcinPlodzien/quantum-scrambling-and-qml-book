#!/usr/bin/env python3
"""
###########################################################################
#  EXERCISE 6.3 — Fragility of the GHZ Metrological Advantage
#
#  Chapter 6, Applications
#  Topic: Quantum metrology, QFI, GHZ state, particle loss
#
#  ---------- EXERCISE STATEMENT ----------
#
#  The N-qubit GHZ state |GHZ> = (|0...0> + |1...1>)/sqrt(2) achieves
#  the Heisenberg limit F_Q = N^2 for the phase generator
#  G = (1/2) sum_i sigma_z^{(i)}.
#
#  (a) Trace out one qubit and find the reduced state rho_{N-1}.
#  (b) Compute the QFI of rho_{N-1} for the reduced generator
#      G' = (1/2) sum_{i=1}^{N-1} sigma_z^{(i)}.
#
#  ---------- ANALYTICAL SOLUTION ----------
#
#  (a) rho_{N-1} = (1/2)(|0><0|^{N-1} + |1><1|^{N-1}).
#      The off-diagonal coherences vanish because <0_N|1_N> = 0.
#      The state is a classical mixture of the two product states.
#
#  (b) The QFI for a mixed state rho = sum_k lambda_k |k><k| is:
#      F_Q = 2 sum_{k,l} (lambda_k - lambda_l)^2/(lambda_k + lambda_l) |<k|G'|l>|^2
#
#      Here lambda_0 = lambda_1 = 1/2, so (lambda_0 - lambda_1)^2 = 0.
#      For the unpopulated branches (lambda_l = 0), G' is diagonal
#      in the Z-basis, so <k|G'|l> = 0 for all l outside {|0...0>, |1...1>}.
#      
#      Therefore F_Q = 0.  Complete loss of metrological utility.
#
#  ---------- NUMERICAL VERIFICATION ----------
###########################################################################
"""
import numpy as np
from scipy.linalg import expm

def qfi_pure(psi, G):
    """QFI for pure state: F_Q = 4 * Var(G) = 4(<G^2> - <G>^2)."""
    exp_G = (psi.conj() @ G @ psi).real
    exp_G2 = (psi.conj() @ G @ G @ psi).real
    return 4 * (exp_G2 - exp_G**2)

def qfi_mixed(rho, G):
    """QFI for a general mixed state via the SLD formula."""
    eigs, V = np.linalg.eigh(rho)
    D = len(eigs)
    fq = 0.0
    for k in range(D):
        for l in range(D):
            denom = eigs[k] + eigs[l]
            if denom < 1e-15:
                continue
            G_kl = V[:, k].conj() @ G @ V[:, l]
            fq += 2 * (eigs[k] - eigs[l])**2 / denom * abs(G_kl)**2
    return fq.real

print("GHZ metrological advantage and its fragility under particle loss")
print("=" * 60)

for N in [2, 3, 4, 5, 6]:
    D = 2**N
    D_red = 2**(N - 1)

    # Construct the GHZ state
    ghz = np.zeros(D, dtype=complex)
    ghz[0] = 1 / np.sqrt(2)     # |0...0>
    ghz[-1] = 1 / np.sqrt(2)    # |1...1>

    # Phase generator G = (1/2) sum sigma_z
    G = np.zeros((D, D), dtype=complex)
    for i in range(N):
        sz_i = np.eye(1, dtype=complex)
        for j in range(N):
            if j == i:
                sz_i = np.kron(sz_i, np.array([[1, 0], [0, -1]], dtype=complex))
            else:
                sz_i = np.kron(sz_i, np.eye(2, dtype=complex))
        G += sz_i / 2

    # QFI of the full GHZ state
    fq_full = qfi_pure(ghz, G)

    # Partial trace: trace out the last qubit
    ghz_mat = ghz.reshape(D_red, 2)
    rho_red = ghz_mat @ ghz_mat.conj().T

    # Reduced generator (first N-1 qubits)
    G_red = np.zeros((D_red, D_red), dtype=complex)
    for i in range(N - 1):
        sz_i = np.eye(1, dtype=complex)
        for j in range(N - 1):
            if j == i:
                sz_i = np.kron(sz_i, np.array([[1, 0], [0, -1]], dtype=complex))
            else:
                sz_i = np.kron(sz_i, np.eye(2, dtype=complex))
        G_red += sz_i / 2

    # QFI of the reduced state
    fq_red = qfi_mixed(rho_red, G_red)

    print(f"\n  N = {N}:")
    print(f"    F_Q(GHZ)     = {fq_full:.4f}  (expected N^2 = {N**2})")
    print(f"    F_Q(reduced) = {fq_red:.4f}  (expected 0)")
    assert abs(fq_full - N**2) < 1e-6, f"Full QFI = {fq_full}, expected {N**2}"
    assert abs(fq_red) < 1e-10, f"Reduced QFI = {fq_red}, expected 0"

    # Verify reduced state is classical mixture
    eigs_red = np.linalg.eigvalsh(rho_red)
    eigs_red = sorted(eigs_red[eigs_red > 1e-12], reverse=True)
    print(f"    rho eigenvalues: {[f'{e:.4f}' for e in eigs_red[:4]]}")
    print(f"    (classical 50/50 mixture, all coherence lost)")

print(f"\n  The QFI drops from N^2 to 0 upon loss of a single qubit.")
print(f"  All metrological information was stored in the global phase")
print(f"  coherence, which the environment irreversibly collapsed.")

