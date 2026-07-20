#!/usr/bin/env python3
"""
###########################################################################
#  EXERCISE 6.2 — Tripartite Mutual Information of Scrambled States
#
#  Chapter 6, Applications
#  Topic: Information delocalization, multipartite entanglement, scrambling
#
#  ---------- EXERCISE STATEMENT ----------
#
#  For a Haar-random pure state on four equally sized subsystems
#  A, B, C, D with d_A = d_B = d_C = d_D = d >> 1:
#
#  Using the Page approximation S(X) ~ min(ln d_X, ln d_Xbar),
#  compute I(A:B), I(A:C), I(A:BC), and show I_3(A:B:C) ~ -2 ln(d).
#
#  ---------- ANALYTICAL SOLUTION ----------
#
#  Subsystem entropies (Page approximation, volume law):
#    S(A) = S(B) = S(C) = S(D) = ln(d)       (1 subsystem, complement = d^3)
#    S(AB) = S(AC) = S(BC) = 2*ln(d)         (2 subsystems = half system)
#    S(ABC) = S(D) = ln(d)                    (3 subsystems, complement = d)
#
#  Mutual informations:
#    I(A:B)  = S(A) + S(B) - S(AB)  = ln(d) + ln(d) - 2*ln(d) = 0
#    I(A:C)  = S(A) + S(C) - S(AC)  = 0
#    I(A:BC) = S(A) + S(BC) - S(ABC) = ln(d) + 2*ln(d) - ln(d) = 2*ln(d)
#
#  Tripartite:
#    I_3(A:B:C) = I(A:B) + I(A:C) - I(A:BC) = 0 + 0 - 2*ln(d) = -2*ln(d)
#
#  Physical meaning: knowing B or C individually reveals nothing about A,
#  but combining them unlocks full information.  This is the hallmark of
#  information scrambling and delocalization.
#
#  ---------- NUMERICAL VERIFICATION ----------
###########################################################################
"""
import numpy as np
from scipy.stats import unitary_group

np.random.seed(42)

def von_neumann_entropy(rho):
    """Compute S(rho) = -Tr(rho ln rho) for a density matrix."""
    eigs = np.linalg.eigvalsh(rho)
    eigs = eigs[eigs > 1e-15]
    return -np.sum(eigs * np.log(eigs))

def partial_trace(psi_mat, dims, keep):
    """
    Compute the reduced density matrix by tracing out the complement.
    psi_mat: state vector reshaped to (d1, d2, ..., dn)
    dims: list of subsystem dimensions
    keep: list of subsystem indices to keep
    """
    n = len(dims)
    psi = psi_mat.reshape(dims)
    rho = np.tensordot(psi, psi.conj(), axes=0)
    # rho has indices (i1,...,in, j1,...,jn)
    # Trace over subsystems NOT in 'keep'
    trace_over = sorted(set(range(n)) - set(keep))
    # Contract pairs from highest index to lowest to avoid index shifts
    for idx in sorted(trace_over, reverse=True):
        rho = np.trace(rho, axis1=idx, axis2=idx + n - (n - rho.ndim // 2))
    # This is complex; use explicit construction instead
    return rho

# For simplicity, use the direct approach: reshape, contract
def reduced_dm(psi, dims, keep):
    """Get reduced density matrix for subsystems in 'keep'."""
    D_total = len(psi)
    n_sub = len(dims)
    psi_t = psi.reshape(dims)

    # Determine which indices to trace out
    trace_out = sorted(set(range(n_sub)) - set(keep))
    keep_sorted = sorted(keep)

    # Permute so kept indices come first
    order = keep_sorted + trace_out
    psi_t = np.transpose(psi_t, order)

    d_keep = int(np.prod([dims[k] for k in keep_sorted]))
    d_trace = int(np.prod([dims[k] for k in trace_out]))

    psi_t = psi_t.reshape(d_keep, d_trace)
    rho = psi_t @ psi_t.conj().T
    return rho

print("Tripartite mutual information for Haar-random states")
print("=" * 60)

for d in [2, 3, 4]:
    D = d**4
    n_samples = min(2000, max(200, 20000 // D))

    I_AB_list, I_AC_list, I_ABC_list, I3_list = [], [], [], []
    dims = [d, d, d, d]

    for _ in range(n_samples):
        psi = unitary_group.rvs(D)[:, 0]

        S_A   = von_neumann_entropy(reduced_dm(psi, dims, [0]))
        S_B   = von_neumann_entropy(reduced_dm(psi, dims, [1]))
        S_C   = von_neumann_entropy(reduced_dm(psi, dims, [2]))
        S_AB  = von_neumann_entropy(reduced_dm(psi, dims, [0, 1]))
        S_AC  = von_neumann_entropy(reduced_dm(psi, dims, [0, 2]))
        S_BC  = von_neumann_entropy(reduced_dm(psi, dims, [1, 2]))
        S_ABC = von_neumann_entropy(reduced_dm(psi, dims, [0, 1, 2]))

        I_AB  = S_A + S_B - S_AB
        I_AC  = S_A + S_C - S_AC
        I_ABC = S_A + S_BC - S_ABC
        I3    = I_AB + I_AC - I_ABC

        I_AB_list.append(I_AB)
        I_AC_list.append(I_AC)
        I_ABC_list.append(I_ABC)
        I3_list.append(I3)

    prediction = -2 * np.log(d)
    print(f"\n  d = {d} (D = {D}):")
    print(f"    <I(A:B)>  = {np.mean(I_AB_list):+.4f}  (predicted: 0)")
    print(f"    <I(A:C)>  = {np.mean(I_AC_list):+.4f}  (predicted: 0)")
    print(f"    <I(A:BC)> = {np.mean(I_ABC_list):+.4f}  (predicted: {2*np.log(d):.4f})")
    print(f"    <I_3>     = {np.mean(I3_list):+.4f}  (predicted: {prediction:.4f})")

print(f"\n  A negative I_3 is the information-theoretic signature of")
print(f"  scrambling: information about A is delocalized across B and C")
print(f"  and can only be recovered by accessing both jointly.")

