"""Unitaries That Commute with a Hamiltonian: numerical verification of the book solution."""
import numpy as np

def commutant_dim(eigs):
    """Real dimension of {X skew-Hermitian : [X, H] = 0} for H = diag(eigs)."""
    D = len(eigs)
    H = np.diag(eigs).astype(complex)
    basis = []
    for i in range(D):
        A = np.zeros((D, D), complex)
        A[i, i] = 1j
        basis.append(A)
        for j in range(i+1, D):
            A = np.zeros((D, D), complex)
            A[i, j], A[j, i] = 1, -1
            basis.append(A)
            A = np.zeros((D, D), complex)
            A[i, j], A[j, i] = 1j, 1j
            basis.append(A)
    M = np.array([(B @ H - H @ B).ravel() for B in basis]).T
    return len(basis) - np.linalg.matrix_rank(np.vstack([M.real, M.imag]))

for pattern in [(1,1,1,1), (2,1,1), (2,2), (3,1), (4,)]:
    eigs = sum([[k]*m for k, m in enumerate(pattern)], [])
    d = commutant_dim(np.array(eigs, float))
    print(pattern, "dim =", d, " sum d_k^2 =", sum(m*m for m in pattern))
    assert d == sum(m*m for m in pattern)

for N in range(1, 6):
    print(f"N={N}: non-degenerate family dimension 2^N = {2**N} of 4^N = {4**N}, fraction {2**N/4**N:.4f}")
print("sum d_k^2 VERIFIED")
