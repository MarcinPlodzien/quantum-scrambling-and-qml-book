"""
magic.py -- stabilizer Renyi entropy (magic) M_2 via a batched Walsh-Hadamard
transform.

    M_2(psi) = -log2[ (1/D) sum_P <psi|P|psi>^4 ],   P over all 4^N Pauli strings.

The naive sum has 4^N terms.  Group the Pauli strings by their X-mask x: the 2^N
Paulis sharing a given x have expectations that are exactly the Walsh-Hadamard
transform (over the Z-mask z) of the correlation vector

    C_x[j] = conj(psi_j) * psi_{j xor x}.

So <psi|P(x,z)|psi> = (H^{otimes N} C_x)[z], and sum_P <...>^4 = sum_x sum_z
|(H C_x)[z]|^4.  We build ALL correlation vectors at once as the D x D matrix
M[x, j] = C_x[j], apply one batched Walsh-Hadamard along the j-axis, and sum the
fourth powers.  Cost O(N * 4^N), fully vectorised (no 4^N enumeration, no per-x
Python/JAX dispatch).  Feasible to ~L=12, where the D x D matrix is 256 MB.

(This replaces an earlier per-x JAX loop that spent all its time in dispatch
overhead; the value is identical -- verified against the brute-force 4^N sum.)
"""
import numpy as np


def _fwht_rows(M, N):
    """In-place-style Walsh-Hadamard transform H^{otimes N} along axis 1 of M,
    batched over the rows (axis 0).  Standard radix-2 butterfly; the OUTPUT order
    is irrelevant here because compute_sre sums |.|^4 over all outputs."""
    rows, D = M.shape
    T = M
    h = 1
    while h < D:
        T = T.reshape(rows, D // (2 * h), 2 * h)
        a = T[:, :, :h]
        b = T[:, :, h:]
        T = np.concatenate([a + b, a - b], axis=2).reshape(rows, D)
        h *= 2
    return T


def compute_sre(psi, n_qubits, chunk=None):
    """M_2 (bits).  0 for stabilizer states; ~ N - log2(3) for Haar-random states.

    The full D x D correlation matrix is 2^{2N} complex numbers -- 256 MB at
    N=12, 4 GB at N=14.  For N >= 13 we process the X-masks in chunks so the
    peak memory stays bounded; the result is identical (the sum over X-masks just
    accumulates)."""
    psi = np.asarray(psi, dtype=complex)
    psi = psi / np.linalg.norm(psi)
    D = 1 << n_qubits
    j = np.arange(D)
    if chunk is None:
        chunk = D if n_qubits <= 12 else (1 << 11)   # cap rows at 2048 for large N
    total = 0.0
    for x0 in range(0, D, chunk):
        xs = np.arange(x0, min(x0 + chunk, D))
        # rows are correlation vectors C_x[j] = conj(psi_j) psi_{j xor x}
        M = np.conj(psi)[None, :] * psi[xs[:, None] ^ j[None, :]]
        F = _fwht_rows(M, n_qubits)
        total += np.sum(np.abs(F) ** 4)
    return float(-np.log2(total / D))
