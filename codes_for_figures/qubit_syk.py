"""
qubit_syk.py -- efficient Jordan-Wigner SYK on qubits.
=====================================================

The Majorana SYK model on L qubits uses N_M = 2L Majorana fermions:

    q=4 (chaotic):      H = sum_{i<j<k<l} J_ijkl  chi_i chi_j chi_k chi_l
    q=2 (free/Gaussian): H = sum_{i<j}     J_ij   (i chi_i chi_j)

with real Gaussian couplings.  Under the Jordan-Wigner transformation each
Majorana becomes a Pauli string (a tensor product of I, X, Y, Z),

    chi_{2k}   = Z_0 Z_1 ... Z_{k-1} X_k ,
    chi_{2k+1} = Z_0 Z_1 ... Z_{k-1} Y_k ,      k = 0 .. L-1.

The one idea that makes this file fast is:

    a product of Majoranas is again a SINGLE Pauli string.

So H is not a 2^L x 2^L matrix; it is a *list* of (coefficient, Pauli string).
Applying one Pauli string to a statevector costs O(2^L) (it just permutes the
amplitudes and flips some signs), so the whole model runs to L ~ 14 without ever
storing a dense matrix.

--------------------------------------------------------------------------------
HOW A PAULI STRING IS STORED: two bitmasks (x, z) and a phase p
--------------------------------------------------------------------------------
Any Pauli string on L qubits can be written, up to a phase, as

        P = i^p * X^x * Z^z ,

where x and z are L-bit integers (bitmasks): bit k of x says "apply X on qubit
k", bit k of z says "apply Z on qubit k".  A single qubit then reads off as

        x_k z_k :  00 -> I,  10 -> X,  01 -> Z,  11 -> XZ = -iY.

The phase p in {0,1,2,3} tracks factors of i (so a genuine Y = i*XZ is x=1,z=1,
p=1).  This (x, z) encoding is often called the *symplectic* representation --
"symplectic" only because, as we will see in pmul(), multiplying two Paulis is
XOR of the masks plus a sign fixed by the cross term popcount(z1 & x2); that
"cross term" is a symplectic inner product on the bit vectors.  The name is not
important here; the (x, z, p) triple is just a compact, exact stand-in for a
Pauli matrix that never needs the matrix itself.

CONVENTION (used everywhere in this file): qubit k lives in integer bit k, i.e.
basis state |j> has qubit k in bit (j >> k) & 1.  This is the natural convention
for statevector code and is the one apply_H, compute_dense, and the FWHT magic
routine all use.  It is the qubit-reversal of the textbook np.kron ordering
(which puts qubit 0 in the most significant bit); magic, entanglement, and the
spectrum are all invariant under that relabelling, so no physics depends on it.
"""
import numpy as np
from itertools import combinations

# ---------------------------------------------------------------------------
# popcount lookup table.  popcount(m) = number of 1-bits in m.  We need the
# PARITY popcount(...) & 1 all over this file (it is the sign a Z-string puts on
# a basis state).  Looking it up in a precomputed 16-bit table is far faster
# than recomputing per call, and it vectorises over a whole numpy array of
# indices at once: _POP16[array_of_masks] returns the array of popcounts.
#
# dtype MUST be signed.  The sign is built as  1 - 2*(popcount & 1)  ->  +1/-1.
# With an unsigned (uint8) table, 1 - 2*1 evaluates to 255 (unsigned underflow),
# not -1, and every odd-parity sign silently becomes +255.  Use int64.
# ---------------------------------------------------------------------------
_POP16 = np.array([bin(i).count("1") for i in range(1 << 16)], dtype=np.int64)

# i^p for p = 0,1,2,3.
_PHASE = np.array([1.0 + 0j, 1j, -1.0 + 0j, -1j])


def majoranas(L):
    """The 2L Jordan-Wigner Majoranas, each as a Pauli string (x, z, p).

    chi_{2k}   = Z_0..Z_{k-1} X_k :  X on qubit k (bit k); Z-string on the low k
                 bits;  no i factor, so p = 0.
    chi_{2k+1} = Z_0..Z_{k-1} Y_k :  same, but Y_k = i X_k Z_k, so qubit k also
                 carries a Z (bit k added to the z-mask) and p = 1 for the i.
    """
    ops = []
    for k in range(L):
        xk = 1 << k                     # X (or Y) acts on qubit k  -> bit k of x
        zstr = (1 << k) - 1             # Z_0..Z_{k-1}              -> low k bits of z
        ops.append((xk, zstr, 0))       # chi_{2k}   = Z..Z X_k
        ops.append((xk, zstr | xk, 1))  # chi_{2k+1} = Z..Z Y_k = i * Z..Z X_k Z_k
    return ops


def pmul(A, B):
    """Multiply two Pauli strings A = i^{p1} X^{x1} Z^{z1},  B = i^{p2} X^{x2} Z^{z2}.

    X and Z on the SAME qubit satisfy X^2 = Z^2 = I, so the X- and Z-masks just
    add mod 2, i.e. XOR:   x = x1 ^ x2,   z = z1 ^ z2.

    The only subtlety is order.  Writing the product out,

        (X^{x1} Z^{z1}) (X^{x2} Z^{z2})

    we must slide Z^{z1} to the right past X^{x2}.  On each qubit where BOTH a Z
    (from z1) and an X (from x2) sit, Z X = -X Z contributes a factor -1.  The
    number of such qubits is popcount(z1 & x2), so the whole reordering yields
    (-1)^popcount(z1 & x2).  Encoding -1 as i^2, the phase exponent is

        p = p1 + p2 + 2 * popcount(z1 & x2)   (mod 4).
    """
    x1, z1, p1 = A
    x2, z2, p2 = B
    x = x1 ^ x2
    z = z1 ^ z2
    p = (p1 + p2 + 2 * bin(z1 & x2).count("1")) & 3   # &3 is mod 4
    return (x, z, p)


def _majorana_product(chi, idx):
    """Fold a tuple of Majorana indices into one Pauli string via repeated pmul."""
    P = chi[idx[0]]
    for a in idx[1:]:
        P = pmul(P, chi[a])
    return P


def build_terms(L, q, rng, J=1.0):
    """Return H as a list of (coeff, x, z, p): the SYK Hamiltonian, Pauli by Pauli.

    q=4: variance 3! J^2 / N_M^3 keeps the energy extensive (<H^2> ~ N).
    q=2: an explicit factor i makes (i chi_i chi_j) Hermitian (chi_i chi_j alone
         is anti-Hermitian), implemented by bumping the phase p by 1.
    """
    chi = majoranas(L)
    Nm = 2 * L
    terms = []
    if q == 4:
        sd = np.sqrt(6.0 * J**2 / Nm**3)                # sqrt(3! J^2 / N_M^3)
        for idx in combinations(range(Nm), 4):
            x, z, p = _majorana_product(chi, idx)
            terms.append((float(rng.normal(0, sd)), x, z, p))
    elif q == 2:
        sd = np.sqrt(J**2 / Nm)                          # extensive bandwidth
        for i, j in combinations(range(Nm), 2):
            x, z, p = pmul(chi[i], chi[j])
            terms.append((float(rng.normal(0, sd)), x, z, (p + 1) & 3))  # the i
    else:
        raise ValueError("q must be 2 or 4")
    return terms


def apply_H(psi, terms, L):
    """Return H|psi>, applying each Pauli string in O(2^L).  No dense matrix.

    A Pauli string O = i^p X^x Z^z acts on a computational basis state |j> as

        O |j> = i^p * (-1)^popcount(z & j) * |j XOR x> .

    Read this off the encoding: Z^z multiplies |j> by (-1) once per qubit where
    both z and j have a 1 -> the sign is (-1)^popcount(z & j); then X^x flips
    exactly the qubits in x -> the ket becomes |j XOR x>.

    Turned around to act on amplitudes: the amplitude that LANDS on index j comes
    from the source index j XOR x, so

        (O psi)[j] = i^p * (-1)^popcount(z & (j^x)) * psi[j^x] .

    Everything below is that formula, vectorised over all j at once:
      * jx = j ^ x        the permutation "flip the qubits in x"
      * z & jx            the Z-mask restricted to the source indices
      * _POP16[...] & 1   its parity, per index, from the lookup table
      * 1 - 2*parity      turns parity {0,1} into the sign {+1,-1}
    Cost per term is O(2^L); the whole H is len(terms) such passes.
    """
    D = 1 << L
    j = np.arange(D, dtype=np.int64)
    out = np.zeros(D, dtype=complex)
    for c, x, z, p in terms:
        jx = j ^ x
        sign = 1 - 2 * (_POP16[z & jx] & 1)             # (-1)^popcount(z & (j^x))
        out += (c * _PHASE[p]) * sign * psi[jx]
    return out


def compute_dense(terms, L):
    """Build the explicit 2^L x 2^L matrix (small L only; for tests / teaching).

    Same rule as apply_H, read as matrix elements.  O = i^p X^x Z^z is nonzero
    only from column b to row b^x, with value  i^p (-1)^popcount(z & b).  Here the
    sign uses the COLUMN index b (= j), because that is the |b> that Z^z acts on.
    (apply_H used z & (j^x): there j is the OUTPUT row, so the source is j^x --
    the same column index, just named differently.)
    """
    D = 1 << L
    H = np.zeros((D, D), dtype=complex)
    j = np.arange(D, dtype=np.int64)
    for c, x, z, p in terms:
        jx = j ^ x
        sign = 1 - 2 * (_POP16[z & j] & 1)              # (-1)^popcount(z & column)
        H[jx, j] += c * _PHASE[p] * sign
    return H


def lanczos_expm(psi, terms, L, t, m=40):
    """e^{-i H t} |psi> by an m-step Lanczos (Krylov) approximation.

    Lanczos builds an orthonormal basis of the Krylov space
    {psi, H psi, H^2 psi, ...} in which the Hermitian H is a small real
    tridiagonal matrix T (diagonal alpha, off-diagonal beta).  We exponentiate
    that tiny m x m matrix exactly and map back.  Because T is real symmetric,
    e^{-i t T} is exactly unitary, so the returned vector has the same norm as
    psi regardless of m (m only controls accuracy).  m ~ 40 is ample for the
    moderate H*t here; the returned state matches scipy.linalg.expm to ~1e-15.
    """
    beta0 = np.linalg.norm(psi)
    V = np.zeros((m, len(psi)), dtype=complex)
    alpha = np.zeros(m)
    beta = np.zeros(m)
    V[0] = psi / beta0
    w = apply_H(V[0], terms, L)
    alpha[0] = np.vdot(V[0], w).real
    w = w - alpha[0] * V[0]
    for k in range(1, m):
        beta[k] = np.linalg.norm(w)
        if beta[k] < 1e-12:                              # Krylov space exhausted
            m = k
            break
        V[k] = w / beta[k]
        w = apply_H(V[k], terms, L)
        alpha[k] = np.vdot(V[k], w).real
        w = w - alpha[k] * V[k] - beta[k] * V[k - 1]     # three-term recurrence
    T = np.diag(alpha[:m]) + np.diag(beta[1:m], 1) + np.diag(beta[1:m], -1)
    ev, evec = np.linalg.eigh(T)
    coeff = evec @ (np.exp(-1j * t * ev) * evec[0].conj())   # e^{-itT} e_0 in eigenbasis
    return beta0 * (coeff @ V[:m])


def half_entanglement(psi, L):
    """von Neumann entanglement entropy across the middle cut (qubits < L//2)."""
    a = L // 2
    M = psi.reshape(1 << a, 1 << (L - a))
    s = np.linalg.svd(M, compute_uv=False)
    p = s ** 2
    p = p[p > 1e-13]
    return float(-np.sum(p * np.log(p)))
