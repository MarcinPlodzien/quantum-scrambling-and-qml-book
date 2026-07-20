"""
qrc_dense_reference.py
======================
Exact, dense (full density-matrix) implementation of the quantum reservoir
computer used throughout the book's Santa Fe forecasting example.  This module is
the GROUND TRUTH: the tensor-network implementations (quimb, TeNPy, ITensor) in
this folder are verified against it at small N, where the dense simulation is
still cheap.

--------------------------------------------------------------------------------
THE MODEL AND PROTOCOL
--------------------------------------------------------------------------------
Reservoir: an N-qubit mixed-field Ising chain (nonintegrable, genuinely
scrambling),

    H = J * sum_i Z_i Z_{i+1}  +  h_x * sum_i X_i  +  h_z * sum_i Z_i .

Pure transverse field (h_z = 0) is Jordan-Wigner integrable and does NOT
scramble; the longitudinal h_z breaks integrability.  A single step evolves the
reservoir by U = exp(-i H t_qrc).

Each step of the online protocol, given the next scalar input s_n in [-1, 1]:

    1. ENCODE   s_n into qubit 0 by an R_y rotation, |psi(s)> with <Z> = s.
    2. RESET    the input register: partial-trace qubit 0 out of rho.
    3. INJECT   the freshly encoded input in its place.
    4. EVOLVE   rho -> U rho U^dagger  (density-matrix / mixed-state evolution).
    5. READ OUT a fixed set of low-weight Pauli observables as the feature
                vector f_n.

Only a linear (ridge) readout on the features {f_n} is trained; the reservoir
dynamics are never optimized.  The reset (a completely positive trace-preserving
map) is what gives the reservoir fading memory: CPTP maps contract trace
distance, so distant inputs are forgotten.

--------------------------------------------------------------------------------
COMPLEXITY
--------------------------------------------------------------------------------
The state is a dense 2^N x 2^N density matrix and U rho U^dagger costs O(8^N) per
step, so this reference is practical only to about N = 8-10.  The tensor-network
versions replace rho by a matrix-product density operator to reach larger N,
efficiently as long as the reservoir's operator entanglement (bond dimension)
stays bounded -- which is exactly the under/edge-of-scrambling regime where a
reservoir is useful.
"""
import numpy as np
from scipy.linalg import expm

I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)

_POP = np.array([bin(i).count("1") for i in range(1 << 16)])
_PHASE = np.array([1, 1j, -1, -1j])


def kron_op(single, site, N):
    """Single-qubit operator `single` on `site` (qubit 0 = leftmost), identity elsewhere."""
    op = np.array([[1.0]], dtype=complex)
    for j in range(N):
        op = np.kron(op, single if j == site else I2)
    return op


def mixed_field_ising(N, J, hx, hz):
    """Nonintegrable mixed-field Ising Hamiltonian.  hz may be a scalar or a
    length-N array (per-site longitudinal field, e.g. for a little disorder)."""
    hz = np.broadcast_to(hz, (N,))
    H = np.zeros((2 ** N, 2 ** N), dtype=complex)
    for i in range(N - 1):
        H += J * (kron_op(Z, i, N) @ kron_op(Z, i + 1, N))
    for i in range(N):
        H += hx * kron_op(X, i, N) + hz[i] * kron_op(Z, i, N)
    return H


def encode(s):
    """R_y(arccos s)|0> : the single-qubit input state with <Z> = s."""
    return np.array([np.sqrt((1 + s) / 2), np.sqrt((1 - s) / 2)], dtype=complex)


def pauli_readout(N):
    """Precompute the readout set {X_i, Y_i, Z_i, X_iX_{i+1}, Y_iY_{i+1},
    Z_iZ_{i+1}} as (column-index, weight) matrices.  For a Pauli P = i^p X^x Z^z,
    <P> = Re sum_b W[b] rho[b, b^x] -- an O(2^N) dot product per observable.
    (Verified elsewhere against brute-force Tr(P rho).)"""
    bit = lambda i: 1 << (N - 1 - i)
    specs = []
    for i in range(N):
        specs += [(bit(i), 0, 0), (bit(i), bit(i), 1), (0, bit(i), 0)]          # X, Y, Z
    for i in range(N - 1):
        m = bit(i) | bit(i + 1)
        specs += [(m, 0, 0), (m, m, 2), (0, m, 0)]                              # XX, YY, ZZ
    b = np.arange(1 << N)
    cols = np.array([b ^ x for (x, z, p) in specs])
    w = np.array([_PHASE[p] * (1 - 2 * (_POP[z & b] & 1)) for (x, z, p) in specs], dtype=complex)
    return cols, w


def run_reservoir(series, N, H, tau_ev, cols=None, w=None):
    """Drive the dense reservoir with `series`; return the (T, n_features) matrix
    of readout expectation values."""
    if cols is None:
        cols, w = pauli_readout(N)
    U = expm(-1j * H * tau_ev)
    Udag = U.conj().T
    D, D_res = 1 << N, 1 << (N - 1)
    rho = np.zeros((D, D), dtype=complex)
    rho[0, 0] = 1.0                                       # start in |0...0>
    b = np.arange(D)
    F = np.zeros((len(series), cols.shape[0]))
    for n, s in enumerate(series):
        rho_res = np.einsum("aiaj->ij", rho.reshape(2, D_res, 2, D_res))  # trace out qubit 0
        psi = encode(s)
        rho = U @ np.kron(np.outer(psi, psi.conj()), rho_res) @ Udag       # inject + evolve
        F[n] = np.real((w * rho[b[None, :], cols]).sum(axis=1))            # readout
    return F


def ridge_forecast(F, series, tau, washout=80, train_frac=0.6, lam=1e-6):
    """Train a ridge readout to predict s_{n+tau} from the features at step n;
    return the test-set R^2 (squared correlation of prediction and truth)."""
    Xf = F[washout: len(series) - tau]
    y = series[washout + tau:]
    Xf = np.hstack([Xf, np.ones((len(Xf), 1))])          # bias column
    ntr = int(train_frac * len(Xf))
    W = np.linalg.solve(Xf[:ntr].T @ Xf[:ntr] + lam * np.eye(Xf.shape[1]), Xf[:ntr].T @ y[:ntr])
    pred, true = Xf[ntr:] @ W, y[ntr:]
    return float(np.corrcoef(pred, true)[0, 1] ** 2)


if __name__ == "__main__":
    # self-test on a short chaotic (Mackey-Glass) series
    n = 500
    x = np.zeros(n + 200); x[0] = 1.2
    for t in range(len(x) - 1):
        xt = x[t - 17] if t >= 17 else 0.0
        x[t + 1] = x[t] + 0.1 * (0.2 * xt / (1 + xt ** 10) - 0.1 * x[t])
    s = x[200:200 + n]; s = 2 * (s - s.min()) / (s.max() - s.min()) - 1
    N = 5
    H = mixed_field_ising(N, 1.0, 1.0, 0.5)
    F = run_reservoir(s, N, H, tau_ev=1.0)
    print(f"dense QRC reference: N={N}, {F.shape[1]} features")
    for tau in (1, 3, 5):
        print(f"  R^2(tau={tau}) = {ridge_forecast(F, s, tau):.3f}")
