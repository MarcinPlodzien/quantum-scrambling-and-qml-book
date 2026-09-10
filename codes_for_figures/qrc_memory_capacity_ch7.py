"""Memory capacity of the four-qubit Haar-scrambled reservoir quoted in Chapter 7
(Sec. on quantum reservoir computing): reproduces the numbers C_M(tau) and C_mem in the text.
Setup: N = 4 qubits, one fixed Haar-random unitary (seed 42), input s_n uniform in [-1, 1]
(seed 0) injected on qubit 1 by re-encoding, readout of the eight expectations <X_j>, <Z_j>,
linear readout with intercept fitted by least squares on T = 700 steps after a washout of 100.
numpy only; runs in a few seconds."""
import numpy as np

X = np.array([[0, 1], [1, 0]], dtype=complex)
Z = np.diag([1.0, -1.0]).astype(complex)
I2 = np.eye(2, dtype=complex)

def op_on(op, j, n):
    mats = [I2] * n
    mats[j] = op
    out = mats[0]
    for m in mats[1:]:
        out = np.kron(out, m)
    return out

def haar_unitary(d, rng):
    z = (rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))) / np.sqrt(2)
    q, r = np.linalg.qr(z)
    ph = np.diag(r) / np.abs(np.diag(r))
    return q * ph

def injection(rho, n, ket):
    """Trace out qubit 0 (the input node) and re-prepare it in |ket>."""
    d_rest = 2 ** (n - 1)
    rho4 = rho.reshape(2, d_rest, 2, d_rest)
    rho_rest = np.einsum('iaib->ab', rho4)
    return np.kron(np.outer(ket, ket.conj()), rho_rest)

def capacity(features, target):
    fmat = np.column_stack([np.ones(len(target)), features])
    w, *_ = np.linalg.lstsq(fmat, target, rcond=None)
    pred = fmat @ w
    return 1 - np.sum((target - pred) ** 2) / np.sum((target - target.mean()) ** 2)

NQ, T, WASH = 4, 700, 100
DQ = 2 ** NQ
U_res = haar_unitary(DQ, np.random.default_rng(42))
s_seq = np.random.default_rng(0).uniform(-1, 1, size=T)
obs = [op_on(X, j, NQ) for j in range(NQ)] + [op_on(Z, j, NQ) for j in range(NQ)]
rho = np.zeros((DQ, DQ), dtype=complex); rho[0, 0] = 1.0
feats = np.zeros((T, len(obs)))
for n in range(T):
    ket = np.array([np.sqrt((1 + s_seq[n]) / 2), np.sqrt((1 - s_seq[n]) / 2)], dtype=complex)
    rho = injection(rho, NQ, ket)
    rho = U_res @ rho @ U_res.conj().T
    feats[n] = [np.trace(o @ rho).real for o in obs]
caps = [capacity(feats[WASH:], s_seq[WASH - tau:T - tau]) for tau in range(10)]
print("C_M(tau), tau = 0..9:", " ".join(f"{c:.3f}" for c in caps))
print(f"sum over tau = 0..4: {sum(caps[:5]):.3f};  sum over tau = 0..9: {sum(caps):.3f}")
