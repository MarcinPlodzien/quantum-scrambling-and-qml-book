"""Sectional Curvature on SU(3): numerical verification of the book solution."""
import numpy as np

g = lambda A, B: np.real(np.trace(A.conj().T @ B))   # Hilbert-Schmidt metric, = -Tr(AB) on skew-Hermitian
comm = lambda A, B: A @ B - B @ A
E = lambda i, j: np.outer(np.eye(3)[i], np.eye(3)[j])

X1 = -1j*np.diag([1, -1, 0])/np.sqrt(2)
Y1 = -1j*np.diag([1, 1, -2])/np.sqrt(6)
X2 = -1j*(E(0,1) + E(1,0))/np.sqrt(2)
Y2 = (E(1,0) - E(0,1))/np.sqrt(2)
Y3 = -1j*(E(1,2) + E(2,1))/np.sqrt(2)

for name, (X, Y) in {"(X1,Y1)": (X1, Y1), "(X2,Y2)": (X2, Y2), "(X2,Y3)": (X2, Y3)}.items():
    assert np.allclose(X.conj().T, -X) and np.allclose(Y.conj().T, -Y)          # skew-Hermitian
    assert np.isclose(np.trace(X), 0) and np.isclose(np.trace(Y), 0)            # traceless
    assert np.isclose(g(X, X), 1) and np.isclose(g(Y, Y), 1) and np.isclose(g(X, Y), 0)
    C = comm(X, Y)
    K = g(C, C)/4
    print(f"{name}: |[X,Y]|^2 = {g(C,C):.4f},  K = {K:.4f}")

assert np.isclose(g(comm(X1,Y1), comm(X1,Y1))/4, 0)
assert np.isclose(g(comm(X2,Y2), comm(X2,Y2))/4, 1/2)
assert np.isclose(g(comm(X2,Y3), comm(X2,Y3))/4, 1/8)
print("K = 0, 1/2, 1/8 VERIFIED")
