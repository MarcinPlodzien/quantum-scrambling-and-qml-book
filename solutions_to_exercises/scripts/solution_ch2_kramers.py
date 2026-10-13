"""Time Reversal for Two Spins and in a Field: numerical verification of the book solution."""
import numpy as np

sx = np.array([[0, 1], [1, 0]], complex)
sy = np.array([[0, -1j], [1j, 0]])
sz = np.diag([1., -1.]).astype(complex)

# (a) two spins: T = U K
U = np.kron(-1j*sy, -1j*sy)
T = lambda v: U @ v.conj()
assert np.allclose(U @ U.conj(), np.eye(4))
print("(a) T^2 = +1 for two spins")

ket = lambda s: np.eye(4)[int(s, 2)].astype(complex)
for s in ["00", "01", "10", "11"]:
    print(f"    T|{s}> =", np.round(T(ket(s)).real, 3))

basis = [(ket("00")+ket("11"))/np.sqrt(2), 1j*(ket("00")-ket("11"))/np.sqrt(2),
         (ket("01")-ket("10"))/np.sqrt(2), 1j*(ket("01")+ket("10"))/np.sqrt(2)]
B = np.column_stack(basis)
assert all(np.allclose(T(v), v) for v in basis) and np.allclose(B.conj().T @ B, np.eye(4))

S = [sx/2, sy/2, sz/2]
Hh = sum(np.kron(a, a) for a in S)
Hb = B.conj().T @ Hh @ B
assert np.allclose(Hb.imag, 0)
print("    S1.S2 in the T-fixed basis is real, diagonal", np.round(np.diag(Hb).real, 3))

# (b) one spin in a field: T = Ut K
Ut = -1j*sy
Bv = np.array([0.3, 0.5, 0.7])
H1 = -(Bv[0]*sx + Bv[1]*sy + Bv[2]*sz)
assert all(np.allclose(Ut @ s.conj() @ Ut.conj().T, -s) for s in (sx, sy, sz))
assert np.allclose(Ut @ H1.conj() @ Ut.conj().T, -H1)
ev = np.linalg.eigvalsh(H1)
print("(b) T H T^-1 = -H; levels", np.round(ev, 4), " splitting", round(ev[1]-ev[0], 4), "= 2|B| =", round(2*np.linalg.norm(Bv), 4))

# (c) GSE spectrum with both Kramers partners kept: every other spacing is zero, so r_n = 0
rng = np.random.default_rng(0)
M = 200
A = rng.normal(size=(2*M, 2*M)) + 1j*rng.normal(size=(2*M, 2*M))
H = A + A.conj().T
J = np.kron(np.eye(M), -1j*sy)                # T = J K on 2M-dim space, T^2 = -1
Hs = (H + J @ H.conj() @ J.conj().T)/2        # time-reversal-symmetric part: a GSE-type matrix
e = np.sort(np.linalg.eigvalsh(Hs))
s = np.diff(e)
r = np.minimum(s[:-1], s[1:])/np.maximum(s[:-1], s[1:])
print("(c) with both partners kept: <r> =", round(r.mean(), 6))

e1 = e[::2]
s1 = np.diff(e1)
r1 = np.minimum(s1[:-1], s1[1:])/np.maximum(s1[:-1], s1[1:])
print("    one member per pair: <r> =", round(r1.mean(), 3), "(GSE value 0.674, finite M)")
