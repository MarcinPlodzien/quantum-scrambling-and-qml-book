#!/usr/bin/env python3
"""
###########################################################################
#  NUMERICAL EXERCISE -- SYK symmetry classes via Jordan-Wigner
#
#  Chapter 4, Quantum Dynamics & Chaos  (Exercise: SYK symmetry classes)
#  Topic: Majorana SYK on qubits, Wigner-Dyson classes, N_M mod 8 pattern
#
#  ---------- EXERCISE STATEMENT ----------
#
#  Build the Majorana SYK Hamiltonian
#      H = sum_{i<j<k<l} J_{ijkl} chi_i chi_j chi_k chi_l ,   J ~ N(0,1)
#  on L = N_M/2 qubits via the Jordan-Wigner map
#      chi_{2k-1} = (prod_{j<k} Z_j) X_k ,
#      chi_{2k}   = (prod_{j<k} Z_j) Y_k ,
#  which satisfies {chi_a, chi_b} = 2 delta_{ab}.
#
#  Tasks:
#    (a) Verify the Majorana anticommutation relations.
#    (b) For N_M = 8, 10, 12, 14, diagonalize H in one fermion-parity
#        sector and compute the mean gap ratio <r>.  Remove Kramers
#        doublets (GSE case) before forming spacings.
#    (c) Identify the Wigner-Dyson class and confirm the period-8 pattern:
#        N_M mod 8 = 0 -> GOE, 2 -> GUE, 4 -> GSE, 6 -> GUE.
#
#  ---------- ANALYTICAL BACKGROUND ----------
#
#  Reference mean gap ratios:
#      Poisson 0.386, GOE 0.531, GUE 0.600, GSE 0.674.
#  The class is fixed by an antiunitary particle-hole symmetry whose
#  square (+1/-1/absent) depends only on N_M mod 8 -- a reflection of the
#  Bott periodicity of the real Clifford algebras
#  (Garcia-Garcia & Verbaarschot 2016; You, Ludwig & Xu 2017;
#   Cotler et al. 2017).
#
#  ---------- NUMERICAL SOLUTION ----------
###########################################################################
"""
import numpy as np
from numpy.linalg import eigvalsh
from functools import reduce
from itertools import combinations

I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)


def kron(ops):
    return reduce(np.kron, ops)


def majoranas(L):
    """N_M = 2L Majorana operators on L qubits via Jordan-Wigner."""
    g = []
    for k in range(1, L + 1):
        tail = [Z] * (k - 1)
        g.append(kron(tail + [X] + [I2] * (L - k)))   # chi_{2k-1}
        g.append(kron(tail + [Y] + [I2] * (L - k)))   # chi_{2k}
    return g


# ----- Part (a): verify {chi_a, chi_b} = 2 delta_ab -----
print("Part (a): Majorana anticommutation relations")
print("=" * 60)
L_check = 3
g = majoranas(L_check)
NM = 2 * L_check
max_err = 0.0
for a in range(NM):
    for b in range(NM):
        anti = g[a] @ g[b] + g[b] @ g[a]
        target = 2.0 * (a == b) * np.eye(2 ** L_check)
        max_err = max(max_err, np.max(np.abs(anti - target)))
print(f"  L={L_check}: max |{{chi_a,chi_b}} - 2 delta_ab| = {max_err:.2e}")
assert max_err < 1e-12
print("  Jordan-Wigner Majoranas verified.\n")


def gap_ratio(L, nreal, seed=1):
    """Mean gap ratio of Majorana SYK in the +1 parity sector."""
    rng = np.random.default_rng(seed)
    NM = 2 * L
    g = majoranas(L)
    dim = 2 ** L
    pdiag = np.real(np.diag(kron([Z] * L)))   # parity = prod Z
    idx = np.where(pdiag > 0)[0]
    combs = list(combinations(range(NM), 4))
    rs = []
    for _ in range(nreal):
        J = rng.normal(size=len(combs))
        H = np.zeros((dim, dim), dtype=complex)
        for c, jc in zip(combs, J):
            i, j, k, l = c
            H += jc * (g[i] @ g[j] @ g[k] @ g[l])
        H = H[np.ix_(idx, idx)]
        H = (H + H.conj().T) / 2
        E = np.sort(eigvalsh(H).real)
        # remove (Kramers) degeneracies
        Eu = [E[0]]
        for e in E[1:]:
            if e - Eu[-1] > 1e-7:
                Eu.append(e)
        s = np.diff(np.array(Eu))
        s = s[s > 1e-9]
        n = len(s)
        sm = s[int(0.2 * n):int(0.8 * n)]   # central spectrum
        rr = np.minimum(sm[:-1], sm[1:]) / np.maximum(sm[:-1], sm[1:])
        rs.append(np.mean(rr))
    return float(np.mean(rs))


# ----- Parts (b),(c): gap ratio and class identification -----
print("Part (b),(c): gap ratio <r> and Wigner-Dyson class")
print("=" * 60)
print("  Reference: Poisson 0.386, GOE 0.531, GUE 0.600, GSE 0.674\n")

refs = {1: ("GOE", 0.531), 2: ("GUE", 0.600), 4: ("GSE", 0.674)}
expected = {8: "GOE", 10: "GUE", 12: "GSE", 14: "GUE"}
nreal_by_L = {4: 60, 5: 60, 6: 40, 7: 12}

for NM in (8, 10, 12, 14):
    L = NM // 2
    r = gap_ratio(L, nreal_by_L[L])
    # classify by nearest reference value
    cls = min(refs.values(), key=lambda rv: abs(rv[1] - r))[0]
    ok = "OK" if cls == expected[NM] else "MISMATCH"
    print(f"  N_M={NM:2d} (mod 8 = {NM % 8}): <r> = {r:.3f}  ->  {cls}  "
          f"(expected {expected[NM]}) [{ok}]")
    assert cls == expected[NM], f"class mismatch at N_M={NM}"

print("\n  Period-8 pattern confirmed: N_M mod 8 = 0,2,4,6 -> GOE,GUE,GSE,GUE.")
print("  The all-to-all SYK model realizes all three Wigner-Dyson")
print("  ensembles of Chapter 2, selected purely by the Majorana count.")
