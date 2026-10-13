#!/usr/bin/env python3
"""
###########################################################################
###########################################################################
##
##  SOLUTION: Dynamical Lie Algebra of a Three-Qubit Ising Chain
##
##  Book:     Information Scrambling and the Haar Measure
##  Chapter:  1 -- Mathematical Foundations
##  Section:  1.4 -- Lie Algebras and the Exponential Map
##
###########################################################################
##
##  EXERCISE STATEMENT
##  ------------------
##
##  Open three-qubit transverse-field Ising chain, generators
##    Z1, Z2, Z3, X1X2, X2X3.
##
##  (a) Show that two Pauli strings commute or anticommute, and that
##      for anticommuting strings [-iP, -iQ] is a real multiple of -iR
##      for a single Pauli string R proportional to PQ.
##
##  (b) Take commutators until no new strings appear.  Show that the
##      DLA is spanned by fifteen strings, dim g = 15.
##
##  (c) Show that the parity Z1Z2Z3 commutes with all five generators,
##      compare the parity-allowed dimension (30) with dim g and with
##      dim su(8) = 63, and decide whether parity alone accounts for
##      the restriction.
##
###########################################################################
"""

import numpy as np
import itertools

I2 = np.eye(2, dtype=complex)
PAULI = {'I': I2,
         'X': np.array([[0, 1], [1, 0]], dtype=complex),
         'Y': np.array([[0, -1j], [1j, 0]]),
         'Z': np.array([[1, 0], [0, -1]], dtype=complex)}

def string(s):
    """Pauli string, e.g. 'XZY' = sigma_x (x) sigma_z (x) sigma_y."""
    out = np.array([[1]], dtype=complex)
    for c in s:
        out = np.kron(out, PAULI[c])
    return out

def rank(ops):
    return np.linalg.matrix_rank(np.array([o.flatten() for o in ops]), tol=1e-9)

def dla(generators):
    """Close {-i H_j} under commutators; return a list of basis elements."""
    basis = [-1j * H for H in generators]
    changed = True
    while changed:
        changed = False
        for A, B in itertools.product(list(basis), list(basis)):
            C = A @ B - B @ A
            if np.linalg.norm(C) > 1e-9 and rank(basis + [C]) > len(basis):
                basis.append(C)
                changed = True
    return basis

def check(label, ok):
    """Print the result of a check and stop if it fails."""
    print(label, bool(ok))
    assert ok, label

def ising_generators(N):
    """Open transverse-field Ising chain: Z_i and X_i X_{i+1}."""
    zs = ['I' * i + 'Z' + 'I' * (N - i - 1) for i in range(N)]
    xx = ['I' * i + 'XX' + 'I' * (N - i - 2) for i in range(N - 1)]
    return [string(s) for s in zs + xx]

# (b) three-qubit chain
g = dla(ising_generators(3))
print("dim DLA (N = 3):", len(g))
assert len(g) == 15

claimed = ['ZII', 'IZI', 'IIZ',
           'XXI', 'XYI', 'YXI', 'YYI',
           'IXX', 'IXY', 'IYX', 'IYY',
           'XZX', 'XZY', 'YZX', 'YZY']
C = [-1j * string(s) for s in claimed]
check("the 15 listed strings are independent:", rank(C) == 15)
check("and span the DLA:", rank(C + g) == 15)

# first commutators quoted in the solution
comm = lambda A, B: A @ B - B @ A
check("[-iZ1, -iX1X2] = -2i Y1X2:", np.allclose(comm(-1j*string('ZII'), -1j*string('XXI')), -2j*string('YXI')))
check("[X1X2, X2X3] = 0:", np.allclose(comm(string('XXI'), string('IXX')), 0))
check("[-iX1Y2, -iX2X3] = 2i X1Z2X3:", np.allclose(comm(-1j*string('XYI'), -1j*string('IXX')), 2j*string('XZX')))

# (c) parity
P = string('ZZZ')
check("Z1Z2Z3 commutes with every generator:", all(np.allclose(P @ H, H @ P) for H in ising_generators(3)))
in_dla = rank(C + [-1j * P]) == 15
print("-i Z1Z2Z3 lies in the DLA:", in_dla)
assert not in_dla   # the parity commutes with the DLA but is not generated
print("parity-allowed dimension su(4)+su(4) = 30; dim su(8) = 63")

# growth with N
for N in (2, 3, 4):
    assert len(dla(ising_generators(N))) == N*(2*N-1)
    print(f"N = {N}: dim DLA = {len(dla(ising_generators(N)))},  N(2N-1) = {N*(2*N-1)},  4^N - 1 = {4**N - 1}")
