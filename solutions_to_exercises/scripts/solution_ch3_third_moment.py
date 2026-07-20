#!/usr/bin/env python3
"""
###########################################################################
#  NUMERICAL EXERCISE -- Third moment E[|Tr U|^6] via full S_3 calculus
#
#  Chapter 3, Haar Ensembles  (Exercise 3.7c)
#  Topic: Weingarten calculus, S_3 permutations, trace moments
#
#  ---------- EXERCISE STATEMENT ----------
#
#  For Haar-random U in U(D), show that
#      E[|Tr U|^{2k}] = k!   for all D >= k,
#  and in particular E[|Tr U|^6] = 6 for D >= 3, by carrying out the
#  full S_3 Weingarten sum
#      E[|Tr U|^6] = 6 * sum_{tau in S_3} D^{#cycles(tau)} Wg(tau, D)
#                  = 6 * [ D^3 Wg(id) + 3 D^2 Wg((12)) + 2 D Wg((123)) ]
#                  = 6 * 1 = 6.
#
#  ---------- NUMERICAL SOLUTION ----------
#    (1) Build the S_3 Gram matrix G_{s,t} = D^{#cycles(s^{-1} t)} and invert
#        it to get the Weingarten functions; verify the bracket equals 1.
#    (2) Monte-Carlo check E[|Tr U|^6] -> 6 for several D.
###########################################################################
"""
import numpy as np
from itertools import permutations
from scipy.stats import unitary_group


def cycles(perm):
    """Number of cycles of a permutation given as a tuple (0-indexed image)."""
    n = len(perm)
    seen = [False] * n
    c = 0
    for i in range(n):
        if not seen[i]:
            c += 1
            j = i
            while not seen[j]:
                seen[j] = True
                j = perm[j]
    return c


def compose(p, q):
    """(p o q)(i) = p[q[i]]."""
    return tuple(p[q[i]] for i in range(len(p)))


def inverse(p):
    inv = [0] * len(p)
    for i, pi in enumerate(p):
        inv[pi] = i
    return tuple(inv)


def weingarten_S3(D):
    """Return dict perm -> Wg(perm, D) by inverting the Gram matrix."""
    S3 = list(permutations(range(3)))
    G = np.array([[D ** cycles(compose(inverse(s), t)) for t in S3] for s in S3],
                 dtype=float)
    W = np.linalg.inv(G)
    # Wg depends only on cycle type; read it off row for s = identity
    idx = {p: i for i, p in enumerate(S3)}
    e = (0, 1, 2)
    return {p: W[idx[e], idx[p]] for p in S3}, S3


print("Part (1): S_3 Weingarten functions and the bracket identity")
print("=" * 60)
for D in (3, 4, 5, 10):
    Wg, S3 = weingarten_S3(D)
    bracket = sum(D ** cycles(p) * Wg[p] for p in S3)
    # closed-form values (D>=3)
    wid = (D ** 2 - 2) / (D * (D ** 2 - 1) * (D ** 2 - 4))
    wt = -1 / ((D ** 2 - 1) * (D ** 2 - 4))
    wc = 2 / (D * (D ** 2 - 1) * (D ** 2 - 4))
    e = (0, 1, 2)
    transp = (1, 0, 2)
    cyc = (1, 2, 0)
    err = max(abs(Wg[e] - wid), abs(Wg[transp] - wt), abs(Wg[cyc] - wc))
    moment = 6 * bracket
    print(f"  D={D:2d}: sum_tau D^c Wg = {bracket:.6f}  ->  "
          f"E[|Tr U|^6] = 6*{bracket:.4f} = {moment:.4f} "
          f"(closed-form Wg err {err:.1e})")
    assert abs(bracket - 1.0) < 1e-9
    assert abs(moment - 6.0) < 1e-9
print("  Bracket = 1 exactly; analytic E[|Tr U|^6] = 6 for D >= 3.\n")

print("Part (2): Monte-Carlo check")
print("=" * 60)
rng = np.random.default_rng(0)
for D in (3, 4, 5):
    vals = [abs(np.trace(unitary_group.rvs(D, random_state=rng))) ** 6
            for _ in range(200000)]
    m = np.mean(vals)
    print(f"  D={D}: MC E[|Tr U|^6] = {m:.3f}  (expect 6)")
    assert abs(m - 6.0) < 0.15
print("\n  Confirmed: E[|Tr U|^{2k}] = k! (here k=3 -> 6), exact for D >= k.")
