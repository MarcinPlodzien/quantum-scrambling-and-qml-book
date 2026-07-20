#!/usr/bin/env python3
"""
###########################################################################
#  EXERCISE 3.5 — The SWAP Trick in Action
#
#  Chapter 3, Haar Ensembles
#  Topic: SWAP operator, purity, Bell states, Haar-averaged purity
#
#  ---------- EXERCISE STATEMENT ----------
#
#  (a) Prove Tr[(A x B) F] = Tr(AB) for the SWAP operator F.
#  (b) For the Bell state |Phi+> = (|00>+|11>)/sqrt(2), compute rho_A,
#      and verify Tr(rho_A^2) = Tr[(rho_A x rho_A) F_A] = 1/2.
#  (c) Compare with Haar-averaged purity E[gamma] = (m+n)/(mn+1)
#      at m=n=2.  Is the Bell state more or less entangled than typical?
#
#  ---------- ANALYTICAL SOLUTION ----------
#
#  (a) Tr[(A x B) F] = sum_{i,j} <i,j| (A x B) |j,i> = sum_{i,j} A_ij B_ji
#                     = Tr(AB).
#
#  (b) rho_A = Tr_B|Phi+><Phi+| = I/2.
#      Direct: Tr(rho_A^2) = Tr(I/4) = 1/2.
#      SWAP:   (I/2 x I/2) = I_4/4.  Tr[I_4/4 * F] = (1/4)*Tr(F) = (1/4)*2 = 1/2.
#
#  (c) E[gamma] = (2+2)/(2*2+1) = 4/5 = 0.8.
#      Bell purity = 0.5 < 0.8.  Bell state is MORE entangled than typical.
#      At D=4, typicality does not concentrate near maximal entanglement.
#
#  ---------- NUMERICAL VERIFICATION ----------
###########################################################################
"""
import numpy as np

# ========================================================================
# Part (a): SWAP trick identity Tr[(A x B) F] = Tr(AB)
# ========================================================================
print("Part (a): SWAP trick identity for several random matrix pairs")
print("=" * 60)

for D in [2, 3, 4]:
    SWAP = np.zeros((D**2, D**2))
    for i in range(D):
        for j in range(D):
            SWAP[i*D+j, j*D+i] = 1

    A = np.random.randn(D, D) + 1j * np.random.randn(D, D)
    B = np.random.randn(D, D) + 1j * np.random.randn(D, D)

    lhs = np.trace(np.kron(A, B) @ SWAP)
    rhs = np.trace(A @ B)
    print(f"  D={D}: Tr[(AxB)F] = {lhs:.4f}, Tr(AB) = {rhs:.4f}")
    assert abs(lhs - rhs) < 1e-10

# ========================================================================
# Part (b): Bell state purity via direct and SWAP methods
# ========================================================================
print(f"\nPart (b): Bell state |Phi+> purity")
print("=" * 60)

# |Phi+> = (|00> + |11>)/sqrt(2)
bell = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
rho_full = np.outer(bell, bell.conj())

# Partial trace over B (second qubit)
rho_A = np.zeros((2, 2), dtype=complex)
for b in range(2):
    # <b|rho|b> on the second register
    proj = np.zeros((2, 4), dtype=complex)
    proj[0, b] = 1  # maps |0,b> -> |0>
    proj[1, 2+b] = 1  # maps |1,b> -> |1>
    rho_A += proj @ rho_full @ proj.T

print(f"  rho_A = I/2: {np.allclose(rho_A, np.eye(2)/2)}")

# Direct purity
purity_direct = np.trace(rho_A @ rho_A).real
print(f"  Tr(rho_A^2) = {purity_direct:.6f}  (expected 0.5)")
assert abs(purity_direct - 0.5) < 1e-12

# SWAP trick purity
SWAP_2 = np.zeros((4, 4))
for i in range(2):
    for j in range(2):
        SWAP_2[i*2+j, j*2+i] = 1
purity_swap = np.trace(SWAP_2 @ np.kron(rho_A, rho_A)).real
print(f"  Tr[F(rho_A x rho_A)] = {purity_swap:.6f}  (expected 0.5)")
assert abs(purity_swap - 0.5) < 1e-12

# ========================================================================
# Part (c): Comparison with Haar-averaged purity
# ========================================================================
print(f"\nPart (c): Bell state vs Haar-typical purity")
print("=" * 60)

m, n = 2, 2
gamma_haar = (m + n) / (m * n + 1)
print(f"  Bell state purity:      gamma = {purity_direct:.4f}")
print(f"  Haar-averaged purity:   E[gamma] = (m+n)/(mn+1) = {gamma_haar:.4f}")
print(f"  {purity_direct:.4f} < {gamma_haar:.4f}: Bell state is MORE entangled than typical")
print(f"\n  At D=4, the Hilbert space is too small for typicality to")
print(f"  concentrate near maximal entanglement.  The exact Page entropy")
print(f"  for m=n=2 is E[S] = sum_{{k=3}}^4 1/k - 1/4 = 1/3 ~ 0.33,")
print(f"  well below S_max = ln(2) ~ 0.69 achieved by the Bell state.")

