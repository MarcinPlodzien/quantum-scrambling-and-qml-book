#!/usr/bin/env python3
"""
###########################################################################
#  EXERCISE 4.1 — The C-F Identity (OTOC-Commutator Relation)
#
#  Chapter 4, Scrambling Dynamics
#  Topic: OTOCs, squared commutator, information scrambling
#
#  ---------- EXERCISE STATEMENT ----------
#
#  For unitary W, V and thermal average <.> = Tr(.)/D:
#    F(t) = <W^dag(t) V^dag W(t) V>       (four-point OTOC)
#    C(t) = <[W(t),V]^dag [W(t),V]>       (squared commutator)
#
#  (a) Show C(t) = 2 - 2 Re F(t).
#  (b) At t=0 with [W,V]=0: C(0)=0, F(0)=1.
#      F -> -1 gives C=4;  F -> 0 gives C=2.
#  (c) Why F -> 0 (not -1) at late times for Haar-random W(t)?
#
#  ---------- ANALYTICAL SOLUTION ----------
#
#  (a) Expand [W(t),V]^dag [W(t),V]:
#      = V^dag W^dag W V - V^dag W^dag VW - W^dag V^dag WV + W^dag V^dag VW
#      First & last terms = I (unitarity).
#      <.> gives: C = 2 - F - F* = 2 - 2 Re(F).
#
#  (b) [W,V]=0 => F(0) = <W^dag V^dag WV> = <I> = 1,  C(0) = 0.
#      F=-1: C = 2-2(-1) = 4.   F=0: C = 2.
#
#  (c) For Haar-random W(t), the 4-point function averages to 0
#      by the Weingarten calculus (no preferred phase alignment).
#
#  ---------- NUMERICAL VERIFICATION ----------
###########################################################################
"""
import numpy as np
from scipy.stats import unitary_group
from scipy.linalg import expm

np.random.seed(42)

# --- Pauli matrices ---
sx = np.array([[0,1],[1,0]], dtype=complex)
sz = np.array([[1,0],[0,-1]], dtype=complex)

# ========================================================================
# Part (a): Verify C = 2 - 2 Re(F) numerically
# ========================================================================
print("Part (a): C(t) = 2 - 2 Re F(t)")
print("=" * 60)

D = 4  # Two qubits
W = np.kron(sx, np.eye(2))  # sigma_x on qubit 1
V = np.kron(np.eye(2), sz)  # sigma_z on qubit 2

# Random Hamiltonian for time evolution
H = np.random.randn(D, D) + 1j * np.random.randn(D, D)
H = (H + H.conj().T) / 2

for t in [0.0, 0.5, 1.0, 2.0, 5.0, 10.0]:
    U = expm(-1j * H * t)
    Wt = U.conj().T @ W @ U  # Heisenberg evolution W(t) = U^dag W U

    # OTOC: F = Tr(W(t)^dag V^dag W(t) V) / D
    F = np.trace(Wt.conj().T @ V.conj().T @ Wt @ V) / D

    # Squared commutator: C = Tr([W(t),V]^dag [W(t),V]) / D
    comm = Wt @ V - V @ Wt
    C = np.trace(comm.conj().T @ comm).real / D

    C_from_F = 2 - 2 * F.real

    print(f"  t={t:5.1f}: F = {F.real:+.4f}{F.imag:+.4f}i, "
          f"C = {C:.4f}, 2-2Re(F) = {C_from_F:.4f}")
    assert abs(C - C_from_F) < 1e-10, "C-F identity failed!"

# ========================================================================
# Part (b): Initial conditions F(0)=1, C(0)=0
# ========================================================================
print(f"\nPart (b): Initial conditions and boundary values")
print("=" * 60)
# At t=0, [W,V] = [sx x I, I x sz] = 0
comm_0 = W @ V - V @ W
print(f"  [W,V] = 0 at t=0: {np.allclose(comm_0, 0)}")
print(f"  F(0) = 1, C(0) = 0 (verified above)")
print(f"  If F=-1: C = 2-2(-1) = 4")
print(f"  If F= 0: C = 2-2(0)  = 2")

# ========================================================================
# Part (c): Late-time F -> 0 for Haar-random W(t)
# ========================================================================
print(f"\nPart (c): Haar-random W(t) gives F -> 0 at late times")
print("=" * 60)

F_values = []
for _ in range(5000):
    # Replace W(t) by a Haar-random unitary (fully scrambled limit)
    Wt = unitary_group.rvs(D)
    F = np.trace(Wt.conj().T @ V.conj().T @ Wt @ V) / D
    F_values.append(F)

mean_F = np.mean(F_values)
print(f"  <F>_Haar = {mean_F.real:.4f} + {mean_F.imag:.4f}i  (expected ~0)")
print(f"  |<F>|   = {abs(mean_F):.4f}")
assert abs(mean_F) < 0.05, f"|<F>| = {abs(mean_F)} too large"

print(f"\n  By the Weingarten calculus, the four-point function of")
print(f"  Haar-random unitaries vanishes for D >> 1: no preferred")
print(f"  phase alignment exists between W(t) and V.")

