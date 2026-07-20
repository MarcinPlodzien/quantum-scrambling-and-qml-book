#!/usr/bin/env python3
"""
###########################################################################
#  EXERCISE 5.1 — OTOCs and the Failure of Clifford Chaos
#
#  Chapter 5, Unitary Designs
#  Topic: Clifford circuits, Pauli group, discrete vs continuous scrambling
#
#  ---------- EXERCISE STATEMENT ----------
#
#  Prove that if U(t) is Clifford and W,V are single-qubit Paulis,
#  the squared commutator C(t) can only take values 0 or 4.
#  Conclude: Clifford circuits have no Lyapunov exponent.
#
#  ---------- ANALYTICAL SOLUTION ----------
#
#  Clifford unitaries map Paulis to Paulis: W(t) = U^dag W U in P_N.
#  Any two Pauli operators either commute or anticommute.
#
#  Case 1: [W(t), V] = 0  =>  C(t) = 0.
#
#  Case 2: {W(t), V} = 0  =>  [W(t), V] = 2 W(t) V.
#    C(t) = (1/D) Tr(4 V^dag W^dag W V) = 4 Tr(I)/D = 4.
#
#  No intermediate values are possible.
#  Continuous exponential growth C ~ eps * exp(2*lambda*t) is
#  impossible with only discrete jumps {0, 4}.
#
#  ---------- NUMERICAL VERIFICATION ----------
###########################################################################
"""
import numpy as np
from itertools import product

# --- Pauli matrices ---
I2 = np.eye(2, dtype=complex)
sx = np.array([[0,1],[1,0]], dtype=complex)
sy = np.array([[0,-1j],[1j,0]])
sz = np.array([[1,0],[0,-1]], dtype=complex)
paulis_1q = [I2, sx, sy, sz]

# --- Common Clifford gates ---
H_gate = np.array([[1,1],[1,-1]], dtype=complex) / np.sqrt(2)
S_gate = np.array([[1,0],[0,1j]], dtype=complex)
CNOT = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]], dtype=complex)

def random_clifford_2q():
    """Generate a random 2-qubit Clifford by composing elementary gates."""
    U = np.eye(4, dtype=complex)
    for _ in range(10):
        choice = np.random.randint(5)
        if choice == 0: U = np.kron(H_gate, I2) @ U
        elif choice == 1: U = np.kron(I2, H_gate) @ U
        elif choice == 2: U = np.kron(S_gate, I2) @ U
        elif choice == 3: U = np.kron(I2, S_gate) @ U
        elif choice == 4: U = CNOT @ U
    return U

print("Verifying C(t) in {0, 4} for Clifford circuits on 2 qubits")
print("=" * 60)

np.random.seed(42)
D = 4

# Generate all 2-qubit Pauli operators (excluding identity)
pauli_ops = []
pauli_labels = []
for i, Pi in enumerate(paulis_1q):
    for j, Pj in enumerate(paulis_1q):
        if i == 0 and j == 0:
            continue  # skip identity
        pauli_ops.append(np.kron(Pi, Pj))
        labels = ['I', 'X', 'Y', 'Z']
        pauli_labels.append(f"{labels[i]}{labels[j]}")

n_trials = 100
all_C_values = set()

for trial in range(n_trials):
    U = random_clifford_2q()

    # Pick random Pauli operators W and V
    w_idx = np.random.randint(len(pauli_ops))
    v_idx = np.random.randint(len(pauli_ops))
    W = pauli_ops[w_idx]
    V = pauli_ops[v_idx]

    # W(t) = U^dag W U  (Heisenberg evolution under Clifford)
    Wt = U.conj().T @ W @ U

    # Squared commutator: C = Tr([W(t),V]^dag [W(t),V]) / D
    comm = Wt @ V - V @ Wt
    C = np.trace(comm.conj().T @ comm).real / D

    # C should be exactly 0 or 4
    assert np.isclose(C, 0) or np.isclose(C, 4), \
        f"Trial {trial}: C = {C} not in {{0, 4}} for W={pauli_labels[w_idx]}, V={pauli_labels[v_idx]}"
    all_C_values.add(round(C))

print(f"  Tested {n_trials} random (Clifford, W, V) combinations.")
print(f"  All observed C values: {sorted(all_C_values)}")
print(f"  Only {{0, 4}} observed, as predicted.")
print(f"\n  Since C(t) jumps discretely between 0 and 4, no smooth")
print(f"  exponential growth regime C ~ eps*exp(2*lambda*t) exists.")
print(f"  Clifford circuits therefore have no Lyapunov exponent.")

