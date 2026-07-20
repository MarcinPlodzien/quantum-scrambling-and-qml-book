#!/usr/bin/env python3
"""
###########################################################################
#  NUMERICAL EXERCISE -- Classical Shadows: Full Tomography Pipeline
#
#  Chapter 6, Applications
#  Topic: Randomized single-qubit measurements, shadow tomography,
#         observable estimation, sample complexity
#
#  ---------- EXERCISE STATEMENT ----------
#
#  Implement the classical shadow protocol for N-qubit states using
#  randomized single-qubit Clifford measurements.
#
#  Protocol:
#    1. Prepare the unknown state rho.
#    2. For each shot: choose a random Pauli basis (X, Y, or Z) for
#       each qubit independently, measure, obtain bitstring b.
#    3. Construct the single-shot "shadow snapshot":
#         rho_hat = tensor_j (3 * U_j^dag |b_j><b_j| U_j - I)
#    4. Average over shots: rho_shadow = (1/T) sum_t rho_hat^{(t)}.
#
#  Tasks:
#    (a) Implement the shadow protocol for N = 2 qubits.
#    (b) Reconstruct a known state (e.g., the Bell state |Phi+>)
#        and verify ||rho_shadow - rho_true||_F -> 0 as T grows.
#    (c) Estimate Pauli observables <XX>, <ZZ>, <ZI> from the shadows
#        and compare to exact values.
#    (d) Study the sample complexity: plot the estimation error vs T.
#
#  ---------- ANALYTICAL BACKGROUND ----------
#
#  The key identity underlying classical shadows is the channel inversion:
#    M(sigma) = E_{U,b}[U^dag |b><b| U * Tr(U sigma U^dag |b><b|)]
#             = (sigma + Tr(sigma)*I) / 3    (for single-qubit Cliffords)
#
#  The inverse is M^{-1}(tau) = 3*tau - Tr(tau)*I, giving:
#    rho_hat = 3*U^dag|b><b|U - I    (single-shot unbiased estimator)
#
#  For N qubits with independent single-qubit measurements:
#    rho_hat = tensor_j (3*U_j^dag|b_j><b_j|U_j - I_j)
#
#  Observable estimation:
#    <O>_hat = Tr(O * rho_hat)
#  This is an unbiased estimator: E[<O>_hat] = Tr(O * rho).
#
#  The variance depends on the locality of O:
#    Var[<O>_hat] <= 3^k * ||O||_shadow^2 / T
#  where k is the number of qubits O acts on non-trivially.
#  Key advantage: estimating k-local observables requires only
#  O(3^k / epsilon^2) shadows, INDEPENDENT of system size N.
#
#  ---------- NUMERICAL SOLUTION ----------
###########################################################################
"""
import numpy as np

np.random.seed(42)

# ========================================================================
# Setup: Pauli eigenstates for measurement bases
# ========================================================================
# Each qubit is measured in a randomly chosen Pauli basis (X, Y, or Z).
# The eigenstates of each basis are:
#   Z: |0> = [1,0],  |1> = [0,1]
#   X: |+> = [1,1]/sqrt(2),  |-> = [1,-1]/sqrt(2)
#   Y: |+i> = [1,i]/sqrt(2),  |-i> = [1,-i]/sqrt(2)

eigenstates = {
    'Z': [np.array([1, 0], dtype=complex), np.array([0, 1], dtype=complex)],
    'X': [np.array([1, 1], dtype=complex)/np.sqrt(2),
          np.array([1, -1], dtype=complex)/np.sqrt(2)],
    'Y': [np.array([1, 1j], dtype=complex)/np.sqrt(2),
          np.array([1, -1j], dtype=complex)/np.sqrt(2)],
}
basis_names = ['X', 'Y', 'Z']

def single_qubit_shadow(basis_choice, outcome):
    """
    Construct the single-qubit shadow snapshot:
      rho_hat_j = 3 * |phi><phi| - I
    where |phi> is the eigenstate corresponding to (basis_choice, outcome).
    """
    phi = eigenstates[basis_choice][outcome]
    return 3 * np.outer(phi, phi.conj()) - np.eye(2)

# ========================================================================
# Part (a): Classical shadow protocol for N=2 qubits
# ========================================================================
print("Part (a): Classical shadow protocol for the Bell state |Phi+>")
print("=" * 60)

N = 2
D = 2**N

# Target state: |Phi+> = (|00> + |11>)/sqrt(2)
bell = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
rho_true = np.outer(bell, bell.conj())

def run_shadow_protocol(rho, N, T):
    """
    Run the classical shadow protocol with T measurement rounds.
    Returns the list of shadow snapshots.
    """
    D = 2**N
    snapshots = []

    for _ in range(T):
        # Step 1: Choose random Pauli basis for each qubit
        bases = [basis_names[np.random.randint(3)] for _ in range(N)]

        # Step 2: Compute Born probabilities and sample outcome
        # Construct the tensor product of eigenstates
        for outcome_bits in range(D):
            # Build the measurement projector |b><b|
            ket = np.array([1], dtype=complex)
            for q in range(N):
                bit = (outcome_bits >> (N - 1 - q)) & 1
                ket = np.kron(ket, eigenstates[bases[q]][bit])

            prob = abs(ket.conj() @ rho @ ket)**2
            if np.random.rand() < prob:
                # This outcome was selected
                # Step 3: Construct multi-qubit shadow snapshot
                snapshot = np.array([[1]], dtype=complex)
                for q in range(N):
                    bit = (outcome_bits >> (N - 1 - q)) & 1
                    sq = single_qubit_shadow(bases[q], bit)
                    snapshot = np.kron(snapshot, sq)
                snapshots.append(snapshot)
                break
        else:
            # Fallback: sample from the full distribution
            probs = np.zeros(D)
            for ob in range(D):
                ket = np.array([1], dtype=complex)
                for q in range(N):
                    bit = (ob >> (N - 1 - q)) & 1
                    ket = np.kron(ket, eigenstates[bases[q]][bit])
                probs[ob] = abs(ket.conj() @ rho @ ket)**2
            probs /= probs.sum()
            outcome_bits = np.random.choice(D, p=probs)
            snapshot = np.array([[1]], dtype=complex)
            for q in range(N):
                bit = (outcome_bits >> (N - 1 - q)) & 1
                sq = single_qubit_shadow(bases[q], bit)
                snapshot = np.kron(snapshot, sq)
            snapshots.append(snapshot)

    return snapshots

# ========================================================================
# Part (b): Reconstruct the Bell state and measure convergence
# ========================================================================
print("\nPart (b): State reconstruction error vs number of shadows T")
print("=" * 60)

for T in [100, 500, 1000, 5000, 10000]:
    snapshots = run_shadow_protocol(rho_true, N, T)
    rho_shadow = np.mean(snapshots, axis=0)

    # Frobenius norm error
    err_F = np.linalg.norm(rho_shadow - rho_true, 'fro')
    # Trace distance
    delta = rho_shadow - rho_true
    err_td = 0.5 * np.sum(np.abs(np.linalg.eigvalsh(delta)))

    print(f"  T={T:6d}: ||rho_sh - rho||_F = {err_F:.4f}, "
          f"trace distance = {err_td:.4f}")

# ========================================================================
# Part (c): Observable estimation from shadows
# ========================================================================
print("\nPart (c): Estimating Pauli observables from T=5000 shadows")
print("=" * 60)

# Build Pauli operators
sx = np.array([[0,1],[1,0]], dtype=complex)
sy = np.array([[0,-1j],[1j,0]], dtype=complex)
sz = np.array([[1,0],[0,-1]], dtype=complex)
I2 = np.eye(2, dtype=complex)

observables = {
    'ZZ': np.kron(sz, sz),
    'XX': np.kron(sx, sx),
    'ZI': np.kron(sz, I2),
    'IZ': np.kron(I2, sz),
    'XY': np.kron(sx, sy),
}

T = 5000
snapshots = run_shadow_protocol(rho_true, N, T)

for name, O in observables.items():
    # Exact value
    exact = np.trace(O @ rho_true).real

    # Shadow estimate: average of Tr(O * rho_hat) over snapshots
    estimates = [np.trace(O @ snap).real for snap in snapshots]
    shadow_est = np.mean(estimates)
    shadow_err = np.std(estimates) / np.sqrt(T)

    print(f"  <{name}>: exact = {exact:+.4f}, "
          f"shadow = {shadow_est:+.4f} +/- {shadow_err:.4f}")

# ========================================================================
# Part (d): Sample complexity -- error scaling with T
# ========================================================================
print("\nPart (d): Sample complexity for <ZZ> estimation")
print("=" * 60)
print("  The error should scale as 1/sqrt(T):")
print(f"  {'T':>8s}  {'|error|':>10s}  {'1/sqrt(T)':>10s}  {'ratio':>8s}")

O_ZZ = np.kron(sz, sz)
exact_ZZ = np.trace(O_ZZ @ rho_true).real

for T in [50, 200, 500, 2000, 10000]:
    snapshots = run_shadow_protocol(rho_true, N, T)
    estimates = [np.trace(O_ZZ @ snap).real for snap in snapshots]
    err = abs(np.mean(estimates) - exact_ZZ)
    expected = 1.0 / np.sqrt(T)
    print(f"  {T:8d}  {err:10.4f}  {expected:10.4f}  {err/expected:8.2f}")

print(f"\n  The shadow protocol achieves 1/sqrt(T) convergence for")
print(f"  k-local observables, with a prefactor that depends only on")
print(f"  the locality k (here k=2 for ZZ), not the system size N.")
print(f"  This is the central advantage over full state tomography,")
print(f"  which requires O(D^2) = O(4^N) measurements.")
