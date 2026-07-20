#!/usr/bin/env python3
"""
get_fig_ch6_shadow_convergence.py
=================================
Classical shadow convergence for the two-qubit Bell state (Chapter 6).

Produces Fig. 6.3 -- figures/ch6/fig_ch6_shadow_convergence.pdf

    Panel (a): running mean of the Pauli-shadow estimator for ⟨Z⊗Z⟩ on |Φ+⟩
               against the number of snapshots T, with a ±2σ̂/√T band and the
               exact value +1
    Panel (b): the shadow norm ‖P_k‖²_sh = 3^k against the locality k of the
               target Pauli string (analytic curve, no sampling involved)

============================================================
PHYSICS BACKGROUND: WHAT A CLASSICAL SHADOW IS
============================================================
Classical shadows turn randomized measurements into an unbiased estimate of
*any* observable, chosen after the data has been taken. Each of T independent
rounds does three things:

    1. draw a random unitary U from a fixed ensemble and apply it to ρ,
    2. measure in the computational basis, obtaining a bitstring x,
    3. store the classical record (U, x).

The record is post-processed into a snapshot

    ρ_sh = M^{-1}( U† |x⟩⟨x| U ),

where M is the measurement channel of the ensemble. M is a depolarizing channel,
so it is invertible on the traceless part and M^{-1} can be applied on paper. The
construction is built so that

    E[ρ_sh] = ρ        (unbiased, exactly, at every T)

and therefore the sample mean of Tr[O ρ_sh] over rounds estimates Tr[O ρ] for
any O the reader cares to name later.

This script uses the LOCAL PAULI ensemble: each qubit independently gets one of
the three single-qubit basis rotations {X, Y, Z}, i.e. every round measures a
uniformly random Pauli string. The channel then factorizes qubit by qubit, and
so does its inverse:

    ρ_sh = ⊗_j ( 3 u_j† |x_j⟩⟨x_j| u_j  −  I ).

============================================================
WHERE THE FACTOR OF 3 COMES FROM (INVERSE MEASUREMENT CHANNEL)
============================================================
For a single qubit the ensemble average of the post-measurement projector is the
depolarizing channel M(σ) = (σ + Tr(σ) I)/(d+1) with d = 2. Inverting it by the
ansatz M^{-1}(τ) = a τ + b Tr(τ) I and demanding M^{-1} ∘ M = id gives

    M^{-1}(τ) = (d+1) τ − Tr(τ) I ,

and on a post-measurement state τ = u†|x⟩⟨x|u (which has Tr τ = 1) this is the
3 (·) − I above. The 3 is d+1, not a fitted constant.

The factor is what makes the estimator unbiased, and it is also what makes it
noisy. Note 3 u†|x⟩⟨x|u − I is not a state: it has a negative eigenvalue. It is a
formal object whose *average* is ρ.

============================================================
WHY 3^k, AND WHY THE SAMPLE COMPLEXITY LOOKS THE WAY IT DOES
============================================================
Take a k-local Pauli string P (k qubits carry a non-identity factor). Since
Tr[Z] = 0, the single-qubit trace against a snapshot factor is

    Tr[ Z (3 u†|x⟩⟨x|u − I) ] =  3(−1)^x   if that qubit happened to be measured
                                            in the matching (Z) basis,
                              =  0         if it was measured in X or Y.

So Tr[P ρ_sh] is 3^k (±1) when ALL k active qubits landed in the matching basis,
and exactly 0 otherwise. The matching happens with probability (1/3)^k. The
3^k therefore compensates the (1/3)^k hit rate exactly, which is precisely how
unbiasedness survives: rare informative rounds are reweighted to make up for the
many blank ones.

That reweighting sets the noise. The per-snapshot estimate ranges over {0, ±3^k},
so the single-shot variance is bounded by the shadow norm

    ‖P_k‖²_sh = 3^k        (panel (b)).

Two distinct scalings follow, and it is worth keeping them apart:

  * STATISTICAL, 1/√T. At fixed observable, the error of the sample mean is
    σ/√T with σ² ≤ ‖O‖²_sh. This is ordinary Monte Carlo, nothing quantum in it.
    Setting ε = ‖O‖_sh/√T and solving gives T ~ ‖O‖²_sh/ε². This is what panel
    (a) shows: the band narrows as 1/√T, and the exponent is 1/2 because the
    snapshots are i.i.d., not because of any property of ρ.

  * COMBINATORIAL, 3^k and log K. The 3^k is the hit-rate cost above; it is
    exponential in the locality of the observable but independent of the total
    system size N. The log K comes from union-bounding over K observables at
    once and needs the median-of-means estimator (see below). Together:

        T ≳ max_i ‖O_i‖²_sh · log K / ε² .

FOR THIS FIGURE, CONCRETELY. The target is O = Z⊗Z on |Φ+⟩, so k = 2 and
‖O‖²_sh = 9. Both qubits land in the Z basis with probability 1/9. When they do,
the Bell state only ever yields outcomes 00 or 11, so (−1)^{x_1+x_2} = +1 every
time and the snapshot reports +9. Hence the per-snapshot estimate is

    9 with probability 1/9,   0 with probability 8/9,

giving mean 1 (= the exact ⟨Z⊗Z⟩, as it must) and variance 9 − 1 = 8, i.e. a
standard deviation of 2√2 ≈ 2.83. The text quotes the shadow-norm bound
√(3^k/T) = 3/√T for k = 2; the true spread sits just under it, which is the
expected relationship between a variance bound and a variance.

THE TAKEAWAY FOR THE READER: a single shadow snapshot is wildly wrong (it says 9
or 0, never 1), yet the mean of many is right, and the approach to the right
answer is the plain 1/√T of Monte Carlo. What the quantum problem contributes is
not a different exponent in T but the size of the constant in front, 3^k, which
is set by the locality of the observable.

============================================================
ALGORITHM
============================================================
  * The Bell state ρ = |Φ+⟩⟨Φ+| is built explicitly as a 4x4 density matrix, and
    the reference value exact_ZZ = Tr[(Z⊗Z) ρ] = +1 is computed by direct trace.
    That reference is fully independent of the sampling: the figure compares a
    Monte Carlo estimate against exact linear algebra, not against itself.
  * sample_shadow_ZZ() runs T_max = 5000 independent rounds. Each round draws a
    basis per qubit uniformly from {X, Y, Z}, samples an outcome from the exact
    Born probabilities of the rotated state, and returns Tr[(Z⊗Z) ρ_sh] for that
    round.
  * The running mean over the first T of those estimates is evaluated at 80
    log-spaced values of T. The band is ±2 σ̂_T/√T, with σ̂_T the sample standard
    deviation of the same first T estimates (so the band is measured from the
    data, not drawn from the 3/√T bound).
  * Panel (b) is the analytic 3^k for k = 1..6. No sampling, so no agreement to
    check: it is the cost curve, plotted to be read off.

============================================================
IMPLEMENTATION NOTES
============================================================
1. THE EARLY RETURN OF 0.0 IS EXACT, NOT AN APPROXIMATION.
   When either qubit is measured in X or Y, sample_shadow_ZZ() appends 0.0 and
   skips the round without building the snapshot. This is not a shortcut that
   trades accuracy for speed: as shown above, Tr[Z (3 u†|x⟩⟨x|u − I)] is
   identically zero for a mismatched basis, so 0.0 is the exact contribution of
   that round. Those rounds are kept in the average (rather than discarded) on
   purpose. They are what dilutes the mean from 9 down to 1, and dropping them
   would bias the estimator upward by a factor of 9.

2. WHY A PLAIN MEAN AND NOT MEDIAN-OF-MEANS.
   The book's complexity bound is stated for the median-of-means estimator: split
   the T snapshots into L groups, average within each group, and report the median
   of the group means. Taking L ~ log K restores exponential confidence from a
   variance bound alone, and that is where the log K in the bound comes from.

   None of that applies to this figure, and it is worth being exact about why,
   because the loose version of the argument ("heavy tails, so the plain mean is
   not enough") is false here. This figure estimates ONE observable (K = 1), and
   for the Bell state the ZZ snapshot takes only the values 0 and +9. It is
   BOUNDED. The empirical mean of a bounded variable already concentrates
   exponentially by Hoeffding, with no median needed. The Chebyshev-only rate the
   bound assumes is what is PROVABLE knowing nothing but the variance, not what
   this estimator actually does.

   Measured failure probability P(|estimate - 1| > 0.5), 4000 repetitions, L = 8:

       T          plain mean      median-of-means
       50            0.163              0.336
      100            0.078              0.100
      200            0.012              0.023
      500            0.000              0.001

   Median-of-means is strictly WORSE at every T, because splitting into groups
   discards information the plain mean keeps. Plotting it here would show it
   losing, directly beneath text arguing for it. The plain mean is the correct
   object for this figure, its measured spread is exactly what the +/- 2 sigma/sqrt(T)
   band reports, and the caption says "running mean" for that reason.

   The median earns its place elsewhere: when only a variance bound is available,
   and when K observables must be controlled at once. Neither holds for K = 1 with
   bounded snapshots.

3. THE ROTATE-THEN-READ-DIAGONAL STEP IS WRITTEN GENERALLY BUT ONLY EVER RUNS
   WITH U = I. Because of note 1, the code reaches the U = kron(u_1, u_2) line
   only when both bases are 'Z', where both factors are the identity. It is left
   in the general form so the round reads as the protocol of the book's shadows
   figure (rotate, then measure in the computational basis) rather than as a
   Z-basis special case. The 'X' (Hadamard) and 'Y' entries of pauli_bases are
   present for the same reason: they define the ensemble that is being sampled
   from, even though this particular observable never uses them.

4. SEEDING. np.random.seed(42) fixes the legacy global RNG, so the published
   figure is reproducible bit for bit. The estimator is stochastic and the
   running mean at small T wanders visibly; that wander is the point of panel
   (a) and is not a seed artifact.

5. NO VECTORIZATION. The 5000 rounds run in a plain Python loop. At 4x4 matrices
   this costs well under a second, so the loop is kept for readability.

6. OUTPUT PATH IS RELATIVE TO THE WORKING DIRECTORY ('../figures/ch6/...'), not
   to __file__. Run this script from codes_for_figures/ or it will fail to find
   the directory.

============================================================
RUNTIME
============================================================
Under a second, single-core. No caching, no flags: just re-run it.
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_style import apply_book_style, load_or_compute, panel_label

# ── Bell state |Φ+⟩ = (|00⟩ + |11⟩)/√2 ──────────────────────────
rho_bell = np.zeros((4, 4), dtype=complex)
phi_plus = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
rho_bell = np.outer(phi_plus, phi_plus.conj())

# Pauli matrices
I2 = np.eye(2, dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)

# ZZ observable. The reference value comes from exact linear algebra, entirely
# independent of the sampling below, so panel (a) is a real test of the estimator.
ZZ = np.kron(Z, Z)
exact_ZZ = np.real(np.trace(ZZ @ rho_bell))  # = +1

# Single-qubit basis rotations u_j defining the local Pauli ensemble.
# Measuring in the P basis = rotate by u_P, then measure in the computational basis.
pauli_bases = {
    'Z': I2,
    'X': np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2),  # Hadamard
    'Y': np.array([[1, -1j], [1, 1j]], dtype=complex) / np.sqrt(2),
}

def sample_shadow_ZZ(rho, n_shots):
    """Sample n_shots shadow snapshots and estimate ⟨Z⊗Z⟩.

    Returns the per-round values Tr[(Z⊗Z) ρ_sh], one per snapshot, NOT their
    mean: the caller needs the individual estimates to form running means and
    the empirical spread. Each entry is 0 or ±9 (see module docstring).
    """
    estimates = []
    basis_names = ['X', 'Y', 'Z']

    for _ in range(n_shots):
        # Random Pauli basis for each qubit
        b1 = np.random.choice(basis_names)
        b2 = np.random.choice(basis_names)

        # Tr[Z (3 u†|x⟩⟨x|u − I)] = 0 exactly for a mismatched basis, so a round
        # missing either Z contributes an exact 0.0. It is still recorded: these
        # blank rounds are 8/9 of the sample and are what dilutes the surviving
        # +9 down to the true value 1. Dropping them would bias the mean by 9x.
        if b1 != 'Z' or b2 != 'Z':
            estimates.append(0.0)  # No contribution
            continue

        U = np.kron(pauli_bases[b1], pauli_bases[b2])
        # Rotate state and measure. Written in the general protocol form, though
        # this line is only reached with b1 = b2 = 'Z', where U is the identity.
        rotated_rho = U @ rho @ U.conj().T
        probs = np.real(np.diag(rotated_rho))
        probs = np.abs(probs)          # guard against small negative round-off
        probs /= probs.sum()

        # Sample outcome; bits[0] is qubit 1 (most significant), bits[1] qubit 2
        outcome = np.random.choice(4, p=probs)
        bits = [(outcome >> 1) & 1, outcome & 1]

        # Shadow estimator for ZZ: 3^2 * (-1)^(x1+x2).
        # The 3^k = 9 is the inverse-channel factor, and it exactly compensates
        # the (1/3)^k = 1/9 probability of having landed in the matching basis.
        est = 9.0 * (-1) ** (bits[0] + bits[1])
        estimates.append(est)

    return np.array(estimates)


# ── Monte Carlo convergence ──────────────────────────────────────
np.random.seed(42)
T_max = 5000
estimates = sample_shadow_ZZ(rho_bell, T_max)

# Running mean at logarithmically spaced points. Nested prefixes of ONE sample
# path (estimates[:T]) rather than independent reruns per T: this shows the
# convergence of a single experiment as snapshots accumulate, which is what a
# reader running the protocol would see.
T_values = np.unique(np.logspace(1, np.log10(T_max), 80).astype(int))
running_means = []
running_stds = []
for T in T_values:
    mean = np.mean(estimates[:T])
    # Standard error measured from the data, not the 3/sqrt(T) shadow-norm bound.
    std = np.std(estimates[:T]) / np.sqrt(T)
    running_means.append(mean)
    running_stds.append(std)

running_means = np.array(running_means)
running_stds = np.array(running_stds)

# ── Plot ──────────────────────────────────────────────────────────
apply_book_style()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
panel_label(ax1, "a", loc="upper left")
panel_label(ax2, "b", loc="upper right")

# Panel (a): Convergence of ⟨Z⊗Z⟩ estimator
ax1.fill_between(T_values, running_means - 2*running_stds,
                 running_means + 2*running_stds,
                 alpha=0.25, color='steelblue', label=r'$\pm 2\sigma/\sqrt{T}$')
ax1.plot(T_values, running_means, '-', color='steelblue', lw=1.5,
         label=r'Shadow estimator $\hat{o}_T$')
ax1.axhline(exact_ZZ, color='crimson', ls='--', lw=1.2,
            label=r'Exact $\langle \hat{Z}\otimes\hat{Z}\rangle = 1$')
ax1.set_xlabel(r'Number of snapshots $T$')
ax1.set_ylabel(r'shadow estimate of $\langle \hat{Z}\!\otimes\!\hat{Z}\rangle$')
ax1.set_xscale('log')
ax1.set_xlim(10, T_max)
ax1.set_ylim(-1, 3)
ax1.legend(loc='upper right', framealpha=0.9)

# Panel (b): Shadow norm scaling 3^k
ks = np.arange(1, 7)
shadow_norms = 3.0 ** ks
ax2.semilogy(ks, shadow_norms, 'o-', color='darkorange', lw=2, markersize=8,
             label=r'$\|\hat{P}_k\|_{\mathrm{sh}}^2 = 3^k$')
ax2.set_xlabel(r'Locality $k$')
ax2.set_ylabel(r'Shadow norm $\|\hat{P}_k\|_{\mathrm{sh}}^2$')
ax2.set_xticks(ks)
ax2.legend(loc='upper left', framealpha=0.9)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../figures/ch6/fig_ch6_shadow_convergence.pdf')
plt.savefig('../figures/ch6/fig_ch6_shadow_convergence.png')
print(f"Exact ⟨Z⊗Z⟩ = {exact_ZZ}")
print(f"Shadow estimate (T={T_max}): {running_means[-1]:.4f} ± {running_stds[-1]:.4f}")
print("Saved: figures/ch6/fig_ch6_shadow_convergence.pdf")
