#!/usr/bin/env python3
"""
get_fig_ch7_qrc_3qubit_features.py
==================================
Verification script for the three-qubit QRC worked example (Chapter 7,
Sec. "Temporal inputs: quantum reservoir computing").

FIGURE STATUS: this is a diagnostic figure, NOT included in the final text.
The worked example is carried in prose and equations only; ch7_qml.tex contains
exactly one \\includegraphics (fig_ch7_kernel_concentration.pdf), there is no
\\label{fig:qrc...} anywhere in the book, and no figure number was ever assigned
to this output. The script is kept because its checks back the analytic claims
of that section, so it should be read as a verification harness that happens to
draw a picture, not as figure-generation code. Do not cite a figure number for
it.

Writes (creating the directory if absent):
    ../figures/ch7/fig_ch7_qrc_3qubit_features.pdf

    Panel (a): features ⟨O_α⟩ after one QRC step against the input s, against
               the reference line y = s, showing nonlinearity out of a linear
               encoding
    Panel (b): step-2 features against the PAST input s_1 at fixed s_2, showing
               that the reservoir remembers

Console checks:
  - the encoding is linear:            ⟨Z_1⟩ = s
  - the step-1 features are not:       max deviation from a linear fit in s
  - the reservoir has memory:          step-2 feature range over s_1
  - the QELM POVM Haar collapse:       E[E_b] = I_in / D   (Eq. qelm_povm_haar)
  - the QELM feature variance:         Var[f_b] = (D−1)/(D²(D+1))
                                                            (Eq. qelm_feature_var)

============================================================
PHYSICS BACKGROUND: WHAT QRC IS AND WHAT THE "FEATURES" ARE
============================================================
Quantum reservoir computing is a trainable-readout model. The quantum circuit is
FIXED and never trained; it acts as a nonlinear feature extractor. Only a
classical linear map is fitted, by ridge regression, on top of measured
expectation values. Because that fit is convex with a closed-form solution, the
variational loop is gone and with it the barren plateau. The exponential cost is
not removed, it moves: it reappears in the number of shots needed to resolve the
features (see the last section below).

The features are just expectation values. After each step the reservoir state
ρ_n is measured against a fixed set of observables {O_α} (here Z_1, Z_2, Z_3,
Z_1 Z_2), giving a feature vector f_n = (⟨O_1⟩, ..., ⟨O_d⟩), and the prediction
is ŷ_n = w^T f_n with w the only trained object.

QRC is the temporal version: the input is a time series {s_n}. At each step the
input node (qubit 1) is traced out and re-encoded with the next value,

    E_{s_n}(ρ_{n-1}) = Tr_node[ρ_{n-1}] ⊗ |ψ(s_n)⟩⟨ψ(s_n)| ,

and the state is then mixed by the reservoir Hamiltonian,

    ρ_n = U ( E_{s_n}(ρ_{n-1}) ) U† ,   U = exp(−i H τ) .

The encoding is the amplitude encoding

    |ψ(s)⟩ = √((1+s)/2) |0⟩ + √((1−s)/2) |1⟩  ,   so  ⟨Z_1⟩ = s ,

a LINEAR encoding of the data, which is what makes panel (a) interesting.

============================================================
WHERE THE NONLINEARITY ACTUALLY COMES FROM (PANEL (a))
============================================================
This is the point of the worked example and it is easy to get backwards. The
unitary channel ρ ↦ U ρ U† is a LINEAR map on density matrices, and so are the
injection map and the measurement. Nothing in the quantum evolution can create
nonlinearity in ρ.

The nonlinearity in the classical input s enters through the ENCODING. The full
Bloch representation of the encoded qubit is

    ρ_in(s) = ½ ( I + s Z + √(1−s²) X ) .

⟨Z⟩ = s is linear, but the state is pure, and purity (|r| = 1) forces the
transverse component √(1−s²), which is not linear in s. So the map s ↦ ρ_in(s)
is already nonlinear, even though the observable used to define the encoding
reports s exactly.

The entangling Hamiltonian then does the useful work: it rotates and mixes these
Bloch components across all three qubits, spreading that one √(1−s²) into every
multi-qubit observable, each with its own J τ- and h τ-dependent coefficients.
Panel (a) shows the result: ⟨Z_1⟩ starts on the line y = s and the other features
depart from any straight line in s. The script quantifies this by fitting a
degree-1 polynomial in s and reporting the maximum residual.

So: the encoding supplies the nonlinearity, the scrambling dynamics diversifies
it across the operator basis. A richer reservoir does not make a more nonlinear
function of s, it makes MORE INDEPENDENT nonlinear functions of s available to
the linear readout. That is the expressivity the model trades on.

============================================================
WHERE THE MEMORY COMES FROM (PANEL (b))
============================================================
The partial trace in the injection map is irreversible, and that irreversibility
is the source of fading memory. This matters more than it first appears: a closed
finite-dimensional system under a time-independent Hamiltonian has a discrete
spectrum and is subject to the quantum recurrence theorem, so it would eventually
replay echoes of old inputs and could never settle into processing a stream. The
partial trace acts as a periodic sink that breaks the recurrence and forces
genuine forgetting.

Memory survives one step because Tr_1[ρ_1], the reduced state on qubits 2 and 3
after step 1, still carries s_1-dependent correlations that the Hamiltonian wrote
there. Step 2 overwrites qubit 1 with s_2 but inherits those correlations, so the
step-2 features are functions of BOTH s_1 and s_2 (the text notes that ⟨Z_1 Z_2⟩
at step 2 picks up cross terms in s_1 s_2 and s_2 √(1−s_1²)). Panel (b) sweeps
s_1 at fixed s_2 = 0.3 and shows the step-2 features moving: if the reservoir had
no memory these curves would be flat.

============================================================
HOW THIS CONNECTS TO SCRAMBLING (VERIFICATION 3)
============================================================
The last check is why the whole example sits in a scrambling book. Fold the fixed
unitary and the readout together into an effective POVM on the input alone,

    E_b = ⟨0|_res ( U† Π_b U ) |0⟩_res ,     f_b(x) = Tr_in[E_b ρ_in(x)] ,

so the span of {E_b} is exactly the information the model can access, and no
amount of training data can recover a target outside it. Now push the reservoir
toward Haar. The first-moment twirl gives E[U† Π_b U] = Tr[Π_b] I / D = I/D, and
sandwiching with the reservoir reference state gives

    E[E_b] = I_in / D                                     (Eq. qelm_povm_haar)

The POVM has collapsed to a multiple of the identity: E[f_b(x)] = 1/D for every
input x, so the feature map has lost all discriminating power. The signal is what
is left over as fluctuations around that mean, and for a pure input the second
moment gives

    Var[f_b] = (D−1)/(D²(D+1)) ≈ 1/D²                     (Eq. qelm_feature_var)

The feature spread is of order 1/D. Resolving it against shot noise (a Bernoulli
outcome with p_b ≈ 1/D estimated from ν shots has standard deviation ≈ 1/√(νD))
needs ν = O(D) shots per feature, exponential in the qubit count. This is where
the cost that the convex readout seemed to eliminate comes back.

Hence: too little mixing and the features barely depend on the input, too much
and the POVM concentrates toward I/D and even recent inputs wash out. The useful
window is the edge of scrambling (Fig. 7.3).

============================================================
ALGORITHM
============================================================
  * H = J (Z_1 Z_2 + Z_2 Z_3) + h (X_1 + X_2 + X_3), transverse-field Ising, with
    J = 1.0, h = 0.8, τ = 1.0. U = expm(−i H τ) is formed once and reused.
  * Panel (a): for each s on a 200-point grid, prepare |ψ(s)⟩ ⊗ |00⟩, evolve once
    by U, and measure Z_1, Z_2, Z_3, Z_1 Z_2 as Tr[O ρ]. Fit each feature to a
    line in s and report the largest residual; > 0.01 is flagged NONLINEAR.
  * Panel (b): run step 1 as above, then apply qrc_step(), which traces out
    qubit 1, tensors |ψ(s_2)⟩⟨ψ(s_2)| back in, and evolves by U again. Sweep s_1
    at fixed s_2 and report each feature's range over s_1; > 0.01 is flagged
    MEMORY.
  * Verification 3: draw 500 Haar unitaries on the full D = 8 space, build E_b
    for the b = 0 outcome, average, and compare to I_2/8. Separately evaluate
    f_b = Tr[Π_b U σ U†] on a pure σ and compare its sample variance to
    (D−1)/(D²(D+1)).
  * Everything here is exact linear algebra on 8x8 matrices. There is no shot
    noise anywhere except deliberately in the Haar average of Verification 3.

============================================================
IMPLEMENTATION NOTES
============================================================
1. THE HAAR CHECKS ARE NOT CIRCULAR, AND THIS IS LOAD-BEARING.
   E[E_b] = I_in/D and Var[f_b] = (D−1)/(D²(D+1)) are analytic predictions of the
   Haar twirl. The data they are checked against comes from QR-decomposed Ginibre
   matrices and nothing else: the sampling never references I/D, never references
   the variance formula, and never draws from the induced distribution of f_b.
   So agreement is a genuine test of the moment formulas rather than a test of a
   sampler. (The sister script get_fig_ch7_kernel_concentration.py once got this
   wrong, drawing its "Monte Carlo" points from the very Beta law the figure was
   meant to validate. Preserve this property when editing.)

2. THE QR PHASE CORRECTION IS MANDATORY.
   Q = Q @ diag(d/|d|) with d = diag(R) is not cosmetic. numpy's QR fixes no
   phase convention on the columns of Q, so without dividing out the phases of
   diag(R) the result is biased and is NOT Haar-distributed. Removing that line
   silently breaks Verification 3.

3. THE FEATURE-VARIANCE CHECK NEEDS A PURE INPUT.
   Eq. qelm_feature_var is derived from the collision probability with
   Tr[ρ_in²] = 1. rho_in_test is |0⟩⟨0| for exactly this reason; a mixed input has
   Tr[ρ_in²] < 1 and would legitimately give a different (smaller) variance, so
   the check would fail against a formula that was never claimed to cover it.

4. QUBIT ORDERING. kron3(A,B,C) = A ⊗ B ⊗ C puts qubit 1 in the most significant
   slot. Everything downstream depends on this: kron(rho_in, rho_23) assumes the
   input node is leftmost, and the reshape (2, 4, 2, 4) in qrc_step() and the
   (D_in, D_res, D_in, D_res) reshape in Verification 3 both split the leading
   factor as the input qubit and the trailing one as the two reservoir qubits.

5. TOLERANCES AND SEEDING. Verification 3 uses 500 samples with atol = 0.015 on
   E[E_b] and a 15% relative window on the variance ratio. These are loose enough
   to pass at 500 samples but tight enough to catch a wrong formula (a factor of
   2, or 1/D² in place of (D−1)/(D²(D+1)) at D = 8, would both fail). The seed
   default_rng(42) makes pass/fail deterministic rather than a coin flip on a
   marginal tolerance.

============================================================
RUNTIME
============================================================
A few seconds, single-core. No caching and no flags. 400 evolutions of an 8x8
matrix for the two panels, plus 500 QR decompositions at D = 8.
"""

import sys
from pathlib import Path

import numpy as np
from scipy.linalg import expm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_style import apply_book_style, load_or_compute, panel_label

# ══════════════════════════════════════════════════════════════════════════
#  Pauli infrastructure
# ══════════════════════════════════════════════════════════════════════════
I2 = np.eye(2)
X  = np.array([[0, 1], [1, 0]], dtype=complex)
Y  = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z  = np.array([[1, 0], [0, -1]], dtype=complex)

def kron3(a, b, c):
    return np.kron(np.kron(a, b), c)

# 3-qubit observables
Z1   = kron3(Z, I2, I2)
Z2   = kron3(I2, Z, I2)
Z3   = kron3(I2, I2, Z)
Z1Z2 = kron3(Z, Z, I2)
Z2Z3 = kron3(I2, Z, Z)
Z1Z3 = kron3(Z, I2, Z)
X1   = kron3(X, I2, I2)

# ══════════════════════════════════════════════════════════════════════════
#  QRC protocol
# ══════════════════════════════════════════════════════════════════════════
# Hamiltonian: transverse-field Ising
J_coup = 1.0
h_field = 0.8
tau     = 1.0

H = J_coup * (kron3(Z, Z, I2) + kron3(I2, Z, Z)) \
  + h_field * (kron3(X, I2, I2) + kron3(I2, X, I2) + kron3(I2, I2, X))

assert np.allclose(H, H.conj().T), "H not Hermitian"
U = expm(-1j * H * tau)

def encode(s):
    """Encode scalar s ∈ [-1,1] into qubit state with ⟨Z⟩ = s.

    ⟨Z⟩ = s is linear in s, but the state is pure, so its Bloch vector also
    carries a transverse component √(1−s²). That constraint, not the dynamics,
    is the origin of every nonlinearity in this script (see module docstring).
    """
    return np.array([np.sqrt((1+s)/2), np.sqrt((1-s)/2)], dtype=complex)

def qrc_step(rho_prev, s_new, U):
    """
    One QRC step: trace out qubit 1, inject s_new, evolve.
    rho_prev: 8×8 density matrix on 3 qubits.

    The partial trace is the irreversible part of the protocol and is what
    supplies fading memory; the surviving rho_23 is what carries the past.
    """
    # Partial trace over qubit 1  (reshape to (2,4,2,4) → trace axes 0,2).
    # Index layout is (q1, q23, q1', q23') because kron3 puts qubit 1 in the
    # most significant slot.
    rho_reshaped = rho_prev.reshape(2, 4, 2, 4)
    # Redundant leftover: this einsum computes the same partial trace and is
    # immediately overwritten by the np.trace call below. Kept as written.
    rho_23 = np.einsum('iajb->ab', rho_reshaped * np.eye(2)[:, None, :, None])
    # More explicitly: trace over first qubit
    rho_23 = np.trace(rho_reshaped, axis1=0, axis2=2)

    # Inject new input on qubit 1. rho_23 is the memory; only the input node is
    # overwritten, so this is Eq. qrc_injection.
    psi_new = encode(s_new)
    rho_in = np.outer(psi_new, psi_new.conj())
    rho_injected = np.kron(rho_in, rho_23)

    # Hamiltonian evolution
    return U @ rho_injected @ U.conj().T

def measure(rho, obs):
    """Expectation value ⟨O⟩ = Tr(O ρ)."""
    return np.real(np.trace(obs @ rho))


# ══════════════════════════════════════════════════════════════════════════
#  VERIFICATION 1: Linear encoding  ⟨Z⟩ = s
# ══════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("  VERIFICATION 1: Linear encoding ⟨Z⟩ = s")
print("=" * 60)
s_test = np.array([-0.9, -0.5, 0.0, 0.5, 0.9])
for s in s_test:
    psi = encode(s)
    zexp = psi @ Z[:2, :2] @ psi
    print(f"  s = {s:+.1f}  →  ⟨Z⟩ = {np.real(zexp):+.6f}  (error: {abs(np.real(zexp)-s):.2e})")
print()


# ══════════════════════════════════════════════════════════════════════════
#  PANEL (a): Feature nonlinearity after one QRC step
# ══════════════════════════════════════════════════════════════════════════
s_grid = np.linspace(-0.99, 0.99, 200)
psi_00 = np.array([1, 0, 0, 0], dtype=complex)  # |00⟩ for reservoir qubits

features = {r'$\langle \hat{Z}_1 \rangle$': [],
            r'$\langle \hat{Z}_2 \rangle$': [],
            r'$\langle \hat{Z}_3 \rangle$': [],
            r'$\langle \hat{Z}_1\hat{Z}_2 \rangle$': []}
obs_list = [Z1, Z2, Z3, Z1Z2]

for s in s_grid:
    psi_s = encode(s)
    psi_init = np.kron(psi_s, psi_00)
    rho_0p = np.outer(psi_init, psi_init.conj())
    rho_1 = U @ rho_0p @ U.conj().T

    for key, obs in zip(features.keys(), obs_list):
        features[key].append(measure(rho_1, obs))

# Check nonlinearity quantitatively: fit the best straight line in s and report
# the worst residual. A feature that were merely a rescaled copy of the encoding
# would fit exactly; a nonzero residual is the √(1−s²) component made visible.
for key in features:
    vals = np.array(features[key])
    coeffs = np.polyfit(s_grid, vals, 1)
    linear_fit = np.polyval(coeffs, s_grid)
    residual = np.max(np.abs(vals - linear_fit))
    status = "NONLINEAR ✓" if residual > 0.01 else "linear"
    print(f"  {key:30s}  max deviation from linearity: {residual:.4f}  [{status}]")

# ══════════════════════════════════════════════════════════════════════════
#  PANEL (b): Memory — features at step 2 depend on s_1
# ══════════════════════════════════════════════════════════════════════════
print()
print("=" * 60)
print("  VERIFICATION 2: Memory of s_1 at step 2")
print("=" * 60)

s2_fixed = 0.3
s1_grid = np.linspace(-0.99, 0.99, 200)
memory_Z1Z2 = []
memory_Z2   = []
memory_Z3   = []

for s1 in s1_grid:
    # Step 1
    psi_s1 = encode(s1)
    psi_init = np.kron(psi_s1, psi_00)
    rho_0p = np.outer(psi_init, psi_init.conj())
    rho_1 = U @ rho_0p @ U.conj().T
    # Step 2
    rho_2 = qrc_step(rho_1, s2_fixed, U)
    memory_Z1Z2.append(measure(rho_2, Z1Z2))
    memory_Z2.append(measure(rho_2, Z2))
    memory_Z3.append(measure(rho_2, Z3))

memory_Z1Z2 = np.array(memory_Z1Z2)
memory_Z2 = np.array(memory_Z2)
memory_Z3 = np.array(memory_Z3)

# Check that features at step 2 vary with s1
for name, arr in [("⟨Z₁Z₂⟩", memory_Z1Z2), ("⟨Z₂⟩", memory_Z2), ("⟨Z₃⟩", memory_Z3)]:
    spread = np.max(arr) - np.min(arr)
    print(f"  {name} at step 2 (s₂={s2_fixed}):  range over s₁ = {spread:.4f}  "
          f"{'MEMORY ✓' if spread > 0.01 else 'NO MEMORY'}")


# ══════════════════════════════════════════════════════════════════════════
#  VERIFICATION 3: QELM POVM Haar collapse  E[E_b] = I_in / D
# ══════════════════════════════════════════════════════════════════════════
print()
print("=" * 60)
print("  VERIFICATION 3: QELM POVM Haar collapse")
print("=" * 60)

D = 8
D_in = 2
D_res = 4
n_samples = 500
b = 0   # computational basis outcome

E_b_sum = np.zeros((D_in, D_in), dtype=complex)
f_b_vals = []

ket_0_res = np.zeros(D_res, dtype=complex)
ket_0_res[0] = 1.0

# Use a PURE input state for feature variance check (formula assumes pure input)
rho_in_test = np.array([[1, 0], [0, 0]], dtype=complex)  # |0><0|

rng = np.random.default_rng(42)

for i in range(n_samples):
    # Haar-random unitary via QR decomposition. The diag(R) phase correction is
    # essential: numpy's QR fixes no phase convention, so without dividing out
    # the phases of diag(R) the result is biased and NOT Haar. Note this route
    # uses only Gaussians and a QR; it never references I/D or the variance
    # formula below, so the comparisons are a genuine test of those formulas.
    Z_mat = (rng.standard_normal((D, D)) + 1j * rng.standard_normal((D, D))) / np.sqrt(2)
    Q, R = np.linalg.qr(Z_mat)
    d = np.diag(R)
    Q = Q @ np.diag(d / np.abs(d))

    # Pi_b = |b><b| on full space
    Pi_b = np.zeros((D, D), dtype=complex)
    Pi_b[b, b] = 1.0

    # E_b = <0_res| U^dag Pi_b U |0_res>, the effective POVM element on the
    # input qubit alone. Reshape splits (input, reservoir) per the kron ordering.
    UdPiU = Q.conj().T @ Pi_b @ Q
    UdPiU_reshaped = UdPiU.reshape(D_in, D_res, D_in, D_res)
    E_b_sample = np.einsum('iajb,a,b->ij', UdPiU_reshaped, ket_0_res, ket_0_res.conj())
    E_b_sum += E_b_sample

    # Feature value for variance check. sigma_test is pure, as Eq.
    # qelm_feature_var requires (its derivation assumes Tr[rho_in^2] = 1).
    sigma_test = np.kron(rho_in_test, np.outer(ket_0_res, ket_0_res.conj()))
    f_b = np.real(np.trace(Pi_b @ Q @ sigma_test @ Q.conj().T))
    f_b_vals.append(f_b)

E_b_avg = E_b_sum / n_samples
expected = np.eye(D_in, dtype=complex) / D

print(f"  E[E_b]:\n{np.real(E_b_avg)}")
print(f"  Expected (I_in/D = I₂/{D}):\n{np.real(expected)}")
print(f"  Match: {np.allclose(E_b_avg, expected, atol=0.015)}")

# Feature variance
f_arr = np.array(f_b_vals)
var_f = np.var(f_arr)
expected_var = (D - 1) / (D**2 * (D + 1))
print(f"\n  Var[f_b] = {var_f:.6f}")
print(f"  Expected (D-1)/(D²(D+1)) = {expected_var:.6f}")
print(f"  Ratio: {var_f / expected_var:.3f}  (should be ≈ 1.0)")
print(f"  Match: {abs(var_f/expected_var - 1) < 0.15}")


# ══════════════════════════════════════════════════════════════════════════
#  FIGURE:  Two-panel  (a) Feature nonlinearity  (b) Memory
# ══════════════════════════════════════════════════════════════════════════
apply_book_style()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.0))

# ── Panel (a): Feature map ──
colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
labels = list(features.keys())
for i, (key, vals) in enumerate(features.items()):
    ax1.plot(s_grid, vals, color=colors[i], lw=1.8, label=key)

# Reference line: y = s (linear encoding)
ax1.plot(s_grid, s_grid, 'k--', lw=0.8, alpha=0.4, label=r'$y = s$ (linear)')

ax1.set_xlabel(r'Input $s$')
ax1.set_ylabel(r'Feature $\langle \hat{O}_\alpha \rangle$')
panel_label(ax1, "a", loc="upper left")
ax1.legend(loc='best', framealpha=0.9, ncol=1)
ax1.set_xlim(-1, 1)
ax1.grid(True, alpha=0.15)

# ── Panel (b): Memory ──
ax2.plot(s1_grid, memory_Z1Z2, color='#A23B72', lw=1.8,
         label=r'$\langle \hat{Z}_1\hat{Z}_2 \rangle$')
ax2.plot(s1_grid, memory_Z2, color='#2E86AB', lw=1.8,
         label=r'$\langle \hat{Z}_2 \rangle$')
ax2.plot(s1_grid, memory_Z3, color='#F18F01', lw=1.8,
         label=r'$\langle \hat{Z}_3 \rangle$')

ax2.set_xlabel(r'Past input $s_1$')
ax2.set_ylabel(r'Feature at step 2')
panel_label(ax2, "b", loc="upper left")
ax2.text(0.97, 0.95, rf'$s_2 = {s2_fixed}$', transform=ax2.transAxes,
         va='top', ha='right')
ax2.legend(loc='best', framealpha=0.9)
ax2.set_xlim(-1, 1)
ax2.grid(True, alpha=0.15)

fig.tight_layout(w_pad=3.0)
outpath = '../figures/ch7/fig_ch7_qrc_3qubit_features.pdf'
import os
os.makedirs(os.path.dirname(outpath), exist_ok=True)
fig.savefig(outpath)
print(f"\n  Figure saved to: {outpath}")

print("\n" + "=" * 60)
print("  ALL VERIFICATIONS COMPLETE")
print("=" * 60)
