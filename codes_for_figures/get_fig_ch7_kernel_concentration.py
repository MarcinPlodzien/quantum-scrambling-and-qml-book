#!/usr/bin/env python3
"""
get_fig_ch7_kernel_concentration.py
===================================
Quantum kernel concentration for Chapter 7 (Quantum Machine Learning).

Produces Fig. 7.4 -- figures/ch7/fig_ch7_kernel_concentration.pdf

    Panel (a): the exact kernel density Beta(1, D-1), for D = 2 ... 128 (log x-axis)
    Panel (b): mean and variance vs D -- exact Weingarten values against a
               genuine Haar Monte Carlo estimate

============================================================
PHYSICS BACKGROUND
============================================================
A fidelity quantum kernel encodes two classical inputs x, y into states via a
feature map V(x), and reads out their overlap:

    kappa(x,y) = |<0| V(x)^dag V(y) |0>|^2

If the feature map is expressive enough that V(x)^dag V(y) behaves like a Haar-random
unitary V on the D-dimensional Hilbert space, the kernel value collapses to the single
random variable

    kappa = |<0| V |0>|^2  =  |V_00|^2 .

The exact law is
                                                            D-2
    kappa ~ Beta(1, D-1),        p(kappa) = (D-1) (1 - kappa)

with the first two moments

    E[kappa]   = 1/D
    Var[kappa] = (D-1) / (D^2 (D+1))   ~   1/D^2   for large D.

The standard deviation therefore tracks the mean: every kernel entry sits within ~1/D
of 1/D. For N qubits, D = 2^N, so the off-diagonal kernel entries are exponentially
close to each other and the Gram matrix approaches (1/D) * Identity. Training data then
becomes indistinguishable to the model, and the number of shots required to resolve two
inputs grows exponentially in N. This is the kernel-method counterpart of the barren
plateau: the obstruction is expressivity itself, not a bad optimizer.

============================================================
WHERE THE MOMENTS COME FROM (Weingarten, second moment)
============================================================
Both moments are Haar integrals over a single matrix entry (Chapter 3):

    E[|U_00|^2] = 1/D              (first moment)
    E[|U_00|^4] = 2/(D(D+1))       (second moment, t=2 Weingarten)

hence

    Var = 2/(D(D+1)) - (1/D)^2 = (D-1) / (D^2 (D+1)) .

These are the "exact" dashed curves in panel (b). They are analytic, and the Monte Carlo
points must be produced *independently* of them or the comparison proves nothing --
see the next section.

============================================================
IMPLEMENTATION NOTE 1: THE MONTE CARLO MUST BE INDEPENDENT
============================================================
An earlier version of this script drew the "Monte Carlo" points with

    kappas = scipy.stats.beta.rvs(1, D-1, size=5000)      # <-- circular!

That samples the very Beta(1, D-1) law whose mean and variance are plotted as the exact
curves. The points then cannot disagree with the curves except through sampling noise:
it validates SciPy's Beta sampler, not the physics. The claim under test -- that a Haar
unitary produces kappa ~ Beta(1, D-1) -- went unchecked.

This version samples actual Haar randomness and never uses the Beta law to generate data.
The agreement in panel (b) is therefore a real test of the Weingarten moments.

============================================================
IMPLEMENTATION NOTE 2: ONLY THE FIRST COLUMN IS NEEDED (O(D) NOT O(D^3))
============================================================
Sampling a full Haar unitary by QR-decomposing a Ginibre matrix costs O(D^3) per sample,
which is why the shortcut above was taken. But kappa depends only on V_00, i.e. only on
the *first column* of V -- and the first column of a Haar-random unitary is exactly a
uniformly random unit vector on the complex sphere in C^D.

A uniform point on that sphere is obtained by normalizing a complex Gaussian vector:

    z ~ CN(0, I_D),    v = z / ||z||,    kappa = |v_0|^2

This costs O(D) per sample instead of O(D^3), is fully vectorizable over samples, and is
*exact* -- not an approximation of the QR construction. It buys ~40x more samples
(200_000 per dimension here, vs 5_000 before) in a fraction of the runtime.

Crucially, this route uses only Gaussian sampling and a normalization. It knows nothing
about the Beta distribution, so reproducing Beta's moments is a genuine result.

A full-QR cross-check at small D is run at the end (CROSS_CHECK_QR) to confirm that the
first-column shortcut agrees with honest QR-sampled Haar unitaries.

============================================================
RUNTIME
============================================================
~10 s single-core for the 200k-sample sweep (dominated by D=128), plus a few seconds
for the QR cross-check. No GPU, no quantum hardware.
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import beta as beta_dist

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_style import apply_book_style, load_or_compute, panel_label

SCRIPT_DIR = Path(__file__).resolve().parent
# The book includes this as ch7/fig_ch7_kernel_concentration.pdf; write there, not into
# the figures/ root (an earlier version did, so re-running regenerated nothing the book used).
OUTPUT_DIR = SCRIPT_DIR.parent / "figures" / "ch7"
DATA_DIR = SCRIPT_DIR / "data" / "ch7"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

N_SAMPLES = 200_000      # per dimension; cheap because we only sample one column
CROSS_CHECK_QR = True    # verify the first-column shortcut against full QR Haar sampling


def haar_random_unitary(D, rng):
    """Haar-random U(D) via QR of a Ginibre matrix.

    The R-diagonal phase correction is essential: numpy's QR fixes no phase convention,
    so without dividing out diag(R)/|diag(R)| the result is biased and NOT Haar.
    Used only for the cross-check -- the main sweep needs just one column (see below).
    """
    Z = (rng.standard_normal((D, D)) + 1j * rng.standard_normal((D, D))) / np.sqrt(2)
    Q, R = np.linalg.qr(Z)
    d = np.diag(R)
    return Q * (d / np.abs(d))


def haar_kernel_samples(D, n, rng):
    """Sample kappa = |<0|V|0>|^2 for n Haar-random V, in O(n*D).

    The first column of a Haar unitary is a uniformly random unit vector on the complex
    sphere, and normalizing a complex Gaussian vector produces exactly that. So we never
    build V: we sample its first column directly. Nothing here references Beta(1, D-1) --
    that law is what the figure is testing.
    """
    z = rng.standard_normal((n, D)) + 1j * rng.standard_normal((n, D))
    z /= np.linalg.norm(z, axis=1, keepdims=True)   # uniform on the complex unit sphere
    return np.abs(z[:, 0]) ** 2


dims_scan = [2, 4, 8, 16, 32, 64, 128]


def compute():
    """Haar Monte Carlo sweep for panel (b): mean and variance of kappa vs D.

    The expensive part of the figure (~10 s, dominated by D=128 at N_SAMPLES).
    The exact Weingarten curves are cached alongside so the plotting code and the
    printed comparison table both read a single consistent object.
    """
    rng = np.random.default_rng(2025)
    mc_means, mc_vars, exact_means, exact_vars = [], [], [], []
    for D in dims_scan:
        kappas = haar_kernel_samples(D, N_SAMPLES, rng)   # genuine Haar, independent of Beta
        mc_means.append(np.mean(kappas))
        mc_vars.append(np.var(kappas))
        exact_means.append(1.0 / D)                        # Weingarten, first moment
        exact_vars.append((D - 1) / (D**2 * (D + 1)))      # Weingarten, second moment
    return {"dims_scan": np.array(dims_scan),
            "mc_means": np.array(mc_means), "mc_vars": np.array(mc_vars),
            "exact_means": np.array(exact_means), "exact_vars": np.array(exact_vars)}


data = load_or_compute(DATA_DIR / "fig_ch7_kernel_concentration.npz", compute)
mc_means, mc_vars = data["mc_means"], data["mc_vars"]
exact_means, exact_vars = data["exact_means"], data["exact_vars"]

# ── Panel (a): Exact kernel distributions (log x-axis) ───────────
dims_a = [2, 4, 8, 16, 32, 64, 128]
colors_a = ['#4361ee', '#3a86a8', '#2a9d5c', '#8ac926', '#e9c46a', '#e76f51', '#d62828']

apply_book_style()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.2))

x_grid = np.logspace(-4, 0, 500)  # log-spaced from 1e-4 to 1

for D, col in zip(dims_a, colors_a):
    # Beta(1, D-1) is the exact law; panel (a) is the analytic prediction, no sampling.
    pdf = beta_dist.pdf(x_grid, 1, D - 1)
    ax1.plot(x_grid, pdf, '-', color=col, lw=2.0,
             label=rf'$D={D}$ ($N={int(np.log2(D))}$)')
    ax1.fill_between(x_grid, pdf, alpha=0.08, color=col)
    ax1.axvline(1.0/D, color=col, ls=':', lw=0.9, alpha=0.5)  # mean sits at 1/D

ax1.set_xscale('log')
ax1.set_xlabel(r'Kernel value $\kappa = |\langle 0|V|0\rangle|^2$')
ax1.set_ylabel('Probability density')
ax1.set_xlim(5e-4, 1.5)
ax1.set_ylim(bottom=0)
ax1.legend(loc='upper right', framealpha=0.9, fontsize=7.5)
panel_label(ax1, "a", loc="upper left")

# ── Panel (b): Mean and variance scaling — Haar MC vs exact Weingarten ──
ax2.semilogy(dims_scan, exact_means, '--', color='#4361ee', lw=1.8, zorder=1,
             label=r'$\mathbb{E}[\kappa] = 1/D$')
ax2.semilogy(dims_scan, mc_means, 'o', color='#4361ee', markersize=8,
             markeredgecolor='white', markeredgewidth=1.0, zorder=2,
             label=r'MC mean $\hat{\mu}$')
ax2.semilogy(dims_scan, exact_vars, '--', color='#e76f51', lw=1.8, zorder=1,
             label=r'$\mathrm{Var}[\kappa] = \frac{D-1}{D^2(D+1)}$')
ax2.semilogy(dims_scan, mc_vars, 's', color='#e76f51', markersize=8,
             markeredgecolor='white', markeredgewidth=1.0, zorder=2,
             label=r'MC variance $\hat{\sigma}^2$')

D_arr = np.array(dims_scan)
ax2.semilogy(D_arr, 1.0/D_arr**2, ':', color='gray', lw=1.2,
             label=r'$1/D^2$ (reference)')

ax2.set_xlabel(r'Hilbert space dimension $D = 2^N$')
ax2.set_ylabel('Value')
ax2.set_xscale('log', base=2)
ax2.legend(loc='upper right', framealpha=0.9, fontsize=8)
panel_label(ax2, "b", loc="lower left")
ax2.grid(True, alpha=0.2, which='both')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'fig_ch7_kernel_concentration.pdf')
plt.savefig(OUTPUT_DIR / 'fig_ch7_kernel_concentration.png')
print(f"Saved: {OUTPUT_DIR / 'fig_ch7_kernel_concentration.pdf'}")

print(f"\nHaar MC ({N_SAMPLES:,} samples/dim) vs exact Weingarten:")
print(f"  {'D':>4} {'MC mean':>10} {'exact':>10} {'ratio':>7} {'MC var':>11} {'exact':>11} {'ratio':>7}")
for D, m, v, em, ev in zip(dims_scan, mc_means, mc_vars, exact_means, exact_vars):
    print(f"  {D:4d} {m:10.6f} {em:10.6f} {m/em:7.4f} {v:11.3e} {ev:11.3e} {v/ev:7.4f}")

if CROSS_CHECK_QR:
    # Confirm the O(D) first-column shortcut reproduces full O(D^3) QR Haar sampling.
    # Small D and few samples only -- this is a consistency check, not a data source.
    print("\nCross-check: first-column shortcut vs full QR Haar unitaries (2,000 samples)")
    print(f"  {'D':>4} {'shortcut':>10} {'full QR':>10} {'exact 1/D':>10}")
    rng_qr = np.random.default_rng(7)
    for D in [2, 4, 8]:
        qr_k = np.array([abs(haar_random_unitary(D, rng_qr)[0, 0])**2 for _ in range(2000)])
        sc_k = haar_kernel_samples(D, 2000, rng_qr)
        print(f"  {D:4d} {sc_k.mean():10.6f} {qr_k.mean():10.6f} {1.0/D:10.6f}")
