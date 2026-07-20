#!/usr/bin/env python3
"""
get_fig_ch4_sff_unfolded.py
===========================
The spectral form factor after UNFOLDING: the ramp slope really is 2 : 1 : 1/2
across the three Wigner-Dyson classes, and it is measured, not fitted.

Produces Fig. 4.2 -- figures/ch4/fig_ch4_sff_unfolded.pdf

This is the companion to Fig. 4.1. Figure 4.1 plots K(t)/D for GOE/GUE/GSE with
ramp guides anchored to each curve, which shows that each ramp is LINEAR but
cannot, by construction, demonstrate the ratio of slopes between the classes.
This figure supplies what that one cannot: after unfolding each spectrum to unit
mean level spacing, the connected form factor is compared against the
PARAMETER-FREE random-matrix predictions, with nothing fitted.

============================================================
WHY UNFOLDING IS NECESSARY
============================================================
The ramp is a statement about SPECTRAL CORRELATIONS on the scale of the mean
level spacing. A Gaussian ensemble has a semicircular density of states, so the
mean spacing varies across the band, and a Hamiltonian with an extensive
bandwidth (any real many-body system) has a strongly non-uniform density. That
non-universal profile enters the raw form factor through the disconnected part
and buries the ramp under the dip: plotting K(t) for such a spectrum shows a dip
and a plateau with no ramp visible at all.

Unfolding removes the non-universal density. One maps each eigenvalue E_i to

    e_i = N_smooth(E_i),

the smooth (ensemble-averaged) integrated density of states evaluated at E_i.
By construction the unfolded levels e_i have unit mean spacing everywhere (Delta
= 1), so the only structure left in their correlations is the universal random-
matrix part.  The Heisenberg time in unfolded units is then t_H = 2 pi / Delta =
2 pi -- the SAME size-independent scale for all three ensembles (the ramp meets
the plateau at tau = t / t_H = 1), which is exactly what Fig. 4.1's single
t_H = pi D / 2 could not provide (the GSE sits on a different bandwidth, so one
raw t_H cannot serve all three spectra).  Do not confuse t_H = 2 pi with the
plateau VALUE, which is the number of levels (normalized to 1 below).

============================================================
WHAT IS PLOTTED, AND WHY IT IS NOT CIRCULAR
============================================================
For unfolded levels {e_i} and tau = t / t_H, the connected form factor is

    K_c(tau) = < |sum_i exp(-2 pi i e_i tau)|^2 > - |< sum_i exp(...) >|^2,

normalized by the number of levels so that the plateau sits at 1. The measured
K_c(tau) is compared against the closed-form random-matrix results (Mehta), which
contain NO free parameters:

    b_GUE(tau) = tau                                   (tau < 1)
    b_GOE(tau) = 2 tau - tau ln(1 + 2 tau)             (tau < 1)
    b_GSE(tau) = tau/2 - (tau/4) ln|1 - tau|           (tau < 2)

The small-tau slopes of these are 2, 1, 1/2 = 2/beta for beta = 1, 2, 4, which is
the class dependence the figure exists to show. Note the GOE curve is visibly
BELOW its tangent 2 tau even at moderate tau because of the -tau ln(1+2 tau)
term, so a naive "slope 2" guide would miss the data; the full formula does not.
The predictions are evaluated independently of the sampled spectra, so agreement
is a genuine test, not a fit.

============================================================
IMPLEMENTATION NOTES
============================================================
- UNFOLDING is done per realization by fitting a low-order polynomial to the
  staircase (E_i -> i) over the central portion of the band, then evaluating it
  at the E_i. The band edges are trimmed (central ~70%) because the semicircle
  edge is where the polynomial unfolding is least reliable and where Tracy-Widom
  edge physics, not the bulk ramp, dominates.
- GSE is built as a self-dual quaternion matrix (2D x 2D complex). Its spectrum
  is doubly degenerate (Kramers), so we keep one eigenvalue of each pair before
  unfolding; forgetting this halves the effective level count and corrupts the
  ramp.
- The connected SFF needs heavy ensemble averaging to be smooth, so results are
  cached in data/ch4/sff_unfolded.npz. Use --regenerate to force recomputation.

RUNTIME: a few minutes at the default sizes; instant from cache with --plot-only.
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from plot_style import apply_book_style, load_or_compute

OUTPUT_DIR = SCRIPT_DIR.parent / "figures" / "ch4"
DATA_DIR = SCRIPT_DIR / "data" / "ch4"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)
CACHE = DATA_DIR / "sff_unfolded.npz"

D_GOE = D_GUE = 300      # matrix size
D_GSE = 150              # doubled internally to 300 by the quaternion construction
N_REAL = 600            # realizations per ensemble (connected SFF is noisy)
N_TAU = 90
TRIM = 0.15             # fraction trimmed from each band edge before unfolding
POLY = 7                # unfolding polynomial order


def goe(D, rng):
    A = rng.standard_normal((D, D))
    return (A + A.T) / np.sqrt(2)


def gue(D, rng):
    A = (rng.standard_normal((D, D)) + 1j * rng.standard_normal((D, D))) / np.sqrt(2)
    return (A + A.conj().T) / np.sqrt(2)


def gse(D, rng):
    """Self-dual quaternion matrix; spectrum is Kramers-degenerate (kept singly later)."""
    X = rng.standard_normal((D, D)) + 1j * rng.standard_normal((D, D))
    Y = rng.standard_normal((D, D)) + 1j * rng.standard_normal((D, D))
    A = np.block([[X, Y], [-Y.conj(), X.conj()]])
    return (A + A.conj().T) / 2


def unfolded_sff(matfn, D, n_real, taus, kramers, rng):
    Ksum = np.zeros(len(taus))
    Zsum = np.zeros(len(taus), dtype=complex)
    nlev = 0
    for _ in range(n_real):
        E = np.linalg.eigvalsh(matfn(D, rng))
        if kramers:
            E = E[::2]                      # one of each Kramers pair
        n = len(E)
        E = E[int(TRIM * n):int((1 - TRIM) * n)]   # trim band edges
        idx = np.arange(len(E))
        p = np.polyfit(E, idx, POLY)        # smooth integrated DOS
        e = np.polyval(p, E)                # unfolded levels, unit mean spacing
        e -= e.mean()
        Z = np.exp(-2j * np.pi * np.outer(taus, e)).sum(axis=1)
        Ksum += np.abs(Z)**2
        Zsum += Z
        nlev = len(e)
    Ksum /= n_real
    Zsum /= n_real
    return (Ksum - np.abs(Zsum)**2) / nlev   # connected, normalized so plateau -> 1


# closed-form random-matrix ramp+plateau (Mehta), no free parameters
def b_gue(t):
    return np.where(t < 1, t, 1.0)


def b_goe(t):
    t = np.clip(t, 1e-9, None)
    small = 2 * t - t * np.log(1 + 2 * t)
    large = 2 - t * np.log((2 * t + 1) / np.clip(2 * t - 1, 1e-9, None))
    return np.where(t < 1, small, large)


def b_gse(t):
    t = np.clip(t, 1e-9, None)
    small = t / 2 - (t / 4) * np.log(np.abs(1 - t))
    return np.where(t < 2, small, 1.0)


def compute():
    """Expensive numerics: 1800 diagonalisations + per-realization unfolding.
    Cached via load_or_compute (a few minutes cold)."""
    rng = np.random.default_rng(0)
    taus = np.linspace(0.01, 1.6, N_TAU)
    K_goe = unfolded_sff(goe, D_GOE, N_REAL, taus, False, rng)
    K_gue = unfolded_sff(gue, D_GUE, N_REAL, taus, False, rng)
    K_gse = unfolded_sff(gse, D_GSE, N_REAL, taus, True, rng)
    return {"taus": taus, "K_goe": K_goe, "K_gue": K_gue, "K_gse": K_gse}


def plot(taus, K_goe, K_gue, K_gse):
    apply_book_style()
    fig, ax = plt.subplots(figsize=(6.5, 4.5), constrained_layout=True)
    tt = np.linspace(0.001, 1.6, 400)
    for K, pred, col, lab in [
        (K_goe, b_goe, '#2a9d5c', r'GOE ($\beta=1$)'),
        (K_gue, b_gue, '#4361ee', r'GUE ($\beta=2$)'),
        (K_gse, b_gse, '#d62828', r'GSE ($\beta=4$)'),
    ]:
        ax.plot(taus, K, 'o', color=col, ms=4, alpha=0.65, label=lab + ', data')
        ax.plot(tt, pred(tt), '-', color=col, lw=1.8)
    ax.plot([], [], 'k-', lw=1.8, label='RMT prediction (no fit)')
    # tangents showing the 2:1:1/2 small-tau slopes
    ts = np.linspace(0, 0.25, 10)
    for slope, col in [(2, '#2a9d5c'), (1, '#4361ee'), (0.5, '#d62828')]:
        ax.plot(ts, slope * ts, ':', color=col, lw=1.0, alpha=0.7)
    ax.axhline(1.0, color='gray', ls='--', lw=1.0, alpha=0.7)
    ax.set_xlabel(r'$\tau = t / t_H$ (unfolded)')
    ax.set_ylabel(r'connected form factor $K_c(\tau)$')
    ax.set_xlim(0, 1.6)
    ax.set_ylim(0, 1.3)
    ax.legend(loc='lower right', framealpha=0.92)
    plt.savefig(OUTPUT_DIR / 'fig_ch4_sff_unfolded.pdf')
    plt.savefig(OUTPUT_DIR / 'fig_ch4_sff_unfolded.png')
    print(f"Saved: {OUTPUT_DIR / 'fig_ch4_sff_unfolded.pdf'}")
    # report measured vs predicted small-tau slope
    w = (taus > 0.05) & (taus < 0.30)
    for K, name, pred in [(K_goe, 'GOE', 2), (K_gue, 'GUE', 1), (K_gse, 'GSE', 0.5)]:
        s = np.polyfit(taus[w], K[w], 1)[0]
        print(f"  {name}: measured window-slope {s:.3f}  (tau->0 limit {pred})")


if __name__ == "__main__":
    # load_or_compute reuses data/ch4/sff_unfolded.npz when present; pass
    # --regenerate (or FIG_RECOMPUTE=1) to force a fresh computation.
    force = "--regenerate" in sys.argv
    d = load_or_compute(CACHE, compute, force=force)
    plot(d['taus'], d['K_goe'], d['K_gue'], d['K_gse'])
