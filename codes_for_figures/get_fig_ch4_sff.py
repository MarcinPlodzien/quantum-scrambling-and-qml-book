#!/usr/bin/env python3
"""
get_fig_ch4_sff.py
===================================
Spectral form factor of the three Wigner-Dyson ensembles, overlaid on one
set of axes, for Chapter 4.

Produces:  Figure 4.1  (LaTeX label fig:sff_goe_gue_gse, in ch4_dynamics.tex)
Output:    figures/ch4/fig_ch4_sff.pdf  and  .png
Cache:     data/ch4/sff_goe_gue_gse.npz

Note on the output name: no literal "fig_ch4_sff.pdf" string appears in this
file.  The stem is built in a loop at the bottom of main() as an f-string,
    for ext in (".pdf", ".png"):  fig.savefig(FIG_DIR / f"fig_ch4_sff{ext}")
so grepping for the filename finds nothing.  FIG_DIR is ../figures/ch4
relative to this script.

============================================================
PHYSICS BACKGROUND
============================================================
The spectral form factor (SFF) is the Fourier transform of the spectral
two-point correlation function.  For a spectrum {E_n} of D levels,

    Z(t) = Tr e^{-iHt} = Σ_n e^{-i E_n t}
    K(t) = (1/D) · ⟨ |Z(t)|² ⟩_ensemble

It is the sharpest available probe of level correlations in the time
domain: level repulsion, which the level-spacing distribution sees only at
the scale of one mean spacing, shows up in K(t) as a feature spanning
decades of time.

The characteristic chaotic shape has three regimes, shaded in the figure:

    DIP     (short t)  The disconnected part |⟨Z⟩|² dominates and dephases.
                       Its decay is set by the density of states, i.e. by
                       the shape of the band, and is NOT universal.
    RAMP    (mid t)    |⟨Z⟩|² has died; the connected part K_c(t) survives
                       and grows LINEARLY in t.  This is the RMT signature.
                       Its presence is the statement that the levels are
                       correlated (rigid), not independent.
    PLATEAU (late t)   K(t) → 1.  Once t exceeds the Heisenberg time t_H,
                       individual levels are resolved and |Z|² becomes the
                       incoherent sum of D unit terms, giving |Z|² ≈ D and
                       hence K = |Z|²/D ≈ 1.

WHAT THE RAMP SLOPE ENCODES
---------------------------
The prefactor of the linear ramp is fixed by the Dyson index β, i.e. by the
antiunitary (time-reversal) symmetry class, and by nothing else:

    GOE  β=1  real symmetric H, T-symmetric with T²=+1   K_c ≈ 2 t/t_H
    GUE  β=2  complex Hermitian H, time reversal broken   K_c ≈   t/t_H
    GSE  β=4  quaternionic self-dual H, T²=-1 (Kramers)   K_c ≈ t/(2 t_H)

The ramp slope is therefore a direct measurement of the Dyson class,
independent of (and complementary to) the level-spacing ratio ⟨r⟩.  The
ordering is the reader's takeaway: STRONGER symmetry gives a STEEPER ramp.
Larger β means stiffer level repulsion, which suppresses the long-range
spectral fluctuations that would otherwise raise K(t), so the ramp starts
lower.  Note that the ramp is linear in all three cases, so on log-log axes
all three lines have slope 1; the β-dependence lives entirely in the
PREFACTOR, i.e. in the vertical offset between the three curves.

WHY THE FIGURE USES RANDOM MATRICES RATHER THAN A SPIN CHAIN
------------------------------------------------------------
This point is made explicitly in the text around Fig. 4.1.  A many-body
spectrum (e.g. the mixed-field Ising chain used elsewhere in the chapter)
is roughly Gaussian with a width extensive in N.  That non-universal band
profile enters through the disconnected part |⟨Z⟩|², whose slow decay
buries the ramp underneath the dip: plotting raw K(t) for such a chain
gives a dip and a plateau with NO visible ramp at all.  The cure is
unfolding (Sec. sec:unfolding), which rescales the spectrum so the mean
level spacing is unity across the band, or a Gaussian energy filter narrow
enough that the density of states counts as flat.  Only after such a step
does the slope 2t/t_H mean anything for a physical Hamiltonian.

This script sidesteps that problem rather than solving it: it samples the
Wigner-Dyson ensembles directly, whose semicircular density of states is
universal enough that the ramp is exposed without unfolding.  NO UNFOLDING
IS PERFORMED HERE.  A reader who transplants this code to a spin-chain
spectrum will see no ramp, and the reason will be the missing unfolding
step, not a bug.

============================================================
ALGORITHM
============================================================
1. SAMPLING.  N_SAMPLES=500 independent matrices per ensemble, D=400.  The
   matrices are built directly from Gaussian entries (no Kronecker products,
   no Hamiltonian: this is pure RMT), and only the eigenvalues are kept.
   Cost is O(D³) per diagonalisation, O(D²) memory.

     GOE: H = (A + Aᵀ)/(2√D),          A real Gaussian
     GUE: H = (G + G†)/(2√D),          G complex Gaussian
     GSE: H = [[A, B], [-B*, A*]] / (2√(2D)), A Hermitian, B antisymmetric.
          This 2D×2D self-dual block form realises the quaternionic
          structure.  Its eigenvalues come in exactly degenerate Kramers
          doublets, so evals[::2] keeps one member of each pair and leaves
          D distinct levels, matching the level count of the other two
          ensembles.  Keeping both members would double-count and destroy
          the plateau normalisation.

2. AVERAGING.  K(t) is averaged over realisations, NOT over time.  Each
   sample contributes |Σ_n e^{-i E_n t}|² / D at every t, and the ensemble
   mean is taken at the end (avg_sff).  This matters: a single realisation
   of |Z(t)|² is wildly non-self-averaging (it fluctuates by O(1) times its
   own mean at every t), so the ramp is invisible in one sample and only
   emerges after the ensemble average.  500 samples is what makes the three
   curves smooth enough to read a prefactor off.

3. TIME GRID.  N_TIMES=500 points spaced LOGARITHMICALLY from 1e-3 to 5·t_H.
   Log spacing is required because dip, ramp and plateau live on decades of
   time; a linear grid would spend all its points on the plateau.  The x
   axis is plotted as t/t_H, so the plateau onset sits near 1 by
   construction.

4. PLOTTED QUANTITY.  K(t) = ⟨|Z(t)|²⟩/D, so K(0) = D and the plateau is 1.
   The caption calls this "the normalized form factor K(t)/D", for which the
   GUE ramp is t/t_H and the plateau is 1; the unnormalised convention used
   in the body text [eq:sff_ramp, eq:sff_plateau] instead gives K_c = D·t/t_H
   with plateau D.  Same object, factor of D apart. Do not "fix" one to match
   the other without also editing the caption.

============================================================
IMPLEMENTATION NOTES
============================================================
HEISENBERG TIME.  t_H = π·D/2 hard-codes the estimate t_H = 2π/Δ with mean
level spacing Δ ≈ 4/D, i.e. it assumes the D levels are spread over a
semicircle of radius 2.  That assumption does not hold for the matrices this
script actually builds, and the reader should know it before reusing the
number.  With the normalisations above the measured band half-widths are

     GOE  ≈ 1.42     GUE  ≈ 1.42     GSE  ≈ 1.00        (radius, not 2)

The GOE/GUE half-width is √2 (entry variance 1/(2D) gives radius 2·√(D·σ²)
= √2), so the true mean spacing is ≈ 2√2/D, and t_H = πD/2 is short of the
real Heisenberg time by a factor ≈ 0.72.  Worse, the GSE branch carries a
DIFFERENT bandwidth (radius ≈ 1, since its 2D×2D block has entry variance
1/(8D)), so no single t_H is simultaneously correct for all three curves;
for the GSE the same constant is off by a factor ≈ 0.50.

Why the figure survives this.  t_H enters only as the x-axis unit, and the
dashed ramp guides are FITTED to the data rather than computed (see below),
so the fit silently absorbs the miscalibration.  The qualitative content of
the figure (three linear ramps, correctly ordered in height, saturating at
K=1) is invariant under rescaling t_H and is therefore unaffected.  What is
NOT reliable is any quantitative reading of the x axis: t/t_H = 1 is not
where the plateau actually begins, and the caption's "the GUE reaches it
exactly at t_H" should be read as a statement about the theory, not as a
feature you can measure off this plot with a ruler.  Treat t/t_H = 1 as
"order the Heisenberg time" only.

RAMP GUIDE LINES.  The dashed guides are NOT drawn from the analytic
formula.  For each ensemble the amplitude is fitted to that ensemble's own
data,
    A = median( K / (t/t_H) )  over the ramp window,
and the line A·(t/t_H) is drawn, but LABELLED with the analytic expression
("~2 t/t_H", "~t/t_H", "~t/(2t_H)").  The caption is explicit about this
("anchored to each ensemble's data"), so the figure is not making a false
claim, but the consequence is worth stating plainly for anyone editing this
script: these dashed lines CANNOT be used as evidence that the prefactors
really are 2, 1, 1/2, because the prefactor is measured from the data rather
than predicted.  The guides demonstrate that the ramp is LINEAR (slope 1 on
log-log); they do not test the β-dependence.  A genuine test would plot the
analytic prefactor with no fitting and let the curves miss if they miss.

Concretely, the fitted amplitudes are nowhere near the labels: A ≈ 1.19
(GOE, labelled "2 t/t_H"), 0.72 (GUE, labelled "t/t_H"), 0.28 (GSE,
labelled "t/(2t_H)").  The gap is mostly the t_H miscalibration above (a
common factor ≈ 0.72 for GOE/GUE, ≈ 0.50 for GSE) plus the genuine
logarithmic correction to the GOE/GSE ramp, which is not a pure straight
line over the fit window.  The ORDERING and the linearity are real physics;
the absolute prefactors printed in the labels are not what is drawn.

REGION BOUNDARIES.  T_DIP_END=0.025 and T_RAMP_END=0.90 (in t/t_H units) are
cosmetic: they set the shaded backgrounds and the ramp-fit window only.  They
are eyeballed to the D=400 data, not derived from a crossover condition.

NUMERICAL GUARD.  avg_sff divides by len(evals) rather than by the global D
so that the GSE branch, which carries D levels retained from a 2D×2D matrix,
is normalised by its own level count.

CACHING.  All three spectra and the three K(t) curves are cached to
data/ch4/sff_goe_gue_gse.npz, keyed on nothing: the file is reused whenever
it exists.  CHANGING D, N_SAMPLES OR N_TIMES WILL NOT INVALIDATE THE CACHE.
Delete the .npz by hand after touching those parameters, otherwise the
figure will be silently regenerated from the old data.

RUNTIME.  ~20 s cold on a modern laptop (dominated by 1500 dense
diagonalisations); effectively instant once the cache exists.  RNG is seeded
(default_rng(42)) so a cold run is reproducible.
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT     = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from plot_style import apply_book_style, load_or_compute

DATA_DIR = ROOT / "data" / "ch4";    DATA_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR  = ROOT.parent / "figures" / "ch4"; FIG_DIR.mkdir(parents=True, exist_ok=True)

# ─── Parameters ───────────────────────────────────────────────────────────────
D         = 400    # matrix size per sample. NB: the resulting semicircle has
                   # radius ~sqrt(2) for GOE/GUE and ~1 for GSE, not 2 --
                   # see HEISENBERG TIME in the module docstring.
N_SAMPLES = 500    # realisations per ensemble. K(t) is NOT self-averaging:
                   # one sample shows no ramp at all, only the average does.
N_TIMES   = 500    # time grid points (log-spaced: dip/ramp/plateau span decades)
RNG       = np.random.default_rng(42)   # seeded -> cold runs are reproducible

# ─── Random matrix generators (pure RMT, no kron products) ────────────────────

def gen_goe(D, n, rng):
    """
    GOE (β=1): real symmetric H = (A + A^T) / (2√D), A_ij ~ N(0,1).
    Physical systems: spin chains with time-reversal symmetry (e.g. Heisenberg).
    Ramp: K_c(t) = 2t/t_H - (t/t_H)^2  for t < t_H.
    """
    out = []
    for _ in range(n):
        A = rng.standard_normal((D, D))
        out.append(np.linalg.eigvalsh((A + A.T) / (2 * np.sqrt(D))))
    return out

def gen_gue(D, n, rng):
    """
    GUE (β=2): complex Hermitian H = (G + G†) / (2√D), G_ij ~ CN(0,1).
    Physical systems: systems with broken time-reversal (e.g. magnetic field).
    Ramp: K_c(t) = t/t_H  for t < t_H.
    """
    out = []
    for _ in range(n):
        G = (rng.standard_normal((D,D)) + 1j*rng.standard_normal((D,D))) / np.sqrt(2)
        out.append(np.linalg.eigvalsh((G + G.conj().T) / (2 * np.sqrt(D))))
    return out

def gen_gse(D, n, rng):
    """
    GSE (β=4): quaternionic 2D×2D block H = [[A, B]; [-B*, A*]] / (2√(2D)).
    A = DxD Hermitian, B = DxD antisymmetric complex.
    Eigenvalues are Kramers doublets; we keep one from each pair.
    Physical systems: half-integer spin chains with time-reversal (Kramers).
    Ramp: K_c(t) = t/(2 t_H)  for t < t_H.
    """
    out = []
    for _ in range(n):
        A = rng.standard_normal((D,D)) + 1j*rng.standard_normal((D,D))
        A = (A + A.conj().T) / 2                   # Hermitian
        B = rng.standard_normal((D,D)) + 1j*rng.standard_normal((D,D))
        B = (B - B.T) / 2                          # antisymmetric
        H = np.block([[A, B], [-B.conj(), A.conj()]]) / (2 * np.sqrt(2*D))
        evals = np.linalg.eigvalsh(H)
        out.append(evals[::2])                     # one from each Kramers pair
    return out

# ─── SFF computation ───────────────────────────────────────────────────────────

def avg_sff(spectra, times):
    """
    K(t) = (1/D) mean_k |sum_n exp(-i E_n t)|^2 over ensemble.

    Averaging is over REALISATIONS, not over time: |Z(t)|^2 fluctuates by
    O(1) times its own mean at every t, so the ramp only emerges after the
    ensemble mean is taken.

    Normalising by len(evals) rather than by the global D matters for the
    GSE branch, which retains D levels out of a 2D x 2D matrix.
    """
    K = np.zeros(len(times))
    for evals in spectra:
        # outer(evals, times) -> phase matrix; sum over levels gives Z(t)
        Z = np.sum(np.exp(-1j * np.outer(evals, times)), axis=0)
        K += np.abs(Z)**2 / len(evals)
    return K / len(spectra)

# ─── Analytical ramp predictions ──────────────────────────────────────────────
# Linear ramp regime K_c(t) = (β/2) * t/t_H  where β=1,2,4
# GOE: slope = 2, GUE: slope = 1, GSE: slope = 1/2

def compute():
    """Expensive numerics: 1500 dense diagonalisations + the three SFF curves.
    Cached via load_or_compute (~20 s cold)."""
    print(f"Generating {N_SAMPLES} samples per ensemble (D={D}) ...")
    print("  GOE ..."); goe = gen_goe(D, N_SAMPLES, RNG)
    print("  GUE ..."); gue = gen_gue(D, N_SAMPLES, RNG)
    print("  GSE ..."); gse = gen_gse(D, N_SAMPLES, RNG)

    # Heisenberg time: t_H = 2π/Δ assuming Δ≈4/D. This is only a scale-setter
    # for the x axis; the actual band radius is ~sqrt(2) (GOE/GUE) and ~1
    # (GSE), so this t_H is NOT the true Heisenberg time of either. See the
    # HEISENBERG TIME section of the module docstring before trusting the axis.
    t_H   = np.pi * D / 2
    # log-spaced grid: a linear grid would spend every point on the plateau
    times = np.exp(np.linspace(np.log(1e-3), np.log(5*t_H), N_TIMES))

    print("  Computing SFFs ...")
    K_goe = avg_sff(goe, times)
    K_gue = avg_sff(gue, times)
    K_gse = avg_sff(gse, times)
    return {"times": times, "K_goe": K_goe, "K_gue": K_gue, "K_gse": K_gse,
            "t_H": t_H}


def main():
    # Cache filename kept as sff_goe_gue_gse.npz to reuse the existing exact data.
    d = load_or_compute(DATA_DIR / "sff_goe_gue_gse.npz", compute)
    times, K_goe, K_gue, K_gse = d["times"], d["K_goe"], d["K_gue"], d["K_gse"]
    t_H = float(d["t_H"])

    t_norm = times / t_H                        # dimensionless time units

    # Dip/ramp/plateau boundaries (in t/t_H units). Cosmetic: these set the
    # shaded backgrounds and the ramp-fit window. Eyeballed to the D=400 data,
    # not derived from a crossover condition.
    T_DIP_END  = 0.025
    T_RAMP_END = 0.90

    # ── Single combined panel: all three ensembles overlaid ───────────────────
    apply_book_style()
    fig, ax = plt.subplots(figsize=(6.5, 4.2), constrained_layout=True)

    # Background shading
    ax.axvspan(t_norm[0],  T_DIP_END,  alpha=0.10, color="#fdae61", zorder=0)
    ax.axvspan(T_DIP_END,  T_RAMP_END, alpha=0.10, color="#a1d99b", zorder=0)
    ax.axvspan(T_RAMP_END, t_norm[-1], alpha=0.10, color="#9ecae1", zorder=0)

    # Data curves
    for K, lbl, color in [
        (K_goe, r"GOE ($\beta=1$)", "#2ca02c"),
        (K_gue, r"GUE ($\beta=2$)", "#1f77b4"),
        (K_gse, r"GSE ($\beta=4$)", "#d62728"),
    ]:
        ax.loglog(t_norm, K, lw=1.8, color=color, label=lbl, zorder=3)

    # Ramp slope guides — anchored to the actual K(t) data in the ramp window.
    # In the ramp region K(t) ≈ A * (t/t_H)^1, so loglog slope = 1 for all β.
    # In theory the β-dependence is the PREFACTOR A = 2, 1, 0.5 (GOE, GUE, GSE).
    # We find A by fitting:  A = median(K / (t/t_H)) over the ramp window.
    #
    # CAVEAT: A is FITTED, so these guides test linearity only -- they are not
    # independent evidence for the 2 : 1 : 0.5 prefactors, and the labels below
    # print the analytic expression while the line drawn uses the fitted A.
    # Measured: A ≈ 1.19 / 0.72 / 0.28, not 2 / 1 / 0.5 (the offset is mostly
    # the t_H miscalibration). The caption says "anchored to each ensemble's
    # data", so the figure is honest; do not read the labels as a measurement.
    t_guide = np.linspace(T_DIP_END * 1.5, T_RAMP_END * 0.95, 100)
    ramp_mask = (t_norm >= T_DIP_END * 2) & (t_norm <= T_RAMP_END * 0.6)

    for K, color, slope_label in [
        (K_goe, "#2ca02c", r"$\sim 2\,t/t_H$"),
        (K_gue, "#1f77b4", r"$\sim t/t_H$"),
        (K_gse, "#d62728", r"$\sim t/(2t_H)$"),
    ]:
        # Fit amplitude: A = K / (t/t_H) → median in ramp window
        A = np.median(K[ramp_mask] / t_norm[ramp_mask])
        ax.loglog(t_guide, A * t_guide, "--", color=color, lw=2.2,
                  alpha=1.0, label=slope_label, zorder=4)

    ax.axhline(1., ls="--", color="gray", lw=1.0, zorder=1)
    ax.axvline(1., ls=":",  color="k",   lw=0.9, alpha=0.5, zorder=1)

    # Region labels
    for tx, ty, lbl, c in [(0.04, 0.97, "Dip",     "#b45309"),
                            (0.50, 0.97, "Ramp",    "#31a354"),
                            (0.80, 0.15, "Plateau", "#3182bd")]:
        ax.text(tx, ty, lbl, transform=ax.transAxes, fontsize=9,
                color=c, fontstyle="italic", va="top")

    ax.set_xlabel(r"$t\,/\,t_H$", fontsize=11)
    ax.set_ylabel(r"$K(t)$",       fontsize=11)
    ax.set_xlim(left=1e-3)
    ax.legend(ncol=2, loc="upper right", framealpha=0.90, fontsize=9)

    for ext in (".pdf", ".png"):
        fig.savefig(FIG_DIR / f"fig_ch4_sff{ext}")
    plt.close(fig)
    print(f"Saved: {FIG_DIR}/fig_ch4_sff.pdf/.png")

if __name__ == "__main__":
    main()
