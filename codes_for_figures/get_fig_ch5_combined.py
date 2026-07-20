#!/usr/bin/env python3
"""
get_fig_ch5_combined.py
===================================
Combined Chapter 5 figure: Marchenko-Pastur convergence of the entanglement
spectrum (top row, panels a-d) + D_KL versus circuit depth (bottom, panel e).

Produces:  Figure 5.2  (LaTeX label fig:mp_convergence, in ch5_designs.tex)
Output:    figures/ch5/fig_ch5_mp_convergence.pdf  and  .png

============================================================
OUTPUT FILENAME COLLISION -- READ THIS FIRST
============================================================
TWO scripts in this directory write figures/ch5/fig_ch5_mp_convergence.pdf:

    get_fig_ch5_mp_convergence.py  ->  4 panels  (a)-(d), figsize (16, 3.8)
    get_fig_ch5_combined.py (THIS) ->  5 panels  (a)-(e), figsize (16, 7.5)

Whichever runs last wins.  The book's Fig. 5.2 comes from THIS script: its
caption describes panels (a)-(d) AND a panel (e) carrying D_KL versus depth,
which the other script does not draw at all.  The committed PDF confirms it
(page size 956 x 464 pt, aspect ~2.1, matching this script's 16:7.5; the
4-panel script would give aspect ~4.2).

Consequence: running get_fig_ch5_mp_convergence.py after this script will
silently overwrite Fig. 5.2 with a version missing panel (e), and the caption
will then describe a panel that does not exist.  This is documented, not
fixed.  If you regenerate figures in bulk, this script must run LAST.

============================================================
THIS SCRIPT COMPUTES NOTHING
============================================================
It is a pure plotting front-end.  Every number it draws is loaded from .npz
caches produced by two other scripts:

    spectra_ry_N20.npz     <- get_fig_ch5_mp_convergence.py   (panels a-d)
    spectra_haar_N20.npz   <- get_fig_ch5_mp_convergence.py   (panels a-d)
    dkl_ry_N20.npz         <- get_fig_ch5_dkl_convergence.py  (panel e)
    dkl_cliff_N20.npz      <- get_fig_ch5_dkl_convergence.py  (panel e)

There is no generation path and no --regenerate flag: if a cache is missing
this script raises FileNotFoundError.  Run those two scripts first.  The
physics, the simulator and the D_KL estimator all live in them, not here;
this file only decides layout.  Runtime is a couple of seconds.

============================================================
PHYSICS BACKGROUND
============================================================
Both halves of the figure ask one question: has a deterministic brickwork
circuit scrambled well enough to look Haar-random?

THE ENTANGLEMENT SPECTRUM.  Split N=20 qubits into equal halves A and B
(k=10, d_A = d_B = 1024).  Reshape the state into a d_A x d_B matrix M; the
reduced density matrix rho_A = M M^dagger is of Wishart form.  Its
eigenvalues p_i define the entanglement spectrum xi_i = -ln p_i, which is
what the histograms show.

THE TWO EXTREMES.  The spectrum's SHAPE is the diagnostic:

  Stabilizer (Clifford) states: rho_A is proportional to a rank-chi
      projector, so every nonzero eigenvalue is equal and the entanglement
      spectrum is a DELTA FUNCTION at xi = r ln 2.  Flat spectrum = classical
      simulability.  A Clifford circuit can be maximally entangled by the
      entropy and still be trivially simulable, so entropy alone cannot
      detect scrambling; the spectrum can.
  Haar-random states: the eigenvalues follow the Marchenko-Pastur law with
      aspect ratio c = d_A/d_B = 1 (equal cut), giving a BROAD distribution.

So convergence of the histogram to MP is the statement that the circuit has
reached Haar-like entanglement structure, and the distance from MP measures
how far it still has to go.

MARCHENKO-PASTUR IN THE xi VARIABLE.  With x = d_A p rescaled to mean 1,

    rho_MP(x) = sqrt((x_+ - x)(x - x_-)) / (2 pi c x),   x_+- = (1 +- sqrt(c))^2

For the equal cut c=1, so x_- = 0 and x_+ = 4, and the density has a
1/sqrt(x) singularity at the origin.  Changing variables to xi = -ln p with
x = 2^k e^{-xi} brings in the Jacobian |dx/dxi| = x:

    rho_xi(xi) = rho_MP(x) * x = sqrt((x_+ - x)(x - x_-)) / (2 pi c)

The 1/x cancels: in the xi representation MP is a smooth curve vanishing as
a square root at both edges, which is why it is pleasant to plot and why the
book works in xi.  This is Eq. eq:mp_xi of ch5_designs.tex, and mp_density_xi
below implements exactly it (rho computed with the 1/x, then multiplied back
by x -- the cancellation is left explicit rather than simplified away).

WHY A D_KL PANEL IS COMBINED WITH THE HISTOGRAMS
------------------------------------------------
The histograms and the D_KL panel answer complementary halves of the
question, and neither is sufficient alone:

  (a)-(d) show WHAT converges.  The eye can see the spectrum broaden and
      acquire the MP peak.  But "agrees at plot resolution" is exactly as
      strong a claim as the plot's resolution, and by eye a converged
      histogram and a nearly-converged one are indistinguishable.
  (e) shows HOW FAST, and how far.  D_KL(P_data || P_MP) on a log axis
      resolves the last two decades that the histograms cannot, exposing the
      sharp transition near L ~ N/2 and the floor at D_KL ~ 0.005.

More importantly, panel (e) carries the NEGATIVE CONTROL that the histograms
lack.  The Clifford curve (blue) plateaus at D_KL ~ O(1) and never descends:
the Clifford barrier, the spectral fingerprint of a circuit that entangles
without scrambling.  Without that control, the red curve's descent would only
show that SOME circuit reaches MP; with it, the figure shows that MAGIC
(non-Cliffordness) is what does the work.  That contrast is the reason the
two originally separate figures were merged into one.

The convergence shown is an EMPIRICAL observation about this one
deterministic circuit, not a consequence of the design theorems.  The text is
explicit about this: unlike random quantum circuits, where Haar convergence is
proven, deterministic layouts can be trapped in non-ergodic sectors or
sublinear-growth regimes when gate angles are fine-tuned or Clifford-heavy.

============================================================
WHAT IS ACTUALLY PLOTTED
============================================================
Panels (a)-(d), at L = 5, 10, 15, 20 (i.e. up to L = N):
    gray  : pooled Haar-random reference spectra (20 samples, all xi pooled
            into one histogram -- the empirical target)
    red   : the Ry(pi/4)+CX brickwork circuit spectrum at depth L (a SINGLE
            deterministic state, no averaging: the circuit has no randomness
            to average over)
    black : the analytic MP density rho_xi (the theoretical target)
    Both histograms are density-normalised, so the single circuit spectrum
    and the 20-sample Haar pool are directly comparable.

Panel (e): D_KL(P_data || P_MP) versus depth L on a semilog-y axis, red
    circles = non-Clifford (Ry+CX), blue squares = Clifford. Curves are
    labelled by text annotations placed past the right edge of the data
    rather than by a legend.

============================================================
IMPLEMENTATION NOTES
============================================================
DUPLICATED FUNCTION.  mp_density_xi is a verbatim copy of the function in
get_fig_ch5_mp_convergence.py.  The two must stay in sync or the black curve
in Fig. 5.2 will disagree with the MP reference used to compute the cached
D_KL values in panel (e).

THE 1e-25 GUARD.  eigs[eigs > 1e-25] filters eigenvalues that are numerically
zero before taking -log.  Necessary: rho_A has 1024 eigenvalues but a
low-depth circuit has Schmidt rank far below that, so most are ~0 and would
map to xi = +inf, poisoning the histogram range.  (The upstream simulator
already clips at 1e-30; this is the second line of defence.)

NaN MASKING.  mask_ry / mask_cl drop NaN entries from the cached D_KL arrays
so semilogy does not silently break the line. Note the sentinel convention of
the upstream script: degenerate (stabilizer) spectra return 100.0 rather than
inf, and converged ones are floored at 1e-4.

c AND m.  c = m/n with m = 2^k and n = 2^(N-k). For N=20, k=10 this is
exactly c=1, so x_- = 0 and the MP support is [0, 4].

HARD-CODED GEOMETRY.  N_Q=20 and HIST_DEPTHS=[5,10,15,20] must match the
cached files (spectra_ry_N20.npz stores keys "L5".."L20"); they are not read
back from the caches. bins = linspace(2, 28, 55) and ylim (0, 0.37) are tuned
to the N=20 data and will clip other N.

xi_grid starts at 1.0, not 0: the MP curve is only evaluated where the mask
(x_- < x < x_+) can be satisfied, and the plotted window is xlim (2, 28).

fig.axes[0] is used for the shared y-label and relies on creation order (the
top-left panel is added first).
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_style import apply_book_style, load_or_compute, panel_label

BOOK_DIR   = Path(__file__).resolve().parent.parent
DATA_DIR   = Path(__file__).resolve().parent / "data" / "ch5"
OUTPUT_DIR = BOOK_DIR / "figures" / "ch5"
N_Q = 20                      # must match the cached spectra_*_N20.npz files
HIST_DEPTHS = [5, 10, 15, 20]  # panel depths; keys "L5".."L20" in the ry cache

# NOTE: this script computes no numerics of its own -- it is a pure plotting
# front-end that loads the four upstream .npz caches produced by
# get_fig_ch5_mp_convergence.py and get_fig_ch5_dkl_convergence.py.  There is
# therefore no load_or_compute() cache here; the analytic MP curve is trivial.

def mp_density_xi(xi, c, m):
    """
    Marchenko-Pastur density in the entanglement-spectrum variable xi = -ln p.
    Implements Eq. eq:mp_xi of ch5_designs.tex.

    VERBATIM COPY of the function in get_fig_ch5_mp_convergence.py -- keep the
    two in sync, or the black curve here will disagree with the MP reference
    used to compute the cached D_KL in panel (e).
    """
    x = m * np.exp(-xi)              # x = 2^k e^{-xi}, the mean-1 rescaled eigenvalue
    xm = (1 - np.sqrt(c))**2         # x_-  (= 0 for the equal cut, c=1)
    xp = (1 + np.sqrt(c))**2         # x_+  (= 4 for the equal cut, c=1)
    rho = np.zeros_like(x)
    mask = (x > xm) & (x < xp)       # MP has compact support: 0 outside [x_-, x_+]
    rho[mask] = np.sqrt((xp - x[mask]) * (x[mask] - xm)) / \
                (2 * np.pi * c * x[mask])
    # Jacobian |dx/dxi| = x. This cancels the 1/x above, leaving a smooth curve
    # with square-root edges; the cancellation is left explicit, not simplified.
    return rho * x


def plot_combined():
    # No generation path: these four caches come from get_fig_ch5_mp_convergence.py
    # (spectra) and get_fig_ch5_dkl_convergence.py (dkl). Missing -> FileNotFoundError.
    print("  Loading cached data...")
    ry_spec   = np.load(DATA_DIR / f"spectra_ry_N{N_Q}.npz")
    haar_data = np.load(DATA_DIR / f"spectra_haar_N{N_Q}.npz")
    ry_dkl    = np.load(DATA_DIR / f"dkl_ry_N{N_Q}.npz")
    cl_dkl    = np.load(DATA_DIR / f"dkl_cliff_N{N_Q}.npz")

    k = N_Q // 2
    m = 1 << k                       # d_A = 2^k = 1024
    n = (1 << N_Q) >> k              # d_B = 2^(N-k) = 1024
    c = m / n                        # aspect ratio; exactly 1 for the equal cut
    xi_grid = np.linspace(1.0, 30.0, 2000)
    mp_curve = mp_density_xi(xi_grid, c, m)

    # Pool all 20 Haar samples into one histogram: the empirical target that
    # the single (deterministic) circuit spectrum is compared against.
    haar_sp = haar_data["spectra"]
    xi_haar = -np.log(haar_sp[haar_sp > 1e-25])   # guard: drop numerical zeros

    # ── Create combined figure ───────────────────────────────
    apply_book_style()
    fig = plt.figure(figsize=(16, 7.5))
    
    # Top row: 4 panels for MP convergence
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 0.85],
                          hspace=0.35, wspace=0.08)
    
    bins = np.linspace(2, 28, 55)
    letters = ["a", "b", "c", "d"]

    for i, L in enumerate(HIST_DEPTHS):
        ax = fig.add_subplot(gs[0, i])
        
        # Gray Haar histogram
        ax.hist(xi_haar, bins=bins, density=True, alpha=0.35,
                color="gray", edgecolor="white", linewidth=0.4, zorder=1)
        
        # Red circuit histogram: a SINGLE deterministic state, not an average
        eigs = ry_spec[f"L{L}"]
        xi = -np.log(eigs[eigs > 1e-25])   # low depth -> most eigs ~0, must filter
        ax.hist(xi, bins=bins, density=True, alpha=0.7,
                color="#E63946", edgecolor="white", linewidth=0.4, zorder=2)
        
        # MP curve
        ax.plot(xi_grid, mp_curve, "k-", lw=2.5, alpha=0.85, zorder=10)
        
        ax.set_xlim(2, 28)
        ax.set_ylim(0, 0.37)
        # Panel letter via the shared helper; the physics (depth L) stays as a
        # separate annotation stacked just below it in the same corner.
        panel_label(ax, letters[i], loc="upper right")
        ax.text(0.97, 0.86, f"$L={L}$", transform=ax.transAxes, fontsize=12,
                va="top", ha="right")
        ax.set_xlabel(r"$\xi = -\ln\,\lambda$")
        if i > 0:
            ax.tick_params(labelleft=False)
    
    fig.axes[0].set_ylabel("Density")   # relies on creation order: top-left panel
    
    # Bottom row: single wide DKL panel
    ax_dkl = fig.add_subplot(gs[1, :])
    
    depths = ry_dkl["depths"]
    kl_ry = ry_dkl["kl"]
    kl_cl = cl_dkl["kl"]
    
    # Drop NaNs so semilogy does not break the line. Upstream sentinels: 100.0
    # for degenerate (stabilizer) spectra, floor 1e-4 for converged ones.
    mask_ry = ~np.isnan(kl_ry)
    mask_cl = ~np.isnan(kl_cl)
    
    ax_dkl.semilogy(depths[mask_ry], kl_ry[mask_ry], "o-", color="#E63946",
                    lw=2.5, markersize=7, zorder=3)
    ax_dkl.semilogy(depths[mask_cl], kl_cl[mask_cl], "s--", color="#457B9D",
                    lw=2, markersize=6, zorder=3)
    
    ax_dkl.text(max(depths) + 0.5, kl_ry[mask_ry][-1],
                "non-Clifford", fontsize=12, color="#E63946",
                va="center", ha="left", fontweight="bold")
    ax_dkl.text(max(depths) + 0.5, kl_cl[mask_cl][-1],
                "Clifford", fontsize=12, color="#457B9D",
                va="center", ha="left", fontweight="bold")
    
    ax_dkl.set_xlabel("Circuit depth $L$")
    ax_dkl.set_ylabel(
        r"$D_{\mathrm{KL}}\!\left(P_{\mathrm{data}} \| P_{\mathrm{MP}}\right)$")
    ax_dkl.set_xlim(0, max(depths) + 4)
    ax_dkl.set_xticks(np.arange(2, max(depths) + 1, 2))
    panel_label(ax_dkl, "e", loc="upper left")

    # Save. NOTE: get_fig_ch5_mp_convergence.py writes this SAME path with only
    # panels (a)-(d). Book Fig. 5.2 is this 5-panel version, so if you rerun
    # both, this script must go last. See the collision note in the docstring.
    outpath = OUTPUT_DIR / "fig_ch5_mp_convergence.pdf"
    fig.savefig(outpath)
    fig.savefig(outpath.with_suffix(".png"))
    print(f"    → PDF: {outpath}")
    print(f"    → PNG: {outpath.with_suffix('.png')}")
    plt.close(fig)


if __name__ == "__main__":
    print("=" * 55)
    print("  Ch5 — Combined MP + DKL figure")
    print("=" * 55)
    plot_combined()
    print("Done")
