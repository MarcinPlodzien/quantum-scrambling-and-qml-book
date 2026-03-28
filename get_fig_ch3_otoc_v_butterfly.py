#!/usr/bin/env python3
"""
===============================================================================
EXTRACTING THE BUTTERFLY VELOCITY v_B FROM SPATIALLY-RESOLVED OTOCs
===============================================================================

This script demonstrates how to extract the butterfly velocity v_B from
the spatially-resolved OTOC data computed by get_fig_ch3_otoc.py.

===============================================================================
1. WHAT IS THE BUTTERFLY VELOCITY?
===============================================================================

The butterfly velocity v_B characterizes the speed at which quantum
information spreads through a many-body system.  It is defined through
the spatially-resolved squared commutator:

    C_{0j}(t) = (1/D) Tr( [Ŵ_0(t), V̂_j]† [Ŵ_0(t), V̂_j] )

where Ŵ_0 = σ^z_0 is localized at site 0 and V̂_j = σ^z_j is a probe
at site j.  At t = 0, the operators commute and C_{0j}(0) = 0.

As the Heisenberg-evolved operator Ŵ_0(t) grows under time evolution,
its support spreads across the chain.  The commutator becomes nonzero
when Ŵ_0(t) reaches site j.  This defines an onset time t*(j):

    C_{0j}(t*(j)) = ε     (threshold crossing)

In chaotic systems, the onset time grows linearly with distance:

    t*(j) ≈ j / v_B + t_offset

The slope of the t*(j) vs j curve gives the butterfly velocity:

    v_B = Δj / Δt*

which is the speed of the "operator front" — the boundary between
the scrambled and unscrambled regions of the chain.

===============================================================================
2. RELATIONSHIP TO LIEB-ROBINSON VELOCITY
===============================================================================

The Lieb-Robinson bound guarantees that for any local Hamiltonian,
commutators between spatially separated operators satisfy:

    ‖[Ô_x(t), Ô_y]‖ ≤ c · exp(−(|x−y| − v_LR t)/ξ)

This implies an ABSOLUTE speed limit v_LR on information propagation.
The butterfly velocity is always bounded:

    v_B ≤ v_LR

In practice, v_B is often strictly less than v_LR because:
• v_LR is a worst-case bound over ALL operators and ALL initial states
• v_B characterizes the TYPICAL spreading of a SPECIFIC operator
• The butterfly front can have a diffusive broadening ∝ √t that
  makes the effective front slower than the strict LR bound

For the Mixed-Field Ising model with h_x = 1.05, h_z = 0.5, the
nearest-neighbor coupling J=1 sets the energy scale, and we expect
v_B ~ O(1) in units of lattice sites per unit time.

===============================================================================
3. HOW TO EXTRACT v_B: THE THRESHOLD METHOD
===============================================================================

Step 1: Choose a threshold ε.
    The threshold must be:
    • Large enough that numerical noise (C ~ 10^{-12} at early times)
      doesn't trigger false crossings
    • Small enough that it captures the ONSET, not the saturation
    Good choices: ε ∈ [0.01, 0.1] for Tr/D normalization (max C ~ 2)

Step 2: For each probe site j = 1, ..., N-1, find the first time
    t*(j) where C_{0j}(t) ≥ ε.

Step 3: Plot t*(j) vs j and perform a linear fit:
    t*(j) = j / v_B + t_offset

    The slope gives 1/v_B.  The offset t_offset accounts for the
    finite time needed for the operator to develop nonzero support
    even at j = 0 (it starts as a pure σ^z, which already commutes
    with the nearest-neighbor ZZ term).

Step 4: Check robustness by varying ε.  If v_B is stable across
    an order of magnitude in ε, the measurement is reliable.

CAVEATS:
• At N = 10, we only have 9 data points — the fit is indicative,
  not a precision measurement.
• The integrable model may not have a well-defined v_B because the
  operator front is not sharp (it has diffusive broadening and
  interference patterns).
• For true precision, one would use N = 20+ with TEBD/MPS methods.

===============================================================================
4. CHAOTIC vs INTEGRABLE: WHAT TO EXPECT
===============================================================================

CHAOTIC (MFI, h_z = 0.5):
    • Sharp operator front → clean linear t*(j) vs j
    • v_B well-defined and threshold-independent
    • Front broadening is diffusive: width ~ √t (KPZ universality)

INTEGRABLE (TFI, h_z = 0):
    • Operator front is NOT sharp — interference fringes
    • t*(j) vs j may still look roughly linear at short distances
    • v_B is NOT well-defined in the thermodynamic sense
    • Better described by quasiparticle velocities (max group velocity
      of the free-fermion dispersion relation)

===============================================================================
5. CONNECTION TO THE OTOC HEATMAP
===============================================================================

In the heatmap C_{0j}(t) plotted by get_fig_ch3_otoc.py:
    • The BLACK region (C ≈ 0) is the "unscrambled" zone
    • The BRIGHT region (C > 0) is the "scrambled" zone
    • The boundary between them is the OPERATOR LIGHT CONE
    • The SLOPE of this boundary = 1/v_B

The threshold method finds this boundary numerically by drawing a
horizontal line at C = ε through the heatmap and recording where
each row (= site j) first crosses it.

===============================================================================
"""

import numpy as np
import scipy.linalg as la
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# =============================================================================
# ─── Pathing ─────────────────────────────────────────────────────────────────
# =============================================================================
try:
    ROOT = Path(__file__).parent
except NameError:
    ROOT = Path.cwd()

DATA_DIR = ROOT / "data" / "ch3"
FIG_DIR  = ROOT.parent / "figures" / "ch3"
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "text.usetex"    : False,
    "font.family"    : "serif",
    "axes.labelsize" : 10,
    "axes.titlesize" : 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8.5,
})

# =============================================================================
# ─── Parameters ──────────────────────────────────────────────────────────────
# =============================================================================
N_SPIN = 10
THRESHOLDS = [0.01, 0.05, 0.1, 0.5]   # Multiple ε to test robustness

# =============================================================================
# ─── Onset-time extraction ───────────────────────────────────────────────────
# =============================================================================
def extract_onset_times(otoc_map, times, threshold):
    """
    For each probe site j, find the first time t*(j) where C_{0j}(t) ≥ ε.

    Parameters:
        otoc_map  : ndarray (N_times, N_sites), C_{0j}(t) for j = 1..N-1
        times     : 1D array of time values
        threshold : float, the crossing threshold ε

    Returns:
        sites     : 1D array of site indices j where crossing was found
        t_star    : 1D array of onset times t*(j)
    """
    n_sites = otoc_map.shape[1]
    sites = []
    t_star = []

    for j in range(n_sites):
        # Find first index where C_{0,j+1}(t) >= threshold
        crossings = np.where(otoc_map[:, j] >= threshold)[0]
        if len(crossings) > 0:
            idx = crossings[0]
            # Linear interpolation for sub-grid accuracy
            if idx > 0:
                c_prev = otoc_map[idx - 1, j]
                c_curr = otoc_map[idx, j]
                frac = (threshold - c_prev) / (c_curr - c_prev + 1e-30)
                t_cross = times[idx - 1] + frac * (times[idx] - times[idx - 1])
            else:
                t_cross = times[idx]
            sites.append(j + 1)   # site index (1-based)
            t_star.append(t_cross)

    return np.array(sites), np.array(t_star)


# =============================================================================
# ─── Main ────────────────────────────────────────────────────────────────────
# =============================================================================
def main():
    # Load cached heatmap data from get_fig_ch3_otoc.py
    cache_path = DATA_DIR / f"otoc_heatmap_N{N_SPIN}_t15.npz"
    if not cache_path.exists():
        print(f"ERROR: cached data not found at {cache_path}")
        print("Run get_fig_ch3_otoc.py first to generate the heatmap data.")
        return

    data = np.load(cache_path)
    times          = data["times"]
    map_chaotic    = data["map_chaotic"]
    map_integrable = data["map_integrable"]
    print(f"Loaded heatmap data: N={N_SPIN}, {len(times)} time steps, "
          f"{map_chaotic.shape[1]} sites")

    # =========================================================================
    # Extract v_B for multiple thresholds
    # =========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8), constrained_layout=True)

    colors = ["#2166ac", "#4393c3", "#d6604d", "#b2182b"]

    for ax, (model_name, otoc_map) in zip(
            axes, [("Chaotic (MFI)", map_chaotic),
                   ("Integrable (TFI)", map_integrable)]):

        for eps, col in zip(THRESHOLDS, colors):
            sites, t_star = extract_onset_times(otoc_map, times, eps)
            if len(sites) < 2:
                continue

            ax.plot(sites, t_star, "o-", ms=4, lw=1.2, color=col,
                    label=rf"$\varepsilon = {eps}$")

            # Linear fit: t* = j / v_B + offset
            if len(sites) >= 3:
                coeffs = np.polyfit(sites, t_star, 1)
                v_B = 1.0 / coeffs[0]
                j_fit = np.linspace(0, sites[-1] + 0.5, 50)
                ax.plot(j_fit, np.polyval(coeffs, j_fit), "--", color=col,
                        lw=0.8, alpha=0.6)
                print(f"  {model_name}, ε={eps:.2f}: "
                      f"v_B = {v_B:.3f} sites/time  "
                      f"(slope={coeffs[0]:.3f}, offset={coeffs[1]:.3f})")

        ax.set_xlabel(r"Site $j$")
        ax.set_ylabel(r"Onset time $t^*(j)$")
        ax.legend(framealpha=0.85, fontsize=7.5)
        ax.set_xlim(0, N_SPIN)
        ax.set_ylim(0, None)

    axes[0].text(0.04, 0.96, r"$\mathbf{(a)}$  Chaotic (MFI)",
                 transform=axes[0].transAxes, fontsize=10,
                 va="top", ha="left")
    axes[1].text(0.04, 0.96, r"$\mathbf{(b)}$  Integrable (TFI)",
                 transform=axes[1].transAxes, fontsize=10,
                 va="top", ha="left")

    out_path = FIG_DIR / "fig_ch3_otoc_v_butterfly.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), bbox_inches="tight", dpi=300)
    print(f"\nSaved: {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
