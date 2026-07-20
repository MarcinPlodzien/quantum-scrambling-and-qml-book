#!/usr/bin/env python3
"""
get_fig_ch4_syk_dynamics.py
===========================
Running SYK on a quantum computer: how entanglement and magic GROW in time when
the Jordan-Wigner qubit-SYK model is evolved as a circuit from |0..0>.

Produces Fig. (ch4) -- figures/ch4/fig_ch4_syk_dynamics.pdf

    Panel (a): half-cut entanglement entropy S(t).
    Panel (b): stabilizer Renyi entropy (magic) M_2(t).

for q=4 (chaotic) and q=2 (free fermions), each averaged over disorder.

STORY.  The initial register |0..0> is a stabilizer state: entanglement 0 and
magic 0.  Under SYK evolution BOTH rise on the scrambling time scale and saturate
at plateaus that coincide with the mid-spectrum eigenstate values (the companion
static figure get_fig_ch5_syk_qubits.py).  Crucially the q=2 (free-fermion) magic
plateau sits right alongside the q=4 (chaotic) one -- scrambling generates magic
in the free model too, even though that model is classically simulable
(matchgates).  Magic growth alone therefore does not certify hardness.

METHOD.  No dense matrix.  H is a list of Pauli strings (qubit_syk.py); |0..0> is
propagated with a Lanczos matrix-exponential, stepping between successive times so
the state is reused rather than re-evolved from scratch; magic (magic.py) and
half-cut entanglement are read at each step and averaged over disorder.

Cached in data/ch4/fig_ch4_syk_dynamics.npz (FIG_RECOMPUTE=1 to recompute).
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_style import apply_book_style, load_or_compute, panel_label
from qubit_syk import build_terms, lanczos_expm, half_entanglement
from magic import compute_sre

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR.parent / "figures" / "ch4"
DATA_DIR = SCRIPT_DIR / "data" / "ch4"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

L = 10                                   # system size for the dynamics
N_REAL = 8                               # disorder realisations
# fine sampling early (magic saturates by t ~ 1), coarse later to show the plateau
TIMES = np.concatenate([np.arange(0.0, 2.01, 0.1), np.array([2.5, 3, 4, 5, 6, 8, 10])])


def compute():
    out = {"times": TIMES, "L": L}
    for q in (2, 4):
        S = np.zeros((N_REAL, len(TIMES)))
        M = np.zeros((N_REAL, len(TIMES)))
        for r in range(N_REAL):
            terms = build_terms(L, q, np.random.default_rng(1234 + r))
            psi = np.zeros(1 << L, dtype=complex)
            psi[0] = 1.0                          # |0..0>: stabilizer, S=0, M2=0
            t_prev = 0.0
            for it, t in enumerate(TIMES):
                if t > t_prev:
                    psi = lanczos_expm(psi, terms, L, t - t_prev, m=40)  # step, reuse state
                    t_prev = t
                S[r, it] = half_entanglement(psi, L)
                M[r, it] = compute_sre(psi, L)
            print(f"  q={q} realisation {r + 1}/{N_REAL} done")
        out[f"S_q{q}"] = S
        out[f"M_q{q}"] = M
    return out


data = load_or_compute(DATA_DIR / "fig_ch4_syk_dynamics.npz", compute)
ts = data["times"]

apply_book_style()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6), constrained_layout=True)

styles = {
    "q4": dict(color="#c1121f", label="q=4 SYK (chaotic)"),
    "q2": dict(color="#4361ee", label="q=2 SYK (free fermions)"),
}

for key, kw in styles.items():
    S = data[f"S_{key}"]
    ax1.plot(ts, S.mean(0), lw=1.6, **kw)
    ax1.fill_between(ts, S.mean(0) - S.std(0), S.mean(0) + S.std(0), color=kw["color"], alpha=0.15)
ax1.set_xlabel(r"time $t$")
ax1.set_ylabel(r"half-cut entanglement $S(t)$")
panel_label(ax1, "a", loc="lower right")
ax1.legend(loc="upper left")

for key, kw in styles.items():
    M = data[f"M_{key}"]
    ax2.plot(ts, M.mean(0), lw=1.6, **kw)
    ax2.fill_between(ts, M.mean(0) - M.std(0), M.mean(0) + M.std(0), color=kw["color"], alpha=0.15)
ax2.set_xlabel(r"time $t$")
ax2.set_ylabel(r"Stabilizer Rényi Entropy $M_2(t)$")
panel_label(ax2, "b", loc="lower right")
ax2.legend(loc="upper left")

plt.savefig(OUTPUT_DIR / "fig_ch4_syk_dynamics.pdf")
plt.savefig(OUTPUT_DIR / "fig_ch4_syk_dynamics.png")
print(f"  Saved: {OUTPUT_DIR / 'fig_ch4_syk_dynamics.pdf'}")
