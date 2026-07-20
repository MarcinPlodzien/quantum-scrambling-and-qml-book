#!/usr/bin/env python3
"""
get_fig_ch7_qsbm.py
===================
Quantum scrambling Born machine (QSBM) with an ideal Haar scrambler: a fixed
Haar-random unitary (reused every layer) supplies all the entanglement, only
single-qubit rotations are trained.  Per layer (on |0>^{N_Q}): pre-rotations
Rx,Rz -> fixed Haar U_S -> post-rotation Ry, on all N_Q qubits.  N_A=2 ancillas
are traced out, so the measured register is n = N_Q - N_A = 8 qubits (256 bins);
the mixed reduced state gives smooth output distributions.  Target: 5-peak
Gaussian mixture.  Adam on the exact NLL; KLD reported from N_shots=5000 samples
(20 realizations, mean +/- 1 sigma).

Produces  figures/ch7/fig_ch7_qsbm.pdf

    Top:    converged KLD vs number of trainable layers L (L = 1..6).
    Bottom: target (filled) vs learned distribution (mean, +/-1 sigma band) at
            L = 1, 2, 4.

THE POINT.  More trainable single-qubit layers steadily lower the KLD toward the
5000-shot floor even though no entangling gate is trained: with the entanglement
supplied for free by the fixed scrambler, the whole parameter budget goes to the
single-qubit readout, and a handful of layers already reproduces the target.

Cached in data/ch7/fig_ch7_qsbm.npz (FIG_RECOMPUTE=1 to recompute).
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_style import apply_book_style, load_or_compute, panel_label

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR.parent / "figures" / "ch7"
DATA_DIR = SCRIPT_DIR / "data" / "ch7"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

NQ, NA = 10, 2                       # 8 measured qubits -> 256 bins; 5000-shot floor ~0.026
NSYS = NQ - NA
NSEED, STEPS, LR = 20, 350, 0.02
LS = [1, 2, 3, 4, 5, 6, 8, 10]
PANEL_LS = [1, 2, 4, 6]


def compute():
    import jax, jax.numpy as jnp
    from jax import lax
    cdt = jnp.complex64
    I2 = jnp.eye(2, dtype=cdt)
    X = jnp.array([[0, 1], [1, 0]], cdt); Y = jnp.array([[0, -1j], [1j, 0]], cdt); Z = jnp.array([[1, 0], [0, -1]], cdt)
    rx = lambda t: jnp.cos(t / 2).astype(cdt) * I2 - 1j * jnp.sin(t / 2).astype(cdt) * X
    ry = lambda t: jnp.cos(t / 2).astype(cdt) * I2 - 1j * jnp.sin(t / 2).astype(cdt) * Y
    rz = lambda t: jnp.cos(t / 2).astype(cdt) * I2 - 1j * jnp.sin(t / 2).astype(cdt) * Z
    ap1 = lambda p, U, j: jnp.moveaxis(jnp.tensordot(U, p, axes=([1], [j])), 0, j)

    def haar(dim, r):
        z = (r.standard_normal((dim, dim)) + 1j * r.standard_normal((dim, dim))) / np.sqrt(2)
        q, rr = np.linalg.qr(z); ph = np.diagonal(rr); return (q * (ph / np.abs(ph))).astype(np.complex64)

    x = np.arange(2 ** NSYS); npr = np.random.RandomState(123); tgt = np.zeros(2 ** NSYS)
    for i in range(5):
        c = int((i + 0.5) * 2 ** NSYS / 5); s = 2 ** NSYS / 20; h = npr.rand() + 0.5
        tgt += h * np.exp(-((x - c) ** 2) / (2 * s ** 2))
    tgt /= tgt.sum(); q = jnp.array(tgt.astype(np.float32)); P0 = jnp.zeros((2 ** NQ,), cdt).at[0].set(1.0)

    def make_fns(Lv):
        def dist(theta, US):
            psi = P0
            for l in range(Lv):
                t = psi.reshape((2,) * NQ)
                for j in range(NQ): t = ap1(t, rx(theta[l, j, 0]), j); t = ap1(t, rz(theta[l, j, 1]), j)
                psi = US @ t.ravel(); t = psi.reshape((2,) * NQ)
                for j in range(NQ): t = ap1(t, ry(theta[l, j, 2]), j)
                psi = t.ravel()
            return (jnp.abs(psi) ** 2).reshape(2 ** NSYS, 2 ** NA).sum(axis=1)
        nll = lambda th, US: -jnp.sum(q * jnp.log(jnp.clip(dist(th, US), 1e-10, None)))
        gnll = jax.grad(nll)
        def train(th0, US):
            b1, b2, eps = 0.9, 0.999, 1e-8
            def body(c, _):
                th, m, v, t = c; g = gnll(th, US); t = t + 1.0
                m = b1 * m + (1 - b1) * g; v = b2 * v + (1 - b2) * g * g
                return (th - LR * (m / (1 - b1 ** t)) / (jnp.sqrt(v / (1 - b2 ** t)) + eps), m, v, t), None
            z = jnp.zeros_like(th0)
            (th, *_), _ = lax.scan(body, (th0, z, z, 0.0), None, length=STEPS)
            return th
        return jax.jit(jax.vmap(train, (0, 0))), jax.jit(jax.vmap(dist, (0, 0)))

    rng_shot = np.random.default_rng(2024)
    def klds(P):                               # sampled KLD, 5000 shots, add-1 smoothing
        out = []
        for p in np.clip(np.asarray(P), 0, None):
            p = p / p.sum()
            phat = (rng_shot.multinomial(5000, p) + 1.0) / (5000 + p.size)
            out.append(float(np.sum(tgt * np.log(np.clip(tgt, 1e-12, None) / phat))))
        return np.array(out)

    US = jnp.stack([jnp.array(haar(2 ** NQ, np.random.default_rng(200 + s))) for s in range(NSEED)])
    rng = np.random.default_rng(7)
    km, ks, panels = [], [], {}
    for Lv in LS:
        tv, dv = make_fns(Lv)
        th0 = jnp.array(rng.uniform(0, 2 * np.pi, (NSEED, Lv, NQ, 3)).astype(np.float32))
        th = tv(th0, US); P = np.array(dv(th, US)); kl = klds(P)
        km.append(float(kl.mean())); ks.append(float(kl.std()))
        if Lv in PANEL_LS: panels[Lv] = (P.mean(0), P.std(0), float(kl.mean()))
        print(f"Haar L={Lv}  KLD={kl.mean():.4f} +/- {kl.std():.4f}", flush=True)

    pl = np.array(PANEL_LS)
    return {"Ls": np.array(LS), "km": np.array(km), "ks": np.array(ks),
            "x": x, "target": tgt, "panel_Ls": pl,
            "panel_mean": np.array([panels[L][0] for L in pl]),
            "panel_std": np.array([panels[L][1] for L in pl]),
            "panel_kl": np.array([panels[L][2] for L in pl])}


data = load_or_compute(DATA_DIR / "fig_ch7_qsbm.npz", compute)

apply_book_style()
x = data["x"]
fig = plt.figure(figsize=(11, 5.0))
gs = fig.add_gridspec(2, len(PANEL_LS), height_ratios=[1.15, 1.0], hspace=0.55, wspace=0.28)

ax = fig.add_subplot(gs[0, :])
Ls, m, s = data["Ls"], data["km"], data["ks"]
ax.fill_between(Ls, m - s, m + s, color="#1f6f8b", alpha=0.20)
ax.plot(Ls, m, "o-", color="#1f6f8b", lw=2.0, ms=7)
ax.set_yscale("log"); ax.set_xlabel("trainable layers $L$")
ax.set_ylabel(r"$D_{\mathrm{KL}}(q\,\|\,p_\theta)$")
panel_label(ax, "a", loc="upper right")

for idx, Lp in enumerate(list(data["panel_Ls"])):
    axp = fig.add_subplot(gs[1, idx])
    axp.fill_between(x, data["target"], color="0.78", lw=0, zorder=1, label="target")
    pm, ps = data["panel_mean"][idx], data["panel_std"][idx]
    axp.fill_between(x, pm - ps, pm + ps, color="#2166ac", alpha=0.30, lw=0, zorder=2)
    axp.plot(x, pm, color="#2166ac", lw=1.4, zorder=3, label="QSBM")
    axp.set_title(rf"$L={int(Lp)}$,  $D_{{\mathrm{{KL}}}}={float(data['panel_kl'][idx]):.3f}$", fontsize=11)
    axp.set_xlabel("$x$"); axp.set_yticks([])
    panel_label(axp, "bcde"[idx], loc="upper left")
    if idx == 0: axp.legend(frameon=False, loc="upper right", fontsize=9)

fig.savefig(OUTPUT_DIR / "fig_ch7_qsbm.pdf")
fig.savefig(OUTPUT_DIR / "fig_ch7_qsbm.png")
print(f"  Saved: {OUTPUT_DIR / 'fig_ch7_qsbm.pdf'}")
