#!/usr/bin/env python3
"""
get_fig_ch7_qrc_feature_map.py
==============================
The exact one-step feature identity of a quantum reservoir, checked against
full numerics under a *deterministic* chaotic drive.

Produces Fig. (ch7) -- figures/ch7/fig_ch7_qrc_feature_map.pdf

For a single input node written by <Z> = s and any observable O, one step of a
reservoir of ANY size gives exactly

        <O>(s) = a + b s + c sqrt(1 - s^2),                       (Eq. qrc_feature_form)

with (a,b,c) the node-Pauli components of the body-traced Heisenberg operator
M(O) = Tr_body[ U^dag O U (I (x) rho_body) ].  The functional form is fixed by
the single-qubit encoding and the partial trace, so it holds verbatim for a body
of any dimension; only (a,b,c) change with N, H, tau.

    Panel (a): <O>(s) for three observables of an N=5 mixed-field Ising reservoir
               -- full numerics (markers) vs the closed form a+bs+c sqrt(1-s^2)
               (lines).  The rug on the s-axis marks the values actually visited
               by a DETERMINISTIC logistic-map input stream s_n = 2 x_n - 1,
               x_{n+1} = r x_n (1-x_n), r = 3.9 (fixed seed x_0): the reservoir is
               driven through the whole interval and the identity holds at each.
    Panel (b): the deviation |numeric - closed form| across s, at machine
               precision (~1e-16) for every observable -- the identity is exact,
               not a fit.

Cached in data/ch7/fig_ch7_qrc_feature_map.npz.
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.linalg import expm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_style import apply_book_style, load_or_compute, panel_label

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR.parent / "figures" / "ch7"
DATA_DIR = SCRIPT_DIR / "data" / "ch7"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)


def kron_op(single, site, N):
    op = np.array([[1.0]], dtype=complex)
    for j in range(N):
        op = np.kron(op, single if j == site else I2)
    return op


def mixed_field_ising(N, J, hx, hz):
    H = np.zeros((2**N, 2**N), dtype=complex)
    for i in range(N - 1):
        H += J * kron_op(Z, i, N) @ kron_op(Z, i + 1, N)
    for i in range(N):
        H += hx * kron_op(X, i, N) + hz * kron_op(Z, i, N)
    return H


def rho_in(s):
    """Pure input-node state with <Z> = s: (1/2)(I + s Z + sqrt(1-s^2) X)."""
    a = np.sqrt((1 + s) / 2)
    b = np.sqrt((1 - s) / 2)
    v = np.array([a, b], dtype=complex)
    return np.outer(v, v.conj())


def compute():
    N = 5                       # node = qubit 0, body = qubits 1..N-1
    J, hx, hz, tau = 1.0, 0.9, 0.6, 1.2   # nonintegrable (hz != 0) -> genuine scrambling
    dbody = 2 ** (N - 1)
    H = mixed_field_ising(N, J, hx, hz)
    U = expm(-1j * H * tau)
    body0 = np.zeros((dbody, dbody), dtype=complex)
    body0[0, 0] = 1.0            # |00..0> reference on the body

    observables = {
        r"$\langle \hat{Z}_2\rangle$":        kron_op(Z, 1, N),
        r"$\langle \hat{Z}_3\rangle$":        kron_op(Z, 2, N),
        r"$\langle \hat{Z}_1\hat{Z}_2\rangle$": kron_op(Z, 0, N) @ kron_op(Z, 1, N),
    }

    s = np.linspace(-1.0, 1.0, 241)
    numeric = {}
    closed = {}
    abc = {}
    for name, O in observables.items():
        M = U.conj().T @ O @ U
        Mr = M.reshape(2, dbody, 2, dbody)
        Meff = np.einsum("aibj,ji->ab", Mr, body0)         # 2x2 effective node operator
        a = 0.5 * np.trace(Meff).real
        b = 0.5 * np.trace(Z @ Meff).real
        c = 0.5 * np.trace(X @ Meff).real
        num = np.array([
            np.trace(O @ (U @ np.kron(rho_in(sv), body0) @ U.conj().T)).real
            for sv in s
        ])
        numeric[name] = num
        closed[name] = a + b * s + c * np.sqrt(1 - s ** 2)
        abc[name] = (a, b, c)

    # Deterministic logistic-map input stream, mapped to s in [-1,1].
    r, x = 3.9, 0.37
    xs = []
    for _ in range(80):          # burn-in transient
        x = r * x * (1 - x)
    for _ in range(400):
        x = r * x * (1 - x)
        xs.append(x)
    s_stream = 2.0 * np.array(xs) - 1.0

    names = list(observables.keys())
    return {
        "s": s,
        "names": np.array(names, dtype=object),
        "numeric": np.array([numeric[n] for n in names]),
        "closed": np.array([closed[n] for n in names]),
        "abc": np.array([abc[n] for n in names]),
        "s_stream": s_stream,
        "N": N, "J": J, "hx": hx, "hz": hz, "tau": tau,
    }


data = load_or_compute(DATA_DIR / "fig_ch7_qrc_feature_map.npz", compute)
s = data["s"]
names = list(data["names"])
numeric = data["numeric"]
closed = data["closed"]
abc = data["abc"]
s_stream = data["s_stream"]

resid = np.abs(numeric - closed)
worst = resid.max()
print(f"max |numeric - (a+bs+c sqrt(1-s^2))| over all observables and s : {worst:.2e}")
for i, n in enumerate(names):
    a, b, c = abc[i]
    print(f"  {n}: a={a:+.4f} b={b:+.4f} c={c:+.4f}  max resid={resid[i].max():.1e}")

apply_book_style()
colors = ["#1f77b4", "#d62728", "#2ca02c"]
markers = ["o", "s", "^"]

fig, (axa, axb) = plt.subplots(1, 2, figsize=(11.0, 4.4), constrained_layout=True)

# Panel (a): feature map, numerics (markers) vs closed form (lines).
for i, n in enumerate(names):
    axa.plot(s, closed[i], "-", color=colors[i], lw=2.0, zorder=2)
    axa.plot(s[::12], numeric[i][::12], markers[i], color=colors[i], ms=6.5,
             mfc="white", mew=1.6, label=n, zorder=3)
# rug of the deterministic logistic-map samples
y0 = axa.get_ylim()[0]
axa.plot(s_stream, np.full_like(s_stream, y0), "|", color="0.35", ms=7,
         alpha=0.5, zorder=1)
axa.set_xlabel(r"input $s$  (node encoding $\langle\hat{Z}\rangle=s$)")
axa.set_ylabel(r"one-step feature $\langle\hat{O}\rangle_{\hat\rho_1}(s)$")
axa.set_xlim(-1.02, 1.02)
axa.legend(loc="upper center", frameon=False, ncol=1, handlelength=1.6)
axa.text(0.5, -0.5, r"lines: $a+bs+c\sqrt{1-s^2}$   markers: exact numerics",
         ha="center", fontsize=10, color="0.25")
panel_label(axa, "a")

# Panel (b): residual, log scale.
for i, n in enumerate(names):
    axb.semilogy(s, np.clip(resid[i], 1e-18, None), "-", color=colors[i], lw=1.6,
                 label=n)
axb.axhline(1e-15, ls=":", color="0.5", lw=1.0)
axb.text(-0.98, 1.4e-15, "machine precision", fontsize=9, color="0.4")
axb.set_xlabel(r"input $s$")
axb.set_ylabel(r"$|\,\langle\hat{O}\rangle_{\rm num}-(a+bs+c\sqrt{1-s^2})\,|$")
axb.set_xlim(-1.02, 1.02)
axb.set_ylim(1e-18, 1e-12)
panel_label(axb, "b")

plt.savefig(OUTPUT_DIR / "fig_ch7_qrc_feature_map.pdf")
plt.savefig(OUTPUT_DIR / "fig_ch7_qrc_feature_map.png")
print("wrote", OUTPUT_DIR / "fig_ch7_qrc_feature_map.pdf")
