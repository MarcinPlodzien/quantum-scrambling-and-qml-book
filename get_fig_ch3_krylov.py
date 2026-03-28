#!/usr/bin/env python3
"""
get_fig_ch3_krylov_fast_version.py
===================================
FAST VERSION of the Krylov complexity figure generator for Chapter 3.

This script computes and plots the operator Krylov complexity for the
chaotic Mixed-Field Ising model (Kim & Huse, Phys. Rev. Lett. 111, 2013),
illustrating the Universal Operator Growth Hypothesis:

    "For chaotic systems, the Lanczos coefficients grow linearly: b_n ~ α·n"

This implies C_K(t) ~ sinh²(αt) at early times (before finite-size saturation).

============================================================
PHYSICS BACKGROUND
============================================================
The Krylov complexity measures how quickly a local operator spreads in
operator space under Heisenberg time evolution:

    O(t) = e^{iHt} O(0) e^{-iHt}

The operator Lanczos algorithm recursively builds an orthonormal basis
{O_n} of nested commutators:

    L(O_n) = b_{n+1} O_{n+1} + b_n O_{n-1}

where L(·) = [H, ·] is the Liouvillian (superoperator).

The Krylov complexity is then:
    C_K(t) = Σ_n  n · |φ_n(t)|²
where |φ_n(t)|² is the probability of finding O(t) in the n-th Krylov basis element.

============================================================
KEY OPTIMIZATION: PARITY TRICK (2× SPEEDUP)
============================================================
The Liouvillian L(O) = H·O − O·H requires TWO sparse×dense matrix
multiplications: H·O and O·H. But since our initial operator O_0 = σ_z
is symmetric, the Krylov basis elements alternate symmetry:

    n even: O_n is symmetric       → L(O_n) = X − X^T   (anti-symmetric)
    n odd:  O_n is anti-symmetric  → L(O_n) = X + X^T   (symmetric)

where X = H·O_n. This means we only need ONE sparse×dense matmul per step,
then get L(O_n) via a transpose and add/subtract — a 2× speedup on the
dominant computational cost!

============================================================
BITWISE HAMILTONIAN CONSTRUCTION
============================================================
The Hamiltonian H = Σ Z_i Z_{i+1} − 1.05 Σ X_i + 0.5 Σ Z_i is built
using bitwise operations on computational basis state indices:

    |s⟩ = |s_{N-1} s_{N-2} ... s_0⟩    (binary representation)

    Z_i |s⟩ = (−1)^{s_i} |s⟩           → diagonal: 1−2·bit(s,i)
    X_i |s⟩ = |s ⊕ 2^i⟩                → off-diagonal: flip bit i

This constructs the full D×D sparse Hamiltonian in O(N·D) time,
compared to O(N·D²) for Kronecker products. For N=14 (D=16384),
this is the difference between seconds and hours.

============================================================
MEMORY LAYOUT
============================================================
The Lanczos algorithm stores only O_curr and O_prev (both D×D dense),
plus the sparse Hamiltonian:

    N=12: D=4096,   each D×D matrix = 128 MB,  total ~400 MB
    N=14: D=16384,  each D×D matrix = 2 GB,    total ~6 GB

The sparse Hamiltonian has nnz = D·(N+1) entries, negligible compared
to the dense operator matrices.

============================================================
FILES
============================================================
Caches data to:  data/ch3/krylov_data.npz   (avoids recomputation)
Saves figure to: figures/ch3/fig_ch3_krylov.pdf
"""

import numpy as np
import scipy.sparse as sp
import scipy.linalg as la
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use("Agg")    # non-interactive backend for headless servers
import matplotlib.pyplot as plt
from pathlib import Path

# ─── Directory setup ──────────────────────────────────────────────────────────
ROOT     = Path(__file__).parent
DATA_DIR = ROOT / "data" / "ch3"
FIG_DIR  = ROOT.parent / "figures" / "ch3"
DATA_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ─── Matplotlib defaults for publication figures ─────────────────────────────
plt.rcParams.update({
    "text.usetex"    : False,     # Set True if LaTeX is available
    "font.family"    : "serif",   # Matches LaTeX document font
    "axes.labelsize" : 10,
    "axes.titlesize" : 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8.5,
})

# ─── Simulation parameters ───────────────────────────────────────────────────
N_SPIN    = 12       # Number of spin-1/2 sites. D = 2^N is the Hilbert space dim.
                     # N=12: D=4096 (fast, ~1 min).  N=14: D=16384 (~6 min w/ parity).
K_LANCZOS = 120      # Number of Lanczos steps (Krylov depth).
                     # Should be large enough to see saturation of b_n.
N_TIMES   = 300      # Number of time points for C_K(t).
T_MAX     = 3.5      # Maximum time for C_K(t) evaluation.


# ═════════════════════════════════════════════════════════════════════════════
# SPARSE HAMILTONIAN BUILDER
# ═════════════════════════════════════════════════════════════════════════════

def build_mixed_field_hamiltonian(N):
    """
    Build the Mixed-Field Ising Hamiltonian as a sparse CSR matrix.

    H = Σ_{i=0}^{N-2} Z_i Z_{i+1}  −  1.05 Σ_{i=0}^{N-1} X_i  +  0.5 Σ_{i=0}^{N-1} Z_i

    Parameters from Kim & Huse (2013): the transverse field h_x = 1.05
    and longitudinal field h_z = 0.5 break integrability, making the
    model chaotic (no conserved quantities beyond energy).

    CONSTRUCTION STRATEGY (bitwise, O(N·D) complexity):
    ────────────────────────────────────────────────────
    Each computational basis state |s⟩ is labeled by an integer s ∈ {0,...,D−1}.
    The binary digits of s give the spin configuration: bit i of s = spin at site i.

    • Z_i Z_{i+1} is DIAGONAL: contributes +1 if spins i,i+1 are aligned, −1 if not.
      Formula: diag[s] += 1 − 2·(bit_i ⊕ bit_{i+1})

    • Z_i is DIAGONAL: contributes ±1 depending on spin i.
      Formula: diag[s] += 0.5·(1 − 2·bit_i)

    • X_i is OFF-DIAGONAL: flips spin i, connecting |s⟩ ↔ |s ⊕ 2^i⟩.
      Formula: H[s, s ⊕ 2^i] = −1.05

    The matrix has exactly nnz = D·(N+1) non-zero entries:
    D diagonal entries + D·N off-diagonal entries (one per spin flip per state).
    """
    D = 2**N
    # Integer labels for all 2^N basis states
    s = np.arange(D, dtype=np.int32)

    # ── DIAGONAL PART: ZZ coupling + Z field ─────────────────────────────
    diag = np.zeros(D, dtype=np.float64)

    # ZZ coupling: Σ Z_i Z_{i+1}
    # bit_i XOR bit_{i+1} = 0 if aligned (→ +1), 1 if anti-aligned (→ −1)
    for i in range(N - 1):
        bi = (s >> (N - 1 - i)) & 1           # extract bit i from state s
        bj = (s >> (N - 1 - (i + 1))) & 1     # extract bit i+1
        diag += 1.0 - 2.0 * (bi ^ bj)         # +1 if aligned, −1 if not

    # Z field: 0.5 Σ Z_i
    # bit_i = 0 → spin up → Z eigenvalue +1; bit_i = 1 → spin down → −1
    for i in range(N):
        bi = (s >> (N - 1 - i)) & 1
        diag += 0.5 * (1.0 - 2.0 * bi)

    # ── PRE-ALLOCATE SPARSE MATRIX ARRAYS ────────────────────────────────
    # Total non-zeros: D diagonal + N·D off-diagonal = D·(N+1)
    nnz = D * (N + 1)
    rows = np.empty(nnz, dtype=np.int32)
    cols = np.empty(nnz, dtype=np.int32)
    vals = np.empty(nnz, dtype=np.float64)

    # Fill diagonal block
    rows[:D] = s;  cols[:D] = s;  vals[:D] = diag

    # ── OFF-DIAGONAL: X field (single bit flips) ─────────────────────────
    # X_i flips bit i: maps |s⟩ → |s ⊕ 2^{N−1−i}⟩
    for i in range(N):
        sl = slice(D * (i + 1), D * (i + 2))   # slice for this spin's block
        rows[sl] = s                             # row = original state
        cols[sl] = s ^ (1 << (N - 1 - i))       # col = bit-flipped state
        vals[sl] = -1.05                         # transverse field strength

    # Assemble into efficient CSR format for fast matrix-vector products
    return sp.csr_matrix((vals, (rows, cols)), shape=(D, D), dtype=np.float64)


# ═════════════════════════════════════════════════════════════════════════════
# OPERATOR LANCZOS ALGORITHM (WITH PARITY OPTIMIZATION)
# ═════════════════════════════════════════════════════════════════════════════

def lanczos_operator(H_sparse, O0, K):
    """
    Operator Lanczos algorithm with the PARITY TRICK.

    The standard operator Lanczos computes:
        L(O) = [H, O] = H·O − O·H

    This requires TWO sparse×dense matrix multiplications per step.
    Each such multiplication dominates the runtime at O(nnz × D).

    ──── THE PARITY TRICK ────
    Key insight: if O_0 is symmetric (like σ_z), then the Krylov basis
    elements ALTERNATE between symmetric and anti-symmetric matrices:

        O_0 = symmetric  →  L(O_0) = anti-symmetric  →  O_1 ∝ anti-symmetric
        O_1 = anti-symmetric → L(O_1) = symmetric    →  O_2 ∝ symmetric
        ...

    This is because [H, S] is anti-symmetric when S is symmetric and H is
    symmetric, and [H, A] is symmetric when A is anti-symmetric.

    For a symmetric O:   H·O − O·H = X − X^T   where X = H·O
    For anti-symmetric O: H·O − O·H = X + X^T   where X = H·O

    In both cases, we only need ONE matmul (X = H·O), then get the
    commutator via a simple transpose and add/subtract.

    This gives an exact 2× speedup with no approximation!

    ──── MEMORY ────
    We use the 3-term recurrence relation to keep only O_curr and O_prev
    in memory (each is a D×D dense matrix):
        L(O_n) = b_{n+1} O_{n+1} + b_n O_{n-1}

    For N=14: each D×D = 16384² × 8 bytes = 2 GB, so total ~6 GB.

    ──── INNER PRODUCT ────
    The Hilbert-Schmidt inner product for operators is:
        ⟨A, B⟩_HS = Tr(A† B) / D

    For real matrices: ‖A‖²_HS = Tr(A^T A)/D = Σ_{ij} A²_{ij} / D
    We compute this as np.linalg.norm(A)² / D = ‖A‖²_F / D.
    """
    D = H_sparse.shape[0]
    sqrt_D = np.sqrt(D)

    # Normalize the initial operator in the Hilbert-Schmidt norm
    # ‖O‖_HS = ‖O‖_Frobenius / √D
    norm0 = np.linalg.norm(O0) / sqrt_D
    O_curr = O0 / norm0
    O_prev = np.zeros_like(O_curr)

    b = []      # List of Lanczos coefficients b_1, b_2, ...
    for n in range(K):
        # Step 1: Single sparse×dense matmul (THE EXPENSIVE STEP)
        # X = H · O_curr, cost: O(nnz × D)
        X = H_sparse.dot(O_curr)

        # Step 2: Compute L(O_curr) = [H, O_curr] using the parity trick
        # n even → O_curr symmetric     → L = X − X^T (anti-symmetric result)
        # n odd  → O_curr anti-symmetric → L = X + X^T (symmetric result)
        A = (X - X.T) if n % 2 == 0 else (X + X.T)

        # Step 3: Orthogonalize against the previous basis element
        # From the 3-term recurrence: A = b_{n+1}·O_{n+1} + b_n·O_{n-1}
        # We subtract the O_{n-1} component:
        if n > 0:
            A -= b[-1] * O_prev

        # Step 4: Compute the Lanczos coefficient b_{n+1} = ‖A‖_HS
        b_n = np.linalg.norm(A) / sqrt_D
        b.append(b_n)

        if n % 20 == 0:
            print(f"    step {n:3d}/{K}  b_n = {b_n:.6f}", flush=True)
        if b_n < 1e-14:
            print(f"  Krylov space exhausted at n={n}")
            break

        # Step 5: Advance the 3-term recurrence
        O_prev = O_curr
        A /= b_n          # In-place division to avoid extra allocation
        O_curr = A         # O_{n+1} = A / b_{n+1}

    return np.array(b)


# ═════════════════════════════════════════════════════════════════════════════
# KRYLOV COMPLEXITY FROM LANCZOS COEFFICIENTS
# ═════════════════════════════════════════════════════════════════════════════

def krylov_complexity_from_b(b, times):
    """
    Compute C_K(t) from the Lanczos coefficients {b_n}.

    The time evolution in Krylov space is governed by the K×K tridiagonal
    "Krylov Hamiltonian":

        H_K = -i · M,    where M_{n,n+1} = b_n,  M_{n+1,n} = -b_n

    The Krylov complexity is:
        C_K(t) = Σ_n  n · |⟨n| e^{-iH_K t} |0⟩|²

    This is computed via exact diagonalization of H_K (a K×K matrix,
    much smaller than the D×D operator space).

    VECTORIZED IMPLEMENTATION:
    Instead of looping over time points, we compute:
        exp_t[t, k] = e^{i·λ_k·t}           (phase factors for all t, k)
        φ_t[t, n] = Σ_k exp_t[t,k] · c0[k] · V[n,k]   (amplitudes)
        C_K[t] = Σ_n n · |φ_t[t,n]|²

    This is O(K² · N_times) via matrix products, much faster than
    a Python loop over N_times.
    """
    K = len(b)

    # Build the K×K anti-symmetric tridiagonal matrix
    # M_{n,n+1} = +b_n, M_{n+1,n} = -b_n  (anti-Hermitian structure)
    M = np.diag(b[:-1], 1) - np.diag(b[:-1], -1)

    # The "Krylov Hamiltonian" H_K = -iM is Hermitian
    H_k = -1j * M
    evals, evecs = la.eigh(H_k)     # Real eigenvalues, unitary eigenvectors

    # Initial state: all weight on the 0-th Krylov basis element
    c0 = evecs[0, :].conj()           # ⟨0|V⟩ in the eigenbasis
    n_arr = np.arange(K, dtype=float)  # Position operator: n = 0, 1, 2, ...

    # Vectorized time evolution
    #   exp_t shape: (N_times, K) — phase factors e^{iλ_k t} for each (t, k)
    #   phi_t shape: (N_times, K) — amplitudes φ_n(t) in Krylov basis
    exp_t = np.exp(1j * np.outer(times, evals))    # (N_times, K)
    phi_t = (exp_t * c0) @ evecs.T                 # (N_times, K)

    # C_K(t) = Σ_n n · |φ_n(t)|²
    return np.sum(np.abs(phi_t)**2 * n_arr, axis=1)


# ═════════════════════════════════════════════════════════════════════════════
# MAIN: COMPUTE + PLOT
# ═════════════════════════════════════════════════════════════════════════════

def main():
    cache_path = DATA_DIR / "krylov_data.npz"

    # ── Load or compute ───────────────────────────────────────────────────
    if cache_path.exists():
        print(f"Loading cached data from {cache_path}")
        data = np.load(cache_path)
        b_cha = data["b_cha"]
        times = data["times"]
        CK_cha = data["CK_cha"]
        # Check if cache matches current N_SPIN setting
        N_cached = int(data["N_spin"]) if "N_spin" in data else -1
        if N_cached != N_SPIN:
            print(f"  Cache N={N_cached} != {N_SPIN}. Re-running.")
            cache_path.unlink()
            return main()    # Recursive call after deleting stale cache
    else:
        D = 2**N_SPIN
        print(f"N={N_SPIN}, D={D}, memory per D×D matrix: {D**2*8/1e6:.0f} MB")

        # Initial operator: σ_z at the centre site (a simple local operator)
        # σ_z is diagonal: eigenvalue +1 for spin-up, −1 for spin-down
        site = N_SPIN // 2
        s = np.arange(D, dtype=np.int32)
        diag = (1.0 - 2.0 * ((s >> (N_SPIN - 1 - site)) & 1)).astype(np.float64)
        O_init = np.diag(diag)     # D×D dense diagonal matrix

        # Build the sparse Hamiltonian
        print(f"\nBuilding Mixed-Field Ising Hamiltonian ...")
        H = build_mixed_field_hamiltonian(N_SPIN)
        print(f"  nnz: {H.nnz:,}")

        # Run the operator Lanczos with the parity trick
        print(f"\nRunning operator Lanczos (K={K_LANCZOS}) ...")
        b_cha = lanczos_operator(H, O_init, K_LANCZOS)
        del H, O_init     # Free ~2-4 GB of memory immediately

        # Compute C_K(t) from the Lanczos coefficients
        times = np.linspace(0, T_MAX, N_TIMES)
        print(f"\nComputing C_K(t) ...")
        CK_cha = krylov_complexity_from_b(b_cha, times)

        # Cache for fast figure iteration
        np.savez(cache_path, b_cha=b_cha, times=times, CK_cha=CK_cha,
                 N_spin=N_SPIN)
        print(f"Saved: {cache_path}")

    # ── FIT 1: b_n = α·n + γ ─────────────────────────────────────────────
    # We fit a linear function with intercept to the Lanczos coefficients
    # in the range n ∈ [8, 30] where b_n is cleanly linear.
    # The early b_n (n < 5) are affected by the specific initial operator
    # structure and don't follow the universal linear growth.
    n_arr = np.arange(1, len(b_cha) + 1)
    i_fit = (n_arr >= 8) & (n_arr <= 30)
    coeffs = np.polyfit(n_arr[i_fit].astype(float), b_cha[i_fit], 1)
    alpha_bn, gamma_bn = float(coeffs[0]), float(coeffs[1])
    print(f"\n  b_n fit: α = {alpha_bn:.4f}, γ = {gamma_bn:.4f}")

    # ── FIT 2: A·sinh²(αt) to C_K(t) ─────────────────────────────────────
    # The analytical prediction C_K(t) = sinh²(αt) holds in the
    # thermodynamic limit with pure b_n = α·n. At finite N, we fit
    # a 2-parameter model A·sinh²(αt) to absorb the prefactor mismatch
    # from the non-ideal early b_n.
    CK_max = np.max(CK_cha)
    fit_mask = (times > 0.3) & (CK_cha > 0.01) & (CK_cha < 0.4 * CK_max)

    def model(t, A, a):
        return A * np.sinh(a * t)**2

    popt, _ = curve_fit(model, times[fit_mask], CK_cha[fit_mask],
                        p0=[1.0, alpha_bn], bounds=([0.001, 0.01], [100, 10.0]))
    A_fit, alpha_ck = float(popt[0]), float(popt[1])
    print(f"  C_K(t) fit: α = {alpha_ck:.4f}, A = {A_fit:.4f}")

    # ─────────────────────────────────────────────────────────────────────────
    # PLOT — clean minimal style (all annotations in caption)
    # ─────────────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.3), constrained_layout=True)

    # ── Panel (a): Lanczos coefficients b_n ───────────────────────────────
    # Shows the hallmark of chaos: b_n grows linearly with n.
    # The first few points (n < 5) sit below the fit — a finite-size effect
    # where the initial operator structure dominates over the universal trend.
    ax = axes[0]
    n_show = 30     # Truncate to focus on the linear growth regime
    n_cha = np.arange(1, n_show + 1)

    # Fit line (black dashed, no legend — described in caption)
    n_line = np.linspace(0, n_show, 100)
    fit_line = np.maximum(alpha_bn * n_line + gamma_bn, 0)
    ax.plot(n_line, fit_line, "--k", lw=2.2, alpha=0.7, zorder=1)

    # Data points
    ax.plot(n_cha, b_cha[:n_show], "s", ms=3.0, color="#d7191c",
            alpha=0.85, zorder=2, label="Chaotic spin chain")

    ax.text(0.04, 0.96, r"$\mathbf{(a)}$", transform=ax.transAxes,
            fontsize=11, va="top", ha="left")
    ax.set_xlabel(r"Krylov index $n$")
    ax.set_ylabel(r"$b_n$")
    ax.set_xlim(0, n_show + 1)
    ax.set_ylim(0, None)
    ax.legend(framealpha=0.85, loc="lower right", fontsize=7.5)

    # ── Panel (b): Krylov complexity C_K(t) on log-log ────────────────────
    # On log-log axes, sinh²(αt) ~ e^{2αt}/4 appears as an upward-curving
    # line (exponential growth). The clean overlap between numerical and
    # analytical curves validates the operator growth hypothesis.
    ax = axes[1]

    # Numerical data (skip t=0 to avoid log(0))
    ax.loglog(times[1:], np.maximum(CK_cha[1:], 1e-10),
              color="#d7191c", lw=1.8, zorder=2, label="Numerical")

    # Analytical prediction
    t_anal = np.linspace(0.01, T_MAX, 500)
    CK_anal = A_fit * np.sinh(alpha_ck * t_anal)**2
    ax.loglog(t_anal, np.maximum(CK_anal, 1e-10),
              "--k", lw=2.2, alpha=0.7, zorder=3,
              label=rf"$\sim\!\sinh^2(\alpha t)$, $\alpha\!\approx\!{alpha_ck:.2f}$")

    ax.text(0.04, 0.96, r"$\mathbf{(b)}$", transform=ax.transAxes,
            fontsize=11, va="top", ha="left")
    ax.set_xlabel(r"Time $t$")
    ax.set_ylabel(r"$\mathcal{C}_K(t)$")
    ax.legend(framealpha=0.85, loc="lower right", fontsize=7.5)
    ax.set_ylim(1e-4, None)

    # ── Save ──────────────────────────────────────────────────────────────
    out_path = FIG_DIR / "fig_ch3_krylov.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(out_path.with_suffix('.png'), bbox_inches="tight", dpi=300)
    print(f"\nSaved: {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
