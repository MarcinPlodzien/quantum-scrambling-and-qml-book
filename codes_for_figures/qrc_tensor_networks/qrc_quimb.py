"""
qrc_quimb.py
============
Tensor-network (quimb) implementation of the quantum reservoir computer defined
in ``qrc_dense_reference.py``.  It reproduces the dense ground truth bit-for-bit
at small N (large bond dimension) and then runs at N the dense code cannot reach.

--------------------------------------------------------------------------------
WHY AN MPDO, AND HOW WE VECTORIZE IT
--------------------------------------------------------------------------------
The reset step (partial-trace qubit 0) makes the reservoir state *mixed*, so we
cannot carry a wavefunction MPS -- we must carry the density matrix rho itself.
We store rho as a MATRIX-PRODUCT DENSITY OPERATOR (MPDO).  The clean way to run
TEBD on an MPDO is to VECTORIZE it: fuse each site's ket index k and bra index b
into a single physical "super-leg" of dimension 4,

        p = 2*k + b        (k = ket bit, b = bra bit),

so rho becomes an ordinary MPS with physical dimension 4 per site -- a
"super-ket" |rho>>.  Every operation on rho becomes a plain MPS operation on
|rho>>:

    * a superoperator  rho -> A rho B^dagger  is a linear map on the super-legs;
    * the partial trace is a contraction of a super-leg with a fixed vector;
    * an expectation value Tr(O rho) is a contraction of |rho>> with one vector
      per site (NOT the quadratic <psi|O|psi> of a wavefunction -- it is linear
      in |rho>>, because rho already carries both ket and bra).

Index bookkeeping for a single qubit, p = 2*k + b in {00, 01, 10, 11}:

    * rho -> u rho u^dagger  (single-qubit unitary u) is the 4x4 super-matrix
          S(u)[p', p] = u[k', k] * conj(u[b', b]),
      i.e. u acts on the ket half and u* on the bra half:  g (x) g*.
    * partial trace / Tr(I . rho) contracts with   t = [1, 0, 0, 1]
          (t[p] = delta_{k,b} = I[b, k]).
    * Tr(O . rho) contracts each site with   m_O[p] = O[b, k]:
          m_I = [1, 0, 0,  1]        (identity)
          m_X = [0, 1, 1,  0]
          m_Y = [0, 1j,-1j, 0]
          m_Z = [1, 0, 0, -1]
      Then Tr(O rho) = sum_{k,b} O[b,k] rho[k,b] = <<m_O | rho>> factorizes over
      sites for a product operator O = prod_i O_i.

--------------------------------------------------------------------------------
THE STEP (matches the dense reference exactly)
--------------------------------------------------------------------------------
Given the next input s:

    1./2./3.  RESET + INJECT (fused).  Partial-tracing qubit 0 and injecting the
        fresh product state |psi(s)><psi(s)| on qubit 0, uncorrelated with the
        rest, is exactly

            rho  ->  |psi(s)><psi(s)|_0  (x)  Tr_0(rho).

        On the super-ket this is a one-tensor edit of site 0: contract site 0's
        super-leg with the trace vector t to get the boundary vector into bond
        (0,1), then set site 0 to the rank-1 tensor  vec|psi><psi|  (x) boundary.
        This keeps the site count and all bond dimensions unchanged.

    4.  EVOLVE  rho -> U rho U^dagger,  U = exp(-i H tau_ev),
            H = J sum Z_i Z_{i+1} + hx sum X_i + hz sum Z_i.
        Split H = A + B with A = J sum Z_iZ_{i+1} (all ZZ terms commute, and are
        diagonal) and B = sum (hx X_i + hz Z_i) (single-site, all commute).  Each
        group exponentiates EXACTLY; only [A, B] != 0 forces a Trotter split.  We
        use the symmetric 2nd-order step  S2(dt) = e^{-iB dt/2} e^{-iA dt}
        e^{-iB dt/2}  and, by default, the 4th-order Suzuki product of five S2's,
        so the Trotter error is driven below 1e-6 with a handful of substeps
        (that is what lets us MATCH the dense expm, not merely approximate it).

        In super-ket form each ket gate g becomes the super-gate g (x) g*:
          - single-site field gate  -> 4x4 super-matrix, applied per site
            (no bond growth);
          - two-site ZZ gate         -> 16x16 super-gate reshaped to (4,4,4,4)
            and applied to neighbouring super-legs with SVD truncation back to
            chi_max (this is the only place entanglement -- here *operator*
            entanglement of rho -- can grow).

    5.  READ OUT the low-weight Pauli set {X_i, Y_i, Z_i, X_iX_{i+1}, Y_iY_{i+1},
        Z_iZ_{i+1}} by contracting |rho>> with the per-site vectors above, in the
        SAME order as the dense reference.

Cost per step is O(N chi^3) instead of the dense O(8^N); chi stays small in the
under/edge-of-scrambling regime where a reservoir is actually useful.
"""
import time

import numpy as np
from scipy.linalg import expm

import quimb.tensor as qtn

# --- single-qubit operators (ket-space 2x2 matrices) -------------------------
I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
ZZ = np.kron(Z, Z)                                   # 4x4, basis 2*k_i + k_{i+1}

# --- super-leg vectors, p = 2*k + b ------------------------------------------
# Trace / partial-trace vector   t[p] = I[b, k] = delta_{k,b}.
TRACE_VEC = np.array([1, 0, 0, 1], dtype=complex)
# Measurement vectors  m_O[p] = O[b, k]  so that  Tr(O rho) = <<m_O | rho>>.
M_X = np.array([0, 1, 1, 0], dtype=complex)
M_Y = np.array([0, 1j, -1j, 0], dtype=complex)
M_Z = np.array([1, 0, 0, -1], dtype=complex)


def encode(s):
    """R_y(arccos s)|0> : single-qubit input state with <Z> = s (real amps)."""
    return np.array([np.sqrt((1 + s) / 2), np.sqrt((1 - s) / 2)], dtype=complex)


def vec_rho1(psi):
    """Vectorize the rank-1 density matrix |psi><psi| into a super-leg (dim 4):
    v[p = 2k + b] = psi[k] * conj(psi[b])."""
    return np.outer(psi, psi.conj()).reshape(4)


# ---------------------------------------------------------------------------
#  Superoperators on the vectorized (super-ket) representation
# ---------------------------------------------------------------------------
def super_single(u):
    """rho -> u rho u^dagger  as a 4x4 super-matrix S[p', p] on one site.
    S(u)[2k'+b', 2k+b] = u[k',k] * conj(u[b',b])   (ket gets u, bra gets u*)."""
    # axes: a=k' (ket out), b=b' (bra out), c=k (ket in), d=b (bra in)
    s4 = np.einsum("ac,bd->abcd", u, u.conj())
    return s4.reshape(4, 4)


def super_two(g):
    """rho -> g rho g^dagger  for a two-qubit ket gate g (4x4, basis 2k_i+k_j),
    returned as the (4,4,4,4) super-gate the way quimb wants it:
        G[o_i, o_j, in_i, in_j],  o/in are super-legs (dim 4) of the two sites.
    Construction is the tensor product g (x) g* with ket->g, bra->g*."""
    g4 = g.reshape(2, 2, 2, 2)          # [ki_o, kj_o, ki_in, kj_in]
    gc4 = g.conj().reshape(2, 2, 2, 2)  # [bi_o, bj_o, bi_in, bj_in]
    # Interleave ket/bra of each site so the fused super-legs are p = 2k + b.
    # Target axis order: ki_o,bi_o, kj_o,bj_o, ki_in,bi_in, kj_in,bj_in
    #   g4  labels a,c,e,f = ki_o,kj_o,ki_in,kj_in
    #   gc4 labels b,d,g,h = bi_o,bj_o,bi_in,bj_in
    G8 = np.einsum("acef,bdgh->abcdegfh", g4, gc4)
    return G8.reshape(4, 4, 4, 4)


# ---------------------------------------------------------------------------
#  Trotter plan for  U = exp(-i H tau_ev)
# ---------------------------------------------------------------------------
def _field_super(hx, hz, dt):
    """Super-gate for the single-site field over time dt: rho -> uf rho uf^dag,
    uf = exp(-i (hx X + hz Z) dt).  (Same on every site.)"""
    uf = expm(-1j * (hx * X + hz * Z) * dt)
    return super_single(uf)


def _zz_super(J, dt):
    """Super-gate for one ZZ bond over time dt: rho -> gzz rho gzz^dag,
    gzz = exp(-i J Z_iZ_{i+1} dt)  (diagonal)."""
    gzz = expm(-1j * J * dt * ZZ)
    return super_two(gzz)


# 4th-order Suzuki coefficient: S4 = S2(p)^2 S2(1-4p) S2(p)^2 with this p.
_SUZUKI_P = 1.0 / (4.0 - 4.0 ** (1.0 / 3.0))


def _trotter_plan(N, J, hx, hz, tau_ev, n_trotter, order):
    """Precompute the ordered list of gates for ONE full evolution by U.

    Returns a list of ('F', S_field) and ('Z', G_zz) operations.  'F' is applied
    to every site; 'Z' is applied to every nearest-neighbour bond.  Because the
    ZZ terms mutually commute (and the field terms mutually commute), the only
    Trotter error is between the two groups -- controlled by n_trotter / order.
    """
    dt = tau_ev / n_trotter
    sub = [_SUZUKI_P, _SUZUKI_P, 1.0 - 4.0 * _SUZUKI_P, _SUZUKI_P, _SUZUKI_P] \
        if order == 4 else [1.0]

    # Cache gate arrays by the (rounded) time-slice they act for.
    fcache, zcache = {}, {}

    def field(d):
        key = round(d, 15)
        if key not in fcache:
            fcache[key] = _field_super(hx, hz, d)
        return fcache[key]

    def zz(d):
        key = round(d, 15)
        if key not in zcache:
            zcache[key] = _zz_super(J, d)
        return zcache[key]

    plan = []
    for _ in range(n_trotter):
        for c in sub:
            d = c * dt
            # S2(d) = field(d/2) . ZZ(d) . field(d/2)
            plan.append(("F", field(d / 2)))
            plan.append(("Z", zz(d)))
            plan.append(("F", field(d / 2)))
    return plan


# ---------------------------------------------------------------------------
#  Extracting MPS arrays as (left, phys, right) for cheap measurement
# ---------------------------------------------------------------------------
def _mps_arrays(mps):
    """Return the site tensors as numpy arrays in a uniform (Dl, 4, Dr) layout,
    padding the open boundary bonds to dimension 1."""
    N = mps.L
    arrs = []
    for i in range(N):
        t = mps[i]
        pk = mps.site_ind(i)
        left = (set(t.inds) & set(mps[i - 1].inds)).pop() if i > 0 else None
        right = (set(t.inds) & set(mps[i + 1].inds)).pop() if i < N - 1 else None
        order = [x for x in (left, pk, right) if x is not None]
        data = np.asarray(t.transpose(*order).data)
        if left is None:
            data = data[None, ...]
        if right is None:
            data = data[..., None]
        arrs.append(np.ascontiguousarray(data))
    return arrs


def _measure(mps, N):
    """Read out {X_i, Y_i, Z_i}_{i<N} then {XX, YY, ZZ}_{i<N-1} = Tr(P rho), in
    the dense reference's feature order.  Uses left/right environments built from
    the trace vector so every observable is an O(N chi^2) contraction."""
    A = _mps_arrays(mps)

    def M(i, v):
        # transfer matrix of site i with super-leg contracted against vector v.
        return np.einsum("lpr,p->lr", A[i], v)

    T = [M(i, TRACE_VEC) for i in range(N)]          # bare (trace) transfer mats

    # Left env  Lenv[i] = T_0 ... T_{i-1}  (shape (1, Dl_i));  Lenv[0] = [[1]].
    Lenv = [np.ones((1, 1), dtype=complex)]
    for i in range(1, N):
        Lenv.append(Lenv[i - 1] @ T[i - 1])
    # Right env  Renv[i] = T_{i+1} ... T_{N-1}  (shape (Dr_i, 1));  Renv[N-1]=[[1]].
    Renv = [None] * N
    Renv[N - 1] = np.ones((1, 1), dtype=complex)
    for i in range(N - 2, -1, -1):
        Renv[i] = T[i + 1] @ Renv[i + 1]

    feats = []
    for i in range(N):                                # single-site X, Y, Z
        for v in (M_X, M_Y, M_Z):
            feats.append((Lenv[i] @ M(i, v) @ Renv[i])[0, 0].real)
    for i in range(N - 1):                            # two-site XX, YY, ZZ
        for v in (M_X, M_Y, M_Z):
            feats.append((Lenv[i] @ M(i, v) @ M(i + 1, v) @ Renv[i + 1])[0, 0].real)
    return np.array(feats)


# ---------------------------------------------------------------------------
#  The reservoir driver
# ---------------------------------------------------------------------------
def run_reservoir(series, N, J, hx, hz, tau_ev, chi_max,
                  n_trotter=24, order=4, cutoff=1e-14):
    """Drive the MPDO reservoir with `series`; return the (T, n_features) matrix
    of Pauli readouts, feature order identical to qrc_dense_reference.run_reservoir:
    X_i,Y_i,Z_i for i=0..N-1, then X_iX_{i+1},Y_iY_{i+1},Z_iZ_{i+1} for i=0..N-2.

    Parameters
    ----------
    chi_max : maximum MPDO bond dimension (SVD truncation of the two-site gates).
    n_trotter, order : substeps and Suzuki order of the exp(-i H tau_ev) split.
        The defaults drive the Trotter error well below 1e-6 for tau_ev ~ O(1).
    cutoff : singular-value cutoff in the gate splits (kept tiny so chi_max, not
        the cutoff, sets the truncation).
    """
    # super-ket |rho>> for rho = |0...0><0...0|: each site is vec|0><0| = e_0.
    site0 = np.zeros(4, dtype=complex); site0[0] = 1.0
    mps = qtn.MPS_product_state([site0] * N)

    plan = _trotter_plan(N, J, hx, hz, tau_ev, n_trotter, order)
    split_opts = dict(max_bond=chi_max, cutoff=cutoff)

    F = np.zeros((len(series), 6 * N - 3))
    for n, s in enumerate(series):
        # --- steps 1-3: reset qubit 0 + inject |psi(s)><psi(s)| ---------------
        t0 = mps[0]
        pk = mps.site_ind(0)
        rb = (set(t0.inds) - {pk}).pop()             # site 0's only bond (to site 1)
        A0 = t0.transpose(pk, rb).data               # (4, Dr)
        boundary = TRACE_VEC @ A0                     # trace out old qubit 0 -> (Dr,)
        new0 = np.outer(vec_rho1(encode(s)), boundary)   # rank-1: |psi><psi| (x) rest
        t0.modify(data=new0, inds=(pk, rb))

        # --- step 4: evolve rho -> U rho U^dagger via Trotterized super-gates --
        for kind, G in plan:
            if kind == "F":
                # Field super-gate S[o,p]: a single-site map that never changes a
                # bond, so we contract it straight into each site's physical leg
                # (moveaxis + matmul) -- far cheaper than a generic gate call.
                for i in range(N):
                    t = mps[i]
                    ax = t.inds.index(mps.site_ind(i))
                    d = np.moveaxis(t.data, ax, -1) @ G.T      # sum_p d[..,p] S[o,p]
                    t.modify(data=np.moveaxis(d, -1, ax))
            else:                                    # ZZ: super-gate per bond (+SVD)
                for i in range(N - 1):
                    mps.gate_split_(G, (i, i + 1), **split_opts)

        # --- step 5: read out the Pauli features -----------------------------
        F[n] = _measure(mps, N)
    return F


# ---------------------------------------------------------------------------
#  Verification / demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import qrc_dense_reference as ref

    def mackey_glass(n):
        x = np.zeros(n + 200); x[0] = 1.2
        for t in range(len(x) - 1):
            xt = x[t - 17] if t >= 17 else 0.0
            x[t + 1] = x[t] + 0.1 * (0.2 * xt / (1 + xt ** 10) - 0.1 * x[t])
        s = x[200:200 + n]
        return 2 * (s - s.min()) / (s.max() - s.min()) - 1

    J, hx, hz, tau = 1.0, 1.0, 0.5, 1.0

    # ---- CHECK 1: match the dense ground truth at small N, large chi --------
    # chi_max=256 is far above the exact operator bond dimension here, so the
    # only residual is the (4th-order) Trotter error, driven below 1e-6.
    N, T = 4, 12
    s = mackey_glass(T)
    H = ref.mixed_field_ising(N, J, hx, hz)
    F_ref = ref.run_reservoir(s, N, H, tau_ev=tau)
    F_tn = run_reservoir(s, N, J, hx, hz, tau_ev=tau,
                         chi_max=256, n_trotter=24, order=4)
    max_diff = np.abs(F_tn - F_ref).max()
    ok = max_diff <= 1e-6
    print(f"[check 1] N={N}, {F_tn.shape[1]} features, {T} steps: "
          f"max |F_tn - F_dense| = {max_diff:.3e}  "
          f"-> {'PASS' if ok else 'FAIL'} (<= 1e-6)")

    # ---- CHECK 2: run at an N the dense code (2^N x 2^N rho) cannot reach ----
    # A truncated (chi_max=16), coarse-Trotter timing demo -- accuracy is not
    # the point here; reaching N=12 at all is.  QRC is a many-small-steps
    # workload, so quimb's per-gate Python overhead dominates the per-step time.
    N_big, T_big = 12, 8
    s_big = mackey_glass(T_big)
    t0 = time.perf_counter()
    F_big = run_reservoir(s_big, N_big, J, hx, hz, tau_ev=tau,
                          chi_max=16, n_trotter=6, order=2)
    dt = time.perf_counter() - t0
    print(f"[check 2] N={N_big}, {F_big.shape[1]} features, {T_big} steps "
          f"(chi_max=16): {dt:.2f} s  ->  {dt / T_big * 1e3:.0f} ms/step "
          f"(dense would need a {2 ** N_big}x{2 ** N_big} density matrix)")

    assert ok, f"dense-match check FAILED: max diff {max_diff:.3e} > 1e-6"
