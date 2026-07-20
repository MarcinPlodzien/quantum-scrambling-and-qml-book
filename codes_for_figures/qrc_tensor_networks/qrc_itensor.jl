# =============================================================================
# qrc_itensor.jl
# =============================================================================
# Tensor-network (matrix-product density operator, MPDO) implementation of the
# quantum reservoir computer (QRC) whose EXACT dense ground truth lives in
# `qrc_dense_reference.py`.  This file is the ITensors.jl back end referenced by
# README.md, and it is verified against the dense reference at the bottom of the
# file (the `1e-6` machine-precision check plus a large-N timing run).
#
# -----------------------------------------------------------------------------
# THE MODEL AND PROTOCOL  (identical to the dense reference)
# -----------------------------------------------------------------------------
# Reservoir: an N-qubit nonintegrable mixed-field Ising chain
#
#     H = J * sum_i Z_i Z_{i+1}  +  h_x * sum_i X_i  +  h_z * sum_i Z_i .
#
# The longitudinal h_z is what breaks Jordan-Wigner integrability and makes the
# chain genuinely scramble.  One reservoir step, given the next input s in [-1,1]:
#
#     1. ENCODE   s into qubit 0 by R_y(arccos s)|0>  (so <Z> = s);
#     2. RESET    the input register: partial-trace qubit 0 out of rho;
#     3. INJECT   |psi(s)><psi(s)| in its place;
#     4. EVOLVE   rho -> U rho U^dagger  with  U = exp(-i H tau_ev);
#     5. READ OUT the Pauli features {X_i,Y_i,Z_i, X_iX_{i+1},Y_iY_{i+1},Z_iZ_{i+1}}.
#
# The reset is a CPTP map; CPTP maps contract trace distance, which is the origin
# of the reservoir's fading memory.  Because of the reset the state is MIXED, so
# it cannot be a pure-state MPS -- we carry the full density operator rho as an
# MPDO and evolve it with density-matrix TEBD.
#
# -----------------------------------------------------------------------------
# THE MPDO / INDEX STRUCTURE
# -----------------------------------------------------------------------------
# rho is stored as an ITensors.jl `MPO`: one tensor per site, each carrying TWO
# physical legs -- the KET leg (unprimed site index `s`) and the BRA leg (primed
# site index `s'`) -- plus the usual virtual/link legs between neighbours:
#
#         s1        s2              sN        (ket legs,  prime level 0)
#         |         |               |
#     [A_1]--l1--[A_2]--l2-- ... --[A_N]      (link legs l1..l_{N-1})
#         |         |               |
#         s1'       s2'             sN'       (bra legs,  prime level 1)
#
# so that  rho = sum_{k,l}  <k| rho |l>  |k><l|,  the unprimed and primed site
# legs being the two physical indices of the density operator.  We build rho with
# `outer(psi', psi)` (verified: <Z>=+1 and Tr=1 on |0..0>); ITensors fixes the
# internal ket/bra layout of that MPO, and the only place it is visible to us is
# the Pauli readout, where it shows up as an operator transpose -- see the note in
# `trace_expect`.  The bond dimension of this MPO measures the OPERATOR
# entanglement of rho; truncating it to `chi_max` is the whole point of the method
# (cost O(N chi^3) per step vs the dense O(8^N)).
#
# Julia site j (1-indexed) corresponds to Python/dense qubit j-1 (0-indexed), and
# qubit 0 -- the input register, leftmost / most significant bit in the dense
# code's big-endian `kron` ordering -- is Julia site 1.  All feature orderings and
# the trace-out/inject on site 1 are chosen to match the dense code bit-for-bit.
#
# -----------------------------------------------------------------------------
# TIME EVOLUTION  (density-matrix TEBD)
# -----------------------------------------------------------------------------
# We Trotterize U = exp(-i H tau_ev) into two-site gates.  Each bond (j,j+1) gets
# the gate exp(-i h_{j,j+1} dt) where h_{j,j+1} bundles the ZZ coupling on that
# bond together with the single-site fields, assigned to bonds so that every site
# field is counted exactly once (site j's field -> bond j; the last site's field
# -> the last bond).  A single symmetric (2nd-order Strang) substep is the forward
# sweep of half-gates followed by its reverse; `n_trotter` such substeps of size
# dt = tau_ev / n_trotter realize U to O(dt^2) accuracy.
#
# For a MIXED state we need U rho U^dagger, not U|psi>.  ITensors.jl does this in
# one call: `apply(gate, rho; apply_dag=true)` applies the gate to the ket legs
# AND its conjugate-transpose to the bra legs, then compresses the bond back down
# to `maxdim = chi_max` (with an SVD `cutoff`).  That truncation is the only
# approximation beyond the Trotter error; at large chi_max it is exact to machine
# precision, which is what the verification below exploits.
# =============================================================================

using ITensors
using ITensorMPS

# -----------------------------------------------------------------------------
# encode(s): the single-qubit input KET with <Z> = s, i.e. R_y(arccos s)|0>.
# Amplitudes [sqrt((1+s)/2), sqrt((1-s)/2)] over the computational basis {|0>,|1>}
# reproduce encode() in qrc_dense_reference.py exactly.
# -----------------------------------------------------------------------------
function encode_ket(s::Real, site::Index)
    a = sqrt((1 + s) / 2)
    b = sqrt((1 - s) / 2)
    return ITensor(ComplexF64[a, b], site)
end

# -----------------------------------------------------------------------------
# build_trotter_gates: the ordered list of two-site gates for ONE symmetric
# 2nd-order Trotter substep of size dt.  Applying this list realizes
# exp(-i H dt) to O(dt^3) per substep, hence O(dt^2) global error over tau_ev.
#
#   h_{j,j+1} = J Z_j Z_{j+1}
#             + (hx X_j + hz Z_j)               [site j's field -> bond j]
#             + (hx X_N + hz Z_N)  if j+1 == N  [last site's field -> last bond]
#
# The returned vector is [forward half-gates on bonds 1..N-1, then their reverse].
# Applying `forward then reverse` is the Strang symmetrization.
# -----------------------------------------------------------------------------
function build_trotter_gates(sites, J, hx, hz, dt)
    N = length(sites)
    forward = ITensor[]
    for j in 1:(N - 1)
        s1, s2 = sites[j], sites[j + 1]
        hbond = J * op("Z", s1) * op("Z", s2)
        hbond += hx * op("X", s1) * op("Id", s2) + hz * op("Z", s1) * op("Id", s2)
        if j + 1 == N
            hbond += hx * op("Id", s1) * op("X", s2) + hz * op("Id", s1) * op("Z", s2)
        end
        push!(forward, exp(-1im * (dt / 2) * hbond))   # half step for symmetry
    end
    return vcat(forward, reverse(forward))             # forward sweep + reverse sweep
end

# -----------------------------------------------------------------------------
# reset_inject!: steps 2+3 of the protocol on the MPDO, in place.
#
#   rho_new = |psi(s)><psi(s)|  (x)  Tr_{site 1}(rho)
#
# Implementation as local tensor surgery (keeps the fixed N-site index set):
#   * trace out site 1's physical legs -> a tensor on the (1,2) link;
#   * absorb it into site 2 (this removes the old (1,2) bond);
#   * overwrite site 1 with the rank-1 tensor |psi><psi|, reconnected to site 2
#     through a fresh dim-1 link L (the injected qubit is a product state, so the
#     bond dimension across it is 1 -- exactly the fading-memory reset).
# Trace-preserving by construction (verified: Tr stays 1).
# -----------------------------------------------------------------------------
function reset_inject!(rho::MPO, sites, s::Real)
    # (2) partial trace over site 1: contract its ket leg into its bra leg.
    tr1 = rho[1] * delta(sites[1], sites[1]')      # -> tensor carrying only link(1,2)
    newsite2 = tr1 * rho[2]                         # fold reduced state into site 2

    # (3) inject the fresh input as |psi><psi| on site 1.
    ket = encode_ket(s, sites[1])
    rho1 = ket * prime(dag(ket))                    # |psi> (s1) times <psi| (s1')
    L = Index(1, "Link,l=1")                        # dim-1 bond across the product site
    rho[1] = rho1 * onehot(L => 1)
    rho[2] = newsite2 * onehot(L => 1)
    return rho
end

# -----------------------------------------------------------------------------
# trace_expect: <O> = Tr(O rho) for a product observable O given as a
# Dict(site => opname).  Sites not in the Dict are traced out.
#
# Per site k the "cap" tensor is:
#   * operator site:  rho[k] * transpose(op(P, s_k))  -- sums the ket/bra legs
#                                                        against P;
#   * traced  site:   rho[k] * delta(s_k, s_k')        -- ordinary partial trace.
# The caps then contract along the links to a scalar.
#
# NOTE on the transpose.  ITensors builds `op(P, s)` with element
# op(s=b, s'=a) = <a|P|b>, and `outer(psi', psi)` stores rho with the physical
# legs laid out so that the bare contraction rho[k] * op(P, s_k) evaluates
# Tr(P^T rho) rather than Tr(P rho).  For the symmetric Paulis X, Z (P^T = P)
# this is already correct, but for the antisymmetric Y (Y^T = -Y) it flips the
# sign.  Feeding the transpose  swapprime(op, 0, 1) = P^T  makes every site
# evaluate Tr(P rho) exactly, matching the dense reference (verified below to
# 1e-6, including all Y_i and Y_iY_{i+1} features).
# We divide by Tr(rho) for numerical hygiene (it is 1 to rounding).
# -----------------------------------------------------------------------------
function trace_expect(rho::MPO, sites, ops::Dict{Int,String})
    T = ITensor(1.0)
    Z = ITensor(1.0)
    for k in 1:length(rho)
        if haskey(ops, k)
            T = T * (rho[k] * swapprime(op(ops[k], sites[k]), 0, 1))
        else
            T = T * (rho[k] * delta(sites[k], sites[k]'))
        end
        Z = Z * (rho[k] * delta(sites[k], sites[k]'))   # running Tr(rho)
    end
    return real(scalar(T) / scalar(Z))
end

# -----------------------------------------------------------------------------
# run_reservoir: drive the MPDO reservoir with `series` and return the
# (T x n_features) real feature matrix, with the SAME feature ordering as the
# dense reference:
#     for i in 1..N     : X_i, Y_i, Z_i
#     for i in 1..N-1   : X_iX_{i+1}, Y_iY_{i+1}, Z_iZ_{i+1}
# => 3N + 3(N-1) features.
#
# Positional signature matches the task spec exactly:
#     run_reservoir(series, N, J, hx, hz, tau_ev, chi_max)
# Keyword knobs (evolution accuracy / truncation) have safe defaults:
#     n_trotter : number of symmetric substeps used to realize exp(-i H tau_ev)
#                 (Trotter error ~ (tau_ev/n_trotter)^2);
#     cutoff    : SVD truncation threshold used alongside maxdim = chi_max.
# -----------------------------------------------------------------------------
function run_reservoir(series, N::Int, J::Real, hx::Real, hz::Real,
                       tau_ev::Real, chi_max::Int;
                       n_trotter::Int = 200, cutoff::Float64 = 1e-14)
    sites = siteinds("Qubit", N)

    # rho starts in |0...0><0...0| (matches rho[0,0]=1 in the dense code).
    rho = outer(MPS(sites, "0")', MPS(sites, "0"))

    # Precompute the Trotter gate list for a single substep of size dt.
    dt = tau_ev / n_trotter
    gates = build_trotter_gates(sites, J, hx, hz, dt)

    # Precompute the readout observable specs (list of Dict(site=>op)) in order.
    specs = Dict{Int,String}[]
    for i in 1:N
        push!(specs, Dict(i => "X"))
        push!(specs, Dict(i => "Y"))
        push!(specs, Dict(i => "Z"))
    end
    for i in 1:(N - 1)
        push!(specs, Dict(i => "X", i + 1 => "X"))
        push!(specs, Dict(i => "Y", i + 1 => "Y"))
        push!(specs, Dict(i => "Z", i + 1 => "Z"))
    end

    T = length(series)
    F = zeros(Float64, T, length(specs))

    for (n, s) in enumerate(series)
        # (2)+(3) reset the input register and inject the fresh encoded input.
        reset_inject!(rho, sites, s)

        # (4) evolve rho -> U rho U^dagger via n_trotter density-matrix TEBD
        #     substeps, truncating the operator bond to chi_max each time.
        for _ in 1:n_trotter
            rho = apply(gates, rho; apply_dag = true, maxdim = chi_max, cutoff = cutoff)
        end

        # (5) read out the Pauli feature vector.
        for (c, spec) in enumerate(specs)
            F[n, c] = trace_expect(rho, sites, spec)
        end
    end
    return F
end

# =============================================================================
# VERIFICATION  (runs when this file is executed as a script)
# =============================================================================
# 1. small-N, large-chi comparison against the dense Python ground truth
#    (max abs feature difference must be <= 1e-6);
# 2. a larger-N run reporting the per-step wall time.
# =============================================================================
function _read_csv_matrix(path::String)
    rows = Vector{Vector{Float64}}()
    for line in eachline(path)
        isempty(strip(line)) && continue
        push!(rows, parse.(Float64, split(strip(line), ",")))
    end
    return reduce(vcat, (reshape(r, 1, :) for r in rows))
end

if abspath(PROGRAM_FILE) == @__FILE__
    here = @__DIR__

    # ---- shared test problem -------------------------------------------------
    Nver   = 5
    J, hx, hz = 1.0, 1.0, 0.5
    tau_ev = 1.0
    # a short deterministic input series in [-1,1]
    series = [sin(0.7 * n) * cos(0.3 * n) for n in 0:11]

    # ---- (A) dump the dense reference via Python -----------------------------
    series_path = joinpath(tempdir(), "qrc_series.csv")
    ref_path    = joinpath(tempdir(), "qrc_ref.csv")
    open(series_path, "w") do io
        println(io, join(string.(series), ","))
    end
    pyscript = """
import sys, numpy as np
sys.path.insert(0, r"$here")
import qrc_dense_reference as q
series = np.loadtxt(r"$series_path", delimiter=",")
H = q.mixed_field_ising($Nver, $J, $hx, $hz)
F = q.run_reservoir(series, $Nver, H, tau_ev=$tau_ev)
np.savetxt(r"$ref_path", F, delimiter=",")
print("dense reference:", F.shape)
"""
    println("[verify] computing dense reference via Python ...")
    run(pipeline(`python3 -c $pyscript`))
    Fref = _read_csv_matrix(ref_path)

    # ---- (B) ITensor MPDO with large chi (essentially exact) -----------------
    # For N=5 the MPDO's operator bond is capped at 4^2 = 16, so chi_max = 64 is
    # already lossless; the residual vs. the dense reference is pure Trotter error,
    # driven below 1e-6 by n_trotter = 400 substeps of exp(-i H tau_ev).
    println("[verify] computing ITensor MPDO feature matrix (N=$Nver, large chi) ...")
    Ftn = run_reservoir(series, Nver, J, hx, hz, tau_ev, 64; n_trotter = 400)

    maxdiff = maximum(abs.(Ftn .- Fref))
    println("\n================ VERIFICATION ================")
    println("feature matrix size : ", size(Ftn), "  (dense: ", size(Fref), ")")
    println("max abs difference  : ", maxdiff)
    passed = maxdiff <= 1e-6
    println("1e-6 check          : ", passed ? "PASS" : "FAIL")
    println("==============================================\n")

    # ---- (C) larger-N timing -------------------------------------------------
    Nbig = 12
    chi_big = 48
    n_trot_big = 16
    println("[timing] warming up N=$Nbig, chi=$chi_big ...")
    run_reservoir([0.2, -0.3], Nbig, J, hx, hz, tau_ev, chi_big; n_trotter = n_trot_big)  # compile/warmup
    nsteps = 6
    tbig = [0.5 * sin(0.9n) for n in 1:nsteps]
    t0 = time()
    run_reservoir(tbig, Nbig, J, hx, hz, tau_ev, chi_big; n_trotter = n_trot_big)
    dt_total = time() - t0
    println("[timing] N=$Nbig chi=$chi_big n_trotter=$n_trot_big : ",
            round(dt_total / nsteps * 1000; digits=1), " ms/step ",
            "(", nsteps, " steps in ", round(dt_total; digits=2), " s)")

    passed || error("VERIFICATION FAILED: max abs diff = $maxdiff > 1e-6")
end
