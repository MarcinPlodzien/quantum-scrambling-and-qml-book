# Quantum Reservoir Computing with Tensor Networks

Tensor-network implementations of the quantum reservoir computer (QRC) used in
the Santa Fe forecasting example, so that reservoirs larger than the dense
`~8`-qubit limit can be simulated. Three back ends are provided and each is
checked against the same dense ground truth.

## The model and protocol

Reservoir: a nonintegrable **mixed-field Ising chain**

```
H = J Σ Z_i Z_{i+1}  +  h_x Σ X_i  +  h_z Σ Z_i
```

(the longitudinal `h_z` is what breaks Jordan–Wigner integrability and makes the
chain genuinely scramble). One reservoir step, given the next input `s ∈ [-1,1]`:

1. **encode** `s` into qubit 0 by `R_y(arccos s)|0>` (so `<Z> = s`);
2. **reset** the input register — partial-trace qubit 0 out of `ρ`;
3. **inject** the freshly encoded input in its place;
4. **evolve** `ρ → U ρ U†` with `U = exp(-i H t_qrc)`;
5. **read out** the Pauli observables `{X_i, Y_i, Z_i, X_iX_{i+1}, Y_iY_{i+1},
   Z_iZ_{i+1}}` as the feature vector.

Only a linear (ridge) readout on the features is trained. The reset is a CPTP
map, and CPTP maps contract trace distance — that is the origin of the
reservoir's *fading memory*.

## Why tensor networks

Because of the reset, the reservoir state is **mixed**, so it is represented as a
**matrix-product density operator (MPDO)**: an MPO whose two physical legs per
site are the ket and bra indices of `ρ`. The step becomes

- trace qubit 0  → contract site 0's ket/bra legs;
- inject         → prepend a rank-1 site `|ψ(s)><ψ(s)|`;
- evolve         → TEBD: Trotterize `U` into two-site gates and apply each to the
                   ket legs and its conjugate to the bra legs, compressing the
                   bond dimension back to `χ_max`;
- read out       → local expectation values from the MPDO.

The cost is `O(N · χ³)` per step instead of the dense `O(8^N)`. The catch is that
`χ` grows with the reservoir's *operator entanglement*: it stays small in the
under- and edge-of-scrambling regimes (where a reservoir is actually useful) and
blows up deep in the over-scrambled regime — so the tensor-network speed-up is
largest exactly where QRC operates.

## Files

| file | back end | notes |
|------|----------|-------|
| `qrc_dense_reference.py` | NumPy (dense) | ground truth, exact to `~N=8–10`; the others are verified against it |
| `qrc_quimb.py`   | [quimb](https://quimb.readthedocs.io) (Python) | flexible general-TN back end; jax/GPU capable |
| `qrc_tenpy.py`   | [TeNPy](https://tenpy.readthedocs.io) (Python) | purpose-built 1-D MPO/TEBD machinery |
| `qrc_itensor.jl` | [ITensors.jl](https://itensor.github.io) (Julia) | compiled; lowest per-step overhead |
| `benchmark.py`   | — | times the back ends against the dense reference |

## Which is fastest?

QRC is a *many-small-steps* workload (a long input series, swept over
parameters), so per-step overhead dominates at moderate bond dimension. Ranked
expectation, to be confirmed by `benchmark.py`:

- **ITensors.jl** — compiled loop, lowest overhead → fastest for many small steps;
- **quimb** — fastest Python option, especially with the `jax` backend / GPU;
- **TeNPy** — excellent MPO/TEBD, but more Python overhead per step.

At *large* `χ` the heavy contraction is BLAS-bound and the back ends converge, so
the language advantage is a small-to-moderate `χ` effect.

## Verification

Every back end reproduces the dense reference's feature matrix (and hence its
forecast `R²`) at small `N` and large `χ_max` to machine precision. Run
`benchmark.py` to reproduce the check and the timings.
