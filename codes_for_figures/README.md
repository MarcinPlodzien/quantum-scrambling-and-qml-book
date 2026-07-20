# Quantum Chaos and Information Scrambling — Figure Codes

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)

**GitHub repository:** [`quantum-scrambling-and-qml-book`](https://github.com/MarcinPlodzien/quantum-scrambling-and-qml-book)

Companion code for the book:

> **M. Płodzień**, *Quantum Chaos and Information Scrambling: From Random Matrix Theory to Quantum Machine Learning*,
> Lecture Notes in Physics, Springer (2026).

This repository contains Python scripts that generate all publication-quality figures in the monograph.

---

## Book Structure

| Chapter | Title | Figures |
|---------|-------|---------|
| 1 | Mathematical Structures of Hilbert Space | — |
| 2 | Random Matrix Theory and Spectral Statistics | Level repulsion, spacing distributions, Wigner semicircle |
| 3 | Haar Random Ensembles and Weingarten Calculus | Marchenko–Pastur density, Page curve, entropy concentration |
| 4 | Quantum Dynamics and Information Scrambling | Spectral form factor, OTOCs, butterfly velocity, Krylov complexity |
| 5 | Unitary Designs, Circuit Complexity, and Magic | Entanglement scaling, Marchenko–Pastur convergence, DKL, stabilizer Rényi entropy |
| 6 | Benchmarking Quantum Hardware | — |
| 7 | Quantum Machine Learning | — |
| 8 | Conclusions | — |

## Repository Structure

```
codes_for_figures/
├── get_fig_ch2.py                          # Ch.2: RMT spectral statistics
├── get_fig_ch3.py                          # Ch.3: Marchenko–Pastur, Page curve, concentration
├── get_fig_ch4_sff.py                      # Ch.4: Spectral form factor (Fig 4.1)
├── get_fig_ch4_sff_unfolded.py             # Ch.4: SFF ramp β-dependence on unfolded spectra (Fig 4.2)
├── get_fig_ch4_otoc.py                     # Ch.4: OTOC dynamics
├── get_fig_ch4_otoc_v_butterfly.py         # Ch.4: Butterfly velocity
├── get_fig_ch4_otoc_expm_and_krylov_jax.py # Ch.4: OTOC (JAX) + Krylov complexity
├── get_fig_ch4_krylov.py                   # Ch.4: Krylov complexity (Lanczos)
├── get_fig_ch5_combined.py                 # Ch.5: Combined MP + DKL convergence
├── get_fig_ch5_dkl_convergence.py          # Ch.5: DKL to Marchenko–Pastur
├── get_fig_ch5_entanglement_scaling_analog.py   # Ch.5: Entanglement scaling (analog)
├── get_fig_ch5_clifford_rigidity.py       # Ch.5: Clifford eigenphases have no spectral rigidity (Fig 5.1)
├── get_fig_ch5_entanglement_scaling_digital.py  # Ch.5: Entanglement scaling (digital)
├── get_fig_ch5_mp_convergence.py           # Ch.5: Marchenko–Pastur convergence
├── get_fig_ch5_sre_analog.py              # Ch.5: Stabilizer Rényi entropy (analog)
├── get_fig_ch5_sre_digital.py             # Ch.5: Stabilizer Rényi entropy (digital)
├── get_fig_ch5_sre_scaling.py             # Ch.5: SRE scaling with system size
└── README.md
```

Each script is named for the chapter whose figures it generates: `get_fig_chN_*.py` writes `fig_chN_*.pdf` into `figures/chN/`, and caches its intermediate arrays in `data/chN/`.

## Requirements

```
numpy
scipy
matplotlib
jax
jaxlib
```

Install with:
```bash
pip install numpy scipy matplotlib jax jaxlib
```

Some scripts for large system sizes (N ≥ 12) benefit from GPU acceleration via JAX.

## Usage

Each script is self-contained and generates one or more PDF figures:

```bash
python get_fig_ch2.py          # → fig_ch2_*.pdf
python get_fig_ch4_otoc.py     # → fig_ch4_otoc.pdf
python get_fig_ch5_combined.py # → fig_ch5_mp_dkl.pdf
```

Output figures are saved in the corresponding chapter directories (`figures/ch2/`, ..., `figures/ch7/`).

Scripts that perform expensive exact diagonalization cache their results under `data/chN/` and accept:

```bash
python get_fig_ch2.py --plot-only    # re-plot from cached data (fast)
python get_fig_ch2.py --regenerate   # force recomputation from scratch
```

## Citation

If you use this code or find it helpful, please cite the book:

```bibtex
@book{plodzien2026scrambling,
  title   = {Quantum Chaos and Information Scrambling:
             From Random Matrix Theory to Quantum Machine Learning},
  author  = {P{\l}odzi{\'e}{\'n}, Marcin},
  series  = {Lecture Notes in Physics},
  year    = {2026},
  publisher = {Springer}
}
```

## License

This code is released under the [MIT License](LICENSE).
