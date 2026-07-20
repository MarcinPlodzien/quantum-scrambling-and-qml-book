# Companion code — *Quantum Chaos and Information Scrambling: From Random Matrix Theory to Quantum Machine Learning*

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/)

Reproducible code for the Springer monograph by **Marcin Płodzień**. Everything needed to
regenerate the book's figures and to work through its exercises with fully worked solutions.

## Contents

| Path | What it is |
|------|------------|
| `codes_for_figures/` | Scripts that regenerate every figure in the book (`get_fig_*.py`), plus the shared helpers `plot_style.py`, `magic.py`, `qubit_syk.py`. |
| `codes_for_figures/data/` | Pre-generated numerical data (`.npz`) so the figures reproduce immediately, without recomputing the heavy random-matrix and dynamics runs. |
| `notebooks/` | 63 solution notebooks, one per exercise, named `solution_ch<N>_<topic>.ipynb`. Each is self-contained: problem statement, worked derivation, and a numerical check. |

## Installation

Tested on **Python 3.13**. Versions are pinned with major-version caps so a future release
cannot silently break the code.

```bash
pip install -r requirements.txt
# optional: only for the tensor-network reservoir figure (codes_for_figures/qrc_tensor_networks/)
pip install -r requirements-optional.txt
```

## Usage

```bash
# regenerate a figure (reads codes_for_figures/data/ if present, else recomputes)
python codes_for_figures/get_fig_ch4_otoc.py

# open a worked solution
jupyter lab notebooks/
```

## Related

- **Interactive tutorial website** (rendered notebooks, hands-on walkthroughs):
  <https://github.com/MarcinPlodzien/quantum-chaos-and-scrambling>

## License

MIT — see [LICENSE](LICENSE).
