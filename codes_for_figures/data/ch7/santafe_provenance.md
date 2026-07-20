# Santa Fe laser dataset — provenance

`santafe_laser.npy` is **Data Set A of the Santa Fe Time Series Prediction
Competition** (1991–1992): the intensity of a far-infrared NH$_3$ (FIR) laser in a
chaotic (Lorenz-like) regime, sampled and quantised to 8 bits. It is the standard
chaotic-forecasting benchmark of reservoir computing.

## What is in the file
- `numpy` array, shape `(10093, 1)`, `dtype=int64`, values in `0..255` (8-bit).
- First five samples: `86, 141, 95, 41, 22`.
- `sha256(tobytes())` begins `cc4ace0cb515b4feb5f70ac3998ebe0d`.

These are the fingerprints the loader/downloader below checks against, so any copy
you obtain can be verified to be the same series used for the figure.

## Original source and citation
The data was recorded by U. Hübner and collaborators and released as Set A of the
competition edited by Weigend and Gershenfeld. Cite the competition volume:

> A. S. Weigend and N. A. Gershenfeld (eds.), *Time Series Prediction: Forecasting
> the Future and Understanding the Past*, Santa Fe Institute Studies in the
> Sciences of Complexity, Addison-Wesley (1993).

The competition data has been public and widely redistributed for three decades
(e.g. bundled with `reservoirpy` and many reservoir-computing repositories); it is
not proprietary. `download_santafe.py` fetches a copy and verifies it against the
fingerprints above, so the figure is reproducible from a clean checkout.

## How it is used in the book
`get_fig_ch7_santafe.py` rescales the raw 8-bit series to `[-1, 1]`, drives the
quantum reservoir with it one sample at a time, and trains a linear (ridge) readout
to forecast the intensity `tau` steps ahead (Fig. `fig:santafe`, Chapter 7).
