"""
Shared plotting style and data-cache helper for all book figures.

Every figure script should:

    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from plot_style import apply_book_style, load_or_compute, panel_label

    apply_book_style()

    data = load_or_compute(DATA_DIR / "fig_chN_name.npz", compute)
    # ... plot from `data` ...

This guarantees (i) one unified set of font sizes across the whole book, and
(ii) that expensive numerics run once and are reloaded on every later re-render
(e.g. a font-size change), unless a recompute is forced.

Force a recompute for a single run with the environment variable:

    FIG_RECOMPUTE=1 python get_fig_chN_name.py
"""

import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Unified house style.
#
# KEY POINT: what the reader sees is the font size ON THE PRINTED PAGE, which is
#     matplotlib_font_size * (include_width / figsize_width).
# Almost every figure is included at the same \textwidth, but a multi-panel
# figure has a much wider figsize than a single-panel one, so it is scaled DOWN
# more and its text would come out smaller with an identical font.size.  To make
# the PRINTED text uniform, the matplotlib font size must scale WITH the figure
# width.  apply_book_style(fig_width) does this automatically: a wider (more-
# panel) figure gets proportionally larger fonts, so all figures print at the
# same on-page size.
#
# Change the base sizes or REF_WIDTH_IN here and re-run any script to restyle
# from cached data (the numerics do not recompute).
# ---------------------------------------------------------------------------

# Fixed house sizes, matching the book's established figures (e.g. the
# entanglement-scaling figures).  Every figure uses the SAME font sizes; a
# figure looks right by choosing a sensible figsize and aspect ratio, not by
# rescaling its fonts.  Change these once here to restyle every figure.
_BOOK_RCPARAMS = {
    "font.family":     "serif",
    "font.size":       12,
    "axes.labelsize":  14,
    "axes.titlesize":  13,
    "legend.fontsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "figure.dpi":      150,
    "savefig.dpi":     300,   # PNG companions all at 300
    "savefig.bbox":    "tight",
    "text.usetex":     False,
}


def apply_book_style(fig_width_in=None, textwidth_frac=None):
    """Apply the unified book figure style (fixed house font sizes).

    The ``fig_width_in`` / ``textwidth_frac`` arguments are accepted for backward
    compatibility but ignored: font sizes are fixed across all figures, and each
    figure's on-page appearance is set by its ``figsize`` and the width at which
    it is included.  Aim for panels with a roughly 5:4 (width:height) aspect and
    include at ``\\textwidth`` for results consistent with the rest of the book.
    Call before creating the figure.
    """
    plt.rcParams.update(_BOOK_RCPARAMS)


def load_or_compute(data_file, compute, force=None):
    """Load cached figure data, or compute and cache it.

    Parameters
    ----------
    data_file : path-like
        Where the ``.npz`` cache for this figure lives.
    compute : callable
        Zero-argument function returning a ``dict`` of numpy arrays / scalars
        (the everything the plotting code needs).  Called only on a cache miss.
    force : bool or None
        If True, always recompute.  If None (default), recompute when the
        environment variable ``FIG_RECOMPUTE`` is set to a non-empty, non-"0"
        value.  Otherwise load from ``data_file`` when it exists.

    Returns
    -------
    dict
        The arrays, either loaded from disk or freshly computed.
    """
    data_file = Path(data_file)
    if force is None:
        force = os.environ.get("FIG_RECOMPUTE", "") not in ("", "0")

    if data_file.exists() and not force:
        with np.load(data_file, allow_pickle=True) as d:
            return {k: d[k] for k in d.files}

    result = compute()
    if not isinstance(result, dict):
        raise TypeError("compute() must return a dict of arrays/scalars")
    data_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez(data_file, **result)
    return result


def panel_label(ax, letter, loc="upper left", pad=0.03, **kwargs):
    """Draw a consistent bold panel label, e.g. ``(a)``, on an axis.

    loc is one of 'upper left', 'upper right', 'lower left', 'lower right'.
    """
    x = pad if "left" in loc else 1 - pad
    y = 1 - pad if "upper" in loc else pad
    ha = "left" if "left" in loc else "right"
    va = "top" if "upper" in loc else "bottom"
    txt = letter if letter.startswith("(") else f"({letter})"
    style = dict(fontsize=plt.rcParams["axes.titlesize"], fontweight="bold",
                 ha=ha, va=va)
    style.update(kwargs)
    ax.text(x, y, txt, transform=ax.transAxes, **style)
