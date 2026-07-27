"""Publication-grade plotting style for :mod:`qjax`, themed on ``magma``.

:func:`use_qjax_style` configures Matplotlib for research-grade, vector output:
serif text with Computer-Modern math, embedded fonts, thin in-pointing ticks,
and a ``magma`` color cycle. :func:`qcolors` samples a discrete sequence from
``magma`` so a family of curves indexed by ``q`` shares a coherent identity,
and :func:`save_figure` writes a tight, font-embedded PDF.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler

#: The colormap used throughout qjax figures.
CMAP = "magma"


def qcolors(n: int, lo: float = 0.1, hi: float = 0.85) -> list:
    """Sample ``n`` evenly spaced colors from the ``magma`` colormap.

    The default ``[lo, hi]`` window avoids the near-black and near-white
    extremes so every curve stays legible on a white background.

    Args:
        n: Number of colors to return.
        lo: Lower bound of the colormap window, in ``[0, 1]``.
        hi: Upper bound of the colormap window, in ``[0, 1]``.

    Returns:
        A list of ``n`` RGBA tuples.
    """
    cmap = mpl.colormaps[CMAP]
    positions = np.linspace(lo, hi, max(n, 1))
    return [cmap(p) for p in positions]


def use_qjax_style() -> None:
    """Apply the qjax publication style (serif math, vector PDF, magma cycle)."""
    plt.rcParams.update(
        {
            # Typography: serif body with Computer-Modern math (no system LaTeX
            # required). pdf/ps fonttype 42 embeds editable TrueType outlines.
            "text.usetex": False,
            "font.family": "serif",
            "font.serif": ["CMU Serif", "Times New Roman", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "axes.formatter.use_mathtext": True,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.fontsize": 9.5,
            # Color: magma colormap and a matching discrete cycle.
            "image.cmap": CMAP,
            "axes.prop_cycle": cycler(color=qcolors(5)),
            # Figure / output: single-column default, high-resolution rasters.
            "figure.figsize": (6.0, 4.0),
            "figure.dpi": 150,
            "savefig.dpi": 600,
            "savefig.format": "pdf",
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
            "savefig.transparent": False,
            # Axes, lines, and ticks.
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "lines.linewidth": 1.8,
            "lines.markersize": 5,
            "legend.frameon": False,
            "legend.handlelength": 1.6,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.minor.visible": True,
            "ytick.minor.visible": True,
            "xtick.major.size": 4.0,
            "ytick.major.size": 4.0,
            "xtick.minor.size": 2.0,
            "ytick.minor.size": 2.0,
        }
    )


def save_figure(fig: plt.Figure, path: str | Path, transparent: bool = False) -> Path:
    """Save ``fig`` as a tight, font-embedded vector PDF.

    The extension is forced to ``.pdf`` and the parent directory is created if
    needed, so callers can pass a bare stem like ``figures/q_gaussian``.

    Args:
        fig: The figure to write.
        path: Destination path; any extension is replaced with ``.pdf``.
        transparent: If ``True``, write with a transparent background so the
            figure blends with whatever it is placed on.

    Returns:
        The resolved output path.
    """
    out = Path(path).with_suffix(".pdf")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", pad_inches=0.03, transparent=transparent)
    return out


__all__ = ["CMAP", "qcolors", "use_qjax_style", "save_figure"]
