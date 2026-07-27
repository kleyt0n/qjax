"""Publication-grade plotting style for `qjax`, themed on the brand ramp.

`use_qjax_style` configures Matplotlib for research-grade, vector output:
serif text with Computer-Modern math, embedded fonts, thin in-pointing ticks,
and a color cycle drawn from the qjax ramp. `qcolors` samples a discrete
sequence from that ramp so a family of curves indexed by ``q`` shares a coherent
identity, and `save_figure` writes a tight, font-embedded PDF.

The ramp
--------
`QJAX_RAMP` is the ten-step green-blue scale the logo and documentation are
built from, ordered light to dark. It is registered with Matplotlib as
``"qjax"`` (plus ``"qjax_r"``), so `CMAP` works anywhere a colormap name does.

It is a *sequential* scale: it encodes magnitude, which is exactly what a family
of curves indexed by ``q`` needs. Two consequences worth knowing:

- **`qcolors` windows the ramp** to ``[0.40, 1.0]`` by default. The three
  lightest steps sit between 1.3:1 and 1.7:1 against white — invisible as thin
  lines. The window starts where the ramp first clears the 2:1 floor for a
  sequential light end.
- **The ramp cannot supply a categorical palette.** Exhaustive search over all
  1820 four-colour subsets (including interpolated mid-steps) found none that
  passes the categorical checks: every subset with usable separation
  (normal-vision OKLab ΔE >= 15) buys it from the extremes that fall outside the
  lightness band and below 3:1 contrast. Where a figure distinguishes *methods*
  rather than magnitudes, carry identity with linestyle and markers and let
  colour be the secondary cue.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler
from matplotlib.colors import LinearSegmentedColormap

#: The qjax brand ramp, light to dark. Shared with the logo and the docs theme;
#: rebranding means swapping these ten values and nothing else.
QJAX_RAMP: tuple[str, ...] = (
    "#d9ed92",
    "#b5e48c",
    "#99d98c",
    "#76c893",
    "#52b69a",
    "#34a0a4",
    "#168aad",
    "#1a759f",
    "#1e6091",
    "#184e77",
)

#: The colormap used throughout qjax figures.
CMAP = "qjax"

#: Default window into the ramp for discrete curve colors. The lower bound is
#: set where the ramp first clears a 2:1 contrast ratio against white; below it
#: a thin line is not reliably visible.
QCOLORS_LO = 0.40
QCOLORS_HI = 1.0


def _register_colormap() -> None:
    """Register the qjax ramp (and its reverse) with Matplotlib, idempotently."""
    for name, colors in ((CMAP, QJAX_RAMP), (f"{CMAP}_r", tuple(reversed(QJAX_RAMP)))):
        if name not in mpl.colormaps:
            mpl.colormaps.register(LinearSegmentedColormap.from_list(name, colors, N=256))


_register_colormap()


def qcolors(n: int, lo: float = QCOLORS_LO, hi: float = QCOLORS_HI) -> list:
    """Sample ``n`` evenly spaced colors from the qjax ramp.

    Intended for curves indexed by an ordered parameter — a family of ``q``
    values, a noise sweep — where the reader should see the ordering in the
    color. For unordered categories (competing methods, class labels) the ramp
    cannot give reliable separation; see the module docstring.

    The default ``[lo, hi]`` window starts partway down the ramp so the lightest
    curve still reads against a white page.

    Args:
        n: Number of colors to return.
        lo: Lower bound of the colormap window, in ``[0, 1]``.
        hi: Upper bound of the colormap window, in ``[0, 1]``.

    Returns:
        A list of ``n`` RGBA tuples.
    """
    cmap = mpl.colormaps[CMAP]
    if n <= 0:
        return []
    if n == 1:
        # A lone curve should get a mid-ramp color, not the faintest step.
        return [cmap(0.5 * (lo + hi))]
    return [cmap(p) for p in np.linspace(lo, hi, n)]


#: A Matplotlib linestyle: a named style or an ``(offset, on/off dashes)`` pair.
LineStyle = str | tuple[float, tuple[float, ...]]

#: Dash patterns for distinguishing unordered categories. Ordered by how quickly
#: they separate from a solid line, so the first two are the most distinct.
QLINESTYLES: tuple[LineStyle, ...] = ("-", "--", ":", "-.", (0.0, (3.0, 1.0, 1.0, 1.0, 1.0, 1.0)))


def qlinestyles(n: int) -> list[LineStyle]:
    """Return ``n`` distinguishable Matplotlib linestyles.

    The brand ramp is sequential, so colour alone cannot separate more than about
    three unordered categories: sampling it for four competing *methods* puts two
    dark blues side by side that measure well under the readability floor (OKLab
    ΔE ~6 against a floor of 15). Pairing `qcolors` with these dash patterns
    supplies the second, non-colour channel, which also keeps figures readable in
    grayscale print and for colour-vision deficiencies.

    Args:
        n: Number of linestyles to return.

    Returns:
        A list of ``n`` linestyle specifications, cycling if ``n`` exceeds the
        number of defined patterns.
    """
    if n <= 0:
        return []
    return [QLINESTYLES[i % len(QLINESTYLES)] for i in range(n)]


def use_qjax_style() -> None:
    """Apply the qjax publication style (serif math, vector PDF, brand-ramp cycle)."""
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
            # Color: the qjax ramp and a matching discrete cycle.
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


__all__ = [
    "CMAP",
    "QJAX_RAMP",
    "QLINESTYLES",
    "LineStyle",
    "qcolors",
    "qlinestyles",
    "use_qjax_style",
    "save_figure",
]
