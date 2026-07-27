"""Plotting helpers for qjax, themed with the ``magma`` colormap.

Matplotlib is an optional dependency: importing ``qjax`` for the mathematics
does not pull it in. Install it with the ``plots`` extra::

    pip install "qjax[plots]"
"""

try:
    import matplotlib  # noqa: F401
except ImportError as exc:  # pragma: no cover - exercised only without matplotlib
    raise ImportError(
        "qjax.plots requires matplotlib, which is an optional dependency. "
        'Install it with:  pip install "qjax[plots]"'
    ) from exc

from qjax.plots.distributions import plot_q_gaussian
from qjax.plots.functions import plot_q_exp, plot_q_log
from qjax.plots.style import CMAP, qcolors, save_figure, use_qjax_style

__all__ = [
    "CMAP",
    "qcolors",
    "use_qjax_style",
    "save_figure",
    "plot_q_log",
    "plot_q_exp",
    "plot_q_gaussian",
]
