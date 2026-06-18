"""Plotting helpers for qjax, themed with the ``magma`` colormap."""

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
