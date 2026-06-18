"""Plots of the ``q``-deformed elementary functions across a range of ``q``."""

from __future__ import annotations

from collections.abc import Sequence

import jax.numpy as jnp
import matplotlib.pyplot as plt

from qjax.core.functions import q_exp, q_log
from qjax.plots.style import qcolors, use_qjax_style


def plot_q_log(
    q_values: Sequence[float] = (0.5, 0.8, 1.0, 1.5, 2.0),
    x_range: tuple[float, float] = (0.05, 4.0),
    num: int = 400,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Plot the ``q``-logarithm for several entropic indices.

    Args:
        q_values: Entropic indices to draw, one curve each.
        x_range: ``(min, max)`` of the (positive) domain.
        num: Number of sample points.
        ax: Existing axis to draw on; a new one is created if ``None``.

    Returns:
        The axis containing the plot.
    """
    use_qjax_style()
    if ax is None:
        _, ax = plt.subplots()
    x = jnp.linspace(x_range[0], x_range[1], num)
    for q, color in zip(q_values, qcolors(len(q_values)), strict=False):
        ax.plot(x, q_log(x, q), color=color, label=f"q = {q:g}")
    ax.axhline(0.0, color="0.6", lw=0.8)
    ax.set(xlabel="x", ylabel=r"$\ln_q(x)$", title="q-logarithm")
    ax.legend()
    return ax


def plot_q_exp(
    q_values: Sequence[float] = (0.5, 0.8, 1.0, 1.5, 2.0),
    x_range: tuple[float, float] = (-3.0, 2.0),
    num: int = 400,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Plot the ``q``-exponential for several entropic indices.

    Args:
        q_values: Entropic indices to draw, one curve each.
        x_range: ``(min, max)`` of the domain.
        num: Number of sample points.
        ax: Existing axis to draw on; a new one is created if ``None``.

    Returns:
        The axis containing the plot.
    """
    use_qjax_style()
    if ax is None:
        _, ax = plt.subplots()
    x = jnp.linspace(x_range[0], x_range[1], num)
    for q, color in zip(q_values, qcolors(len(q_values)), strict=False):
        ax.plot(x, q_exp(x, q), color=color, label=f"q = {q:g}")
    ax.set(xlabel="x", ylabel=r"$\exp_q(x)$", title="q-exponential")
    ax.legend()
    return ax


__all__ = ["plot_q_log", "plot_q_exp"]
