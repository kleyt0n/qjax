"""Plots of the ``q``-Gaussian density across a range of ``q``."""

from __future__ import annotations

from collections.abc import Sequence

import jax.numpy as jnp
import matplotlib.pyplot as plt

from qjax.core.distributions import q_gaussian_pdf
from qjax.plots.style import qcolors, use_qjax_style


def plot_q_gaussian(
    q_values: Sequence[float] = (0.5, 1.0, 1.5, 2.0, 2.5),
    beta: float = 1.0,
    x_range: tuple[float, float] = (-5.0, 5.0),
    num: int = 500,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Plot the ``q``-Gaussian density for several entropic indices.

    Lower ``q`` gives compact support; ``q -> 1`` is the Gaussian; higher ``q``
    (up to 3) gives progressively heavier tails.

    Args:
        q_values: Entropic indices to draw, one curve each (``q < 3``).
        beta: Shared inverse-width parameter.
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
        ax.plot(x, q_gaussian_pdf(x, q, beta), color=color, label=f"q = {q:g}")
    ax.set(xlabel="x", ylabel="density", title=f"q-Gaussian (β = {beta:g})")
    ax.legend()
    return ax


__all__ = ["plot_q_gaussian"]
