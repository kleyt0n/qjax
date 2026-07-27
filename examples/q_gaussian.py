"""Visualize the q-Gaussian family and compare samples against the density.

Run with: ``uv run python examples/q_gaussian.py``
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

import qjax
from qjax.plots import plot_q_gaussian, qcolors, save_figure, use_qjax_style

FIG_DIR = Path(__file__).parent / "figures"


def main() -> None:
    use_qjax_style()
    fig, (ax_pdf, ax_hist) = plt.subplots(1, 2, figsize=(12, 4.5))

    # Left: the density family across q (ramp-colored).
    plot_q_gaussian(q_values=(0.5, 1.0, 1.5, 2.0, 2.5), beta=1.0, ax=ax_pdf)

    # Right: histogram of samples (1 <= q < 3) overlaid on the analytic density.
    key = jax.random.PRNGKey(0)
    x_grid = jnp.linspace(-6.0, 6.0, 400)
    qs = (1.0, 1.5, 2.0)
    for q, color in zip(qs, qcolors(len(qs)), strict=False):
        samples = qjax.sample(key, q=q, beta=1.0, shape=(50_000,))
        # Drop (not clip) tail samples outside the view so heavy tails don't
        # pile up as spikes at the plot edges.
        samples = samples[jnp.abs(samples) <= 6.0]
        ax_hist.hist(samples, bins=120, density=True, color=color, alpha=0.35)
        ax_hist.plot(x_grid, qjax.q_gaussian_pdf(x_grid, q, 1.0), color=color, label=f"q = {q:g}")
    ax_hist.set(xlabel="x", ylabel="density", title="samples vs. density")
    ax_hist.legend()

    fig.tight_layout()
    print(f"saved {save_figure(fig, FIG_DIR / 'q_gaussian')}")


if __name__ == "__main__":
    main()
