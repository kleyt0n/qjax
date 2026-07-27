"""q-deformed weighting for derivative-free optimization (animated, 2-D).

A cross-entropy-method (CEM) optimizer reweights a sampled Gaussian population by
``exp_q(-cost / T)`` and refits the population to the weighted samples. The
entropic index ``q`` controls the tail of the weighting:

- ``q = 1`` is the Boltzmann weight ``exp(-cost/T)`` — light (exponential) tails
  that concentrate greedily on the current best samples.
- ``q > 1`` gives heavy (power-law) tails that keep weight on far-from-best
  candidates, sustaining exploration.

On a deceptive landscape — a deep global well plus a shallower decoy well near the
initialization — greedy ``q = 1`` collapses into the decoy, while heavy-tailed
``q = 2.5`` keeps exploring and reaches the global optimum.

This script renders a GIF (``examples/figures/optimization.gif``) with a filled
contour view (live population + path) beside a 3-D surface with the optimization
path traced on it; the final frame is also saved as a PDF.

Run with: ``uv run python examples/optimization.py``
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.animation as animation
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np

import qjax
from qjax.plots import CMAP, save_figure, use_qjax_style

FIG_DIR = Path(__file__).parent / "figures"

GLOBAL_MIN = jnp.array([2.0, 2.0])
DECOY = jnp.array([-2.0, -1.0])
INIT_MU = jnp.array([-2.0, -2.0])
INIT_SIGMA = jnp.array([2.3, 2.3])
STEPS = 48
POP = 120
TEMPERATURE = 0.5
QS = (1.0, 2.5)
GIF_DPI = 150  # render resolution of the animation

# 3-D camera: fixed elevation, azimuth swept slowly across the animation.
VIEW_ELEV = 34.0
AZIM_START = -78.0
AZIM_SWEEP = 80.0  # total degrees rotated over the run


def objective(p: jnp.ndarray) -> jnp.ndarray:
    """Deceptive 2-D cost: deep, wide global well at (2,2), decoy well at (-2,-1)."""
    x, y = p[..., 0], p[..., 1]
    global_well = -3.0 * jnp.exp(-0.5 * ((x - 2.0) ** 2 + (y - 2.0) ** 2) / 2.0)
    decoy_well = -2.2 * jnp.exp(-0.5 * ((x + 2.0) ** 2 + (y + 1.0) ** 2) / 0.7)
    bowl = 0.04 * (x**2 + y**2)
    ripple = 0.30 * jnp.sin(2.0 * x) * jnp.sin(2.0 * y)
    return global_well + decoy_well + bowl + ripple


def optimize(key: jax.Array, q: float):
    """q-weighted CEM; return the mean path (STEPS+1, 2) and populations (STEPS, POP, 2)."""
    mu = INIT_MU
    sigma = INIT_SIGMA
    means, pops = [mu], []
    for _ in range(STEPS):
        key, sub = jax.random.split(key)
        samples = mu + sigma * jax.random.normal(sub, (POP, 2))
        costs = objective(samples)
        weights = qjax.q_exp(-(costs - costs.min()) / TEMPERATURE, q)
        weights = weights / jnp.sum(weights)
        mu = jnp.sum(weights[:, None] * samples, axis=0)
        sigma = jnp.sqrt(jnp.sum(weights[:, None] * (samples - mu) ** 2, axis=0)) + 1e-3
        means.append(mu)
        pops.append(samples)
    return np.asarray(jnp.stack(means)), np.asarray(jnp.stack(pops))


def main() -> None:
    use_qjax_style()
    key = jax.random.PRNGKey(0)
    # The surface is filled with the (green-blue) qjax ramp, so the paths are
    # warm and deliberately off-ramp: hue alone separates them from the backdrop.
    # Luminance cannot, because the ramp spans nearly the full lightness range —
    # hence the contrasting stroke added to every path artist below.
    colors = ("#f7b267", "#d1495b")  # amber, crimson

    runs = {}
    for q in QS:
        means, pops = optimize(key, q)
        runs[q] = (means, pops)
        final = means[-1]
        cost = float(objective(jnp.asarray(final)))
        print(f"q={q:g}: final=({final[0]:+.2f}, {final[1]:+.2f})  cost={cost:.3f}")

    # Objective grid for the contour and surface.
    g = np.linspace(-5.0, 5.0, 160)
    gx, gy = np.meshgrid(g, g)
    gz = np.asarray(objective(jnp.stack([gx, gy], axis=-1)))

    fig = plt.figure(figsize=(15.0, 6.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.5], wspace=0.05)
    ax2d = fig.add_subplot(gs[0, 0])
    ax3d = fig.add_subplot(gs[0, 1], projection="3d")
    # Solid white background, legible on any page/theme.
    fig.patch.set_facecolor("white")

    # --- static backdrops ---
    cf = ax2d.contourf(gx, gy, gz, levels=30, cmap=CMAP)
    ax2d.grid(False)
    ax2d.scatter(*GLOBAL_MIN, marker="*", s=160, color="white", edgecolor="k", zorder=5)
    ax2d.scatter(*DECOY, marker="X", s=90, color="0.8", edgecolor="k", zorder=5)
    ax2d.set(xlabel="$x_1$", ylabel="$x_2$", title="cost landscape + population")
    ax2d.set_aspect("equal")
    fig.colorbar(cf, ax=ax2d, fraction=0.046, label="cost")

    ax3d.plot_surface(gx, gy, gz, cmap=CMAP, alpha=0.55, linewidth=0, antialiased=True)
    # Strip the axes, panes and grid for a "floating in space" surface, and zoom
    # in so it fills the (wider) panel; the camera azimuth is rotated in update().
    ax3d.set_axis_off()
    ax3d.set_box_aspect((1, 1, 0.55), zoom=1.5)
    ax3d.text2D(0.5, 0.96, "optimization path on surface", transform=ax3d.transAxes, ha="center")

    # --- dynamic artists, per q ---
    pop_scatter, path2d, head2d, path3d, head3d = {}, {}, {}, {}, {}
    halo = [path_effects.withStroke(linewidth=3.6, foreground="white")]
    for q, color in zip(QS, colors, strict=False):
        label = f"$q={q:g}$ (BGS)" if q == 1.0 else f"$q={q:g}$ (Tsallis)"
        pop_scatter[q] = ax2d.scatter([], [], s=6, color=color, alpha=0.35, zorder=3)
        (path2d[q],) = ax2d.plot(
            [], [], color=color, lw=2.0, zorder=4, label=label, path_effects=halo
        )
        (head2d[q],) = ax2d.plot([], [], "o", color=color, mec="k", ms=7, zorder=6)
        (path3d[q],) = ax3d.plot([], [], [], color=color, lw=2.5, path_effects=halo)
        (head3d[q],) = ax3d.plot([], [], [], "o", color=color, mec="k", ms=6)
    ax2d.legend(loc="upper left")

    def zlift(xy: np.ndarray) -> np.ndarray:
        """Cost values along a path, lifted slightly so the line sits above the surface."""
        return np.asarray(objective(jnp.asarray(xy))) + 0.15

    def update(frame: int):
        # Slowly rotate the camera for a better view of the surface.
        azim = AZIM_START + AZIM_SWEEP * frame / max(STEPS - 1, 1)
        ax3d.view_init(elev=VIEW_ELEV, azim=azim)
        artists = []
        for q in QS:
            means, pops = runs[q]
            path = means[: frame + 2]  # mean path through this step
            pts = pops[frame]
            pop_scatter[q].set_offsets(pts)
            path2d[q].set_data(path[:, 0], path[:, 1])
            head2d[q].set_data([path[-1, 0]], [path[-1, 1]])
            zc = zlift(path)
            path3d[q].set_data_3d(path[:, 0], path[:, 1], zc)
            head3d[q].set_data_3d([path[-1, 0]], [path[-1, 1]], [zc[-1]])
            artists += [pop_scatter[q], path2d[q], head2d[q], path3d[q], head3d[q]]
        return artists

    fig.suptitle("q-weighted CEM — BGS ($q=1$) vs Tsallis ($q=2.5$)", y=0.98)
    anim = animation.FuncAnimation(fig, update, frames=STEPS, interval=120, blit=False)
    fig.tight_layout()

    gif_path = FIG_DIR / "optimization.gif"
    anim.save(
        gif_path,
        writer=animation.PillowWriter(fps=12),
        dpi=GIF_DPI,
        savefig_kwargs={"facecolor": "white"},
    )
    print(f"saved {gif_path}")

    update(STEPS - 1)  # leave the figure on the final frame
    print(f"saved {save_figure(fig, FIG_DIR / 'optimization')}")


if __name__ == "__main__":
    main()
