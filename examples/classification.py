"""Label-noise robustness: Tsallis cross-entropy vs. the Shannon baseline.

We train the *same* over-parameterized MLP on a synthetic 4-class problem under
increasing levels of complexity — a growing fraction of corrupted training
labels — and vary a single knob: the entropic index ``q`` of the
``tsallis_cross_entropy`` loss applied to softmax outputs.

- ``q = 1`` is *exactly* the **Shannon baseline** (standard cross-entropy). The
  loss ``-log p_c`` is unbounded, so a confidently mislabeled example produces an
  arbitrarily large gradient and the network memorizes the noise.
- ``q < 1`` gives a **bounded** Tsallis loss ``-ln_q p_c = (1 - p_c^{1-q})/(1-q)``
  (the generalized cross-entropy of Zhang & Sabuncu, 2018). Its gradient
  saturates on hard/mislabeled points, so clean-set accuracy degrades far more
  gracefully as noise rises.
- **learnable ``q``**: rather than fixing ``q`` we make it a trainable parameter
  optimized jointly with the network (``qjax.nn.bounded_q``).
  Minimizing the bounded Tsallis loss over the training set monotonically favors
  smaller ``q`` (it down-weights unfittable, noisy points), so ``q`` *descends to
  the robust end* of its allowed range. The upshot is practical: instead of
  grid-searching ``q``, gradient descent **discovers** the robust regime on its
  own and matches — or slightly beats — the best hand-tuned fixed ``q`` at every
  noise level. (Because the descent favors small ``q`` regardless of noise, we
  start ``q`` already inside the robust regime to avoid memorizing noise early.)

Each configuration is averaged over several seeds and evaluated on a *clean*
held-out test set.

Run with: ``uv run python examples/classification.py``
"""

from __future__ import annotations

from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

import qjax
from qjax.nn import bounded_q
from qjax.plots import CMAP, qcolors, save_figure, use_qjax_style

FIG_DIR = Path(__file__).parent / "figures"
NUM_CLASSES = 4
NOISE_LEVELS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)
SEEDS = (0, 1, 2, 3)
HIDDEN = 128
STEPS = 3000
LR = 3e-3  # Adam step size

# Learnable-q parameterization: q = bounded_q(q_raw, Q_MIN, Q_MAX) in (0.3, 1.3),
# spanning the robust (q < 1) and standard (q = 1) regimes. We start inside the
# robust regime (q ~ 0.5): starting near Shannon (q ~ 1) lets the network
# memorize the noise early, before q has annealed down.
Q_MIN, Q_MAX, Q_RAW_INIT = 0.3, 1.3, -1.4  # init q ~ 0.50

# (label, q_fixed, is_learnable) — q = 1 is the Shannon baseline; q < 1 are robust.
METHODS = (
    ("Shannon (q = 1)", 1.0, False),
    ("Tsallis q = 0.7", 0.7, False),
    ("Tsallis q = 0.4", 0.4, False),
    ("learnable q", 0.0, True),
)


def make_data(key: jax.Array, n: int):
    """Four isotropic Gaussian blobs at the corners of a square (overlapping)."""
    centers = jnp.array([[2.0, 2.0], [-2.0, 2.0], [-2.0, -2.0], [2.0, -2.0]])
    k_label, k_noise = jax.random.split(key)
    labels = jax.random.randint(k_label, (n,), 0, NUM_CLASSES)
    x = centers[labels] + 1.4 * jax.random.normal(k_noise, (n, 2))
    return x, labels


def corrupt_labels(key: jax.Array, labels: jnp.ndarray, eta: float) -> jnp.ndarray:
    """Reassign a fraction ``eta`` of labels to a uniformly random class."""
    k_mask, k_new = jax.random.split(key)
    flip = jax.random.bernoulli(k_mask, eta, labels.shape)
    random_labels = jax.random.randint(k_new, labels.shape, 0, NUM_CLASSES)
    return jnp.where(flip, random_labels, labels)


def init_params(key: jax.Array) -> dict:
    """Initialize a 2 -> HIDDEN -> HIDDEN -> NUM_CLASSES MLP (incl. ``q_raw``)."""
    k1, k2, k3 = jax.random.split(key, 3)
    return {
        "w1": jax.random.normal(k1, (2, HIDDEN)) * jnp.sqrt(2.0 / 2),
        "b1": jnp.zeros(HIDDEN),
        "w2": jax.random.normal(k2, (HIDDEN, HIDDEN)) * jnp.sqrt(2.0 / HIDDEN),
        "b2": jnp.zeros(HIDDEN),
        "w3": jax.random.normal(k3, (HIDDEN, NUM_CLASSES)) * jnp.sqrt(2.0 / HIDDEN),
        "b3": jnp.zeros(NUM_CLASSES),
        "q_raw": jnp.array(Q_RAW_INIT),  # only used when q is learnable
    }


def resolve_q(params: dict, q_fixed, is_learnable: bool):
    """Return the loss entropic index in use: a constant, or the learned one."""
    if is_learnable:
        return bounded_q(params["q_raw"], Q_MIN, Q_MAX)
    return q_fixed


def logits(params: dict, x: jnp.ndarray) -> jnp.ndarray:
    """Forward pass returning class logits."""
    h = jnp.tanh(x @ params["w1"] + params["b1"])
    h = jnp.tanh(h @ params["w2"] + params["b2"])
    return h @ params["w3"] + params["b3"]


@partial(jax.jit, static_argnames=("is_learnable",))
def train(params: dict, x, y_onehot, q_fixed, is_learnable: bool):
    """Adam optimization of the (possibly learnable-q) Tsallis cross-entropy.

    Returns the trained params and the per-step trajectory of the entropic
    index ``q`` (constant for the fixed-q methods).
    """
    b1, b2, eps = 0.9, 0.999, 1e-8
    m = jax.tree_util.tree_map(jnp.zeros_like, params)
    v = jax.tree_util.tree_map(jnp.zeros_like, params)

    def loss_fn(params):
        q = resolve_q(params, q_fixed, is_learnable)
        p = jnp.clip(jax.nn.softmax(logits(params, x), axis=-1), 1e-7, 1.0)
        return jnp.mean(qjax.tsallis_cross_entropy(p, y_onehot, q=q, axis=-1))

    def step(carry, t):
        params, m, v = carry
        grads = jax.grad(loss_fn)(params)
        m = jax.tree_util.tree_map(lambda m, g: b1 * m + (1 - b1) * g, m, grads)
        v = jax.tree_util.tree_map(lambda v, g: b2 * v + (1 - b2) * g * g, v, grads)
        bc1, bc2 = 1 - b1 ** (t + 1), 1 - b2 ** (t + 1)
        params = jax.tree_util.tree_map(
            lambda p, m, v: p - LR * (m / bc1) / (jnp.sqrt(v / bc2) + eps),
            params,
            m,
            v,
        )
        return (params, m, v), resolve_q(params, q_fixed, is_learnable)

    (params, _, _), q_hist = jax.lax.scan(step, (params, m, v), jnp.arange(STEPS))
    return params, q_hist


def accuracy(params: dict, x, y) -> float:
    """Fraction of correct argmax predictions."""
    return float(jnp.mean(jnp.argmax(logits(params, x), axis=-1) == y))


# --------------------------------------------------------------------------- #
# Extra experiment: decision boundaries across shapes, noise and losses.
#
# A compact 3-class classifier on two shapes (blobs, spiral) makes the robustness
# *visible*: from clean (no noise) to 40% label noise, the Shannon baseline
# (``q = 1``) carves spurious wrong-class islands around the mislabeled points
# while the bounded Tsallis loss (``q = 0.3``) keeps clean regions. The comparison
# is fair — within each shape both losses share the same init, data, noisy labels
# and optimizer; only ``q`` differs (and ``q = 1`` is exactly Shannon).
# --------------------------------------------------------------------------- #
BND_CLASSES, BND_HIDDEN, BND_STEPS, BND_LR = 3, (64, 64), 3400, 5e-3
BND_TEAL = "#168aad"  # brand accent framing the Tsallis (robust) columns
BND_COLS = (
    ("BGS", 1.0, 0.0),
    ("Tsallis", 0.3, 0.0),
    ("BGS", 1.0, 0.2),
    ("Tsallis", 0.3, 0.2),
    ("BGS", 1.0, 0.4),
    ("Tsallis", 0.3, 0.4),
)


def _blobs_shape(key, n):
    """``BND_CLASSES`` Gaussian blobs on a ring."""
    m = n // BND_CLASSES
    ang = jnp.pi / 2 + jnp.arange(BND_CLASSES) * (2 * jnp.pi / BND_CLASSES)
    centers = 1.7 * jnp.stack([jnp.cos(ang), jnp.sin(ang)], -1)
    y = jnp.repeat(jnp.arange(BND_CLASSES), m)
    return centers[y] + 0.62 * jax.random.normal(key, (BND_CLASSES * m, 2)), y


def _spiral_shape(key, n):
    """``BND_CLASSES`` interleaved spiral arms."""
    k1, k2 = jax.random.split(key)
    m = n // BND_CLASSES
    t = jnp.sqrt(jax.random.uniform(k1, (BND_CLASSES, m)))
    ang = t * 2 * jnp.pi * 0.85 + jnp.arange(BND_CLASSES)[:, None] * (2 * jnp.pi / BND_CLASSES)
    r = 0.35 + 1.2 * t
    x = jnp.stack([r * jnp.cos(ang), r * jnp.sin(ang)], -1).reshape(-1, 2)
    return x + 0.05 * jax.random.normal(k2, x.shape), jnp.repeat(jnp.arange(BND_CLASSES), m)


BND_SHAPES = (("blobs", _blobs_shape), ("spiral", _spiral_shape))


def _bnd_init(key):
    k1, k2, k3 = jax.random.split(key, 3)
    h1, h2 = BND_HIDDEN
    return {
        "w1": jax.random.normal(k1, (2, h1)) * jnp.sqrt(2.0 / 2),
        "b1": jnp.zeros(h1),
        "w2": jax.random.normal(k2, (h1, h2)) * jnp.sqrt(2.0 / h1),
        "b2": jnp.zeros(h2),
        "w3": jax.random.normal(k3, (h2, BND_CLASSES)) * jnp.sqrt(2.0 / h2),
        "b3": jnp.zeros(BND_CLASSES),
    }


def _bnd_logits(p, x):
    h = jnp.tanh(x @ p["w1"] + p["b1"])
    h = jnp.tanh(h @ p["w2"] + p["b2"])
    return h @ p["w3"] + p["b3"]


@jax.jit
def _bnd_train(params, x, y_oh, q):
    """Adam on the (fixed-``q``) Tsallis cross-entropy; returns final params."""
    b1, b2, eps = 0.9, 0.999, 1e-8
    m = jax.tree_util.tree_map(jnp.zeros_like, params)
    v = jax.tree_util.tree_map(jnp.zeros_like, params)

    def loss_fn(p):
        probs = jnp.clip(jax.nn.softmax(_bnd_logits(p, x), axis=-1), 1e-7, 1.0)
        return jnp.mean(qjax.tsallis_cross_entropy(probs, y_oh, q=q, axis=-1))

    def step(carry, t):
        p, m, v = carry
        gr = jax.grad(loss_fn)(p)
        m = jax.tree_util.tree_map(lambda a, b: b1 * a + (1 - b1) * b, m, gr)
        v = jax.tree_util.tree_map(lambda a, b: b2 * a + (1 - b2) * b * b, v, gr)
        bc1, bc2 = 1 - b1 ** (t + 1), 1 - b2 ** (t + 1)
        p = jax.tree_util.tree_map(
            lambda p, m, v: p - BND_LR * (m / bc1) / (jnp.sqrt(v / bc2) + eps), p, m, v
        )
        return (p, m, v), None

    (params, _, _), _ = jax.lax.scan(step, (params, m, v), jnp.arange(BND_STEPS))
    return params


def _bnd_flip(key, y, eta):
    """Flip a fraction ``eta`` of labels to a *different* class."""
    k1, k2 = jax.random.split(key)
    other = (y + 1 + jax.random.randint(k2, y.shape, 0, BND_CLASSES - 1)) % BND_CLASSES
    return jnp.where(jax.random.bernoulli(k1, eta, y.shape), other, y)


def decision_boundary_figure() -> None:
    """Render the shapes x (method x noise) decision-boundary grid."""
    use_qjax_style()
    cmap = ListedColormap(qcolors(BND_CLASSES))
    lim, gridn = 2.7, 200
    grid_axis = np.linspace(-lim, lim, gridn)
    xx, yy = np.meshgrid(grid_axis, grid_axis)
    grid = jnp.asarray(np.stack([xx.ravel(), yy.ravel()], axis=-1))

    fig, axes = plt.subplots(
        len(BND_SHAPES), len(BND_COLS), figsize=(15.6, 5.7), layout="constrained"
    )
    base = jax.random.PRNGKey(0)
    for i, (sname, make) in enumerate(BND_SHAPES):
        ks = jax.random.split(jax.random.fold_in(base, i), 4)
        x_tr, y0 = make(ks[0], 360)
        x_te, y_te = make(ks[1], 4500)
        mean, std = x_tr.mean(0), x_tr.std(0)
        x_tr = (x_tr - mean) / std * 1.25
        x_te = (x_te - mean) / std * 1.25
        params0 = _bnd_init(ks[3])  # shared init (fairness)
        noisy = {
            eta: _bnd_flip(jax.random.fold_in(ks[2], int(eta * 100)), y0, eta)
            for eta in {c[2] for c in BND_COLS}
        }  # shared noisy labels

        for j, (mlabel, q, eta) in enumerate(BND_COLS):
            y_tr = noisy[eta]
            trained = _bnd_train(params0, x_tr, jax.nn.one_hot(y_tr, BND_CLASSES), jnp.float32(q))
            acc = float(jnp.mean(jnp.argmax(_bnd_logits(trained, x_te), -1) == y_te))
            regions = np.asarray(jnp.argmax(_bnd_logits(trained, grid), -1)).reshape(xx.shape)
            print(f"{sname:<7} {mlabel:<8} eta={eta:.1f}  acc={acc:.3f}")

            ax = axes[i, j]
            ax.imshow(
                regions,
                extent=[-lim, lim, -lim, lim],
                origin="lower",
                cmap=cmap,
                vmin=-0.5,
                vmax=BND_CLASSES - 0.5,
                alpha=0.42,
                interpolation="bilinear",
            )
            ax.scatter(
                np.asarray(x_tr[:, 0]),
                np.asarray(x_tr[:, 1]),
                c=np.asarray(y_tr),
                cmap=cmap,
                vmin=-0.5,
                vmax=BND_CLASSES - 0.5,
                s=24,
                edgecolor="white",
                linewidth=0.5,
                alpha=0.95,
            )
            ax.set(xlim=(-lim, lim), ylim=(-lim, lim))
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect("equal")
            is_tsa = mlabel == "Tsallis"
            for sp in ax.spines.values():
                sp.set_visible(is_tsa)
                if is_tsa:
                    sp.set_color(BND_TEAL)
                    sp.set_linewidth(2.6)
            ax.text(
                0.05,
                0.95,
                f"{acc * 100:.0f}%",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=11,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.24", fc="white", ec="0.7", alpha=0.9),
            )
            if i == 0:
                ax.set_title(
                    f"{mlabel}\n$\\eta = {int(eta * 100)}\\%$",
                    fontsize=11.5,
                    color=BND_TEAL if is_tsa else "0.15",
                    fontweight="bold" if is_tsa else "normal",
                )
            if j == 0:
                ax.set_ylabel(f"{sname}\n({BND_CLASSES} classes)", fontsize=12.5)

    print(f"saved {save_figure(fig, FIG_DIR / 'classification_boundaries')}")


def main() -> None:
    use_qjax_style()
    base = jax.random.PRNGKey(0)
    labels = [label for label, _, _ in METHODS]

    results = {label: [] for label in labels}
    stderr = {label: [] for label in labels}
    learned_q = []  # final learned q, per noise level
    q_trajectories: dict[float, jnp.ndarray] = {}  # noise level -> q trajectory (seed 0)

    for eta in NOISE_LEVELS:
        per_method = {label: [] for label in labels}
        per_lq = []
        for seed in SEEDS:
            keys = jax.random.split(jax.random.fold_in(base, seed), 4)
            x_tr, y_tr_clean = make_data(keys[0], n=400)
            x_te, y_te = make_data(keys[1], n=2000)  # clean test set
            y_tr = corrupt_labels(keys[2], y_tr_clean, eta)
            y_tr_onehot = jax.nn.one_hot(y_tr, NUM_CLASSES)
            params0 = init_params(keys[3])
            for label, q_fixed, is_learnable in METHODS:
                qf = jnp.float32(q_fixed)
                trained, q_hist = train(params0, x_tr, y_tr_onehot, qf, is_learnable)
                per_method[label].append(accuracy(trained, x_te, y_te))
                if is_learnable:
                    per_lq.append(float(resolve_q(trained, jnp.float32(q_fixed), True)))
                    if seed == 0:
                        q_trajectories[eta] = q_hist
        for label in labels:
            accs = jnp.array(per_method[label])
            results[label].append(float(jnp.mean(accs)))
            stderr[label].append(float(jnp.std(accs) / jnp.sqrt(len(SEEDS))))
        learned_q.append(float(jnp.mean(jnp.array(per_lq))))
        summary = "  ".join(f"{lbl.split()[0]}:{results[lbl][-1]:.3f}" for lbl in labels)
        print(f"eta={eta:.1f}  {summary}  learned_q={learned_q[-1]:.2f}")

    # ---- Figure: (a) accuracy vs noise, (b) q learning dynamics, (c) the task ----
    fig, (ax_acc, ax_q, ax_data) = plt.subplots(1, 3, figsize=(15.0, 4.3), layout="constrained")
    colors = qcolors(len(METHODS))
    markers = ("o", "s", "D", "^")
    for label, color, marker in zip(labels, colors, markers, strict=False):
        mean, err = jnp.array(results[label]), jnp.array(stderr[label])
        ax_acc.plot(NOISE_LEVELS, mean, color=color, marker=marker, label=label)
        ax_acc.fill_between(NOISE_LEVELS, mean - err, mean + err, color=color, alpha=0.15)
    ax_acc.set(
        xlabel=r"label-noise rate $\eta$",
        ylabel="clean test accuracy",
        title="(a) robustness to label noise",
    )
    ax_acc.legend(loc="lower left")

    # (b) q descends from its init into the robust regime during training.
    shown = (NOISE_LEVELS[0], NOISE_LEVELS[len(NOISE_LEVELS) // 2], NOISE_LEVELS[-1])
    for eta, color in zip(shown, qcolors(len(shown)), strict=False):
        ax_q.plot(q_trajectories[eta], color=color, label=rf"$\eta={eta:g}$")
    ax_q.axhline(0.4, color="0.4", ls=":", lw=1.0, label="Tsallis $q=0.4$")
    ax_q.set(
        xlabel="training step",
        ylabel="learned entropic index $q$",
        title="(b) $q$ descends to the robust regime",
    )
    ax_q.legend(loc="best", fontsize=8)

    x_demo, y_demo = make_data(jax.random.PRNGKey(7), n=800)
    sc = ax_data.scatter(
        x_demo[:, 0], x_demo[:, 1], c=y_demo, cmap=CMAP, s=10, edgecolor="none", alpha=0.8
    )
    ax_data.set(
        xlabel=r"$x_1$", ylabel=r"$x_2$", title=f"(c) task: {NUM_CLASSES} overlapping classes"
    )
    ax_data.set_aspect("equal")
    cbar = fig.colorbar(sc, ax=ax_data, ticks=range(NUM_CLASSES), fraction=0.046)
    cbar.set_label("class")

    print(f"saved {save_figure(fig, FIG_DIR / 'classification')}")

    # extra experiment: decision boundaries across shapes, noise and losses
    decision_boundary_figure()


if __name__ == "__main__":
    main()
