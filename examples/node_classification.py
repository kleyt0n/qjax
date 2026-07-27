"""Node classification under label noise: Tsallis GNN vs. the Shannon baseline.

We train the *same* 2-layer Graph Convolutional Network (GCN) on a synthetic
graph — a stochastic block model whose communities *are* the classes — under an
increasing fraction of corrupted training labels, and vary a single knob: the
entropic index ``q`` of the ``tsallis_cross_entropy`` loss on the softmax output.

- ``q = 1`` is *exactly* the **Shannon baseline** (standard cross-entropy). The
  loss ``-log p_c`` is unbounded, so a confidently mislabeled node produces an
  arbitrarily large gradient and the GCN propagates that error across the graph.
- **learnable Tsallis ``q``**: instead of fixing ``q`` we make it a trainable
  parameter optimized jointly with the network (``qjax.nn.bounded_q``).
  The bounded Tsallis loss ``-ln_q p_c = (1 - p_c^{1-q})/(1-q)`` (Zhang & Sabuncu,
  2018) has a gradient that saturates on hard/mislabeled nodes, so minimizing it
  over the noisy training set drives ``q`` *down into the robust regime* on its
  own — no grid search — and clean-set accuracy degrades far more gracefully.

Both methods share the same graph, init, noisy labels and optimizer; only ``q``
differs. Each configuration is averaged over several seeds and evaluated on a
*clean* held-out set of test nodes (transductive setting).

Run with: ``uv run python examples/node_classification.py``
"""

from __future__ import annotations

from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D

import qjax
from qjax.nn import bounded_q
from qjax.plots import qcolors, save_figure, use_qjax_style

FIG_DIR = Path(__file__).parent / "figures"

NUM_CLASSES = 4
N_NODES = 600
FEAT_DIM = NUM_CLASSES  # weak per-node signal; the graph carries the rest
P_IN, P_OUT = 0.055, 0.004  # SBM edge probs: intra-community >> inter-community
FEAT_NOISE = 1.6  # so features alone are *not* separable — structure must help
TRAIN_FRAC = 0.25

NOISE_LEVELS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)
SEEDS = (0, 1, 2, 3)
HIDDEN = 32
STEPS = 400
LR = 1e-2  # Adam step size

# Learnable-q parameterization: q = bounded_q(q_raw, Q_MIN, Q_MAX) in (0.3, 1.3),
# spanning the robust (q < 1) and standard (q = 1) regimes. We start inside the
# robust regime (q ~ 0.5) so the GCN does not memorize noise before q anneals down.
Q_MIN, Q_MAX, Q_RAW_INIT = 0.3, 1.3, -1.4  # init q ~ 0.50

# (label, q_fixed, is_learnable) — q = 1 is the Shannon baseline.
METHODS = (
    ("Shannon (q = 1)", 1.0, False),
    ("learnable Tsallis q", 0.0, True),
)

VIZ_ETA = 0.4  # label-noise rate shown in the graph panels (c, d)
N_VIZ = 200  # nodes drawn in the graph panels (a legible induced subgraph)


def make_graph(key: jax.Array):
    """Stochastic block model: ``NUM_CLASSES`` communities that *are* the classes.

    Returns node features ``X``, a symmetric 0/1 adjacency ``A`` (no self-loops),
    and the integer community labels.
    """
    k_label, k_edge, k_feat = jax.random.split(key, 3)
    labels = jax.random.randint(k_label, (N_NODES,), 0, NUM_CLASSES)

    # Edge probability is P_IN within a community and P_OUT across communities.
    same = labels[:, None] == labels[None, :]
    probs = jnp.where(same, P_IN, P_OUT)
    draw = jax.random.bernoulli(k_edge, probs)
    upper = jnp.triu(draw, k=1)  # keep upper triangle, drop the diagonal
    adj = (upper | upper.T).astype(jnp.float32)  # symmetric, zero diagonal

    # Weak, noisy per-node features: a faint class prototype plus large noise.
    prototype = jnp.eye(NUM_CLASSES, FEAT_DIM)
    feats = prototype[labels] + FEAT_NOISE * jax.random.normal(k_feat, (N_NODES, FEAT_DIM))
    return feats, adj, labels


def normalize_adj(adj: jnp.ndarray) -> jnp.ndarray:
    """Symmetric-normalized propagation matrix ``D^{-1/2} (A + I) D^{-1/2}``."""
    a = adj + jnp.eye(adj.shape[0])
    d_inv_sqrt = jax.lax.rsqrt(jnp.sum(a, axis=1))
    return d_inv_sqrt[:, None] * a * d_inv_sqrt[None, :]


def split_mask(key: jax.Array) -> jnp.ndarray:
    """Boolean train mask selecting a ``TRAIN_FRAC`` fraction of nodes."""
    return jax.random.bernoulli(key, TRAIN_FRAC, (N_NODES,))


def corrupt_labels(key: jax.Array, labels: jnp.ndarray, eta: float) -> jnp.ndarray:
    """Reassign a fraction ``eta`` of labels to a uniformly random class."""
    k_mask, k_new = jax.random.split(key)
    flip = jax.random.bernoulli(k_mask, eta, labels.shape)
    random_labels = jax.random.randint(k_new, labels.shape, 0, NUM_CLASSES)
    return jnp.where(flip, random_labels, labels)


def init_params(key: jax.Array) -> dict:
    """Initialize a FEAT_DIM -> HIDDEN -> NUM_CLASSES GCN (incl. ``q_raw``)."""
    k1, k2 = jax.random.split(key)
    return {
        "w1": jax.random.normal(k1, (FEAT_DIM, HIDDEN)) * jnp.sqrt(2.0 / FEAT_DIM),
        "w2": jax.random.normal(k2, (HIDDEN, NUM_CLASSES)) * jnp.sqrt(2.0 / HIDDEN),
        "q_raw": jnp.array(Q_RAW_INIT),  # only used when q is learnable
    }


def resolve_q(params: dict, q_fixed, is_learnable: bool):
    """Return the loss entropic index in use: a constant, or the learned one."""
    if is_learnable:
        return bounded_q(params["q_raw"], Q_MIN, Q_MAX)
    return q_fixed


def gcn_logits(params: dict, x: jnp.ndarray, a_hat: jnp.ndarray) -> jnp.ndarray:
    """Two-layer GCN forward pass returning per-node class logits."""
    h = jax.nn.relu(a_hat @ (x @ params["w1"]))
    return a_hat @ (h @ params["w2"])


@partial(jax.jit, static_argnames=("is_learnable",))
def train(params: dict, x, a_hat, y_onehot, train_mask, q_fixed, is_learnable: bool):
    """Adam optimization of the (possibly learnable-q) Tsallis cross-entropy.

    The loss is masked to the training nodes. Returns the trained params and the
    per-step trajectory of the entropic index ``q`` (constant for fixed-q).
    """
    b1, b2, eps = 0.9, 0.999, 1e-8
    m = jax.tree_util.tree_map(jnp.zeros_like, params)
    v = jax.tree_util.tree_map(jnp.zeros_like, params)
    w = train_mask.astype(jnp.float32)

    def loss_fn(params):
        q = resolve_q(params, q_fixed, is_learnable)
        p = jnp.clip(jax.nn.softmax(gcn_logits(params, x, a_hat), axis=-1), 1e-7, 1.0)
        ce = qjax.tsallis_cross_entropy(p, y_onehot, q=q, axis=-1)
        return jnp.sum(ce * w) / jnp.sum(w)  # mean over training nodes only

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


def predictions(params: dict, x, a_hat) -> jnp.ndarray:
    """Per-node argmax class predictions."""
    return jnp.argmax(gcn_logits(params, x, a_hat), axis=-1)


def masked_accuracy(preds, labels, mask) -> float:
    """Fraction of correct predictions over the nodes selected by ``mask``."""
    return float(jnp.sum((preds == labels) & mask) / jnp.sum(mask))


def make_seed_problem(seed: int, eta: float):
    """Reproducibly build the graph, masks and noisy labels for one seed/noise."""
    keys = jax.random.split(jax.random.fold_in(jax.random.PRNGKey(0), seed), 4)
    x, adj, y_clean = make_graph(keys[0])
    a_hat = normalize_adj(adj)
    train_mask = split_mask(keys[1])
    y_noisy = jnp.where(train_mask, corrupt_labels(keys[2], y_clean, eta), y_clean)
    return x, adj, a_hat, y_clean, y_noisy, train_mask, keys[3]


def spring_layout(adj: np.ndarray, iters: int = 140, seed: int = 1) -> np.ndarray:
    """Fruchterman--Reingold force-directed layout (deterministic, NumPy-only).

    Connected nodes attract and all nodes repel, so the communities settle into
    well-separated blobs — a far cleaner picture than a Laplacian eigenmap, which
    degenerates on sparse stochastic block models. Run on the *full* graph so the
    structure is intact; callers draw whatever subset they like.
    """
    n = adj.shape[0]
    pos = np.random.default_rng(seed).normal(size=(n, 2))
    k = np.sqrt(1.0 / n)  # natural edge length
    t = 0.1  # temperature: caps per-step displacement, cooled each iteration
    for _ in range(iters):
        delta = pos[:, None, :] - pos[None, :, :]
        dist = np.sqrt((delta**2).sum(-1)) + 1e-9
        unit = delta / dist[..., None]
        repulse = ((k * k / dist)[..., None] * unit).sum(1)
        attract = ((adj * dist / k)[..., None] * unit).sum(1)
        disp = repulse - attract
        length = np.sqrt((disp**2).sum(-1)) + 1e-9
        pos += disp / length[..., None] * np.minimum(length, t)[..., None]
        t *= 0.97
    return pos


def draw_graph(ax, coords, adj_sub, pred, true, test_mask, cmap, title, acc, lim):
    """Draw one induced subgraph: nodes by predicted class, wrong test nodes ringed.

    ``lim`` is a shared ``(xmin, xmax, ymin, ymax)`` box so the Shannon and Tsallis
    panels sit on *identical* axes and are directly comparable.
    """
    ii, jj = np.nonzero(np.triu(adj_sub, k=1))
    segments = [(coords[a], coords[b]) for a, b in zip(ii, jj, strict=False)]
    ax.add_collection(LineCollection(segments, colors="0.6", linewidths=0.4, alpha=0.35, zorder=1))
    wrong = (pred != true) & test_mask  # misclassified held-out nodes
    ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c=pred,
        cmap=cmap,
        vmin=-0.5,
        vmax=NUM_CLASSES - 0.5,
        s=44,
        edgecolor="white",
        linewidth=0.6,
        zorder=3,
    )
    ax.scatter(
        coords[wrong, 0],
        coords[wrong, 1],
        facecolors="none",
        edgecolors="#d62728",
        s=132,
        linewidths=1.9,
        zorder=4,
    )
    ax.set_title(title, pad=8)
    ax.set(xlim=lim[:2], ylim=lim[2:])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_color("0.8")
        sp.set_linewidth(0.8)
    ax.text(
        0.035,
        0.965,
        f"acc {acc * 100:.0f}%",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=11.5,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", alpha=0.92),
    )


def main() -> None:
    use_qjax_style()
    labels = [label for label, _, _ in METHODS]

    results = {label: [] for label in labels}
    stderr = {label: [] for label in labels}
    learned_q = []  # mean final learned q, per noise level
    q_trajectories: dict[float, jnp.ndarray] = {}  # noise level -> q trajectory (seed 0)

    for eta in NOISE_LEVELS:
        per_method = {label: [] for label in labels}
        per_lq = []
        for seed in SEEDS:
            x, _, a_hat, y_clean, y_noisy, train_mask, k_init = make_seed_problem(seed, eta)
            test_mask = ~train_mask
            y_onehot = jax.nn.one_hot(y_noisy, NUM_CLASSES)
            params0 = init_params(k_init)
            for label, q_fixed, is_learnable in METHODS:
                qf = jnp.float32(q_fixed)
                trained, q_hist = train(params0, x, a_hat, y_onehot, train_mask, qf, is_learnable)
                preds = predictions(trained, x, a_hat)
                per_method[label].append(masked_accuracy(preds, y_clean, test_mask))
                if is_learnable:
                    per_lq.append(float(resolve_q(trained, qf, True)))
                    if seed == 0:
                        q_trajectories[eta] = q_hist
        for label in labels:
            accs = jnp.array(per_method[label])
            results[label].append(float(jnp.mean(accs)))
            stderr[label].append(float(jnp.std(accs) / jnp.sqrt(len(SEEDS))))
        learned_q.append(float(jnp.mean(jnp.array(per_lq))))
        summary = "  ".join(f"{lbl.split()[0]}:{results[lbl][-1]:.3f}" for lbl in labels)
        print(f"eta={eta:.1f}  {summary}  learned_q={learned_q[-1]:.2f}")

    # The results are saved as two standalone, publication-ready figures: the
    # metric curves, and the graph visualizations.
    method_colors = qcolors(len(METHODS))
    markers = ("o", "D")

    # --- Figure 1: metrics. (a) accuracy vs noise, (b) the q descent. ---
    fig, (ax_acc, ax_q) = plt.subplots(1, 2, figsize=(11.0, 4.5), layout="constrained")
    for label, color, marker in zip(labels, method_colors, markers, strict=False):
        mean, err = jnp.array(results[label]), jnp.array(stderr[label])
        ax_acc.plot(
            NOISE_LEVELS, mean, color=color, marker=marker, markersize=6, label=label, zorder=3
        )
        ax_acc.fill_between(
            NOISE_LEVELS, mean - err, mean + err, color=color, alpha=0.16, linewidth=0
        )
    ax_acc.set(
        xlabel=r"label-noise rate $\eta$",
        ylabel="clean test accuracy",
        title="(a) robustness to label noise",
        xlim=(NOISE_LEVELS[0], NOISE_LEVELS[-1]),
    )
    ax_acc.margins(y=0.08)
    ax_acc.legend(loc="lower left", title="GNN loss")

    shown = (NOISE_LEVELS[0], NOISE_LEVELS[len(NOISE_LEVELS) // 2], NOISE_LEVELS[-1])
    for eta, color in zip(shown, qcolors(len(shown)), strict=False):
        ax_q.plot(q_trajectories[eta], color=color, label=rf"$\eta = {eta:g}$")
    ax_q.axhline(1.0, color="0.45", ls=":", lw=1.1)
    ax_q.text(
        0.012 * STEPS, 1.0, "Shannon $q = 1$", color="0.4", va="bottom", ha="left", fontsize=8.5
    )
    ax_q.set(
        xlabel="training step",
        ylabel="learned entropic index $q$",
        title=r"(b) $q$ descends to the robust regime",
        xlim=(0, STEPS),
        ylim=(Q_MIN, 1.06),
    )
    ax_q.legend(loc="upper right", title="label noise", ncol=1)
    print(f"saved {save_figure(fig, FIG_DIR / 'node_classification_metrics')}")

    # --- Figure 2: the same graph at high noise, colored by each GNN's predictions. ---
    cmap = ListedColormap(qcolors(NUM_CLASSES))
    x, adj, a_hat, y_clean, y_noisy, train_mask, k_init = make_seed_problem(0, VIZ_ETA)
    test_mask = ~train_mask
    y_onehot = jax.nn.one_hot(y_noisy, NUM_CLASSES)
    params0 = init_params(k_init)

    coords_full = spring_layout(np.asarray(adj))  # lay out the full graph
    sub = jax.random.permutation(jax.random.PRNGKey(1), N_NODES)[:N_VIZ]
    sub_np = np.asarray(sub)
    adj_sub = np.asarray(adj)[np.ix_(sub_np, sub_np)]
    coords = coords_full[sub_np]
    true_sub = np.asarray(y_clean)[sub_np]
    test_sub = np.asarray(test_mask)[sub_np]

    # Shared, slightly padded axes box so the two panels are directly comparable.
    pad = 0.06 * (coords.max(0) - coords.min(0))
    lo, hi = coords.min(0) - pad, coords.max(0) + pad
    lim = (lo[0], hi[0], lo[1], hi[1])

    fig, (ax_shannon, ax_tsallis) = plt.subplots(1, 2, figsize=(10.4, 5.4), layout="constrained")
    panels = ((ax_shannon, "(a) Shannon GNN"), (ax_tsallis, "(b) learnable-Tsallis GNN"))
    for (ax, title), (_label, q_fixed, is_learnable) in zip(panels, METHODS, strict=False):
        trained, _ = train(
            params0, x, a_hat, y_onehot, train_mask, jnp.float32(q_fixed), is_learnable
        )
        preds = predictions(trained, x, a_hat)
        acc = masked_accuracy(preds, y_clean, test_mask)
        draw_graph(
            ax,
            coords,
            adj_sub,
            np.asarray(preds)[sub_np],
            true_sub,
            test_sub,
            cmap,
            rf"{title}  ($\eta = {VIZ_ETA:g}$)",
            acc,
            lim,
        )

    # Shared figure legend: the node colors (predicted class) and the red ring.
    class_colors = qcolors(NUM_CLASSES)
    handles = [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            markersize=8,
            markerfacecolor=c,
            markeredgecolor="white",
            markeredgewidth=0.6,
            label=f"class {i}",
        )
        for i, c in enumerate(class_colors)
    ]
    handles.append(
        Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            markersize=10,
            markerfacecolor="none",
            markeredgecolor="#d62728",
            markeredgewidth=1.8,
            label="misclassified test node",
        )
    )
    fig.legend(
        handles=handles,
        loc="outside lower center",
        ncol=len(handles),
        frameon=False,
        fontsize=9.5,
        columnspacing=1.5,
        handletextpad=0.3,
    )
    print(f"saved {save_figure(fig, FIG_DIR / 'node_classification_graphs')}")


if __name__ == "__main__":
    main()
