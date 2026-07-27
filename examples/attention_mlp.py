"""Sparse self-attention with Tsallis entmax: finding signal among distractors.

A single-head **attention-pooling classifier** is trained end-to-end on a
well-posed sequence task and we vary one knob — the entropic index ``q`` of the
attention map ``tsallis_entmax(scores, q)``:

- ``q = 1`` is ordinary **softmax** attention (dense): every position, including
  pure-noise distractors, receives a non-zero weight.
- ``q > 1`` is **entmax / sparsemax** attention (sparse): irrelevant positions
  are assigned *exactly* zero weight.
- **learnable ``q``**: instead of fixing ``q`` we make it a trainable parameter
  (``qjax.nn.bounded_q``) and let gradient descent discover the
  attention sparsity that best fits the task — the library's central thesis that
  ``q`` is just another differentiable parameter.

Task. Each length-``L`` sequence contains ``K`` informative tokens that carry a
shared "signal" direction plus the sequence's class prototype; the remaining
``L - K`` tokens are pure Gaussian noise. The model must learn a query that
locates the informative tokens and pools their class evidence. As ``L`` grows the
distractor count grows, so dense softmax leaks attention onto noise and the
pooled representation is diluted — whereas entmax zeros the noise out.

Run with: ``uv run python examples/attention_mlp.py``
"""

from __future__ import annotations

from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

import qjax
from qjax.nn import bounded_q
from qjax.plots import CMAP, qcolors, save_figure, use_qjax_style

FIG_DIR = Path(__file__).parent / "figures"

D_MODEL = 16  # token / model dimension
NUM_CLASSES = 4  # class prototypes live on basis axes 0..C-1
K_SIGNAL = 3  # informative tokens per sequence
PROTO_SCALE = 2.5  # magnitude of the class-prototype component
MARKER_SCALE = 2.5  # magnitude of the shared "informative" marker direction
NOISE_STD = 1.0  # per-feature Gaussian noise
SEQ_LENGTHS = (4, 8, 16, 32)
VIZ_LENGTH = 16  # sequence length used for the attention-map panels
SEEDS = (0, 1, 2)
STEPS = 5000
LR = 5e-3
ENTMAX_ITERS = 25  # bisection steps inside entmax during training

# Learnable-q parameterization: q = bounded_q(q_raw, Q_MIN, Q_MAX) in (1.1, 2.8),
# kept strictly above 1 to avoid the softmax singularity of the entmax solver.
Q_MIN, Q_MAX, Q_RAW_INIT = 1.1, 2.8, -1.0

# (label, q_fixed, is_learnable) — q = 1 is softmax; learnable q is trained.
METHODS = (
    ("softmax (q = 1)", 1.0, False),
    ("entmax (q = 1.5)", 1.5, False),
    ("sparsemax (q = 2)", 2.0, False),
    ("learnable q", 0.0, True),
)


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def make_batch(key: jax.Array, length: int, n: int):
    """Sample ``n`` sequences; return tokens, labels, and the informative mask.

    Class prototypes sit on basis axes ``0..C-1`` and the shared informative
    marker on axis ``C``; the ``K`` informative tokens carry both (plus noise),
    distractors carry noise only.
    """
    k_label, k_pos, k_noise = jax.random.split(key, 3)
    labels = jax.random.randint(k_label, (n,), 0, NUM_CLASSES)

    # Pick K informative positions per sequence via top-K of a random score.
    rank = jax.random.uniform(k_pos, (n, length))
    kth = jnp.sort(rank, axis=-1)[:, -K_SIGNAL]
    info_mask = rank >= kth[:, None]  # (n, L), exactly K True per row

    prototypes = jnp.eye(NUM_CLASSES, D_MODEL)  # unit vectors on axes 0..C-1
    marker = jax.nn.one_hot(NUM_CLASSES, D_MODEL)  # axis C, orthogonal to all
    signal = PROTO_SCALE * prototypes[labels][:, None, :] + MARKER_SCALE * marker
    x = NOISE_STD * jax.random.normal(k_noise, (n, length, D_MODEL))
    x = x + info_mask[..., None] * signal
    return x, labels, info_mask


# --------------------------------------------------------------------------- #
# Single-head attention-pooling classifier
# --------------------------------------------------------------------------- #
def init_params(key: jax.Array) -> dict:
    """Initialize the attention-pooling classifier parameters (incl. ``q_raw``)."""
    k_k, k_v, k_q, k_o = jax.random.split(key, 4)
    scale = 1.0 / jnp.sqrt(D_MODEL)
    return {
        "w_key": jax.random.normal(k_k, (D_MODEL, D_MODEL)) * scale,
        "w_val": jax.random.normal(k_v, (D_MODEL, D_MODEL)) * scale,
        "query": jax.random.normal(k_q, (D_MODEL,)) * scale,
        "w_out": jax.random.normal(k_o, (D_MODEL, NUM_CLASSES)) * scale,
        "b_out": jnp.zeros(NUM_CLASSES),
        "q_raw": jnp.array(Q_RAW_INIT),  # only used when q is learnable
    }


def resolve_q(params: dict, q_fixed, is_learnable: bool):
    """Return the entropic index in use: a constant, or the learned one."""
    if is_learnable:
        return bounded_q(params["q_raw"], Q_MIN, Q_MAX)
    return q_fixed


def forward(params: dict, x: jnp.ndarray, q):
    """Return class logits and the attention weights for a batch of sequences."""
    keys = x @ params["w_key"]
    values = x @ params["w_val"]
    scores = keys @ params["query"] / jnp.sqrt(D_MODEL)  # (n, L)
    attn = qjax.tsallis_entmax(scores, q=q, axis=-1, num_iters=ENTMAX_ITERS)
    context = jnp.einsum("nl,nld->nd", attn, values)  # (n, D)
    logits = context @ params["w_out"] + params["b_out"]  # (n, C)
    return logits, attn


@partial(jax.jit, static_argnames=("is_learnable",))
def train(params: dict, x, y_onehot, q_fixed, is_learnable: bool):
    """Adam optimization of the cross-entropy loss; also logs the q trajectory."""
    b1, b2, eps = 0.9, 0.999, 1e-8
    m = jax.tree_util.tree_map(jnp.zeros_like, params)
    v = jax.tree_util.tree_map(jnp.zeros_like, params)

    def loss_fn(params):
        q = resolve_q(params, q_fixed, is_learnable)
        logits, _ = forward(params, x, q)
        return -jnp.mean(jnp.sum(y_onehot * jax.nn.log_softmax(logits, axis=-1), axis=-1))

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


def evaluate(params: dict, x, y, info_mask, q):
    """Return (accuracy, mean attention mass placed on informative tokens)."""
    logits, attn = forward(params, x, q)
    acc = jnp.mean(jnp.argmax(logits, axis=-1) == y)
    info_mass = jnp.mean(jnp.sum(attn * info_mask, axis=-1))
    return float(acc), float(info_mass)


# --------------------------------------------------------------------------- #
# Experiment + figure
# --------------------------------------------------------------------------- #
def attention_map_panel(ax, params, q, title: str):
    """Plot learned attention (examples x positions) with true signal marked."""
    x, _, info_mask = make_batch(jax.random.PRNGKey(99), VIZ_LENGTH, n=18)
    _, attn = forward(params, x, q)
    order = jnp.argsort(jnp.argmax(info_mask, axis=-1))  # group similar layouts
    attn, info_mask = attn[order], info_mask[order]

    im = ax.imshow(attn, cmap=CMAP, vmin=0.0, vmax=1.0, aspect="auto")
    rows, cols = jnp.nonzero(info_mask)
    ax.scatter(  # outline the ground-truth informative positions
        cols, rows, s=22, facecolors="none", edgecolors="white", linewidths=0.8
    )
    ax.set(title=title, xlabel="sequence position", ylabel="example")
    ax.grid(False)
    return im


def main() -> None:
    use_qjax_style()
    base = jax.random.PRNGKey(0)
    labels = [label for label, _, _ in METHODS]

    acc = {label: [] for label in labels}
    acc_se = {label: [] for label in labels}
    mass = {label: [] for label in labels}
    learned_q = []  # mean learned q per sequence length
    q_trajectories = {}  # length -> learned q trajectory (seed 0)
    viz = {}  # label -> (params, resolved_q) at VIZ_LENGTH, seed 0

    for length in SEQ_LENGTHS:
        per_acc = {label: [] for label in labels}
        per_mass = {label: [] for label in labels}
        per_lq = []
        for seed in SEEDS:
            keys = jax.random.split(jax.random.fold_in(base, seed), 3)
            x_tr, y_tr, _ = make_batch(keys[0], length, n=512)
            x_te, y_te, m_te = make_batch(keys[1], length, n=2000)
            y_tr_oh = jax.nn.one_hot(y_tr, NUM_CLASSES)
            params0 = init_params(keys[2])
            for label, q_fixed, is_learnable in METHODS:
                trained, q_hist = train(params0, x_tr, y_tr_oh, jnp.float32(q_fixed), is_learnable)
                q_used = resolve_q(trained, jnp.float32(q_fixed), is_learnable)
                a, mm = evaluate(trained, x_te, y_te, m_te, q_used)
                per_acc[label].append(a)
                per_mass[label].append(mm)
                if is_learnable:
                    per_lq.append(float(q_used))
                    if seed == 0:
                        q_trajectories[length] = q_hist
                if length == VIZ_LENGTH and seed == 0:
                    viz[label] = (trained, float(q_used))
        for label in labels:
            a = jnp.array(per_acc[label])
            acc[label].append(float(jnp.mean(a)))
            acc_se[label].append(float(jnp.std(a) / jnp.sqrt(len(SEEDS))))
            mass[label].append(float(jnp.mean(jnp.array(per_mass[label]))))
        learned_q.append(float(jnp.mean(jnp.array(per_lq))))
        best = max(labels, key=lambda label: acc[label][-1])
        print(f"L={length:>2}  best={best} ({acc[best][-1]:.3f})  learned_q={learned_q[-1]:.2f}")

    # ---- Figure: 2x3 research panel ----
    fig, axes = plt.subplots(2, 3, figsize=(15.0, 8.6), layout="constrained")
    (ax_acc, ax_mass, ax_q), (ax_soft, ax_sparse, ax_learn) = axes
    colors = qcolors(len(METHODS))
    markers = ("o", "s", "D", "^")

    for label, color, marker in zip(labels, colors, markers, strict=False):
        mean, se = jnp.array(acc[label]), jnp.array(acc_se[label])
        ax_acc.plot(SEQ_LENGTHS, mean, color=color, marker=marker, label=label)
        ax_acc.fill_between(SEQ_LENGTHS, mean - se, mean + se, color=color, alpha=0.15)
        ax_mass.plot(SEQ_LENGTHS, mass[label], color=color, marker=marker, label=label)
    ax_acc.set(
        xlabel="sequence length $L$ (distractor count grows)",
        ylabel="clean test accuracy",
        title="(a) accuracy vs. sequence length",
    )
    ax_acc.set_xscale("log", base=2)
    ax_acc.legend(loc="lower left")
    ax_mass.set(
        xlabel="sequence length $L$",
        ylabel="attention mass on informative tokens",
        title="(b) attention focus on signal",
    )
    ax_mass.set_xscale("log", base=2)
    ax_mass.set_ylim(0.0, 1.02)

    # (c) learned-q trajectories during training, one per sequence length.
    for length, color in zip(SEQ_LENGTHS, qcolors(len(SEQ_LENGTHS)), strict=False):
        ax_q.plot(q_trajectories[length], color=color, label=f"$L={length}$")
    ax_q.axhline(2.0, color="0.4", ls=":", lw=1.0, label="sparsemax ($q=2$)")
    ax_q.set(
        xlabel="training step",
        ylabel="learned entropic index $q$",
        title="(c) $q$ learned by gradient descent",
    )
    ax_q.legend(loc="best", ncol=2)

    soft_params, soft_q = viz["softmax (q = 1)"]
    sparse_params, sparse_q = viz["sparsemax (q = 2)"]
    learn_params, learn_q = viz["learnable q"]
    attention_map_panel(ax_soft, soft_params, jnp.float32(soft_q), "(d) softmax ($q=1$)")
    attention_map_panel(ax_sparse, sparse_params, jnp.float32(sparse_q), "(e) sparsemax ($q=2$)")
    im = attention_map_panel(
        ax_learn, learn_params, jnp.float32(learn_q), f"(f) learnable ($q={learn_q:.2f}$)"
    )
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.015, pad=0.01)
    cbar.set_label("attention weight")
    fig.suptitle(
        "Tsallis-entmax self-attention: a learnable $q$ recovers sparse, signal-focused attention"
    )

    print(f"saved {save_figure(fig, FIG_DIR / 'attention_mlp')}")


if __name__ == "__main__":
    main()
