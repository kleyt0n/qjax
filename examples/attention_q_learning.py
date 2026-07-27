"""Watch the entropic index ``q`` being learned inside an attention mechanism.

This is the dynamic companion to ``attention_mlp.py``. We train a single-head
attention-pooling classifier whose attention map is ``tsallis_entmax(scores, q)``
and make ``q`` itself a trained parameter. As gradient descent raises ``q`` from
its initialization toward sparsemax (``q ≈ 2``), the attention sharpens from a
diffuse cloud onto the few informative tokens.

The script renders a GIF (``examples/figures/attention_q_learning.gif``) with two
synchronized panels — the ``q``-learning curve (with a moving marker) and the
attention map recomputed at each checkpoint — plus a final-frame PDF.

Run with: ``uv run python examples/attention_q_learning.py``
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

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
SEQ_LENGTH = 16  # sequence length
N_TRAIN = 512
N_VIZ = 18  # sequences shown in the attention-map panel
STEPS = 4000
LR = 5e-3
ENTMAX_ITERS = 25  # bisection steps inside entmax
NUM_FRAMES = 140  # animation frames (front-loaded over training)
GIF_DPI = 150  # render resolution of the animation

# Learnable-q parameterization: q = bounded_q(q_raw, Q_MIN, Q_MAX) in (1.1, 2.8),
# kept strictly above 1 to avoid the softmax singularity of the entmax solver.
Q_MIN, Q_MAX, Q_RAW_INIT = 1.1, 2.8, -1.0


# --------------------------------------------------------------------------- #
# Data + model (shared with attention_mlp.py)
# --------------------------------------------------------------------------- #
def make_batch(key: jax.Array, length: int, n: int):
    """Sample ``n`` sequences; return tokens, labels, and the informative mask.

    Class prototypes sit on basis axes ``0..C-1`` and the shared informative
    marker on axis ``C``; the ``K`` informative tokens carry both (plus noise),
    distractors carry noise only.
    """
    k_label, k_pos, k_noise = jax.random.split(key, 3)
    labels = jax.random.randint(k_label, (n,), 0, NUM_CLASSES)

    rank = jax.random.uniform(k_pos, (n, length))
    kth = jnp.sort(rank, axis=-1)[:, -K_SIGNAL]
    info_mask = rank >= kth[:, None]  # (n, L), exactly K True per row

    prototypes = jnp.eye(NUM_CLASSES, D_MODEL)
    marker = jax.nn.one_hot(NUM_CLASSES, D_MODEL)
    signal = PROTO_SCALE * prototypes[labels][:, None, :] + MARKER_SCALE * marker
    x = NOISE_STD * jax.random.normal(k_noise, (n, length, D_MODEL))
    x = x + info_mask[..., None] * signal
    return x, labels, info_mask


def init_params(key: jax.Array) -> dict:
    """Initialize the attention-pooling classifier (including ``q_raw``)."""
    k_k, k_v, k_q, k_o = jax.random.split(key, 4)
    scale = 1.0 / jnp.sqrt(D_MODEL)
    return {
        "w_key": jax.random.normal(k_k, (D_MODEL, D_MODEL)) * scale,
        "w_val": jax.random.normal(k_v, (D_MODEL, D_MODEL)) * scale,
        "query": jax.random.normal(k_q, (D_MODEL,)) * scale,
        "w_out": jax.random.normal(k_o, (D_MODEL, NUM_CLASSES)) * scale,
        "b_out": jnp.zeros(NUM_CLASSES),
        "q_raw": jnp.array(Q_RAW_INIT),
    }


def learned_q(params: dict):
    """The entropic index implied by the current parameters."""
    return bounded_q(params["q_raw"], Q_MIN, Q_MAX)


def forward(params: dict, x: jnp.ndarray, q):
    """Return class logits and the attention weights for a batch of sequences."""
    keys = x @ params["w_key"]
    values = x @ params["w_val"]
    scores = keys @ params["query"] / jnp.sqrt(D_MODEL)
    attn = qjax.tsallis_entmax(scores, q=q, axis=-1, num_iters=ENTMAX_ITERS)
    context = jnp.einsum("nl,nld->nd", attn, values)
    logits = context @ params["w_out"] + params["b_out"]
    return logits, attn


# --------------------------------------------------------------------------- #
# Training, recording q and the attention map at every step
# --------------------------------------------------------------------------- #
@jax.jit
def train(params: dict, x, y_onehot, x_viz):
    """Adam loop; log per-step ``q`` and the attention map on a fixed viz batch."""
    b1, b2, eps = 0.9, 0.999, 1e-8
    m = jax.tree_util.tree_map(jnp.zeros_like, params)
    v = jax.tree_util.tree_map(jnp.zeros_like, params)

    def loss_fn(params):
        logits, _ = forward(params, x, learned_q(params))
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
        q = learned_q(params)
        _, attn_viz = forward(params, x_viz, q)  # attention on the fixed viz batch
        return (params, m, v), (q, attn_viz)

    (params, _, _), (q_hist, attn_hist) = jax.lax.scan(step, (params, m, v), jnp.arange(STEPS))
    return params, q_hist, attn_hist


def main() -> None:
    use_qjax_style()
    keys = jax.random.split(jax.random.PRNGKey(0), 2)
    x_tr, y_tr, _ = make_batch(keys[0], SEQ_LENGTH, N_TRAIN)
    y_tr_oh = jax.nn.one_hot(y_tr, NUM_CLASSES)

    # Fixed visualization batch + a fixed row order so rows never jump in the GIF.
    x_viz, _, info_mask = make_batch(jax.random.PRNGKey(99), SEQ_LENGTH, N_VIZ)
    order = np.asarray(jnp.argsort(jnp.argmax(info_mask, axis=-1)))
    info_mask = np.asarray(info_mask)[order]
    sig_rows, sig_cols = np.nonzero(info_mask)

    _, q_hist, attn_hist = train(init_params(keys[1]), x_tr, y_tr_oh, x_viz)
    q_hist = np.asarray(q_hist)
    attn_hist = np.asarray(attn_hist)[:, order]  # (STEPS, N_VIZ, L), display order
    print(f"learned q: {q_hist[0]:.2f} (init) -> {q_hist[-1]:.2f} (final)")

    # Front-loaded frames: linger on the fast early phase, not the flat tail.
    frac = np.linspace(0.0, 1.0, NUM_FRAMES) ** 1.5
    frames = np.unique((frac * (STEPS - 1)).astype(int))

    # --- figure ---
    fig = plt.figure(figsize=(11.5, 4.7))
    fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.15], wspace=0.25)
    ax_q = fig.add_subplot(gs[0, 0])
    ax_att = fig.add_subplot(gs[0, 1])
    color = qcolors(3)[1]  # mid-magma for the q curve

    ax_q.plot(q_hist, color=color, alpha=0.35, lw=1.5)
    ax_q.axhline(2.0, color="0.4", ls=":", lw=1.0)
    ax_q.text(STEPS, 2.0, "  sparsemax", va="center", fontsize=8, color="0.4")
    ax_q.axhline(Q_MIN, color="0.6", ls=":", lw=1.0)
    ax_q.text(STEPS, Q_MIN, r"  $\to$ softmax", va="center", fontsize=8, color="0.6")
    (marker,) = ax_q.plot([], [], "o", color=color, mec="k", ms=9, zorder=5)
    ax_q.set(xlabel="training step", ylabel="entropic index $q$", ylim=(Q_MIN - 0.1, 2.85))

    im = ax_att.imshow(
        attn_hist[0], cmap=CMAP, vmin=0.0, vmax=1.0, aspect="auto", interpolation="nearest"
    )
    ax_att.scatter(sig_cols, sig_rows, s=22, facecolors="none", edgecolors="white", linewidths=0.8)
    ax_att.set(xlabel="sequence position", ylabel="example")
    ax_att.grid(False)
    fig.colorbar(im, ax=ax_att, fraction=0.046, label="attention weight")

    def update(frame_idx: int):
        step = int(frames[frame_idx])
        q_now = q_hist[step]
        marker.set_data([step], [q_now])
        im.set_array(attn_hist[step])
        ax_q.set_title(f"(a) $q$ learned by gradient descent — $q={q_now:.2f}$")
        ax_att.set_title(f"(b) attention at step {step} ($q={q_now:.2f}$)")
        return [marker, im]

    anim = animation.FuncAnimation(fig, update, frames=len(frames), interval=80, blit=False)
    fig.suptitle("Learning the attention sparsity: $q$ rises, attention sharpens", y=0.99)
    fig.tight_layout()

    gif_path = FIG_DIR / "attention_q_learning.gif"
    anim.save(
        gif_path,
        writer=animation.PillowWriter(fps=14),
        dpi=GIF_DPI,
        savefig_kwargs={"facecolor": "white"},
    )
    print(f"saved {gif_path}")

    update(len(frames) - 1)  # leave the figure on the final frame
    print(f"saved {save_figure(fig, FIG_DIR / 'attention_q_learning')}")


if __name__ == "__main__":
    main()
