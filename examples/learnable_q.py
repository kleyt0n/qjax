"""Treat the entropic index q as a learnable parameter.

We generate data from a q-Gaussian with a hidden ``q_true`` and recover it by
maximizing the q-Gaussian log-likelihood with gradient descent — demonstrating
that ``q`` is differentiable and can be fit end-to-end.

Run with: ``uv run python examples/learnable_q.py``
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

import qjax
from qjax.plots import save_figure, use_qjax_style

FIG_DIR = Path(__file__).parent / "figures"


def neg_log_likelihood(params: dict, x: jnp.ndarray) -> jnp.ndarray:
    """Mean negative q-Gaussian log-likelihood of ``x`` under ``params``."""
    # softplus keeps beta > 0; q is constrained to (1, 3) via a scaled sigmoid.
    beta = jax.nn.softplus(params["beta_raw"]) + 1e-3
    q = 1.0 + 2.0 * jax.nn.sigmoid(params["q_raw"])
    return -jnp.mean(qjax.q_gaussian_logpdf(x, q, beta))


def main() -> None:
    use_qjax_style()
    key = jax.random.PRNGKey(0)
    q_true, beta_true = 1.6, 0.8
    data = qjax.sample(key, q=q_true, beta=beta_true, shape=(20_000,))

    params = {"q_raw": jnp.array(0.0), "beta_raw": jnp.array(0.0)}
    loss_and_grad = jax.jit(jax.value_and_grad(neg_log_likelihood))

    lr, history = 0.05, []
    for step in range(400):
        loss, grads = loss_and_grad(params, data)
        params = {k: v - lr * grads[k] for k, v in params.items()}
        q_hat = float(1.0 + 2.0 * jax.nn.sigmoid(params["q_raw"]))
        history.append((float(loss), q_hat))
        if step % 100 == 0:
            print(f"step {step:3d}  loss={loss:.4f}  q_hat={q_hat:.4f}")

    q_final = 1.0 + 2.0 * jax.nn.sigmoid(params["q_raw"])
    beta_final = jax.nn.softplus(params["beta_raw"]) + 1e-3
    print(
        f"recovered q={float(q_final):.3f} (true {q_true}), "
        f"beta={float(beta_final):.3f} (true {beta_true})"
    )

    losses, q_hats = zip(*history, strict=False)
    fig, (ax_loss, ax_q) = plt.subplots(1, 2, figsize=(12, 4.5))
    ax_loss.plot(losses, color="#bc3754")
    ax_loss.set(xlabel="step", ylabel="negative log-likelihood", title="optimization")
    ax_q.plot(q_hats, color="#bc3754", label=r"$\hat q$")
    ax_q.axhline(q_true, color="0.4", ls="--", label=r"$q_\mathrm{true}$")
    ax_q.set(xlabel="step", ylabel="q", title="recovering q")
    ax_q.legend()

    fig.tight_layout()
    print(f"saved {save_figure(fig, FIG_DIR / 'learnable_q')}")


if __name__ == "__main__":
    main()
