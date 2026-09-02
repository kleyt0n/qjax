r"""A masked autoregressive network over binary spins (MADE).

Variational methods in statistical mechanics need a distribution over
$2^N$ spin configurations that can be both *sampled* and *evaluated*
exactly -- the second is what a mean-field ansatz gives up and what makes a
variational free energy computable without a nested Monte Carlo estimate. An
autoregressive factorization

$$p_\theta(s) = \prod_{i=1}^{N} p_\theta(s_i \mid s_1, \dots, s_{i-1})$$

gives both: sampling is $N$ sequential passes, but the log-probability of a
*given* configuration is a single pass, because every conditional is read off
the same forward computation.

MADE (Germain et al., 2015) enforces the factorization by masking the weights of
an ordinary MLP: each unit carries a degree, and a connection is kept only when
it cannot leak information from $s_i$ into the conditional for $s_i$ itself.
The result is an exactly normalized distribution with no architectural
machinery beyond element-wise masks.

Kept here rather than in `qjax.physics` because nothing about it is
``q``-deformed or physical: it is a neural-network building block, framework-
agnostic like the rest of `qjax.nn`, and the same masked MLP serves any
autoregressive model over binary variables.

References:
    Germain, M., Gregor, K., Murray, I. & Larochelle, H. (2015). MADE: Masked
        Autoencoder for Distribution Estimation. *ICML*.
    Wu, D., Wang, L. & Zhang, P. (2019). Solving statistical mechanics using
        variational autoregressive networks. *Phys. Rev. Lett.* **122**, 080602.
"""

from __future__ import annotations

from collections.abc import Sequence

import jax
import jax.numpy as jnp

from qjax.shared.types import Array

#: A MADE parameter set: ``{"weights": [...], "biases": [...]}``, a pytree.
Params = dict[str, list[jax.Array]]


def made_masks(num_spins: int, hidden: Sequence[int]) -> list[jax.Array]:
    r"""Binary masks enforcing the autoregressive property, one per weight matrix.

    Input $s_i$ is given degree $i+1$ and output $i$ the same. A
    connection into a hidden unit of degree $d$ is kept when the incoming
    degree is $\le d$; a connection from a hidden unit into output $i$ is
    kept when $d < i+1$. Composing the two, output $i$ can depend on
    input $j$ only if $j < i$ -- which is exactly the autoregressive
    condition, and is asserted directly in the tests via a Jacobian.

    Args:
        num_spins: Number of spins ``N``.
        hidden: Widths of the hidden layers, at least one.

    Returns:
        A list of ``len(hidden) + 1`` masks aligned with the weight matrices of
        `made_init`; mask ``k`` has shape ``(in_k, out_k)``.

    Raises:
        ValueError: If ``hidden`` is empty or ``num_spins`` is below 2.
    """
    if not hidden:
        raise ValueError("MADE needs at least one hidden layer.")
    if num_spins < 2:
        raise ValueError(f"MADE needs at least 2 spins; got {num_spins}.")

    input_degrees = jnp.arange(1, num_spins + 1)
    # Hidden degrees cycle through 1..N-1: degree N would let a unit see every
    # input, and no output could then use it.
    hidden_degrees = [1 + jnp.arange(width) % (num_spins - 1) for width in hidden]

    masks = []
    previous = input_degrees
    for degrees in hidden_degrees:
        masks.append((previous[:, None] <= degrees[None, :]).astype(jnp.result_type(float)))
        previous = degrees
    masks.append((previous[:, None] < input_degrees[None, :]).astype(jnp.result_type(float)))
    return masks


def made_init(key: jax.Array, num_spins: int, hidden: Sequence[int]) -> Params:
    """Initialize MADE parameters with Glorot-scaled weights and zero biases.

    Args:
        key: PRNG key.
        num_spins: Number of spins ``N``.
        hidden: Widths of the hidden layers.

    Returns:
        A pytree ``{"weights": [...], "biases": [...]}`` whose weight shapes
        match the masks from `made_masks`.
    """
    widths = [num_spins, *hidden, num_spins]
    keys = jax.random.split(key, len(widths) - 1)
    weights, biases = [], []
    for layer_key, fan_in, fan_out in zip(keys, widths[:-1], widths[1:], strict=True):
        scale = jnp.sqrt(2.0 / (fan_in + fan_out))
        weights.append(jax.random.normal(layer_key, (fan_in, fan_out)) * scale)
        biases.append(jnp.zeros((fan_out,)))
    return {"weights": weights, "biases": biases}


def made_conditionals(params: Params, masks: Sequence[jax.Array], spins: Array) -> jax.Array:
    r"""Logits of $p(s_i = +1 \mid s_{<i})$ for every site, in one forward pass.

    Args:
        params: Parameters from `made_init`.
        masks: Masks from `made_masks`.
        spins: Configurations of shape ``(B, N)`` with entries in ``{-1, +1}``.
            Entries at or after position ``i`` cannot influence logit ``i``, so
            partially filled configurations are safe to pass during sampling.

    Returns:
        Logits of shape ``(B, N)``.
    """
    weights, biases = params["weights"], params["biases"]
    activation = jnp.asarray(spins, dtype=jnp.result_type(float))
    for weight, bias, mask in zip(weights[:-1], biases[:-1], masks[:-1], strict=True):
        activation = jnp.tanh(activation @ (weight * mask) + bias)
    return activation @ (weights[-1] * masks[-1]) + biases[-1]


def made_log_prob(params: Params, masks: Sequence[jax.Array], spins: Array) -> jax.Array:
    r"""Exact log-probability $\log p_\theta(s)$ of each configuration.

    With $p(s_i = +1) = \sigma(z_i)$ the two cases collapse into one:
    $\log p(s_i) = \log \sigma(s_i z_i) = -\mathrm{softplus}(-s_i z_i)$,
    which is also the numerically stable form.

    Args:
        params: Parameters from `made_init`.
        masks: Masks from `made_masks`.
        spins: Configurations of shape ``(B, N)`` with entries in ``{-1, +1}``.

    Returns:
        Log-probabilities of shape ``(B,)``. Summed over all ``2**N``
        configurations these exponentiate to exactly ``1``.
    """
    spins = jnp.asarray(spins, dtype=jnp.result_type(float))
    logits = made_conditionals(params, masks, spins)
    return -jnp.sum(jax.nn.softplus(-spins * logits), axis=-1)


def made_sample(
    key: jax.Array, params: Params, masks: Sequence[jax.Array], num_samples: int
) -> jax.Array:
    """Draw exact samples by filling in one spin at a time.

    Args:
        key: PRNG key.
        params: Parameters from `made_init`.
        masks: Masks from `made_masks`.
        num_samples: Number of configurations to draw.

    Returns:
        Configurations of shape ``(num_samples, N)`` with entries in
        ``{-1.0, +1.0}``, distributed exactly as `made_log_prob` says.
    """
    num_spins = masks[0].shape[0]
    Carry = tuple[jax.Array, jax.Array]

    def step(carry: Carry, site: jax.Array) -> tuple[Carry, None]:
        chain_key, state = carry
        chain_key, subkey = jax.random.split(chain_key)
        logits = made_conditionals(params, masks, state)
        probability = jax.nn.sigmoid(logits[:, site])
        draw = jax.random.uniform(subkey, (num_samples,)) < probability
        return (chain_key, state.at[:, site].set(jnp.where(draw, 1.0, -1.0))), None

    initial: Carry = (key, jnp.zeros((num_samples, num_spins), dtype=jnp.result_type(float)))
    (_, spins), _ = jax.lax.scan(step, initial, jnp.arange(num_spins))
    return spins


__all__ = ["Params", "made_masks", "made_init", "made_conditionals", "made_log_prob", "made_sample"]
