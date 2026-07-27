r"""Attention whose normalizer is `tsallis_entmax`.

Replacing the softmax in an attention block with ``entmax`` makes the *sparsity*
of the attention map a property of the entropic index rather than a fixed choice:
``q = 1`` reproduces ordinary softmax attention, ``q = 2`` gives ``sparsemax``
(most positions receive exactly zero weight), and ``q < 1`` spreads the weight
more evenly than softmax. Because ``q`` is differentiable, it can be learned
jointly with the rest of the network — see `bounded_q`.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from qjax.core.activations import tsallis_entmax
from qjax.shared.types import Array, Scalar


def entmax_attention(
    queries: Array,
    keys: Array,
    values: Array,
    q: Scalar = 2.0,
    mask: Array | None = None,
    scale: Scalar | None = None,
    num_iters: int = 50,
) -> tuple[jax.Array, jax.Array]:
    r"""Scaled dot-product attention normalized by `tsallis_entmax`.

    Computes ``entmax_q(Q K^T / sqrt(d)) V`` over the last (key) axis. Leading
    axes broadcast, so the same call serves single-head, multi-head, and batched
    inputs.

    Args:
        queries: Query vectors, shape ``(..., n_queries, d_key)``. A single
            query per batch element may be passed as ``(..., d_key)``.
        keys: Key vectors, shape ``(..., n_keys, d_key)``.
        values: Value vectors, shape ``(..., n_keys, d_value)``.
        q: Entropic index (scalar), ``q > 0``. ``1`` is softmax attention,
            ``2`` is sparsemax attention.
        mask: Optional boolean array broadcastable to the score shape
            ``(..., n_queries, n_keys)``. ``False`` positions are excluded.
        scale: Divisor applied to the scores. Defaults to ``sqrt(d_key)``.
        num_iters: Bisection steps for the ``entmax`` threshold search.

    Returns:
        A ``(context, attention)`` pair. ``context`` has shape
        ``(..., n_queries, d_value)`` and ``attention`` has shape
        ``(..., n_queries, n_keys)`` and sums to one over the last axis.

    Example:
        >>> import jax.numpy as jnp
        >>> from qjax.nn import entmax_attention
        >>> q_vec = jnp.ones((2, 4))
        >>> k = jnp.ones((2, 5, 4))
        >>> v = jnp.ones((2, 5, 3))
        >>> context, attn = entmax_attention(q_vec, k, v, q=2.0)
        >>> context.shape, attn.shape
        ((2, 3), (2, 5))
    """
    queries = jnp.asarray(queries, dtype=jnp.result_type(float))
    keys = jnp.asarray(keys, dtype=jnp.result_type(float))
    values = jnp.asarray(values, dtype=jnp.result_type(float))

    # A single query per batch element is the common case in attention pooling;
    # add the query axis, then drop it again on the way out.
    squeeze_query = queries.ndim == keys.ndim - 1
    if squeeze_query:
        queries = queries[..., None, :]

    if scale is None:
        scale = jnp.sqrt(jnp.asarray(keys.shape[-1], dtype=queries.dtype))
    scores = jnp.einsum("...qd,...kd->...qk", queries, keys) / scale

    if mask is not None:
        # -inf scores are driven to exactly zero weight by entmax for every q,
        # and keep the masked positions out of the threshold search.
        scores = jnp.where(jnp.asarray(mask, dtype=bool), scores, -jnp.inf)

    attention = tsallis_entmax(scores, q=q, axis=-1, num_iters=num_iters)
    context = jnp.einsum("...qk,...kd->...qd", attention, values)

    if squeeze_query:
        context = context[..., 0, :]
        attention = attention[..., 0, :]
    return context, attention


__all__ = ["entmax_attention"]
