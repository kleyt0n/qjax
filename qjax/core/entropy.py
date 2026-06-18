r"""Tsallis entropy, cross-entropy, and relative entropy (``q``-divergence).

These information measures generalize Shannon's entropy and the
Kullback–Leibler divergence through the entropic index ``q``, recovering them in
the limit ``q -> 1``. They are the natural objective functions for
non-extensive learning.
"""

from __future__ import annotations

import jax.numpy as jnp

from qjax.core.functions import q_log
from qjax.shared.types import Array, Scalar
from qjax.shared.validation import as_scalar_q, near_one


def tsallis_entropy(p: Array, q: Scalar, axis: int = -1) -> jnp.ndarray:
    r"""Tsallis entropy :math:`S_q(p) = (1 - \sum_i p_i^q)/(q - 1)`.

    Recovers the Shannon entropy :math:`-\sum_i p_i \log p_i` as ``q -> 1``.
    The entropy is concave in ``p`` and non-negative for probability vectors.

    Args:
        p: Probability mass values along ``axis`` (assumed non-negative and,
            ideally, normalized).
        q: Entropic index (scalar).
        axis: Axis over which the distribution is defined.

    Returns:
        Tsallis entropy reduced over ``axis``.
    """
    p = jnp.asarray(p, dtype=jnp.result_type(float))
    q = as_scalar_q(q)
    q_minus_one = q - 1.0
    safe_denom = jnp.where(q_minus_one == 0.0, 1.0, q_minus_one)
    deformed = (1.0 - jnp.sum(p ** q, axis=axis)) / safe_denom
    # Shannon entropy with the 0*log(0) = 0 convention. The double-``where``
    # guards the log *input* as well as the output: a bare ``p * jnp.log(p)``
    # is masked to 0 in value but still back-propagates ``0 * log(0) = 0 * -inf
    # = NaN`` at a zero coordinate (even when this branch is unselected, e.g.
    # q = 2). Feeding ``jnp.log`` a safe argument keeps the gradient finite.
    safe_p = jnp.where(p > 0.0, p, 1.0)
    plogp = jnp.where(p > 0.0, p * jnp.log(safe_p), 0.0)
    shannon = -jnp.sum(plogp, axis=axis)
    return jnp.where(near_one(q), shannon, deformed)


def tsallis_cross_entropy(p: Array, y: Array, q: Scalar, axis: int = -1) -> jnp.ndarray:
    r"""Tsallis cross-entropy :math:`H_q(y, p) = -\sum_i y_i \ln_q p_i`.

    A drop-in ``q``-deformed classification loss. With one-hot ``y`` it reduces
    to ``-q_log(p_correct, q)``, and to the standard cross-entropy as ``q -> 1``.

    Args:
        p: Predicted probabilities along ``axis``.
        y: Target distribution along ``axis`` (e.g. one-hot labels).
        q: Entropic index (scalar).
        axis: Axis over which the distributions are defined.

    Returns:
        Tsallis cross-entropy reduced over ``axis``.
    """
    p = jnp.asarray(p, dtype=jnp.result_type(float))
    y = jnp.asarray(y, dtype=jnp.result_type(float))
    # Apply the 0 * ln_q(0) = 0 convention. Where the target mass is zero the
    # term is dropped, and ``q_log`` is evaluated on a safe argument so a zero
    # prediction at an *unused* class (common for sparse ``tsallis_entmax``
    # outputs) yields neither a NaN value nor a NaN gradient. A zero prediction
    # *on* a positive-target class is a genuine +inf loss and is left intact.
    safe_p = jnp.where(y > 0.0, p, 1.0)
    contrib = jnp.where(y > 0.0, y * q_log(safe_p, q), 0.0)
    return -jnp.sum(contrib, axis=axis)


def tsallis_divergence(p: Array, r: Array, q: Scalar, axis: int = -1) -> jnp.ndarray:
    r"""Tsallis relative entropy :math:`D_q(p\,\|\,r)`.

    Defined as :math:`D_q(p\|r) = \big(\sum_i p_i^q r_i^{1-q} - 1\big)/(q - 1)`,
    equivalently :math:`-\sum_i p_i \ln_q(r_i / p_i)`. Recovers the
    Kullback–Leibler divergence as ``q -> 1`` and is non-negative.

    Args:
        p: First distribution along ``axis``.
        r: Second (reference) distribution along ``axis``.
        q: Entropic index (scalar).
        axis: Axis over which the distributions are defined.

    Returns:
        Tsallis divergence reduced over ``axis``.
    """
    p = jnp.asarray(p, dtype=jnp.result_type(float))
    r = jnp.asarray(r, dtype=jnp.result_type(float))
    q = as_scalar_q(q)
    q_minus_one = q - 1.0
    safe_denom = jnp.where(q_minus_one == 0.0, 1.0, q_minus_one)
    deformed = (jnp.sum(p ** q * r ** (1.0 - q), axis=axis) - 1.0) / safe_denom
    # KL divergence with the 0*log(0) = 0 convention. As in ``tsallis_entropy``,
    # the ratio is evaluated on a safe argument so the masked terms contribute
    # no NaN to the gradient (``p/r`` can also be ``0/0`` or ``x/0``).
    mask = (p > 0.0) & (r > 0.0)
    safe_ratio = jnp.where(mask, p / r, 1.0)
    ratio = jnp.where(mask, p * jnp.log(safe_ratio), 0.0)
    kl = jnp.sum(ratio, axis=axis)
    return jnp.where(near_one(q), kl, deformed)


__all__ = ["tsallis_entropy", "tsallis_cross_entropy", "tsallis_divergence"]
