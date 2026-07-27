r"""Tsallis entropy, cross-entropy, and relative entropy (``q``-divergence).

These information measures generalize Shannon's entropy and the
Kullback–Leibler divergence through the entropic index ``q``, recovering them in
the limit ``q -> 1``. They are the natural objective functions for
non-extensive learning.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from qjax.core.functions import q_log
from qjax.shared.series import expm1_over_t
from qjax.shared.types import Array, Scalar
from qjax.shared.validation import as_scalar_q


def tsallis_entropy(p: Array, q: Scalar, axis: int = -1) -> jax.Array:
    r"""Tsallis entropy $S_q(p) = (1 - \sum_i p_i^q)/(q - 1)$.

    Recovers the Shannon entropy $-\sum_i p_i \log p_i$ as ``q -> 1``.
    The entropy is concave in ``p`` and non-negative for probability vectors.

    Computed in the equivalent form $\sum_i p_i \ln_q(1/p_i)$, which
    agrees with the definition above whenever ``p`` sums to one and, unlike it,
    stays finite and correctly differentiable in ``q`` at ``q = 1``. The two
    forms differ on an unnormalized ``p``, for which the definition above is
    genuinely singular at ``q = 1``.

    Args:
        p: Probability mass values along ``axis``: non-negative and normalized.
        q: Entropic index (scalar).
        axis: Axis over which the distribution is defined.

    Returns:
        Tsallis entropy reduced over ``axis``.
    """
    p = jnp.asarray(p, dtype=jnp.result_type(float))
    q = as_scalar_q(q)

    # Evaluated in the equivalent form S_q(p) = sum_i p_i ln_q(1/p_i), written
    # through the entire function (e^t - 1)/t. This is exact for a normalized p
    # and, unlike a hard `|q - 1| < eps` branch onto the Shannon expression,
    # keeps both the value and the derivative *with respect to q* correct
    # through q = 1 (the classical branch is q-independent, so its q-derivative
    # would be an identically zero cliff).
    #
    # The 0*log(0) = 0 convention is applied by masking on a sanitized p: a bare
    # p * log(p) is masked to 0 in value but still back-propagates
    # 0 * log(0) = 0 * -inf = NaN at a zero coordinate.
    support = p > 0.0
    safe_p = jnp.where(support, p, 1.0)
    log_p = jnp.log(safe_p)
    terms = jnp.where(support, -safe_p * log_p * expm1_over_t((q - 1.0) * log_p), 0.0)
    return jnp.sum(terms, axis=axis)


def tsallis_cross_entropy(p: Array, y: Array, q: Scalar, axis: int = -1) -> jax.Array:
    r"""Tsallis cross-entropy $H_q(y, p) = -\sum_i y_i \ln_q p_i$.

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


def tsallis_divergence(p: Array, r: Array, q: Scalar, axis: int = -1) -> jax.Array:
    r"""Tsallis relative entropy $D_q(p\,\|\,r)$.

    Defined as $D_q(p\|r) = \big(\sum_i p_i^q r_i^{1-q} - 1\big)/(q - 1)$,
    equivalently $-\sum_i p_i \ln_q(r_i / p_i)$. Recovers the
    Kullback–Leibler divergence as ``q -> 1`` and is non-negative.

    The second form is the one evaluated: it agrees with the first whenever
    ``p`` sums to one and, unlike it, stays correctly differentiable in ``q`` at
    ``q = 1``.

    Args:
        p: First distribution along ``axis``, non-negative and normalized.
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

    # Evaluated as D_q(p||r) = -sum_i p_i ln_q(r_i / p_i) — the equivalent form
    # already named in the docstring — written through the entire function
    # (e^t - 1)/t so that the Kullback-Leibler limit, and its q-derivative, come
    # out of the same expression instead of a separate q-independent branch.
    #
    # One mask serves both regimes. An earlier version guarded only the KL
    # branch, which still leaked NaN: the deformed branch evaluates
    # r ** (1-q) = 0 ** negative = inf at a zero reference and back-propagates
    # NaN even when unselected.
    mask = (p > 0.0) & (r > 0.0)
    safe_p = jnp.where(mask, p, 1.0)
    safe_r = jnp.where(mask, r, 1.0)
    log_ratio = jnp.log(safe_p / safe_r)
    supported = safe_p * log_ratio * expm1_over_t(q_minus_one * log_ratio)

    # Where p > 0 but r == 0 the reference assigns no mass to an outcome p deems
    # possible: ln_q(0) is -inf for q >= 1, giving a genuinely divergent term,
    # and -1/(1-q) for q < 1, giving the finite p/(1-q). Where p == 0 the term
    # vanishes under the 0 * ln_q(0) = 0 convention.
    safe_gap = jnp.where(q_minus_one < 0.0, -q_minus_one, 1.0)
    unsupported = jnp.where(q_minus_one < 0.0, p / safe_gap, jnp.inf)
    terms = jnp.where(mask, supported, jnp.where(p > 0.0, unsupported, 0.0))
    return jnp.sum(terms, axis=axis)


__all__ = ["tsallis_entropy", "tsallis_cross_entropy", "tsallis_divergence"]
