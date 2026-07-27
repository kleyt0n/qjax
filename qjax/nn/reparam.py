r"""Keeping a learnable entropic index inside its valid range.

Treating ``q`` as a free parameter and optimizing it directly does not work: the
Tsallis primitives are undefined at ``q <= 0``, the ``q``-Gaussian requires
``q < 3``, and gradient descent will happily step outside either bound. The
standard remedy — used verbatim in six of this repository's examples before it
lived here — is to optimize an unconstrained real ``q_raw`` and squash it:

$$
q = q_{\min} + (q_{\max} - q_{\min})\,\sigma(q_{\mathrm{raw}}).
$$
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from qjax.shared.types import Scalar


def bounded_q(q_raw: Scalar, lo: Scalar = 1.0, hi: Scalar = 3.0) -> jax.Array:
    r"""Map an unconstrained parameter to an entropic index in ``(lo, hi)``.

    The map is smooth and strictly monotone, so gradients flow to ``q_raw``
    everywhere and the optimizer can never leave the interval. ``q_raw = 0``
    corresponds to the midpoint.

    The interval is open in exact arithmetic but closed in floating point: the
    sigmoid saturates to exactly ``0`` or ``1`` once ``|q_raw|`` exceeds roughly
    ``37`` (float64) or ``17`` (float32), so ``q`` can reach ``lo`` or ``hi``
    exactly. Choose ``lo`` strictly inside the valid domain — ``q > 0`` for the
    Tsallis primitives, ``q < 3`` for the ``q``-Gaussian — rather than relying
    on strict inequality here.

    Args:
        q_raw: Unconstrained real parameter, any shape.
        lo: Lower bound of the open interval (exclusive).
        hi: Upper bound of the open interval (exclusive).

    Returns:
        The entropic index, same shape as ``q_raw``.

    Example:
        >>> import jax.numpy as jnp
        >>> from qjax.nn import bounded_q
        >>> float(bounded_q(jnp.asarray(0.0), 1.0, 3.0))
        2.0
    """
    lo = jnp.asarray(lo, dtype=jnp.result_type(float))
    hi = jnp.asarray(hi, dtype=jnp.result_type(float))
    return lo + (hi - lo) * jax.nn.sigmoid(jnp.asarray(q_raw, dtype=jnp.result_type(float)))


def inverse_bounded_q(q: Scalar, lo: Scalar = 1.0, hi: Scalar = 3.0) -> jax.Array:
    """Invert `bounded_q` to initialize ``q_raw`` at a chosen ``q``.

    Useful for starting training from a meaningful index — ``q = 1`` for a
    softmax-like attention map, say — rather than from the interval midpoint.

    Args:
        q: Target entropic index, strictly inside ``(lo, hi)``.
        lo: Lower bound used by `bounded_q`.
        hi: Upper bound used by `bounded_q`.

    Returns:
        The ``q_raw`` for which ``bounded_q(q_raw, lo, hi) == q``.
    """
    q = jnp.asarray(q, dtype=jnp.result_type(float))
    lo = jnp.asarray(lo, dtype=jnp.result_type(float))
    hi = jnp.asarray(hi, dtype=jnp.result_type(float))
    unit = (q - lo) / (hi - lo)
    return jnp.log(unit) - jnp.log1p(-unit)


__all__ = ["bounded_q", "inverse_bounded_q"]
