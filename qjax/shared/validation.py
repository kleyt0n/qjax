"""Validation and broadcasting helpers for the entropic index ``q``.

Tsallis primitives are parameterized by a single real number ``q`` (the entropic
index). These helpers normalize ``q`` to a JAX scalar and provide a numerically
robust mask for the ``q -> 1`` limit, where most closed-form expressions become
indeterminate (``0 / 0``).
"""

from __future__ import annotations

import jax.numpy as jnp

from qjax.shared.types import Scalar

#: Absolute distance from ``q = 1`` below which the Boltzmann–Gibbs limit is used.
Q_EPS: float = 1e-6


def as_scalar_q(q: Scalar) -> jnp.ndarray:
    """Coerce an entropic index to a floating-point JAX scalar.

    Args:
        q: The entropic index, as a Python number or array-like.

    Returns:
        A 0-d :class:`jax.Array` of floating dtype.
    """
    return jnp.asarray(q, dtype=jnp.result_type(float))


def near_one(q: Scalar, eps: float = Q_EPS) -> jnp.ndarray:
    """Boolean mask for indices that should use the ``q -> 1`` (classical) limit.

    Args:
        q: The entropic index.
        eps: Half-width of the neighborhood around ``q = 1``.

    Returns:
        A boolean array, broadcastable against ``q``, that is ``True`` where
        ``|q - 1| < eps``.
    """
    return jnp.abs(as_scalar_q(q) - 1.0) < eps


__all__ = ["Q_EPS", "as_scalar_q", "near_one"]
