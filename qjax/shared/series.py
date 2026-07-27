r"""Entire-function ratios used to take the ``q -> 1`` limit stably.

Every Tsallis closed form is a ``0/0`` indeterminate at ``q = 1``, and the
textbook way to evaluate it — select a separate Boltzmann-Gibbs expression when
``|q - 1|`` is small — has two defects:

1. Between the switch-over point and the region where the deformed form is
   accurate there is a band in which *neither* is: subtracting ``1`` from
   ``x^{1-q} ~ 1`` loses most of the mantissa. In float32 the relative error of
   ``(x^{1-q} - 1)/(1-q)`` peaks near ``4e-3`` around ``q = 1.00001``.
2. The classical branch does not depend on ``q``, so its derivative with respect
   to ``q`` is identically zero. A learnable entropic index that wanders into
   the switch-over window sees a hard zero gradient.

Both vanish if the indeterminate quotient is written through the *entire*
functions below, which are analytic at the origin and equal ``1`` there. The
classical limit then falls out of the same expression, with no branch on ``q``
and no loss of accuracy.

Near the origin each ratio is evaluated by its Taylor series rather than by the
direct form, which keeps the value *and* the derivative correct.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

#: Below this magnitude the Taylor series is used. Its truncation error is
#: ``O(t^4/120)``, already below float64 epsilon at ``t = 1e-4``, while the
#: direct form is accurate above it.
SERIES_CUTOFF: float = 1e-4


def expm1_over_t(t: jax.Array) -> jax.Array:
    """Entire function ``(exp(t) - 1)/t``, equal to ``1`` at ``t = 0``."""
    small = jnp.abs(t) < SERIES_CUTOFF
    safe_t = jnp.where(small, 1.0, t)
    series = 1.0 + t * (0.5 + t * (1.0 / 6.0 + t / 24.0))
    return jnp.where(small, series, jnp.expm1(safe_t) / safe_t)


def log1p_over_t(t: jax.Array) -> jax.Array:
    """Entire function ``log(1 + t)/t``, equal to ``1`` at ``t = 0``."""
    small = jnp.abs(t) < SERIES_CUTOFF
    safe_t = jnp.where(small, 1.0, t)
    series = 1.0 - t * (0.5 - t * (1.0 / 3.0 - t * 0.25))
    return jnp.where(small, series, jnp.log1p(safe_t) / safe_t)


__all__ = ["SERIES_CUTOFF", "expm1_over_t", "log1p_over_t"]
