"""Type aliases shared across :mod:`qjax`.

These aliases keep signatures readable without imposing a hard dependency on a
specific array backend. At runtime every value is a :class:`jax.Array`, but the
aliases also accept Python scalars and NumPy arrays, which JAX promotes
automatically.
"""

from __future__ import annotations

import jax

#: Anything JAX can treat as an array: a JAX array, a Python/NumPy scalar, or a
#: nested sequence that :func:`jax.numpy.asarray` can materialize.
Array = jax.Array | float | int

#: A scalar entropic index ``q`` (or any single real number).
Scalar = jax.Array | float | int

__all__ = ["Array", "Scalar"]
