"""Type aliases shared across `qjax`.

These aliases keep signatures readable without imposing a hard dependency on a
specific array backend. At runtime every value is a `jax.Array`, but the
aliases also accept Python scalars and NumPy arrays, which JAX promotes
automatically.

``Array`` and ``Scalar`` were previously the *same* alias, so the distinction
between "an array of any shape" and "a single real number" was documentation
only and a type checker could not act on it. ``Scalar`` is now the narrower of
the two: it excludes the nested sequences that ``Array`` accepts, which is the
real constraint on an entropic index.
"""

from __future__ import annotations

from collections.abc import Sequence

import jax

#: A single real number: a Python scalar or a 0-d `jax.Array`. Used for
#: the entropic index ``q`` and other scalar parameters such as ``beta``.
Scalar = jax.Array | float | int

#: Anything JAX can treat as an array: a JAX array, a Python/NumPy scalar, or a
#: sequence that `jax.numpy.asarray` can materialize.
Array = jax.Array | float | int | Sequence[float]

__all__ = ["Array", "Scalar"]
