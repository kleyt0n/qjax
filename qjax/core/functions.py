r"""``q``-deformed elementary functions and the Tsallis ``q``-algebra.

This module implements the two foundational maps of non-extensive statistics,
the ``q``-logarithm and ``q``-exponential, together with the ``q``-deformed
arithmetic they induce. Each function is a pure, differentiable JAX expression
that recovers its Boltzmann–Gibbs counterpart as ``q -> 1``.

Definitions
-----------
The ``q``-logarithm and its inverse, the ``q``-exponential, are

.. math::

    \ln_q(x) = \frac{x^{1-q} - 1}{1 - q}, \qquad
    \exp_q(x) = \big[1 + (1 - q)\,x\big]_+^{\frac{1}{1-q}},

where :math:`[\cdot]_+ = \max(\cdot, 0)`. Both reduce to :math:`\ln` and
:math:`\exp` as :math:`q \to 1`.
"""

from __future__ import annotations

import jax.numpy as jnp

from qjax.shared.types import Array, Scalar
from qjax.shared.validation import as_scalar_q, near_one


def q_log(x: Array, q: Scalar) -> jnp.ndarray:
    r"""``q``-logarithm :math:`\ln_q(x) = (x^{1-q} - 1)/(1-q)`.

    Recovers the natural logarithm as ``q -> 1``. The implementation uses the
    double-``where`` trick so that the gradient is finite exactly at ``q = 1``.

    Args:
        x: Positive input, any shape.
        q: Entropic index (scalar).

    Returns:
        Element-wise ``q``-logarithm, same shape as ``x``.
    """
    x = jnp.asarray(x, dtype=jnp.result_type(float))
    q = as_scalar_q(q)
    one_minus_q = 1.0 - q
    # Guard the division so the unused branch never produces NaN gradients.
    safe_denom = jnp.where(one_minus_q == 0.0, 1.0, one_minus_q)
    deformed = (x ** one_minus_q - 1.0) / safe_denom
    return jnp.where(near_one(q), jnp.log(x), deformed)


def q_exp(x: Array, q: Scalar) -> jnp.ndarray:
    r"""``q``-exponential :math:`\exp_q(x) = [1 + (1-q)x]_+^{1/(1-q)}`.

    Inverse of :func:`q_log` and the limit of :func:`math.exp` as ``q -> 1``.
    The base is clipped at zero (the *Tsallis cut-off*), so the result is ``0``
    wherever ``1 + (1-q)x < 0``.

    Args:
        x: Input, any shape.
        q: Entropic index (scalar).

    Returns:
        Element-wise ``q``-exponential, same shape as ``x``.
    """
    x = jnp.asarray(x, dtype=jnp.result_type(float))
    q = as_scalar_q(q)
    one_minus_q = 1.0 - q
    base = jnp.maximum(1.0 + one_minus_q * x, 0.0)
    safe_denom = jnp.where(one_minus_q == 0.0, 1.0, one_minus_q)
    deformed = base ** (1.0 / safe_denom)
    return jnp.where(near_one(q), jnp.exp(x), deformed)


def q_add(a: Array, b: Array, q: Scalar) -> jnp.ndarray:
    r"""``q``-addition :math:`a \oplus_q b = a + b + (1-q)\,a\,b`.

    The deformed sum satisfies ``q_log(x*y) = q_add(q_log(x), q_log(y))`` and
    reduces to ordinary addition as ``q -> 1``.

    Args:
        a: First operand.
        b: Second operand.
        q: Entropic index (scalar).

    Returns:
        Element-wise ``q``-sum.
    """
    a = jnp.asarray(a, dtype=jnp.result_type(float))
    b = jnp.asarray(b, dtype=jnp.result_type(float))
    one_minus_q = 1.0 - as_scalar_q(q)
    return a + b + one_minus_q * a * b


def q_diff(a: Array, b: Array, q: Scalar) -> jnp.ndarray:
    r"""``q``-subtraction :math:`a \ominus_q b = (a - b)/(1 + (1-q)b)`.

    Inverse of :func:`q_add` in its first argument: ``q_add(q_diff(a, b), b) == a``.

    Args:
        a: First operand.
        b: Second operand.
        q: Entropic index (scalar).

    Returns:
        Element-wise ``q``-difference.
    """
    a = jnp.asarray(a, dtype=jnp.result_type(float))
    b = jnp.asarray(b, dtype=jnp.result_type(float))
    one_minus_q = 1.0 - as_scalar_q(q)
    return (a - b) / (1.0 + one_minus_q * b)


def q_prod(a: Array, b: Array, q: Scalar) -> jnp.ndarray:
    r"""``q``-product :math:`a \otimes_q b = [a^{1-q} + b^{1-q} - 1]_+^{1/(1-q)}`.

    Dual to :func:`q_add`: it satisfies ``q_exp(x+y) = q_prod(q_exp(x), q_exp(y))``
    and reduces to ordinary multiplication as ``q -> 1``. Defined for ``a, b >= 0``.

    Args:
        a: First operand (non-negative).
        b: Second operand (non-negative).
        q: Entropic index (scalar).

    Returns:
        Element-wise ``q``-product.
    """
    a = jnp.asarray(a, dtype=jnp.result_type(float))
    b = jnp.asarray(b, dtype=jnp.result_type(float))
    q = as_scalar_q(q)
    one_minus_q = 1.0 - q
    safe_exp = jnp.where(one_minus_q == 0.0, 1.0, one_minus_q)
    base = jnp.maximum(a ** safe_exp + b ** safe_exp - 1.0, 0.0)
    deformed = base ** (1.0 / safe_exp)
    return jnp.where(near_one(q), a * b, deformed)


def q_div(a: Array, b: Array, q: Scalar) -> jnp.ndarray:
    r"""``q``-division :math:`a \oslash_q b = [a^{1-q} - b^{1-q} + 1]_+^{1/(1-q)}`.

    Inverse of :func:`q_prod` in its first argument and the limit of ordinary
    division as ``q -> 1``. Defined for ``a, b >= 0``.

    Args:
        a: Numerator (non-negative).
        b: Denominator (non-negative).
        q: Entropic index (scalar).

    Returns:
        Element-wise ``q``-quotient.
    """
    a = jnp.asarray(a, dtype=jnp.result_type(float))
    b = jnp.asarray(b, dtype=jnp.result_type(float))
    q = as_scalar_q(q)
    one_minus_q = 1.0 - q
    safe_exp = jnp.where(one_minus_q == 0.0, 1.0, one_minus_q)
    base = jnp.maximum(a ** safe_exp - b ** safe_exp + 1.0, 0.0)
    deformed = base ** (1.0 / safe_exp)
    return jnp.where(near_one(q), a / b, deformed)


__all__ = ["q_log", "q_exp", "q_add", "q_diff", "q_prod", "q_div"]
