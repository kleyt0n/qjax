r"""``q``-deformed elementary functions and the Tsallis ``q``-algebra.

This module implements the two foundational maps of non-extensive statistics,
the ``q``-logarithm and ``q``-exponential, together with the ``q``-deformed
arithmetic they induce. Each function is a pure, differentiable JAX expression
that recovers its Boltzmann–Gibbs counterpart as ``q -> 1``.

Definitions
-----------
The ``q``-logarithm and its inverse, the ``q``-exponential, are

$$
\ln_q(x) = \frac{x^{1-q} - 1}{1 - q}, \qquad
\exp_q(x) = \big[1 + (1 - q)\,x\big]_+^{\frac{1}{1-q}},
$$

where $[\cdot]_+ = \max(\cdot, 0)$. Both reduce to $\ln$ and
$\exp$ as $q \to 1$.

Numerical form
--------------
Evaluating $(x^{1-q} - 1)/(1-q)$ directly loses catastrophically to
cancellation as $q \to 1$ — in float32 the relative error peaks near
``4e-3`` around ``q = 1.00001``. Both functions are therefore written through
`jax.numpy.expm1` / `jax.numpy.log1p`, which are exact in that
regime:

$$
\ln_q(x) = \log x \cdot \frac{e^{u} - 1}{u},\quad u = (1-q)\log x,
\qquad
\exp_q(x) = \exp\!\Big(x \cdot \frac{\log(1 + a)}{a}\Big),\quad a = (1-q)x.
$$

The shared factor is the entire function $(e^t - 1)/t \to 1$ as
$t \to 0$, so no ``q = 1`` special case is needed at all: the classical
limit falls out of the same expression. Near $t = 0$ a short Taylor series
replaces the ratio, which keeps not only the value but also the *derivative with
respect to* ``q`` correct — a hard ``where(q == 1, ...)`` branch would return a
``q``-independent expression and hence a spurious zero ``q``-gradient.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from qjax.shared.series import expm1_over_t, log1p_over_t
from qjax.shared.types import Array, Scalar
from qjax.shared.validation import as_scalar_q, near_one


def _safe_power(value: jax.Array, exponent: jax.Array) -> jax.Array:
    """Compute ``value ** exponent`` for ``value >= 0`` without NaN gradients.

    At ``value == 0`` the result is ``0`` for a positive exponent and ``+inf``
    for a negative one. Both are supplied as constants, so no gradient path runs
    through the singular ``0 ** negative``, which would otherwise back-propagate
    ``NaN`` into any expression that merely *mentions* this term.
    """
    positive = value > 0.0
    safe_value = jnp.where(positive, value, 1.0)
    at_zero = jnp.where(exponent > 0.0, 0.0, jnp.inf)
    return jnp.where(positive, safe_value**exponent, at_zero)


def _clipped_power(base: jax.Array, exponent: jax.Array) -> jax.Array:
    """Compute ``[base]_+ ** (1/exponent)`` with the Tsallis cut-off.

    Past the cut-off (``base <= 0``) the sign of the exponent decides the value:
    ``0`` when ``1/exponent > 0`` and ``+inf`` when it is negative. The base is
    sanitized before the power so the clipped region contributes neither a NaN
    value nor a NaN gradient — a bare ``jnp.maximum(base, 0) ** (1/exponent)``
    back-propagates ``NaN`` through ``0 ** negative``.
    """
    positive = base > 0.0
    safe_base = jnp.where(positive, base, 1.0)
    cut_off = jnp.where(exponent > 0.0, 0.0, jnp.inf)
    return jnp.where(positive, safe_base ** (1.0 / exponent), cut_off)


def q_log(x: Array, q: Scalar) -> jax.Array:
    r"""``q``-logarithm $\ln_q(x) = (x^{1-q} - 1)/(1-q)$.

    Recovers the natural logarithm as ``q -> 1``, continuously and with the
    correct derivative in ``q`` (see the module docstring).

    At ``x = 0`` the limit is ``-1/(1-q)`` for ``q < 1`` and ``-inf`` for
    ``q >= 1``. Negative ``x`` is outside the domain and yields ``NaN``.

    Args:
        x: Non-negative input, any shape.
        q: Entropic index (scalar).

    Returns:
        Element-wise ``q``-logarithm, same shape as ``x``.
    """
    x = jnp.asarray(x, dtype=jnp.result_type(float))
    q = as_scalar_q(q)
    one_minus_q = 1.0 - q

    # Evaluate log on a sanitized argument so that x <= 0 contributes no NaN to
    # the gradient of the in-domain branch.
    positive = x > 0.0
    log_x = jnp.log(jnp.where(positive, x, 1.0))
    deformed = log_x * expm1_over_t(one_minus_q * log_x)

    safe_denom = jnp.where(one_minus_q == 0.0, 1.0, one_minus_q)
    at_zero = jnp.where(one_minus_q > 0.0, -1.0 / safe_denom, -jnp.inf)
    return jnp.where(positive, deformed, jnp.where(x == 0.0, at_zero, jnp.nan))


def q_exp(x: Array, q: Scalar) -> jax.Array:
    r"""``q``-exponential $\exp_q(x) = [1 + (1-q)x]_+^{1/(1-q)}$.

    Inverse of `q_log` and the limit of `math.exp` as ``q -> 1``.

    Past the *Tsallis cut-off* — that is, wherever ``1 + (1-q)x <= 0`` — the
    exponent ``1/(1-q)`` decides the value: the result is ``0`` for ``q < 1``
    (positive exponent) and ``+inf`` for ``q > 1`` (negative exponent), matching
    the genuine divergence of the ``q``-exponential there.

    Args:
        x: Input, any shape.
        q: Entropic index (scalar).

    Returns:
        Element-wise ``q``-exponential, same shape as ``x``.
    """
    x = jnp.asarray(x, dtype=jnp.result_type(float))
    q = as_scalar_q(q)
    one_minus_q = 1.0 - q
    a = one_minus_q * x

    # Sanitize before log1p so the clipped region contributes no NaN gradient.
    in_support = a > -1.0
    safe_a = jnp.where(in_support, a, 0.0)
    finite = jnp.exp(x * log1p_over_t(safe_a))

    cut_off = jnp.where(one_minus_q > 0.0, 0.0, jnp.inf)
    return jnp.where(in_support, finite, cut_off)


def q_add(a: Array, b: Array, q: Scalar) -> jax.Array:
    r"""``q``-addition $a \oplus_q b = a + b + (1-q)\,a\,b$.

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


def q_diff(a: Array, b: Array, q: Scalar) -> jax.Array:
    r"""``q``-subtraction $a \ominus_q b = (a - b)/(1 + (1-q)b)$.

    Inverse of `q_add` in its first argument: ``q_add(q_diff(a, b), b) == a``.

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


def q_prod(a: Array, b: Array, q: Scalar) -> jax.Array:
    r"""``q``-product $a \otimes_q b = [a^{1-q} + b^{1-q} - 1]_+^{1/(1-q)}$.

    Dual to `q_add`: it satisfies ``q_exp(x+y) = q_prod(q_exp(x), q_exp(y))``
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
    base = _safe_power(a, safe_exp) + _safe_power(b, safe_exp) - 1.0
    return jnp.where(near_one(q), a * b, _clipped_power(base, safe_exp))


def q_div(a: Array, b: Array, q: Scalar) -> jax.Array:
    r"""``q``-division $a \oslash_q b = [a^{1-q} - b^{1-q} + 1]_+^{1/(1-q)}$.

    Inverse of `q_prod` in its first argument and the limit of ordinary
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
    base = _safe_power(a, safe_exp) - _safe_power(b, safe_exp) + 1.0
    # Guard the classical quotient too: a bare ``a / b`` back-propagates NaN from
    # a zero denominator even when this branch is unselected (e.g. q = 2). The
    # divide-by-zero limit is spelled with constants rather than as ``a * inf``,
    # whose derivative is ``inf`` and would still poison the zero cotangent.
    nonzero_b = b != 0.0
    signed_inf = jnp.where(a > 0.0, jnp.inf, jnp.where(a < 0.0, -jnp.inf, jnp.nan))
    classical = jnp.where(nonzero_b, a / jnp.where(nonzero_b, b, 1.0), signed_inf)
    return jnp.where(near_one(q), classical, _clipped_power(base, safe_exp))


__all__ = ["q_log", "q_exp", "q_add", "q_diff", "q_prod", "q_div"]
