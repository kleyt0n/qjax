r"""Tsallis ``entmax``: the ``q``-deformed softmax / sparsemax family.

``entmax`` is the probability mapping obtained by regularizing the
maximum-score problem with Tsallis entropy:

.. math::

    \mathrm{entmax}_q(z) = \arg\max_{p \in \Delta}\;
        \langle p, z \rangle + S_q^{T}(p),

with Tsallis entropy :math:`S_q^{T}(p) = \tfrac{1}{q(q-1)}(1 - \sum_i p_i^q)`.
Its solution has the closed form (Peters, Niculae & Martins, 2019)

.. math::

    p_i = \big[(q - 1)\,z_i - \tau\big]_+^{\,1/(q-1)},

where the threshold :math:`\tau` enforces :math:`\sum_i p_i = 1`. For ``q = 1``
it is the ordinary softmax (dense); for ``q = 2`` it is ``sparsemax`` (sparse).
Intermediate ``q`` trade off smoothness against sparsity.

The threshold is found by bisection on a tight bracket, which is exact in the
limit and fully compatible with :func:`jax.jit`, :func:`jax.grad`, and
:func:`jax.vmap`.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from qjax.shared.types import Array, Scalar
from qjax.shared.validation import as_scalar_q, near_one


def _entmax_p(z: jnp.ndarray, q: jnp.ndarray, tau: jnp.ndarray) -> jnp.ndarray:
    """Unnormalized ``entmax`` map ``[(q-1) z - tau]_+^{1/(q-1)}`` (last axis).

    The double-``where`` keeps the gradient finite in the sparse regime: at a
    clipped (zero) coordinate the naive ``0 ** p`` has gradient ``0 * log(0)``,
    which is ``NaN`` with respect to both ``z`` and the entropic index ``q``.
    Returning an exact ``0`` through a separate branch avoids evaluating the
    power (and its ``log``) there, so a learnable ``q`` trains stably.
    """
    base = (q - 1.0) * z - tau[..., None]
    positive = base > 0.0
    safe_base = jnp.where(positive, base, 1.0)
    return jnp.where(positive, safe_base ** (1.0 / (q - 1.0)), 0.0)


def tsallis_entmax(
    z: Array,
    q: Scalar = 2.0,
    axis: int = -1,
    num_iters: int = 50,
) -> jnp.ndarray:
    r"""Tsallis ``entmax`` over a simplex axis.

    Solves for the threshold ``tau`` such that ``sum([(q-1)z - tau]_+^{1/(q-1)})``
    equals one, by bisection on the tight bracket
    ``[(q-1)·max(z) - 1, (q-1)·max(z)]``. ``q = 1`` short-circuits to a
    numerically stable softmax.

    Args:
        z: Scores / logits; the distribution is formed along ``axis``.
        q: Entropic index (scalar), ``q >= 1``. ``q = 1`` -> softmax,
            ``q = 2`` -> sparsemax.
        axis: Axis over which to normalize.
        num_iters: Number of bisection steps for the threshold search.

    Returns:
        Probabilities with the same shape as ``z`` that sum to one along ``axis``.
    """
    z = jnp.asarray(z, dtype=jnp.result_type(float))
    q = as_scalar_q(q)

    # Move the working axis to the end for a uniform reduction layout.
    z = jnp.moveaxis(z, axis, -1)

    def softmax(_):
        z_shift = z - jnp.max(z, axis=-1, keepdims=True)
        e = jnp.exp(z_shift)
        return e / jnp.sum(e, axis=-1, keepdims=True)

    def entmax(_):
        # p is increasing in z and decreasing in tau, so the total mass is
        # monotonically decreasing in tau. At tau = (q-1)·max(z) the mass is 0;
        # at tau = (q-1)·max(z) - 1 the top term alone is >= 1.
        scaled_max = (q - 1.0) * jnp.max(z, axis=-1)
        hi = scaled_max          # mass(hi) = 0  <= 1
        lo = scaled_max - 1.0    # mass(lo) >= 1

        def step(_, bounds):
            lo, hi = bounds
            mid = 0.5 * (lo + hi)
            mass = jnp.sum(_entmax_p(z, q, mid), axis=-1)
            too_much = mass > 1.0
            lo = jnp.where(too_much, mid, lo)
            hi = jnp.where(too_much, hi, mid)
            return lo, hi

        lo, hi = jax.lax.fori_loop(0, num_iters, step, (lo, hi))
        tau = 0.5 * (lo + hi)
        p = _entmax_p(z, q, tau)
        return p / jnp.sum(p, axis=-1, keepdims=True)

    p = jax.lax.cond(near_one(q), softmax, entmax, operand=None)
    return jnp.moveaxis(p, -1, axis)


__all__ = ["tsallis_entmax"]
