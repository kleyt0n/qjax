r"""Tsallis ``entmax``: the ``q``-deformed softmax / sparsemax family.

``entmax`` is the probability mapping obtained by regularizing the
maximum-score problem with Tsallis entropy:

$$
\mathrm{entmax}_q(z) = \arg\max_{p \in \Delta}\;
    \langle p, z \rangle + S_q^{T}(p),
$$

with Tsallis entropy $S_q^{T}(p) = \tfrac{1}{q(q-1)}(1 - \sum_i p_i^q)$.
Its solution has the closed form (Peters, Niculae & Martins, 2019)

$$
p_i = \big[(q - 1)\,z_i - \tau\big]_+^{\,1/(q-1)},
$$

where the threshold $\tau$ enforces $\sum_i p_i = 1$. For ``q = 1``
it is the ordinary softmax; ``q = 2`` is ``sparsemax``. Larger ``q`` gives
sparser distributions, ``q < 1`` gives distributions *denser* (higher entropy)
than softmax, and ``q -> 0^+`` approaches the uniform distribution.

Differentiation
---------------
The threshold is located by bisection, but the mapping is **not** differentiated
through that loop — doing so yields a wrong (and non-symmetric) Jacobian. The
solve is wrapped in `jax.lax.stop_gradient` and the derivative is supplied
by a `jax.custom_jvp` rule built from the implicit function theorem.

Writing $s_i = p_i^{2-q}$ and $h_i = -p_i \log p_i$ (both zero off
the support), and letting $T(v) = v - s\,\sum_j v_j / \sum_j s_j$, both
tangents share a single form:

$$
\dot p = T\!\Big(s \odot \dot z
         + \big(h + s \odot z\big)\tfrac{\dot q}{q - 1}\Big),
$$

so the Jacobian with respect to ``z`` is
$J = \mathrm{diag}(s) - s s^\top / \sum_i s_i$ — symmetric, positive
semi-definite, and annihilating the all-ones vector (``entmax`` is invariant to
a constant shift of ``z``). Because ``custom_jvp`` is used rather than
``custom_vjp``, forward mode, reverse mode, and higher-order derivatives are all
exact.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from qjax.shared.types import Array, Scalar
from qjax.shared.validation import Q_EPS, as_scalar_q, near_one, positive_q_or_nan


def _entmax_p(z: jax.Array, q: jax.Array, tau: jax.Array) -> jax.Array:
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


def _support_terms(p: jax.Array, q: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Return ``(s, h)`` with ``s = p^(2-q)`` and ``h = -p log p``, zero off support.

    Both are evaluated on a sanitized ``p = 1`` off the support, where
    ``1 ** (2-q) = 1`` and ``-1 * log(1) = 0`` are finite with finite
    derivatives in both ``p`` and ``q``, so second-order derivatives stay clean.
    """
    support = p > 0.0
    p_safe = jnp.where(support, p, 1.0)
    s = jnp.where(support, p_safe ** (2.0 - q), 0.0)
    h = jnp.where(support, -p_safe * jnp.log(p_safe), 0.0)
    return s, h


def _bracket(zc: jax.Array, q: jax.Array) -> tuple[jax.Array, jax.Array]:
    r"""Threshold bracket ``(heavy, light)`` with ``mass(heavy) >= 1 >= mass(light)``.

    ``zc`` must be centred so that ``max(zc) == 0`` along the last axis, hence
    the scaled maximum ``(q-1) max(zc)`` is ``0``.

    Endpoints are labelled by their *mass property* rather than by numeric
    order, because the mass is decreasing in ``tau`` for ``q > 1`` but
    *increasing* for ``q < 1``. Labelling this way lets one bisection serve both
    regimes.

    - ``heavy = -1``: the top coordinate alone contributes exactly ``1``, so the
      total mass is at least ``1``. Valid for either sign of ``q - 1``.
    - ``light = 0`` for ``q > 1``: every base is non-positive, so the mass is
      ``0``.
    - ``light = -n^(1-q)`` for ``q < 1``: every base is at least ``n^(1-q)``, so
      every ``p_i`` is at most ``1/n`` and the mass is at most ``1``.
    """
    n = float(zc.shape[-1])
    ones = jnp.ones(zc.shape[:-1], dtype=zc.dtype)
    heavy = -ones
    light = jnp.where(q > 1.0, 0.0, -(n ** (1.0 - q))) * ones
    return heavy, light


def _solve_tau(zc: jax.Array, q: jax.Array, num_iters: int) -> jax.Array:
    """Bisect, then Newton-polish, for the threshold ``tau`` on centred scores."""
    heavy, light = _bracket(zc, q)

    def step(_, bounds):
        heavy, light = bounds
        mid = 0.5 * (heavy + light)
        too_much = jnp.sum(_entmax_p(zc, q, mid), axis=-1) > 1.0
        return jnp.where(too_much, mid, heavy), jnp.where(too_much, light, mid)

    heavy, light = jax.lax.fori_loop(0, num_iters, step, (heavy, light))
    tau = 0.5 * (heavy + light)

    # One safeguarded Newton step. With F(tau) = mass(tau) - 1 the derivative is
    # F'(tau) = -sum(s) / (q - 1), so the update below is correct for both signs
    # of q - 1. This drives |sum(p) - 1| to roughly machine epsilon, which the
    # JVP rule relies on (it assumes tau solves the constraint exactly) and which
    # makes the result largely insensitive to ``num_iters``.
    p = _entmax_p(zc, q, tau)
    s, _ = _support_terms(p, q)
    s_sum = jnp.sum(s, axis=-1)
    polished = tau + (q - 1.0) * (jnp.sum(p, axis=-1) - 1.0) / s_sum
    return jnp.clip(polished, jnp.minimum(heavy, light), jnp.maximum(heavy, light))


def _centre(z: jax.Array) -> jax.Array:
    """Shift scores so the last-axis maximum is ``0``.

    ``entmax`` is invariant to a constant shift, and so is the JVP projector, but
    only if the shift carries no gradient — centring by a *differentiated* max
    would inject a spurious term. Hence `jax.lax.stop_gradient`.
    """
    return z - jax.lax.stop_gradient(jnp.max(z, axis=-1, keepdims=True))


@partial(jax.custom_jvp, nondiff_argnums=(2,))
def _entmax_core(z: jax.Array, q: jax.Array, num_iters: int) -> jax.Array:
    """``entmax`` over the last axis for ``q != 1``, with an exact custom JVP."""
    zc = _centre(z)
    tau = jax.lax.stop_gradient(
        _solve_tau(jax.lax.stop_gradient(zc), jax.lax.stop_gradient(q), num_iters)
    )
    p = _entmax_p(zc, q, tau)
    return p / jnp.sum(p, axis=-1, keepdims=True)


@_entmax_core.defjvp
def _entmax_core_jvp(num_iters, primals, tangents):
    r"""Exact JVP ``dp = T(s*dz + (h + s*z) dq/(q-1))`` with ``T(v) = v - s sum(v)/sum(s)``.

    Note the recursive call to `_entmax_core` for the primal: it routes
    higher-order derivatives back through this rule rather than through the
    bisection, and costs nothing extra because JAX only invokes the rule when a
    derivative is actually requested.

    Numerical caveat: for ``q > 2`` a coordinate sitting essentially on the kink
    has a tiny ``p_i`` and hence an enormous ``s_i = p_i^(2-q)``, so the ``O(n)``
    projection loses precision to cancellation (roughly 3 of 7 digits at
    ``q = 2.8`` in float32). The exact-arithmetic result is well conditioned; a
    stable ``O(n^2)`` rewrite exists but is not worth the cost.
    """
    z, q = primals
    z_dot, q_dot = tangents
    p = _entmax_core(z, q, num_iters)
    s, h = _support_terms(p, q)
    s_sum = jnp.sum(s, axis=-1, keepdims=True)
    v = s * z_dot + (h + s * _centre(z)) * (q_dot / (q - 1.0))
    p_dot = v - s * (jnp.sum(v, axis=-1, keepdims=True) / s_sum)
    return p, p_dot


@jax.custom_jvp
def _softmax_core(z: jax.Array, q: jax.Array) -> jax.Array:
    """Softmax over the last axis. ``q`` is inert in the primal but not in the JVP."""
    del q
    e = jnp.exp(z - jnp.max(z, axis=-1, keepdims=True))
    return e / jnp.sum(e, axis=-1, keepdims=True)


@_softmax_core.defjvp
def _softmax_core_jvp(primals, tangents):
    r"""JVP at ``q = 1``, including the non-zero ``q``-derivative.

    Taking ``q -> 1`` in the ``entmax`` JVP gives

    $$
    \frac{\partial p_i}{\partial q}\Big|_{q=1}
        = -\tfrac12\,p_i\big(\log^2 p_i - \textstyle\sum_j p_j \log^2 p_j\big),
    $$

    which is *not* zero. Supplying it here removes what would otherwise be a
    gradient cliff at the ``Q_EPS`` boundary between this branch and
    `_entmax_core`.
    """
    z, q = primals
    z_dot, q_dot = tangents
    p = _softmax_core(z, q)
    u = jnp.log(p)
    dz = p * z_dot - p * jnp.sum(p * z_dot, axis=-1, keepdims=True)
    dq = -0.5 * p * (u * u - jnp.sum(p * u * u, axis=-1, keepdims=True)) * q_dot
    return p, dz + dq


def tsallis_entmax(
    z: Array,
    q: Scalar = 2.0,
    axis: int = -1,
    num_iters: int = 50,
) -> jax.Array:
    r"""Tsallis ``entmax`` over a simplex axis.

    Solves for the threshold ``tau`` such that ``sum([(q-1)z - tau]_+^{1/(q-1)})``
    equals one. ``q = 1`` short-circuits to a numerically stable softmax.

    Gradients are exact: the threshold search is not differentiated through, and
    a `jax.custom_jvp` rule supplies the implicit-function derivative with
    respect to both ``z`` and ``q``. The Jacobian in ``z`` is
    ``diag(s) - s s^T / sum(s)`` with ``s = p ** (2 - q)``.

    Args:
        z: Scores / logits; the distribution is formed along ``axis``.
        q: Entropic index (scalar), ``q > 0``. ``q = 1`` -> softmax,
            ``q = 2`` -> sparsemax. Values above ``1`` give sparse outputs;
            values below ``1`` give outputs denser than softmax.
        axis: Axis over which to normalize.
        num_iters: Number of bisection steps for the threshold search. A Newton
            polish step follows, so the result is accurate even for small values.

    Returns:
        Probabilities with the same shape as ``z`` that sum to one along ``axis``.

    Raises:
        ValueError: If ``q`` is a concrete value and ``q <= 0``. A traced,
            non-positive ``q`` yields ``NaN`` instead.
    """
    z = jnp.asarray(z, dtype=jnp.result_type(float))
    q = positive_q_or_nan(as_scalar_q(q))

    # Move the working axis to the end for a uniform reduction layout.
    z = jnp.moveaxis(z, axis, -1)

    near = near_one(q)
    # Under vmap over a batched q, lax.cond lowers to a select and *both*
    # branches execute, so keep the entmax branch clear of the 1/(q-1) pole.
    # The clamped region is exactly the region the select discards.
    q_eff = jnp.where(near, jnp.where(q < 1.0, 1.0 - Q_EPS, 1.0 + Q_EPS), q)

    p = jax.lax.cond(
        near,
        lambda _: _softmax_core(z, q),
        lambda _: _entmax_core(z, q_eff, num_iters),
        operand=None,
    )
    return jnp.moveaxis(p, -1, axis)


__all__ = ["tsallis_entmax"]
