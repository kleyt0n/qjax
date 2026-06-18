r"""The ``q``-Gaussian distribution.

The ``q``-Gaussian maximizes Tsallis entropy under a fixed second moment, just
as the Gaussian maximizes Shannon entropy. Its density is

.. math::

    p(x) = \frac{\sqrt{\beta}}{C_q}\,\exp_q\!\big(-\beta x^2\big),

with normalization constant :math:`C_q`. It interpolates between heavy-tailed
distributions (``1 < q < 3``; Student-``t`` like) and compactly supported ones
(``q < 1``), recovering the Gaussian as ``q -> 1``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.scipy.special import gammaln

from qjax.core.functions import q_exp
from qjax.shared.types import Array, Scalar
from qjax.shared.validation import as_scalar_q, near_one


def normalization(q: Scalar) -> jnp.ndarray:
    r"""Normalization constant :math:`C_q` of the unit-``\beta`` ``q``-Gaussian.

    Defined piecewise (see Tsallis, 2009) so that :math:`\int p(x)\,dx = 1`:

    - ``q < 1``:   :math:`C_q = \frac{2\sqrt{\pi}\,\Gamma(1/(1-q))}
      {(3-q)\sqrt{1-q}\,\Gamma(\tfrac{3-q}{2(1-q)})}`
    - ``q = 1``:   :math:`C_q = \sqrt{\pi}`
    - ``1<q<3``:   :math:`C_q = \frac{\sqrt{\pi}\,\Gamma(\tfrac{3-q}{2(q-1)})}
      {\sqrt{q-1}\,\Gamma(1/(q-1))}`

    Args:
        q: Entropic index (scalar), required to satisfy ``q < 3``.

    Returns:
        The scalar normalization constant ``C_q``.
    """
    q = as_scalar_q(q)
    sqrt_pi = jnp.sqrt(jnp.pi)

    def below(qb):  # q < 1
        a = 1.0 - qb
        log_c = (
            jnp.log(2.0) + 0.5 * jnp.log(jnp.pi) + gammaln(1.0 / a)
            - jnp.log(3.0 - qb) - 0.5 * jnp.log(a) - gammaln((3.0 - qb) / (2.0 * a))
        )
        return jnp.exp(log_c)

    def above(qa):  # 1 < q < 3
        b = qa - 1.0
        log_c = (
            0.5 * jnp.log(jnp.pi) + gammaln((3.0 - qa) / (2.0 * b))
            - 0.5 * jnp.log(b) - gammaln(1.0 / b)
        )
        return jnp.exp(log_c)

    # Each branch is evaluated with a sanitized argument that stays strictly
    # inside its own domain, so the unused branch produces neither a NaN value
    # nor a NaN gradient before the final selection by the sign of (q - 1).
    q_safe_below = jnp.where(q < 1.0, q, 0.0)
    q_safe_above = jnp.where(q > 1.0, q, 2.0)
    c_below = below(q_safe_below)
    c_above = above(q_safe_above)
    return jnp.where(q < 1.0, c_below, jnp.where(q > 1.0, c_above, sqrt_pi))


def q_gaussian_pdf(x: Array, q: Scalar = 1.0, beta: Scalar = 1.0) -> jnp.ndarray:
    r"""Density of the ``q``-Gaussian, :math:`\sqrt{\beta}/C_q\,\exp_q(-\beta x^2)`.

    Args:
        x: Evaluation points, any shape.
        q: Entropic index (scalar), ``q < 3``.
        beta: Inverse-width parameter ``beta > 0``.

    Returns:
        Density values, same shape as ``x``.
    """
    x = jnp.asarray(x, dtype=jnp.result_type(float))
    beta = jnp.asarray(beta, dtype=jnp.result_type(float))
    return jnp.sqrt(beta) / normalization(q) * q_exp(-beta * x ** 2, q)


def q_gaussian_logpdf(x: Array, q: Scalar = 1.0, beta: Scalar = 1.0) -> jnp.ndarray:
    r"""Log-density of the ``q``-Gaussian.

    Computed as :math:`\tfrac12\log\beta - \log C_q + \log\!\exp_q(-\beta x^2)`.
    Returns ``-inf`` outside the (compact) support when ``q < 1``, where the
    density vanishes.

    Args:
        x: Evaluation points, any shape.
        q: Entropic index (scalar), ``q < 3``.
        beta: Inverse-width parameter ``beta > 0``.

    Returns:
        Log-density values, same shape as ``x``.
    """
    x = jnp.asarray(x, dtype=jnp.result_type(float))
    beta = jnp.asarray(beta, dtype=jnp.result_type(float))
    log_prefactor = 0.5 * jnp.log(beta) - jnp.log(normalization(q))
    return log_prefactor + jnp.log(q_exp(-beta * x ** 2, q))


def sample(
    key: jax.Array,
    q: Scalar = 1.0,
    beta: Scalar = 1.0,
    shape: tuple[int, ...] = (),
) -> jnp.ndarray:
    r"""Draw ``q``-Gaussian samples for ``1 <= q < 3``.

    For ``1 < q < 3`` the ``q``-Gaussian is a rescaled Student-``t``: with
    :math:`\nu = (3-q)/(q-1)` degrees of freedom,

    .. math::

        X = \frac{T_\nu}{\sqrt{(3-q)\,\beta}}, \qquad T_\nu \sim \mathrm{t}(\nu),

    where :math:`T_\nu = Z/\sqrt{W/\nu}` with :math:`Z \sim \mathcal N(0,1)` and
    :math:`W \sim \chi^2_\nu`. This yields exactly the family variance
    :math:`1/((5-3q)\beta)` for ``q < 5/3``. At ``q = 1`` the Gaussian
    :math:`Z/\sqrt{2\beta}` is returned.

    Args:
        key: A :func:`jax.random.PRNGKey`.
        q: Entropic index (scalar), ``1 <= q < 3``.
        beta: Inverse-width parameter ``beta > 0``.
        shape: Output shape.

    Returns:
        Samples of the requested ``shape``.
    """
    q = as_scalar_q(q)
    beta = jnp.asarray(beta, dtype=jnp.result_type(float))
    k_z, k_w = jax.random.split(key)
    z = jax.random.normal(k_z, shape)

    def gaussian(_):
        return z / jnp.sqrt(2.0 * beta)

    def student_t(_):
        nu = (3.0 - q) / (q - 1.0)
        w = 2.0 * jax.random.gamma(k_w, nu / 2.0, shape)  # chi-square with nu dof
        t = z / jnp.sqrt(w / nu)
        return t / jnp.sqrt((3.0 - q) * beta)

    return jax.lax.cond(near_one(q), gaussian, student_t, operand=None)


__all__ = ["normalization", "q_gaussian_pdf", "q_gaussian_logpdf", "sample"]
