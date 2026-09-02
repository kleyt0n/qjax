r"""Tsallis-Stariolo generalized-simulated-annealing temperature schedules.

Generalized simulated annealing (Tsallis & Stariolo, *Physica A* **233**, 395,
1996) replaces the two Boltzmann ingredients of classical annealing -- the
Gaussian proposal and the exponential acceptance -- by their $q$-deformed
counterparts, and cools with the matching $q$-deformed schedule

$$T_q(t) = T_q(1)\,\frac{2^{q-1} - 1}{(1 + t)^{q-1} - 1}, \qquad t \ge 1.$$

At $q = 2$ this is the Cauchy machine of Szu & Hartley; at $q = 1$ it must
reduce to the Geman-Geman logarithmic schedule $T_1(t) = T_1(1)\ln 2 /
\ln(1+t)$ -- but the expression above is $0/0$ there, exactly the pathology
that `qjax.shared.series` exists to defeat.

The fix needs no new code at all. Since

$$\ln_{2-q}(x) = \frac{x^{q-1} - 1}{q - 1},$$

both $(q-1)$ factors cancel and the whole schedule is a *ratio of two
`qjax.q_log` calls*:

$$T_q(t) = T_q(1)\,\frac{\ln_{2-q} 2}{\ln_{2-q}(1 + t)}.$$

Written this way the classical limit falls out of the same expression with no
branch on $q$, and -- because `qjax.q_log` carries the limit through the
entire function $(e^t-1)/t$ rather than switching to a $q$-independent
formula -- the derivative with respect to $q$ stays correct and non-zero at
$q = 1$. A learnable cooling index is therefore just another parameter.

References:
    Tsallis, C. & Stariolo, D. A. (1996). *Physica A* **233**, 395.
    Szu, H. & Hartley, R. (1987). *Phys. Lett. A* **122**, 157.
    Geman, S. & Geman, D. (1984). *IEEE TPAMI* **6**, 721.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from qjax.core.functions import q_log
from qjax.shared.types import Array, Scalar


def tsallis_schedule(step: Array, initial: Scalar, q: Scalar) -> jax.Array:
    r"""The Tsallis cooling law $T_q(1)\,\ln_{2-q} 2 / \ln_{2-q}(1+t)$.

    Args:
        step: Annealing step ``t``, counted from ``1``; any shape. At ``t = 1``
            the schedule returns ``initial`` for every ``q``. At ``t = 0`` the
            denominator vanishes and the result is ``+inf``, which is the
            correct limit of the schedule rather than an error.
        initial: Temperature at ``t = 1``.
        q: Cooling index. ``q = 1`` is the Geman-Geman logarithmic schedule,
            ``q = 2`` the Cauchy schedule ``initial / t``, and ``q > 2`` cools
            faster still.

    Returns:
        The temperature at each ``step``, same shape as ``step``.
    """
    step = jnp.asarray(step, dtype=jnp.result_type(float))
    index = 2.0 - jnp.asarray(q, dtype=jnp.result_type(float))
    return jnp.asarray(initial, dtype=jnp.result_type(float)) * (
        q_log(2.0, index) / q_log(1.0 + step, index)
    )


def visiting_temperature(step: Array, initial: Scalar, q_visit: Scalar) -> jax.Array:
    r"""Visiting temperature $T_V(t)$ controlling the proposal step length.

    Sets the width of the $q$-Gaussian from which trial moves are drawn. With
    ``q_visit > 1`` the proposal has power-law tails, so the walk mixes
    occasional long Levy-like flights with local moves and escapes a metastable
    basin without waiting for a thermally activated crossing.

    Args:
        step: Annealing step ``t``, counted from ``1``.
        initial: Visiting temperature at ``t = 1``.
        q_visit: Visiting index ``q_V``. ``1`` recovers classical (Boltzmann)
            annealing, ``2`` the Cauchy machine.

    Returns:
        The visiting temperature at each ``step``.
    """
    return tsallis_schedule(step, initial, q_visit)


def acceptance_temperature(step: Array, initial: Scalar, q_accept: Scalar) -> jax.Array:
    r"""Acceptance temperature $T_A(t)$ entering $\exp_{q_A}(-\Delta E / T_A)$.

    Scheduled by the same Tsallis law as the visiting temperature but with its
    own index, so the two deformations -- how far the walk proposes and how
    readily it accepts an uphill move -- can be varied independently.

    Args:
        step: Annealing step ``t``, counted from ``1``.
        initial: Acceptance temperature at ``t = 1``.
        q_accept: Cooling index for the acceptance temperature.

    Returns:
        The acceptance temperature at each ``step``.
    """
    return tsallis_schedule(step, initial, q_accept)


__all__ = ["tsallis_schedule", "visiting_temperature", "acceptance_temperature"]
