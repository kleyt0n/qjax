r"""Estimators for locating and characterizing a phase transition.

These are the finite-size-scaling tools that turn a family of curves measured at
several lattice sizes into a number that can be compared against an exact
critical temperature or exponent. They are deliberately model-agnostic: the
input is a curve over a temperature grid, whether it came from a Monte Carlo
observable or from a neural network's output.

All of them are jittable and take static shapes; the index searches are written
with masks rather than Python control flow so they work under `jax.jit`.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from qjax.shared.types import Array, Scalar


def binder_cumulant(magnetization: Array, axis: int = -1) -> jax.Array:
    r"""Binder fourth-order cumulant $U_4 = 1 - \langle m^4\rangle / (3\langle m^2\rangle^2)$.

    $U_4$ is dimensionless at the critical point, so curves measured at
    different lattice sizes cross there -- the standard way to locate $T_c$
    without knowing it. It equals $2/3$ deep in the ordered phase (where
    $m$ is a two-delta distribution at $\pm m_0$) and $0$ in the
    disordered phase (where $m$ is Gaussian).

    Args:
        magnetization: Per-configuration magnetizations.
        axis: Axis to average over.

    Returns:
        The cumulant, with ``axis`` reduced.
    """
    magnetization = jnp.asarray(magnetization, dtype=jnp.result_type(float))
    second = jnp.mean(magnetization**2, axis=axis)
    fourth = jnp.mean(magnetization**4, axis=axis)
    safe = jnp.where(second == 0.0, 1.0, second)
    return jnp.where(second == 0.0, jnp.nan, 1.0 - fourth / (3.0 * safe**2))


def _interpolate(
    x: jax.Array, y: jax.Array, low: jax.Array, high: jax.Array, level: Scalar
) -> jax.Array:
    """Linearly interpolate the abscissa at which ``y`` crosses ``level``."""
    y_low, y_high = y[low], y[high]
    span = y_high - y_low
    fraction = jnp.where(span == 0.0, 0.0, (level - y_low) / jnp.where(span == 0.0, 1.0, span))
    return x[low] + fraction * (x[high] - x[low])


def crossing_temperature(temperatures: Array, curve: Array, level: Scalar = 0.5) -> jax.Array:
    """Temperature of the first crossing of ``level``, by linear interpolation.

    Used to read a transition temperature off a monotone indicator, such as the
    probability a classifier assigns to the ordered phase.

    Args:
        temperatures: Strictly ordered temperature grid, shape ``(T,)``.
        curve: Values on that grid, shape ``(T,)``.
        level: The level to cross.

    Returns:
        A 0-d array with the crossing temperature, or ``NaN`` if the curve never
        crosses ``level`` on the grid.
    """
    x = jnp.asarray(temperatures, dtype=jnp.result_type(float))
    y = jnp.asarray(curve, dtype=jnp.result_type(float))
    above = y >= level
    changes = above[:-1] != above[1:]
    index = jnp.argmax(changes)
    crossed = jnp.any(changes)
    return jnp.where(crossed, _interpolate(x, y, index, index + 1, level), jnp.nan)


def peak_temperature(temperatures: Array, curve: Array) -> jax.Array:
    """Temperature of a curve's maximum, refined by a three-point parabola.

    Reads off a *pseudo-critical* temperature from a peaked indicator (a
    susceptibility, a heat capacity, or the Tsallis entropy of a classifier's
    output). The parabolic vertex through the grid maximum and its two
    neighbours recovers sub-grid resolution and handles a non-uniform grid.

    Args:
        temperatures: Ordered temperature grid, shape ``(T,)`` with ``T >= 3``.
        curve: Values on that grid, shape ``(T,)``.

    Returns:
        A 0-d array with the peak temperature.
    """
    x = jnp.asarray(temperatures, dtype=jnp.result_type(float))
    y = jnp.asarray(curve, dtype=jnp.result_type(float))
    centre = jnp.clip(jnp.argmax(y), 1, x.shape[0] - 2)
    x1, x2, x3 = x[centre - 1], x[centre], x[centre + 1]
    y1, y2, y3 = y[centre - 1], y[centre], y[centre + 1]

    d1 = (x1 - x2) * (x1 - x3)
    d2 = (x2 - x1) * (x2 - x3)
    d3 = (x3 - x1) * (x3 - x2)
    quadratic = y1 / d1 + y2 / d2 + y3 / d3
    linear = -(y1 * (x2 + x3) / d1 + y2 * (x1 + x3) / d2 + y3 * (x1 + x2) / d3)

    degenerate = quadratic == 0.0
    safe = jnp.where(degenerate, 1.0, quadratic)
    return jnp.where(degenerate, x2, -0.5 * linear / safe)


def half_width(temperatures: Array, curve: Array) -> jax.Array:
    r"""Full width at half maximum of a peaked curve.

    The half level is taken relative to the curve's own minimum,
    $\tfrac12(\max + \min)$, so a peak sitting on a non-zero background is
    measured correctly. For a critical indicator the width scales as
    $w(L) \sim L^{-1/\nu}$, which is how the examples recover $\nu$.

    Args:
        temperatures: Ordered temperature grid, shape ``(T,)``.
        curve: Values on that grid, shape ``(T,)``.

    Returns:
        A 0-d array with the width, or ``NaN`` if the curve does not fall back
        below the half level on both sides of its maximum.
    """
    x = jnp.asarray(temperatures, dtype=jnp.result_type(float))
    y = jnp.asarray(curve, dtype=jnp.result_type(float))
    size = y.shape[0]
    level = 0.5 * (jnp.max(y) + jnp.min(y))

    peak = jnp.argmax(y)
    index = jnp.arange(size)
    below = y < level
    left = jnp.max(jnp.where(below & (index < peak), index, -1))
    right = jnp.min(jnp.where(below & (index > peak), index, size))

    bracketed = (left >= 0) & (right < size)
    safe_left = jnp.clip(left, 0, size - 2)
    safe_right = jnp.clip(right, 1, size - 1)
    lower = _interpolate(x, y, safe_left, safe_left + 1, level)
    upper = _interpolate(x, y, safe_right - 1, safe_right, level)
    return jnp.where(bracketed, upper - lower, jnp.nan)


def finite_size_extrapolation(
    sizes: Array, estimates: Array, nu: Scalar = 1.0
) -> tuple[jax.Array, jax.Array, jax.Array]:
    r"""Extrapolate a size-dependent estimate to the thermodynamic limit.

    Fits $T_c(L) = T_c(\infty) + a L^{-1/\nu}$ by ordinary least squares in
    $x = L^{-1/\nu}$ -- the leading finite-size correction for a
    pseudo-critical temperature. With the exact $\nu$ supplied, the intercept
    is the quantity to compare against the exact $T_c$.

    Args:
        sizes: Linear lattice sizes ``L``, shape ``(S,)``.
        estimates: The size-dependent estimates, shape ``(S,)``.
        nu: Correlation-length exponent used to build the abscissa.

    Returns:
        ``(intercept, slope, intercept_stderr)``. The standard error is the
        usual OLS one and is ``0`` for a perfect fit.
    """
    x = jnp.asarray(sizes, dtype=jnp.result_type(float)) ** (-1.0 / nu)
    y = jnp.asarray(estimates, dtype=jnp.result_type(float))
    count = x.shape[0]

    x_mean, y_mean = jnp.mean(x), jnp.mean(y)
    sxx = jnp.sum((x - x_mean) ** 2)
    sxy = jnp.sum((x - x_mean) * (y - y_mean))
    slope = sxy / sxx
    intercept = y_mean - slope * x_mean

    residual = y - (intercept + slope * x)
    dof = max(count - 2, 1)
    variance = jnp.sum(residual**2) / dof
    stderr = jnp.sqrt(variance * (1.0 / count + x_mean**2 / sxx))
    return intercept, slope, stderr


__all__ = [
    "binder_cumulant",
    "crossing_temperature",
    "peak_temperature",
    "half_width",
    "finite_size_extrapolation",
]
