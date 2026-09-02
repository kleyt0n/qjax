r"""Anomalous diffusion: scaling relations and estimators for the entropic index.

Nonextensive statistics makes a *falsifiable* prediction about anomalous
diffusion. The nonlinear (porous-medium) Fokker-Planck equation

$$\frac{\partial p}{\partial t} = D\,\frac{\partial^2 p^{\,2-q}}{\partial x^2}$$

has the self-similar $q$-Gaussian solution of Tsallis & Bukman (1996), whose
width obeys $\langle x^2 \rangle \propto t^{\alpha}$ with

$$\alpha = \frac{2}{3 - q}.$$

So the entropic index and the anomalous diffusion exponent are not two
independent fitted parameters: measuring the shape of the distribution predicts
the growth of its width, and vice versa. That is what makes $q$ a *measured
physical quantity* here rather than a hyperparameter -- and it is checkable, in
this module, by two independent estimators.

A second, experimentally realized mechanism yields the same distribution from a
*linear*-noise Langevin equation with saturating (Sisyphus) friction,

$$dp = -\frac{\alpha\,p}{1 + (p/p_c)^2}\,dt + \sqrt{2 D_0}\,dW,$$

whose exact stationary solution is $P(p) \propto [1 + p^2/p_c^2]^{-\alpha
p_c^2 / (2 D_0)}$, a $q$-Gaussian with

$$q = 1 + \frac{2 D_0}{\alpha\,p_c^2}, \qquad \beta = \frac{\alpha}{2 D_0}.$$

This is the cold-atom case: Lutz (2003) evaluated the three coefficients
semiclassically for atoms in a dissipative optical lattice and obtained
$q = 1 + 44 E_R / U_0$, confirmed experimentally by Douglas, Bergamini &
Renzoni (2006).

References:
    Plastino, A. R. & Plastino, A. (1995). *Physica A* **222**, 347.
    Tsallis, C. & Bukman, D. J. (1996). *Phys. Rev. E* **54**, R2197.
    Lutz, E. (2003). *Phys. Rev. A* **67**, 051402(R).
    Douglas, P., Bergamini, S. & Renzoni, F. (2006). *Phys. Rev. Lett.* **96**,
        110601.
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp

from qjax.core.distributions import normalization, q_gaussian_pdf
from qjax.shared.types import Array, Scalar

#: Largest entropic index for which the porous-medium exponent ``m = 2 - q`` is
#: positive. At ``q = 2`` the equation degenerates (``m = 0``) and above it the
#: diffusion term changes character entirely, so the closed-form solution below
#: is restricted to ``q < 2``.
NLFP_MAX_INDEX: float = 2.0


def nlfp_exponent(q: Scalar) -> jax.Array:
    r"""Anomalous diffusion exponent $\alpha = 2/(3-q)$ of the nonlinear FP equation.

    ``q = 1`` gives normal diffusion (``alpha = 1``), ``q > 1`` superdiffusion
    and ``q < 1`` subdiffusion.

    Args:
        q: Entropic index, ``q < 3``.

    Returns:
        The exponent ``alpha`` in ``<x^2> ~ t**alpha``.
    """
    return 2.0 / (3.0 - jnp.asarray(q, dtype=jnp.result_type(float)))


def nlfp_index(exponent: Scalar) -> jax.Array:
    r"""Invert `nlfp_exponent`: $q = 3 - 2/\alpha$.

    The second, independent route to the entropic index. It is the *only* route
    for a subdiffusive ($q < 1$) process, because the $q$-Gaussian then has
    compact support: the log-likelihood is ``-inf`` outside it, so a
    gradient-based fit initialized at ``q > 1`` can never cross into ``q < 1``.

    Args:
        exponent: Measured anomalous exponent ``alpha``.

    Returns:
        The entropic index implied by ``alpha``.
    """
    exponent = jnp.asarray(exponent, dtype=jnp.result_type(float))
    return 3.0 - 2.0 / exponent


def nlfp_scaling_beta(
    time: Array, q: Scalar, beta_initial: Scalar, reference_time: Scalar = 1.0
) -> jax.Array:
    r"""Self-similar width parameter $\beta(t)$ of the Tsallis-Bukman solution.

    The solution keeps its $q$-Gaussian *shape* for all time and only
    rescales, with $\beta(t) \propto t^{-\alpha}$ and the same
    $\alpha = 2/(3-q)$ that governs the mean-squared displacement. The
    diffusivity $D$ enters only through the pair
    ``(beta_initial, reference_time)``, which the initial condition fixes, so it
    is not an argument here.

    Args:
        time: Time(s) at which to evaluate the width, any shape.
        q: Entropic index.
        beta_initial: Width parameter at ``reference_time``.
        reference_time: Time at which ``beta_initial`` is quoted.

    Returns:
        ``beta(t)``, same shape as ``time``.
    """
    time = jnp.asarray(time, dtype=jnp.result_type(float))
    scale = time / jnp.asarray(reference_time, dtype=jnp.result_type(float))
    return jnp.asarray(beta_initial, dtype=jnp.result_type(float)) * scale ** (-nlfp_exponent(q))


def nlfp_rate(q: Scalar, diffusivity: Scalar) -> jax.Array:
    r"""Rate constant $K$ of the width ODE $\dot\beta = -K\,\beta^{(5-q)/2}$.

    Substituting the normalized $q$-Gaussian
    $p = (\sqrt\beta / C_q)\exp_q(-\beta x^2)$ into
    $\partial_t p = D\,\partial_{xx} p^{\,2-q}$ makes every $x$-dependent
    factor cancel identically, leaving a scalar ordinary differential equation
    for the width alone with

    $$K = \frac{4 D (2-q)}{C_q^{\,1-q}},$$

    where $C_q$ is `qjax.normalization`. At $q = 1$ this is $K = 4D$, and
    `nlfp_width` then reduces to the heat kernel exactly -- which is the check
    that pins the constant.

    Args:
        q: Entropic index, strictly below `NLFP_MAX_INDEX`.
        diffusivity: The coefficient ``D`` in the equation.

    Returns:
        The scalar rate constant ``K``, with the dimensions of ``D``.

    Raises:
        ValueError: If ``q`` is a concrete value at or above
            `NLFP_MAX_INDEX`, where the equation is no longer of
            porous-medium type.
    """
    q = jnp.asarray(q, dtype=jnp.result_type(float))
    try:
        static = float(q)
    except (TypeError, jax.errors.ConcretizationTypeError, jax.errors.TracerArrayConversionError):
        pass
    else:
        if static >= NLFP_MAX_INDEX:
            raise ValueError(
                f"the porous-medium exponent 2 - q must be positive, so q < "
                f"{NLFP_MAX_INDEX}; got {static}."
            )
    diffusivity = jnp.asarray(diffusivity, dtype=jnp.result_type(float))
    return 4.0 * diffusivity * (2.0 - q) / normalization(q) ** (1.0 - q)


def nlfp_offset(q: Scalar, diffusivity: Scalar, beta_initial: Scalar) -> jax.Array:
    r"""Time offset $t_\star$ placing width $\beta_0$ at $t = 0$.

    The self-similar solution is a power law in $t + t_\star$; the offset is what
    replaces the singular point-source initial condition by a $q$-Gaussian of
    finite width, so that ``t = 0`` is an ordinary regular point.

    Args:
        q: Entropic index, strictly below `NLFP_MAX_INDEX`.
        diffusivity: The coefficient ``D``.
        beta_initial: Width parameter at ``t = 0``.

    Returns:
        The scalar offset ``t_star``.
    """
    q = jnp.asarray(q, dtype=jnp.result_type(float))
    beta_initial = jnp.asarray(beta_initial, dtype=jnp.result_type(float))
    rate = nlfp_rate(q, diffusivity)
    return 2.0 / ((3.0 - q) * rate) * beta_initial ** (-(3.0 - q) / 2.0)


def nlfp_width(
    time: Array,
    q: Scalar,
    diffusivity: Scalar,
    beta_initial: Scalar,
    initial_time: Scalar = 0.0,
) -> jax.Array:
    r"""Exact width parameter $\beta(t)$ of the Tsallis-Bukman solution.

    Solving $\dot\beta = -K\beta^{(5-q)/2}$ from `nlfp_rate` gives

    $$\beta(t) = \Big[\tfrac{3-q}{2}\,K\,(t - t_0 + t_\star)\Big]^{-\frac{2}{3-q}},$$

    with $t_\star$ from `nlfp_offset` chosen so that
    $\beta(t_0) = \beta_0$.

    This is the full solution, diffusivity included. `nlfp_scaling_beta` is the
    weaker statement -- the same power law with the prefactor left to the initial
    condition -- and is what to use when ``D`` is unknown.

    Two properties worth knowing, both pinned by the test suite: at $q = 1$
    this is exactly the heat kernel width $1/(4D(t + t_\star))$, and for every
    $q$ it decays as $t^{-\alpha}$ with the same
    $\alpha = 2/(3-q)$ that `nlfp_exponent` returns.

    Args:
        time: Time(s) at which to evaluate the width, any shape.
        q: Entropic index, strictly below `NLFP_MAX_INDEX`.
        diffusivity: The coefficient ``D``.
        beta_initial: Width parameter at ``initial_time``.
        initial_time: Time at which ``beta_initial`` is quoted.

    Returns:
        ``beta(t)``, same shape as ``time``.
    """
    time = jnp.asarray(time, dtype=jnp.result_type(float))
    q = jnp.asarray(q, dtype=jnp.result_type(float))
    rate = nlfp_rate(q, diffusivity)
    offset = nlfp_offset(q, diffusivity, beta_initial)
    elapsed = time - jnp.asarray(initial_time, dtype=jnp.result_type(float)) + offset
    return (0.5 * (3.0 - q) * rate * elapsed) ** (-2.0 / (3.0 - q))


def nlfp_density(
    x: Array,
    time: Array,
    q: Scalar,
    diffusivity: Scalar,
    beta_initial: Scalar,
    initial_time: Scalar = 0.0,
) -> jax.Array:
    r"""The exact solution of the nonlinear Fokker-Planck equation.

    A normalized $q$-Gaussian whose width follows `nlfp_width`:

    $$p(x, t) = \frac{\sqrt{\beta(t)}}{C_q}\,\exp_q\!\big(-\beta(t)\,x^2\big).$$

    The shape never changes -- only the width -- which is what "self-similar"
    means here, and is why one scalar ODE captures the whole solution of a
    nonlinear partial differential equation.

    Args:
        x: Position(s), broadcast against ``time``.
        time: Time(s), broadcast against ``x``.
        q: Entropic index, strictly below `NLFP_MAX_INDEX`.
        diffusivity: The coefficient ``D``.
        beta_initial: Width parameter at ``initial_time``.
        initial_time: Time at which ``beta_initial`` is quoted.

    Returns:
        The density, broadcast over ``x`` and ``time``. Exactly zero beyond
        `nlfp_front` when ``q < 1``.
    """
    width = nlfp_width(time, q, diffusivity, beta_initial, initial_time)
    return q_gaussian_pdf(x, q, width)


def nlfp_front(
    time: Array,
    q: Scalar,
    diffusivity: Scalar,
    beta_initial: Scalar,
    initial_time: Scalar = 0.0,
) -> jax.Array:
    r"""Edge of the support, $x_f(t) = 1/\sqrt{(1-q)\beta(t)}$, for $q < 1$.

    Below $q = 1$ the $q$-Gaussian is compactly supported, so the solution has
    a genuine moving free boundary: the density is *exactly* zero beyond
    $x_f$, not merely small. That front is the sharpest thing to measure a
    numerical solution against, and it is the feature a strictly positive
    parameterization cannot represent at all.

    Args:
        time: Time(s) at which to locate the front, any shape.
        q: Entropic index, strictly below `NLFP_MAX_INDEX`.
        diffusivity: The coefficient ``D``.
        beta_initial: Width parameter at ``initial_time``.
        initial_time: Time at which ``beta_initial`` is quoted.

    Returns:
        The front position, same shape as ``time``; ``+inf`` for ``q >= 1``,
        where the support is the whole line.
    """
    q = jnp.asarray(q, dtype=jnp.result_type(float))
    width = nlfp_width(time, q, diffusivity, beta_initial, initial_time)
    compact = q < 1.0
    # Sanitize before the reciprocal square root so the unselected branch
    # contributes neither a NaN value nor a NaN gradient.
    safe = jnp.where(compact, (1.0 - q) * width, 1.0)
    return jnp.where(compact, 1.0 / jnp.sqrt(safe), jnp.inf)


def nlfp_residual(
    density_fn: Callable[[jax.Array, jax.Array], jax.Array],
    x: Scalar,
    time: Scalar,
    q: Scalar,
    diffusivity: Scalar,
) -> jax.Array:
    r"""Residual $\partial_t p - D\,\partial_{xx} p^{\,2-q}$ of a candidate solution.

    Takes a callable mapping a scalar ``(x, t)`` to a scalar density and
    differentiates it, so one operator serves two purposes: it *validates* the
    exact solution (its residual must vanish, which is what gates the derivation
    of `nlfp_rate`) and it *trains* a neural network (the residual is the loss).
    Vectorize over collocation points with `jax.vmap`.

    The second derivative is taken of the pressure variable $v = p^{\,2-q}$
    directly, matching the equation as written. Near a $q < 1$ front,
    $p \sim (x_f - x)^{1/(1-q)}$ gives
    $v \sim (x_f - x)^{(2-q)/(1-q)}$ -- cubic at $q = 1/2$ -- so
    $v''$ is continuous and vanishing there rather than singular.

    Args:
        density_fn: A callable ``(x, t) -> p`` on scalars.
        x: Position at which to evaluate the residual.
        time: Time at which to evaluate the residual.
        q: Entropic index, strictly below `NLFP_MAX_INDEX`.
        diffusivity: The coefficient ``D``.

    Returns:
        The scalar residual. Zero for an exact solution.
    """
    x = jnp.asarray(x, dtype=jnp.result_type(float))
    time = jnp.asarray(time, dtype=jnp.result_type(float))
    exponent = 2.0 - jnp.asarray(q, dtype=jnp.result_type(float))

    def pressure(position: jax.Array, instant: jax.Array) -> jax.Array:
        density = density_fn(position, instant)
        # Sanitize before the fractional power: at a q < 1 front the density is
        # exactly zero, and ``0 ** exponent`` would back-propagate NaN even
        # though the value itself is fine.
        positive = density > 0.0
        return jnp.where(positive, jnp.where(positive, density, 1.0) ** exponent, 0.0)

    time_derivative = jax.grad(density_fn, argnums=1)(x, time)
    curvature = jax.grad(jax.grad(pressure, argnums=0), argnums=0)(x, time)
    return time_derivative - jnp.asarray(diffusivity, dtype=jnp.result_type(float)) * curvature


def saturating_langevin_q(diffusion: Scalar, friction: Scalar, momentum_scale: Scalar) -> jax.Array:
    r"""Entropic index $q = 1 + 2 D_0 / (\alpha p_c^2)$ of the Sisyphus Langevin process.

    Exact for the stationary state of ``dp = -alpha p / (1 + (p/p_c)**2) dt +
    sqrt(2 D_0) dW``. Because the three coefficients are chosen by the caller,
    this is a *rigorous internal reference* for a fit: the true ``q`` is known.

    Args:
        diffusion: Momentum diffusion coefficient ``D_0``.
        friction: Friction coefficient ``alpha`` at small momentum.
        momentum_scale: Saturation momentum ``p_c``.

    Returns:
        The exact stationary entropic index.
    """
    diffusion = jnp.asarray(diffusion, dtype=jnp.result_type(float))
    friction = jnp.asarray(friction, dtype=jnp.result_type(float))
    momentum_scale = jnp.asarray(momentum_scale, dtype=jnp.result_type(float))
    return 1.0 + 2.0 * diffusion / (friction * momentum_scale**2)


def saturating_langevin_beta(diffusion: Scalar, friction: Scalar) -> jax.Array:
    r"""Width parameter $\beta = \alpha / (2 D_0)$ of the Sisyphus stationary state.

    The companion of `saturating_langevin_q`: together they specify the exact
    stationary $q$-Gaussian, so both fitted parameters have a known target.

    Args:
        diffusion: Momentum diffusion coefficient ``D_0``.
        friction: Friction coefficient ``alpha`` at small momentum.

    Returns:
        The exact stationary width parameter.
    """
    friction = jnp.asarray(friction, dtype=jnp.result_type(float))
    diffusion = jnp.asarray(diffusion, dtype=jnp.result_type(float))
    return friction / (2.0 * diffusion)


def lutz_q(recoil_over_depth: Array) -> jax.Array:
    r"""Lutz's cold-atom prediction $q = 1 + 44 E_R / U_0$.

    Obtained by evaluating the three Sisyphus coefficients semiclassically for
    atoms in a dissipative optical lattice of depth $U_0$ and recoil energy
    $E_R$; confirmed experimentally by Douglas, Bergamini & Renzoni (2006).
    Unlike `saturating_langevin_q`, this is a prediction about a *real
    experiment*, not about a simulation.

    Args:
        recoil_over_depth: The ratio ``E_R / U_0``, any shape.

    Returns:
        The predicted entropic index, same shape as the input.
    """
    return 1.0 + 44.0 * jnp.asarray(recoil_over_depth, dtype=jnp.result_type(float))


def mean_squared_displacement(snapshots: Array, origin: Array | None = None) -> jax.Array:
    """Ensemble mean-squared displacement from a set of snapshots.

    Args:
        snapshots: Positions of shape ``(T, P)`` for one dimension or
            ``(T, P, D)`` for ``D`` dimensions, where ``P`` is the number of
            independent walkers.
        origin: Starting positions, shape matching one snapshot. Defaults to
            ``snapshots[0]``.

    Returns:
        The mean-squared displacement at each snapshot time, shape ``(T,)``.
    """
    snapshots = jnp.asarray(snapshots, dtype=jnp.result_type(float))
    start = snapshots[0] if origin is None else jnp.asarray(origin, dtype=snapshots.dtype)
    offsets = snapshots - start
    squared = offsets**2 if snapshots.ndim == 2 else jnp.sum(offsets**2, axis=-1)
    return jnp.mean(squared, axis=-1)


def fit_power_law(
    x: Array, y: Array, low: int = 0, high: int | None = None
) -> tuple[jax.Array, jax.Array, jax.Array]:
    r"""Least-squares fit of $y = c\,x^{a}$ on a log-log scale.

    Args:
        x: Abscissa, strictly positive, shape ``(n,)``.
        y: Ordinate, strictly positive, shape ``(n,)``.
        low: First index to include; use it to drop an early transient.
        high: One past the last index to include. Defaults to ``n``.

    Returns:
        ``(exponent, prefactor, exponent_stderr)``. The standard error is the
        usual OLS one and is ``0`` for a perfect power law.
    """
    log_x = jnp.log(jnp.asarray(x, dtype=jnp.result_type(float))[low:high])
    log_y = jnp.log(jnp.asarray(y, dtype=jnp.result_type(float))[low:high])
    count = log_x.shape[0]

    x_mean, y_mean = jnp.mean(log_x), jnp.mean(log_y)
    sxx = jnp.sum((log_x - x_mean) ** 2)
    exponent = jnp.sum((log_x - x_mean) * (log_y - y_mean)) / sxx
    intercept = y_mean - exponent * x_mean

    residual = log_y - (intercept + exponent * log_x)
    dof = max(count - 2, 1)
    stderr = jnp.sqrt(jnp.sum(residual**2) / dof / sxx)
    return exponent, jnp.exp(intercept), stderr


def histogram_density(samples: Array, edges: Array) -> jax.Array:
    """Normalized histogram density over the given bin edges.

    Written with a scatter-add rather than `jax.numpy.histogram` so it is
    cheap to call inside a `jax.lax.scan` -- the particle simulations need the
    density at every step, since the nonlinear Fokker-Planck drift depends on it.

    Args:
        samples: Values to bin, any shape (flattened).
        edges: Monotone bin edges, shape ``(B + 1,)``.

    Returns:
        Density in each bin, shape ``(B,)``, integrating to ``1`` against the
        bin widths (or all zeros if no sample falls inside).
    """
    samples = jnp.asarray(samples, dtype=jnp.result_type(float)).reshape(-1)
    edges = jnp.asarray(edges, dtype=jnp.result_type(float))
    bins = edges.shape[0] - 1
    widths = jnp.diff(edges)

    index = jnp.searchsorted(edges, samples, side="right") - 1
    inside = (index >= 0) & (index < bins)
    counts = (
        jnp.zeros(bins, dtype=samples.dtype)
        .at[jnp.where(inside, index, 0)]
        .add(jnp.where(inside, 1.0, 0.0))
    )
    total = jnp.sum(counts)
    return jnp.where(total > 0.0, counts / jnp.where(total > 0.0, total, 1.0) / widths, 0.0)


def interpolate_density(x: Array, edges: Array, density: Array) -> jax.Array:
    """Evaluate a binned density at arbitrary points by linear interpolation.

    Args:
        x: Points at which to evaluate, any shape.
        edges: Bin edges the density was built on, shape ``(B + 1,)``.
        density: Density per bin, shape ``(B,)``.

    Returns:
        Interpolated density, same shape as ``x``, clamped to ``0`` outside the
        binned range.
    """
    edges = jnp.asarray(edges, dtype=jnp.result_type(float))
    centres = 0.5 * (edges[:-1] + edges[1:])
    return jnp.interp(
        jnp.asarray(x, dtype=jnp.result_type(float)),
        centres,
        jnp.asarray(density, dtype=jnp.result_type(float)),
        left=0.0,
        right=0.0,
    )


__all__ = [
    "NLFP_MAX_INDEX",
    "nlfp_exponent",
    "nlfp_index",
    "nlfp_scaling_beta",
    "nlfp_rate",
    "nlfp_offset",
    "nlfp_width",
    "nlfp_density",
    "nlfp_front",
    "nlfp_residual",
    "saturating_langevin_q",
    "saturating_langevin_beta",
    "lutz_q",
    "mean_squared_displacement",
    "fit_power_law",
    "histogram_density",
    "interpolate_density",
]
