import jax
import jax.numpy as jnp
import pytest

import qjax
from qjax.physics import diffusion as D

jax.config.update("jax_enable_x64", True)


def test_scaling_relations():
    # alpha = 2 / (3 - q): normal diffusion at q = 1, super above, sub below.
    assert float(D.nlfp_exponent(1.0)) == pytest.approx(1.0)
    assert float(D.nlfp_exponent(1.5)) == pytest.approx(4.0 / 3.0)
    assert float(D.nlfp_exponent(0.5)) == pytest.approx(0.8)
    assert float(D.nlfp_exponent(2.0)) == pytest.approx(2.0)  # ballistic
    # nlfp_index is its exact inverse -- the second, independent estimator of q.
    for q in (0.2, 0.5, 1.0, 1.5, 2.0, 2.5):
        assert float(D.nlfp_index(D.nlfp_exponent(q))) == pytest.approx(q, abs=1e-12)


def test_lutz_cold_atom_prediction():
    assert float(D.lutz_q(0.0)) == pytest.approx(1.0)
    assert float(D.lutz_q(0.01)) == pytest.approx(1.44)
    # Vectorized, and monotone in the recoil-to-depth ratio.
    ratios = jnp.array([0.005, 0.01, 0.02])
    assert jnp.allclose(D.lutz_q(ratios), 1.0 + 44.0 * ratios)


def test_saturating_langevin_reproduces_its_exact_stationary_state():
    # dp = -alpha p / (1 + (p/p_c)^2) dt + sqrt(2 D_0) dW has the exact
    # stationary solution P(p) ~ [1 + p^2/p_c^2]^{-alpha p_c^2 / 2 D_0}. If the
    # (q, beta) formulas are right, the q-exponential must reproduce it exactly.
    diffusion, friction, momentum_scale = 0.5, 2.0, 1.3
    q = D.saturating_langevin_q(diffusion, friction, momentum_scale)
    beta = D.saturating_langevin_beta(diffusion, friction)
    momenta = jnp.linspace(-8.0, 8.0, 401)

    exponent = -friction * momentum_scale**2 / (2.0 * diffusion)
    exact = (1.0 + momenta**2 / momentum_scale**2) ** exponent
    assert float(jnp.max(jnp.abs(qjax.q_exp(-beta * momenta**2, q) / exact - 1.0))) < 1e-12
    # A shallower lattice (more diffusion relative to friction) is more heavy-tailed.
    assert float(D.saturating_langevin_q(1.0, 2.0, 1.3)) > float(q)
    # And the classical limit is recovered as the noise vanishes.
    assert float(D.saturating_langevin_q(1e-9, 2.0, 1.3)) == pytest.approx(1.0, abs=1e-8)


def test_nlfp_scaling_beta():
    # beta(t) ~ t^{-alpha}: the solution keeps its shape and only rescales.
    times = jnp.array([1.0, 4.0, 16.0])
    widths = D.nlfp_scaling_beta(times, q=1.5, beta_initial=2.0)
    assert float(widths[0]) == pytest.approx(2.0)
    slope = jnp.diff(jnp.log(widths)) / jnp.diff(jnp.log(times))
    assert jnp.allclose(slope, -D.nlfp_exponent(1.5), atol=1e-12)
    # The width parameter is the inverse of a squared length, so 1/beta grows
    # with exactly the mean-squared-displacement exponent.
    assert float(D.nlfp_scaling_beta(1.0, 1.0, 1.0, reference_time=2.0)) == pytest.approx(2.0)


def test_msd_and_power_law_fit():
    times = jnp.logspace(0.0, 3.0, 60)
    exponent, prefactor, stderr = D.fit_power_law(times, 3.0 * times**1.37)
    assert float(exponent) == pytest.approx(1.37, abs=1e-9)
    assert float(prefactor) == pytest.approx(3.0, rel=1e-9)
    assert float(stderr) == pytest.approx(0.0, abs=1e-12)
    # The window arguments drop a transient.
    noisy = jnp.concatenate([jnp.array([100.0, 50.0]), 3.0 * times[2:] ** 1.37])
    assert float(D.fit_power_law(times, noisy, low=2)[0]) == pytest.approx(1.37, abs=1e-9)


def test_mean_squared_displacement_on_a_brownian_walk():
    steps, walkers, variance = 4000, 20_000, 2e-3
    increments = jax.random.normal(jax.random.PRNGKey(0), (steps, walkers)) * jnp.sqrt(variance)
    trajectory = jnp.cumsum(increments, axis=0)
    indices = jnp.array([100, 200, 400, 800, 1600, 3200])
    snapshots = jnp.concatenate([jnp.zeros((1, walkers)), trajectory[indices]])
    times = jnp.concatenate([jnp.array([1.0]), indices.astype(float)])

    displacement = D.mean_squared_displacement(snapshots)
    assert displacement.shape == (7,)
    assert float(displacement[0]) == 0.0
    # Free Brownian motion: <x^2> = variance * t exactly, so alpha = 1.
    exponent, prefactor, stderr = D.fit_power_law(times[1:], displacement[1:])
    assert float(exponent) == pytest.approx(1.0, abs=4.0 * float(stderr) + 0.01)
    assert float(prefactor) == pytest.approx(variance, rel=0.05)


def test_mean_squared_displacement_multidimensional_and_offset_origin():
    walkers = 1000
    snapshots = jnp.zeros((3, walkers, 2)).at[1].set(1.0).at[2].set(2.0)
    # |x - x_0|^2 summed over both components: 0, 2, 8.
    assert jnp.allclose(D.mean_squared_displacement(snapshots), jnp.array([0.0, 2.0, 8.0]))
    shifted = D.mean_squared_displacement(snapshots, origin=jnp.ones((walkers, 2)))
    assert jnp.allclose(shifted, jnp.array([2.0, 0.0, 2.0]))


def test_histogram_density_is_normalized():
    edges = jnp.linspace(-5.0, 5.0, 101)
    samples = jax.random.normal(jax.random.PRNGKey(1), (200_000,))
    density = D.histogram_density(samples, edges)
    assert density.shape == (100,)
    assert float(jnp.sum(density * jnp.diff(edges))) == pytest.approx(1.0, abs=1e-12)
    # It recovers the generating density.
    centres = 0.5 * (edges[:-1] + edges[1:])
    truth = jnp.exp(-(centres**2) / 2.0) / jnp.sqrt(2.0 * jnp.pi)
    assert float(jnp.max(jnp.abs(density - truth))) < 0.02
    # Non-uniform bins are handled by dividing through the widths.
    uneven = jnp.array([-5.0, -1.0, 0.0, 3.0, 5.0])
    assert float(jnp.sum(D.histogram_density(samples, uneven) * jnp.diff(uneven))) == pytest.approx(
        1.0, abs=1e-12
    )
    # Everything out of range gives zero rather than NaN.
    assert jnp.allclose(D.histogram_density(jnp.array([99.0, -99.0]), edges), 0.0)


def test_interpolate_density_and_scan_compatibility():
    edges = jnp.linspace(-5.0, 5.0, 101)
    samples = jax.random.normal(jax.random.PRNGKey(2), (200_000,))
    density = D.histogram_density(samples, edges)
    assert float(D.interpolate_density(0.0, edges, density)) == pytest.approx(0.399, abs=0.02)
    # Outside the binned range the density is clamped to zero, which is what
    # keeps the nonlinear-Fokker-Planck drift finite in the tails.
    assert float(D.interpolate_density(50.0, edges, density)) == 0.0
    assert float(D.interpolate_density(-50.0, edges, density)) == 0.0

    # Both are called at every step of the particle simulations, so they must
    # work inside a scan with static shapes.
    def step(carry, _):
        estimated = D.histogram_density(carry, edges)
        return carry + 0.01 * D.interpolate_density(carry, edges, estimated), None

    final, _ = jax.lax.scan(jax.jit(step), samples[:1000], None, length=5)
    assert bool(jnp.all(jnp.isfinite(final)))


# --------------------------------------------------------------------------- #
# The exact solution of the nonlinear Fokker-Planck equation
# --------------------------------------------------------------------------- #
DIFFUSIVITY, BETA_INITIAL = 1.0, 100.0


def exact_density(q, diffusivity=DIFFUSIVITY, beta_initial=BETA_INITIAL):
    """Curry `nlfp_density` down to the ``(x, t) -> p`` callable the residual wants."""

    def density(x, t):
        return D.nlfp_density(x, t, q, diffusivity, beta_initial)

    return density


def test_nlfp_width_is_the_heat_kernel_at_q_one():
    # The single sharpest check on the derived rate constant K: at q = 1 the
    # porous-medium equation *is* the heat equation, whose width is 1/(4 D t).
    # If K carried a wrong q-dependent factor, it would have to be one that
    # happens to equal 1 at q = 1 -- and test_nlfp_residual_vanishes below
    # closes that loophole at every other q.
    offset = D.nlfp_offset(1.0, DIFFUSIVITY, BETA_INITIAL)
    assert float(offset) == pytest.approx(1.0 / (4.0 * DIFFUSIVITY * BETA_INITIAL), rel=1e-12)
    for time in (0.0, 0.1, 1.0, 4.0):
        width = D.nlfp_width(time, 1.0, DIFFUSIVITY, BETA_INITIAL)
        assert float(width) == pytest.approx(
            1.0 / (4.0 * DIFFUSIVITY * (time + float(offset))), rel=1e-12
        )
    # And K itself is just 4 D there, because C_1^{1-1} = 1.
    assert float(D.nlfp_rate(1.0, DIFFUSIVITY)) == pytest.approx(4.0 * DIFFUSIVITY, rel=1e-12)


@pytest.mark.parametrize("q", [0.4, 0.5, 0.7, 1.0, 1.3, 1.5, 1.7])
@pytest.mark.parametrize("time", [0.05, 0.5, 1.0])
def test_nlfp_residual_vanishes_on_the_exact_solution(q, time):
    # THE GATE on the whole construction. beta_dot = -K beta^{(5-q)/2} with
    # K = 4 D (2-q) / C_q^{1-q} was derived by hand; this is what proves it.
    # Any error in K, in the exponent, or in the offset shows up here as a
    # residual of order the terms themselves rather than of order eps.
    positions = jnp.linspace(-2.0, 2.0, 41)
    residual = jax.vmap(lambda x: D.nlfp_residual(exact_density(q), x, time, q, DIFFUSIVITY))(
        positions
    )
    assert bool(jnp.all(jnp.isfinite(residual)))
    assert float(jnp.max(jnp.abs(residual))) < 1e-10


def test_nlfp_residual_is_not_vacuously_zero():
    # Feed the residual a density that solves the equation for a *different* q.
    # If the operator were returning zero for everything, this would pass too.
    positions = jnp.linspace(-1.5, 1.5, 31)
    wrong = jax.vmap(lambda x: D.nlfp_residual(exact_density(0.5), x, 0.5, 1.5, DIFFUSIVITY))(
        positions
    )
    assert float(jnp.max(jnp.abs(wrong))) > 1e-2
    # A rescaled density is not a solution either: the equation is nonlinear, so
    # multiplying p by a constant does not commute with it.
    scaled = jax.vmap(
        lambda x: D.nlfp_residual(lambda y, t: 2.0 * exact_density(1.5)(y, t), x, 0.5, 1.5, 1.0)
    )(positions)
    assert float(jnp.max(jnp.abs(scaled))) > 1e-2


@pytest.mark.parametrize("q", [0.4, 0.7, 1.0, 1.3, 1.7])
def test_nlfp_density_conserves_mass(q):
    # The equation is a continuity equation, so the exact solution integrates to
    # one at every time -- and stays a *normalized* q-Gaussian, not just a
    # self-similar shape.
    #
    # The grid has to be wide for the heavy tails: at q = 1.7 the second moment
    # is already infinite (that happens above q = 5/3) and the density decays only
    # as |x|^{-2.86}, so truncating at |x| = 60 loses 5e-3 of the mass. Widening
    # the window is the honest fix; loosening the tolerance would hide it.
    positions = jnp.linspace(-5000.0, 5000.0, 2_000_001)
    for time in (0.0, 0.25, 1.0, 4.0):
        density = D.nlfp_density(positions, time, q, DIFFUSIVITY, BETA_INITIAL)
        assert float(jnp.trapezoid(density, positions)) == pytest.approx(1.0, abs=1e-4)


@pytest.mark.parametrize("q", [0.4, 0.7, 1.0, 1.3, 1.7])
def test_nlfp_density_is_the_q_gaussian_at_the_exact_width(q):
    # The content of `nlfp_density` beyond `nlfp_width` is only that it passes the
    # width through to a normalized q-Gaussian. Assert that directly, so the claim
    # holds at every q including the heavy tails where a quadrature check of the
    # mass is dominated by truncation rather than by the code.
    positions = jnp.linspace(-4.0, 4.0, 201)
    for time in (0.0, 0.5, 2.0):
        width = D.nlfp_width(time, q, DIFFUSIVITY, BETA_INITIAL)
        assert jnp.allclose(
            D.nlfp_density(positions, time, q, DIFFUSIVITY, BETA_INITIAL),
            qjax.q_gaussian_pdf(positions, q, width),
            rtol=0.0,
            atol=0.0,
        )


@pytest.mark.parametrize("q", [0.5, 1.0, 1.5])
def test_nlfp_width_recovers_the_anomalous_exponent(q):
    # beta ~ t^{-alpha} with the same alpha = 2/(3-q) that nlfp_exponent returns,
    # so the exact solution and the scaling relation cannot drift apart.
    times = jnp.logspace(2.0, 5.0, 40)  # far beyond t_star, where the offset is negligible
    width = D.nlfp_width(times, q, DIFFUSIVITY, BETA_INITIAL)
    slope, _, stderr = D.fit_power_law(times, width)
    # beta is a power law in (t + t_star), not in t, so a pure power-law fit
    # carries a bias of order t_star / t_min -- about 1e-4 at q = 1.5, where
    # t_star is largest. That is a property of the exact solution, not an error.
    assert float(slope) == pytest.approx(-float(D.nlfp_exponent(q)), abs=1e-4)
    assert float(stderr) < 1e-4
    # And it agrees with the weaker proportional-only form anchored at one time.
    anchored = D.nlfp_scaling_beta(
        times, q, float(D.nlfp_width(1e3, q, DIFFUSIVITY, BETA_INITIAL)), reference_time=1e3
    )
    assert jnp.allclose(width, anchored, rtol=2e-3)


@pytest.mark.parametrize("q", [0.3, 0.5, 0.8])
def test_nlfp_front_bounds_a_compact_support(q):
    time = 0.4
    front = float(D.nlfp_front(time, q, DIFFUSIVITY, BETA_INITIAL))
    assert jnp.isfinite(front)
    inside = D.nlfp_density(
        jnp.array([0.0, 0.5 * front, 0.99 * front]), time, q, DIFFUSIVITY, BETA_INITIAL
    )
    assert bool(jnp.all(inside > 0.0))
    outside = D.nlfp_density(
        jnp.array([1.01 * front, 2.0 * front, 50.0]), time, q, DIFFUSIVITY, BETA_INITIAL
    )
    assert bool(jnp.all(outside == 0.0))
    # The front is where the q-exponential's own cut-off sits.
    assert front == pytest.approx(
        1.0 / jnp.sqrt((1.0 - q) * D.nlfp_width(time, q, DIFFUSIVITY, BETA_INITIAL)), rel=1e-12
    )


@pytest.mark.parametrize("q", [1.0, 1.4, 1.9])
def test_nlfp_front_is_infinite_without_compact_support(q):
    assert bool(jnp.isinf(D.nlfp_front(0.5, q, DIFFUSIVITY, BETA_INITIAL)))
    # The support really is the whole line, but "positive everywhere" is a
    # statement about the mathematics, not about float64: at q = 1 the Gaussian
    # tail reaches exp(-10^4) by x = 200 and underflows to exactly zero. Only the
    # genuine power-law tails of q > 1 survive that far out.
    reach = 200.0 if q > 1.0 else 20.0
    assert float(D.nlfp_density(reach, 0.5, q, DIFFUSIVITY, BETA_INITIAL)) > 0.0


@pytest.mark.parametrize("q", [1.2, 1.5, 1.8])
def test_nlfp_density_has_the_exact_power_law_tail(q):
    # p ~ |x|^{-2/(q-1)} for q > 1: the feature an exponential parameterization
    # has to learn as a logarithm rather than get structurally.
    time = 0.5
    positions = jnp.logspace(2.0, 4.0, 30)
    density = D.nlfp_density(positions, time, q, DIFFUSIVITY, BETA_INITIAL)
    slope, _, stderr = D.fit_power_law(positions, density)
    assert float(slope) == pytest.approx(-2.0 / (q - 1.0), rel=1e-4)
    assert float(stderr) < 1e-3


@pytest.mark.parametrize("q", [0.4, 0.7, 1.3, 1.7])
def test_deformed_logarithm_of_the_exact_solution_is_quadratic(q):
    # The example's central claim, pinned at the library level. By the q-algebra,
    #   ln_q(A exp_q(-beta x^2)) = ln_q A - beta x^2 [1 + (1-q) ln_q A],
    # so in the coordinates of a q-exponential output head the exact solution is
    # *exactly* a quadratic in x -- no approximation, no fitting slack.
    time = 0.3
    front = float(D.nlfp_front(time, q, DIFFUSIVITY, BETA_INITIAL))
    limit = min(front * 0.95, 3.0) if jnp.isfinite(front) else 3.0
    positions = jnp.linspace(-limit, limit, 121)

    density = D.nlfp_density(positions, time, q, DIFFUSIVITY, BETA_INITIAL)
    deformed = qjax.q_log(density, q)

    # Least squares against {1, x, x^2}; the linear coefficient must vanish too.
    design = jnp.stack([jnp.ones_like(positions), positions, positions**2], axis=-1)
    coefficients, *_ = jnp.linalg.lstsq(design, deformed, rcond=None)
    prediction = design @ coefficients
    spread = float(jnp.max(jnp.abs(deformed - jnp.mean(deformed))))
    assert float(jnp.max(jnp.abs(deformed - prediction))) < 1e-10 * max(spread, 1.0)
    assert float(jnp.abs(coefficients[1])) < 1e-10 * max(spread, 1.0)

    # And the coefficients are the ones the identity predicts.
    width = D.nlfp_width(time, q, DIFFUSIVITY, BETA_INITIAL)
    amplitude = jnp.sqrt(width) / qjax.normalization(q)
    deformed_amplitude = qjax.q_log(amplitude, q)
    assert float(coefficients[0]) == pytest.approx(float(deformed_amplitude), rel=1e-9)
    assert float(coefficients[2]) == pytest.approx(
        float(-width * (1.0 + (1.0 - q) * deformed_amplitude)), rel=1e-9
    )


def test_nlfp_gradients_are_finite_across_a_front():
    # Collocation points fall outside the compact support, where the density is
    # exactly zero. The PINN only works if that region back-propagates a zero
    # gradient rather than NaN.
    q, time = 0.5, 0.4
    front = float(D.nlfp_front(time, q, DIFFUSIVITY, BETA_INITIAL))
    positions = jnp.array([0.0, 0.9 * front, front, 1.1 * front, 5.0 * front])

    space = jax.vmap(jax.grad(lambda x: D.nlfp_density(x, time, q, DIFFUSIVITY, BETA_INITIAL)))
    clock = jax.vmap(
        lambda x: jax.grad(lambda t: D.nlfp_density(x, t, q, DIFFUSIVITY, BETA_INITIAL))(time)
    )
    assert bool(jnp.all(jnp.isfinite(space(positions))))
    assert bool(jnp.all(jnp.isfinite(clock(positions))))
    # Outside the support both derivatives are exactly zero.
    assert float(space(positions)[-1]) == 0.0
    assert float(clock(positions)[-1]) == 0.0


def test_nlfp_rate_rejects_a_degenerate_index():
    # At q = 2 the porous-medium exponent m = 2 - q vanishes and the equation is
    # no longer of this type at all, so a statically-known q >= 2 is a bug.
    for bad in (2.0, 2.5, 3.0):
        with pytest.raises(ValueError, match="must be positive"):
            D.nlfp_rate(bad, DIFFUSIVITY)
    with pytest.raises(ValueError, match="must be positive"):
        D.nlfp_width(1.0, 2.0, DIFFUSIVITY, BETA_INITIAL)


def test_nlfp_helpers_are_jittable_and_differentiable_in_q():
    jitted = jax.jit(lambda q: D.nlfp_width(0.5, q, DIFFUSIVITY, BETA_INITIAL))
    assert float(jitted(0.7)) == pytest.approx(
        float(D.nlfp_width(0.5, 0.7, DIFFUSIVITY, BETA_INITIAL))
    )
    # normalization() is branch-safe, so the width is differentiable in q -- which
    # is what an inverse problem on the entropic index would need.
    gradient = jax.grad(lambda q: D.nlfp_width(0.5, q, DIFFUSIVITY, BETA_INITIAL))(0.7)
    assert jnp.isfinite(gradient)
    assert float(jnp.abs(gradient)) > 0.0
