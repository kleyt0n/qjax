import jax
import jax.numpy as jnp
import pytest

from qjax.physics import observables as O

jax.config.update("jax_enable_x64", True)


def test_binder_cumulant_limits():
    # Deep in the ordered phase m is a two-delta distribution at +/- m0.
    two_delta = jnp.array([0.7, -0.7, 0.7, -0.7, -0.7, 0.7])
    assert float(O.binder_cumulant(two_delta)) == pytest.approx(2.0 / 3.0)
    # Deep in the disordered phase m is Gaussian, where <m^4> = 3 <m^2>^2.
    gaussian = jax.random.normal(jax.random.PRNGKey(0), (2_000_000,))
    assert float(O.binder_cumulant(gaussian)) == pytest.approx(0.0, abs=5e-3)
    # Scale invariance: the cumulant is dimensionless.
    assert float(O.binder_cumulant(3.5 * gaussian)) == pytest.approx(
        float(O.binder_cumulant(gaussian))
    )


def test_binder_cumulant_reduces_the_requested_axis():
    batch = jax.random.normal(jax.random.PRNGKey(1), (4, 5000))
    assert O.binder_cumulant(batch, axis=-1).shape == (4,)
    assert O.binder_cumulant(batch, axis=0).shape == (5000,)
    # An all-zero magnetization has no scale, so the cumulant is undefined.
    assert bool(jnp.isnan(O.binder_cumulant(jnp.zeros(8))))


def test_crossing_and_peak_temperature_on_planted_curves():
    grid = jnp.linspace(1.0, 4.0, 601)
    centre = 2.2691853

    # A monotone sigmoid crosses 1/2 exactly at its centre.
    sigmoid = 1.0 / (1.0 + jnp.exp((grid - centre) * 8.0))
    assert float(O.crossing_temperature(grid, sigmoid)) == pytest.approx(centre, abs=1e-4)
    # And at another level, where the analytic crossing is known too.
    offset = centre + jnp.log(3.0) / 8.0  # sigmoid = 1/4 there
    assert float(O.crossing_temperature(grid, sigmoid, 0.25)) == pytest.approx(
        float(offset), abs=1e-4
    )
    # A curve that never reaches the level has no crossing.
    assert bool(jnp.isnan(O.crossing_temperature(grid, jnp.zeros_like(grid), 0.5)))

    # A Lorentzian peak: the parabolic refinement beats the grid spacing. The
    # window is wide enough that the curve's own minimum -- which sets the half
    # level -- is negligible, so the measured width is the textbook FWHM.
    wide = jnp.linspace(1.0, 6.0, 1001)
    width = 0.4
    lorentzian = 1.0 / (1.0 + ((wide - centre) / (0.5 * width)) ** 2)
    assert float(O.peak_temperature(wide, lorentzian)) == pytest.approx(centre, abs=1e-3)
    assert float(O.half_width(wide, lorentzian)) == pytest.approx(width, abs=5e-3)


def test_half_width_is_measured_from_the_curves_own_baseline():
    # A peak sitting on a raised background: the half level is (max + min) / 2,
    # so the width is the peak's own, not the width at half of the total height.
    grid = jnp.linspace(-4.0, 4.0, 1601)
    peak = jnp.exp(-(grid**2) / 2.0)
    expected = 2.0 * jnp.sqrt(2.0 * jnp.log(2.0))  # FWHM of a unit Gaussian
    assert float(O.half_width(grid, peak)) == pytest.approx(float(expected), abs=1e-2)
    assert float(O.half_width(grid, peak + 5.0)) == pytest.approx(float(expected), abs=1e-2)


def test_peak_temperature_handles_a_non_uniform_grid():
    # The three-point parabola must not assume equal spacing.
    grid = jnp.array([1.0, 1.2, 1.9, 2.0, 2.1, 2.9, 3.4])
    peak = 2.03
    curve = -((grid - peak) ** 2)
    assert float(O.peak_temperature(grid, curve)) == pytest.approx(peak, abs=1e-9)


def test_peak_temperature_degenerate_and_edge_cases():
    grid = jnp.linspace(0.0, 1.0, 5)
    # A flat curve has no vertex; fall back to the grid maximum.
    assert jnp.isfinite(O.peak_temperature(grid, jnp.ones_like(grid)))
    # A maximum at the boundary is clipped inward rather than reading out of range.
    rising = grid
    assert float(O.peak_temperature(grid, rising)) <= float(grid[-1]) + 1e-9


def test_half_width_needs_the_peak_bracketed():
    grid = jnp.linspace(0.0, 1.0, 21)
    assert bool(jnp.isnan(O.half_width(grid, grid)))  # monotone: never comes back down


def test_finite_size_extrapolation_recovers_a_planted_intercept():
    sizes = jnp.array([8.0, 10.0, 12.0, 16.0])
    intercept, slope, stderr = O.finite_size_extrapolation(sizes, 2.2 + 0.7 / sizes, nu=1.0)
    assert float(intercept) == pytest.approx(2.2, abs=1e-10)
    assert float(slope) == pytest.approx(0.7, abs=1e-10)
    assert float(stderr) == pytest.approx(0.0, abs=1e-12)

    # With nu != 1 the abscissa changes, and the planted law must match it.
    nu = 0.63
    exact = 2.2 + 0.7 * sizes ** (-1.0 / nu)
    intercept, slope, _ = O.finite_size_extrapolation(sizes, exact, nu=nu)
    assert float(intercept) == pytest.approx(2.2, abs=1e-10)
    assert float(slope) == pytest.approx(0.7, abs=1e-10)


def test_finite_size_extrapolation_reports_a_nonzero_error_when_noisy():
    sizes = jnp.array([8.0, 10.0, 12.0, 16.0, 20.0])
    noisy = 2.2 + 0.7 / sizes + jnp.array([0.01, -0.01, 0.008, -0.006, 0.004])
    intercept, _, stderr = O.finite_size_extrapolation(sizes, noisy)
    assert float(stderr) > 0.0
    assert float(abs(intercept - 2.2)) < 4.0 * float(stderr)


def test_estimators_are_jittable():
    grid = jnp.linspace(1.0, 4.0, 101)
    curve = jnp.exp(-((grid - 2.5) ** 2))
    for function in (O.peak_temperature, O.half_width):
        assert float(jax.jit(function)(grid, curve)) == pytest.approx(float(function(grid, curve)))
    assert float(jax.jit(O.crossing_temperature)(grid, curve, 0.5)) == pytest.approx(
        float(O.crossing_temperature(grid, curve, 0.5))
    )
