import jax
import jax.numpy as jnp
import pytest

from qjax.physics import annealing as A

jax.config.update("jax_enable_x64", True)

STEPS = jnp.array([1.0, 2.0, 10.0, 99.0, 1000.0])


def test_visiting_temperature_takes_the_geman_geman_limit():
    # The schedule is 0/0 at q = 1. Written as a ratio of two q-logarithms it
    # returns the Geman-Geman log schedule *exactly*, with no branch on q.
    geman_geman = jnp.log(2.0) / jnp.log1p(STEPS)
    assert float(jnp.max(jnp.abs(A.visiting_temperature(STEPS, 1.0, 1.0) - geman_geman))) < 1e-12
    # And it stays accurate through the neighbourhood of q = 1, where the naive
    # (x**(q-1) - 1) form loses the whole mantissa to cancellation.
    for offset in (1e-9, -1e-9, 1e-6, -1e-6):
        error = jnp.max(jnp.abs(A.visiting_temperature(STEPS, 1.0, 1.0 + offset) - geman_geman))
        assert float(error) < 1e-5


def test_visiting_temperature_has_a_finite_nonzero_q_gradient_at_one():
    # This is the point of routing the schedule through qjax.q_log: a hard
    # where(q == 1, ...) branch would return a q-independent expression and hence
    # a spurious zero gradient, so a learnable cooling index would never move.
    gradient = jax.grad(lambda q: A.visiting_temperature(99.0, 1.0, q))(1.0)
    assert bool(jnp.isfinite(gradient))
    assert float(abs(gradient)) > 1e-3
    # Analytic value: d/dq [ln_{2-q}2 / ln_{2-q}(1+t)] = (L2/Lt)(L2 - Lt)/2.
    log_two, log_step = jnp.log(2.0), jnp.log(100.0)
    expected = (log_two / log_step) * (log_two - log_step) / 2.0
    assert float(gradient) == pytest.approx(float(expected), rel=1e-6)


def test_cauchy_and_faster_schedules():
    # q = 2 is Szu & Hartley's Cauchy machine, T(t) = T(1) / t.
    assert jnp.allclose(A.visiting_temperature(STEPS, 1.0, 2.0), 1.0 / STEPS, atol=1e-12)
    # q = 3 cools as 1/t**2 (the general law is 1/t**(q-1) at large t).
    assert jnp.allclose(A.visiting_temperature(STEPS, 1.0, 3.0), 3.0 / (STEPS * (STEPS + 2.0)))


def test_schedule_starts_at_the_initial_temperature_for_every_index():
    for q in (0.2, 0.5, 1.0, 1.5, 2.0, 2.7, 3.5):
        assert float(A.visiting_temperature(1.0, 3.7, q)) == pytest.approx(3.7, rel=1e-12)


def test_schedule_is_monotone_decreasing_and_faster_for_larger_q():
    fine = jnp.arange(1.0, 200.0)
    previous = None
    for q in (1.0, 1.5, 2.0, 2.7):
        temperatures = A.visiting_temperature(fine, 1.0, q)
        assert bool(jnp.all(jnp.diff(temperatures) < 0.0))
        if previous is not None:
            assert bool(jnp.all(temperatures[1:] < previous[1:]))
        previous = temperatures


def test_schedule_diverges_at_step_zero():
    # t = 0 is outside the schedule's domain; +inf is the correct limit, not an
    # error, so a caller that forgets to start at 1 sees it immediately.
    assert bool(jnp.isinf(A.visiting_temperature(0.0, 1.0, 1.0)))


def test_acceptance_temperature_shares_the_law():
    assert jnp.allclose(
        A.acceptance_temperature(STEPS, 2.0, 1.4), A.tsallis_schedule(STEPS, 2.0, 1.4)
    )
    assert jnp.allclose(
        A.visiting_temperature(STEPS, 2.0, 1.4), A.acceptance_temperature(STEPS, 2.0, 1.4)
    )


def test_schedule_is_jittable_and_vmappable_over_q():
    jitted = jax.jit(A.visiting_temperature)
    assert jnp.allclose(jitted(STEPS, 1.0, 2.0), A.visiting_temperature(STEPS, 1.0, 2.0))
    indices = jnp.array([1.0, 1.5, 2.0, 2.7])
    batched = jax.vmap(lambda q: A.visiting_temperature(STEPS, 1.0, q))(indices)
    assert batched.shape == (4, 5)
    for row, q in zip(batched, indices, strict=True):
        assert jnp.allclose(row, A.visiting_temperature(STEPS, 1.0, float(q)))
