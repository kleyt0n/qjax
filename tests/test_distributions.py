import jax
import jax.numpy as jnp
import pytest

from qjax.core.distributions import (
    normalization,
    q_gaussian_logpdf,
    q_gaussian_pdf,
    sample,
)

jax.config.update("jax_enable_x64", True)


# Light/moderate tails: a finite grid captures essentially all the mass. Heavier
# tails (q >= 2) are checked analytically below instead of by truncated quadrature.
@pytest.mark.parametrize("q", [0.0, 0.5, 1.0, 1.5])
@pytest.mark.parametrize("beta", [0.5, 1.0, 2.0])
def test_pdf_integrates_to_one(q, beta):
    x = jnp.linspace(-60.0, 60.0, 400_001)
    pdf = q_gaussian_pdf(x, q, beta)
    mass = jnp.trapezoid(pdf, x)
    assert jnp.allclose(mass, 1.0, atol=2e-3)


def test_pdf_q2_is_cauchy():
    # The q = 2 q-Gaussian with beta = 1 is the standard Cauchy 1/(pi (1 + x^2)).
    x = jnp.linspace(-8.0, 8.0, 400)
    expected = 1.0 / (jnp.pi * (1.0 + x**2))
    assert jnp.allclose(q_gaussian_pdf(x, 2.0, 1.0), expected, atol=1e-6)


def test_pdf_matches_gaussian_at_one():
    x = jnp.linspace(-4.0, 4.0, 200)
    beta = 1.0
    expected = jnp.sqrt(beta / jnp.pi) * jnp.exp(-beta * x**2)
    assert jnp.allclose(q_gaussian_pdf(x, 1.0, beta), expected, atol=1e-6)


def test_logpdf_consistent_with_pdf():
    x = jnp.linspace(-3.0, 3.0, 100)
    for q in (0.5, 1.0, 1.5, 2.5):
        lp = q_gaussian_logpdf(x, q, 1.3)
        p = q_gaussian_pdf(x, q, 1.3)
        # Compare in log-space where the density is positive (inside the support).
        mask = p > 1e-10
        assert jnp.allclose(lp[mask], jnp.log(p[mask]), atol=1e-6)


def test_normalization_positive():
    for q in (0.0, 0.5, 1.0, 1.5, 2.5):
        assert normalization(q) > 0


@pytest.mark.parametrize("q", [0.5, 1.2, 1.6, 2.5])
def test_normalization_gradient_finite(q):
    # Regression: the unused piecewise branch must not poison the gradient
    # (0 * NaN) when differentiating C_q with respect to q.
    g = jax.grad(normalization)(q)
    assert jnp.isfinite(g)


def test_logpdf_gradient_finite():
    x = jnp.linspace(-3.0, 3.0, 64)
    g = jax.grad(lambda q: jnp.mean(q_gaussian_logpdf(x, q, 0.8)))(1.6)
    assert jnp.isfinite(g)


@pytest.mark.parametrize("q", [1.2, 1.4])
def test_sample_variance_matches_theory(q):
    # Var = 1 / (beta (5 - 3q)) for q < 5/3.
    beta = 1.0
    key = jax.random.PRNGKey(0)
    x = sample(key, q, beta, shape=(400_000,))
    expected_var = 1.0 / (beta * (5.0 - 3.0 * q))
    assert jnp.allclose(jnp.var(x), expected_var, rtol=0.1)


def test_sample_is_zero_mean_and_shaped():
    key = jax.random.PRNGKey(1)
    x = sample(key, 1.5, 1.0, shape=(5000,))
    assert x.shape == (5000,)
    assert jnp.abs(jnp.mean(x)) < 0.2
