import math

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


# ---------------------------------------------------------------------------
# Heavy tails (q >= 2).
#
# These cases were previously removed from the parametrization because a
# truncated `trapezoid` under-integrates a tail that decays like |x|^(-2/(q-1)):
# at q = 2.9 a grid out to +/-2000 still misses ~64% of the mass. The
# normalization constant itself is correct, so the right fix is a quadrature
# that handles the tail rather than dropping the coverage.
# ---------------------------------------------------------------------------


def _closed_form_normalization(q):
    """C_q from the Gamma-function closed form, evaluated in log space."""
    if q < 1.0:
        a = 1.0 - q
        log_c = (
            math.log(2.0)
            + 0.5 * math.log(math.pi)
            + math.lgamma(1.0 / a)
            - math.log(3.0 - q)
            - 0.5 * math.log(a)
            - math.lgamma((3.0 - q) / (2.0 * a))
        )
    elif q > 1.0:
        b = q - 1.0
        log_c = (
            0.5 * math.log(math.pi)
            + math.lgamma((3.0 - q) / (2.0 * b))
            - 0.5 * math.log(b)
            - math.lgamma(1.0 / b)
        )
    else:
        return math.sqrt(math.pi)
    return math.exp(log_c)


@pytest.mark.parametrize("q", [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 2.9])
def test_normalization_matches_closed_form(q):
    assert jnp.allclose(normalization(q), _closed_form_normalization(q), rtol=1e-10)


@pytest.mark.parametrize("q", [1.5, 2.0])
@pytest.mark.parametrize("beta", [0.5, 1.0, 2.0])
def test_heavy_tailed_pdf_integrates_to_one(q, beta):
    # Substitute x = tan(t) so the whole real line maps to (-pi/2, pi/2). The
    # density decays like |x|^(-2/(q-1)) and the Jacobian sec^2(t) grows like
    # x^2, so the transformed integrand is bounded only while 2/(q-1) >= 2, i.e.
    # q <= 2. Beyond that it stays integrable but becomes unbounded at the
    # endpoints, which uniform-grid quadrature handles badly -- those q are
    # covered by the exact Student-t identity below instead.
    t = jnp.linspace(-jnp.pi / 2, jnp.pi / 2, 200_001)[1:-1]
    x = jnp.tan(t)
    mass = jnp.trapezoid(q_gaussian_pdf(x, q, beta) / jnp.cos(t) ** 2, t)
    assert jnp.allclose(mass, 1.0, atol=1e-4)


@pytest.mark.parametrize("q", [1.5, 2.0, 2.5, 2.9])
@pytest.mark.parametrize("beta", [0.5, 1.0])
def test_pdf_matches_student_t(q, beta):
    # For 1 < q < 3 the q-Gaussian is exactly a rescaled Student-t with
    # nu = (3-q)/(q-1) degrees of freedom and scale 1/sqrt((3-q) beta). Matching
    # that (already-normalized) density pointwise is a stronger statement than a
    # coarse numerical mass check, and it holds for arbitrarily heavy tails.
    nu = (3.0 - q) / (q - 1.0)
    scale = 1.0 / math.sqrt((3.0 - q) * beta)
    x = jnp.linspace(-6.0, 6.0, 401)
    u = x / scale
    log_t = (
        math.lgamma((nu + 1) / 2)
        - math.lgamma(nu / 2)
        - 0.5 * math.log(nu * math.pi)
        - ((nu + 1) / 2) * jnp.log1p(u**2 / nu)
    )
    expected = jnp.exp(log_t) / scale
    assert jnp.allclose(q_gaussian_pdf(x, q, beta), expected, rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("q", [2.0, 2.5])
def test_heavy_tailed_logpdf_consistent_with_pdf(q):
    x = jnp.linspace(-5.0, 5.0, 201)
    assert jnp.allclose(q_gaussian_logpdf(x, q, 1.0), jnp.log(q_gaussian_pdf(x, q, 1.0)), atol=1e-9)


@pytest.mark.parametrize("q", [2.0, 2.5, 2.9])
def test_normalization_gradient_finite_for_heavy_tails(q):
    assert jnp.isfinite(jax.grad(normalization)(q))


def test_pdf_is_normalized_for_compact_support():
    # q < 1 gives compact support [-1/sqrt((1-q) beta), ...]; integrate inside it.
    q, beta = 0.5, 1.0
    edge = 1.0 / math.sqrt((1.0 - q) * beta)
    x = jnp.linspace(-edge, edge, 200_001)
    assert jnp.allclose(jnp.trapezoid(q_gaussian_pdf(x, q, beta), x), 1.0, atol=1e-6)
    # And vanishes strictly outside it.
    assert q_gaussian_pdf(edge * 1.01, q, beta) == 0.0
