import jax
import jax.numpy as jnp
import pytest

from qjax.core.functions import q_add, q_diff, q_div, q_exp, q_log, q_prod

jax.config.update("jax_enable_x64", True)

QS = [0.3, 0.7, 1.0, 1.5, 2.5]


@pytest.mark.parametrize("q", QS)
def test_q_log_exp_are_inverse(q):
    x = jnp.linspace(0.1, 3.0, 50)
    assert jnp.allclose(q_exp(q_log(x, q), q), x, atol=1e-6)


def test_q_log_recovers_natural_log_at_one():
    x = jnp.linspace(0.1, 5.0, 50)
    assert jnp.allclose(q_log(x, 1.0), jnp.log(x), atol=1e-7)


def test_q_exp_recovers_exp_at_one():
    x = jnp.linspace(-3.0, 2.0, 50)
    assert jnp.allclose(q_exp(x, 1.0), jnp.exp(x), atol=1e-7)


def test_q_log_continuous_near_one():
    x = jnp.array([0.5, 1.0, 2.0])
    left = q_log(x, 1.0 - 1e-4)
    right = q_log(x, 1.0 + 1e-4)
    assert jnp.allclose(left, right, atol=1e-3)


def test_gradient_finite_at_q_equals_one():
    # The double-where trick must keep d/dq finite exactly at q = 1.
    g = jax.grad(lambda q: q_log(2.0, q))(1.0)
    assert jnp.isfinite(g)
    g2 = jax.grad(lambda q: q_exp(0.5, q))(1.0)
    assert jnp.isfinite(g2)


@pytest.mark.parametrize("q", QS)
def test_q_add_diff_inverse(q):
    a = jnp.array([0.2, 0.5, 1.3])
    b = jnp.array([0.1, -0.4, 0.7])
    assert jnp.allclose(q_add(q_diff(a, b, q), b, q), a, atol=1e-6)


# For q > 1 the q-product/division carry a Tsallis cut-off, so the round trip is
# exact only where the deformed quotient stays inside the (a^{1-q} - b^{1-q} + 1 >= 0)
# domain. These operands stay inside it for the chosen q values.
@pytest.mark.parametrize("q", [0.3, 0.7, 1.3, 1.5])
def test_q_prod_div_inverse(q):
    a = jnp.array([0.4, 1.0, 1.2])
    b = jnp.array([0.5, 1.1, 0.9])
    assert jnp.allclose(q_prod(q_div(a, b, q), b, q), a, atol=1e-5)


def test_q_log_distributes_over_q_product():
    # ln_q(x y) = ln_q(x) (+)_q ln_q(y)
    x, y, q = 1.7, 2.3, 1.4
    lhs = q_log(x * y, q)
    rhs = q_add(q_log(x, q), q_log(y, q), q)
    assert jnp.allclose(lhs, rhs, atol=1e-6)


def test_jit_and_vmap():
    f = jax.jit(jax.vmap(q_log, in_axes=(0, None)))
    out = f(jnp.linspace(0.1, 2.0, 16), 1.8)
    assert out.shape == (16,)
