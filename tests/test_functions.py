import math

import jax
import jax.numpy as jnp
import pytest
from jax.test_util import check_grads

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
# domain. These operands stay inside it for every q in QS, including q = 2.5.
@pytest.mark.parametrize("q", QS)
def test_q_prod_div_inverse(q):
    a = jnp.array([0.4, 1.0, 1.2])
    b = jnp.array([0.5, 1.1, 0.9])
    assert jnp.allclose(q_prod(q_div(a, b, q), b, q), a, atol=1e-5)
    # The dual direction: q_div undoes q_prod.
    assert jnp.allclose(q_div(q_prod(a, b, q), b, q), a, atol=1e-5)


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


# ---------------------------------------------------------------------------
# Gradient correctness (not merely finiteness) and the q -> 1 limit.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("q", [0.4, 1.0, 1.6, 2.5])
def test_check_grads_q_log(q):
    check_grads(
        q_log, (jnp.array([0.5, 2.0, 3.0]), jnp.asarray(q)), order=2, modes=["fwd", "rev"], eps=1e-6
    )


@pytest.mark.parametrize("q", [0.4, 1.0, 1.6, 2.5])
def test_check_grads_q_exp(q):
    check_grads(
        q_exp,
        (jnp.array([-0.3, 0.2, 0.5]), jnp.asarray(q)),
        order=2,
        modes=["fwd", "rev"],
        eps=1e-6,
    )


def test_dq_gradient_correct_at_q_one():
    # Regression: the closed form used to be selected by a hard `|q-1| < eps`
    # branch whose classical side is q-independent, so d/dq was a hard zero in a
    # 2e-6 window around q = 1. The true values are analytic.
    x = 2.0
    log_x = math.log(x)
    grad = jax.grad(lambda q: q_log(x, q))(1.0)
    assert jnp.allclose(grad, -0.5 * log_x**2, atol=1e-12)
    assert not jnp.allclose(grad, 0.0)

    second = jax.grad(jax.grad(lambda q: q_log(x, q)))(1.0)
    assert jnp.allclose(second, log_x**3 / 3.0, atol=1e-12)


def test_dq_gradient_continuous_across_q_one():
    # No cliff: the derivative in q varies smoothly through q = 1.
    grads = [
        float(jax.grad(lambda q: q_log(2.0, q))(qv))
        for qv in [1 - 1e-4, 1 - 1e-6, 1.0, 1 + 1e-6, 1 + 1e-4]
    ]
    assert max(grads) - min(grads) < 1e-4


@pytest.mark.parametrize("q", [1.0 + 10.0**-k for k in range(3, 9)])
def test_q_log_accurate_near_one(q):
    # The expm1 form removes the catastrophic cancellation of (x^(1-q)-1)/(1-q).
    expected = (2.0 ** (1.0 - q) - 1.0) / (1.0 - q)
    assert jnp.allclose(q_log(2.0, q), expected, rtol=1e-9)


# ---------------------------------------------------------------------------
# Domain edges: zero, the Tsallis cut-off, and gradient safety around them.
# ---------------------------------------------------------------------------


def test_q_log_at_zero():
    # ln_q(0) = -1/(1-q) for q < 1, and -inf for q >= 1.
    assert jnp.allclose(q_log(0.0, 0.5), -2.0)
    assert jnp.allclose(q_log(0.0, 0.0), -1.0)
    assert q_log(0.0, 1.0) == -jnp.inf
    assert q_log(0.0, 2.0) == -jnp.inf
    assert jnp.isnan(q_log(-1.0, 1.5))


def test_q_exp_cut_off_sign_depends_on_q():
    # Past the cut-off the exponent 1/(1-q) decides: 0 for q < 1, +inf for q > 1.
    assert q_exp(-5.0, 0.5) == 0.0
    assert q_exp(2.0, 2.0) == jnp.inf
    assert q_exp(1.0, 2.0) == jnp.inf
    assert jnp.isfinite(q_exp(-2.0, 2.0))


@pytest.mark.parametrize("q", [0.5, 1.0, 2.0, 2.5])
@pytest.mark.parametrize("operands", [(0.0, 2.0), (2.0, 0.0), (0.0, 0.0)])
@pytest.mark.parametrize("op", [q_prod, q_div], ids=["q_prod", "q_div"])
def test_q_prod_div_gradients_finite_at_zero_operands(q, operands, op):
    # Regression: `0 ** negative` inside q_prod/q_div back-propagated NaN even
    # when its branch was unselected, and the classical `a / b` branch did the
    # same for a zero denominator.
    a, b = operands
    assert jnp.isfinite(jax.grad(lambda t: op(t, b, q))(a))
    assert jnp.isfinite(jax.grad(lambda t: op(a, t, q))(b))


def test_q_div_by_zero_value_is_infinite():
    # The value must still signal the singularity even though the gradient is tamed.
    assert q_div(1.0, 0.0, 2.0) == jnp.inf
    assert q_div(1.0, 0.0, 1.0) == jnp.inf
