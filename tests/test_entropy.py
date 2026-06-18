import jax
import jax.numpy as jnp
import pytest

from qjax.core.entropy import tsallis_cross_entropy, tsallis_divergence, tsallis_entropy

jax.config.update("jax_enable_x64", True)


def _shannon(p):
    plogp = jnp.where(p > 0, p * jnp.log(p), 0.0)
    return -jnp.sum(plogp)


def _kl(p, r):
    return jnp.sum(jnp.where(p > 0, p * jnp.log(p / r), 0.0))


def test_entropy_recovers_shannon_at_one():
    p = jnp.array([0.1, 0.2, 0.3, 0.4])
    assert jnp.allclose(tsallis_entropy(p, 1.0), _shannon(p), atol=1e-7)


@pytest.mark.parametrize("q", [0.5, 1.0, 2.0, 3.0])
def test_entropy_maximal_at_uniform(q):
    n = 5
    uniform = jnp.full(n, 1.0 / n)
    peaked = jnp.array([0.9, 0.04, 0.03, 0.02, 0.01])
    assert tsallis_entropy(uniform, q) >= tsallis_entropy(peaked, q)


@pytest.mark.parametrize("q", [0.5, 1.0, 2.0])
def test_entropy_nonnegative(q):
    p = jnp.array([0.25, 0.25, 0.25, 0.25])
    assert tsallis_entropy(p, q) >= -1e-9


def test_divergence_recovers_kl_at_one():
    p = jnp.array([0.1, 0.2, 0.3, 0.4])
    r = jnp.array([0.25, 0.25, 0.25, 0.25])
    assert jnp.allclose(tsallis_divergence(p, r, 1.0), _kl(p, r), atol=1e-7)


@pytest.mark.parametrize("q", [0.5, 1.0, 1.5, 2.0])
def test_divergence_nonnegative_and_zero_at_identity(q):
    p = jnp.array([0.1, 0.2, 0.3, 0.4])
    r = jnp.array([0.3, 0.3, 0.2, 0.2])
    assert tsallis_divergence(p, r, q) >= -1e-9
    assert jnp.allclose(tsallis_divergence(p, p, q), 0.0, atol=1e-9)


def test_cross_entropy_matches_standard_at_one():
    p = jnp.array([0.7, 0.2, 0.1])
    y = jnp.array([1.0, 0.0, 0.0])
    expected = -jnp.log(p[0])
    assert jnp.allclose(tsallis_cross_entropy(p, y, 1.0), expected, atol=1e-7)


def test_cross_entropy_differentiable():
    y = jnp.array([0.0, 1.0, 0.0])

    def loss(logits):
        p = jax.nn.softmax(logits)
        return tsallis_cross_entropy(p, y, 1.5)

    g = jax.grad(loss)(jnp.array([0.5, -0.2, 1.1]))
    assert jnp.all(jnp.isfinite(g))


def test_batched_axis():
    p = jnp.array([[0.5, 0.5], [0.9, 0.1]])
    out = tsallis_entropy(p, 2.0, axis=-1)
    assert out.shape == (2,)


# Regression: a probability vector with an exact zero entry (e.g. the output of
# sparse tsallis_entmax, or a one-hot distribution) must not poison gradients
# through the 0 * log(0) = 0 * -inf = NaN path of the classical branch.
@pytest.mark.parametrize("q", [1.0, 1.5, 2.0])
def test_entropy_gradient_finite_with_zero_entry(q):
    p = jnp.array([0.5, 0.5, 0.0])
    g = jax.grad(lambda p: tsallis_entropy(p, q))(p)
    assert jnp.all(jnp.isfinite(g))


@pytest.mark.parametrize("q", [1.0, 1.5, 2.0])
def test_divergence_gradient_finite_with_zero_entry(q):
    p = jnp.array([0.5, 0.5, 0.0])
    r = jnp.array([0.3, 0.3, 0.4])
    g = jax.grad(lambda p: tsallis_divergence(p, r, q))(p)
    assert jnp.all(jnp.isfinite(g))


@pytest.mark.parametrize("q", [0.5, 1.0, 1.5, 2.0])
def test_cross_entropy_sparse_prediction_is_finite(q):
    # A zero prediction at a class whose target mass is zero contributes nothing
    # (0 * ln_q(0) = 0): both the value and the gradient must stay finite.
    p = jnp.array([0.7, 0.3, 0.0])
    y = jnp.array([1.0, 0.0, 0.0])
    val = tsallis_cross_entropy(p, y, q)
    grad = jax.grad(lambda p: tsallis_cross_entropy(p, y, q))(p)
    assert jnp.isfinite(val)
    assert jnp.all(jnp.isfinite(grad))


def test_cross_entropy_infinite_on_zero_true_class():
    # Predicting zero probability *on* the true class is a genuine +inf loss and
    # must be preserved (not masked away).
    p = jnp.array([0.0, 1.0])
    y = jnp.array([1.0, 0.0])
    assert jnp.isposinf(tsallis_cross_entropy(p, y, 1.0))
