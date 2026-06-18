import jax
import jax.numpy as jnp
import pytest

from qjax.core.activations import tsallis_entmax

jax.config.update("jax_enable_x64", True)


@pytest.mark.parametrize("q", [1.0, 1.5, 2.0, 3.0])
def test_sums_to_one(q):
    z = jnp.array([2.0, 1.0, 0.1, -1.0, 0.5])
    p = tsallis_entmax(z, q)
    assert jnp.allclose(jnp.sum(p), 1.0, atol=1e-5)
    assert jnp.all(p >= -1e-9)


def test_recovers_softmax_at_one():
    z = jnp.array([2.0, 1.0, 0.1, -1.0, 0.5])
    assert jnp.allclose(tsallis_entmax(z, 1.0), jax.nn.softmax(z), atol=1e-6)


def test_sparsemax_is_sparse():
    # With q = 2 (sparsemax), small-score entries are driven exactly to zero.
    z = jnp.array([3.0, 2.5, -2.0, -3.0])
    p = tsallis_entmax(z, 2.0)
    assert jnp.allclose(jnp.sum(p), 1.0, atol=1e-5)
    assert jnp.sum(p == 0.0) >= 2


def test_sparsemax_closed_form_two_entries():
    # For two logits sparsemax has p1 = clip(0.5 + (z1 - z2)/2, 0, 1).
    # A gap of 0.5 stays interior: 0.5 + 0.25 = 0.75.
    z = jnp.array([0.5, 0.0])
    p = tsallis_entmax(z, 2.0)
    assert jnp.allclose(p, jnp.array([0.75, 0.25]), atol=1e-4)
    # A gap >= 1 saturates to a vertex.
    vertex = tsallis_entmax(jnp.array([1.0, 0.0]), 2.0)
    assert jnp.allclose(vertex, jnp.array([1.0, 0.0]), atol=1e-4)


def test_higher_q_is_sparser():
    z = jnp.array([2.0, 1.0, 0.0, -1.0])
    nz_soft = jnp.sum(tsallis_entmax(z, 1.0) > 1e-6)
    nz_sparse = jnp.sum(tsallis_entmax(z, 2.0) > 1e-6)
    assert nz_sparse <= nz_soft


def test_batched_jit_grad():
    z = jnp.array([[2.0, 1.0, 0.0], [0.0, 1.0, 2.0]])
    p = jax.jit(lambda z: tsallis_entmax(z, 1.5, axis=-1))(z)
    assert p.shape == z.shape
    assert jnp.allclose(jnp.sum(p, axis=-1), 1.0, atol=1e-5)

    g = jax.grad(lambda z: tsallis_entmax(z, 1.5)[0])(jnp.array([1.0, 0.5, 0.2]))
    assert jnp.all(jnp.isfinite(g))


@pytest.mark.parametrize("q", [1.3, 2.0, 2.5])
def test_gradient_wrt_q_finite_in_sparse_regime(q):
    # Regression: with a learnable q, the gradient must stay finite even when
    # entmax zeros out coordinates (where the naive 0**p gradient is NaN).
    z = jnp.array([3.0, 1.0, 0.0, -2.0, -5.0])  # forces several exact zeros

    def objective(q):
        return jnp.sum(tsallis_entmax(z, q) ** 2)

    g = jax.grad(objective)(q)
    assert jnp.isfinite(g)
