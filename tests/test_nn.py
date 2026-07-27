"""Tests for the ``qjax.nn`` building blocks."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from qjax.nn import bounded_q, entmax_attention, inverse_bounded_q, tsallis_cross_entropy_loss

jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# bounded_q
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("lo", "hi"), [(1.0, 3.0), (0.3, 1.3), (1.1, 2.8)])
def test_bounded_q_stays_inside_the_interval(lo, hi):
    # Even at absurd magnitudes the value never escapes [lo, hi]. The endpoints
    # themselves are reachable because the sigmoid saturates in floating point,
    # which is why the docstring documents a closed interval in practice.
    q_raw = jnp.linspace(-50.0, 50.0, 201)
    q = bounded_q(q_raw, lo, hi)
    assert jnp.all(q >= lo)
    assert jnp.all(q <= hi)


@pytest.mark.parametrize(("lo", "hi"), [(1.0, 3.0), (0.3, 1.3)])
def test_bounded_q_is_strictly_interior_in_the_usable_range(lo, hi):
    q_raw = jnp.linspace(-20.0, 20.0, 201)
    q = bounded_q(q_raw, lo, hi)
    assert jnp.all(q > lo)
    assert jnp.all(q < hi)


def test_bounded_q_midpoint_and_monotonicity():
    assert jnp.allclose(bounded_q(0.0, 1.0, 3.0), 2.0)
    q = bounded_q(jnp.linspace(-5.0, 5.0, 50), 1.0, 3.0)
    assert jnp.all(jnp.diff(q) > 0.0)


@pytest.mark.parametrize("q", [1.05, 1.5, 2.0, 2.95])
def test_inverse_bounded_q_round_trips(q):
    assert jnp.allclose(bounded_q(inverse_bounded_q(q, 1.0, 3.0), 1.0, 3.0), q, atol=1e-10)


def test_bounded_q_gradient_is_finite_and_nonzero():
    grad = jax.grad(lambda r: bounded_q(r, 1.0, 3.0))(0.0)
    assert jnp.isfinite(grad)
    assert grad > 0.0
    # Even far out in the saturated tail it must not become NaN.
    assert jnp.isfinite(jax.grad(lambda r: bounded_q(r, 1.0, 3.0))(40.0))


def test_bounded_q_composes_with_entmax_under_jit():
    import qjax

    def loss(q_raw):
        q = bounded_q(q_raw, 1.0, 3.0)
        return jnp.sum(qjax.tsallis_entmax(jnp.array([1.0, 2.0, 0.5]), q) ** 2)

    assert jnp.isfinite(jax.jit(jax.grad(loss))(0.0))


# ---------------------------------------------------------------------------
# entmax_attention
# ---------------------------------------------------------------------------


def test_attention_shapes_with_single_query():
    queries = jnp.ones((2, 4))
    keys = jnp.ones((2, 5, 4))
    values = jnp.ones((2, 5, 3))
    context, attn = entmax_attention(queries, keys, values, q=2.0)
    assert context.shape == (2, 3)
    assert attn.shape == (2, 5)


def test_attention_shapes_with_multiple_queries():
    queries = jnp.ones((2, 7, 4))
    keys = jnp.ones((2, 5, 4))
    values = jnp.ones((2, 5, 3))
    context, attn = entmax_attention(queries, keys, values, q=1.5)
    assert context.shape == (2, 7, 3)
    assert attn.shape == (2, 7, 5)


@pytest.mark.parametrize("q", [0.5, 1.0, 1.5, 2.0])
def test_attention_weights_form_a_distribution(q):
    key = jax.random.PRNGKey(0)
    k_q, k_k, k_v = jax.random.split(key, 3)
    queries = jax.random.normal(k_q, (3, 4))
    keys = jax.random.normal(k_k, (3, 6, 4))
    values = jax.random.normal(k_v, (3, 6, 2))
    _, attn = entmax_attention(queries, keys, values, q=q)
    assert jnp.allclose(jnp.sum(attn, axis=-1), 1.0, atol=1e-9)
    assert jnp.all(attn >= 0.0)


def test_attention_matches_softmax_at_q_one():
    key = jax.random.PRNGKey(1)
    k_q, k_k, k_v = jax.random.split(key, 3)
    queries = jax.random.normal(k_q, (2, 4))
    keys = jax.random.normal(k_k, (2, 5, 4))
    values = jax.random.normal(k_v, (2, 5, 3))

    _, attn = entmax_attention(queries, keys, values, q=1.0)
    scores = jnp.einsum("bd,bkd->bk", queries, keys) / jnp.sqrt(4.0)
    assert jnp.allclose(attn, jax.nn.softmax(scores, axis=-1), atol=1e-9)


def test_attention_mask_excludes_positions():
    queries = jnp.ones((1, 4))
    keys = jnp.ones((1, 5, 4))
    values = jnp.arange(5.0).reshape(1, 5, 1)
    mask = jnp.array([[True, True, False, False, False]])
    _, attn = entmax_attention(queries, keys, values, q=1.0, mask=mask)
    assert jnp.allclose(attn[0, 2:], 0.0, atol=1e-12)
    assert jnp.allclose(jnp.sum(attn), 1.0, atol=1e-9)


def test_attention_is_sparse_at_q_two():
    queries = jnp.array([[1.0, 0.0]])
    keys = jnp.array([[[3.0, 0.0], [0.0, 0.0], [-3.0, 0.0]]])
    values = jnp.ones((1, 3, 2))
    _, attn = entmax_attention(queries, keys, values, q=2.0)
    assert jnp.sum(attn == 0.0) >= 1


def test_attention_gradients_flow_to_q():
    key = jax.random.PRNGKey(2)
    k_q, k_k, k_v = jax.random.split(key, 3)
    queries = jax.random.normal(k_q, (2, 4))
    keys = jax.random.normal(k_k, (2, 5, 4))
    values = jax.random.normal(k_v, (2, 5, 3))

    def loss(q_raw):
        q = bounded_q(q_raw, 1.0, 3.0)
        context, _ = entmax_attention(queries, keys, values, q=q)
        return jnp.sum(context**2)

    grad = jax.grad(loss)(0.0)
    assert jnp.isfinite(grad)
    assert not jnp.allclose(grad, 0.0)


# ---------------------------------------------------------------------------
# tsallis_cross_entropy_loss
# ---------------------------------------------------------------------------


def test_loss_matches_softmax_cross_entropy_at_q_one():
    logits = jnp.array([[2.0, 0.5, -1.0], [0.1, 1.2, 0.3]])
    targets = jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    got = tsallis_cross_entropy_loss(logits, targets, q=1.0)
    expected = jnp.mean(-jnp.sum(targets * jax.nn.log_softmax(logits, axis=-1), axis=-1))
    assert jnp.allclose(got, expected, atol=1e-9)


@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
def test_loss_reductions(reduction):
    logits = jnp.array([[2.0, 0.5, -1.0], [0.1, 1.2, 0.3]])
    targets = jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    out = tsallis_cross_entropy_loss(logits, targets, q=1.5, normalizer_q=1.0, reduction=reduction)
    assert out.shape == ((2,) if reduction == "none" else ())


def test_loss_rejects_unknown_reduction():
    with pytest.raises(ValueError, match="reduction must be"):
        tsallis_cross_entropy_loss(jnp.zeros((1, 2)), jnp.zeros((1, 2)), reduction="average")


def test_loss_accepts_probabilities_directly():
    probs = jnp.array([[0.7, 0.2, 0.1]])
    targets = jnp.array([[1.0, 0.0, 0.0]])
    got = tsallis_cross_entropy_loss(probs, targets, q=1.0, from_logits=False)
    assert jnp.allclose(got, -jnp.log(0.7), atol=1e-9)


@pytest.mark.parametrize("q", [0.3, 0.5, 0.8])
def test_loss_is_bounded_below_one(q):
    # Robustness to label noise lives at q < 1, where ln_q(0) = -1/(1-q) is
    # finite, so a confidently wrong prediction costs at most 1/(1-q) instead of
    # diverging as the q = 1 loss does.
    targets = jnp.array([[1.0, 0.0]])
    confident_and_wrong = jnp.array([[-40.0, 40.0]])
    loss = tsallis_cross_entropy_loss(
        confident_and_wrong, targets, q=q, normalizer_q=1.0, reduction="sum"
    )
    assert loss <= 1.0 / (1.0 - q) + 1e-9
    baseline = tsallis_cross_entropy_loss(
        confident_and_wrong, targets, q=1.0, normalizer_q=1.0, reduction="sum"
    )
    assert baseline > loss


def test_loss_above_one_penalizes_harder_than_log():
    # The mirror image: q > 1 grows like p^(1-q), faster than the logarithm.
    targets = jnp.array([[1.0, 0.0]])
    confident_and_wrong = jnp.array([[-40.0, 40.0]])
    sharp = tsallis_cross_entropy_loss(
        confident_and_wrong, targets, q=2.0, normalizer_q=1.0, reduction="sum"
    )
    baseline = tsallis_cross_entropy_loss(
        confident_and_wrong, targets, q=1.0, normalizer_q=1.0, reduction="sum"
    )
    assert sharp > baseline


@pytest.mark.parametrize("q", [0.3, 0.6, 0.9])
def test_loss_gradient_is_bounded_below_one(q):
    # The point of the bounded loss: the gradient a mislabelled example
    # contributes is capped too, so it cannot dominate the update.
    targets = jnp.array([[1.0, 0.0]])

    def loss(z):
        return tsallis_cross_entropy_loss(z, targets, q=q, normalizer_q=1.0, reduction="sum")

    grad = jax.grad(loss)(jnp.array([[-40.0, 40.0]]))
    assert jnp.all(jnp.isfinite(grad))
    assert jnp.max(jnp.abs(grad)) < 1.0


def test_loss_gradients_are_finite_including_wrt_q():
    logits = jnp.array([[2.0, 0.5, -1.0], [0.1, 1.2, 0.3]])
    targets = jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

    grad_logits = jax.grad(
        lambda z: tsallis_cross_entropy_loss(z, targets, q=1.5, normalizer_q=1.0)
    )
    assert jnp.all(jnp.isfinite(grad_logits(logits)))

    def loss_of_q(q_raw):
        q = bounded_q(q_raw, 1.0, 3.0)
        return tsallis_cross_entropy_loss(logits, targets, q=q, normalizer_q=1.0)

    assert jnp.isfinite(jax.grad(loss_of_q)(0.0))


def test_loss_jits():
    logits = jnp.array([[2.0, 0.5, -1.0]])
    targets = jnp.array([[1.0, 0.0, 0.0]])
    fn = jax.jit(lambda z, t: tsallis_cross_entropy_loss(z, t, q=1.5, normalizer_q=1.0))
    assert jnp.allclose(
        fn(logits, targets),
        tsallis_cross_entropy_loss(logits, targets, q=1.5, normalizer_q=1.0),
        atol=1e-12,
    )
