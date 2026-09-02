import jax
import jax.numpy as jnp
import pytest

from qjax.nn.autoregressive import (
    made_conditionals,
    made_init,
    made_log_prob,
    made_masks,
    made_sample,
)

jax.config.update("jax_enable_x64", True)


def enumerate_spins(num_spins):
    states = jnp.arange(2**num_spins)
    bits = (states[:, None] >> jnp.arange(num_spins)[None, :]) & 1
    return 1.0 - 2.0 * bits.astype(jnp.result_type(float))


def sharpen(params, scale=3.0, shift=0.4):
    """Push a freshly initialized MADE away from the near-uniform distribution."""
    return {
        "weights": [w * scale for w in params["weights"]],
        "biases": [b + shift for b in params["biases"]],
    }


def test_mask_shapes_match_the_weights():
    num_spins, hidden = 8, (32, 16)
    masks = made_masks(num_spins, hidden)
    params = made_init(jax.random.PRNGKey(0), num_spins, hidden)
    assert len(masks) == len(hidden) + 1
    for mask, weight in zip(masks, params["weights"], strict=True):
        assert mask.shape == weight.shape
    assert [b.shape for b in params["biases"]] == [(32,), (16,), (8,)]
    # Masks are binary and not degenerate.
    for mask in masks:
        assert set(jnp.unique(mask).tolist()) <= {0.0, 1.0}
        assert 0.0 < float(jnp.mean(mask)) < 1.0


@pytest.mark.parametrize("num_spins", [4, 8])
def test_made_normalizes_over_the_full_state_space(num_spins):
    hidden = (32, 32)
    masks = made_masks(num_spins, hidden)
    params = sharpen(made_init(jax.random.PRNGKey(1), num_spins, hidden))
    total = jnp.sum(jnp.exp(made_log_prob(params, masks, enumerate_spins(num_spins))))
    assert float(total) == pytest.approx(1.0, abs=1e-10)


def test_made_is_autoregressive():
    num_spins, hidden = 8, (32, 32)
    masks = made_masks(num_spins, hidden)
    params = made_init(jax.random.PRNGKey(2), num_spins, hidden)
    configuration = enumerate_spins(num_spins)[37]

    jacobian = jax.jacfwd(lambda s: made_conditionals(params, masks, s[None])[0])(configuration)
    # Logit i may depend only on inputs j < i: the upper triangle, diagonal
    # included, must be exactly zero. This is the property the masks exist for.
    assert float(jnp.max(jnp.abs(jnp.triu(jacobian)))) == 0.0
    # And the lower triangle must not be trivially zero, or the test is vacuous.
    assert float(jnp.max(jnp.abs(jnp.tril(jacobian, -1)))) > 0.0
    # The first conditional is unconditional: no hidden unit can reach it.
    assert float(jnp.max(jnp.abs(masks[-1][:, 0]))) == 0.0


def test_made_sample_matches_its_own_log_prob():
    num_spins, hidden, draws = 6, (24,), 200_000
    masks = made_masks(num_spins, hidden)
    params = sharpen(made_init(jax.random.PRNGKey(3), num_spins, hidden))

    configurations = enumerate_spins(num_spins)
    probabilities = jnp.exp(made_log_prob(params, masks, configurations))
    assert float(jnp.sum(probabilities)) == pytest.approx(1.0, abs=1e-10)
    # The distribution must be far from uniform, or agreement proves nothing.
    assert float(jnp.max(probabilities) / jnp.min(probabilities)) > 10.0

    samples = made_sample(jax.random.PRNGKey(4), params, masks, draws)
    assert samples.shape == (draws, num_spins)
    assert set(jnp.unique(samples).tolist()) <= {-1.0, 1.0}

    index = ((samples < 0).astype(jnp.int32) * (2 ** jnp.arange(num_spins))).sum(axis=-1)
    frequency = jnp.zeros(2**num_spins).at[index].add(1.0) / draws
    sigma = jnp.sqrt(probabilities * (1.0 - probabilities) / draws)
    assert float(jnp.max(jnp.abs(frequency - probabilities) / sigma)) < 5.0


def test_made_log_prob_is_differentiable_and_jittable():
    num_spins, hidden = 10, (32,)
    masks = made_masks(num_spins, hidden)
    params = made_init(jax.random.PRNGKey(5), num_spins, hidden)
    spins = made_sample(jax.random.PRNGKey(6), params, masks, 16)

    def objective(p):
        return jnp.mean(made_log_prob(p, masks, spins))

    value, grads = jax.value_and_grad(objective)(params)
    assert jnp.isfinite(value)
    assert all(bool(jnp.all(jnp.isfinite(g))) for g in grads["weights"] + grads["biases"])
    # Masked-out weights receive exactly zero gradient, so they can never learn.
    assert float(jnp.max(jnp.abs(grads["weights"][-1] * (1.0 - masks[-1])))) == 0.0
    assert float(jax.jit(objective)(params)) == pytest.approx(float(value))


def test_made_rejects_degenerate_configurations():
    with pytest.raises(ValueError, match="at least one hidden layer"):
        made_masks(8, ())
    with pytest.raises(ValueError, match="at least 2 spins"):
        made_masks(1, (8,))
