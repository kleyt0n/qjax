"""Behaviour in JAX's default single precision.

Every other test module enables ``jax_enable_x64``. Users do not, by default, and
the ``q -> 1`` limit is exactly where float32 hurts: the textbook
``(x^(1-q) - 1)/(1-q)`` loses most of its mantissa to cancellation for ``q``
within about ``1e-4`` of one. These tests pin the accuracy that the
``expm1``/``log1p`` formulation buys back.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import qjax

pytestmark = pytest.mark.float32

# q values that straddle the cancellation band around q = 1.
NEAR_ONE_QS = [1.0 + 10.0**-k for k in range(3, 8)] + [1.0 - 10.0**-k for k in range(3, 8)]


@pytest.mark.parametrize("q", NEAR_ONE_QS)
def test_q_log_accurate_near_one_in_float32(float32_mode, q):
    x = 2.0
    got = float(qjax.q_log(x, q))
    expected = (x ** (1.0 - q) - 1.0) / (1.0 - q)
    assert abs(got - expected) / abs(expected) < 1e-6


@pytest.mark.parametrize("q", NEAR_ONE_QS)
def test_q_exp_accurate_near_one_in_float32(float32_mode, q):
    x = 0.7
    got = float(qjax.q_exp(x, q))
    expected = (1.0 + (1.0 - q) * x) ** (1.0 / (1.0 - q))
    assert abs(got - expected) / abs(expected) < 1e-6


def test_default_dtype_is_float32(float32_mode):
    # Guards the fixture itself: without this the tests above would silently
    # assert float64 behaviour and prove nothing.
    assert qjax.q_log(jnp.array([2.0]), 1.5).dtype == jnp.float32


def test_q_log_recovers_log_at_q_one_in_float32(float32_mode):
    x = jnp.array([0.25, 1.0, 3.0])
    assert jnp.allclose(qjax.q_log(x, 1.0), jnp.log(x), rtol=1e-6)


@pytest.mark.parametrize("q", [0.5, 1.0, 1.5, 2.0])
def test_entmax_sums_to_one_in_float32(float32_mode, q):
    z = jnp.array([1.0, 2.0, 0.5, -1.0])
    p = qjax.tsallis_entmax(z, q)
    assert p.dtype == jnp.float32
    assert abs(float(jnp.sum(p)) - 1.0) < 1e-5
    assert jnp.all(p >= 0.0)


def test_entmax_large_logits_stable_in_float32(float32_mode):
    # A shift of +200 must not overflow: the implementation centres by the max.
    z = jnp.array([1.0, 2.0, 0.5])
    base = qjax.tsallis_entmax(z, 1.5)
    shifted = qjax.tsallis_entmax(z + 200.0, 1.5)
    assert jnp.all(jnp.isfinite(shifted))
    assert jnp.allclose(base, shifted, atol=1e-5)


@pytest.mark.parametrize("q", [0.5, 1.0, 1.5, 2.5])
def test_entropy_matches_closed_form_in_float32(float32_mode, q):
    p = np.array([0.5, 0.3, 0.2])
    got = float(qjax.tsallis_entropy(jnp.asarray(p), q))
    expected = -np.sum(p * np.log(p)) if q == 1.0 else (1.0 - np.sum(p**q)) / (q - 1.0)
    assert abs(got - expected) < 1e-5


def test_gradients_finite_in_float32(float32_mode):
    z = jnp.array([3.0, 1.0, 0.0, -2.0])
    p = jnp.array([0.5, 0.0, 0.5])
    assert jnp.all(jnp.isfinite(jax.grad(lambda t: jnp.sum(qjax.tsallis_entmax(t, 2.0) ** 2))(z)))
    assert jnp.all(jnp.isfinite(jax.grad(lambda t: qjax.tsallis_entropy(t, 2.0))(p)))
    assert jnp.isfinite(jax.grad(lambda q: qjax.q_log(2.0, q))(1.0))


def test_dq_gradient_correct_in_float32(float32_mode):
    grad = float(jax.grad(lambda q: qjax.q_log(2.0, q))(1.0))
    assert abs(grad - (-0.5 * math.log(2.0) ** 2)) < 1e-5
