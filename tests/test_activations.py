import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.test_util import check_grads

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


# ---------------------------------------------------------------------------
# Gradient correctness.
#
# The threshold `tau` is found by bisection, but differentiating *through* that
# loop yields a wrong Jacobian: `dtau/dz` collapses to `(q-1) * onehot(argmax)`
# instead of the implicit-function value, and the error does not shrink with
# `num_iters`. The tests below pin the exact implicit-function derivative.
#
# Note on test design: finite differences and `check_grads` are invalid across a
# support kink, where entmax is only directionally differentiable. Inputs here
# are chosen so every coordinate is either comfortably positive or comfortably
# clipped over the whole perturbation ball.
# ---------------------------------------------------------------------------

INTERIOR_Z = jnp.array([1.0, 2.0, 0.5])


def closed_form_jacobian(p, q):
    """Jacobian of entmax w.r.t. z: diag(s) - s s^T / sum(s), with s = p^(2-q)."""
    p = np.asarray(p)
    support = p > 0.0
    s = np.where(support, np.where(support, p, 1.0) ** (2.0 - q), 0.0)
    return np.diag(s) - np.outer(s, s) / s.sum()


@pytest.mark.parametrize("q", [0.4, 0.8, 1.3, 1.5, 2.0, 2.5, 3.0])
def test_jacobian_matches_closed_form(q):
    p = tsallis_entmax(INTERIOR_Z, q)
    jac = jax.jacobian(lambda z: tsallis_entmax(z, q))(INTERIOR_Z)
    assert jnp.allclose(jac, closed_form_jacobian(p, q), atol=1e-10)


@pytest.mark.parametrize("q", [0.5, 1.3, 1.5, 2.2])
def test_jacobian_matches_finite_differences(q):
    eps = 1e-6

    def col(i):
        plus = tsallis_entmax(INTERIOR_Z.at[i].add(eps), q)
        minus = tsallis_entmax(INTERIOR_Z.at[i].add(-eps), q)
        return (plus - minus) / (2.0 * eps)

    numeric = jnp.stack([col(i) for i in range(INTERIOR_Z.size)], axis=1)
    analytic = jax.jacobian(lambda z: tsallis_entmax(z, q))(INTERIOR_Z)
    assert jnp.allclose(analytic, numeric, atol=1e-6)


@pytest.mark.parametrize("q", [0.4, 1.5, 2.0, 2.5])
def test_jacobian_is_symmetric(q):
    # entmax is the gradient map of a convex conjugate, so its Jacobian must be
    # symmetric. Differentiating through the bisection breaks this.
    jac = jax.jacobian(lambda z: tsallis_entmax(z, q))(INTERIOR_Z)
    assert jnp.allclose(jac, jac.T, atol=1e-12)


@pytest.mark.parametrize("q", [0.6, 1.5, 2.5])
def test_jacfwd_equals_jacrev(q):
    fwd = jax.jacfwd(lambda z: tsallis_entmax(z, q))(INTERIOR_Z)
    rev = jax.jacrev(lambda z: tsallis_entmax(z, q))(INTERIOR_Z)
    assert jnp.allclose(fwd, rev, atol=1e-12)


@pytest.mark.parametrize("q", [0.6, 1.5, 2.2])
def test_check_grads_second_order(q):
    check_grads(
        lambda z, qq: tsallis_entmax(z, qq),
        (INTERIOR_Z, jnp.asarray(q)),
        order=2,
        modes=["fwd", "rev"],
        eps=1e-5,
    )


def test_check_grads_batched():
    z = jnp.array([[1.0, 2.0, 0.5], [0.3, -1.0, 0.8]])
    check_grads(lambda zz: tsallis_entmax(zz, 1.4), (z,), order=2, modes=["fwd", "rev"], eps=1e-5)


@pytest.mark.parametrize("q", [0.5, 1.5, 2.2])
def test_dpdq_matches_finite_differences(q):
    eps = 1e-6
    analytic = jax.jacfwd(lambda qq: tsallis_entmax(INTERIOR_Z, qq))(jnp.asarray(q))
    numeric = (tsallis_entmax(INTERIOR_Z, q + eps) - tsallis_entmax(INTERIOR_Z, q - eps)) / (
        2 * eps
    )
    assert jnp.allclose(analytic, numeric, atol=1e-8)


def test_dpdq_at_q_one_matches_limit():
    # The q -> 1 limit of dp/dq is -1/2 p (log^2 p - E_p[log^2 p]), which is NOT
    # zero. A plain softmax branch would report zero and create a gradient cliff
    # at the Q_EPS boundary.
    p = jax.nn.softmax(INTERIOR_Z)
    u = jnp.log(p)
    expected = -0.5 * p * (u * u - jnp.sum(p * u * u))

    analytic = jax.jacfwd(lambda qq: tsallis_entmax(INTERIOR_Z, qq))(jnp.asarray(1.0))
    assert jnp.allclose(analytic, expected, atol=1e-12)
    assert not jnp.allclose(analytic, 0.0)

    eps = 1e-3
    numeric = (tsallis_entmax(INTERIOR_Z, 1.0 + eps) - tsallis_entmax(INTERIOR_Z, 1.0 - eps)) / (
        2 * eps
    )
    assert jnp.allclose(analytic, numeric, atol=1e-6)


@pytest.mark.parametrize("q", [0.5, 1.5, 2.5])
def test_tangent_stays_on_simplex(q):
    # Rows and columns of the Jacobian sum to zero: entmax is shift-invariant and
    # its output is constrained to the simplex.
    jac = jax.jacobian(lambda z: tsallis_entmax(z, q))(INTERIOR_Z)
    assert jnp.allclose(jnp.sum(jac, axis=0), 0.0, atol=1e-12)
    assert jnp.allclose(jnp.sum(jac, axis=1), 0.0, atol=1e-12)
    dpdq = jax.jacfwd(lambda qq: tsallis_entmax(INTERIOR_Z, qq))(jnp.asarray(q))
    assert jnp.allclose(jnp.sum(dpdq), 0.0, atol=1e-12)


def test_zero_gradient_off_support():
    z = jnp.array([3.0, 1.0, 0.0, -2.0, -5.0])
    q = 2.5
    p = tsallis_entmax(z, q)
    clipped = p == 0.0
    assert jnp.any(clipped)

    jac = jax.jacobian(lambda zz: tsallis_entmax(zz, q))(z)
    assert jnp.all(jac[clipped, :] == 0.0)
    assert jnp.all(jac[:, clipped] == 0.0)

    dpdq = jax.jacfwd(lambda qq: tsallis_entmax(z, qq))(jnp.asarray(q))
    assert jnp.all(dpdq[clipped] == 0.0)

    hess = jax.hessian(lambda zz: jnp.sum(tsallis_entmax(zz, q) ** 2))(z)
    assert jnp.all(jnp.isfinite(hess))


# ---------------------------------------------------------------------------
# The q < 1 regime.
# ---------------------------------------------------------------------------


def test_q_below_one_is_denser_than_softmax():
    # Regression: the bisection bracket used to assume q > 1, so q < 1 silently
    # returned a near-one-hot vector -- sparser than softmax, when it must be
    # denser. Shannon entropy must decrease monotonically in q.
    z = jnp.array([1.0, 2.0, 3.0])
    entropies = []
    for q in [0.3, 0.5, 0.7, 0.9, 1.0, 1.5, 2.0]:
        p = tsallis_entmax(z, q)
        assert jnp.allclose(jnp.sum(p), 1.0, atol=1e-12)
        if q < 1.0:
            assert jnp.all(p > 0.0), "entmax with q < 1 has full support"
        entropies.append(-jnp.sum(jnp.where(p > 0, p * jnp.log(jnp.where(p > 0, p, 1.0)), 0.0)))

    # Pairwise comparison: entropies[1:] is deliberately one shorter.
    assert all(a > b for a, b in zip(entropies, entropies[1:], strict=False)), entropies
    softmax_entropy = entropies[4]
    assert all(e > softmax_entropy for e in entropies[:4])


@pytest.mark.parametrize("q", [0.0, -1.0])
def test_non_positive_q_raises(q):
    with pytest.raises(ValueError, match="q must be positive"):
        tsallis_entmax(INTERIOR_Z, q)


def test_traced_non_positive_q_is_nan():
    # Under jit a Python raise is impossible, so a bad learned q must poison
    # visibly rather than return a plausible-looking distribution.
    out = jax.jit(lambda z, q: tsallis_entmax(z, q))(INTERIOR_Z, -1.0)
    assert jnp.all(jnp.isnan(out))


# ---------------------------------------------------------------------------
# Invariances and the primal contract.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("q", "expected"),
    [
        (1.3, [0.1930210144, 0.7315717295, 0.0754072561]),
        (1.5, [0.1620701071, 0.8146494395, 0.0232804534]),
        (2.0, [0.0, 1.0, 0.0]),
    ],
)
def test_primal_unchanged_for_q_above_one(q, expected):
    # Golden values from the pre-custom_jvp implementation. The gradient fix must
    # not move the primal.
    assert jnp.allclose(tsallis_entmax(INTERIOR_Z, q), jnp.array(expected), atol=1e-9)


@pytest.mark.parametrize("q", [0.5, 1.5, 2.5])
def test_low_num_iters_still_accurate(q):
    # A Newton polish follows the bisection, so few iterations still land on the
    # constraint -- which the JVP rule assumes.
    coarse = tsallis_entmax(INTERIOR_Z, q, num_iters=10)
    fine = tsallis_entmax(INTERIOR_Z, q, num_iters=50)
    assert jnp.allclose(jnp.sum(coarse), 1.0, atol=1e-12)
    assert jnp.allclose(coarse, fine, atol=1e-7)


@pytest.mark.parametrize("q", [0.6, 1.5, 2.0])
def test_shift_invariance(q):
    shifted = tsallis_entmax(INTERIOR_Z + 10.0, q)
    assert jnp.allclose(shifted, tsallis_entmax(INTERIOR_Z, q), atol=1e-10)


@pytest.mark.parametrize("q", [0.6, 1.5, 2.5])
def test_jit_vmap_axis_consistency(q):
    z = jnp.array([[1.0, 2.0, 0.5], [0.3, -1.0, 0.8]])

    assert jnp.allclose(jax.jit(tsallis_entmax)(z, q), tsallis_entmax(z, q), atol=1e-12)
    assert jnp.allclose(jax.vmap(lambda r: tsallis_entmax(r, q))(z), tsallis_entmax(z, q), 1e-12)
    assert jnp.allclose(tsallis_entmax(z.T, q, axis=0).T, tsallis_entmax(z, q, axis=-1), atol=1e-12)

    grad_fn = jax.grad(lambda zz: jnp.sum(tsallis_entmax(zz, q) ** 2))
    assert jnp.allclose(jax.jit(grad_fn)(z), grad_fn(z), atol=1e-12)


def test_vmap_over_q_including_one():
    qs = jnp.array([0.5, 1.0, 1.5, 2.0])
    ps = jax.vmap(lambda q: tsallis_entmax(INTERIOR_Z, q))(qs)
    assert jnp.allclose(jnp.sum(ps, axis=-1), 1.0, atol=1e-10)
    assert jnp.allclose(ps[1], jax.nn.softmax(INTERIOR_Z), atol=1e-6)

    grads = jax.vmap(lambda q: jax.grad(lambda t: tsallis_entmax(INTERIOR_Z, t)[1])(q))(qs)
    assert jnp.all(jnp.isfinite(grads))
