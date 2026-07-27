"""Property-based tests for the algebraic identities the docstrings claim.

The example-based tests pin specific inputs; these search for counterexamples to
the structural laws of the ``q``-algebra and the information measures. They are
the tests most likely to catch a regression in the ``q -> 1`` handling, since
Hypothesis will happily probe ``q`` right up against ``1``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

import qjax

jax.config.update("jax_enable_x64", True)

SETTINGS = settings(
    max_examples=150,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)

# Entropic indices spanning both regimes, including values very close to 1.
qs = st.floats(min_value=0.05, max_value=3.0, allow_nan=False, allow_infinity=False)
positives = st.floats(min_value=1e-3, max_value=1e3, allow_nan=False, allow_infinity=False)
reals = st.floats(min_value=-5.0, max_value=5.0, allow_nan=False, allow_infinity=False)


def simplex(draw, size):
    """Draw a strictly positive probability vector of the given size."""
    weights = draw(
        st.lists(
            st.floats(min_value=1e-2, max_value=1.0, allow_nan=False),
            min_size=size,
            max_size=size,
        )
    )
    arr = np.asarray(weights, dtype=np.float64)
    return arr / arr.sum()


@st.composite
def simplex_vectors(draw, min_size=2, max_size=6):
    """Draw a probability vector with a random length."""
    size = draw(st.integers(min_value=min_size, max_value=max_size))
    return simplex(draw, size)


# ---------------------------------------------------------------------------
# q-algebra.
# ---------------------------------------------------------------------------


@SETTINGS
@given(x=positives, q=qs)
def test_q_exp_inverts_q_log(x, q):
    assert np.isclose(float(qjax.q_exp(qjax.q_log(x, q), q)), x, rtol=1e-8, atol=1e-10)


@SETTINGS
@given(x=positives, y=positives, q=qs)
def test_q_log_distributes_over_product(x, y, q):
    # ln_q(x y) = ln_q(x) (+)_q ln_q(y)
    lhs = float(qjax.q_log(x * y, q))
    rhs = float(qjax.q_add(qjax.q_log(x, q), qjax.q_log(y, q), q))
    assert np.isclose(lhs, rhs, rtol=1e-7, atol=1e-9)


@SETTINGS
@given(a=reals, b=reals, q=qs)
def test_q_add_diff_are_inverse(a, b, q):
    assume(abs(1.0 + (1.0 - q) * b) > 1e-3)
    assert np.isclose(float(qjax.q_add(qjax.q_diff(a, b, q), b, q)), a, rtol=1e-7, atol=1e-9)


@SETTINGS
@given(a=positives, b=positives, q=qs)
def test_q_prod_div_are_inverse(a, b, q):
    # The q-product carries a Tsallis cut-off for q > 1: once a^(1-q) + b^(1-q)
    # falls below 1 the product saturates (to +inf for q > 1, 0 for q < 1) and
    # the operation stops being invertible. Hypothesis finds this immediately --
    # e.g. a = b = 2, q = 3 -- so the identity is asserted only on the interior,
    # which is exactly where it is claimed to hold.
    product = float(qjax.q_prod(a, b, q))
    assume(np.isfinite(product) and product > 0.0)
    round_trip = float(qjax.q_div(product, b, q))
    assume(np.isfinite(round_trip) and round_trip > 0.0)

    # Separately, the round trip is conditioned by the *dynamic range* of the two
    # deformed terms. a is recovered from a^(1-q) summed against b^(1-q), so when
    # those differ by many orders of magnitude the smaller one -- the one
    # carrying a -- is swamped, and only the surviving digits come back. At
    # a = 487, b = 0.0015, q = 2.98 the ratio is 7e10 and the round trip returns
    # 487.2603 against 487.2550: a relative error of 1e-5 that is arithmetic, not
    # a defect. Rather than loosen the tolerance everywhere, skip the inputs
    # where the identity is not numerically recoverable and keep the rest tight.
    # A 1e8 cutoff retains ~99% of draws with no loss at rtol = 1e-6.
    exponent = 1.0 - q
    powers = (abs(a**exponent), abs(b**exponent))
    assume(all(np.isfinite(p) for p in powers) and min(powers) > 0.0)
    assume(max(powers) / min(powers) < 1e8)
    assert np.isclose(round_trip, a, rtol=1e-6, atol=1e-8)


@SETTINGS
@given(x=positives, q=qs)
def test_q_log_matches_closed_form_away_from_one(x, q):
    assume(abs(q - 1.0) > 1e-6)
    expected = (x ** (1.0 - q) - 1.0) / (1.0 - q)
    assert np.isclose(float(qjax.q_log(x, q)), expected, rtol=1e-7, atol=1e-9)


# ---------------------------------------------------------------------------
# Information measures.
# ---------------------------------------------------------------------------


@SETTINGS
@given(p=simplex_vectors(), q=qs)
def test_entropy_matches_closed_form(p, q):
    assume(abs(q - 1.0) > 1e-6)
    expected = (1.0 - np.sum(p**q)) / (q - 1.0)
    assert np.isclose(float(qjax.tsallis_entropy(jnp.asarray(p), q)), expected, rtol=1e-7)


@SETTINGS
@given(p=simplex_vectors(), q=qs)
def test_entropy_non_negative_and_maximal_at_uniform(p, q):
    entropy = float(qjax.tsallis_entropy(jnp.asarray(p), q))
    uniform = np.full_like(p, 1.0 / p.size)
    uniform_entropy = float(qjax.tsallis_entropy(jnp.asarray(uniform), q))
    assert entropy >= -1e-9
    assert entropy <= uniform_entropy + 1e-9


@SETTINGS
@given(p=simplex_vectors(), q=qs)
def test_divergence_non_negative_and_zero_at_identity(p, q):
    pj = jnp.asarray(p)
    assert abs(float(qjax.tsallis_divergence(pj, pj, q))) < 1e-9
    shuffled = jnp.asarray(np.roll(p, 1))
    assert float(qjax.tsallis_divergence(pj, shuffled, q)) >= -1e-9


@SETTINGS
@given(p=simplex_vectors(), q=qs)
def test_divergence_matches_closed_form(p, q):
    assume(abs(q - 1.0) > 1e-6)
    r = np.roll(p, 1)
    expected = (np.sum(p**q * r ** (1.0 - q)) - 1.0) / (q - 1.0)
    got = float(qjax.tsallis_divergence(jnp.asarray(p), jnp.asarray(r), q))
    assert np.isclose(got, expected, rtol=1e-6, atol=1e-9)


# ---------------------------------------------------------------------------
# entmax.
# ---------------------------------------------------------------------------


@SETTINGS
@given(
    z=st.lists(reals, min_size=2, max_size=6),
    q=st.floats(min_value=0.2, max_value=3.0, allow_nan=False),
)
def test_entmax_is_a_distribution(z, q):
    p = qjax.tsallis_entmax(jnp.asarray(z, dtype=jnp.float64), q)
    assert np.isclose(float(jnp.sum(p)), 1.0, atol=1e-9)
    assert bool(jnp.all(p >= 0.0))


@SETTINGS
@given(
    z=st.lists(reals, min_size=2, max_size=6),
    q=st.floats(min_value=0.2, max_value=3.0, allow_nan=False),
    shift=reals,
)
def test_entmax_is_shift_invariant(z, q, shift):
    zj = jnp.asarray(z, dtype=jnp.float64)
    base = qjax.tsallis_entmax(zj, q)
    shifted = qjax.tsallis_entmax(zj + shift, q)
    assert bool(jnp.allclose(base, shifted, atol=1e-8))


@SETTINGS
@given(
    z=st.lists(reals, min_size=2, max_size=5),
    q=st.floats(min_value=0.3, max_value=2.8, allow_nan=False),
)
def test_entmax_jacobian_is_symmetric(z, q):
    zj = jnp.asarray(z, dtype=jnp.float64)
    jac = jax.jacobian(lambda t: qjax.tsallis_entmax(t, q))(zj)
    assert bool(jnp.allclose(jac, jac.T, atol=1e-10))


@SETTINGS
@given(
    z=st.lists(reals, min_size=2, max_size=5),
    q=st.floats(min_value=0.3, max_value=2.8, allow_nan=False),
)
def test_entmax_jacobian_matches_closed_form(z, q):
    zj = jnp.asarray(z, dtype=jnp.float64)
    p = np.asarray(qjax.tsallis_entmax(zj, q))
    support = p > 0.0
    s = np.where(support, np.where(support, p, 1.0) ** (2.0 - q), 0.0)
    expected = np.diag(s) - np.outer(s, s) / s.sum()
    jac = np.asarray(jax.jacobian(lambda t: qjax.tsallis_entmax(t, q))(zj))
    assert np.allclose(jac, expected, atol=1e-9)
