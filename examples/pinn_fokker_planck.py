r"""Heavy-tailed PINN residuals: the Tsallis reading of a Student-t likelihood.

Physics-informed neural networks minimize a PDE residual, and standard practice
minimizes its *mean square*. Abijuru et al. (ICML 2026) point out what that
silently assumes -- "independent Gaussian residuals with a fixed global variance"
-- and show, theoretically and empirically, that PINN residuals are instead
"heterogeneous and heavy-tailed", so that "a small number of large residuals can
disproportionately dominate both the loss and gradient". Their remedy is a
**Student-t residual model**, fitted by an expectation-maximization loop that
alternates residual-dependent weights with a weighted mean-squared objective.

**That is a Tsallis method with the name removed.** The Student-t *is* the
``q``-Gaussian, and the correspondence is exact at ``qjax``'s own relation
``nu = (3-q)/(q-1)``, the one already in ``qjax.sample``. Their EM weight

    w(r) = (nu + 1) / (nu + r^2/s^2)

is, term for term, the score of ``qjax.q_gaussian_logpdf`` divided by the
gradient of a squared residual:

    d/dr [-log G_q(r)] = w(r) * d/dr [beta r^2],   w(r) = 1 / (1 + (q-1) beta r^2).

The two agree to ~1e-15, which ``tests/test_examples_physics.py`` pins, and at
``q = 1`` the weight is identically one -- so the mean-squared residual is the
Boltzmann-Gibbs member of the family, not a separate baseline.

What ``qjax`` adds is that **the entropic index need not be estimated by EM**. The
EM alternation exists because the tail index is treated as a latent variable; in
``qjax`` ``q`` is an ordinary differentiable argument, finite everywhere including
``q = 1``, so plain gradient descent on the ``q``-Gaussian likelihood trains the
network *and* its own residual model together.

**What the run finds, in three parts.**

*The premise holds, and by more than the original paper claims.* Collecting the
residuals of a mean-squared-trained PINN at *held-out* collocation points and
fitting a ``q``-Gaussian to them by maximum likelihood gives
``q_loss = 2.11 +- 0.03`` and ``2.40 +- 0.02`` at the two equation indices --
forty and eighty-five standard errors from the Gaussian ``q_loss = 1`` that a
mean-squared objective assumes. In Student-t terms that is ``nu = 0.80`` and
``0.43``: **fewer than one degree of freedom, heavier-tailed than a Cauchy
distribution**, so the residuals do not merely have a heavy tail, they have no
finite mean. Because this PDE has a closed-form solution, that is measured
against ground truth rather than against another approximation.

*Where their assumptions hold, their remedy works -- and it is one line here.*
At ``q = 1.5``, where the solution is smooth, the deformed likelihood cuts the
relative error to ``0.75x`` (fixed index) and ``0.67x`` (learned index) of the
mean-squared arm, and it does so at **every one of eight seeds**. That matters
because the seed-to-seed spread is larger than the gap between arms: only the
per-seed pairing -- every arm at a seed trains on the same collocation points --
separates the two, which is why the comparison is reported paired.

*At a free boundary it is catastrophic, and the reason is instructive.* At
``q = 0.5`` the same loss is **fifteen times worse**, at zero seeds out of eight.
A robust loss is, by construction, a loss that *tolerates large residuals*; in a
*forward* PDE problem the residual is not a noise model but the only thing
propagating the initial condition into the interior. Downweight it and the
solution decays into the spurious family this equation carries -- *every*
spatially uniform density solves it exactly. Measured: all three arms fit the
initial condition (peak 1.328, 1.274, 1.273 against the exact 1.3258), but by
``t = 1`` the deformed arms hold a mass of 0.005 against the exact 1.0, with
residuals twenty-five times *larger* than the mean-squared arm and a loss that
does not mind.

So the useful distinction is not whether residuals are heavy-tailed -- they are,
everywhere here -- but **whether the tail is noise or signal**. At a free
boundary it is signal: the residual is large because the solution genuinely has a
kink there, and discounting it means declining to fit the only hard part of the
domain.

*Joint descent is not EM.* The learned index settles at 1.28 and 1.04 while the
index fitted to the residuals says 2.11 and 2.40 -- gaps of thirty and eighty-three
standard errors. Alternating -- fit the residual model to a *fixed* network, then
the network to a fixed model, as EM does -- is not the same as descending on both
at once, because a network free to change its residuals can move them to suit the
index instead of the other way round. Notably the learned-index arm is still the
*best* arm at ``q = 1.5``: mild robustness helps even when the index that
delivers it disagrees with the residuals it is fitted to. Which is an argument for
the EM structure and against reading too much into a learned index, arrived at by
removing the alternation.

Two entropic indices appear here and they are unrelated, so they are named apart
throughout: ``q`` is the *equation's* index, fixed by the physics
(``dp/dt = D d^2 p^{2-q}/dx^2``), and ``q_loss`` is the *residual model's* index,
learned. The equation is solved at ``q = 0.5`` (compact support, a moving free
boundary) and ``q = 1.5`` (power-law tails), because the free boundary is where
residuals are most violently heterogeneous.

The benchmark is stronger than the ones the ICML paper uses, in the one way that
matters here: this PDE has a **closed-form solution**
(``qjax.physics.nlfp_density``, whose own residual vanishes at 1e-15), so "the
residuals are heavy-tailed" is measured against ground truth rather than against
another approximation.

Only the output head is held fixed: `jax.nn.softplus`, the conventional choice.
The ablation is entirely in the loss. The baseline holds its residual *scale*
fixed as well, so it is a plain mean-squared residual and not a Gaussian
likelihood with a fitted variance -- the two differ, because a fitted scale
re-weights the residual term against the initial and boundary terms as training
proceeds.

Run with: ``uv run python examples/pinn_fokker_planck.py``
Add ``--full`` (or set ``QJAX_FULL=1``) for the larger configuration.
"""

from __future__ import annotations

import os
import sys
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import qjax
import qjax.physics as qp
from qjax.nn import bounded_q, inverse_bounded_q
from qjax.plots import qcolors, save_figure, use_qjax_style

# x64 throughout: the exact solution is the reference every number is judged
# against, and its own residual is ~1e-15, which float32 cannot represent.
jax.config.update("jax_enable_x64", True)

FIG_DIR = Path(__file__).parent / "figures"

#: The diffusivity is the sibling particle method's; the initial width is chosen
#: so the initial condition is resolvable on a collocation grid (see the
#: cross-reference in ``examples/anomalous_diffusion.py``).
DIFFUSIVITY, BETA_INITIAL, FINAL_TIME = 1.0, 4.0, 1.0

#: The *equation's* entropic index. 0.5 has compact support and a moving free
#: boundary; 1.5 has power-law tails. The free boundary is where the residual is
#: most heterogeneous, which is the regime the heavy-tail claim is about.
INDICES = (0.5, 1.5)

#: (label, q_loss, learnable, learn_scale). The *residual model's* index --
#: unrelated to the equation's. ``q_loss = 1`` with the scale held fixed is
#: exactly the mean-squared residual, i.e. standard PINN training;
#: ``q_loss = 1.5`` is a Student-t with nu = 3 held fixed, which is the ICML
#: setup; the third arm learns the index too.
#:
#: ``learn_scale`` matters more than it looks. A Gaussian likelihood whose scale
#: is fitted is *not* a mean-squared residual: as the residual shrinks the fitted
#: beta grows, so the residual term's weight relative to the fixed initial and
#: boundary weights drifts during training. The baseline therefore holds its
#: scale fixed, matching the paper's plain-MSE baseline, while the two deformed
#: arms fit theirs by maximum likelihood, matching their EM.
LOSS_ARMS = (
    ("MSE ($q_L = 1$)", 1.0, False, False),
    ("fixed $q_L = 1.5$", 1.5, False, True),
    ("learnable $q_L$", 0.0, True, True),
)
BASELINE_ARM = "MSE ($q_L = 1$)"
LEARNED_ARM = "learnable $q_L$"

#: The residual index is confined to ``q_loss >= 1``. Below 1 the q-Gaussian has
#: compact support, so the likelihood is ``-inf`` for any residual past the
#: cut-off -- an infinitely *un*-robust loss, and the exact opposite of the
#: intended direction. The same trap as the ``q <-> 2-q`` duality: the
#: deformation has a sign.
Q_LOSS_MIN, Q_LOSS_MAX, Q_LOSS_INIT = 1.0, 2.9, 1.3

LR = 1e-3
WEIGHT_INITIAL, WEIGHT_BOUNDARY = 100.0, 100.0


def configuration(*, quick: bool, full: bool) -> dict:
    """Network size, collocation counts and step budget for the requested tier."""
    if quick:
        return {
            "hidden": (16, 16),
            "steps": 400,
            "collocation": 256,
            "initial_points": 64,
            "boundary_points": 32,
            "seeds": (0,),
            "grid": 201,
            "fit_steps": 400,
        }
    if full:
        return {
            "hidden": (64, 64, 64),
            "steps": 30_000,
            "collocation": 4096,
            "initial_points": 512,
            "boundary_points": 256,
            "seeds": (0, 1, 2, 3, 4, 5, 6, 7),
            "grid": 801,
            "fit_steps": 8000,
        }
    return {
        "hidden": (32, 32, 32),
        "steps": 6000,
        "collocation": 1024,
        "initial_points": 256,
        "boundary_points": 128,
        "seeds": (0, 1, 2, 3, 4, 5, 6, 7),
        "grid": 401,
        "fit_steps": 3000,
    }


# --------------------------------------------------------------------------- #
# The correspondence: their EM weight is our score
# --------------------------------------------------------------------------- #
def influence_weight(residual, q_loss, beta_loss):
    r"""Weight by which the deformed loss rescales the mean-squared gradient.

    Differentiating $-\log \mathcal G_{q_L}(r)$ gives
    $w(r)\,\partial_r(\beta r^2)$ with

    $$w(r) = \frac{1}{1 + (q_L - 1)\,\beta\,r^2},$$

    so the deformed objective *is* a weighted mean-squared residual whose weights
    fall off with the residual -- which is exactly the M-step of the Student-t EM
    algorithm. At $q_L = 1$ the weight is identically one and nothing is
    reweighted.
    """
    q_loss = jnp.asarray(q_loss, dtype=jnp.result_type(float))
    return 1.0 / (1.0 + (q_loss - 1.0) * beta_loss * residual**2)


def student_t_weight(residual, degrees, scale):
    r"""The Student-t EM weight $ (\nu+1)/(\nu + r^2/s^2)$, for comparison.

    Written out in the source deliberately: the claim that the ICML construction
    is a ``qjax`` construction is checkable rather than rhetorical, and
    `matched_student_t` supplies the parameter map that makes the two coincide.
    """
    return (degrees + 1.0) / (degrees + residual**2 / scale**2)


def matched_student_t(q_loss, beta_loss):
    r"""The $(\nu, s)$ of the Student-t that equals the $q_L$-Gaussian.

    Matching the two scores term by term gives $\nu = (3-q)/(q-1)$ -- the same
    relation ``qjax.sample`` already uses to draw $q$-Gaussian variates -- and
    $\nu s^2 = 1/((q-1)\beta)$.
    """
    q_loss = jnp.asarray(q_loss, dtype=jnp.result_type(float))
    degrees = (3.0 - q_loss) / (q_loss - 1.0)
    return degrees, jnp.sqrt(1.0 / ((q_loss - 1.0) * beta_loss * degrees))


# --------------------------------------------------------------------------- #
# The network
# --------------------------------------------------------------------------- #
def domain_half_width(q) -> float:
    """Spatial half-width for one equation index, by a rule fixed in advance.

    For ``q < 1``, 1.5 times the front position at the final time; for ``q >= 1``,
    the smallest half-width leaving at most 2e-3 of the mass outside. Stated so it
    cannot look tuned after the fact.
    """
    front = float(qp.nlfp_front(FINAL_TIME, q, DIFFUSIVITY, BETA_INITIAL))
    if np.isfinite(front):
        return float(np.ceil(1.5 * front * 4.0) / 4.0)
    for candidate in np.arange(2.0, 200.0, 0.5):
        grid = jnp.linspace(-candidate, candidate, 40_001)
        inside = jnp.trapezoid(
            qp.nlfp_density(grid, FINAL_TIME, q, DIFFUSIVITY, BETA_INITIAL), grid
        )
        if float(1.0 - inside) <= 2e-3:
            return float(candidate)
    raise ValueError(f"no half-width below 200 keeps the q = {q} tail inside")  # pragma: no cover


def truncated_mass(q, half_width: float, times) -> jax.Array:
    """Exact mass inside the domain at each time -- not 1, for a truncated tail."""
    grid = jnp.linspace(-half_width, half_width, 20_001)
    return jnp.stack(
        [jnp.trapezoid(qp.nlfp_density(grid, t, q, DIFFUSIVITY, BETA_INITIAL), grid) for t in times]
    )


def output_map(q, half_width: float) -> tuple[float, float]:
    """Affine pre-activation map ``z = centre + scale * network``, from the IC.

    Fixed by inverting the softplus head on the exact initial condition -- given
    data for the problem, not knowledge of the answer. With the last layer
    initialized to zero it makes the network *start* at a sensible constant
    density instead of at ``softplus(0)``, and it is pure conditioning: identical
    for every loss arm, so it cannot favour one. Measured, dropping it costs a
    factor of fifteen in final error, which is why it is here.
    """
    positions = jnp.linspace(-half_width, half_width, 2001)
    exact = qp.nlfp_density(positions, 0.0, q, DIFFUSIVITY, BETA_INITIAL)
    peak = jnp.max(exact)
    target = jnp.log(jnp.expm1(jnp.maximum(exact, 1e-6 * peak)))
    lowest, highest = float(jnp.min(target)), float(jnp.max(target))
    return 0.5 * (lowest + highest), max(0.5 * (highest - lowest), 1e-6)


def init_params(key: jax.Array, hidden: tuple[int, ...], peak: float) -> dict:
    """Glorot tanh MLP plus the two residual-model parameters.

    ``q_raw`` and ``beta_raw`` live in the same pytree as the weights, so one
    ``jax.grad`` trains the network and its own residual model together -- which
    is the whole point, and what replaces the EM alternation.
    """
    widths = (2, *hidden, 1)
    keys = jax.random.split(key, len(widths) - 1)
    weights, biases = [], []
    for index, (layer_key, fan_in, fan_out) in enumerate(
        zip(keys, widths[:-1], widths[1:], strict=True)
    ):
        last = index == len(widths) - 2
        scale = 0.0 if last else jnp.sqrt(2.0 / (fan_in + fan_out))
        weights.append(jax.random.normal(layer_key, (fan_in, fan_out)) * scale)
        biases.append(jnp.zeros((fan_out,)))
    return {
        "weights": weights,
        "biases": biases,
        "q_raw": inverse_bounded_q(Q_LOSS_INIT, Q_LOSS_MIN, Q_LOSS_MAX),
        # The residual scale starts at the scale of the equation's own terms.
        "beta_raw": jnp.log(jnp.expm1((FINAL_TIME / peak) ** 2)),
    }


def adam_rate(step, total: int):
    """Cosine-annealed learning rate, from ``LR`` down to one per cent of it.

    A function of the *global* step: the training scan runs in blocks, and
    feeding this the within-block index would leave the rate pinned near ``LR``
    for the whole run.
    """
    decay = 0.5 * (1.0 + jnp.cos(jnp.pi * jnp.asarray(step, dtype=jnp.result_type(float)) / total))
    return LR * (0.01 + 0.99 * decay)


def boundary_errors(net, times, half_width: float, q, peak: float):
    """Signed boundary error at each end of the domain, shape ``(n, 2)``.

    Both ends are kept separate on purpose. Penalizing their *sum* -- which is
    the natural one-liner -- would let an error at ``+L`` be cancelled by the
    opposite error at ``-L`` at no cost, leaving an antisymmetric boundary error
    free.
    """
    predicted = jax.vmap(lambda t: jnp.stack([net(half_width, t), net(-half_width, t)]))(times)
    exact = qp.nlfp_density(half_width, times, q, DIFFUSIVITY, BETA_INITIAL)[:, None]
    return (predicted - exact) / peak


def resolve_loss_index(params: dict, q_fixed, learnable: bool):
    """The residual model's index: a constant, or the bounded learnable one."""
    if learnable:
        return bounded_q(params["q_raw"], Q_LOSS_MIN, Q_LOSS_MAX)
    return q_fixed


def density(params: dict, x, t, half_width: float, centre, scale):
    """The network's density, through a softplus head -- fixed for every arm."""
    activation = jnp.stack([x / half_width, 2.0 * t / FINAL_TIME - 1.0])
    for weight, bias in zip(params["weights"][:-1], params["biases"][:-1], strict=True):
        activation = jnp.tanh(activation @ weight + bias)
    raw = (activation @ params["weights"][-1] + params["biases"][-1])[0]
    return jax.nn.softplus(centre + scale * raw)


# --------------------------------------------------------------------------- #
# Losses
# --------------------------------------------------------------------------- #
def sample_points(key: jax.Array, config: dict, q, half_width: float) -> dict:
    """Collocation, initial-condition and boundary points, shared by every arm.

    Time is sampled log-uniformly in ``t + t_star``, the variable the width is a
    power law in, which also concentrates points early where a PINN's causality
    pathology bites. Space is uniform: importance-sampling toward the front would
    require knowing the answer.
    """
    space_key, time_key, initial_key, boundary_key = jax.random.split(key, 4)
    offset = float(qp.nlfp_offset(q, DIFFUSIVITY, BETA_INITIAL))
    fraction = jax.random.uniform(time_key, (config["collocation"],))
    return {
        "x": jax.random.uniform(
            space_key, (config["collocation"],), minval=-half_width, maxval=half_width
        ),
        "t": offset * ((1.0 + FINAL_TIME / offset) ** fraction - 1.0),
        "initial_x": jax.random.uniform(
            initial_key, (config["initial_points"],), minval=-half_width, maxval=half_width
        ),
        "boundary_t": jax.random.uniform(
            boundary_key, (config["boundary_points"],), maxval=FINAL_TIME
        ),
    }


def residuals_of(params, points, q, half_width, centre, scale):
    """PDE residual at every collocation point, non-dimensionalized."""

    def net(x, t):
        return density(params, x, t, half_width, centre, scale)

    return jax.vmap(lambda x, t: qp.nlfp_residual(net, x, t, q, DIFFUSIVITY))(
        points["x"], points["t"]
    )


def total_loss(params, points, q, q_fixed, learnable, learn_scale, half_width, peak, centre, scale):
    r"""Residual negative log-likelihood, plus initial and boundary conditions.

    The residual term is $-\langle \log \mathcal G_{q_L}(r)\rangle$ with the
    *normalized* $q$-Gaussian, which matters: the $\sqrt\beta / C_q$ factor is
    what stops the fit from driving the scale to zero, so ``beta_loss`` is a
    genuine maximum-likelihood estimate rather than a free scale the optimizer can
    game. ``qjax.normalization`` supplies $C_q$ for every index, so the whole
    family is available without a separate normalizing constant per arm.

    The initial and boundary terms stay mean-squared. The heavy-tail claim is
    about the *PDE residual*, and those two terms are also the only thing ruling
    out the spurious minimizers -- every spatially uniform density solves this
    equation exactly, so a residual-only objective has a one-parameter family of
    global minima it can fall into.
    """
    residual = residuals_of(params, points, q, half_width, centre, scale) / (peak / FINAL_TIME)
    q_loss = resolve_loss_index(params, q_fixed, learnable)
    beta_loss = jax.nn.softplus(params["beta_raw"]) + 1e-6
    if not learn_scale:
        beta_loss = jax.lax.stop_gradient(beta_loss)
    physics = -jnp.mean(qjax.q_gaussian_logpdf(residual, q_loss, beta_loss))

    def net(x, t):
        return density(params, x, t, half_width, centre, scale)

    initial = jax.vmap(lambda x: net(x, 0.0))(points["initial_x"])
    exact_initial = qp.nlfp_density(points["initial_x"], 0.0, q, DIFFUSIVITY, BETA_INITIAL)
    boundary = boundary_errors(net, points["boundary_t"], half_width, q, peak)

    return (
        physics
        + WEIGHT_INITIAL * jnp.mean(((initial - exact_initial) / peak) ** 2)
        + WEIGHT_BOUNDARY * jnp.mean(boundary**2)
    )


# --------------------------------------------------------------------------- #
# Fitting an index to a set of residuals
# --------------------------------------------------------------------------- #
@partial(jax.jit, static_argnames=("steps",))
def fit_index(samples, steps: int, learning_rate=0.02):
    r"""Maximum-likelihood $(q, \beta)$ of a $q$-Gaussian fitted to residuals.

    The independent measurement the learned index is checked against. Uses the
    same idiom as ``examples/anomalous_diffusion.py`` -- plain gradient descent on
    ``q_gaussian_logpdf``, with ``q`` an ordinary differentiable parameter -- plus
    an asymptotic Fisher error bar, so "the residuals are heavy-tailed" comes with
    a number of standard errors rather than an adjective.

    Returns:
        ``(q_hat, beta_hat, q_sigma)``.
    """
    samples = jnp.asarray(samples, dtype=jnp.result_type(float))
    # A *robust* scale to anchor the width parameter on. The sample standard
    # deviation is the obvious choice and the wrong one: above q = 5/3 the
    # q-Gaussian has infinite variance, so std is dominated by whichever outliers
    # happened to be drawn, and the fit then cannot reach the right width at all
    # -- measured, a planted q = 2.2 came back as 1.07. The median absolute
    # deviation is finite for every q < 3. (0.6745 makes it agree with the
    # standard deviation in the Gaussian limit.)
    spread = jnp.median(jnp.abs(samples - jnp.median(samples))) / 0.6745 + 1e-12

    def unpack(raw):
        return (
            bounded_q(raw[0], Q_LOSS_MIN, Q_LOSS_MAX),
            jax.nn.softplus(raw[1]) / spread**2 + 1e-12,
        )

    def negative_log_likelihood(raw):
        q, beta = unpack(raw)
        return -jnp.mean(qjax.q_gaussian_logpdf(samples, q, beta))

    raw = jnp.array([inverse_bounded_q(1.5, Q_LOSS_MIN, Q_LOSS_MAX), 0.0])

    # Adam, not plain gradient descent. The bounded_q reparameterization is a
    # sigmoid, so the gradient in the raw coordinate is small wherever the index
    # is far from the middle of its range, and a fixed-step method crawls: at a
    # planted q = 2 it stalls near 1.2 even after 30000 steps, which would have
    # made an under-converged estimator look like a finding.
    b1, b2, eps = 0.9, 0.999, 1e-8
    m = jnp.zeros_like(raw)
    v = jnp.zeros_like(raw)

    def step(state, t):
        raw, m, v = state
        grads = jax.grad(negative_log_likelihood)(raw)
        m = b1 * m + (1.0 - b1) * grads
        v = b2 * v + (1.0 - b2) * grads**2
        bias1, bias2 = 1.0 - b1 ** (t + 1), 1.0 - b2 ** (t + 1)
        raw = raw - learning_rate * (m / bias1) / (jnp.sqrt(v / bias2) + eps)
        return (raw, m, v), None

    (raw, _, _), _ = jax.lax.scan(step, (raw, m, v), jnp.arange(steps))
    q_hat, beta_hat = unpack(raw)

    covariance = jnp.linalg.inv(jax.hessian(negative_log_likelihood)(raw)) / samples.shape[0]
    sensitivity = jax.jacfwd(lambda r: unpack(r)[0])(raw)
    return q_hat, beta_hat, jnp.sqrt(jnp.maximum(sensitivity @ covariance @ sensitivity, 0.0))


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
@partial(jax.jit, static_argnames=("learnable", "learn_scale", "steps", "hidden", "trace_every"))
def solve(
    key,
    points,
    q,
    q_fixed,
    learnable,
    learn_scale,
    steps,
    hidden,
    trace_every,
    half_width,
    peak,
    centre,
    scale,
):
    """Adam on the PINN loss; returns parameters and traces.

    One ``jax.grad`` covers the network *and* the residual model, because
    ``q_loss`` and ``beta_loss`` sit in the same pytree and ``qjax.q_gaussian_logpdf``
    is differentiable in its index. That is what replaces the EM alternation.

    Training is scanned in blocks so the trace can be recorded without keeping a
    per-step history, and the *global* step index is carried through the blocks:
    both Adam's bias correction and the cosine schedule are functions of total
    progress, and feeding either the within-block index restarts the warm-up 40
    times and leaves the learning rate pinned at its initial value.
    """
    params = init_params(key, hidden, peak)
    b1, b2, eps = 0.9, 0.999, 1e-8
    m = jax.tree_util.tree_map(jnp.zeros_like, params)
    v = jax.tree_util.tree_map(jnp.zeros_like, params)

    grid_x = jnp.linspace(-half_width, half_width, 129)
    grid_t = jnp.linspace(0.0, FINAL_TIME, 17)
    mesh_x, mesh_t = jnp.meshgrid(grid_x, grid_t, indexing="ij")
    reference = qp.nlfp_density(mesh_x, mesh_t, q, DIFFUSIVITY, BETA_INITIAL)

    def relative_error(p):
        predicted = jax.vmap(jax.vmap(lambda x, t: density(p, x, t, half_width, centre, scale)))(
            mesh_x, mesh_t
        )
        return jnp.linalg.norm(predicted - reference) / jnp.linalg.norm(reference)

    def step(carry, t):
        p, m, v = carry
        grads = jax.grad(total_loss)(
            p, points, q, q_fixed, learnable, learn_scale, half_width, peak, centre, scale
        )
        norm = jnp.sqrt(
            sum(jnp.sum(g**2) for g in grads["weights"] + grads["biases"])
            + grads["q_raw"] ** 2
            + grads["beta_raw"] ** 2
        )
        factor = jnp.minimum(1.0, 1.0 / jnp.maximum(norm, 1e-12))
        grads = jax.tree_util.tree_map(lambda g: g * factor, grads)
        m = jax.tree_util.tree_map(lambda m, g: b1 * m + (1 - b1) * g, m, grads)
        v = jax.tree_util.tree_map(lambda v, g: b2 * v + (1 - b2) * g * g, v, grads)
        bc1, bc2 = 1 - b1 ** (t + 1), 1 - b2 ** (t + 1)
        rate = adam_rate(t, total)
        p = jax.tree_util.tree_map(
            lambda p, m, v: p - rate * (m / bc1) / (jnp.sqrt(v / bc2) + eps), p, m, v
        )
        return (p, m, v), None

    # The scan runs whole blocks, so the schedule spans what is actually run.
    total = max(steps // trace_every, 1) * trace_every

    def block(carry, first):
        carry, _ = jax.lax.scan(step, carry, first + jnp.arange(trace_every))
        p = carry[0]
        return carry, jnp.stack([relative_error(p), resolve_loss_index(p, q_fixed, learnable)])

    blocks = max(steps // trace_every, 1)
    (params, _, _), trace = jax.lax.scan(block, (params, m, v), trace_every * jnp.arange(blocks))
    return params, trace


# --------------------------------------------------------------------------- #
# Measurement
# --------------------------------------------------------------------------- #
def measure(params, points, q, q_fixed, learnable, config, half_width, peak, centre, scale) -> dict:
    """Accuracy, mass, and the residual distribution the whole example is about."""
    grid_x = jnp.linspace(-half_width, half_width, config["grid"])
    times = jnp.linspace(0.0, FINAL_TIME, 33)

    predicted = jnp.stack(
        [
            jax.vmap(lambda x, t=t: density(params, x, t, half_width, centre, scale))(grid_x)
            for t in times
        ]
    )
    reference = qp.nlfp_density(grid_x[None, :], times[:, None], q, DIFFUSIVITY, BETA_INITIAL)
    error = float(jnp.linalg.norm(predicted - reference) / jnp.linalg.norm(reference))
    mass = jnp.stack([jnp.trapezoid(row, grid_x) for row in predicted])

    residual = residuals_of(params, points, q, half_width, centre, scale) / (peak / FINAL_TIME)
    standardized = residual / (jnp.std(residual) + 1e-12)
    q_hat, beta_hat, q_sigma = fit_index(residual, config["fit_steps"])

    # Excess kurtosis: 0 for a Gaussian, positive for a heavy tail. A second,
    # assumption-free witness that the residuals are not Gaussian.
    kurtosis = float(jnp.mean(standardized**4) - 3.0)

    # Collapse witnesses. Every spatially uniform density solves this equation, so
    # a loss that stops penalizing large residuals can let the solution decay into
    # that family after the initial condition has been fitted.
    initial_peak = float(jnp.max(predicted[0]))
    final_mass = float(jnp.trapezoid(predicted[-1], grid_x))

    return {
        "l2_error": error,
        "mass_error": float(jnp.max(jnp.abs(mass - truncated_mass(q, half_width, times)))),
        "initial_peak": initial_peak,
        "final_mass": final_mass,
        "rms_residual": float(jnp.sqrt(jnp.mean(residual**2))),
        "residual": np.asarray(residual),
        "q_from_residuals": float(q_hat),
        "q_residual_sigma": float(q_sigma),
        "beta_from_residuals": float(beta_hat),
        "kurtosis": kurtosis,
        "q_loss_final": float(resolve_loss_index(params, q_fixed, learnable)),
        "beta_loss_final": float(jax.nn.softplus(params["beta_raw"]) + 1e-6),
        "grid_x": np.asarray(grid_x),
        "profile": np.asarray(predicted[-1]),
    }


def run(config: dict) -> dict:
    """Solve every (equation index, loss arm, seed) and collect the metrics."""
    trace_every = max(config["steps"] // 40, 1)
    results: dict = {
        "traces": {},
        "metrics": {},
        "half_width": {},
        "config": config,
        "trace_every": trace_every,
    }

    for q in INDICES:
        half_width = domain_half_width(q)
        results["half_width"][q] = half_width
        peak = float(qp.nlfp_density(0.0, 0.0, q, DIFFUSIVITY, BETA_INITIAL))
        centre, scale = output_map(q, half_width)
        # One collocation set per seed, shared by every arm at that seed: the arm
        # comparison stays paired, while the seed spread reflects point placement
        # as well as initialization.
        draw = partial(sample_points, config=config, q=q, half_width=half_width)
        points = jax.vmap(draw)(jnp.stack([jax.random.PRNGKey(seed) for seed in config["seeds"]]))
        # A *held-out* collocation set for the residual distribution. The whole
        # finding is about that distribution, so it is measured off the points the
        # arm trained on.
        held_out = jax.vmap(draw)(
            jnp.stack([jax.random.PRNGKey(9000 + seed) for seed in config["seeds"]])
        )
        print(f"  equation q = {q}  (half-width {half_width:.2f})")

        for label, q_fixed, learnable, learn_scale in LOSS_ARMS:
            keys = jnp.stack([jax.random.PRNGKey(100 + seed) for seed in config["seeds"]])
            one_seed = partial(
                solve,
                q=q,
                q_fixed=q_fixed,
                learnable=learnable,
                learn_scale=learn_scale,
                steps=config["steps"],
                hidden=config["hidden"],
                trace_every=trace_every,
                half_width=half_width,
                peak=peak,
                centre=centre,
                scale=scale,
            )
            stacked, traces = jax.vmap(one_seed)(keys, points)
            metrics = [
                measure(
                    jax.tree_util.tree_map(lambda leaf, i=i: leaf[i], stacked),
                    jax.tree_util.tree_map(lambda leaf, i=i: leaf[i], held_out),
                    q,
                    q_fixed,
                    learnable,
                    config,
                    half_width,
                    peak,
                    centre,
                    scale,
                )
                for i in range(len(config["seeds"]))
            ]
            results["traces"][q, label] = np.asarray(traces)
            results["metrics"][q, label] = metrics

            errors = np.array([m["l2_error"] for m in metrics])
            measured = np.array([m["q_from_residuals"] for m in metrics])
            print(
                f"    {label:<20} L2 {np.median(errors):.4f}"
                f" [{errors.min():.4f}, {errors.max():.4f}]"
                f"   q from residuals {np.median(measured):.3f}"
                f"   excess kurtosis {np.median([m['kurtosis'] for m in metrics]):.1f}"
                f"   q_loss {np.median([m['q_loss_final'] for m in metrics]):.3f}",
                flush=True,
            )
    return results


# --------------------------------------------------------------------------- #
# Figure
# --------------------------------------------------------------------------- #
def plot_main(results: dict, path: Path) -> None:
    """The residual distribution, the correspondence, and what learning q buys."""
    labels = [label for label, *_ in LOSS_ARMS]
    arm_colors = dict(zip(labels, qcolors(len(labels)), strict=True))

    fig, axes = plt.subplots(2, 3, figsize=(13.4, 7.4))
    fig.subplots_adjust(hspace=0.38, wspace=0.31)

    # (a) the physics: the solution the residuals belong to.
    ax = axes[0, 0]
    for shade, q in zip((0.35, 0.8), INDICES, strict=True):
        reference = results["metrics"][q, BASELINE_ARM][0]
        grid = reference["grid_x"]
        ax.semilogy(
            grid,
            np.asarray(qp.nlfp_density(grid, FINAL_TIME, q, DIFFUSIVITY, BETA_INITIAL)),
            color=str(1.0 - shade),
            lw=2.6,
            alpha=0.45,
            label=f"exact, $q={q}$",
        )
        ax.semilogy(
            grid,
            reference["profile"],
            color=qcolors(2)[int(shade > 0.5)],
            lw=1.3,
            ls="--",
            label=f"PINN, $q={q}$",
        )
    ax.set_ylim(1e-6, 3.0)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$p(x, 1)$")
    ax.set_title("(a) the solution, at both equation indices", fontsize=10)
    ax.legend(fontsize=7.4)

    # (b) THE ICML CLAIM, MEASURED: the residuals are not Gaussian. Shown as a
    #     survival function on log-log axes rather than a histogram -- with ~1e3
    #     residuals per seed a histogram is all spikes out where the tail is, and
    #     the tail is the whole question.
    ax = axes[0, 1]
    q = INDICES[0]
    pooled = np.concatenate([m["residual"] for m in results["metrics"][q, BASELINE_ARM]])
    pooled = np.abs(pooled) / (np.median(np.abs(pooled)) / 0.6745)
    ordered = np.sort(pooled)
    survival = 1.0 - np.arange(ordered.size) / ordered.size
    keep = ordered > 0.2
    ax.loglog(
        ordered[keep],
        survival[keep],
        color=arm_colors[BASELINE_ARM],
        lw=1.6,
        label="residuals of the MSE arm",
    )

    grid = np.logspace(np.log10(0.2), np.log10(max(ordered.max(), 10.0)), 200)
    ax.loglog(
        grid,
        2.0 * (1.0 - 0.5 * (1.0 + jax.scipy.special.erf(grid / np.sqrt(2.0)))),
        color="0.35",
        lw=1.2,
        ls="--",
        label="Gaussian (what MSE assumes)",
    )

    fitted_q = float(
        np.median([m["q_from_residuals"] for m in results["metrics"][q, BASELINE_ARM]])
    )
    # Survival of the fitted q-Gaussian, by quadrature on its own density.
    fine = jnp.linspace(0.0, float(max(ordered.max(), 10.0)) * 3.0, 40_001)
    scaled_beta = 1.0 / ((5.0 - 3.0 * min(fitted_q, 1.6)) if fitted_q < 5 / 3 else 1.0)
    pdf = qjax.q_gaussian_pdf(fine, fitted_q, scaled_beta)
    tail = 1.0 - 2.0 * jnp.concatenate(
        [jnp.zeros(1), jnp.cumsum(0.5 * (pdf[1:] + pdf[:-1]) * jnp.diff(fine))]
    )
    ax.loglog(
        np.asarray(fine),
        np.maximum(np.asarray(tail), 1e-6),
        color=arm_colors[LEARNED_ARM],
        lw=2.6,
        alpha=0.45,
        label=rf"fitted $q$-Gaussian, $\hat q_L={fitted_q:.2f}$",
    )

    ax.set_xlim(0.2, max(ordered.max(), 10.0))
    ax.set_ylim(1.0 / ordered.size, 1.5)
    ax.set_xlabel("standardized $|$residual$|$")
    ax.set_ylabel(r"$P(|r| > u)$")
    ax.set_title(rf"(b) the residuals are heavy-tailed (equation $q={q}$)", fontsize=10)
    ax.legend(fontsize=7.0, loc="lower left")

    # (c) the correspondence: their EM weight IS our score.
    ax = axes[0, 2]
    grid = np.linspace(0.0, 8.0, 400)
    for colour, q_loss in zip(qcolors(4), (1.0, 1.3, 1.7, 2.3), strict=True):
        ours = np.asarray(influence_weight(jnp.asarray(grid), q_loss, 1.0))
        ax.plot(grid, ours, color=colour, lw=1.5, label=rf"$q_L={q_loss}$")
        if q_loss > 1.0:
            # Their EM weight, expressed against the same denominator as ours --
            # the mean-squared gradient 2 beta r -- so the two are directly
            # comparable rather than differing by a convention.
            degrees, scale = matched_student_t(q_loss, 1.0)
            probe = jnp.asarray(np.maximum(grid, 1e-9))
            theirs = (student_t_weight(probe, degrees, scale) * probe / scale**2) / (
                2.0 * 1.0 * probe
            )
            ax.plot(grid, np.asarray(theirs), color="0.2", lw=3.0, alpha=0.22)
    ax.set_xlabel("standardized residual $r$")
    ax.set_ylabel("weight on the squared-error gradient")
    ax.set_title(r"(c) $q$-Gaussian score $=$ Student-$t$ EM weight", fontsize=9.5)
    ax.annotate(
        "thick grey: Student-$t$ EM weight\nat $\\nu=(3-q_L)/(q_L-1)$",
        (0.40, 0.62),
        xycoords="axes fraction",
        fontsize=6.8,
        color="0.3",
    )
    ax.legend(fontsize=7.6)

    # (d) THE CONSISTENCY CHECK: learned index vs measured index.
    ax = axes[1, 0]
    steps = (np.arange(results["traces"][INDICES[0], BASELINE_ARM].shape[1]) + 1) * results[
        "trace_every"
    ]
    for shade, q in zip((0.4, 0.85), INDICES, strict=True):
        trace = results["traces"][q, LEARNED_ARM][:, :, 1]
        colour = qcolors(2)[int(shade > 0.5)]
        ax.plot(
            steps,
            np.median(trace, axis=0),
            color=colour,
            lw=1.5,
            label=rf"learned $q_L$, equation $q={q}$",
        )
        ax.fill_between(steps, trace.min(axis=0), trace.max(axis=0), color=colour, alpha=0.15, lw=0)
        measured = np.median([m["q_from_residuals"] for m in results["metrics"][q, BASELINE_ARM]])
        ax.axhline(measured, color=colour, lw=1.1, ls="--")
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(plt.matplotlib.ticker.LogFormatterSciNotation())
    ax.xaxis.set_minor_formatter(plt.matplotlib.ticker.NullFormatter())
    ax.set_xlabel("Adam step")
    ax.set_ylabel(r"$q_L$")
    ax.set_title(r"(d) learned $q_L$ vs. $q_L$ fitted to the residuals (dashed)", fontsize=9.5)
    ax.legend(fontsize=7.2)

    # (e) accuracy against the exact solution.
    ax = axes[1, 1]
    for q, style in zip(INDICES, ("-", "--"), strict=True):
        for label in labels:
            trace = results["traces"][q, label][:, :, 0]
            ax.loglog(
                steps,
                np.median(trace, axis=0),
                color=arm_colors[label],
                ls=style,
                lw=1.3,
                label=f"$q={q}$, {label}",
            )
    ax.xaxis.set_major_formatter(plt.matplotlib.ticker.LogFormatterSciNotation())
    ax.xaxis.set_minor_formatter(plt.matplotlib.ticker.NullFormatter())
    ax.set_xlabel("Adam step")
    ax.set_ylabel(r"relative $L^2$ vs. exact")
    ax.set_title("(e) held-out error (median over seeds)", fontsize=10)
    ax.legend(fontsize=6.4, ncols=2, loc="lower left")

    # (f) the comparison, paired by seed. Arm-to-arm differences at q = 1.5 are
    #     smaller than the seed-to-seed spread, so plotting medians with a seed
    #     range hides the result: every seed's own ratio is what settles it.
    ax = axes[1, 2]
    deformed = [label for label in labels if label != BASELINE_ARM]
    positions = np.arange(len(INDICES) * len(deformed), dtype=float)
    ax.axhline(1.0, color="0.35", lw=1.0, ls=":", zorder=1)
    ticks = []
    for slot, (q, label) in enumerate([(q, label) for q in INDICES for label in deformed]):
        baseline = np.array([m["l2_error"] for m in results["metrics"][q, BASELINE_ARM]])
        ratio = np.array([m["l2_error"] for m in results["metrics"][q, label]]) / baseline
        jitter = np.linspace(-0.13, 0.13, len(ratio))
        ax.scatter(
            positions[slot] + jitter,
            ratio,
            s=17,
            color=arm_colors[label],
            edgecolor="white",
            linewidth=0.4,
            zorder=3,
        )
        ax.hlines(
            float(np.median(ratio)),
            positions[slot] - 0.24,
            positions[slot] + 0.24,
            color=arm_colors[label],
            lw=2.2,
            zorder=4,
        )
        wins = int(np.sum(ratio < 1.0))
        ax.annotate(
            f"{wins}/{len(ratio)}",
            (positions[slot], float(np.median(ratio))),
            textcoords="offset points",
            xytext=(0, -15 if np.median(ratio) < 1.0 else 9),
            ha="center",
            fontsize=7.2,
            color=arm_colors[label],
        )
        ticks.append(f"$q={q}$\n{label}")
    ax.set_yscale("log")
    ax.set_xticks(positions)
    ax.set_xticklabels(ticks, fontsize=6.8)
    ax.set_ylabel(r"$L^2$ error $\div$ MSE arm, same seed")
    ax.set_title("(f) paired by seed (dashes: median)", fontsize=10)
    ax.text(
        0.03,
        0.5,
        "below 1: the\ndeformed loss helps",
        transform=ax.transAxes,
        fontsize=6.8,
        color="0.35",
        va="center",
    )

    save_figure(fig, path)
    plt.close(fig)


# --------------------------------------------------------------------------- #
def report(results: dict) -> None:
    """Print the validation table the docs page reproduces."""
    labels = [label for label, *_ in LOSS_ARMS]

    print("\n  validation")
    print(f"    {'quantity':<50}{'measured':>14}{'exact':>14}")

    # 1. The exact solution's own residual: what validates the closed form.
    worst, scale = 0.0, 0.0
    for q in INDICES:

        def exact(x, t, q=q):
            return qp.nlfp_density(x, t, q, DIFFUSIVITY, BETA_INITIAL)

        positions = jnp.linspace(-1.0, 1.0, 41)
        for t in (0.05, 0.5, 1.0):
            residual = jax.vmap(lambda x, t=t, q=q: qp.nlfp_residual(exact, x, t, q, DIFFUSIVITY))(
                positions
            )
            rate = jax.vmap(lambda x, t=t: jax.grad(exact, argnums=1)(x, t))(positions)
            worst = max(worst, float(jnp.max(jnp.abs(residual))))
            scale = max(scale, float(jnp.max(jnp.abs(rate))))
    print(f"    {'residual of the exact solution (max)':<50}{worst:>14.2e}{0.0:>14.2e}")
    name = "  ... relative to max |dp/dt| on the same points"
    print(f"    {name:<50}{worst / scale:>14.2e}{0.0:>14.2e}")

    # 2. The correspondence, checked here and not only asserted in the docstring.
    #    Compare the two *scores* -- the quantity that actually enters a gradient
    #    step -- rather than weights, which differ by a convention-dependent factor.
    probes = jnp.linspace(-6.0, 6.0, 241)
    gap = 0.0
    for q_loss in (1.2, 1.5, 2.0, 2.5):
        degrees, scale_t = matched_student_t(q_loss, 1.0)
        ours = jax.vmap(jax.grad(lambda r, q=q_loss: -qjax.q_gaussian_logpdf(r, q, 1.0)))(probes)
        theirs = student_t_weight(probes, degrees, scale_t) * probes / scale_t**2
        gap = max(gap, float(jnp.max(jnp.abs(ours - theirs))))
    name = "q-Gaussian score vs Student-t EM score (max gap)"
    print(f"    {name:<50}{gap:>14.2e}{0.0:>14.2e}")
    name = "weight at q_L = 1 (MSE: no reweighting)"
    print(f"    {name:<50}{float(influence_weight(3.0, 1.0, 1.0)):>14.6f}{1.0:>14.6f}")

    print("\n    are the residuals actually heavy-tailed? (the ICML premise, measured)")
    columns = ("equation q", "q from residuals", "sigma", "excess kurtosis", "Student-t nu")
    print(f"      {columns[0]:<12}{columns[1]:>18}{columns[2]:>9}{columns[3]:>18}{columns[4]:>15}")
    for q in INDICES:
        metrics = results["metrics"][q, BASELINE_ARM]
        measured = np.median([m["q_from_residuals"] for m in metrics])
        sigma = np.median([m["q_residual_sigma"] for m in metrics])
        kurt = np.median([m["kurtosis"] for m in metrics])
        degrees = (3.0 - measured) / (measured - 1.0) if measured > 1.0 else float("inf")
        print(f"      {q:<12}{measured:>18.4f}{sigma:>9.4f}{kurt:>18.1f}{degrees:>15.2f}")
    print("      (a Gaussian would give q = 1 and excess kurtosis 0)")

    print("\n    the consistency check: learned index vs index fitted to the residuals")
    for q in INDICES:
        learned = np.array([m["q_loss_final"] for m in results["metrics"][q, LEARNED_ARM]])
        fitted = np.array([m["q_from_residuals"] for m in results["metrics"][q, BASELINE_ARM]])
        sigma = np.median([m["q_residual_sigma"] for m in results["metrics"][q, BASELINE_ARM]])
        difference = abs(np.median(learned) - np.median(fitted))
        verdict = "consistent" if difference <= 3.0 * max(sigma, 1e-9) else "DIFFER"
        print(
            f"      equation q = {q}: learned {np.median(learned):.4f}"
            f"   fitted {np.median(fitted):.4f}   gap {difference:.4f}"
            f" ({difference / max(sigma, 1e-9):.1f} sigma)   {verdict}"
        )

    header = ("arm", "relative L2 (median [min, max])", "mass", "q_L final")
    print(f"\n    {header[0]:<22}{header[1]:>34}{header[2]:>12}{header[3]:>12}")
    for q in INDICES:
        for label in labels:
            metrics = results["metrics"][q, label]
            errors = np.array([m["l2_error"] for m in metrics])
            plain = f"q={q}, {label.replace('$', '').replace(chr(92), '')}"
            summary = f"{np.median(errors):.4f} [{errors.min():.4f}, {errors.max():.4f}]"
            print(
                f"    {plain:<22}{summary:>34}"
                f"{max(m['mass_error'] for m in metrics):>12.2e}"
                f"{np.median([m['q_loss_final'] for m in metrics]):>12.3f}"
            )

    print("\n    the mechanism: a robust loss tolerates large residuals, and in a")
    print("    forward problem the residual is what carries the initial condition inward")
    columns = ("arm", "peak p(.,0)", "mass at t=1", "RMS residual")
    print(f"      {columns[0]:<26}{columns[1]:>14}{columns[2]:>14}{columns[3]:>15}")
    for q in INDICES:
        exact_peak = float(qp.nlfp_density(0.0, 0.0, q, DIFFUSIVITY, BETA_INITIAL))
        for label in labels:
            metrics = results["metrics"][q, label]
            plain = f"q={q}, {label.replace('$', '').replace(chr(92), '')}"
            print(
                f"      {plain:<26}"
                f"{np.median([m['initial_peak'] for m in metrics]):>14.4f}"
                f"{np.median([m['final_mass'] for m in metrics]):>14.4f}"
                f"{np.median([m['rms_residual'] for m in metrics]):>15.4f}"
            )
        print(f"      {'  exact':<26}{exact_peak:>14.4f}{1.0:>14.4f}{0.0:>15.4f}")

    # Paired by seed, because the seed-to-seed spread is comparable to the
    # difference between arms: every arm at a given seed trained on the *same*
    # collocation points, so the per-seed ratio is the informative statistic and
    # the count of seeds that favour the deformed arm is the honest summary.
    print("\n    does the deformed residual likelihood help? (paired by seed)")
    print(f"      {'arm':<28}{'median ratio':>14}{'range':>18}{'seeds favouring':>18}")
    for q in INDICES:
        baseline = np.array([m["l2_error"] for m in results["metrics"][q, BASELINE_ARM]])
        for label in labels:
            if label == BASELINE_ARM:
                continue
            arm = np.array([m["l2_error"] for m in results["metrics"][q, label]])
            ratio = arm / baseline
            wins = int(np.sum(ratio < 1.0))
            verdict = "better" if np.median(ratio) < 1.0 else "worse"
            name = f"q = {q}, {label.replace('$', '')}"
            print(
                f"      {name:<28}{np.median(ratio):>13.2f}x"
                f"{f'[{ratio.min():.2f}, {ratio.max():.2f}]':>18}"
                f"{f'{wins}/{len(ratio)}':>18}   {verdict}"
            )


def main(*, quick: bool = False, full: bool = False) -> None:
    """Solve every arm and write the figure."""
    use_qjax_style()
    config = configuration(quick=quick, full=full)
    print(
        f"Heavy-tailed PINN residuals: hidden {config['hidden']},"
        f" {config['collocation']} collocation points, {config['steps']} steps,"
        f" {len(config['seeds'])} seeds"
    )
    results = run(config)
    report(results)
    plot_main(results, FIG_DIR / "pinn_fokker_planck")


if __name__ == "__main__":
    main(
        quick="--quick" in sys.argv,
        full="--full" in sys.argv or bool(os.environ.get("QJAX_FULL")),
    )
