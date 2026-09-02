r"""Generalized simulated annealing on Lennard-Jones clusters.

Generalized simulated annealing (Tsallis & Stariolo, *Physica A* **233**, 395,
1996) replaces the two Boltzmann ingredients of classical annealing by their
``q``-deformed counterparts, and in ``qjax`` each is a single call:

- the **visiting distribution** becomes a ``q``-Gaussian,
  ``qjax.sample(key, q=q_V, beta=1/T_V**2, ...)``, drawn *isotropically*: one unit
  direction in configuration space times one heavy-tailed radial step. For
  ``q_V > 1`` it has power-law tails, so the walk mixes occasional long Levy-like
  flights with local moves and can leave a basin without waiting for a thermally
  activated crossing. (Drawing all ``3n`` coordinates independently is a different
  and much worse proposal -- see `bounded_step`, which measures the difference.)
- the **acceptance** becomes ``qjax.q_exp(-dE/T_A, q_A)``. For ``q_A < 1`` this
  has compact support: uphill moves beyond ``T_A/(1-q_A)`` are rejected outright.
  That cut-off is the physics, not an underflow.

Setting ``q_V = q_A = 1`` recovers Kirkpatrick's Boltzmann machine exactly, and
``q_V = 2`` is Szu & Hartley's Cauchy machine, so the ``q = 1`` arm is a genuine
baseline rather than a strawman.

The schedule is where the library earns its place. The Tsallis cooling law

    T_q(t) = T_q(1) (2^{q-1} - 1) / ((1+t)^{q-1} - 1)

is ``0/0`` at ``q = 1`` -- the same pathology ``qjax.shared.series`` exists to
defeat. Substituting ``x^{q-1} - 1 = (q-1) ln_{2-q}(x)`` cancels both ``(q-1)``
factors and leaves a *ratio of two* ``q_log`` *calls*, which returns the
Geman-Geman logarithmic schedule at ``q = 1`` exactly, with no branch on ``q`` and
with a correct non-zero derivative in ``q``. See
``qjax.physics.annealing.tsallis_schedule``; the limit and the gradient are unit
tested in ``tests/test_physics_annealing.py``.

The benchmark is verifiable rather than illustrative. Lennard-Jones clusters have
tabulated global minima, and the three smallest are closed forms (a regular
simplex at ``r = 2**(1/6) sigma`` contributes ``-epsilon`` per pair, giving
``-1``, ``-3``, ``-6``). We report:

- **LJ7** (``-16.505384``), the fast sanity arm: every method solves it.
- **LJ13** (``-44.326801``, the Mackay icosahedron), the main comparison and the
  visiting-index scan: success rate against the number of function evaluations.
- **LJ38** (``-173.928427``, an fcc truncated octahedron), the paradigmatic
  double funnel. Its global minimum is *not* icosahedral, and the wide
  icosahedral basin around ``-173.252378`` is only 0.38 % higher in energy.
  Nothing in a budget of this size reliably finds the global minimum, so that
  panel is framed as a **funnel-escape rate** -- the final-energy distribution
  against *both* minima -- and the rates are reported as measured.

What the run measures is a **negative result with a mechanism**, and it is
reported as such. Deforming the visiting distribution does not accelerate global
optimization here. On LJ13 the classical Gaussian proposal has the *highest*
success rate of the four arms, and the Cauchy machine (``q_V = 2``) is several
standard errors worse; the visiting-index scan declines as the deformation grows.
The reason is not the width -- the Tsallis-Stariolo coupling
``sigma = T_V^{1/(3-q_V)}`` is used, and its exponent already shrinks the width
faster at large ``q_V`` -- but the **tail index**

    nu = (3 - q_V) / (q_V - 1),

which sets how often a proposal is catastrophic: a displacement exceeds ``k``
width units with probability ``~k^-nu``. At ``q_V = 2.7``, ``nu = 0.18``, and a
*majority* of proposals then exceed the container diameter however small the width
is (measured, and asserted in ``tests/test_examples_physics.py``), so the cluster
is scattered at every step and the cooling schedule never gets to act.

This is a statement about *this* parameterization, not a refutation of
Tsallis & Stariolo. Three things differ from their setup and any of them could
matter: the proposal is an isotropic radial ``q``-Gaussian rather than their exact
``D``-dimensional visiting distribution, the local minimizer is Adam rather than
a quasi-Newton method, and the cluster is confined to a ball. What the example
does establish is that the entropic index is not a free win on this landscape, and
that ``nu`` -- not ``q_V`` -- is the quantity to reason about when choosing it.

Two further measured details, both stated because they are easy to get wrong:

- The proposal must be **isotropic** -- one direction times one radial draw. Using
  ``3n`` independent ``q``-Gaussian coordinates gives ``3n`` chances per step for a
  catastrophic component, and costs LJ13 more than half its depth. See
  `bounded_step`.
- With a local quench in the loop (basin hopping), the visiting index matters much
  less: the quench, not the proposal, is then doing the exploring.


Run with: ``uv run python examples/generalized_annealing.py``
Add ``--full`` (or set ``QJAX_FULL=1``) for the larger configuration.
"""

from __future__ import annotations

import os
import sys
from functools import cache, partial
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import qjax
import qjax.physics as qp
from qjax.plots import CMAP, qcolors, qlinestyles, save_figure, use_qjax_style

FIG_DIR = Path(__file__).parent / "figures"

#: (label, q_visit, q_accept). The first two are the classical special cases.
ARMS = (
    ("CSA", "CSA ($q_V=1$, $q_A=1$)", 1.0, 1.0),
    ("GSA", "GSA ($q_V=1.5$, $q_A=1$)", 1.5, 1.0),
    ("FSA", "FSA ($q_V=2$, $q_A=1$)", 2.0, 1.0),
    ("GSA-greedy", "GSA ($q_V=1.5$, $q_A=0.6$)", 1.5, 0.6),
)

#: Visiting indices for the scan in panel (e). Reported against the tail index
#: ``nu = (3-q_V)/(q_V-1)`` as well as against ``q_V``, because ``nu`` is what
#: actually controls how often a proposal destroys the cluster.
VISITING_SCAN = (1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.3, 2.7)

#: Acceptance indices for the companion scan. Below 1 the acceptance has compact
#: support -- uphill moves beyond ``T_A/(1-q_A)`` are rejected outright -- and
#: above 1 it has power-law tails and tolerates them more readily than Boltzmann.
ACCEPTANCE_SCAN = (0.4, 0.6, 0.8, 1.0, 1.3, 1.6, 2.0, 2.4)

#: Soft spherical wall radius per cluster size, in units of sigma. Comfortably
#: larger than the cluster, so the wall stays inert -- which `main` asserts.
CONTAINER = {7: 2.8, 13: 3.4, 38: 4.8}
INITIAL_VISITING = 0.9  # T_V(1), in sigma: the proposal scale at the first step
INITIAL_ACCEPTANCE = 1.2  # T_A(1), in epsilon
QUENCH_RATE = 0.02

#: Neighbour cutoff for drawing bonds, in units of sigma: between the first and
#: second neighbour shells of a close-packed cluster.
BOND_CUTOFF = 1.35
#: Depth of the final polish applied to the best configuration of each restart.
#: Basin hopping reports quenched energies, and the shallow in-loop quench is a
#: basin *selector*, not a converged minimizer: 300 Adam steps at this rate reach
#: the tabulated minima to better than 1e-4, which 30 steps do not come close to.
POLISH_STEPS, POLISH_RATE = 300, 0.02

#: Success on the *polished* energy. The tabulated references are quoted to six
#: decimals and float32 can land a few 1e-5 below them, so this also sets the
#: margin used to check that no arm reports a physically impossible energy.
SUCCESS_TOLERANCE = 1e-2

#: Success on the *unpolished* best-so-far trace, used for the rate-vs-budget
#: curve. It asks the weaker question "has the walk found the right basin yet",
#: and 0.5 epsilon answers it unambiguously: the next isomer above the global
#: minimum sits 0.57 higher for LJ7, 2.86 for LJ13, and 0.68 for LJ38 (the
#: icosahedral funnel), so no other basin can be mistaken for the right one.
BASIN_TOLERANCE = 0.5


def configuration(*, quick: bool, full: bool) -> dict:
    """Trial counts, step budgets and quench depths for the requested tier."""
    if quick:
        return {
            "sanity": {"atoms": 7, "trials": 8, "steps": 200, "quench": 20},
            "scan": {"atoms": 13, "trials": 8, "steps": 200, "quench": 20},
            "main": {"atoms": 13, "trials": 8, "steps": 300, "quench": 20},
            "funnel": {"atoms": 38, "trials": 4, "steps": 200, "quench": 20},
        }
    if full:
        return {
            "sanity": {"atoms": 7, "trials": 128, "steps": 4000, "quench": 40},
            "scan": {"atoms": 13, "trials": 512, "steps": 8000, "quench": 40},
            "main": {"atoms": 13, "trials": 256, "steps": 12000, "quench": 40},
            "funnel": {"atoms": 38, "trials": 256, "steps": 8000, "quench": 60},
        }
    return {
        "sanity": {"atoms": 7, "trials": 32, "steps": 1500, "quench": 30},
        "scan": {"atoms": 13, "trials": 96, "steps": 2500, "quench": 30},
        "main": {"atoms": 13, "trials": 64, "steps": 4000, "quench": 30},
        "funnel": {"atoms": 38, "trials": 24, "steps": 2500, "quench": 40},
    }


# --------------------------------------------------------------------------- #
# The GSA engine
# --------------------------------------------------------------------------- #
def bounded_step(key, q_visit, visiting, num_atoms: int, radius):
    r"""An *isotropic* ``q``-Gaussian displacement, capped at the container size.

    One unit direction in the full ``3n``-dimensional configuration space times one
    heavy-tailed radial step. Drawing each of the ``3n`` coordinates independently
    from a ``q``-Gaussian instead looks equivalent and is not: at ``q_V = 2.7``
    the tail index is ``nu = (3-q)/(q-1) = 0.18``, so a single coordinate exceeds
    100 scale units with probability ``100**-0.18 ~ 0.09``, and across 39
    coordinates *something* blows up on essentially every step. The cluster is
    then scattered across the container at every proposal and the annealing
    schedule never gets to act -- measured, that costs LJ13 more than half its
    depth. One radial draw per step gives the intended behaviour: mostly local
    moves with occasional long Levy-like flights, which the cooling schedule then
    shortens.

    The cap is exact rather than a fudge: the walk is confined to a ball of radius
    ``R``, so a per-atom displacement beyond ``2R`` lands outside it from any
    starting point and is projected back to the same place as a ``2R`` step. It is
    also necessary, because `qjax.sample` builds the Student-$t$ from a
    ``chi2_nu`` variate and in float32 a shape-0.09 gamma draw underflows to
    exactly zero often enough (about 1 in 2000 at ``q_V = 2.7``) that the division
    returns ``+inf``. That infinity is finite precision, not a real value, but left
    alone it turns the configuration into NaN at the first such draw.

    Args:
        key: PRNG key.
        q_visit: Visiting index ``q_V``.
        visiting: Visiting temperature ``T_V(t)``; the radial scale is
            ``T_V^{1/(3-q_V)}``.
        num_atoms: Cluster size ``n``.
        radius: Container radius ``R``.

    Returns:
        A displacement of shape ``(num_atoms, 3)``.
    """
    direction_key, magnitude_key = jax.random.split(key)
    direction = jax.random.normal(direction_key, (num_atoms, 3))
    direction /= jnp.linalg.norm(direction) + 1e-12

    # Tsallis & Stariolo tie the proposal width to T_V^{1/(3-q_V)}, not to T_V.
    # The exponent matters: at q_V = 2.7 it is 3.33, so the width collapses far
    # faster than the temperature and partly offsets the heavy tail.
    width = visiting ** (1.0 / (3.0 - q_visit))
    magnitude = jnp.abs(qjax.sample(magnitude_key, q=q_visit, beta=1.0 / width**2, shape=()))
    limit = 2.0 * radius * jnp.sqrt(float(num_atoms))
    magnitude = jnp.nan_to_num(magnitude, nan=0.0, posinf=limit)
    return direction * jnp.minimum(magnitude, limit)


@partial(jax.jit, static_argnames=("num_atoms", "steps", "quench"))
def gsa_trial(key, num_atoms: int, q_visit, q_accept, steps: int, quench: int, radius):
    r"""One generalized-simulated-annealing run; returns the best-so-far trace.

    Each step proposes ``x + qGaussian(q_V, T_V(t))``, optionally quenches the
    proposal to the bottom of its basin (Wales-Doye basin hopping, so the walk
    explores the graph of local minima rather than the raw landscape), and accepts
    with probability ``min(1, exp_{q_A}(-dE / T_A(t)))``.

    The configuration is re-centred on its centre of mass after every quench.
    Translation is a zero mode of the bare potential, so without re-centring the
    cluster drifts into the confining wall and the reported energy would silently
    include a wall term.

    Args:
        key: PRNG key.
        num_atoms: Cluster size ``n``.
        q_visit: Visiting index ``q_V`` (traced: do **not** ``vmap`` over it, see
            the note in ``sweep``).
        q_accept: Acceptance index ``q_A``.
        steps: Annealing steps.
        quench: Local-minimization steps per proposal; ``0`` disables quenching.
        radius: Soft-wall radius.

    Returns:
        ``(best_positions, polished_energy, best_energy_trace)``. The polished
        energy is the run's result; the trace is the unpolished best-so-far and
        drives the rate-vs-budget curve.
    """
    start_key, walk_key = jax.random.split(key)
    positions = qp.lj_random_cluster(start_key, num_atoms, radius * 0.55)

    def energy(x):
        return qp.lj_energy_confined(x, radius)

    def confine(x):
        """Project every atom back inside the container ball.

        Not a numerical guard bolted on: generalized simulated annealing is
        defined on a *bounded* domain, and the container is that domain. It also
        has to be enforced rather than assumed, because a q-Gaussian proposal at
        ``q_V = 2.7`` has tail index ``nu = (3-q)/(q-1) = 0.18`` -- so heavy that
        single draws reach ``1e20`` and the squared separations then overflow
        float32 to ``inf``, poisoning the quench gradient with NaN.
        """
        distance = jnp.linalg.norm(x, axis=-1, keepdims=True)
        scale = jnp.minimum(1.0, radius / jnp.maximum(distance, 1e-12))
        return x * scale

    def refine(x):
        x = confine(x)
        if quench:
            x, _ = qp.lj_quench(x, quench, QUENCH_RATE)
        return confine(x - jnp.mean(x, axis=0, keepdims=True))

    positions = refine(positions)
    current = energy(positions)

    def step(carry, t):
        chain_key, state, state_energy, best, best_energy = carry
        chain_key, proposal_key, accept_key = jax.random.split(chain_key, 3)

        visiting = qp.visiting_temperature(t, INITIAL_VISITING, q_visit)
        accepting = qp.acceptance_temperature(t, INITIAL_ACCEPTANCE, q_accept)

        jump = bounded_step(proposal_key, q_visit, visiting, num_atoms, radius)
        proposal = refine(state + jump)
        proposed_energy = energy(proposal)

        delta = proposed_energy - state_energy
        probability = jnp.minimum(qjax.q_exp(-delta / accepting, q_accept), 1.0)
        accepted = (delta <= 0.0) | (jax.random.uniform(accept_key) < probability)

        state = jnp.where(accepted, proposal, state)
        state_energy = jnp.where(accepted, proposed_energy, state_energy)
        improved = proposed_energy < best_energy
        best = jnp.where(improved, proposal, best)
        best_energy = jnp.minimum(proposed_energy, best_energy)
        return (chain_key, state, state_energy, best, best_energy), best_energy

    initial = (walk_key, positions, current, positions, current)
    (_, _, _, best, _), trace = jax.lax.scan(
        step, initial, jnp.arange(1, steps + 1, dtype=jnp.result_type(float))
    )
    polished, polish_trace = qp.lj_quench(best, POLISH_STEPS, POLISH_RATE)
    # Report the bare potential, not the confined one, and assert (in `main`) that
    # the polished cluster really is inside the container so the wall cannot have
    # contributed to it.
    return polished, jnp.min(polish_trace), trace


@cache
def _batched_trial(num_atoms: int, steps: int, quench: int):
    """Compile the vmapped run once per shape triple, and cache it.

    Without the cache this is the single biggest cost in the script. ``q_visit``
    and ``q_accept`` are *traced* arguments, so one compilation serves every point
    of the ``(q_V, q_A)`` grid -- but only if JAX is handed the same callable each
    time. Wrapping the run in a fresh ``lambda`` per call, which is the obvious way
    to write it, makes every one of the 49 grid cells a new traced function and
    recompiles a 1500-step scan for each.
    """

    def run(keys, q_visit, q_accept, radius):
        return jax.vmap(gsa_trial, in_axes=(0, None, None, None, None, None, None))(
            keys, num_atoms, q_visit, q_accept, steps, quench, radius
        )

    return jax.jit(run)


def sweep(key, settings: dict, q_visit: float, q_accept: float):
    """Run every restart of one arm in parallel; returns positions, energies, traces.

    ``vmap`` covers the *trials* only. It must not cover ``q_visit``:
    ``qjax.core.distributions.sample`` dispatches on ``lax.cond(near_one(q), ...)``
    and under ``vmap`` with a batched ``q`` that lowers to ``lax.select``, so both
    branches execute -- and the Student-t branch computes ``nu = (3-q)/(q-1)``,
    which is ``inf`` at ``q_V = 1``. The primal survives the select but a gradient
    would not, so the arms are looped in Python instead.
    """
    keys = jax.random.split(key, settings["trials"])
    radius = CONTAINER[settings["atoms"]]
    batched = _batched_trial(settings["atoms"], settings["steps"], settings["quench"])
    return batched(keys, jnp.asarray(q_visit), jnp.asarray(q_accept), jnp.asarray(radius))


def evaluations(settings: dict) -> np.ndarray:
    """Cumulative energy-and-gradient evaluations after each annealing step."""
    per_step = settings["quench"] + 1
    return np.arange(1, settings["steps"] + 1) * per_step


def success_rate(best: np.ndarray, reference: float) -> tuple[float, float]:
    """Fraction of restarts within `SUCCESS_TOLERANCE` of the reference, with its error."""
    hits = np.asarray(best <= reference + SUCCESS_TOLERANCE, dtype=float)
    rate = float(hits.mean())
    return rate, float(np.sqrt(max(rate * (1.0 - rate), 1e-12) / hits.size))


# --------------------------------------------------------------------------- #
# Experiments
# --------------------------------------------------------------------------- #
def run_benchmark(settings: dict, key_offset: int) -> dict:
    """Run all four arms on one cluster size."""
    atoms = settings["atoms"]
    reference = qp.LJ_REFERENCE_MINIMA[atoms]
    results: dict = {
        "atoms": atoms,
        "reference": reference,
        "evaluations": evaluations(settings),
        "traces": {},
        "best": {},
        "positions": {},
    }
    print(f"  LJ{atoms} (reference {reference:.6f}), {settings['trials']} restarts")
    for index, (_, label, q_visit, q_accept) in enumerate(ARMS):
        positions, polished, traces = sweep(
            jax.random.PRNGKey(key_offset + index), settings, q_visit, q_accept
        )
        traces = np.asarray(traces)
        best = np.asarray(polished)
        results["traces"][label] = traces
        results["best"][label] = best
        results["positions"][label] = np.asarray(positions[int(np.argmin(best))])
        rate, spread = success_rate(best, reference)
        print(
            f"    {label:<28} best {best.min():.6f}   median {np.median(best):.4f}"
            f"   success {100 * rate:5.1f} +- {100 * spread:4.1f} %"
        )
    return results


def run_scan(settings: dict, axis: str) -> dict:
    r"""Success rate along one entropic index, holding the other classical.

    Two 1-D scans at high restart counts, rather than one 2-D grid at a low one.
    The grid is the tempting figure and the wrong one: at any trial count this
    script can afford, a ``(q_V, q_A)`` cell carries a binomial error of order 12
    percentage points -- larger than any effect being looked for -- so it renders
    as a blocky pattern that invites over-reading. Scanning each index separately
    at six times the restarts answers the question the four fixed arms only sample
    at two points, with error bars small enough to mean something.

    The visiting sweep is also reported against the tail index
    ``nu = (3-q_V)/(q_V-1)``, which is the quantity with operational meaning: a
    proposal exceeds ``k`` width units with probability ``~k**-nu``, so
    ``nu -> inf`` (``q_V -> 1``) is a purely local Gaussian walk and ``nu -> 0``
    (``q_V -> 3``) scatters the cluster on a finite fraction of every step,
    whatever the width.

    Args:
        settings: Tier settings for this scan.
        axis: ``"visiting"`` to sweep ``q_V`` at ``q_A = 1``, or ``"acceptance"``
            to sweep ``q_A`` at ``q_V = 1``.

    Returns:
        A dict with ``indices``, ``rates``, ``spreads``, ``axis`` and ``atoms``.
    """
    reference = qp.LJ_REFERENCE_MINIMA[settings["atoms"]]
    indices = VISITING_SCAN if axis == "visiting" else ACCEPTANCE_SCAN
    rates, spreads = [], []
    swept = "q_V at q_A = 1" if axis == "visiting" else "q_A at q_V = 1"
    print(f"  {axis} scan on LJ{settings['atoms']}, sweeping {swept}")
    for index, value in enumerate(indices):
        q_visit, q_accept = (value, 1.0) if axis == "visiting" else (1.0, value)
        _, polished, _ = sweep(
            jax.random.PRNGKey(9000 + index), settings, float(q_visit), float(q_accept)
        )
        rate, spread = success_rate(np.asarray(polished), reference)
        rates.append(rate)
        spreads.append(spread)
        extra = ""
        if axis == "visiting":
            tail = (3.0 - value) / (value - 1.0) if value > 1.0 else float("inf")
            extra = f" (tail index nu = {tail:6.2f})"
        print(
            f"    {value:.1f}{extra}   success {100 * rate:5.1f} +- {100 * spread:4.1f} %",
            flush=True,
        )
    return {
        "indices": np.asarray(indices),
        "rates": np.asarray(rates),
        "spreads": np.asarray(spreads),
        "axis": axis,
        "atoms": settings["atoms"],
    }


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def plot_main(sanity: dict, main: dict, funnel: dict, scan_result: dict, path: Path) -> None:
    """Schedules, proposal tails, the acceptance cut-off, and the measured optimum."""
    colors = qcolors(len(ARMS))
    styles = qlinestyles(len(ARMS))
    labels = [label for _, label, _, _ in ARMS]

    fig, axes = plt.subplots(2, 3, figsize=(13.4, 7.6))
    fig.subplots_adjust(hspace=0.36, wspace=0.29)

    # (a) The Tsallis schedule, and its branch-free q -> 1 limit.
    ax = axes[0, 0]
    steps = np.logspace(0.0, 4.0, 200)
    distinct = sorted({q_visit for _, _, q_visit, _ in ARMS} | {2.7})
    for color, style, q_visit in zip(
        qcolors(len(distinct)), qlinestyles(len(distinct)), distinct, strict=True
    ):
        ax.loglog(
            steps,
            np.asarray(qp.visiting_temperature(jnp.asarray(steps), INITIAL_VISITING, q_visit)),
            color=color,
            ls=style,
            lw=1.4,
            label=rf"$q_V={q_visit}$",
        )
    ax.loglog(
        steps,
        INITIAL_VISITING * np.log(2.0) / np.log1p(steps),
        color="0.25",
        lw=2.4,
        alpha=0.35,
        label=r"Geman-Geman $\ln$",
    )
    ax.set_xlabel("annealing step $t$")
    ax.set_ylabel(r"$T_V(t)$")
    ax.set_title(r"(a) $T_q(t) = T_q(1)\,\ln_{2-q}2 / \ln_{2-q}(1+t)$", fontsize=9.5)
    ax.legend(fontsize=7.6)

    # (b) The proposal tails, validating qjax.sample against its own density.
    ax = axes[0, 1]
    edges = np.linspace(-6.0, 6.0, 121)
    centres = 0.5 * (edges[:-1] + edges[1:])
    for color, q_visit in zip(qcolors(3), (1.0, 2.0, 2.7), strict=True):
        draws = qjax.sample(jax.random.PRNGKey(3), q=q_visit, beta=1.0, shape=(200_000,))
        # histogram_density normalizes over the samples that land inside the range.
        # At q_V = 2.7 most of them do not, so the core would be inflated relative
        # to the true density unless it is scaled back by the mass retained.
        inside = float(jnp.mean(jnp.abs(draws) < edges[-1]))
        density = inside * np.asarray(qp.histogram_density(draws, jnp.asarray(edges)))
        ax.semilogy(
            centres, np.maximum(density, 1e-6), color=color, lw=1.1, label=rf"$q_V={q_visit}$"
        )
        ax.semilogy(
            centres,
            np.asarray(qjax.q_gaussian_pdf(jnp.asarray(centres), q=q_visit, beta=1.0)),
            color=color,
            lw=2.6,
            alpha=0.3,
        )
    ax.set_ylim(1e-5, 1.2)
    ax.set_xlabel("proposed displacement")
    ax.set_ylabel("density")
    ax.set_title(r"(b) $q$-Gaussian proposals (thin: samples, thick: pdf)", fontsize=9.5)
    ax.legend(fontsize=7.6)

    # (c) The acceptance rule, with the q_A < 1 cut-off marked.
    ax = axes[0, 2]
    delta = np.linspace(0.0, 4.0, 400)
    for color, q_accept in zip(qcolors(4), (0.5, 1.0, 1.6, 2.2), strict=True):
        curve = np.asarray(qjax.q_exp(-jnp.asarray(delta) / 1.0, q_accept))
        ax.plot(delta, np.minimum(curve, 1.0), color=color, lw=1.5, label=rf"$q_A={q_accept}$")
        if q_accept < 1.0:
            cut = 1.0 / (1.0 - q_accept)
            ax.axvline(cut, color=color, lw=0.9, ls=":")
            ax.annotate(
                rf"cut-off $T_A/(1-q_A)={cut:.0f}$",
                (cut, 0.55),
                fontsize=7,
                rotation=90,
                ha="right",
                va="center",
                color=color,
            )
    ax.set_xlabel(r"uphill cost $\Delta E$   (at $T_A = 1$)")
    ax.set_ylabel("acceptance probability")
    ax.set_title(r"(c) $\min(1, \exp_{q_A}(-\Delta E/T_A))$", fontsize=9.5)
    ax.legend(fontsize=7.6)

    # (d) LJ13: best-so-far energy against function evaluations.
    ax = axes[1, 0]
    budget = main["evaluations"]
    for color, style, label in zip(colors, styles, labels, strict=True):
        traces = main["traces"][label]
        median = np.median(traces, axis=0)
        low, high = np.percentile(traces, (25, 75), axis=0)
        ax.plot(budget, median, color=color, ls=style, lw=1.4, label=label)
        ax.fill_between(budget, low, high, color=color, alpha=0.16, lw=0)
    ax.axhline(
        main["reference"], color="0.2", lw=1.2, ls="--", label=f"exact {main['reference']:.4f}"
    )
    ax.set_xscale("log")
    ax.set_xlabel("function evaluations")
    ax.set_ylabel("best energy so far")
    ax.set_title(rf"(d) LJ$_{{{main['atoms']}}}$: median and IQR", fontsize=9.5)
    ax.legend(fontsize=7.2, loc="upper right")

    # (e) The visiting-index scan: an interior optimum at moderate deformation.
    ax = axes[1, 1]
    scan = scan_result
    ax.errorbar(
        scan["indices"],
        100 * scan["rates"],
        yerr=100 * scan["spreads"],
        color=colors[2],
        lw=1.5,
        marker="o",
        ms=5,
        capsize=2,
    )
    best = int(np.argmax(scan["rates"]))
    ax.axvline(scan["indices"][best], color=colors[2], lw=0.9, ls=":")
    ax.axvline(1.0, color="0.3", lw=1.0, ls="--")
    ax.annotate(
        "CSA",
        (1.0, ax.get_ylim()[1]),
        fontsize=7.5,
        color="0.3",
        ha="left",
        va="top",
        xytext=(3, -2),
        textcoords="offset points",
    )
    ax.set_xlabel(r"visiting index $q_V$   (at $q_A = 1$)")
    ax.set_ylabel("success rate (%)")
    ax.set_title(
        rf"(e) LJ$_{{{scan['atoms']}}}$: no advantage, worst at the smallest $\nu$",
        fontsize=9.5,
    )
    # The tail index is the quantity with operational meaning, but it diverges at
    # q_V = 1, so it is annotated per point rather than put on a second axis.
    for q_visit, rate in zip(scan["indices"], scan["rates"], strict=True):
        tail = r"$\infty$" if q_visit <= 1.0 else f"{(3.0 - q_visit) / (q_visit - 1.0):.1f}"
        ax.annotate(
            tail,
            (q_visit, 100 * rate),
            fontsize=6.4,
            color="0.35",
            ha="center",
            va="bottom",
            xytext=(0, 7),
            textcoords="offset points",
        )
    ax.annotate(
        r"labels: tail index $\nu = (3-q_V)/(q_V-1)$",
        (0.98, 0.04),
        xycoords="axes fraction",
        fontsize=6.8,
        color="0.35",
        ha="right",
    )

    # (f) LJ38: where each arm ends up, relative to the two funnel minima.
    ax = axes[1, 2]
    reference = funnel["reference"]
    everything = np.concatenate([funnel["best"][label] for label in labels])
    edges = np.linspace(min(everything.min(), reference) - 0.4, np.percentile(everything, 96), 26)
    for color, style, label in zip(colors, styles, labels, strict=True):
        counts, _ = np.histogram(funnel["best"][label], bins=edges)
        ax.step(
            edges[:-1],
            100 * counts / len(funnel["best"][label]),
            where="post",
            color=color,
            ls=style,
            lw=1.4,
            label=label,
        )
    ax.axvline(reference, color="0.2", lw=1.3, ls="--")
    ax.axvline(qp.LJ38_ICOSAHEDRAL, color="0.5", lw=1.1, ls=":")
    for value, name, colour, offset in (
        (reference, "fcc global", "0.2", 0.06),
        (qp.LJ38_ICOSAHEDRAL, "icosahedral", "0.45", 0.24),
    ):
        ax.annotate(
            f"{name}\n{value:.3f}",
            (value, offset),
            xycoords=ax.get_xaxis_transform(),
            fontsize=6.6,
            ha="left",
            va="bottom",
            color=colour,
            rotation=0,
        )
    ax.set_xlabel("final energy")
    ax.set_ylabel("restarts (%)")
    ax.set_title(rf"(f) LJ$_{{{funnel['atoms']}}}$ double funnel", fontsize=9.5)
    ax.legend(fontsize=6.6, loc="upper left")

    save_figure(fig, path)
    plt.close(fig)


def plot_landscape(acceptance: dict, funnel: dict, path: Path) -> None:
    """The acceptance-index scan, and the best cluster actually found."""
    fig = plt.figure(figsize=(11.6, 4.6))
    grid = fig.add_gridspec(1, 2, width_ratios=(1.1, 1.0), wspace=0.20)

    ax = fig.add_subplot(grid[0, 0])
    # The compact-support side is qualitatively different, not merely smaller: for
    # q_A < 1 an uphill move beyond T_A/(1-q_A) is rejected outright.
    ax.axvspan(float(acceptance["indices"][0]) - 0.05, 1.0, color=qcolors(6)[0], alpha=0.22, lw=0)
    ax.errorbar(
        acceptance["indices"],
        100 * acceptance["rates"],
        yerr=100 * acceptance["spreads"],
        color=qcolors(4)[3],
        lw=1.5,
        marker="o",
        ms=5,
        capsize=2,
    )
    ax.axvline(1.0, color="0.3", lw=1.0, ls="--")
    ax.set_ylim(bottom=0.0)
    ax.annotate(
        "Boltzmann",
        (1.0, 0.97),
        xycoords=("data", "axes fraction"),
        fontsize=7.5,
        color="0.3",
        ha="left",
        va="top",
        xytext=(4, 0),
        textcoords="offset points",
    )
    ax.annotate(
        "compact support:\nuphill moves capped",
        (0.5, 0.06),
        xycoords=("data", "axes fraction"),
        fontsize=6.8,
        color="0.35",
        ha="center",
    )
    ax.set_xlabel(r"acceptance index $q_A$   (at $q_V = 1$)")
    ax.set_ylabel("success rate (%)")
    ax.set_title(rf"(a) LJ$_{{{acceptance['atoms']}}}$: the acceptance deformation", fontsize=10)

    ax = fig.add_subplot(grid[0, 1], projection="3d")
    labels = [label for _, label, _, _ in ARMS]
    winner = min(labels, key=lambda label: funnel["best"][label].min())
    positions = funnel["positions"][winner]
    coordination = np.asarray(qp.coordination_numbers(jnp.asarray(positions)))

    # Draw the bonds first, behind the atoms. Without them a projected point cloud
    # reads as a flat scatter; with them the shell structure is legible, which is
    # the whole reason for showing a cluster rather than a number.
    separations = np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=-1)
    first, second = np.triu_indices(positions.shape[0], k=1)
    bonded = separations[first, second] < BOND_CUTOFF
    for i, j in zip(first[bonded], second[bonded], strict=True):
        ax.plot(
            positions[[i, j], 0],
            positions[[i, j], 1],
            positions[[i, j], 2],
            color="0.55",
            lw=0.7,
            alpha=0.55,
            zorder=1,
        )
    scatter = ax.scatter(
        positions[:, 0],
        positions[:, 1],
        positions[:, 2],
        c=coordination,
        cmap=CMAP,
        s=110,
        depthshade=True,
        edgecolor="0.2",
        linewidth=0.5,
        zorder=3,
    )
    span = float(np.max(np.abs(positions))) * 1.1
    for setter in (ax.set_xlim, ax.set_ylim, ax.set_zlim):
        setter(-span, span)
    # Equal box aspect, or a roughly spherical cluster renders as a pancake.
    ax.set_box_aspect((1.0, 1.0, 1.0))
    ax.view_init(elev=22.0, azim=-62.0)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.grid(visible=False)
    ax.set_title(
        rf"(b) best LJ$_{{{funnel['atoms']}}}$ found: "
        rf"$E = {funnel['best'][winner].min():.4f}$"
        "\n"
        rf"({winner}; colour = coordination number)",
        fontsize=9,
    )
    fig.colorbar(scatter, ax=ax, fraction=0.03, pad=0.02).ax.tick_params(labelsize=7)

    save_figure(fig, path)
    plt.close(fig)


# --------------------------------------------------------------------------- #
def report(sanity: dict, main: dict, funnel: dict, scan_result: dict, acceptance: dict) -> None:
    """Print the validation table the docs page reproduces."""
    labels = [label for _, label, _, _ in ARMS]
    print("\n  validation")
    print(f"    {'quantity':<44}{'measured':>14}{'reference':>14}")
    for num_atoms in (2, 3, 4):
        positions = qp.equidistant_cluster(num_atoms, qp.LJ_PAIR_DISTANCE)
        print(
            f"    {f'LJ{num_atoms} closed form (regular simplex)':<44}"
            f"{float(qp.lj_energy(positions)):>14.9f}"
            f"{qp.LJ_REFERENCE_MINIMA[num_atoms]:>14.9f}"
        )
    for source in (sanity, main, funnel):
        best = min(source["best"][label].min() for label in labels)
        name = f"LJ{source['atoms']} best found over all arms"
        print(f"    {name:<44}{best:>14.6f}{source['reference']:>14.6f}")

    print("\n    no arm reports an energy below its reference (a physical impossibility):")
    for source in (sanity, main, funnel):
        floor = min(source["best"][label].min() for label in labels)
        margin = floor - source["reference"]
        verdict = "ok" if margin > -SUCCESS_TOLERANCE else "IMPOSSIBLE"
        print(f"      LJ{source['atoms']:<3} margin above the reference: {margin:+.6f}  {verdict}")

    print("\n    success rates within 1e-2 of the reference")
    for source in (sanity, main, funnel):
        rates = "  ".join(
            f"{short}: {100 * success_rate(source['best'][label], source['reference'])[0]:5.1f}%"
            for short, label, _, _ in ARMS
        )
        print(f"      LJ{source['atoms']:<3} {rates}")

    def tail_index(q_visit: float) -> float:
        """``nu = (3-q)/(q-1)``, which diverges at the classical point."""
        return (3.0 - q_visit) / (q_visit - 1.0) if q_visit > 1.0 else float("inf")

    best = int(np.argmax(scan_result["rates"]))
    optimum = float(scan_result["indices"][best])
    tail = tail_index(optimum)
    classical = 100 * scan_result["rates"][0]
    print(f"\n    visiting-index scan on LJ{scan_result['atoms']} (q_A = 1)")
    print(f"      classical q_V = 1: {classical:.1f} +- {100 * scan_result['spreads'][0]:.1f} %")
    print(
        f"      best in the scan:  q_V = {optimum:.1f} (nu = {tail:.2f}),"
        f" {100 * scan_result['rates'][best]:.1f}"
        f" +- {100 * scan_result['spreads'][best]:.1f} %"
    )
    worst = int(np.argmin(scan_result["rates"]))
    worst_q = float(scan_result["indices"][worst])
    worst_tail = tail_index(worst_q)
    print(
        f"      worst in the scan: q_V = {worst_q:.1f} (nu = {worst_tail:.2f}),"
        f" {100 * scan_result['rates'][worst]:.1f}"
        f" +- {100 * scan_result['spreads'][worst]:.1f} %"
    )

    print(f"\n    acceptance-index scan on LJ{acceptance['atoms']} (q_V = 1)")
    boltzmann = int(np.argmin(np.abs(acceptance["indices"] - 1.0)))
    print(
        f"      Boltzmann q_A = 1: {100 * acceptance['rates'][boltzmann]:.1f}"
        f" +- {100 * acceptance['spreads'][boltzmann]:.1f} %"
    )
    for name, index in (
        ("compact support (q_A < 1)", acceptance["indices"] < 1.0),
        ("heavy tailed  (q_A > 1)", acceptance["indices"] > 1.0),
    ):
        side = acceptance["rates"][index]
        print(f"      {name}: {100 * side.mean():.1f} % on average over {side.size} points")

    print("\n    LJ38 funnel occupancy (fraction of restarts at or below each minimum)")
    for label in labels:
        best = funnel["best"][label]
        fcc = float(np.mean(best <= funnel["reference"] + SUCCESS_TOLERANCE))
        icosahedral = float(np.mean(best <= qp.LJ38_ICOSAHEDRAL + SUCCESS_TOLERANCE))
        print(f"      {label:<28} fcc {100 * fcc:5.1f}%   icosahedral+ {100 * icosahedral:5.1f}%")


def main(*, quick: bool = False, full: bool = False) -> None:
    """Run all three benchmarks plus the sweep, and write both figures."""
    use_qjax_style()
    config = configuration(quick=quick, full=full)
    print("Generalized simulated annealing on Lennard-Jones clusters")
    sanity = run_benchmark(config["sanity"], 1000)
    main_result = run_benchmark(config["main"], 2000)
    funnel = run_benchmark(config["funnel"], 3000)
    visiting = run_scan(config["scan"], "visiting")
    acceptance = run_scan(config["scan"], "acceptance")
    report(sanity, main_result, funnel, visiting, acceptance)
    plot_main(sanity, main_result, funnel, visiting, FIG_DIR / "generalized_annealing")
    plot_landscape(acceptance, funnel, FIG_DIR / "generalized_annealing_landscape")


if __name__ == "__main__":
    main(
        quick="--quick" in sys.argv,
        full="--full" in sys.argv or bool(os.environ.get("QJAX_FULL")),
    )
