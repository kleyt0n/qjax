r"""Measuring the entropic index from trajectories: anomalous diffusion and cold atoms.

The other examples treat ``q`` as something to choose or to learn. Here it is a
**physical quantity to be measured**, with a value predicted in advance and two
independent ways to estimate it from data.

Nonextensive statistics makes a falsifiable prediction about anomalous diffusion.
The nonlinear (porous-medium) Fokker-Planck equation

    dp/dt = D d^2 p^{2-q} / dx^2

has the self-similar ``q``-Gaussian solution of Tsallis & Bukman (1996), whose
width obeys ``<x^2> ~ t^alpha`` with

    alpha = 2 / (3 - q).

So the shape of the distribution and the growth of its width are not two free
parameters: either one predicts the other. Fitting ``q`` to the density and
reading it off the mean-squared displacement are therefore *independent*
measurements of the same number, and they can disagree.

A second, experimentally realized mechanism gives the same distribution from a
*linear*-noise Langevin equation with saturating (Sisyphus) friction,

    dp = -alpha p / (1 + (p/p_c)^2) dt + sqrt(2 D_0) dW,

whose exact stationary solution is a ``q``-Gaussian with

    q = 1 + 2 D_0 / (alpha p_c^2),   beta = alpha / (2 D_0).

Because the three coefficients are ours to choose, this is a *rigorous internal
reference*: the true ``q`` is known to machine precision. Separately, Lutz (2003)
evaluated those coefficients semiclassically for atoms in a dissipative optical
lattice and obtained ``q = 1 + 44 E_R / U_0``, confirmed experimentally by
Douglas, Bergamini & Renzoni (2006). The two are kept distinct throughout: the
first is exact for *our simulation*, the second is a prediction about a *real
experiment*.

Four arms, sharing particle count, step count, ``dt``, snapshot times, PRNG
stream and estimator settings; only the generating dynamics differ:

1. **Brownian control**, ``q = 1``. Additive noise, linear friction. Must return
   ``alpha = 1.00`` -- it is both the Boltzmann-Gibbs baseline and a null control on
   the estimators themselves. Its density fit approaches ``q = 1`` from above
   without attaining it (see `fit_q_beta`), so there ``q_hat`` reads as an upper
   bound.
2. **Superdiffusive NLFP**, ``q = 1.5``, hence ``alpha = 4/3`` exactly.
3. **Subdiffusive NLFP**, ``q = 0.5``, hence ``alpha = 0.8`` exactly.
4. **Cold-atom Langevin** at three values of ``E_R / U_0``.

One subtlety is stated rather than hidden. For ``q < 1`` the ``q``-Gaussian has
*compact support*, so ``q_gaussian_logpdf`` is ``-inf`` outside it and the
negative log-likelihood is ``+inf``: a gradient-based fit started at ``q > 1`` can
never cross into ``q < 1``. The maximum-likelihood arms are therefore bounded to
``q > 1``, and the subdiffusive arm's index is measured through the scaling
relation ``q = 3 - 2/alpha`` instead. That is a real constraint of compact-support
likelihoods, not a limitation of this script.

``examples/pinn_fokker_planck.py`` attacks the same equation from the opposite
direction -- solving it by a PDE residual rather than sampling it -- and finds a
second, unrelated entropic index in the distribution of those residuals.

Run with: ``uv run python examples/anomalous_diffusion.py``
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
from qjax.plots import qcolors, qlinestyles, save_figure, use_qjax_style

FIG_DIR = Path(__file__).parent / "figures"

#: Maximum-likelihood fits are confined to q > 1, because the q-Gaussian has
#: compact support below 1 and the likelihood is then -inf outside it.
Q_FIT_MIN, Q_FIT_MAX = 1.0, 2.9

#: Cold-atom lattice depths, as the recoil-to-depth ratio E_R / U_0. Lutz's law
#: q = 1 + 44 E_R/U_0 puts these at q ~ 1.22, 1.35, 1.48.
#:
#: The upper end is chosen, not arbitrary. Past q = 5/3 the q-Gaussian's second
#: moment diverges, and the *tail* of the stationary state then equilibrates as a
#: power law rather than exponentially: at q = 1.75 the measured index is still
#: 3 % low after 150 relaxation times and creeping upward. That is a genuine
#: difficulty in measuring a heavy-tailed stationary state, and staying below
#: 5/3 keeps this example's error bars meaningful instead of dominated by it.
RECOIL_RATIOS = (0.005, 0.008, 0.011)

#: Simulated time for the cold-atom arms, in units of the friction relaxation
#: time 1/alpha. Matched across arms, so a heavily damped arm is not given an
#: unfair advantage over a weakly damped one.
RELAXATIONS = {"quick": 40, "default": 150, "full": 400}

#: The Douglas-Bergamini-Renzoni (2006) experiment reported entropic indices in
#: roughly this range as the lattice depth was varied.
EXPERIMENT_RANGE = (1.4, 1.7)

GRID_LIMIT = 40.0  # half-width of the density grid, in units of the initial spread


def configuration(*, quick: bool, full: bool) -> dict:
    """Particle counts, step counts and grid resolution for the requested tier."""
    if quick:
        return {
            "particles": 2000,
            "segments": 8,
            "per_segment": 40,
            "dt": 5e-4,
            "bins": 128,
            "fit_steps": 600,
            "relaxations": RELAXATIONS["quick"],
        }
    if full:
        return {
            "particles": 100_000,
            "segments": 60,
            "per_segment": 400,
            "dt": 2e-4,
            "bins": 1024,
            "fit_steps": 8000,
            "relaxations": RELAXATIONS["full"],
        }
    return {
        "particles": 20_000,
        "segments": 40,
        "per_segment": 200,
        "dt": 5e-4,
        "bins": 512,
        "fit_steps": 3000,
        "relaxations": RELAXATIONS["default"],
    }


# --------------------------------------------------------------------------- #
# Dynamics
# --------------------------------------------------------------------------- #
@partial(jax.jit, static_argnames=("steps", "bins"))
def nlfp_segment(key, positions, q, diffusivity, dt, steps: int, bins: int, limit):
    r"""Advance the interacting-particle system whose mean-field limit is the NLFP.

    The porous-medium equation ``dp/dt = D d^2 p^{2-q}/dx^2`` is the mean-field
    limit of the Ito process

        dx = sqrt(2 D (2-q) p(x,t)^{1-q}) dW,

    so the noise amplitude depends on the *density at the particle's own position*.
    That density is re-estimated from the ensemble at every step, which is why
    `qjax.physics.histogram_density` and `interpolate_density` are written to be
    cheap inside a scan. Stating it as a particle system rather than as "solving
    the NLFP" is the honest description: the equation is recovered only as the
    particle number grows.

    Args:
        key: PRNG key.
        positions: Current particle positions, shape ``(P,)``.
        q: Entropic index of the equation.
        diffusivity: ``D``.
        dt: Time step.
        steps: Steps in this segment.
        bins: Density grid resolution.
        limit: Half-width of the density grid.

    Returns:
        ``(final_positions, final_key)``.
    """
    edges = jnp.linspace(-limit, limit, bins + 1)

    def step(carry, _):
        chain_key, state = carry
        chain_key, subkey = jax.random.split(chain_key)
        density = qp.histogram_density(state, edges)
        local = qp.interpolate_density(state, edges, density)
        # Floor the density so a particle that has run into the sparse tail still
        # diffuses instead of freezing; for q > 1 the amplitude would otherwise
        # diverge there rather than vanish.
        amplitude = jnp.sqrt(2.0 * diffusivity * (2.0 - q) * jnp.maximum(local, 1e-9) ** (1.0 - q))
        noise = jax.random.normal(subkey, state.shape) * jnp.sqrt(dt)
        return (chain_key, state + amplitude * noise), None

    (key, positions), _ = jax.lax.scan(step, (key, positions), None, length=steps)
    return positions, key


@partial(jax.jit, static_argnames=("steps",))
def sisyphus_segment(key, momenta, friction, diffusion, momentum_scale, dt, steps: int):
    r"""Advance the saturating-friction Langevin process of the cold-atom problem.

    ``dp = -alpha p / (1 + (p/p_c)^2) dt + sqrt(2 D_0) dW``. The noise is ordinary
    additive Gaussian noise; the heavy tail comes entirely from the friction
    *saturating* at large momentum, so a fast atom is barely damped. The stationary
    state is an exact ``q``-Gaussian -- see `qjax.physics.saturating_langevin_q`.

    Args:
        key: PRNG key.
        momenta: Current momenta, shape ``(P,)``.
        friction: ``alpha``, the friction at small momentum.
        diffusion: ``D_0``.
        momentum_scale: ``p_c``, where the friction saturates.
        dt: Time step.
        steps: Steps in this segment.

    Returns:
        ``(final_momenta, final_key)``.
    """

    def step(carry, _):
        chain_key, state = carry
        chain_key, subkey = jax.random.split(chain_key)
        drag = -friction * state / (1.0 + (state / momentum_scale) ** 2)
        noise = jax.random.normal(subkey, state.shape) * jnp.sqrt(2.0 * diffusion * dt)
        return (chain_key, state + drag * dt + noise), None

    (key, momenta), _ = jax.lax.scan(step, (key, momenta), None, length=steps)
    return momenta, key


def simulate(key, config: dict, kind: str, **parameters):
    """Run one arm, keeping only the snapshot at the end of each segment.

    An outer Python loop over segments with a jitted inner scan: the full
    trajectory would be ``segments * per_segment * particles`` floats, and only the
    segment endpoints are ever used.
    """
    particles = config["particles"]
    key, start_key = jax.random.split(key)
    if kind == "sisyphus":
        state = jnp.zeros((particles,))
    elif kind == "nlfp" and parameters["q"] >= 1.0:
        # Start from the exact self-similar profile. The Tsallis-Bukman solution
        # keeps its q-Gaussian *shape* and only rescales, so a run started on it is
        # self-similar from the first step and the mean-squared displacement is a
        # pure power law: measured, that moves the fitted exponent for q = 1.5 from
        # 1.246 to 1.324 against the exact 4/3. `qjax.sample` supplies the initial
        # condition for its own equation -- but only for 1 <= q < 3, which is where
        # the sampler is exact, hence the Gaussian fallback below.
        state = 0.1 * qjax.sample(start_key, q=parameters["q"], beta=1.0, shape=(particles,))
    else:
        state = 0.1 * jax.random.normal(start_key, (particles,))

    snapshots, times = [np.asarray(state)], [0.0]
    dt, per_segment = config["dt"], config["per_segment"]
    segments = config["segments"]
    if kind == "sisyphus":
        # Equal numbers of relaxation times, not equal wall-clock: the friction
        # differs by a factor of four across the arms, so a fixed step budget
        # would leave the weakly damped (most heavy-tailed) arm unequilibrated
        # and bias its entropic index low.
        total = config["relaxations"] / parameters["friction"]
        per_segment = max(int(total / dt / segments), 1)
    for index in range(segments):
        if kind == "nlfp":
            state, key = nlfp_segment(
                key,
                state,
                parameters["q"],
                parameters["diffusivity"],
                dt,
                per_segment,
                config["bins"],
                GRID_LIMIT,
            )
        elif kind == "brownian":
            state, key = sisyphus_segment(
                key, state, 0.0, parameters["diffusivity"], 1.0, dt, per_segment
            )
        else:
            state, key = sisyphus_segment(
                key,
                state,
                parameters["friction"],
                parameters["diffusion"],
                parameters["momentum_scale"],
                dt,
                per_segment,
            )
        snapshots.append(np.asarray(state))
        times.append((index + 1) * per_segment * dt)
    return np.stack(snapshots), np.asarray(times)


# --------------------------------------------------------------------------- #
# Estimating q
# --------------------------------------------------------------------------- #
@partial(jax.jit, static_argnames=("steps",))
def fit_q_beta(samples, q_init, beta_init, steps: int, learning_rate=0.3):
    r"""Maximum-likelihood ``(q, beta)`` on `qjax.q_gaussian_logpdf`, with Fisher errors.

    The fit itself is the idiom of ``examples/learnable_q.py`` -- plain gradient
    descent on the negative log-likelihood, with ``q`` an ordinary differentiable
    parameter. What is added here is an error bar, because the comparison against
    theory has to be quantitative: the asymptotic covariance is
    ``inv(hessian(mean NLL)) / n`` at the optimum, pushed through the
    ``bounded_q``/softplus reparameterization by `jax.jacfwd`, so ``q_hat``
    arrives with a sigma.

    One limitation is inherent rather than incidental. ``bounded_q`` maps the real
    line onto the *open* interval ``(Q_FIT_MIN, Q_FIT_MAX)``, so ``q = 1`` is
    approached only asymptotically: for genuinely Boltzmann-Gibbs data the fit
    crawls toward the boundary and lands a little above it (about ``1.04`` at the
    default step count, ``1.013`` at ten times more). The Brownian control's
    ``q_hat`` is therefore an *upper bound* on ``q``, not an unbiased estimate --
    which is stated rather than tuned away, and does not affect the ``q > 1`` arms,
    where the fit is accurate to its own sigma.

    Args:
        samples: Observations, shape ``(n,)``.
        q_init: Starting entropic index.
        beta_init: Starting width parameter.
        steps: Gradient-descent steps.
        learning_rate: Step size on the unconstrained parameters.

    Returns:
        ``(q_hat, beta_hat, q_sigma, trace)`` where ``trace`` has shape
        ``(steps, 2)`` and holds the NLL and ``q`` at each step.
    """
    samples = jnp.asarray(samples, dtype=jnp.result_type(float))

    def unpack(raw):
        return bounded_q(raw[0], Q_FIT_MIN, Q_FIT_MAX), jax.nn.softplus(raw[1]) + 1e-3

    def negative_log_likelihood(raw):
        q, beta = unpack(raw)
        return -jnp.mean(qjax.q_gaussian_logpdf(samples, q, beta))

    raw = jnp.array(
        [
            inverse_bounded_q(q_init, Q_FIT_MIN, Q_FIT_MAX),
            jnp.log(jnp.expm1(jnp.maximum(beta_init - 1e-3, 1e-3))),
        ]
    )

    def step(state, _):
        value, grads = jax.value_and_grad(negative_log_likelihood)(state)
        return state - learning_rate * grads, jnp.stack([value, unpack(state)[0]])

    raw, trace = jax.lax.scan(step, raw, None, length=steps)
    q_hat, beta_hat = unpack(raw)

    # Fisher information on the unconstrained parameters, then the delta method.
    curvature = jax.hessian(negative_log_likelihood)(raw)
    covariance = jnp.linalg.inv(curvature) / samples.shape[0]
    sensitivity = jax.jacfwd(lambda r: unpack(r)[0])(raw)
    variance = sensitivity @ covariance @ sensitivity
    return q_hat, beta_hat, jnp.sqrt(jnp.maximum(variance, 0.0)), trace


def measure(
    snapshots: np.ndarray, times: np.ndarray, config: dict, q_guess: float, diffusive: bool
) -> dict:
    """The estimators of ``q`` that apply to this arm.

    The mean-squared-displacement route is available only for a *spreading*
    process. The cold-atom arms relax to a stationary momentum distribution, so
    their second moment saturates instead of growing and ``alpha = 2/(3-q)`` does
    not apply to them at all -- there the density likelihood is the only estimator,
    which is also true of the real experiment.
    """
    displacement = qp.mean_squared_displacement(jnp.asarray(snapshots))
    # Fit on the late half only. The self-similar Tsallis-Bukman regime is reached
    # asymptotically from a narrow initial condition, and including the approach
    # biases the exponent low -- measurably so: for q = 1.5 the last three quarters
    # give 1.308 against the exact 4/3, the last half gives 1.324.
    start = max(len(times) // 2, 1)
    exponent, prefactor, exponent_sigma = qp.fit_power_law(
        jnp.asarray(times), displacement, low=start
    )

    final = snapshots[-1]
    spread = float(np.std(final))
    q_hat, beta_hat, q_sigma, trace = fit_q_beta(
        final, q_guess, 1.0 / max(spread**2, 1e-6), config["fit_steps"]
    )
    # A Fisher sigma that underflows means the optimum ran into the bounded_q
    # boundary, where the sigmoid saturates and the curvature carries no
    # information. Report it as unavailable rather than as an implausibly tight
    # error bar.
    sigma = float(q_sigma)
    reliable = sigma > 1e-5 and Q_FIT_MIN + 1e-3 < float(q_hat) < Q_FIT_MAX - 1e-3

    return {
        "msd": np.asarray(displacement),
        "diffusive": diffusive,
        "exponent": float(exponent) if diffusive else float("nan"),
        "exponent_sigma": float(exponent_sigma) if diffusive else float("nan"),
        "prefactor": float(prefactor),
        "msd_start": start,
        "q_from_density": float(q_hat),
        "q_density_sigma": sigma if reliable else float("nan"),
        "beta_from_density": float(beta_hat),
        "q_from_msd": float(qp.nlfp_index(exponent)) if diffusive else float("nan"),
        "q_msd_sigma": (float(2.0 * exponent_sigma / exponent**2) if diffusive else float("nan")),
        "trace": np.asarray(trace),
    }


# --------------------------------------------------------------------------- #
# Experiment
# --------------------------------------------------------------------------- #
def run(config: dict) -> list[dict]:
    """Every arm: simulate, then measure ``q`` two ways against the known value."""
    arms: list[dict] = []

    specification = [
        ("Brownian ($q=1$)", "brownian", {"diffusivity": 1.0}, 1.0, "control"),
        ("NLFP $q=1.5$", "nlfp", {"q": 1.5, "diffusivity": 1.0}, 1.5, "nlfp"),
        ("NLFP $q=0.5$", "nlfp", {"q": 0.5, "diffusivity": 1.0}, 0.5, "nlfp"),
    ]
    for ratio in RECOIL_RATIOS:
        # Choose (alpha, D_0, p_c) so the exact stationary index matches Lutz's
        # prediction for this lattice depth; then q is known two ways before any
        # data is generated.
        target = float(qp.lutz_q(ratio))
        momentum_scale, diffusion = 1.0, 0.5
        friction = 2.0 * diffusion / ((target - 1.0) * momentum_scale**2)
        specification.append(
            (
                rf"cold atoms $E_R/U_0={ratio:.3f}$",
                "sisyphus",
                {"friction": friction, "diffusion": diffusion, "momentum_scale": momentum_scale},
                target,
                "sisyphus",
            )
        )

    for index, (label, kind, parameters, truth, family) in enumerate(specification):
        snapshots, times = simulate(jax.random.PRNGKey(100 + index), config, kind, **parameters)
        guess = min(max(truth, 1.05), 2.5)
        result = measure(snapshots, times, config, guess, diffusive=family != "sisyphus")
        result.update(
            {
                "label": label,
                "kind": kind,
                "family": family,
                "true_q": truth,
                "true_exponent": float(qp.nlfp_exponent(truth)),
                "snapshots": snapshots,
                "times": times,
                "parameters": parameters,
            }
        )
        if family == "sisyphus":
            result["true_beta"] = float(
                qp.saturating_langevin_beta(parameters["diffusion"], parameters["friction"])
            )
            result["recoil_ratio"] = parameters and float(
                2.0
                * parameters["diffusion"]
                / (44.0 * parameters["friction"] * parameters["momentum_scale"] ** 2)
            )
        arms.append(result)
        print(
            f"    {label:<28} true q {truth:.4f}"
            f"   q(density) {result['q_from_density']:.4f} +- {result['q_density_sigma']:.4f}"
            f"   alpha {result['exponent']:.4f} +- {result['exponent_sigma']:.4f}"
            f"   q(MSD) {result['q_from_msd']:.4f}",
            flush=True,
        )
    return arms


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def plot_main(arms: list[dict], path: Path) -> None:
    """Trajectories, the self-similar collapse, the MSD, and the two estimators."""
    colors = qcolors(len(arms))
    styles = qlinestyles(len(arms))

    fig, axes = plt.subplots(2, 3, figsize=(13.4, 7.6))
    fig.subplots_adjust(hspace=0.36, wspace=0.30)

    # (a) A handful of trajectories from each regime: visibly sub, normal, super.
    ax = axes[0, 0]
    for color, arm in zip(colors[:3], arms[:3], strict=True):
        for walker in range(4):
            ax.plot(
                arm["times"],
                arm["snapshots"][:, walker],
                color=color,
                lw=0.9,
                alpha=0.8,
                label=arm["label"] if walker == 0 else None,
            )
    ax.set_xlabel("time $t$")
    ax.set_ylabel("position $x$")
    ax.set_title("(a) sample trajectories", fontsize=10)
    ax.legend(fontsize=7.4)

    # (b) The exact self-similar solution: rescale by t^{alpha/2} and the density
    # at different times must fall on one q-Gaussian.
    ax = axes[0, 1]
    arm = arms[1]
    exponent = arm["true_exponent"]
    indices = [len(arm["times"]) // 3, 2 * len(arm["times"]) // 3, len(arm["times"]) - 1]
    for color, index in zip(qcolors(3), indices, strict=True):
        snapshot = arm["snapshots"][index]
        scale = arm["times"][index] ** (0.5 * exponent)
        edges = np.linspace(-6.0, 6.0, 121)
        centres = 0.5 * (edges[:-1] + edges[1:])
        density = np.asarray(
            qp.histogram_density(jnp.asarray(snapshot / scale), jnp.asarray(edges))
        )
        ax.semilogy(
            centres,
            np.maximum(density, 1e-5),
            color=color,
            lw=1.1,
            label=rf"$t={arm['times'][index]:.2f}$",
        )
    # The width comes from the arm's own maximum-likelihood fit, transported into
    # the rescaled coordinate exactly: x -> x / t^(alpha/2) multiplies beta by
    # t^alpha. Estimating beta from the empirical variance instead biases the
    # overlay wide, because the sample variance of a q = 1.5 q-Gaussian is
    # dominated by exactly the tail this window cuts off.
    beta = arm["beta_from_density"] * arm["times"][indices[-1]] ** exponent
    ax.semilogy(
        centres,
        np.asarray(qjax.q_gaussian_pdf(jnp.asarray(centres), q=arm["q_from_density"], beta=beta)),
        color="0.25",
        lw=2.6,
        alpha=0.35,
        label=rf"fitted $q$-Gaussian, $\hat q={arm['q_from_density']:.3f}$",
    )
    ax.set_ylim(1e-4, 2.0)
    ax.set_xlabel(r"$x / t^{\alpha/2}$")
    ax.set_ylabel("density")
    ax.set_title(rf"(b) self-similar collapse, $q={arm['true_q']}$", fontsize=10)
    ax.legend(fontsize=7.2)

    # (c) MSD against the exact 2/(3-q) reference slopes.
    ax = axes[0, 2]
    diffusive = [arm for arm in arms if arm["diffusive"]]
    for color, style, arm in zip(colors, styles, diffusive, strict=False):
        times, msd = arm["times"][1:], arm["msd"][1:]
        ax.loglog(
            times,
            msd,
            color=color,
            ls=style,
            lw=1.4,
            label=rf"{arm['label']}: $\alpha={arm['exponent']:.3f}$",
        )
        start = arm["msd_start"]
        reference = arm["prefactor"] * times ** arm["true_exponent"]
        ax.loglog(times[start:], reference[start:], color=color, lw=2.6, alpha=0.28)
    ax.set_xlabel("time $t$")
    ax.set_ylabel(r"$\langle x^2 \rangle$")
    ax.set_title(r"(c) MSD; thick lines are the exact $\alpha = 2/(3-q)$", fontsize=10)
    ax.legend(fontsize=7.2, loc="upper left")

    # (d) The maximum-likelihood run, in the idiom of examples/learnable_q.py.
    ax = axes[1, 0]
    for color, style, arm in zip(colors, styles, arms, strict=True):
        if arm["true_q"] < Q_FIT_MIN:
            continue  # compact support: the MLE cannot reach q < 1
        trace = arm["trace"]
        ax.plot(trace[:, 1], color=color, ls=style, lw=1.3, label=arm["label"])
        ax.axhline(arm["true_q"], color=color, lw=0.7, ls=":")
    ax.set_xlabel("gradient-descent step")
    ax.set_ylabel(r"$\hat q$")
    ax.set_title(r"(d) fitting $q$ by maximum likelihood", fontsize=10)
    ax.legend(fontsize=6.8, ncols=2)

    # (e) The money panel: two independent estimators against the known q.
    ax = axes[1, 1]
    span = np.array([0.4, 2.1])
    ax.plot(span, span, color="0.3", lw=1.0, ls="--", label="exact")
    truth = np.array([arm["true_q"] for arm in arms])
    density = np.array([arm["q_from_density"] for arm in arms])
    density_sigma = np.array([arm["q_density_sigma"] for arm in arms])
    from_msd = np.array([arm["q_from_msd"] for arm in arms])
    msd_sigma = np.array([arm["q_msd_sigma"] for arm in arms])
    fittable = truth >= Q_FIT_MIN
    ax.errorbar(
        truth[fittable],
        density[fittable],
        yerr=2 * density_sigma[fittable],
        ls="none",
        marker="o",
        ms=6,
        capsize=3,
        color=colors[4],
        label=r"density MLE ($\pm 2\sigma$)",
    )
    ax.errorbar(
        truth,
        from_msd,
        yerr=2 * msd_sigma,
        ls="none",
        marker="s",
        ms=6,
        capsize=3,
        mfc="none",
        color=colors[1],
        label=r"$3 - 2/\alpha_{\rm MSD}$ ($\pm 2\sigma$)",
    )
    ax.set_xlabel("true entropic index $q$")
    ax.set_ylabel(r"measured $\hat q$")
    ax.set_title("(e) two independent estimators vs. theory", fontsize=10)
    ax.legend(fontsize=7.4, loc="upper left")

    # (f) Cold atoms against Lutz's law and the reported experimental range.
    ax = axes[1, 2]
    ratios = np.linspace(0.0, 0.02, 100)
    ax.plot(
        ratios,
        np.asarray(qp.lutz_q(jnp.asarray(ratios))),
        color="0.25",
        lw=1.3,
        ls="--",
        label=r"Lutz: $q = 1 + 44\,E_R/U_0$",
    )
    ax.axhspan(*EXPERIMENT_RANGE, color=colors[1], alpha=0.16, lw=0, label="Douglas et al. (2006)")
    cold = [arm for arm in arms if arm["family"] == "sisyphus"]
    measured_ratios = np.array([(arm["true_q"] - 1.0) / 44.0 for arm in cold])
    ax.errorbar(
        measured_ratios,
        [arm["q_from_density"] for arm in cold],
        yerr=[2 * np.nan_to_num(arm["q_density_sigma"]) for arm in cold],
        ls="none",
        marker="o",
        ms=6,
        capsize=3,
        color=colors[4],
        label=r"measured from the simulation ($\pm 2\sigma$)",
    )
    ax.set_xlabel(r"$E_R / U_0$")
    ax.set_ylabel(r"$q$")
    ax.set_title("(f) cold atoms in a dissipative lattice", fontsize=10)
    ax.legend(fontsize=7.2, loc="upper left")

    save_figure(fig, path)
    plt.close(fig)


# --------------------------------------------------------------------------- #
def report(arms: list[dict]) -> None:
    """Print the validation table the docs page reproduces."""
    print("\n  validation")
    header = (
        f"    {'arm':<28}{'q true':>8}{'q from density':>22}{'q from MSD':>22}{'alpha / exact':>18}"
    )
    print(header)
    for arm in arms:
        if arm["true_q"] >= Q_FIT_MIN:
            sigma = arm["q_density_sigma"]
            density = (
                f"{arm['q_from_density']:.4f} +- {sigma:.4f}"
                if np.isfinite(sigma)
                else f"{arm['q_from_density']:.4f} (at bound)"
            )
        else:
            density = "n/a (compact support)"
        if arm["diffusive"]:
            from_msd = f"{arm['q_from_msd']:.4f} +- {arm['q_msd_sigma']:.4f}"
            alpha = f"{arm['exponent']:.4f} / {arm['true_exponent']:.4f}"
        else:
            from_msd = "n/a (stationary)"
            alpha = "n/a (stationary)"
        label = arm["label"].replace("$", "")
        print(f"    {label:<28}{arm['true_q']:>8.4f}{density:>22}{from_msd:>22}{alpha:>18}")

    control = arms[0]
    print("\n    the Brownian control is the null test of both estimators:")
    print(
        f"      alpha = {control['exponent']:.4f} +- {control['exponent_sigma']:.4f} (exact 1.0)"
        f"   q(density) = {control['q_from_density']:.4f}"
        f" +- {control['q_density_sigma']:.4f} (exact 1.0)"
    )

    print("\n    agreement with theory, in sigma (blank where the estimator does not apply)")
    for arm in arms:
        pieces = []
        if arm["diffusive"]:
            gap = abs(arm["q_from_msd"] - arm["true_q"])
            pieces.append(f"MSD {gap / arm['q_msd_sigma']:5.1f}")
        if arm["true_q"] >= Q_FIT_MIN and np.isfinite(arm["q_density_sigma"]):
            gap = abs(arm["q_from_density"] - arm["true_q"])
            pieces.append(f"density {gap / arm['q_density_sigma']:5.1f}")
        print(f"      {arm['label'].replace('$', ''):<28}" + "   ".join(pieces))

    print("\n    Lutz's law, for the cold-atom arms")
    for arm in arms:
        if arm["family"] != "sisyphus":
            continue
        ratio = (arm["true_q"] - 1.0) / 44.0
        print(
            f"      E_R/U_0 = {ratio:.4f}   Lutz q = {float(qp.lutz_q(ratio)):.4f}"
            f"   exact stationary q = {arm['true_q']:.4f}"
            f"   measured {arm['q_from_density']:.4f} +- {arm['q_density_sigma']:.4f}"
        )


def main(*, quick: bool = False, full: bool = False) -> None:
    """Run every arm and write the figure."""
    use_qjax_style()
    config = configuration(quick=quick, full=full)
    print(
        f"Anomalous diffusion: {config['particles']} particles,"
        f" {config['segments'] * config['per_segment']} steps"
    )
    arms = run(config)
    report(arms)
    plot_main(arms, FIG_DIR / "anomalous_diffusion")


if __name__ == "__main__":
    main(
        quick="--quick" in sys.argv,
        full="--full" in sys.argv or bool(os.environ.get("QJAX_FULL")),
    )
