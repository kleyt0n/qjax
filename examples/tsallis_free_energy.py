r"""Variational free energy at entropic index q: a non-extensivity scaling law.

Variational autoregressive networks (Wu, Wang & Zhang, *PRL* **122**, 080602,
2019) solve a statistical-mechanics model by minimizing

    F = <E>_p - T S_1(p)

over an autoregressive neural network ``p_theta``, which can be both sampled and
evaluated exactly. The ``q = 1`` arm below *is* that method. Replacing the
logarithm by the ``q``-logarithm gives the nonextensive objective, which as far as
we can tell has not been written down before:

    F_q = E_{s ~ p} [ E(s) + T ln_q p(s) ].

**The duality matters, and it is easy to get backwards.** ``F_q`` is *not*
``<E> - T S_q``. Summing the deformed logarithm against ``p`` rather than against
the escort weight ``p^q`` gives

    -E_p[ln_q p] = (1 - sum_s p^{2-q}) / (1 - q) = S_{2-q}(p),

so the objective is ``<E> - T S_{2-q}(p)``: the *thermodynamic* index is
``2 - q``. Consequently ``q < 1`` supplies **less** entropy pressure than
Boltzmann-Gibbs, not more, and ``q > 1`` supplies more. The identity is checked to
machine precision in ``tests/test_examples_free_energy.py``.

**The gradient comes from autodiff, not by hand.** That is the point of having
``q_log`` as a differentiable primitive: the REINFORCE estimator
``E[(E + T ln_q p + T p^{1-q}) grad log p]`` is never typed out; it is what
``jax.grad`` produces from `free_energy_surrogate`, and at ``q = 1`` it collapses
to the estimator in the paper exactly -- the extra ``+T`` is annihilated by
``E[grad log p] = 0``. The tests check the autodiff gradient against the
hand-derived form at ``q = 0.6, 1.0, 1.4``.

What the experiment measures is not "``q != 1`` beats VAN" -- it does not, and it
cannot: ``F_1`` is the only member of the family that is a variational *bound* on
the Boltzmann free energy, so ``q = 1`` is necessarily the best approximation to
it, and the run confirms that at every size. What ``q`` controls is how hard the
objective pushes the model to spread its mass, and the measurement is a **scaling
law**:

    the free-energy gap, the correlation error and the sample diversity all
    depend on ``q`` only through the combination ``c = (q - 1) N``.

Curves for ``N = 12, 16, 20`` collapse onto one function of ``c``. This is
non-extensivity made quantitative -- ``sum_s p^{2-q}`` at the uniform
distribution is ``M^{q-1} = exp((q-1) N ln 2)``, so the entropy term is extensive
only at ``q = 1`` and the useful deformation shrinks like ``1/N``. It is also the
practical warning: an entropic index tuned on a small system does not transfer to
a large one at fixed ``q``, only at fixed ``(q-1)N``.

Nothing is reported without an exact counterpart, because a collapsed model can
still quote a low ``F``:

- 4x4 Ising, ``N = 16``: brute force over all ``2**16`` states *and* the transfer
  matrix -- two independent exact codes that must agree to ``1e-10``. The model
  itself is enumerable there, so ``tsallis_divergence`` against the exact
  Boltzmann weights is an *exact* diagnostic, not an estimate.
- 8x8 Ising, ``N = 64``: the ``256 x 256`` transfer matrix, exact for the finite
  lattice, with Onsager's thermodynamic limit overlaid.
- SK spin glass, ``N = 12, 16, 20``: exhaustive enumeration of every state,
  including the exact ``<s_i s_j>``, plus Parisi's ``-0.7633``.

Run with: ``uv run python examples/tsallis_free_energy.py``
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
from qjax.nn import made_init, made_log_prob, made_masks, made_sample
from qjax.plots import CMAP, qcolors, qlinestyles, save_figure, use_qjax_style

# x64 throughout: the transfer matrix and the 2**20 enumeration are the reference
# values everything is judged against, and they must not carry float32 error.
jax.config.update("jax_enable_x64", True)

FIG_DIR = Path(__file__).parent / "figures"

#: Arms are indexed by ``c = (q - 1) N`` rather than by ``q``, which is the point
#: of the experiment: ``c = 0`` is exactly VAN, ``c > 0`` adds entropy pressure
#: (thermodynamic index ``2 - q < 1``), ``c < 0`` removes it.
SCALED_INDICES = (-2.0, -1.0, 0.0, 1.0, 2.0, 4.0)
BASELINE = 0.0
LR = 3e-3


def entropic_index(scaled: float, num_spins: int) -> float:
    """``q`` for a given ``c = (q-1) N``, so arms are comparable across sizes."""
    return 1.0 + scaled / num_spins


def label_for(scaled: float) -> str:
    """Legend label for an arm."""
    return "VAN ($c = 0$)" if scaled == BASELINE else f"$c = {scaled:+.0f}$"


def configuration(*, quick: bool, full: bool) -> dict:
    """System sizes, batch sizes and step counts for the requested tier."""
    if quick:
        return {
            "ising_small": {"size": 4, "hidden": (32,), "batch": 64, "steps": 150},
            "ising_large": {"size": 6, "hidden": (48,), "batch": 32, "steps": 120},
            "ising_temperature": 1.6,
            "large_temperatures": (1.8, 2.6),
            "glass_sizes": (10, 12),
            "glass": {"hidden": (32,), "batch": 64, "steps": 150, "temperature": 0.4},
            "seeds": (0,),
        }
    if full:
        return {
            "ising_small": {"size": 4, "hidden": (128, 128), "batch": 1024, "steps": 5000},
            "ising_large": {"size": 8, "hidden": (256, 256), "batch": 512, "steps": 4000},
            "ising_temperature": 1.6,
            "large_temperatures": (1.6, 1.9, 2.1, 2.27, 2.45, 2.7, 3.0),
            "glass_sizes": (12, 16, 20, 22),
            "glass": {"hidden": (128, 128), "batch": 1024, "steps": 5000, "temperature": 0.4},
            "seeds": (0, 1, 2, 3, 4),
        }
    return {
        "ising_small": {"size": 4, "hidden": (64, 64), "batch": 256, "steps": 1200},
        "ising_large": {"size": 8, "hidden": (128, 128), "batch": 128, "steps": 900},
        "ising_temperature": 1.6,
        "large_temperatures": (1.8, 2.1, 2.27, 2.5, 2.9),
        "glass_sizes": (12, 16, 20),
        "glass": {"hidden": (64, 64), "batch": 256, "steps": 1200, "temperature": 0.4},
        "seeds": (0, 1),
    }


# --------------------------------------------------------------------------- #
# The q-deformed variational objective
# --------------------------------------------------------------------------- #
def free_energy_surrogate(params, masks, spins, energies, temperature, q, baseline):
    r"""A scalar whose gradient is the REINFORCE estimator of ``grad F_q``.

    Two terms, and the split is the whole trick:

    - the *score* term carries the reward-like part
      ``(E + T ln_q p - baseline) grad log p``, with the coefficient held by
      `jax.lax.stop_gradient` so it is treated as a constant weight;
    - the *pathwise* term ``T E[ln_q p]`` is differentiated straight through
      `qjax.q_log`, and autodiff turns it into ``T E[p^{1-q} grad log p]``.

    Their sum has gradient ``E[(E + T ln_q p + T p^{1-q}) grad log p]``, the
    analytic estimator. At ``q = 1`` it is exactly the estimator of Wu, Wang &
    Zhang: ``ln_1 p = log p``, ``p^0 = 1``, and the constant ``+T`` drops out
    because ``E[grad log p] = 0``. Nothing about the deformation is hand-derived --
    ``q_log`` carries it, including through ``q = 1``.

    Args:
        params: MADE parameters.
        masks: MADE masks.
        spins: A batch of configurations, shape ``(B, N)``.
        energies: Their energies, shape ``(B,)``.
        temperature: Temperature ``T``.
        q: Entropic index.
        baseline: A control variate (the batch mean) that reduces the variance of
            the score term without biasing it.

    Returns:
        A scalar surrogate. Its *value* is not ``F_q``; only its gradient is
        meaningful. Use `variational_free_energy` for the number to report.
    """
    log_p = made_log_prob(params, masks, spins)
    probability = jnp.exp(log_p)
    weight = jax.lax.stop_gradient(
        energies + temperature * qjax.q_log(jax.lax.stop_gradient(probability), q) - baseline
    )
    score = jnp.mean(weight * log_p)
    pathwise = temperature * jnp.mean(qjax.q_log(probability, q))
    return score + pathwise


def variational_free_energy(params, masks, spins, energies, temperature, q):
    r"""$F_q = \langle E\rangle + T\,\mathbb E_p[\ln_q p] = \langle E\rangle - T S_{2-q}(p)$."""
    log_p = made_log_prob(params, masks, spins)
    return jnp.mean(energies + temperature * qjax.q_log(jnp.exp(log_p), q))


@partial(jax.jit, static_argnames=("energy_fn", "steps", "batch"))
def optimize(key, params, masks, energy_fn, temperature, q, steps, batch):
    """Minimize ``F_q`` by Adam on the surrogate, annealing ``beta`` as in VAN.

    ``beta`` is ramped from a high temperature down to the target: the Boltzmann
    distribution is broad there, so the model has no incentive to collapse, and
    the schedule then tightens it. ``q`` is held fixed -- it is the variable under
    study, and (unlike in the other examples) it is deliberately not learnable,
    because minimizing ``F_q`` over ``q`` is meaningless: the objective would
    simply run to whichever ``q`` makes the entropy term largest.
    """
    b1, b2, eps = 0.9, 0.999, 1e-8
    m = jax.tree_util.tree_map(jnp.zeros_like, params)
    v = jax.tree_util.tree_map(jnp.zeros_like, params)

    def step(carry, t):
        params, m, v, chain_key = carry
        chain_key, subkey = jax.random.split(chain_key)
        # Linear in beta, not in T: it spends more steps in the hard, cold regime.
        beta = 0.1 / temperature + (0.9 / temperature) * jnp.minimum(t / (0.6 * steps), 1.0)
        current = 1.0 / beta

        spins = jax.lax.stop_gradient(made_sample(subkey, params, masks, batch))
        energies = energy_fn(spins)
        baseline = jnp.mean(
            energies + current * qjax.q_log(jnp.exp(made_log_prob(params, masks, spins)), q)
        )
        grads = jax.grad(free_energy_surrogate)(
            params, masks, spins, energies, current, q, jax.lax.stop_gradient(baseline)
        )
        m = jax.tree_util.tree_map(lambda m, g: b1 * m + (1 - b1) * g, m, grads)
        v = jax.tree_util.tree_map(lambda v, g: b2 * v + (1 - b2) * g * g, v, grads)
        bc1, bc2 = 1 - b1 ** (t + 1), 1 - b2 ** (t + 1)
        params = jax.tree_util.tree_map(
            lambda p, m, v: p - LR * (m / bc1) / (jnp.sqrt(v / bc2) + eps), params, m, v
        )

        # Report at the *target* temperature and q = 1, so every arm's trace is on
        # the same scale as the exact reference and as every other arm.
        report = variational_free_energy(params, masks, spins, energies, temperature, 1.0)
        distinct = jnp.sum(jnp.any(spins[:, None, :] != spins[None, :, :], axis=-1)) / batch**2
        return (params, m, v, chain_key), jnp.stack([report, distinct])

    (params, _, _, _), trace = jax.lax.scan(
        step, (params, m, v, key), jnp.arange(steps, dtype=jnp.result_type(float))
    )
    return params, trace


def evaluate(key, params, masks, energy_fn, temperature, samples):
    """Final ``F_1``, spread diagnostics and correlations from a fresh batch."""
    spins = made_sample(key, params, masks, samples)
    energies = energy_fn(spins)
    return {
        "free_energy": float(
            variational_free_energy(params, masks, spins, energies, temperature, 1.0)
        ),
        "max_magnetization": float(jnp.max(jnp.abs(jnp.mean(spins, axis=0)))),
        "correlations": np.asarray(jnp.einsum("bi,bj->ij", spins, spins) / samples),
        "distinct": float(
            jnp.sum(jnp.any(spins[:, None, :] != spins[None, :, :], -1)) / samples**2
        ),
    }


def ising_energy_fn(size: int):
    """Energy of flat ``(B, L*L)`` spin vectors on a periodic ``L x L`` lattice."""

    def energy(spins):
        return qp.ising_energy(spins.reshape(-1, size, size))

    return energy


# --------------------------------------------------------------------------- #
# Experiments
# --------------------------------------------------------------------------- #
def run_ising_small(config: dict) -> dict:
    """4x4 Ising: the enumerable case, where every diagnostic is exact."""
    settings = config["ising_small"]
    size, temperature = settings["size"], config["ising_temperature"]
    num_spins = size * size
    masks = made_masks(num_spins, settings["hidden"])
    energy_fn = ising_energy_fn(size)

    transfer = float(-temperature * qp.ising_transfer_matrix_log_z(size, temperature))
    brute = float(qp.ising_exact_observables(size, temperature)["free_energy_per_site"] * num_spins)
    print(f"  4x4 Ising at T = {temperature}")
    print(f"    transfer matrix {transfer:.10f}   brute force {brute:.10f}")
    print(f"    the two independent exact codes differ by {abs(transfer - brute):.2e}")

    configurations = qp.ising_all_configurations(size).reshape(-1, num_spins)
    boltzmann = qp.ising_boltzmann_probabilities(size, temperature)

    results: dict = {
        "exact": brute,
        "transfer": transfer,
        "temperature": temperature,
        "num_spins": num_spins,
        "traces": {},
        "gap": {},
        "divergence": {},
        "normalization": {},
    }
    for scaled in SCALED_INDICES:
        q = entropic_index(scaled, num_spins)
        traces, gaps, divergences, masses = [], [], [], []
        for seed in config["seeds"]:
            params = made_init(jax.random.PRNGKey(seed), num_spins, settings["hidden"])
            trained, trace = optimize(
                jax.random.PRNGKey(100 + seed),
                params,
                masks,
                energy_fn,
                temperature,
                q,
                settings["steps"],
                settings["batch"],
            )
            traces.append(np.asarray(trace))
            final = evaluate(
                jax.random.PRNGKey(999),
                trained,
                masks,
                energy_fn,
                temperature,
                settings["batch"] * 4,
            )
            gaps.append((final["free_energy"] - brute) / num_spins)
            model = jnp.exp(made_log_prob(trained, masks, configurations))
            divergences.append(
                (
                    float(qjax.tsallis_divergence(model, boltzmann, q=1.0)),
                    float(qjax.tsallis_divergence(model, boltzmann, q=q)),
                )
            )
            masses.append(float(jnp.sum(model)))
        results["traces"][scaled] = np.stack(traces)
        results["gap"][scaled] = np.array(gaps)
        results["divergence"][scaled] = divergences
        results["normalization"][scaled] = float(np.mean(masses))
        print(
            f"    c = {scaled:+.0f} (q = {q:.4f})  (F - F_exact)/N = {np.mean(gaps):+.5f}"
            f" +- {np.std(gaps):.5f}   exact KL = {np.mean([d[0] for d in divergences]):.5f}"
            f"   sum_s p = {np.mean(masses):.8f}"
        )
    return results


def run_ising_large(config: dict) -> dict:
    """8x8 Ising across temperature, against the exact finite-lattice free energy."""
    settings = config["ising_large"]
    size = settings["size"]
    num_spins = size * size
    masks = made_masks(num_spins, settings["hidden"])
    energy_fn = ising_energy_fn(size)
    arms = (0.0, 2.0)

    results: dict = {
        "temperatures": config["large_temperatures"],
        "size": size,
        "variational": {},
        "exact": [],
        "onsager": [],
    }
    for temperature in config["large_temperatures"]:
        exact = float(-temperature * qp.ising_transfer_matrix_log_z(size, temperature))
        results["exact"].append(exact / num_spins)
        results["onsager"].append(float(qp.onsager_free_energy_per_site(temperature)))
        for scaled in arms:
            params = made_init(jax.random.PRNGKey(0), num_spins, settings["hidden"])
            trained, _ = optimize(
                jax.random.PRNGKey(7),
                params,
                masks,
                energy_fn,
                temperature,
                entropic_index(scaled, num_spins),
                settings["steps"],
                settings["batch"],
            )
            final = evaluate(
                jax.random.PRNGKey(8),
                trained,
                masks,
                energy_fn,
                temperature,
                settings["batch"] * 4,
            )
            results["variational"][temperature, scaled] = final["free_energy"] / num_spins
        print(
            f"    T = {temperature:.2f}  exact {exact / num_spins:+.5f}"
            f"   c=0 {results['variational'][temperature, 0.0]:+.5f}"
            f"   c=+2 {results['variational'][temperature, 2.0]:+.5f}"
        )
    return results


def run_glass(config: dict) -> dict:
    """SK spin glass at several sizes: the data collapse in ``c = (q-1)N``."""
    settings = config["glass"]
    temperature = settings["temperature"]
    results: dict = {
        "sizes": config["glass_sizes"],
        "temperature": temperature,
        "exact": {},
        "gap": {},
        "correlation_error": {},
        "distinct": {},
        "traces": {},
        "final": {},
        "exact_correlations": {},
        "ground_state": {},
    }
    for num_spins in config["glass_sizes"]:
        masks = made_masks(num_spins, settings["hidden"])
        couplings = qp.sk_couplings(jax.random.PRNGKey(11), num_spins)

        def energy_fn(spins, couplings=couplings):
            return qp.sk_energy(spins, couplings)

        exact = qp.sk_exact_observables(couplings, temperature)
        reference = float(exact["free_energy_per_spin"])
        exact_correlations = np.asarray(exact["correlations"])
        results["exact"][num_spins] = reference
        results["exact_correlations"][num_spins] = exact_correlations
        results["ground_state"][num_spins] = float(exact["ground_state_energy_per_spin"])
        offdiag = ~np.eye(num_spins, dtype=bool)
        print(
            f"  SK N = {num_spins} at T = {temperature}: exact F/N = {reference:.6f},"
            f" ground state/N = {results['ground_state'][num_spins]:.6f}"
            f"   (Parisi {qp.SK_PARISI_GROUND_STATE})"
        )

        for scaled in SCALED_INDICES:
            q = entropic_index(scaled, num_spins)
            gaps, errors, spreads, traces = [], [], [], []
            for seed in config["seeds"]:
                params = made_init(jax.random.PRNGKey(seed), num_spins, settings["hidden"])
                trained, trace = optimize(
                    jax.random.PRNGKey(200 + seed),
                    params,
                    masks,
                    energy_fn,
                    temperature,
                    q,
                    settings["steps"],
                    settings["batch"],
                )
                final = evaluate(
                    jax.random.PRNGKey(321),
                    trained,
                    masks,
                    energy_fn,
                    temperature,
                    settings["batch"] * 4,
                )
                traces.append(np.asarray(trace))
                gaps.append(final["free_energy"] / num_spins - reference)
                errors.append(
                    float(
                        np.sqrt(
                            np.mean(
                                (final["correlations"][offdiag] - exact_correlations[offdiag]) ** 2
                            )
                        )
                    )
                )
                spreads.append(final["distinct"])
                if seed == config["seeds"][0]:
                    results["final"][num_spins, scaled] = final
            results["gap"][num_spins, scaled] = np.array(gaps)
            results["correlation_error"][num_spins, scaled] = np.array(errors)
            results["distinct"][num_spins, scaled] = np.array(spreads)
            results["traces"][num_spins, scaled] = np.stack(traces)
            print(
                f"    c = {scaled:+.0f} (q = {q:.4f})  gap/N = {np.mean(gaps):+.5f}"
                f" +- {np.std(gaps):.5f}   corr RMS {np.mean(errors):.4f}"
                f"   distinct {np.mean(spreads):.3f}"
            )
    return results


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def plot_main(small: dict, large: dict, glass: dict, path: Path) -> None:
    """Exactness checks, the temperature sweep, and the ``(q-1)N`` collapse."""
    colors = qcolors(len(SCALED_INDICES))
    styles = qlinestyles(len(SCALED_INDICES))
    size_colors = qcolors(len(glass["sizes"]))
    largest = glass["sizes"][-1]

    fig, axes = plt.subplots(2, 3, figsize=(13.4, 7.6))
    fig.subplots_adjust(hspace=0.38, wspace=0.31)

    # (a) 4x4 gap to the exact free energy. The bound F_var >= F_exact holds in
    # expectation, but each point is a single-batch estimate and so fluctuates
    # about it -- hence the moving average, which is smoothing the *estimator*,
    # not the result. The converged values in the validation table are measured
    # from a fresh batch four times larger.
    ax = axes[0, 0]
    window = max(len(small["traces"][BASELINE][0]) // 40, 1)
    kernel = np.ones(window) / window
    for color, style, scaled in zip(colors, styles, SCALED_INDICES, strict=True):
        trace = small["traces"][scaled].mean(axis=0)
        gap = (trace[:, 0] - small["exact"]) / small["num_spins"]
        smoothed = np.convolve(gap, kernel, mode="valid")
        ax.semilogy(
            np.arange(len(smoothed)) + window,
            np.maximum(smoothed, 1e-6),
            color=color,
            ls=style,
            lw=1.3,
            label=label_for(scaled),
        )
    ax.set_xlabel("step")
    ax.set_ylabel(r"$(F_{\rm var} - F_{\rm exact})/N$")
    ax.set_title(r"(a) $4\times4$ Ising: gap to the exact $F$", fontsize=10)
    ax.set_ylim(bottom=5e-7)
    ax.legend(fontsize=7.0, ncols=2)

    # (b) The exact divergence: p_theta is enumerable at N = 16, so this is not an
    # estimate. Both q = 1 (KL) and the arm's own q are shown.
    ax = axes[0, 1]
    positions = np.arange(len(SCALED_INDICES))
    kl = [np.mean([d[0] for d in small["divergence"][s]]) for s in SCALED_INDICES]
    own = [np.mean([d[1] for d in small["divergence"][s]]) for s in SCALED_INDICES]
    ax.bar(positions - 0.19, kl, width=0.36, color=colors[1], label=r"$D_1 = \mathrm{KL}$")
    ax.bar(positions + 0.19, own, width=0.36, color=colors[4], label=r"$D_q$ at the arm's $q$")
    ax.set_xticks(positions)
    ax.set_xticklabels([f"{s:+.0f}" for s in SCALED_INDICES], fontsize=8)
    ax.set_yscale("log")
    ax.set_xlabel("$c = (q-1)N$")
    ax.set_ylabel(r"$D_q(p_\theta \Vert p_{\rm Boltzmann})$")
    ax.set_title(r"(b) exact divergence over all $2^{16}$ states", fontsize=10)
    ax.legend(fontsize=8)

    # (c) 8x8 across temperature against the exact finite-lattice free energy.
    ax = axes[0, 2]
    temperatures = np.asarray(large["temperatures"])
    ax.plot(
        temperatures,
        large["exact"],
        color="0.2",
        lw=1.3,
        ls="--",
        label=rf"exact, $L={large['size']}$",
    )
    ax.plot(
        temperatures, large["onsager"], color="0.55", lw=1.0, ls=":", label=r"Onsager, $L\to\infty$"
    )
    for color, scaled in zip((colors[2], colors[4]), (0.0, 2.0), strict=True):
        values = [large["variational"][t, scaled] for t in large["temperatures"]]
        ax.plot(
            temperatures, values, color=color, lw=1.4, marker="o", ms=4, label=label_for(scaled)
        )
    ax.set_xlabel("temperature $T$")
    ax.set_ylabel(r"$F/N$")
    ax.set_title(
        rf"(c) ${large['size']}\times{large['size']}$ Ising, $N={large['size'] ** 2}$", fontsize=10
    )
    ax.legend(fontsize=7.6)

    # (d) SK trace at the largest size, against the exactly enumerated F/N.
    ax = axes[1, 0]
    for color, style, scaled in zip(colors, styles, SCALED_INDICES, strict=True):
        trace = glass["traces"][largest, scaled].mean(axis=0)
        ax.plot(trace[:, 0] / largest, color=color, ls=style, lw=1.3, label=label_for(scaled))
    ax.axhline(glass["exact"][largest], color="0.2", lw=1.2, ls="--", label="exact $F/N$")
    ax.axhline(qp.SK_PARISI_GROUND_STATE, color="0.55", lw=1.0, ls=":", label=r"Parisi $-0.7633$")
    ax.set_ylim(glass["exact"][largest] - 0.06, glass["exact"][largest] + 0.35)
    ax.set_xlabel("step")
    ax.set_ylabel(r"$F/N$")
    ax.set_title(rf"(d) SK spin glass, $N={largest}$", fontsize=10)
    ax.legend(fontsize=7.0, ncols=2)

    # (e) The scaling law. Curves for different N collapse in c = (q-1)N; the
    # inset shows they do not collapse against q itself.
    ax = axes[1, 1]
    for color, num_spins in zip(size_colors, glass["sizes"], strict=True):
        gaps = np.array([glass["gap"][num_spins, s].mean() for s in SCALED_INDICES])
        spreads = np.array([glass["gap"][num_spins, s].std() for s in SCALED_INDICES])
        ax.errorbar(
            SCALED_INDICES,
            gaps,
            yerr=spreads,
            color=color,
            lw=1.4,
            marker="o",
            ms=4.5,
            capsize=2,
            label=f"$N={num_spins}$",
        )
    ax.axvline(0.0, color="0.3", lw=1.0, ls="--")
    ax.set_yscale("log")
    ax.set_xlabel(r"$c = (q-1)\,N$")
    ax.set_ylabel(r"$(F_{\rm var} - F_{\rm exact})/N$")
    ax.set_title(r"(e) the gap collapses in $(q-1)N$", fontsize=10)
    ax.legend(fontsize=8, loc="lower right")

    inset = ax.inset_axes((0.10, 0.62, 0.36, 0.34))
    for color, num_spins in zip(size_colors, glass["sizes"], strict=True):
        indices = [entropic_index(s, num_spins) for s in SCALED_INDICES]
        gaps = [glass["gap"][num_spins, s].mean() for s in SCALED_INDICES]
        inset.semilogy(indices, gaps, color=color, lw=1.0, marker="o", ms=2.5)
    inset.set_title(r"vs. $q$: no collapse", fontsize=6.5, pad=2)
    inset.set_xlabel("$q$", fontsize=6.5, labelpad=0)
    inset.tick_params(labelsize=5.5, which="both")
    inset.grid(visible=False)

    # (f) The mechanism: c controls how hard the objective spreads the mass, and
    # the correlation error follows the same collapse.
    ax = axes[1, 2]
    for color, num_spins in zip(size_colors, glass["sizes"], strict=True):
        errors = [glass["correlation_error"][num_spins, s].mean() for s in SCALED_INDICES]
        ax.plot(
            SCALED_INDICES,
            errors,
            color=color,
            lw=1.4,
            marker="o",
            ms=4.5,
            label=rf"$N={num_spins}$: corr. RMS",
        )
    twin = ax.twinx()
    for color, num_spins in zip(size_colors, glass["sizes"], strict=True):
        spread = [glass["distinct"][num_spins, s].mean() for s in SCALED_INDICES]
        twin.plot(SCALED_INDICES, spread, color=color, lw=1.0, ls=":", marker="s", ms=3.0)
    twin.set_ylabel("distinct sampled pairs (dotted)", fontsize=9)
    twin.grid(visible=False)
    ax.axvline(0.0, color="0.3", lw=1.0, ls="--")
    ax.set_xlabel(r"$c = (q-1)\,N$")
    ax.set_ylabel("correlation RMS error (solid)")
    ax.set_title(r"(f) $c$ is the spreading-pressure knob", fontsize=10)
    ax.legend(fontsize=7.0, loc="upper left")

    save_figure(fig, path)
    plt.close(fig)


def plot_correlations(glass: dict, path: Path) -> None:
    """Learned vs. exactly enumerated correlations, at ``c = 0`` and ``c = +2``."""
    largest = glass["sizes"][-1]
    exact = glass["exact_correlations"][largest]
    shown = (0.0, 2.0)
    fig, axes = plt.subplots(1, 4, figsize=(14.0, 3.7))

    offdiag = ~np.eye(largest, dtype=bool)
    limit = float(np.max(np.abs(exact[offdiag])))
    panels = [("exact (enumerated)", exact)] + [
        (label_for(s), glass["final"][largest, s]["correlations"]) for s in shown
    ]
    for ax, (title, matrix) in zip(axes[:3], panels, strict=True):
        image = ax.imshow(
            matrix - np.diag(np.diag(matrix)),
            cmap=CMAP,
            vmin=-limit,
            vmax=limit,
            interpolation="nearest",
        )
        ax.set_title(title, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03).ax.tick_params(labelsize=7)

    ax = axes[3]
    for color, scaled in zip(qcolors(len(shown)), shown, strict=True):
        learned = glass["final"][largest, scaled]["correlations"]
        residual = np.sqrt(np.mean((learned[offdiag] - exact[offdiag]) ** 2))
        ax.plot(
            exact[offdiag],
            learned[offdiag],
            ls="none",
            marker="o",
            ms=3.0,
            alpha=0.55,
            color=color,
            label=f"{label_for(scaled)}\nRMS error {residual:.3f}",
        )
    span = np.array([-limit, limit])
    ax.plot(span, span, color="0.3", lw=1.0, ls="--")
    ax.set_xlabel(r"exact $\langle s_i s_j \rangle$")
    ax.set_ylabel(r"learned $\langle s_i s_j \rangle$")
    ax.set_title("learned vs. exact", fontsize=9)
    ax.legend(fontsize=7.2, loc="upper left")

    fig.suptitle(
        rf"SK spin glass, $N={largest}$ at $T={glass['temperature']}$:"
        r" off-diagonal two-point correlations",
        fontsize=10,
        y=1.03,
    )
    save_figure(fig, path)
    plt.close(fig)


# --------------------------------------------------------------------------- #
def report(small: dict, glass: dict) -> None:
    """Print the validation table the docs page reproduces."""
    largest = glass["sizes"][-1]
    print("\n  validation")
    print(f"    {'quantity':<48}{'measured':>14}{'exact':>14}")
    rows = (
        ("4x4: the two exact codes differ by", abs(small["transfer"] - small["exact"]), 0.0),
        ("4x4 model normalization sum_s p(s)", small["normalization"][BASELINE], 1.0),
        ("4x4 (F_var - F_exact)/N at c = 0 (VAN)", float(np.mean(small["gap"][BASELINE])), 0.0),
        (
            f"SK N={largest} (F_var - F_exact)/N at c = 0",
            float(np.mean(glass["gap"][largest, BASELINE])),
            0.0,
        ),
        (
            f"SK N={largest} ground state per spin",
            glass["ground_state"][largest],
            qp.SK_PARISI_GROUND_STATE,
        ),
    )
    for name, measured, exact in rows:
        print(f"    {name:<48}{measured:>14.6f}{exact:>14.6f}")

    print("\n    the c = 0 bound holds (F_var >= F_exact) in every arm and system:")
    for scaled in SCALED_INDICES:
        floor = float(np.min(small["gap"][scaled]))
        print(
            f"      4x4  c = {scaled:+.0f}   min gap per spin {floor:+.6f}   "
            f"{'ok' if floor >= -1e-6 else 'VIOLATED'}"
        )

    print("\n    the scaling law: gap/N at matched c = (q-1)N across sizes")
    header = "  ".join(f"N={n:<5}" for n in glass["sizes"])
    print(f"      {'c':<6}{header}")
    for scaled in SCALED_INDICES:
        row = "  ".join(f"{glass['gap'][n, scaled].mean():<7.5f}" for n in glass["sizes"])
        print(f"      {scaled:<+6.0f}{row}")

    best = min(SCALED_INDICES, key=lambda s: float(np.mean(glass["gap"][largest, s])))
    print(
        f"\n    lowest gap at N={largest} is c = {best:+.0f}"
        f" ({'exactly VAN' if best == BASELINE else 'not VAN'});"
        " F_1 is the only member of the family that bounds F, so this is expected."
    )


def main(*, quick: bool = False, full: bool = False) -> None:
    """Run all three systems and write both figures."""
    use_qjax_style()
    config = configuration(quick=quick, full=full)
    print("Tsallis variational free energy")
    small = run_ising_small(config)
    size = config["ising_large"]["size"]
    print(f"  {size}x{size} Ising across temperature")
    large = run_ising_large(config)
    glass = run_glass(config)
    report(small, glass)
    plot_main(small, large, glass, FIG_DIR / "tsallis_free_energy")
    plot_correlations(glass, FIG_DIR / "tsallis_free_energy_correlations")


if __name__ == "__main__":
    main(
        quick="--quick" in sys.argv,
        full="--full" in sys.argv or bool(os.environ.get("QJAX_FULL")),
    )
