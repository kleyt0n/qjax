"""Tsallis-entmax exploration on a 10-armed bandit (REINFORCE).

The policy over arms is ``tsallis_entmax(preferences, q)``, trained by the
gradient-bandit / REINFORCE rule with a running-average baseline:

- ``q = 1`` is the classic **softmax (Boltzmann)** policy — always assigns every
  arm a non-zero probability, so it keeps paying to explore inferior arms.
- ``q > 1`` is **entmax / sparsemax** — once an arm is clearly inferior it is
  dropped to *exactly* zero probability, concentrating exploration on the
  contenders.
- **learnable ``q``**: ``q`` is itself a policy parameter, updated by the same
  REINFORCE gradient, so the agent *learns how sharp its exploration should be*.

We use the Sutton & Barto 10-armed testbed: each run draws arm means from
``N(0, 1)``, and results are averaged over many independent runs. We report the
three canonical bandit diagnostics — average reward, cumulative regret, and
%-optimal-action — plus the learned-``q`` curve.

Finding. A fixed, very sparse policy (``q = 2``) commits too early and locks
onto a sub-optimal arm (zeroed arms get no gradient and cannot recover). Softmax
(``q = 1``) explores reliably but keeps paying to sample inferior arms. The
**learnable ``q`` wins**: starting near full exploration it raises ``q`` as the
best arm emerges, annealing exploration into exploitation — achieving the lowest
cumulative regret while matching softmax's optimal-action rate.

Run with: ``uv run python examples/reinforcement_learning.py``
"""

from __future__ import annotations

from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

import qjax
from qjax.nn import bounded_q
from qjax.plots import qcolors, qlinestyles, save_figure, use_qjax_style

FIG_DIR = Path(__file__).parent / "figures"

K_ARMS = 10
NUM_RUNS = 300
STEPS = 1000
REWARD_NOISE = 0.3
LR_PREF = 0.1
LR_Q = 0.05
BASELINE_RATE = 0.05
POLICY_ITERS = 30  # entmax bisection steps

# Learnable-q parameterization: q = bounded_q(q_raw, Q_MIN, Q_MAX) in (1.05, 2.55).
# Start near full exploration (q ~ 1.1, almost softmax); REINFORCE then raises q,
# annealing exploration -> exploitation as the best arm becomes clear.
Q_MIN, Q_MAX, Q_RAW_INIT = 1.05, 2.55, -2.5  # init q ~ 1.16

# (label, q_fixed, is_learnable)
METHODS = (
    ("softmax (q = 1)", 1.0, False),
    ("entmax (q = 1.5)", 1.5, False),
    ("sparsemax (q = 2)", 2.0, False),
    ("learnable q", 0.0, True),
)


def resolve_q(q_raw, q_fixed, is_learnable: bool):
    """Return the policy entropic index: a constant, or the learned one."""
    if is_learnable:
        return bounded_q(q_raw, Q_MIN, Q_MAX)
    return q_fixed


def rollout(key: jax.Array, means: jnp.ndarray, q_fixed, is_learnable: bool):
    """Run one bandit episode; return per-step (reward, optimal?, regret, q)."""
    optimal_arm = jnp.argmax(means)
    best_mean = means[optimal_arm]

    def step(carry, k):
        prefs, q_raw, baseline = carry
        q = resolve_q(q_raw, q_fixed, is_learnable)
        k_arm, k_rew = jax.random.split(k)
        probs = qjax.tsallis_entmax(prefs, q=q, num_iters=POLICY_ITERS)
        arm = jax.random.choice(k_arm, K_ARMS, p=probs)
        reward = means[arm] + REWARD_NOISE * jax.random.normal(k_rew)

        def log_prob(prefs, q_raw):
            qq = resolve_q(q_raw, q_fixed, is_learnable)
            pr = jnp.clip(qjax.tsallis_entmax(prefs, q=qq, num_iters=POLICY_ITERS), 1e-9, 1.0)
            return jnp.log(pr[arm])

        g_prefs, g_qraw = jax.grad(log_prob, argnums=(0, 1))(prefs, q_raw)
        advantage = reward - baseline
        prefs = prefs + LR_PREF * advantage * g_prefs  # ascend expected reward
        if is_learnable:
            q_raw = q_raw + LR_Q * advantage * g_qraw
        baseline = baseline + BASELINE_RATE * (reward - baseline)

        info = (reward, (arm == optimal_arm).astype(jnp.float32), best_mean - means[arm], q)
        return (prefs, q_raw, baseline), info

    init = (jnp.zeros(K_ARMS), jnp.array(Q_RAW_INIT), jnp.array(0.0))
    _, infos = jax.lax.scan(step, init, jax.random.split(key, STEPS))
    return infos


@partial(jax.jit, static_argnames=("is_learnable",))
def run_method(key: jax.Array, q_fixed, is_learnable: bool):
    """Average ``rollout`` over ``NUM_RUNS`` independent 10-armed testbeds."""

    def one(run_key):
        k_means, k_play = jax.random.split(run_key)
        means = jax.random.normal(k_means, (K_ARMS,))
        return rollout(k_play, means, q_fixed, is_learnable)

    return jax.vmap(one)(jax.random.split(key, NUM_RUNS))  # each field (NUM_RUNS, STEPS)


def smooth(x: jnp.ndarray, window: int = 25) -> jnp.ndarray:
    """Moving-average smoothing for display."""
    return jnp.convolve(x, jnp.ones(window) / window, mode="valid")


def main() -> None:
    use_qjax_style()
    key = jax.random.PRNGKey(0)
    labels = [label for label, _, _ in METHODS]
    n = jnp.sqrt(NUM_RUNS)

    reward_curve, regret_curve, optimal_curve, q_curve = {}, {}, {}, {}
    for label, q_fixed, is_learnable in METHODS:
        reward, optimal, regret, q = run_method(key, jnp.float32(q_fixed), is_learnable)
        reward_curve[label] = (reward.mean(0), reward.std(0) / n)
        cum_regret = jnp.cumsum(regret, axis=1)
        regret_curve[label] = (cum_regret.mean(0), cum_regret.std(0) / n)
        optimal_curve[label] = optimal.mean(0)
        q_curve[label] = (q.mean(0), q.std(0) / n)
        print(
            f"{label:<20} final reward={reward[:, -100:].mean():.3f}  "
            f"total regret={cum_regret[:, -1].mean():6.1f}  "
            f"%opt={optimal[:, -100:].mean():.2f}  learned_q={q[:, -1].mean():.2f}"
        )

    # Average optimal arm value (same run keys as run_method) for the reference line.
    def best_of(run_key):
        k_means, _ = jax.random.split(run_key)
        return jnp.max(jax.random.normal(k_means, (K_ARMS,)))

    avg_best = float(jax.vmap(best_of)(jax.random.split(key, NUM_RUNS)).mean())

    # ---- Figure: 2x2 research panel ----
    fig, ((ax_r, ax_reg), (ax_opt, ax_q)) = plt.subplots(
        2, 2, figsize=(12.0, 8.6), layout="constrained"
    )
    colors = qcolors(len(METHODS))
    # The brand ramp is sequential, so four methods land on colors too close to
    # separate reliably (the two darkest differ by well under the readability
    # floor). Dash patterns carry the identity; color is the secondary cue. This
    # also keeps the panel readable in grayscale print.
    dashes = qlinestyles(len(METHODS))
    steps = jnp.arange(STEPS)
    sm_x = jnp.arange(24, STEPS)  # x for smoothed (window 25) curves

    for label, color, dash in zip(labels, colors, dashes, strict=False):
        mean, se = reward_curve[label]
        ax_r.plot(sm_x, smooth(mean), color=color, ls=dash, label=label)
        ax_r.fill_between(sm_x, smooth(mean - se), smooth(mean + se), color=color, alpha=0.15)

        rmean, rse = regret_curve[label]
        ax_reg.plot(steps, rmean, color=color, ls=dash, label=label)
        ax_reg.fill_between(steps, rmean - rse, rmean + rse, color=color, alpha=0.15)

        ax_opt.plot(sm_x, 100.0 * smooth(optimal_curve[label]), color=color, ls=dash, label=label)

    ax_r.axhline(avg_best, color="0.4", ls=":", lw=1.0, label="best arm")
    ax_r.set(xlabel="step", ylabel="average reward", title="(a) average reward")
    ax_r.legend(loc="lower right")
    ax_reg.set(xlabel="step", ylabel="cumulative regret", title="(b) cumulative regret")
    ax_opt.axhline(100.0, color="0.4", ls=":", lw=1.0)
    ax_opt.set(xlabel="step", ylabel="% optimal action", title="(c) optimal-action rate")
    ax_opt.set_ylim(0, 105)

    # (d) the learnable policy's q over training, vs the fixed references.
    qm, qse = q_curve["learnable q"]
    ax_q.plot(steps, qm, color=colors[-1], label="learned $q$")
    ax_q.fill_between(steps, qm - qse, qm + qse, color=colors[-1], alpha=0.2)
    for q_ref, name, color in zip((1.0, 1.5, 2.0), labels[:3], colors[:3], strict=False):
        ax_q.axhline(q_ref, color=color, ls=":", lw=1.0, label=name)
    ax_q.set(
        xlabel="step",
        ylabel="policy entropic index $q$",
        title="(d) the policy learns its own $q$",
    )
    ax_q.legend(loc="best", fontsize=8)

    fig.suptitle("Tsallis-entmax policies on the 10-armed testbed (averaged over runs)")
    print(f"saved {save_figure(fig, FIG_DIR / 'reinforcement_learning')}")


if __name__ == "__main__":
    main()
