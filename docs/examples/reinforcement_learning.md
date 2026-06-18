# Exploration on a bandit

> A `tsallis_entmax` policy on the 10-armed testbed, where a *learnable* `q`
> anneals exploration into exploitation and achieves the lowest regret.

## What it shows

The policy over arms maps a vector of preferences $z$ to action probabilities
with `tsallis_entmax`, trained by the gradient-bandit / REINFORCE rule on the
Sutton & Barto 10-armed testbed (averaged over many independent runs). The map
has the closed form

$$
\pi = \operatorname{entmax}_q(z), \qquad
\pi_i = \big[(q-1)\,z_i - \tau\big]_+^{1/(q-1)},
$$

where $\tau$ is set so the probabilities sum to one and the entropic index `q`
controls how the policy explores.

At $q = 1$ this is the classic **softmax (Boltzmann)** policy — every arm keeps a
non-zero probability, so the agent keeps paying to sample inferior arms. For
$q > 1$ it becomes **entmax / sparsemax**, dropping clearly inferior arms to
*exactly* zero probability; but a fixed, very sparse policy ($q = 2$) commits too
early and can lock onto a sub-optimal arm, since zeroed arms receive no gradient
and cannot recover. Making `q` itself a policy parameter — updated by the same
REINFORCE gradient — lets the agent *learn how sharp its exploration should be*
rather than fixing it in advance.

## How it works

The policy is one `tsallis_entmax` call; REINFORCE differentiates through it with
respect to both the preferences and `q`:

```python
import jax, jax.numpy as jnp
import qjax

probs = qjax.tsallis_entmax(prefs, q=q, num_iters=30)
arm = jax.random.choice(key, K_ARMS, p=probs)

def log_prob(prefs, q_raw):
    qq = resolve_q(q_raw, q_fixed, is_learnable)     # q = q_min + span*sigmoid(.)
    pr = jnp.clip(qjax.tsallis_entmax(prefs, q=qq, num_iters=30), 1e-9, 1.0)
    return jnp.log(pr[arm])

# REINFORCE: ascend (reward - baseline) * grad log pi, jointly for prefs and q
g_prefs, g_qraw = jax.grad(log_prob, argnums=(0, 1))(prefs, q_raw)
```

## Result

```{figure} /_static/examples/reinforcement_learning.png
:alt: average reward, cumulative regret, optimal-action rate, and learned q
:width: 100%

(a) Average reward, (b) cumulative regret, (c) %-optimal-action, and (d) the
policy's learned `q` over training. The learnable policy starts near full
exploration and *raises* `q` as the best arm emerges, achieving the lowest
cumulative regret while matching softmax's optimal-action rate.
```

## Takeaways

{func}`~qjax.tsallis_entmax` provides a one-parameter family of policies spanning
dense (softmax) to sparse (sparsemax) exploration. A fixed sparse policy commits
too early, whereas a learnable `q` **anneals** exploration into exploitation and
achieves the lowest cumulative regret. Because `tsallis_entmax` is differentiable
in `q`, this exploration schedule is *learned* by the same policy-gradient update,
not hand-designed.
