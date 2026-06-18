# Derivative-free optimization

> A `q`-exponential-weighted search whose heavy tails (`q > 1`) keep mass on
> far-from-best candidates and escape deceptive local minima.

## What it shows

A cross-entropy-method (CEM) optimizer maintains a Gaussian search distribution,
draws a population of candidates, and refits the distribution to the lowest-cost
ones. Instead of a hard elite cut-off, each candidate $x_i$ is softly weighted by
a `q`-exponential of its cost,

$$
w_i \;\propto\; \exp_q\!\Big(-\frac{c(x_i) - c_{\min}}{T}\Big),
\qquad \exp_q(u) = \big[1 + (1-q)\,u\big]_+^{1/(1-q)},
$$

and the new mean and variance are the weighted moments of the sample.

At $q = 1$ these are the ordinary **Boltzmann** weights $e^{-c/T}$, which decay
exponentially and concentrate greedily on the current best — so the search tends
to collapse into whichever basin it starts near. For $q > 1$ the weights acquire
heavy, power-law tails that retain probability mass on far-from-best candidates,
sustaining exploration. On a deceptive 2-D landscape — a broad global well plus a
shallower decoy near the initialization — greedy $q = 1$ (BGS) falls into the
decoy, while heavy-tailed $q = 2.5$ (Tsallis) keeps exploring and reaches the
global optimum.

## How it works

The `q`-deformed weighting is a one-line change to standard CEM:

```python
import jax, jax.numpy as jnp
import qjax

samples = mu + sigma * jax.random.normal(key, (pop, 2))
costs = objective(samples)

# q-exponential weights (q = 1 -> Boltzmann; q > 1 -> heavy tails)
weights = qjax.q_exp(-(costs - costs.min()) / temperature, q)
weights = weights / jnp.sum(weights)

# refit the search distribution to the reweighted population
mu = jnp.sum(weights[:, None] * samples, axis=0)
sigma = jnp.sqrt(jnp.sum(weights[:, None] * (samples - mu) ** 2, axis=0)) + 1e-3
```

## Result

```{figure} /_static/examples/optimization.png
:alt: 2-D deceptive cost landscape with q=1 and q=2.5 search paths
:width: 100%

Left: the cost landscape with the live population and each method's mean path (the
global minimum marked ★, the decoy ✗). Right: the same surface in 3-D with the
paths traced on it. Greedy `q = 1` (BGS) settles in the decoy well, while
heavy-tailed `q = 2.5` (Tsallis) escapes and reaches the global optimum.
```

## Takeaways

Swapping `exp` for {func}`~qjax.q_exp` turns the temperature into a *tail-weight*
knob: heavier tails ($q > 1$) trade a little greediness for robustness to
deceptive local minima, at no extra cost to the algorithm. The same
`q`-exponential is the building block of the [`q`-Gaussian](q_gaussian.md) density
and of the `q`-deformed activations used elsewhere in the library.
