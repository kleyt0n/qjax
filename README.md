<div align="center">

<img src="https://raw.githubusercontent.com/Kleyt0n/qjax/main/assets/logo_qjax.svg" alt="qjax logo" width="340"/>

#

**Tsallis statistics for artificial intelligence, built on [JAX](https://github.com/jax-ml/jax).**

[![PyPI](https://img.shields.io/pypi/v/qjax.svg?style=flat-square&color=1a759f)](https://pypi.org/project/qjax/)
[![Docs](https://img.shields.io/badge/docs-kleyt0n.github.io%2Fqjax-1a759f.svg?style=flat-square)](https://kleyt0n.github.io/qjax/)
[![Python](https://img.shields.io/badge/python-3.10%2B-3776ab.svg?style=flat-square)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-1e6091.svg?style=flat-square)](LICENSE)
[![Built on JAX](https://img.shields.io/badge/built%20on-JAX-184e77.svg?style=flat-square)](https://github.com/jax-ml/jax)
[![Code style: Ruff](https://img.shields.io/badge/code%20style-ruff-261230.svg?style=flat-square)](https://github.com/astral-sh/ruff)

[Quickstart](#quickstart) • [Building blocks](#building-blocks) • [Example](#label-noise-robustness) • [Installation](#installation)

</div>

## What is qjax?

Tsallis (non-extensive) statistics generalizes Boltzmann–Gibbs–Shannon statistics through a single *entropic index* $q$. As $q \to 1$ every construction collapses back to its classical counterpart — Shannon entropy, the Gaussian, softmax, the Kullback–Leibler divergence — while $q \neq 1$ opens up heavy tails, sparse attention, and tunable exploration.

`qjax` exposes these $q$-deformed primitives as pure, differentiable, `jit`/`vmap`-friendly JAX functions. Because $q$ is just another argument, you can hold it fixed *or* learn it end-to-end by gradient descent.

```python
import jax, jax.numpy as jnp
import qjax

# q-deformed functions (recover log / exp as q -> 1)
qjax.q_log(2.0, q=1.5)
qjax.q_exp(1.0, q=1.5)

# Tsallis information measures
p = jnp.array([0.5, 0.3, 0.2])
qjax.tsallis_entropy(p, q=2.0)         # -> Shannon entropy as q -> 1
qjax.tsallis_divergence(p, p, q=2.0)   # -> KL divergence as q -> 1

# q-Gaussian distribution (heavy-tailed for 1 < q < 3)
x = jnp.linspace(-4, 4, 100)
qjax.q_gaussian_pdf(x, q=1.5, beta=1.0)
samples = qjax.sample(jax.random.PRNGKey(0), q=1.5, beta=1.0, shape=(1000,))

# Sparse softmax: q=1 -> softmax, q=2 -> sparsemax (exact zeros)
qjax.tsallis_entmax(jnp.array([2.0, 1.0, -1.0]), q=2.0)
```

## Contents

- [Building blocks](#building-blocks)
- [A learnable `q`](#a-learnable-q)
- [Label-noise robustness](#label-noise-robustness)
- [Statistical physics](#statistical-physics)
- [Installation](#installation)
- [Contributing](#contributing)
- [License](#license)

## Building blocks

`qjax` is organized as a small set of composable, fully differentiable primitives. Each is a pure function of $(x, q)$. Every primitive is a single closed form in the entropic index $q$, and each recovers its Boltzmann–Gibbs–Shannon counterpart in the $q \to 1$ limit:

| `qjax` | Definition | Limit $q \to 1$ |
| --- | --- | --- |
| `q_log` | $\ln_q x = \dfrac{x^{1-q} - 1}{1 - q}$ | $\ln x$ |
| `q_exp` | $\exp_q x = \big[1 + (1-q)\,x\big]_+^{\frac{1}{1-q}}$ | $e^{x}$ |
| `tsallis_entropy` | $S_q(p) = \dfrac{1 - \sum_i p_i^{\,q}}{q - 1}$ | $-\sum_i p_i \ln p_i$ |
| `tsallis_cross_entropy` | $H_q(y, p) = -\sum_i y_i \ln_q p_i$ | $-\sum_i y_i \ln p_i$ |
| `tsallis_divergence` | $D_q(p \,\Vert\, r) = \dfrac{\sum_i p_i^{\,q}\, r_i^{\,1-q} - 1}{q - 1}$ | $\mathrm{KL}(p \,\Vert\, r)$ |
| `q_gaussian_pdf` | $\mathcal{G}_q(x) = \dfrac{\sqrt{\beta}}{C_q}\,\exp_q(-\beta x^2)$ | $\sqrt{\tfrac{\beta}{\pi}}\,e^{-\beta x^2}$ |
| `tsallis_entmax` | $entmax_q(z) = \arg\max_{p \in \Delta}\,\langle p, z\rangle + S_q(p)$ | $softmax(z)$ |

where $[\,\cdot\,]_+ = \max(\cdot, 0)$ is the Tsallis cut-off, $C_q$ the $q$-Gaussian normalization, and $\Delta$ the probability simplex
(`tsallis_entmax` is exactly **sparsemax** at $q = 2$).

> `qjax` is a research library. The numerics are tested across the $q \to 1$
> limit, gradients, and `jit`/`vmap`, but the API may still evolve.

### Deformed functions and $q$-algebra

`q_log` and `q_exp` are inverse deformations of `log`/`exp`; the accompanying $q$-algebra turns them into homomorphisms (`q_log(a·b) = q_add(q_log a, q_log b)`).

```python
qjax.q_log(x, q=1.5)                                   # (x**(1-q) - 1) / (1-q)
qjax.q_add(qjax.q_log(2.0, 1.4), qjax.q_log(3.0, 1.4), 1.4)   # == q_log(6.0, 1.4)
```

### Information measures

```python
p = jnp.array([0.5, 0.3, 0.2])
r = jnp.array([0.25, 0.25, 0.5])

qjax.tsallis_entropy(p, q=2.0)           # -> Shannon entropy as q -> 1
qjax.tsallis_cross_entropy(p, r, q=2.0)  # q-deformed cross-entropy loss
qjax.tsallis_divergence(p, r, q=2.0)     # -> KL(p || r) as q -> 1
```

### The $q$-Gaussian

A maximum-Tsallis-entropy distribution: heavy-tailed (Student-$t$) for $1 < q < 3$, compactly supported for $q < 1$, Gaussian at $q = 1$.

```python
x = jnp.linspace(-4, 4, 100)
qjax.q_gaussian_pdf(x, q=1.5, beta=1.0)
qjax.q_gaussian_logpdf(x, q=1.5, beta=1.0)
qjax.sample(jax.random.PRNGKey(0), q=1.5, beta=1.0, shape=(1000,))
```

### Sparse activations

`tsallis_entmax` spans the whole family: $q = 1$ is softmax, $q = 2$ is sparsemax
(exact zeros), larger $q$ is sparser still, and $q < 1$ is *denser* than softmax.
A drop-in for sparse attention, with exact gradients w.r.t. both $z$ and $q$.

```python
z = jnp.array([2.0, 1.0, 0.1, -1.0])
qjax.tsallis_entmax(z, q=0.5)   # denser than softmax
qjax.tsallis_entmax(z, q=1.0)   # softmax
qjax.tsallis_entmax(z, q=2.0)   # sparsemax (exact zeros)
```

## A learnable $q$

Because $q$ is an ordinary differentiable argument, it is finite everywhere — including the $q = 1$ limit — so it can be optimized like any other parameter:

```python
import jax

x = jnp.linspace(-3, 3, 200)
nll = lambda q: -jnp.mean(qjax.q_gaussian_logpdf(x, q, 1.0))
grad_q = jax.grad(nll)(1.5)     # well-defined gradient w.r.t. the entropic index
```

This is what makes $q$ more than a hyperparameter: the right amount of non-extensivity can be *discovered* from data.

## Label-noise robustness

When training labels are noisy, ordinary softmax **cross-entropy** is unbounded — a confidently mislabeled example incurs an arbitrarily large loss, so an over-parameterized network ends up *memorizing* the noise. Replacing the logarithm with the deformed $q$-logarithm gives the **Tsallis cross-entropy**, which is *bounded* for $q < 1$: its gradient saturates on unfittable points, so the model ignores label noise instead of fitting it.

For a one-hot target with true class $c$ and softmax probabilities $p$,

$$\mathcal{L}_q(p, c) = -\ln_q p_c = \frac{1 - p_c^{\,1-q}}{1 - q}, \qquad \ln_q x = \frac{x^{1-q} - 1}{1 - q}.$$

As $q \to 1$ this is exactly the standard cross-entropy $-\log p_c$; for $q < 1$ the per-example loss is bounded above by $1/(1-q)$, so mislabeled points cannot dominate the gradient.

The figure trains a small 3-class classifier on two shapes (blobs, spiral) from clean data up to 40% label noise, comparing the Boltzmann–Gibbs–Shannon baseline ($q = 1$) with Tsallis ($q = 0.3$). The comparison is fair — both share the same initialization, data, noisy labels and optimizer; only $q$ differs. Without noise the two match (≈98–99%); as noise grows the baseline carves spurious wrong-class islands while Tsallis keeps clean regions and higher accuracy.

<img src="https://raw.githubusercontent.com/Kleyt0n/qjax/main/docs/img/examples/classification_boundaries.png" alt="Decision boundaries for blobs and spiral across noise levels: Tsallis vs the Boltzmann-Gibbs-Shannon baseline" width="960"/>

See the [classification example](https://kleyt0n.github.io/qjax/examples/classification/) for the full setup.

## Statistical physics

`qjax.physics` pairs the ``q``-primitives with the systems they are meant to
describe, each with something *exact* to be checked against: Onsager's closed
forms for the 2-D Ising model, a transfer matrix, exhaustive enumeration of a
whole state space, the tabulated Lennard-Jones cluster minima, and the anomalous
diffusion scaling relation $\alpha = 2/(3-q)$.

```python
import qjax.physics as qp

qp.ISING_TC                      # 2 / ln(1 + sqrt 2) = 2.2691853, exact
configurations = qp.sample_ising(key, size=16, temperatures=grid,
                                 num_samples=80, sweeps=700)
qp.onsager_free_energy_per_site(2.5)          # exact, thermodynamic limit
qp.ising_transfer_matrix_log_z(8, 2.5)        # exact, finite lattice
qp.sk_exact_observables(couplings, 0.4)       # all 2**N states, streamed

# The Tsallis-Stariolo annealing schedule is a ratio of two q-logarithms, so its
# q -> 1 limit is the Geman-Geman log schedule *exactly*, with no branch on q.
qp.visiting_temperature(step, initial=0.9, q_visit=1.0)
```

Five examples build on it — and report negative results where that is what the
numbers say:

| example | the physics | measured against |
| --- | --- | --- |
| [Ising phases and $T_c$](https://kleyt0n.github.io/qjax/examples/ising_phases/) | finite-size crossover as *physical* label noise | Onsager $T_c$ (0.6 %), $\nu$ and $\beta$ (3 %) |
| [Variational free energy at $q$](https://kleyt0n.github.io/qjax/examples/tsallis_free_energy/) | nonextensive variational autoregressive networks | two independent exact codes; $(q-1)N$ collapse |
| [Generalized simulated annealing](https://kleyt0n.github.io/qjax/examples/generalized_annealing/) | Tsallis & Stariolo (1996) on LJ clusters | closed forms and the Cambridge Cluster Database |
| [Anomalous diffusion](https://kleyt0n.github.io/qjax/examples/anomalous_diffusion/) | $q$ as a *measured* quantity | exact stationary $q$; Lutz's cold-atom law |
| [Heavy-tailed PINN residuals](https://kleyt0n.github.io/qjax/examples/pinn_fokker_planck/) | an ICML 2026 Student-$t$ residual model, read as Tsallis | the closed-form solution; score correspondence to $10^{-15}$ |

<img src="https://raw.githubusercontent.com/Kleyt0n/qjax/main/docs/img/examples/ising_phases.png" alt="Machine learning the 2-D Ising transition: sampler validation against exact enumeration, the measured label noise, and the finite-size extrapolation of T_c" width="960"/>

## Neural-network building blocks

`qjax.nn` holds the pieces that every Tsallis model ends up needing. It is
framework-agnostic — plain arrays and pytrees — so it composes with Flax,
Equinox, Haiku, or hand-rolled JAX without pulling in any of them.

```python
from qjax.nn import bounded_q, entmax_attention, tsallis_cross_entropy_loss

# Keep a learnable q inside its valid range, whatever the optimizer does.
q = bounded_q(params["q_raw"], 1.0, 3.0)

# Attention normalized by entmax instead of softmax.
context, attn = entmax_attention(queries, keys, values, q=q)

# A q-deformed classification loss; q < 1 bounds the penalty on bad labels.
loss = tsallis_cross_entropy_loss(logits, targets, q=0.5, normalizer_q=1.0)
```

## Installation

`qjax` requires Python 3.10+ and depends only on `jax`. Plotting is optional. It is managed with [uv](https://docs.astral.sh/uv/).

| Use case            | Command                                              |
| ------------------- | ---------------------------------------------------- |
| As a dependency     | `uv add qjax`                                        |
| With plotting       | `uv add "qjax[plots]"`   (adds matplotlib)           |
| Development         | `uv sync --extra dev`    (tests + linter + mypy)     |
| Building the docs   | `uv sync --extra docs`   (Material for MkDocs)       |

For GPU/TPU acceleration, install the matching JAX build by following the [JAX installation guide](https://docs.jax.dev/en/latest/installation.html).

## Contributing

Contributions are welcome — new $q$-deformed primitives, examples, docs, and fixes. See [CONTRIBUTING.md](CONTRIBUTING.md) for the development setup, design principles (purity, the $q \to 1$ limit, finite gradients), and the checks CI runs.

## License

Released under the [MIT License](LICENSE).
