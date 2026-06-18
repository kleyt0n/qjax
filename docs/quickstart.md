# Quickstart

Every `qjax` primitive is a pure function of `(x, q)` that is compatible with
{func}`jax.jit`, {func}`jax.grad`, and {func}`jax.vmap`, and recovers its
Boltzmann–Gibbs counterpart as `q → 1`.

## Deformed functions

```python
import jax.numpy as jnp
import qjax

x = jnp.linspace(0.1, 4.0, 5)
qjax.q_log(x, q=1.5)        # (x**(1-q) - 1) / (1-q);  -> log(x) as q -> 1
qjax.q_exp(qjax.q_log(x, 1.5), 1.5)  # exp_q is the inverse of ln_q
```

The `q`-algebra makes `ln_q` and `exp_q` homomorphisms:

```python
qjax.q_add(qjax.q_log(2.0, 1.4), qjax.q_log(3.0, 1.4), 1.4)  # == q_log(6.0, 1.4)
```

## Information measures

```python
p = jnp.array([0.5, 0.3, 0.2])
r = jnp.array([0.25, 0.25, 0.5])

qjax.tsallis_entropy(p, q=2.0)          # -> Shannon entropy as q -> 1
qjax.tsallis_divergence(p, r, q=2.0)    # -> KL(p || r) as q -> 1
qjax.tsallis_cross_entropy(p, r, q=2.0) # q-deformed cross-entropy loss
```

## The q-Gaussian

```python
import jax

x = jnp.linspace(-4, 4, 100)
qjax.q_gaussian_pdf(x, q=1.5, beta=1.0)       # heavy-tailed for 1 < q < 3
samples = qjax.sample(jax.random.PRNGKey(0), q=1.5, beta=1.0, shape=(1000,))
```

## Sparse softmax (entmax)

```python
z = jnp.array([2.0, 1.0, 0.1, -1.0])
qjax.tsallis_entmax(z, q=1.0)   # softmax (dense)
qjax.tsallis_entmax(z, q=2.0)   # sparsemax (exact zeros)
```

## A learnable q

Because `q` is just a differentiable argument, it can be optimized like any
other parameter:

```python
import jax

x = jnp.linspace(-3, 3, 200)
nll = lambda q: -jnp.mean(qjax.q_gaussian_logpdf(x, q, 1.0))
grad_q = jax.grad(nll)(1.5)     # finite everywhere, including q = 1
```

See the [examples](examples.md) for end-to-end demonstrations of a learnable
`q` in classification, attention, and reinforcement learning.
