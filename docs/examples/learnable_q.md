# Fitting q by maximum likelihood

> Recover a hidden generating `q` by gradient descent on the `q`-Gaussian
> log-likelihood — the simplest demonstration that `q` is differentiable.

## What it shows

We draw 20,000 samples from a `q`-Gaussian with a hidden $q_\text{true} = 1.6$ and
$\beta_\text{true} = 0.8$, then recover both parameters by maximum likelihood —
minimizing the negative log-likelihood

$$
\mathcal{L}(q, \beta) = -\frac{1}{N} \sum_{i=1}^{N} \log \mathcal{G}_q(x_i)
$$

with plain gradient descent. Because the gradients $\partial \mathcal{L} /
\partial q$ and $\partial \mathcal{L} / \partial \beta$ are well-defined
everywhere (including at $q = 1$), the optimizer recovers the generating `q` —
concrete evidence that the entropic index is an ordinary learnable parameter, not
a fixed hyperparameter.

## How it works

The only `qjax` call is {func}`~qjax.q_gaussian_logpdf`. The parameters are
optimized in an unconstrained space and mapped into valid ranges (`beta > 0`,
`q ∈ (1, 3)`) so the gradients stay well-defined:

```python
import jax, jax.numpy as jnp
import qjax

def neg_log_likelihood(params, x):
    beta = jax.nn.softplus(params["beta_raw"]) + 1e-3      # beta > 0
    q = 1.0 + 2.0 * jax.nn.sigmoid(params["q_raw"])        # q in (1, 3)
    return -jnp.mean(qjax.q_gaussian_logpdf(x, q, beta))

loss_and_grad = jax.jit(jax.value_and_grad(neg_log_likelihood))

params = {"q_raw": jnp.array(0.0), "beta_raw": jnp.array(0.0)}
for _ in range(400):
    loss, grads = loss_and_grad(params, data)
    params = {k: v - 0.05 * grads[k] for k, v in params.items()}
```

## Result

```{figure} /_static/examples/learnable_q.png
:alt: optimization loss curve and recovered q approaching the true value
:width: 100%

Left: the negative log-likelihood decreasing during optimization. Right: the
estimate `q̂` converging to the true generating `q = 1.6` (dashed line).
```

## Takeaways

Since {func}`~qjax.q_gaussian_logpdf` is differentiable in `q`, maximum-likelihood
estimation of the entropic index is just gradient descent — and the hidden `q` is
recovered accurately, validating `q`-as-a-parameter. The same idea scales to full
models in the [classification](classification.md),
[reinforcement-learning](reinforcement_learning.md) and
[attention](attention_mlp.md) examples, where `q` is learned *jointly* with the
network rather than fitted on its own.
