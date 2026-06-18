# Examples

Every example in the `examples/` directory is a single, runnable script that
trains/evaluates on synthetic data and saves a publication-grade, magma-themed
figure. Run any of them with:

```bash
uv run python examples/<name>.py
```

```{list-table}
:header-rows: 1
:widths: 32 68

* - Example
  - What it shows
* - [The q-Gaussian family](examples/q_gaussian.md)
  - Compact support (`q < 1`), Gaussian (`q = 1`) and heavy tails (`1 < q < 3`),
    with samples overlaid on the analytic density.
* - [Fitting q by maximum likelihood](examples/learnable_q.md)
  - Recovers a hidden generating `q` by gradient descent on the `q`-Gaussian
    log-likelihood — `q` is just a differentiable parameter.
* - [Derivative-free optimization](examples/optimization.md)
  - An animated `q`-exponential-weighted search (contour + 3-D surface) whose
    heavy tails (`q > 1`) escape a decoy minimum that traps greedy `q = 1`.
* - [Label-noise robustness](examples/classification.md)
  - Bounded Tsallis cross-entropy (`q < 1`) vs. the Shannon baseline, and a
    *learnable* `q` that discovers the robust regime on its own.
* - [Exploration on a bandit](examples/reinforcement_learning.md)
  - A `tsallis_entmax` policy whose *learnable* `q` anneals exploration into
    exploitation for the lowest cumulative regret.
* - [Sparse self-attention](examples/attention_mlp.md)
  - Attention pooling with `tsallis_entmax`; a *learnable* `q` recovers sparse,
    signal-focused attention as distractors grow.
* - [Learning `q` in attention](examples/attention_q_learning.md)
  - Animated: watch the attention `q` being learned — as `q` rises toward
    sparsemax, the attention map sharpens onto the informative tokens.
```

## The recurring theme: `q` as a learnable parameter

Four of the six examples make `q` itself trainable, and the headline result is
consistent: **gradient descent reliably discovers a useful entropic index**, with
no grid search.

- [**Classification**](examples/classification.md) — the learned loss `q` settles
  in the robust regime (`q ≈ 0.3`) and matches the best hand-tuned fixed `q` at
  every noise level.
- [**Attention**](examples/attention_mlp.md) — the learned attention `q`
  converges near sparsemax (`q ≈ 2.0`), zeroing out distractor tokens.
- [**Reinforcement learning**](examples/reinforcement_learning.md) — the learned
  policy `q` *rises* over training, annealing exploration into exploitation.

```{toctree}
:maxdepth: 1
:hidden:

examples/q_gaussian
examples/learnable_q
examples/optimization
examples/classification
examples/reinforcement_learning
examples/attention_mlp
examples/attention_q_learning
```
