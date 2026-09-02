# Examples

Every example in the `examples/` directory is a single, runnable script that
trains or evaluates on synthetic data — or on Monte Carlo samples from a
Hamiltonian — and saves a publication-grade, brand-themed figure. Run any of them
with:

```bash
uv run python examples/<name>.py
```

### Three tiers

The five statistical-physics examples are sized in three tiers, and only counts
change between them — never the arms, never the physics — so a claim measured at
one tier means the same at another.

| tier | how | scale | wall clock (laptop CPU) |
| --- | --- | --- | --- |
| reduced | `main(quick=True)` | smallest sizes, one seed | ~10–20 s per script |
| default | `uv run python examples/<name>.py` | as documented on each page | 1–6 min per script |
| full | `--full`, or `QJAX_FULL=1` | large lattices, many seeds | GPU territory |

The reduced tier is what `tests/test_examples_physics.py` runs, so every one of
these scripts is executed end to end by the test suite. Measured default-tier
times: `anomalous_diffusion` ≈ 1.5 min, `ising_phases` ≈ 1 min,
`pinn_fokker_planck` ≈ 3 min, `tsallis_free_energy` ≈ 3.5 min,
`generalized_annealing` ≈ 6 min. Regenerating every figure in the documentation
(`python scripts/build_figures.py`) runs all thirteen examples and takes about
25 minutes.

## Statistical physics

Each of these is measured against something exact — Onsager's closed forms, a
transfer matrix, exhaustive enumeration of the whole state space, tabulated
cluster minima, or a scaling relation — so the claims are verifiable rather than
illustrative, and the negative results are reported as such.

<div class="site-grid" markdown>

<div class="site-card" markdown>
### [Ising phases and `T_c`](examples/ising_phases.md)
Finite-size crossover generates *physical* label noise. `T_c` to 0.6 %, `ν` and
`β` to 3 %, and the bounded loss helps in proportion to the measured noise.
</div>

<div class="site-card" markdown>
### [Variational free energy at index `q`](examples/tsallis_free_energy.md)
The nonextensive version of variational autoregressive networks. `q = 1` is
optimal — necessarily — and the deformation's effect collapses in `(q-1)N`.
</div>

<div class="site-card" markdown>
### [Generalized simulated annealing](examples/generalized_annealing.md)
Tsallis & Stariolo in two `qjax` calls, with a schedule whose `q → 1` limit is
exact. On Lennard-Jones clusters the deformation does not pay, and why.
</div>

<div class="site-card" markdown>
### [Anomalous diffusion](examples/anomalous_diffusion.md)
`q` as a *measured* quantity with an error bar: two independent estimators, an
exactly known target, and Lutz's cold-atom law.
</div>

<div class="site-card" markdown>
### [Heavy-tailed PINN residuals](examples/pinn_fokker_planck.md)
An ICML 2026 Student-t residual model *is* a `q`-Gaussian likelihood. The
residuals are heavier than Cauchy — and the robust loss still backfires.
</div>

</div>

## Machine learning

<div class="site-grid" markdown>

<div class="site-card" markdown>
### [The q-Gaussian family](examples/q_gaussian.md)
Compact support (`q < 1`), Gaussian (`q = 1`) and heavy tails (`1 < q < 3`),
with samples overlaid on the analytic density.
</div>

<div class="site-card" markdown>
### [Fitting q by maximum likelihood](examples/learnable_q.md)
Recovers a hidden generating `q` by gradient descent on the `q`-Gaussian
log-likelihood — `q` is just a differentiable parameter.
</div>

<div class="site-card" markdown>
### [Derivative-free optimization](examples/optimization.md)
An animated `q`-exponential-weighted search (contour + 3-D surface) whose
heavy tails (`q > 1`) escape a decoy minimum that traps greedy `q = 1`.
</div>

<div class="site-card" markdown>
### [Label-noise robustness](examples/classification.md)
Bounded Tsallis cross-entropy (`q < 1`) vs. the Shannon baseline, and a
*learnable* `q` that discovers the robust regime on its own.
</div>

<div class="site-card" markdown>
### [Node classification under noise](examples/node_classification.md)
A GCN with *learnable* Tsallis `q` stays robust to noisy training labels,
while the Shannon baseline propagates the errors across the graph.
</div>

<div class="site-card" markdown>
### [Exploration on a bandit](examples/reinforcement_learning.md)
A `tsallis_entmax` policy whose *learnable* `q` anneals exploration into
exploitation for the lowest cumulative regret.
</div>

<div class="site-card" markdown>
### [Sparse self-attention](examples/attention_mlp.md)
Attention pooling with `tsallis_entmax`; a *learnable* `q` recovers sparse,
signal-focused attention as distractors grow.
</div>

<div class="site-card" markdown>
### [Learning `q` in attention](examples/attention_q_learning.md)
Animated: watch the attention `q` being learned — as `q` rises toward
sparsemax, the attention map sharpens onto the informative tokens.
</div>

</div>

## Four roles for `q`

### `q` as a learnable parameter

Seven of the thirteen examples make `q` itself trainable, and the headline result is
consistent: **gradient descent reliably discovers a useful entropic index**, with
no grid search.

- [**Classification**](examples/classification.md) — the learned loss `q` settles
  in the robust regime (`q ≈ 0.3`) and matches the best hand-tuned fixed `q` at
  every noise level.
- [**Node classification**](examples/node_classification.md) — on a graph, the
  learned GCN loss `q` settles in the robust regime and stays accurate as label
  noise the Shannon baseline amplifies grows.
- [**Attention**](examples/attention_mlp.md) — the learned attention `q`
  converges near sparsemax (`q ≈ 2.0`), zeroing out distractor tokens.
- [**Reinforcement learning**](examples/reinforcement_learning.md) — the learned
  policy `q` *rises* over training, annealing exploration into exploitation.
- [**Ising phases**](examples/ising_phases.md) — the learned `q` is the arm that
  survives a change of *physical* regime, best where the finite-size label noise
  is large and least penalized where it is small.

### `q` as a measured physical quantity

In [**anomalous diffusion**](examples/anomalous_diffusion.md), `q` is not chosen
at all: it is inferred from trajectories by maximum likelihood, with a Fisher
error bar, and checked against a value known exactly in advance — twice over, by
the density and by the mean-squared-displacement exponent, tied together by
`α = 2/(3-q)`. All three cold-atom indices land within one sigma of theory.

### `q` as a residual model

In [**the PINN**](examples/pinn_fokker_planck.md) two indices appear at once: the
equation's, fixed by the physics, and the *residual model's*, learned. The second
turns a state-of-the-art Student-t construction into a `qjax` one-liner — and then
shows that a robust residual loss and a forward PDE are a bad match.

### `q` as a control parameter

In [**generalized annealing**](examples/generalized_annealing.md) and
[**variational free energy**](examples/tsallis_free_energy.md), `q` is *chosen* or
*scheduled*, and both pages say why fitting it there would be meaningless:
minimizing a variational free energy over its own entropic index simply runs to
whichever `q` makes the entropy term largest. These are also the two examples
where the honest answer is that the deformation does not help — and the value of
having exact references is precisely that this can be stated with numbers.
