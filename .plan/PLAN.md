# qjax — Implementation Plan

> **qjax** is a research-oriented Python library built on [JAX](https://github.com/google/jax)
> whose mission is to spread the research and capabilities of **Tsallis statistics**
> (non-extensive statistical mechanics) in **artificial intelligence**.

## 1. Vision

Tsallis statistics generalizes Boltzmann–Gibbs statistics through a single
*entropic index* `q`. As `q → 1` every construction collapses back to its classical
counterpart (Shannon entropy, Gaussian, softmax, KL divergence, …). `qjax` exposes
these `q`-deformed primitives as **pure, differentiable, `jit`/`vmap`-friendly JAX
functions** so that researchers can:

- treat `q` as a *learnable parameter* and let models discover their own statistics;
- explore heavy-tailed / sparse alternatives to standard AI building blocks;
- reproduce and extend Tsallis-statistics results across ML domains.

## 2. Design principles

1. **Purity & composability** — every primitive is a pure function of `(x, q)`;
   no hidden state. Everything works under `jax.jit`, `jax.grad`, `jax.vmap`.
2. **Numerically safe `q → 1` limit** — use the double-`where` trick so gradients
   are finite exactly at `q = 1`.
3. **Single source of truth** — `core/` holds the math; `plots/`, `examples/`,
   and `tests/` only consume the public API.
4. **Precise docstrings** — every public function documents its math (with the
   defining formula and the `q → 1` limit), args, shapes, and returns.
5. **Consistent visuals** — all plots use the **`magma`** colormap via a single
   style module.

## 3. Target package structure

```
qjax/
├── __init__.py            # curated public API
├── core/
│   ├── __init__.py        # re-exports core primitives
│   ├── functions.py       # q_log, q_exp + q-algebra (q_add, q_diff, q_prod, q_div)
│   ├── entropy.py         # tsallis_entropy, tsallis_cross_entropy, tsallis_divergence
│   ├── distributions.py   # q-Gaussian: pdf, logpdf, sample, normalization C_q
│   └── activations.py     # tsallis_entmax (q-softmax / sparsemax family)
├── shared/
│   ├── __init__.py
│   ├── types.py           # Array / Scalar type aliases
│   └── validation.py      # q-range checks, broadcasting helpers
└── plots/
    ├── __init__.py
    ├── style.py           # magma palette + rcParams + qcolors(n)
    ├── functions.py       # plot q-log / q-exp families
    └── distributions.py   # plot q-Gaussian family
examples/                  # runnable scripts (save figures with magma palette)
tests/                     # pytest suite (limits, gradients, jit/vmap, shapes)
docs/                      # theory note + API overview
pyproject.toml             # uv-managed project (jax, matplotlib, pytest, ruff)
```

## 4. Mathematical scope (core)

| Primitive | Formula | `q → 1` limit |
|-----------|---------|---------------|
| `q_log(x, q)` | `(x**(1-q) - 1) / (1-q)` | `log(x)` |
| `q_exp(x, q)` | `[1 + (1-q)x]_+ ** (1/(1-q))` | `exp(x)` |
| `q_add(a, b, q)` | `a + b + (1-q) a b` | `a + b` |
| `tsallis_entropy(p, q)` | `(1 - Σ p**q) / (q-1)` | `-Σ p log p` |
| `tsallis_cross_entropy(p, y, q)` | `-Σ y · q_log(p, q)` | `-Σ y log p` |
| `tsallis_divergence(p, r, q)` | `(Σ p**q r**(1-q) - 1)/(q-1)` | `KL(p‖r)` |
| `q_gaussian_pdf(x, q, β)` | `√β/C_q · q_exp(-β x², q)` | `√(β/π)·exp(-βx²)` |
| `tsallis_entmax(z, q)` | argmax over simplex of `⟨p,z⟩ + Sᵀ_q(p)` | `softmax(z)` |

q-Gaussian sampling uses the **generalized Box–Muller** transform (valid for `q < 3`).

## 5. Enhancements over the bare skeleton

- Add `pyproject.toml` (uv) — the project currently has no build/dependency config.
- Add `shared/validation.py` for consistent `q` handling and clear errors.
- Centralize plot styling on **magma** (`plots/style.py`) instead of ad-hoc colors.
- Fill the six example stubs with runnable, figure-producing demos.
- Add a real pytest suite covering `q → 1` limits, gradient finiteness, `jit`/`vmap`.
- Add a concise theory note in `docs/`.

## 6. Checklist

### Project setup
- [x] `pyproject.toml` for uv (jax, matplotlib, pytest, ruff) + tool config
- [x] `README.md` with install + quickstart
- [x] `docs/theory.md` — concise Tsallis-statistics primer

### Core — math
- [x] `core/functions.py` — `q_log`, `q_exp`, `q_add`, `q_diff`, `q_prod`, `q_div`
- [x] `core/entropy.py` — `tsallis_entropy`, `tsallis_cross_entropy`, `tsallis_divergence`
- [x] `core/distributions.py` — q-Gaussian `pdf`, `logpdf`, `sample`, `normalization`
- [x] `core/activations.py` — `tsallis_entmax` (q-softmax / sparsemax family)
- [x] `core/__init__.py` — re-export core primitives

### Shared
- [x] `shared/types.py` — `Array`, `Scalar` aliases
- [x] `shared/validation.py` — `as_scalar_q`, range checks
- [x] `shared/__init__.py` — re-exports

### Plots (magma palette)
- [x] `plots/style.py` — `use_qjax_style()`, `qcolors(n)`, magma default
- [x] `plots/functions.py` — `plot_q_log`, `plot_q_exp`
- [x] `plots/distributions.py` — `plot_q_gaussian`
- [x] `plots/__init__.py` — re-exports

### Public API
- [x] `qjax/__init__.py` — curated exports + `__version__`

### Examples (runnable, save figures)
- [x] `examples/q_gaussian.py` — q-Gaussian family figure
- [x] `examples/learnable_q.py` — fit `q` by gradient descent to data
- [x] `examples/classification.py` — Tsallis cross-entropy classifier
- [x] `examples/optimization.py` — q-exp annealing / q-deformed objective
- [x] `examples/reinforcement_learning.py` — Tsallis-entmax policy on bandit
- [x] `examples/graph_learning.py` — Tsallis-entropy node attention on a graph

### Tests
- [x] `tests/test_functions.py` — limits, inverses, gradients, jit/vmap
- [x] `tests/test_entropy.py` — Shannon/KL limits, non-negativity
- [x] `tests/test_distributions.py` — normalization, sampling variance
- [x] `tests/test_activations.py` — simplex, softmax/sparsemax limits

### Verification
- [x] `uv run pytest` passes (69 tests)
- [x] `uv run ruff check` passes on `qjax`, `tests`, `examples`
- [x] Run every example end-to-end (6 figures saved to `examples/figures/`)

## 7. Notes from execution

- **Sampler correctness.** The first sampler used a generalized Box–Muller with an
  incorrect scale (variance was ~20% off). Replaced with the exact Student-``t``
  construction ``X = Z / sqrt(W/ν) / sqrt((3-q)β)``, which reproduces the family
  variance ``1/((5-3q)β)`` exactly. Supported range is ``1 <= q < 3``.
- **Gradient bug in `normalization`.** The piecewise ``C_q`` closures captured the
  raw ``q`` instead of their sanitized argument, so the unused branch evaluated
  ``log(1-q)`` with ``q > 1`` and poisoned the gradient via ``0 * NaN``. Fixed and
  covered by `test_normalization_gradient_finite`. This is why `learnable_q` now
  converges (recovers ``q ≈ 1.62`` from ``q_true = 1.6``).
- **`logpdf`.** Must use the ordinary ``log`` of the density, not ``q_log`` (which
  would collapse to ``-βx²`` and drop the normalization curvature).
- All plots default to the **`magma`** colormap via `qjax.plots.use_qjax_style`.

## 8. Publication-grade revision

- **Vector PDF output.** `use_qjax_style` now configures a research-grade style
  (serif body + Computer-Modern math via `mathtext`, embedded fonts with
  `pdf.fonttype=42`, in-pointing major/minor ticks, top/right spines off, magma
  color cycle). Added `qjax.plots.save_figure`, and every example saves a tight
  vector **PDF** to `examples/figures/`.
- **Example rename.** `graph_learning.py` → `attention_mlp.py` (Tsallis-entmax
  sparse attention).
- **Classification redesigned as a baseline study.** Now compares the **Shannon
  baseline** (`q=1`, standard cross-entropy) against robust **Tsallis** losses
  (`q=0.7`, `q=0.4`) across increasing label-noise levels, averaged over seeds on
  a clean test set. Key correction: the *robust* Tsallis cross-entropy regime is
  ``q < 1`` (where ``-ln_q`` is bounded — the generalized cross-entropy), not
  ``q > 1``; all methods share softmax outputs and differ only in the loss `q`.
  Result: clean-test accuracy degrades far more gracefully for `q<1`
  (`q=0.4`: 0.84→0.68) than for Shannon (`q=1`: 0.82→0.59) as `η` rises to 0.5.

## 9. Documentation site

- **Sphinx + Furo.** `docs/` is a full MyST-Markdown documentation site:
  `index`, `installation`, `quickstart`, `theory`, `examples`, and an autodoc
  `api` reference (Napoleon Google-style docstrings, viewcode, intersphinx to
  python/numpy/jax/matplotlib). Math via MyST `dollarmath` + MathJax; magma
  accent colors matching the plot palette.
- Added a `docs` optional-dependency group (`sphinx`, `furo`, `myst-parser`) and
  a `docs/Makefile`. Build: `uv run sphinx-build -b html docs docs/_build/html`
  (clean under `-W`, warnings-as-errors).
