# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - Unreleased

This release corrects a gradient bug in `tsallis_entmax` that affected every
model trained through it. Primal outputs for `q > 1` are unchanged to within
`4e-15`; **gradients change everywhere**, because they were previously wrong.

### Fixed

- **`tsallis_entmax` returned an incorrect gradient.** Autodiff differentiated
  *through* the bisection that locates the threshold `tau`, so `dtau/dz`
  collapsed to `(q-1) * onehot(argmax)` instead of the implicit-function value.
  The resulting Jacobian was not even symmetric, which it must be since entmax
  is the gradient map of a convex conjugate. For `z = [1, 2, 0.5]`, `q = 1.5`
  the maximum absolute error was `0.109` (~30% relative), and it did **not**
  shrink as `num_iters` grew. The solve is now wrapped in `stop_gradient` and
  the derivative supplied by a `jax.custom_jvp` implementing the exact rule
  `dp = T(s*dz + (h + s*z) dq/(q-1))`, with `s = p^(2-q)`, `h = -p log p`, and
  `T(v) = v - s sum(v)/sum(s)`. Verified against the closed form and central
  finite differences, and with `check_grads(order=2, modes=["fwd", "rev"])`.
- **`tsallis_entmax` silently returned near-one-hot output for `q < 1`.** The
  bisection bracket assumed `q > 1`. `entmax([1, 2, 3], q=0.5)` gave
  approximately `[0, 0, 1]` -- sparser than softmax, when `q < 1` must be
  *denser*. The bracket is now selected by mass property rather than numeric
  order and serves both regimes; `q < 1` correctly yields full support and
  entropy above softmax.
- **`d/dq` was identically zero in a `2e-6` window around `q = 1`** for `q_log`,
  `q_exp`, `tsallis_entropy`, and `tsallis_divergence`. The `|q-1| < Q_EPS`
  branch selects a `q`-independent classical expression, so a learnable entropic
  index that entered the window saw no gradient. These functions are now written
  through the entire functions `(e^t - 1)/t` and `log(1+t)/t`, so the classical
  limit emerges from the same expression with no branch on `q`.
- **Catastrophic cancellation near `q = 1` in float32**, the default precision.
  `q_log(2, 1.00001)` had a relative error of `3.85e-3`; it is now `2.88e-8`.
- **NaN gradients** from `q_prod`/`q_div` with a zero operand, from `q_div` with
  a zero denominator, and from `tsallis_divergence` with a zero in the reference
  distribution (the KL branch was guarded but the deformed branch was not).
  Values at these singularities are unchanged; only the gradients are tamed.
- **`q_exp` docstring contradicted its behaviour.** Past the Tsallis cut-off the
  result is `0` for `q < 1` but `+inf` for `q > 1`, not `0` in both cases.
- `mpl.cycler` replaced with a direct `cycler` import in `qjax.plots.style`.

### Added

- `scripts/build_figures.py`: regenerates every example figure and refreshes the
  rasters the documentation serves. The examples write PDFs and GIFs to
  `examples/figures/` while the docs embed PNGs from `docs/img/examples/`, and
  nothing connected the two -- so the palette change updated the PDFs and left
  the site showing the previous theme until this script was run.
- `qjax.nn`: `bounded_q`, `entmax_attention`, and `tsallis_cross_entropy_loss`,
  extracting the learnable-`q` reparameterization that was copy-pasted across
  six examples and the entmax attention block duplicated across three.
- `qjax.shared.validation.positive_q_or_nan`: rejects `q <= 0`, where the
  Tsallis normalizer `1/(q(q-1))` is singular. Raises `ValueError` for a static
  value; propagates `NaN` for a traced one, since a `raise` is impossible under
  `jit`.
- `qjax.shared.series`: the `expm1_over_t` / `log1p_over_t` helpers underlying
  the stable `q -> 1` limits.
- A Newton polish step after the `entmax` bisection, so `num_iters` barely
  affects accuracy (`|sum(p) - 1|` reaches machine epsilon even at
  `num_iters=10`).
- `py.typed`: the package was fully annotated but shipped as untyped.
- Test coverage: gradient-correctness tests (`check_grads`, Jacobian vs closed
  form, symmetry, `jacfwd == jacrev`), a float32 test leg, Hypothesis
  property tests for the algebraic identities, `qjax.plots` smoke tests, and a
  public-API surface test. 295 tests, 100% statement coverage.
- CI: `ruff format --check`, a `mypy` job, `uv lock --check`, macOS and Windows
  legs, coverage upload, an examples smoke test, and a job asserting the core
  imports without matplotlib.

### Changed

- **The plotting theme moved from `magma` to the qjax brand ramp.** The ten-step
  green-blue scale the logo and docs are built from is registered with Matplotlib
  as `"qjax"` (and `"qjax_r"`) and exported as `QJAX_RAMP`, so rebranding means
  swapping ten hex values. Two consequences, both measured rather than eyeballed:
  `qcolors` now windows the ramp to `[0.40, 1.0]`, because its three lightest
  steps sit between 1.3:1 and 1.7:1 against white and vanish as thin lines; and
  because the ramp is *sequential*, it cannot supply a categorical palette — an
  exhaustive search of all 1820 four-colour subsets found none that passes the
  readability checks. `qlinestyles` was added to carry identity for unordered
  categories, and `examples/reinforcement_learning.py` (the one figure that
  separated four methods by colour alone) now uses it.
- Off-ramp colours in the examples were retired: a leftover `magma` pink in
  `learnable_q.py`, an approximate teal in `classification.py`, and the cyan/lime
  optimizer paths in `optimization.py`, which would now blend into the green-blue
  contour surface. The paths are warm (amber/crimson) with a white stroke, since
  the ramp spans nearly the full lightness range and no single colour contrasts
  against all of it. The red misclassification marker in
  `node_classification.py` is a reserved status colour and stays off-ramp.

- **`matplotlib` is now an optional dependency.** Importing `qjax` for the
  mathematics no longer pulls it in. Install plotting with
  `pip install "qjax[plots]"`; `qjax.plots` raises a clear `ImportError`
  otherwise.
- `tsallis_entropy` and `tsallis_divergence` are evaluated in the equivalent
  forms `sum_i p_i ln_q(1/p_i)` and `-sum_i p_i ln_q(r_i/p_i)`. These agree with
  the previous closed forms whenever `p` sums to one. They differ for an
  unnormalized `p`, for which the closed forms are genuinely singular at
  `q = 1`.
- `Array` and `Scalar` in `qjax.shared.types` were the same alias, making the
  distinction documentation-only. `Scalar` is now the narrower of the two.
- Return annotations use `jax.Array` rather than the legacy `jnp.ndarray`.
- The version is single-sourced from `qjax.__version__` via `hatch`, instead of
  being duplicated in `pyproject.toml`.
- `release.yaml` declared a `version` input and never used it, so a release
  could ship whatever was in the tree. It is now verified against
  `qjax.__version__`, and publishing uses PyPI Trusted Publishing (OIDC) rather
  than a long-lived API token.
- **The documentation moved from Sphinx/Read the Docs to Material for MkDocs**,
  built on the Docsforge template and published to GitHub Pages by
  `.github/workflows/docs.yaml`. The site is now at
  <https://kleyt0n.github.io/qjax/>; `qjax.readthedocs.io` is retired. Pages
  keep their paths, so `…/examples/classification/` and friends still resolve.
- **Docstring bodies are Markdown rather than reStructuredText.** `.. math::`
  blocks became `$$…$$`, `:math:` roles became `$…$`, and `:func:` roles became
  code spans, so mkdocstrings renders the formulas that Sphinx used to. No
  signature, argument, or behaviour changed.

### Removed

- `.coverage` and the 12 generated binaries under `examples/figures/` are no
  longer tracked in git.

## [0.1.0] - 2026-06

Initial release: `q`-logarithm and `q`-exponential with the `q`-algebra, Tsallis
entropy / cross-entropy / divergence, the `q`-Gaussian, and `tsallis_entmax`.
