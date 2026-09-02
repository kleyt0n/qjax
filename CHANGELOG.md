# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-09-02

Adds `qjax.physics`, a small set of statistical-mechanics systems paired with
exact reference values, plus five physics examples built on them. The existing
`qjax.*` surface is unchanged; nothing from `qjax.physics` is re-exported at the
top level, so the flat namespace stays the `q`-primitives.

### Added

- **`qjax.physics`** — physical systems with something exact to check against.
  The scope rule is deliberate: pure, cheap, exactly-testable kernels live here,
  while the long runs, controlled comparisons and figures live in `examples/`.
    - `physics.lattice` — the 2-D Ising Hamiltonian, a checkerboard Metropolis
      sampler, and **three mutually independent** routes to the exact free energy:
      exhaustive enumeration of all `2**(L*L)` states, the `2**L x 2**L` transfer
      matrix (exact for the finite periodic lattice), and Onsager's
      thermodynamic-limit solution.
    - `physics.spinglass` — the Sherrington-Kirkpatrick model with streamed exact
      thermodynamics, including the exact `<s_i s_j>`, up to `N = 22`.
    - `physics.clusters` — the Lennard-Jones potential with a soft spherical wall,
      an Adam quench, and geometry helpers. The `n <= 4` minima are closed forms.
    - `physics.annealing` — the Tsallis-Stariolo cooling schedule. Written as a
      *ratio of two `qjax.q_log` calls*, so its `q -> 1` limit is the Geman-Geman
      logarithmic schedule to zero error, with no branch on `q` and a correct
      non-zero derivative in `q`.
    - `physics.diffusion` — the anomalous-diffusion scaling relations
      (`alpha = 2/(3-q)` and its inverse), the exact Sisyphus-Langevin stationary
      `(q, beta)`, Lutz's cold-atom law, and scan-friendly density estimators.
      Also the **closed-form solution of the nonlinear Fokker-Planck equation**
      (`nlfp_rate`, `nlfp_offset`, `nlfp_width`, `nlfp_density`, `nlfp_front`)
      derived from `beta_dot = -K beta^{(5-q)/2}` with
      `K = 4 D (2-q) / C_q^{1-q}`, which reduces to the heat kernel exactly at
      `q = 1`; and `nlfp_residual`, a differential operator taking any
      `(x, t) -> p` callable, so one function both validates the exact solution
      (its residual vanishes at 1e-15, which is what gates the derivation) and
      trains a network.
    - `physics.observables` — Binder cumulant, level crossings, peak location,
      FWHM, and finite-size extrapolation.
    - `physics.reference` — the exact and published constants, each with its
      citation: `ISING_TC`, `ISING_BETA_EXP`, `ISING_NU`, `ISING_ENERGY_AT_TC`,
      `SK_PARISI_GROUND_STATE`, `LJ_REFERENCE_MINIMA`, `LJ38_ICOSAHEDRAL`.
- **`qjax.nn.autoregressive`** — a masked autoregressive network (MADE) over
  binary spins: exactly normalized, exactly autoregressive, sampled in `N`
  sequential passes and evaluated in one. Needed by any variational method in
  statistical mechanics; nothing about it is `q`-deformed, hence `qjax.nn`.
- **Five statistical-physics examples**, each reporting a validation table against
  exact values, and each with `--full` (or `QJAX_FULL=1`) for a larger run:
    - `examples/ising_phases.py` — machine learning the Ising transition, where
      finite-size crossover generates *physical* label noise. Recovers `T_c` to
      0.6 % and `nu`, `beta` to 3 %.
    - `examples/tsallis_free_energy.py` — the nonextensive variational free
      energy. Records the `q <-> 2-q` duality (`-E_p[ln_q p] = S_{2-q}`), that
      `q = 1` is necessarily optimal because it is the only bound, and that the
      deformation's effect collapses in `(q-1)N`.
    - `examples/generalized_annealing.py` — generalized simulated annealing on
      Lennard-Jones clusters. Reports a negative result with a mechanism: the
      deformation does not pay, and the tail index `nu = (3-q_V)/(q_V-1)` is why.
    - `examples/anomalous_diffusion.py` — `q` as a *measured* quantity, with a
      Fisher error bar and two independent estimators. All three cold-atom arms
      land within one sigma of the exact stationary index.
    - `examples/pinn_fokker_planck.py` — the first PDE-residual code in the
      repository, and the first to differentiate a `qjax` primitive twice in
      space. Reads the Student-t residual model of Abijuru et al. (ICML 2026,
      *Heavy-tailed Physics-Informed Neural Networks*) as what it is: a
      `q`-Gaussian likelihood, with their EM weight equal to the score of
      `q_gaussian_logpdf` at qjax's own `nu = (3-q)/(q-1)` to 1e-15, and the
      mean-squared residual as its `q -> 1` member. Confirms their premise more
      strongly than they state it — PINN residuals fit `q = 2.19 +- 0.03`, i.e.
      `nu < 1`, heavier-tailed than Cauchy — and then reports a negative result:
      because `q_gaussian_logpdf` is differentiable in its index the EM loop is
      unnecessary, but a robust residual loss and a *forward* PDE are a bad
      match. Robustness means tolerating large residuals, and in a forward
      problem the residual is the only thing carrying the initial condition
      inward, so the solution decays into the spurious family every constant
      density forms.

- Tests: `tests/test_physics_{lattice,observables,spinglass,clusters,annealing,diffusion}.py`,
  `tests/test_nn_autoregressive.py`, and `tests/test_examples_physics.py`. The last
  gates the claims that live in the scripts — most of all that autodiff through
  `free_energy_surrogate` reproduces the analytic REINFORCE estimator — and runs
  every physics example end to end in a reduced mode, which the CI `examples` job
  did not previously do for any of the heavier scripts.

### Notes

- `qjax.sample` can return `+inf` for `q` near 3 in float32: the Student-`t`
  representation divides by a `chi2_nu` variate, and for `nu ~ 0.2` a shape-0.09
  gamma draw underflows to exactly zero about once in 2000 draws. The value is a
  finite-precision artifact rather than a genuine sample. Library behaviour is
  unchanged; `examples/generalized_annealing.py` documents and handles it.
- Documentation gains four example pages, a `Physics` section in the API
  reference, and a reorganized examples overview.

## [0.1.2] - 2026-07-27

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
  could ship whatever was in the tree. **Releases are now triggered by pushing a
  `v*` git tag** rather than by `workflow_dispatch`: the tag is checked against
  `qjax.__version__` before anything is built, and publishing uses PyPI Trusted
  Publishing (OIDC) rather than a long-lived API token. The workflow no longer
  creates the tag itself. See "Releasing" in `CONTRIBUTING.md`.
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
