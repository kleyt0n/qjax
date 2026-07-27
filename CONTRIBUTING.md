# Contributing to qjax

Thanks for your interest in `qjax`! It is a research library for **Tsallis
statistics in AI**, built on JAX. Contributions of all kinds are welcome: new
`q`-deformed primitives, examples, documentation, tests, and bug fixes.

This guide explains how to set up the project, the conventions we follow, and
how to get a change merged.

## Development setup

`qjax` is managed with [uv](https://docs.astral.sh/uv/) and targets
**Python 3.10+**.

```bash
git clone <repo-url> qjax
cd qjax
uv sync --extra dev          # runtime + tests + linter
```

Useful commands (these mirror what CI runs):

```bash
uv run pytest                                   # test suite
uv run pytest --cov=qjax --cov-report=term-missing   # with coverage
uv run ruff check qjax tests examples           # lint
uv run python examples/q_gaussian.py            # run an example
```

To work on the documentation:

```bash
uv sync --extra docs
uv run mkdocs serve            # live preview on http://127.0.0.1:8000
uv run mkdocs build --strict   # what CI runs: warnings (dead links) are errors
```

## Project layout

```
qjax/
├── core/        # the math: functions, entropy, distributions, activations
├── shared/      # type aliases and q validation helpers
└── plots/       # magma-themed, publication-grade plotting helpers
examples/        # runnable scripts that save figures to examples/figures/
tests/           # pytest suite
docs/            # Material for MkDocs documentation (Markdown)
```

`core/` is the single source of truth for the math; `plots/`, `examples/`, and
`tests/` only consume the public API.

## Design principles

New primitives should follow the conventions already in `core/`:

1. **Pure functions of `(x, q)`.** No hidden state. Every primitive must work
   under `jax.jit`, `jax.grad`, and `jax.vmap`.
2. **Recover the classical limit.** As `q → 1`, the primitive must reduce to its
   Boltzmann–Gibbs–Shannon counterpart (log, exp, Shannon entropy, KL, …).
3. **Finite gradients at `q = 1`.** Closed forms are typically `0/0` at `q = 1`.
   Use the *double-`where`* trick: select the analytic limit with
   `jnp.where(near_one(q), limit, deformed)`, and sanitize the unused branch's
   inputs so it produces neither a `NaN` value nor a `NaN` gradient (a `0 * NaN`
   from the unused branch will still poison `jax.grad`). See `qjax.core.functions`
   and `qjax.core.distributions.normalization` for worked examples.
4. **Validate `q` consistently.** Use the helpers in `qjax.shared.validation`
   (`as_scalar_q`, `near_one`, `Q_EPS`).

## Code style

- **Formatting/linting:** [ruff](https://github.com/astral-sh/ruff), line length
  **100**. Run `uv run ruff check qjax tests examples` before committing; CI
  enforces it.
- **Docstrings:** Google style (enforced by ruff's pydocstyle and rendered by
  mkdocstrings). Every public function documents its **formula**, its
  **`q → 1` limit**, args (with shapes), and returns. Docstring bodies are
  **Markdown**, not reStructuredText: use `$...$` / `$$...$$` for math and
  backticks for code — MathJax renders the former on the API reference page.
- **Type hints:** use the aliases in `qjax.shared.types` (`Array`, `Scalar`).
- Keep the public API curated in `qjax/__init__.py` and the relevant
  `__init__.py` re-exports.

## Tests

Every new primitive needs tests in `tests/`. At minimum, cover:

- the **`q → 1` limit** (matches the classical function to tolerance);
- **gradient finiteness** with `jax.grad`, including at and near `q = 1`;
- **`jit` / `vmap`** compatibility and output shapes;
- any **mathematical identities** or domain edges (e.g. the Tsallis cut-off,
  normalization, simplex constraints).

Tests run with float64 enabled (`jax.config.update("jax_enable_x64", True)`).

```bash
uv run pytest
```

## Plots and examples

- Use `qjax.plots.use_qjax_style()` and save with `qjax.plots.save_figure(...)`,
  which writes vector **PDF** with the project's publication style.
- All figures use the **`magma`** colormap — get colors from
  `qjax.plots.qcolors(n)` or the `CMAP` constant rather than hard-coding.
- Examples should be self-contained, reproducible (fixed PRNG seeds), and save
  their output to `examples/figures/`.

## Submitting a change

1. Create a branch from `main`.
2. Make your change with tests and docstrings.
3. Ensure the full CI set passes locally:
   ```bash
   uv run ruff check qjax tests examples
   uv run pytest --cov=qjax --cov-report=term-missing
   uv run mkdocs build --strict
   ```
4. Open a pull request describing the change and the maths behind it. Link any
   relevant references (the README and `docs/theory.md` cite the key papers).

CI (`.github/workflows/ci.yaml`) runs lint, the test matrix on Python
3.10–3.13, and the documentation build on every push and pull request.

## Reporting issues

When filing a bug, please include a **minimal reproducible example**, the
expected vs. actual behaviour, and your `jax` / `jaxlib` versions
(`uv run python -c "import jax; print(jax.__version__)"`).

## License

By contributing, you agree that your contributions are licensed under the
project's [MIT License](LICENSE).
