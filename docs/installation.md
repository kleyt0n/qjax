# Installation

`qjax` requires **Python 3.10+** and depends only on `jax` and `matplotlib`.

## Install

Add it to your project with [uv](https://docs.astral.sh/uv/) (recommended):

```bash
uv add qjax
```

or install it with pip:

```bash
pip install qjax
```

```{tip}
For GPU/TPU acceleration, install the matching JAX build first by following the
[JAX installation guide](https://docs.jax.dev/en/latest/installation.html);
`qjax` then runs on whatever JAX backend is available.
```

## Contributing (from a clone)

To work on `qjax` itself, clone the repository and create the development
environment with uv:

```bash
git clone <repo-url> qjax
cd qjax
uv sync --extra dev    # runtime + tests and linter (pytest, ruff)
```

Then run the checks:

```bash
uv run pytest          # test suite (q -> 1 limits, gradients, jit/vmap)
uv run ruff check      # lint
```

### Optional dependency groups

```{list-table}
:header-rows: 1
:widths: 20 80

* - Group
  - Installs
* - `dev`
  - `pytest`, `ruff` — run the test suite and linter
* - `docs`
  - `sphinx`, `furo`, `myst-parser` — build this documentation
```

Build the documentation locally:

```bash
uv sync --extra docs
uv run sphinx-build -b html docs docs/_build/html   # or: cd docs && make html
```

Then open `docs/_build/html/index.html`.
