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

!!! tip
    For GPU/TPU acceleration, install the matching JAX build first by following the
    [JAX installation guide](https://docs.jax.dev/en/latest/installation.html);
    `qjax` then runs on whatever JAX backend is available.

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

| Group | Installs |
| --- | --- |
| `dev` | `pytest`, `pytest-cov`, `ruff`, `mypy`, `hypothesis`, `matplotlib` — the full check suite |
| `plots` | `matplotlib` — everything under `qjax.plots` |
| `examples` | `matplotlib`, `numpy` — run the scripts in `examples/` |
| `docs` | `mkdocs-material`, `mkdocstrings` — build this documentation |

Build the documentation locally:

```bash
uv sync --extra docs
uv run mkdocs serve     # live preview on http://127.0.0.1:8000
uv run mkdocs build --strict   # what CI runs: broken links fail the build
```

`mkdocs build` writes the static site to `site/`.
