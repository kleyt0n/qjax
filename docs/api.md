# API reference

The curated public API is re-exported at the top level (e.g. `qjax.q_log`); the
canonical definitions live in the submodules documented below. Every entry on
this page is rendered from the source docstrings by
[mkdocstrings](https://mkdocstrings.github.io/), so it cannot disagree with the
installed version.

!!! note "How to read these pages"
    Signatures show the annotations from the source. Arguments, returns, and
    raises come from the Google-style docstring sections — the same text `help()`
    prints in a REPL. Click **Source** on any entry to see the implementation.

## Core — deformed functions

::: qjax.core.functions
    options:
      heading_level: 3

## Core — entropy and divergences

::: qjax.core.entropy
    options:
      heading_level: 3

## Core — the q-Gaussian distribution

::: qjax.core.distributions
    options:
      heading_level: 3

## Core — activations (entmax)

::: qjax.core.activations
    options:
      heading_level: 3

## Neural-network building blocks

Framework-agnostic pieces for Tsallis models: everything here operates on plain
arrays and pytrees, so it composes with Flax, Equinox, Haiku, or hand-rolled JAX
without adding a dependency on any of them.

::: qjax.nn.reparam
    options:
      heading_level: 3

::: qjax.nn.attention
    options:
      heading_level: 3

::: qjax.nn.losses
    options:
      heading_level: 3

## Shared — types, validation, and series

::: qjax.shared.types
    options:
      heading_level: 3

::: qjax.shared.validation
    options:
      heading_level: 3

::: qjax.shared.series
    options:
      heading_level: 3


## Plots

Plotting requires the optional `plots` extra: `pip install "qjax[plots]"`.

::: qjax.plots.style
    options:
      heading_level: 3

::: qjax.plots.functions
    options:
      heading_level: 3

::: qjax.plots.distributions
    options:
      heading_level: 3
