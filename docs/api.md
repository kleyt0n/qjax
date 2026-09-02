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

::: qjax.nn.autoregressive
    options:
      heading_level: 3

## Physics — systems with exact reference values

`qjax.physics` is not re-exported at the top level: import it explicitly
(`import qjax.physics as qp`) so the flat `qjax.*` namespace stays the
`q`-primitives. Its scope rule is deliberate — pure, cheap, exactly-testable
kernels live here; the long runs, controlled comparisons and figures live in
`examples/`.

::: qjax.physics.reference
    options:
      heading_level: 3

::: qjax.physics.lattice
    options:
      heading_level: 3

::: qjax.physics.observables
    options:
      heading_level: 3

::: qjax.physics.spinglass
    options:
      heading_level: 3

::: qjax.physics.clusters
    options:
      heading_level: 3

::: qjax.physics.annealing
    options:
      heading_level: 3

::: qjax.physics.diffusion
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
