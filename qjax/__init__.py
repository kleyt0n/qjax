"""qjax — Tsallis statistics for artificial intelligence, built on JAX.

Public API (see submodules for details):

- Deformed functions: `q_log`, `q_exp`, and the ``q``-algebra
  (`q_add`, `q_diff`, `q_prod`, `q_div`).
- Information measures: `tsallis_entropy`, `tsallis_cross_entropy`,
  `tsallis_divergence`.
- The ``q``-Gaussian distribution: `q_gaussian_pdf`,
  `q_gaussian_logpdf`, `sample`, `normalization`.
- The ``q``-deformed softmax/sparsemax family: `tsallis_entmax`.

Every function is a pure, differentiable JAX expression that recovers its
Boltzmann–Gibbs counterpart as ``q -> 1``.
"""

from qjax.core import (
    normalization,
    q_add,
    q_diff,
    q_div,
    q_exp,
    q_gaussian_logpdf,
    q_gaussian_pdf,
    q_log,
    q_prod,
    sample,
    tsallis_cross_entropy,
    tsallis_divergence,
    tsallis_entmax,
    tsallis_entropy,
)

__version__ = "0.1.2"

__all__ = [
    "__version__",
    "q_log",
    "q_exp",
    "q_add",
    "q_diff",
    "q_prod",
    "q_div",
    "tsallis_entropy",
    "tsallis_cross_entropy",
    "tsallis_divergence",
    "normalization",
    "q_gaussian_pdf",
    "q_gaussian_logpdf",
    "sample",
    "tsallis_entmax",
]
