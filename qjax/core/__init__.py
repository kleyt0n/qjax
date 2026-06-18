"""Core Tsallis-statistics primitives: deformed functions, entropy, distributions."""

from qjax.core.activations import tsallis_entmax
from qjax.core.distributions import (
    normalization,
    q_gaussian_logpdf,
    q_gaussian_pdf,
    sample,
)
from qjax.core.entropy import (
    tsallis_cross_entropy,
    tsallis_divergence,
    tsallis_entropy,
)
from qjax.core.functions import q_add, q_diff, q_div, q_exp, q_log, q_prod

__all__ = [
    # functions
    "q_log",
    "q_exp",
    "q_add",
    "q_diff",
    "q_prod",
    "q_div",
    # entropy
    "tsallis_entropy",
    "tsallis_cross_entropy",
    "tsallis_divergence",
    # distributions
    "normalization",
    "q_gaussian_pdf",
    "q_gaussian_logpdf",
    "sample",
    # activations
    "tsallis_entmax",
]
