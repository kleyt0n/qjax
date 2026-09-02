"""Neural-network building blocks for Tsallis models.

Deliberately framework-agnostic: everything here operates on plain arrays and
pytrees, so it composes with Flax, Equinox, Haiku, or hand-rolled JAX alike,
without adding a dependency on any of them.

The pieces are the ones that were being rewritten in every example — a
reparameterization that keeps a learnable ``q`` inside its valid range, an
``entmax`` attention block, a ``q``-deformed classification loss, and a masked
autoregressive network for exactly normalized distributions over binary spins.
"""

from qjax.nn.attention import entmax_attention
from qjax.nn.autoregressive import (
    made_conditionals,
    made_init,
    made_log_prob,
    made_masks,
    made_sample,
)
from qjax.nn.losses import tsallis_cross_entropy_loss
from qjax.nn.reparam import bounded_q, inverse_bounded_q

__all__ = [
    "bounded_q",
    "inverse_bounded_q",
    "entmax_attention",
    "tsallis_cross_entropy_loss",
    "made_masks",
    "made_init",
    "made_conditionals",
    "made_log_prob",
    "made_sample",
]
