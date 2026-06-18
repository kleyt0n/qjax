"""Shared utilities: type aliases and entropic-index validation."""

from qjax.shared.types import Array, Scalar
from qjax.shared.validation import Q_EPS, as_scalar_q, near_one

__all__ = ["Array", "Scalar", "Q_EPS", "as_scalar_q", "near_one"]
