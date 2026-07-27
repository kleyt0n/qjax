r"""Loss functions built on the Tsallis information measures.

The Tsallis cross-entropy $H_q(y, p) = -\sum_i y_i \ln_q p_i$ is a drop-in
replacement for the usual cross-entropy that recovers it at ``q = 1``.

Its practical appeal is robustness to label noise, and that lives at ``q < 1``.
There the ``q``-logarithm is bounded below,

$$
\ln_q(0) = \frac{-1}{1 - q} \quad (q < 1),
$$

so a confidently wrong prediction — the signature of a mislabelled example --
contributes at most $1/(1-q)$ instead of diverging, and its gradient is
capped with it. At ``q = 1`` the penalty is unbounded, and for ``q > 1`` it grows
*faster* than the logarithm (like $p^{1-q}$), which sharpens the model on
clean data at the cost of amplifying bad labels.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from qjax.core.activations import tsallis_entmax
from qjax.core.entropy import tsallis_cross_entropy
from qjax.shared.types import Array, Scalar


def tsallis_cross_entropy_loss(
    logits_or_probs: Array,
    targets: Array,
    q: Scalar = 1.0,
    from_logits: bool = True,
    normalizer_q: Scalar | None = None,
    axis: int = -1,
    reduction: str = "mean",
) -> jax.Array:
    r"""``q``-deformed cross-entropy loss.

    Args:
        logits_or_probs: Unnormalized scores if ``from_logits`` (the default),
            otherwise probabilities that already sum to one along ``axis``.
        targets: Target distribution along ``axis``. Either one-hot or soft;
            integer class indices are *not* accepted, so encode them first.
        q: Entropic index of the *loss*. ``1`` is the standard cross-entropy;
            values below ``1`` bound the penalty on confidently wrong
            predictions (robust to label noise); values above ``1`` sharpen it.
        from_logits: Whether to normalize the input first.
        normalizer_q: Entropic index of the `tsallis_entmax` used to
            normalize logits. Defaults to ``q``, coupling the two; pass ``1.0``
            to keep an ordinary softmax under a deformed loss. Ignored when
            ``from_logits`` is ``False``.
        axis: Axis holding the class distribution.
        reduction: ``"mean"``, ``"sum"``, or ``"none"``.

    Returns:
        The reduced loss, or the per-example losses when ``reduction="none"``.

    Raises:
        ValueError: If ``reduction`` is not one of the three accepted values.

    Note:
        A sparse normalizer (``normalizer_q > 1``) can assign exactly zero
        probability to the true class, for which the loss is a genuine ``+inf``.
        That is mathematically correct but hostile to optimization; either pair a
        deformed loss with ``normalizer_q=1.0`` or keep ``q`` close to ``1``
        early in training.
    """
    if reduction not in ("mean", "sum", "none"):
        raise ValueError(f"reduction must be 'mean', 'sum', or 'none'; got {reduction!r}.")

    logits_or_probs = jnp.asarray(logits_or_probs, dtype=jnp.result_type(float))
    targets = jnp.asarray(targets, dtype=jnp.result_type(float))

    if from_logits:
        norm_q = q if normalizer_q is None else normalizer_q
        probs = tsallis_entmax(logits_or_probs, q=norm_q, axis=axis)
    else:
        probs = logits_or_probs

    per_example = tsallis_cross_entropy(probs, targets, q, axis=axis)
    if reduction == "mean":
        return jnp.mean(per_example)
    if reduction == "sum":
        return jnp.sum(per_example)
    return per_example


__all__ = ["tsallis_cross_entropy_loss"]
