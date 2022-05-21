from __future__ import annotations
import jax
import jax.numpy as jnp
import chex

from .one_hot import one_hot


def cross_entropy(
    logits: chex.Array, labels: chex.Array, label_smoothing: float = 0.0
) -> chex.Array:
    """The cross-entropy loss."""
    assert logits.ndim in [labels.ndim, labels.ndim + 1]

    num_classes = logits.shape[-1]
    if logits.ndim == labels.ndim + 1:
        labels = one_hot(labels, num_classes, label_smoothing, dtype=logits.dtype)

    assert logits.shape == labels.shape
    log_preds = jax.nn.log_softmax(logits)
    return -jnp.sum(labels * log_preds, axis=-1)
