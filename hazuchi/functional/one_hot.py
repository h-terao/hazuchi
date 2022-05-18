from __future__ import annotations
import jax
import jax.numpy as jnp
import chex


def one_hot(
    labels: chex.Array,
    num_classes: int,
    label_smoothing: float = 0,
    dtype: chex.ArrayDType = jnp.float32,
) -> chex.Array:
    """Convert to soft labels.

    Args:
        labels: Label indices.
        num_classes (int): Number of classes.
        label_smoothing (float): Label smoothing value.
        dtype: JAX array dtype.
    """
    off_value = label_smoothing / num_classes
    on_value = 1.0 - label_smoothing + off_value
    labels = jax.nn.one_hot(labels, num_classes, dtype=dtype)
    labels = jnp.where(labels > 0.5, on_value, off_value)
    return labels
