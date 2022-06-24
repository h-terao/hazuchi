from __future__ import annotations

import jax
import jax.numpy as jnp
import chex


def one_hot(labels: chex.Array, num_classes: int, label_smoothing: float = 0) -> chex.Array:
    labels = jax.nn.one_hot(labels, num_classes)
    labels = jnp.where(labels > 0.5, 1.0 - label_smoothing, 0)
    labels += label_smoothing / num_classes
    return labels
