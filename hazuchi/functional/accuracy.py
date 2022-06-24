from __future__ import annotations

import jax.numpy as jnp
import chex


def accuracy(preds: chex.Array, targets: chex.Array, k: int = 1) -> chex.Array:
    """Top-k accuracy score.

    Args:
        preds: Predictions that has a shape of [..., C],
            where C denotes the number of classes.
        targets: One-hot encoded labels that has a shape of [..., C].

    Returns:
        Binary array that has a shape of [...].
        You can compute top-k accuracy score by averaging it.
    """
    assert preds.shape == targets.shape
    y = jnp.argsort(preds)[..., -k:]
    t = jnp.argmax(targets, axis=-1, keepdims=True)
    return jnp.sum(y == t, axis=-1)
