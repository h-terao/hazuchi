from __future__ import annotations

import jax
import jax.numpy as jnp
import chex


def cross_entropy(logits: chex.Array, targets: chex.Array) -> chex.Array:
    return -jnp.sum(targets * jax.nn.log_softmax(logits), axis=-1)
