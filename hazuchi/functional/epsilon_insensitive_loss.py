from __future__ import annotations

import jax.numpy as jnp
import chex


def epsilon_insensitive_loss(
    inputs: chex.Array, targets: chex.Array, epsilon: float = 0
) -> chex.Array:
    """Epsilon insensitive loss."""
    return jnp.clip(jnp.abs(inputs - targets) - epsilon, 0)
