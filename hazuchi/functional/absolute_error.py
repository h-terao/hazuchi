import jax.numpy as jnp
import chex


def absolute_error(inputs: chex.Array, targets: chex.Array) -> chex.Array:
    """Computes the absolute difference between inputs and targets."""
    return jnp.abs(inputs - targets)
