import jax.numpy as jnp
import chex


def absolute_error(inputs: chex.Array, targets: chex.Array) -> chex.Array:
    return jnp.abs(inputs - targets)
