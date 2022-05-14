import jax.numpy as jnp
import chex


def squared_error(inputs: chex.Array, targets: chex.Array) -> chex.Array:
    return jnp.square(inputs - targets)
