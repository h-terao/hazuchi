import jax.numpy as jnp
import chex


def huber_loss(inputs: chex.Array, targets: chex.Array, delta: float) -> chex.Array:
    x = inputs - targets
    return jnp.where(
        jnp.abs(x) < delta,
        jnp.square(x) / 2,
        delta * (jnp.abs(x) - delta / 2),
    )
