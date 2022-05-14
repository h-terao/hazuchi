import jax.numpy as jnp
import chex


def permutate(x: chex.Array, indices: chex.Array, inv: bool = False) -> chex.Array:
    n = len(indices)
    if inv:
        indices = jnp.zeros_like(indices).at[indices].set(jnp.arange(n))
    return x[indices]
