from __future__ import annotations
import jax.numpy as jnp
import jax.random as jrandom
import chex

__all__ = ["permutate", "shuffle"]


def permutate(x: chex.Array, indices: chex.Array, inv: bool = False) -> chex.Array:
    n = len(indices)
    if inv:
        indices = jnp.zeros_like(indices).at[indices].set(jnp.arange(n))
    return x[indices]


def shuffle(
    rng: chex.PRNGKey, x: chex.Array, return_indices: bool = False
) -> chex.Array | tuple[chex.Array, chex.Array]:
    indices = jrandom.permutation(rng, len(x))
    if return_indices:
        return x[indices], indices
    else:
        return x[indices]
