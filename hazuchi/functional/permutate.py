from __future__ import annotations

import jax.numpy as jnp
import chex


def permutate(x: chex.Array, index: chex.Array, inv: bool = False) -> chex.Array:
    """Permutate array according to the given index.

    Args:
        x: Array that has a shape of [N, ...].
        index: Index array that has a shape of [N].
        inv (bool): If True, inverse the permutation operation.
            It means that x == permutate(permutate(x, index), index, inv=True).

    Returns:
        Permutated array.
    """
    assert x.shape[0] == index.shape[0]
    if inv:
        n = len(index)
        inv_index = jnp.zeros_like(index).at[index].set(jnp.arange(n))
        return x[inv_index]
    else:
        return x[index]
