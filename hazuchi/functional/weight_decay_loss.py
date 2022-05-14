from __future__ import annotations
from typing import Callable

import jax
import jax.numpy as jnp
import chex


def weight_decay_loss(
    x: chex.PyTreeDef,
    penalty_fun: Callable[[chex.Array], chex.Array],
    filter_fun: Callable[[chex.Array], bool] | None = None,
) -> chex.Array:
    if filter_fun is None:
        filter_fun = lambda v: True  # noqa
    return sum([jnp.sum(penalty_fun(xi)) for xi in jax.tree_leaves(x) if filter_fun(xi)])
