from typing import Callable
import jax
import chex


def triplet_loss(
    inputs: chex.Array,
    positive_inputs: chex.Array,
    negative_inputs: chex.Array,
    loss_fun: Callable[[chex.Array, chex.Array], chex.Array],
    margin: float = 1.0,
) -> chex.Array:
    x = loss_fun(inputs, positive_inputs) - loss_fun(inputs, negative_inputs) + margin
    x = jax.nn.relu(x)
    return x
