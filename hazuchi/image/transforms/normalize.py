import jax.numpy as jnp
import chex

from .utils import flatten, unflatten


def normalize(img: chex.Array, mean: chex.Array | float, std: chex.Array | float) -> chex.Array:
    """Normalize function."""
    img, original_shape = flatten(img)

    mean, std = jnp.array(mean), jnp.array(std)
    assert mean.ndim == 1
    assert std.ndim == 1

    mean = mean.reshape(1, 1, 1, -1)
    std = std.reshape(1, 1, 1, -1)

    img = (img - mean) / std
    img = unflatten(img, original_shape)

    return img


def de_normalize(img: chex.Array, mean: chex.Array | float, std: chex.Array | float) -> chex.Array:
    """De normalize function."""
    img, original_shape = flatten(img)

    mean, std = jnp.array(mean), jnp.array(std)
    assert mean.ndim == 1
    assert std.ndim == 1

    mean = mean.reshape(1, 1, 1, -1)
    std = std.reshape(1, 1, 1, -1)

    img = std * img + mean
    img = unflatten(img, original_shape)

    return img
