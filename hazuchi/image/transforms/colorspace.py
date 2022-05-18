from __future__ import annotations
import jax.numpy as jnp
import chex

__all__ = ["rgb2gray"]


def rgb2gray(img: chex.Array, as_rgb: bool = False) -> chex.Array:
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    v = 0.2989 * r + 0.5870 * g + 0.1140 * b
    v = v[..., None]  # Add channel dim.
    if as_rgb:
        v = jnp.repeat(v, 3, axis=-1)
    return v
