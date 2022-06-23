import jax.numpy as jnp
import chex

from .core import blend
from .colorspace import rgb2gray

__all__ = ["adjust_saturation", "adjust_contrast", "adjust_brightness"]


def adjust_saturation(img: chex.Array, factor: float) -> chex.Array:
    gray = rgb2gray(img)
    return blend(img, gray, factor)


def adjust_contrast(img: chex.Array, factor: float) -> chex.Array:
    degenerate = rgb2gray(img, as_rgb=False)
    degenerate = jnp.mean(degenerate, axis=(-1, -2, -3), keepdims=True)
    return blend(img, degenerate, factor)


def adjust_brightness(img: chex.Array, factor: float) -> chex.Array:
    degenerate = jnp.zeros_like(img)
    return blend(img, degenerate, factor)
