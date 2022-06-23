from __future__ import annotations
import functools

import jax.numpy as jnp
import chex

from .core import convolve

__all__ = ["average_blur", "median_blur"]


def average_blur(img: chex.Array, kernel_size: tuple[int, int]) -> chex.Array:
    """Apply average filter to images."""
    return convolve(img, functools.partial(jnp.mean, axis=(1, 2)), kernel_size)


def median_blur(img: chex.Array, kernel_size: tuple[int, int]) -> chex.Array:
    """Apply median filter to images."""
    return convolve(img, functools.partial(jnp.median, axis=(1, 2)), kernel_size)
