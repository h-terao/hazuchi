import jax.numpy as jnp
import chex

from .core import blend, convolve


def sharpness(img: chex.Array, factor: float) -> chex.Array:
    # Smooth PIL kernel.
    kernel = jnp.array(
        [
            [1, 1, 1],
            [1, 5, 1],
            [1, 1, 1],
        ],
        dtype=img.dtype,
    )
    kernel /= 13.0

    degenerate = convolve(img, kernel)
    return blend(degenerate, img, factor)
