from __future__ import annotations

import jax.numpy as jnp
from jax.scipy import ndimage as ndi
import chex

from ..utils import flatten, unflatten


def warp_perspective(
    img: chex.Array, matrix: chex.Array, order: int = 2, mode: str = "constant", cval: float = 0.0
):
    """Warp an image according to a given coordinate transformations.

    Args:
        img (array): An image array. Shape is (..., height, width, channel).
        matrix (array): An affine transformation matrix.
        order (int): The order of the spline interpolation.
                     Only nearest neighbor (0) and linear interpolation (1) are supported.
        mode (str): The mode parameter determines how the input array is extended
                    beyond its boundaries.
                    "reflect", "grid-mirror", "constant", "grid-contant", "nearest",
                    "mirror", "grid-wrap", and "wrap" are supported.
        cval (int): Value to fill past edges of input if mode is "constant". Default is 128.

    Returns:
        Transformed images.
    """
    img, original_shape = flatten(img)

    batch_size, height, width, channel = img.shape
    img = img.transpose(0, 3, 1, 2).reshape(-1, height, width)  # NHWC -> NCHW
    N = batch_size * channel

    img = jnp.float32(img)
    x_t, y_t = jnp.meshgrid(
        jnp.arange(0, width),
        jnp.arange(0, height),
    )
    pixel_coords = jnp.stack([x_t, y_t, jnp.ones_like(x_t)]).astype(jnp.float32)
    x_coords, y_coords, _ = jnp.einsum("ij,jkl->ikl", matrix, pixel_coords)

    coords_to_map = jnp.stack(
        [
            jnp.tile(jnp.arange(N).reshape(N, 1, 1), reps=(1, height, width)),
            jnp.tile(y_coords.reshape(1, height, width), reps=(N, 1, 1)),
            jnp.tile(x_coords.reshape(1, height, width), reps=(N, 1, 1)),
        ],
        axis=0,
    )

    img = ndi.map_coordinates(img, coords_to_map, order=order, mode=mode, cval=cval)
    img = img.reshape(-1, channel, height, width).transpose(0, 2, 3, 1)
    return unflatten(img, original_shape)
