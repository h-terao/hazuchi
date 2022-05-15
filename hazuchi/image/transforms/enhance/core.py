from __future__ import annotations
from typing import Callable
import jax
import jax.numpy as jnp
import chex

from ..utils import flatten, unflatten


def blend(img1: chex.Array, img2: chex.Array, factor: float) -> chex.Array:
    # factor * img1 + (1-factor) * img2
    return factor * (img1 - img2) + img2


# def apply_kernel(img: chex.Array, kernel: chex.Array, clip_val: bool = True) -> chex.Array:
#     """kernel: 3x3 array"""
#     img, original_shape = flatten(img)

#     # (3, 3) -> (3, 3, 1, 3)
#     kernel = kernel.reshape(3, 3, 1, 1).repeat(3, axis=-1)

#     degenerate = jax.lax.conv_general_dilated(
#         jnp.transpose(img, [0, 3, 1, 2]),  # lhs = NCHW image tensor
#         jnp.transpose(kernel, [3, 2, 0, 1]),  # rhs = OIHW conv kernel tensor
#         (1, 1),  # window strides
#         "VALID",  # padding mode
#         feature_group_count=3,
#     )
#     degenerate = jnp.transpose(degenerate, [0, 2, 3, 1])  # NCHW -> NHWC
#     degenerate = jnp.pad(degenerate, [[0, 0], [1, 1], [1, 1], [0, 0]])

#     # For the borders of the resulting image, fill in the values of the
#     # original image.
#     mask = jnp.ones_like(degenerate)
#     mask = jnp.pad(mask, [[0, 0], [1, 1], [1, 1], [0, 0]], constant_values=0).astype(jnp.bool)

#     degenerate = jnp.where(mask, degenerate, img)
#     return unflatten(degenerate, original_shape, clip_val=clip_val)


def convolve(
    img: chex.Array,
    kernel: chex.Array | Callable[[chex.Array], chex.Array],
    kernel_shape: tuple[int, int] | None = None,
) -> chex.Array:
    """Apply kernel or filter to image.

    Args:
        img: Image array that be applied to convolve operation.
        kernel: A kernel array or the callable object.
                If callable object, it must transform [N,H,W] array to [N] array,
                where H and W are shape of kernel and N is the number of patches to convolve.
        kernel_shape: Height and width of kernel.
                      This value is required when you use the callable object as kernel.
                      If kernel is array, this argument is ignored
                      and use kernel.shape instead of it.

    Example:
        Apply the median filter to images.
        >>> convolve(img, (3, 3), lambda x: jnp.median(x, axis=(1, 2)))

    Todo:
        debug convolve function.
    """
    img, original_shape = flatten(img)
    N, _, _, C = img.shape

    if isinstance(kernel, jnp.ndarray):
        kernel_shape = kernel.shape
    assert kernel_shape is not None

    col = jax.lax.conv_general_dilated_patches(
        img.transpose(0, 3, 1, 2),  # NHWC -> NCHW
        kernel_shape,
        window_strides=(1, 1),
        padding="SAME",
        # padding="valid",
    )
    # [N,[Chw],H,W] -> [N,H,W,[Chw]] -> [N,H,W,C,h,w]
    H, W = col.shape[-2:]
    col = col.transpose(0, 2, 3, 1).reshape(-1, *kernel_shape)

    if isinstance(kernel, jnp.ndarray):
        kernel = kernel[None]
        degenerate = jnp.sum(kernel * col, axis=(-1, -2))
    else:
        degenerate = kernel(col)

    degenerate = degenerate.reshape(N, H, W, C)
    return unflatten(degenerate, original_shape)

    # # copy edge from the original image
    # start_indices = (0, kernel_shape[0] // 2, kernel_shape[1] // 2, 0)
    # degenerate = jax.lax.dynamic_update_slice(img, degenerate, start_indices)
    # return unflatten(degenerate, original_shape)


def rgb2gray(img: chex.Array, as_rgb: bool = True) -> chex.Array:
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    v = 0.2989 * r + 0.5870 * g + 0.1140 * b
    v = v[..., None]  # Add channel dim.
    if as_rgb:
        v = jnp.repeat(v, 3, axis=-1)
    return v
