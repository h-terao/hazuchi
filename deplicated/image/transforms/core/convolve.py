from __future__ import annotations
from typing import Callable
import jax
import jax.numpy as jnp
import chex

from .array_utils import flatten, unflatten


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
