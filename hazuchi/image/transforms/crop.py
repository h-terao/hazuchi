"""crop operation."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import chex

from .utils import flatten, unflatten
from .flip import vflip, hflip

__all__ = [
    "random_crop",
    "center_crop",
    "three_crop",
    "five_crop",
    "ten_crop",
]


def random_crop(
    rng: chex.PRNGKey,
    img: chex.Array,
    size: tuple[int, int],
    pad_size: tuple[int, int] = (0, 0),
    padding_mode: str = "constant",
    cval: float = 0,
) -> chex.Array:
    """Randomly crop the given image into the patch.

    Args:
        rng (PRNGKey): A PRNG key.
        img (array): Image to crop.
        size (tuple of int): Output size (height, width).
        pad_size (tuple of int): Padding size before cropping.
        padding_mode (str): Specify how to pad images.
        cval (float): The padding value if padding_mode is "constant".

    Returns:
        The cropped image.
    """
    img, original_shape = flatten(img)
    img = jnp.pad(
        img,
        [[0, 0], [pad_size[0], pad_size[0]], [pad_size[1], pad_size[1]], [0, 0]],
        mode=padding_mode,
        constant_values=cval,
    )

    batch_size, height, width, channel = img.shape
    x_ratio, y_ratio = jax.random.uniform(rng, (2,))
    y_offset = jnp.int32(y_ratio * (height - size[0] + 1))
    x_offset = jnp.int32(x_ratio * (width - size[1] + 1))

    slice_size = (batch_size, size[0], size[1], channel)
    img = jax.lax.dynamic_slice(img, (0, y_offset, x_offset, 0), slice_size)

    return unflatten(img, original_shape)


def center_crop(
    img: chex.Array,
    size: tuple[int, int],
    pad_size: tuple[int, int] = (0, 0),
    padding_mode: str = "constant",
    cval: float = 0,
) -> chex.Array:
    """Crop the given image into the central patch.

    Args:
        img (array): Image to crop.
        size (tuple of int): Output size.
        pad_size (tuple of int): Padding size before cropping.
        padding_mode (str): Specify how to pad images.
        cval (float): The padding value if padding_mode is "constant".

    Returns:
        The cropped image.
    """
    img, original_shape = flatten(img)
    img = jnp.pad(
        img,
        [[0, 0], [pad_size[0], pad_size[0]], [pad_size[1], pad_size[1]], [0, 0]],
        mode=padding_mode,
        constant_values=cval,
    )

    batch_size, height, width, channel = img.shape
    y_offset = jnp.int32((height - size[0] + 1) / 2)
    x_offset = jnp.int32((width - size[1] + 1) / 2)

    slice_size = (batch_size, size[0], size[1], channel)
    img = jax.lax.dynamic_slice(img, (0, y_offset, x_offset, 0), slice_size)

    return unflatten(img, original_shape)


def three_crop(
    img: chex.Array,
    size: tuple[int, int],
    pad_size: tuple[int, int] = (0, 0),
    padding_mode: str = "constant",
    cval: float = 0,
    interpolation="nearest",
) -> tuple[chex.Array, ...]:
    """Apply three crop operation.

    Crop the given image into three crop to cover the entire of image,
    and resize them into the desired size. Currently, only square size
    (i.e., size[0]==size[1]) is supported.

    Args:
        img (array): Image to crop.
        size (tuple of int): Output size. It must be the square.
        pad_size (tuple of int): Padding size before cropping.
        padding_mode (str): Specify how to pad images.
        cval (float): The padding value if padding_mode is "constant".

    Returns:
        Tuple of the cropped images.
    """
    img, original_shape = flatten(img)
    img = jnp.pad(
        img,
        [[0, 0], [pad_size[0], pad_size[0]], [pad_size[1], pad_size[1]], [0, 0]],
        mode=padding_mode,
        constant_values=cval,
    )
    batch_size, height, width, channel = img.shape

    assert size[0] == size[1], "three_crop only supports size[0]==size[1]."

    @jax.jit
    def horizontal_three_crop(img: chex.Array):
        y_offset = 0  # fixed.

        # left
        x_offset = 0
        left = jax.lax.dynamic_slice(img, (0, y_offset, x_offset, 0), (batch_size, height, height, channel))

        # center
        x_offset = jnp.int32((width - height + 1) / 2)
        center = jax.lax.dynamic_slice(img, (0, y_offset, x_offset, 0), (batch_size, height, height, channel))

        # right
        x_offset = width - height
        right = jax.lax.dynamic_slice(img, (0, y_offset, x_offset, 0), (batch_size, height, height, channel))

        return left, center, right

    @jax.jit
    def vertical_three_crop(img: chex.Array):
        x_offset = 0  # fixed.

        y_offset = 0
        top = jax.lax.dynamic_slice(img, (0, y_offset, x_offset, 0), (batch_size, width, width, channel))

        y_offset = jnp.int32((height - width + 1) / 2)
        center = jax.lax.dynamic_slice(img, (0, y_offset, x_offset, 0), (batch_size, width, width, channel))

        y_offset = height - width
        bottom = jax.lax.dynamic_slice(img, (0, y_offset, x_offset, 0), (batch_size, width, width, channel))

        return top, center, bottom

    patches = jax.lax.cond(width < height, vertical_three_crop, horizontal_three_crop, img)
    patches = jax.tree_util.tree_map(
        lambda x: unflatten(jax.image.resize(x, (size[0], size[1], channel), method=interpolation), original_shape),
        patches,
    )

    return patches


def five_crop(
    img: chex.Array,
    size: tuple[int, int],
    pad_size: tuple[int, int] = (0, 0),
    padding_mode: str = "constant",
    cval: float = 0,
) -> tuple[chex.Array, ...]:
    """Apply five crop operation.

    Crop the given image into four courners and the central crop.

    Args:
        img (array): Image to crop.
        size (tuple of int): Output size.
        pad_size (tuple of int): Padding size before cropping.
        padding_mode (str): Specify how to pad images.
        cval (float): The padding value if padding_mode is "constant".

    Returns:
        Tuple of the cropped images.
    """
    img, original_shape = flatten(img)
    img = jnp.pad(
        img,
        [[0, 0], [pad_size[0], pad_size[0]], [pad_size[1], pad_size[1]], [0, 0]],
        mode=padding_mode,
        constant_values=cval,
    )
    batch_size, height, width, channel = img.shape

    # upper left.
    y_offset = 0
    x_offset = 0
    upper_left = unflatten(
        jax.lax.dynamic_slice(img, (0, y_offset, x_offset, 0), (batch_size, size[0], size[1], channel)),
        original_shape,
    )

    # lower left.
    y_offset = height - size[0]
    x_offset = 0
    lower_left = unflatten(
        jax.lax.dynamic_slice(img, (0, y_offset, x_offset, 0), (batch_size, size[0], size[1], channel)),
        original_shape,
    )

    # center crop
    y_offset = jnp.int32((height - size[0] + 1) / 2)
    x_offset = jnp.int32((width - size[1] + 1) / 2)
    center = unflatten(
        jax.lax.dynamic_slice(img, (0, y_offset, x_offset, 0), (batch_size, size[0], size[1], channel)),
        original_shape,
    )

    # upper right
    y_offset = 0
    x_offset = width - size[1]
    upper_right = unflatten(
        jax.lax.dynamic_slice(img, (0, y_offset, x_offset, 0), (batch_size, size[0], size[1], channel)),
        original_shape,
    )

    # lower right
    y_offset = height - size[0]
    x_offset = width - size[1]
    lower_right = unflatten(
        jax.lax.dynamic_slice(img, (0, y_offset, x_offset, 0), (batch_size, size[0], size[1], channel)),
        original_shape,
    )

    return (upper_left, lower_left, center, upper_right, lower_right)


def ten_crop(
    img: chex.Array,
    size: tuple[int, int],
    pad_size: tuple[int, int] = (0, 0),
    padding_mode: str = "constant",
    cval: float = 0,
    vertical: bool = False,
) -> tuple[chex.Array, ...]:
    """Apply ten crop operation to images.

    Crop the given image into four corners and the central crop plus the flipped version of them.

    Args:
        img (array): Image to crop.
        size (tuple of int): Output size.
        pad_size (tuple of int): Padding size before cropping.
        padding_mode (str): Specify how to pad images.
        cval (float): The padding value if padding_mode is "constant".
        vertical (bool): If True, use vertical flip to crop.
                         Otherwise, use horizontal flip.

    Returns:
        Tuple of the cropped images.
    """
    flipped_img = jax.lax.cond(vertical, vflip, hflip, img)
    return (
        *five_crop(img, size, pad_size, padding_mode, cval),
        *five_crop(flipped_img, size, pad_size, padding_mode, cval),
    )
