from __future__ import annotations
import jax
import jax.numpy as jnp
import chex

from .core import warp_perspective, flatten, unflatten


__all__ = ["affine", "rotate", "rot90", "shear", "translate"]


@jax.jit
def rot_matrix(angle: float, center: tuple[float, float], dtype: chex.ArrayDType = jnp.float32):
    center_y, center_x = center
    angle *= jnp.pi / 180

    shift_x = center_x - center_x * jnp.cos(angle) + center_y * jnp.sin(angle)
    shift_y = center_y - center_x * jnp.sin(angle) - center_y * jnp.cos(angle)
    return jnp.array(
        [
            [jnp.cos(angle), -jnp.sin(angle), shift_x],
            [jnp.sin(angle), jnp.cos(angle), shift_y],
            [0, 0, 1],
        ],
        dtype=dtype,
    )


@jax.jit
def shear_matrix(angles: tuple[float, float] = (0, 0), dtype: chex.ArrayDType = jnp.float32):
    angle_y, angle_x = angles
    angle_x = angle_x * jnp.pi / 180
    angle_y = angle_y * jnp.pi / 180
    return jnp.array(
        [
            [1, jnp.tan(angle_x), 0],
            [jnp.tan(angle_y), 1, 0],
            [0, 0, 1],
        ],
        dtype=dtype,
    )


@jax.jit
def translate_matrix(translation: tuple[int, int] = (0, 0), dtype: chex.ArrayDType = jnp.float32):
    shift_y, shift_x = translation
    return jnp.array(
        [
            [1, 0, -shift_x],
            [0, 1, -shift_y],
            [0, 0, 1],
        ],
        dtype=dtype,
    )


@jax.jit
def scale_matrix(scale: float, dtype: chex.ArrayDType = jnp.float32):
    return jnp.array(
        [
            [scale, 0, 0],
            [0, scale, 0],
            [0, 0, 1],
        ],
        dtype=dtype,
    )


def affine(
    img: chex.Array,
    angle: float = 0.0,
    scale: float = 1.0,
    shear: tuple[float, float] = (0, 0),
    translation: tuple[int, int] = (0, 0),
    center: tuple[float, float] | None = None,
    order: int = 0,
    mode: str = "constant",
    cval: float = 0,
) -> chex.Array:
    """Rotate a image.

    Args:
        img (array): An image array. Shape is (..., height, width, channel).
        angle (float): Rotation angle value in degrees.
        center (tuple[float, float], optional):
            Center of rotation (height, width). Origin is the upper left corner.
            If None, center of the image is used.
        order (int): The order of the spline interpolation.
                     Only nearest neighbor (0) and linear interpolation (1) are supported.
        mode (str): The mode parameter determines how the input array is extended beyond
                    its boundaries. "reflect", "grid-mirror", "constant", "grid-contant", "nearest",
                    "mirror", "grid-wrap", and "wrap" are supported.
        cval (float): Value to fill past edges of input if mode is "constant".

    Returns:
        Transformed images.
    """
    if center is None:
        *_, height, width, _ = img.shape
        center = ((height - 1) / 2, (width - 1) / 2)

    matrix = rot_matrix(angle, center)
    matrix @= scale_matrix(scale)
    matrix @= shear_matrix(shear)
    matrix @= translate_matrix(translation)

    return warp_perspective(img, matrix, order, mode, cval)


def rotate(
    img: chex.Array,
    angle: float,
    center: tuple[float, float] | None = None,
    order: int = 0,
    mode: str = "constant",
    cval: float = 0,
) -> chex.Array:
    """Rotate a image.

    Args:
        img (array): An image array. Shape is (..., height, width, channel).
        angle (float): Rotation angle value in degrees.
        center (tuple[float, float], optional):
            Center of rotation (height, width). Origin is the upper left corner.
            If None, center of the image is used.
        order (int): The order of the spline interpolation.
                     Only nearest neighbor (0) and linear interpolation (1) are supported.
        mode (str): The mode parameter determines how the input array is extended beyond
                    its boundaries. "reflect", "grid-mirror", "constant", "grid-contant", "nearest",
                    "mirror", "grid-wrap", and "wrap" are supported.
        cval (float): Value to fill past edges of input if mode is "constant".

    Returns:
        Transformed images.
    """
    if center is None:
        *_, height, width, _ = img.shape
        center = ((height - 1) / 2, (width - 1) / 2)
    return warp_perspective(img, rot_matrix(angle, center), order, mode, cval)


def rot90(img: chex.Array, n: int = 1) -> chex.Array:
    img, original_shape = flatten(img)
    img = jnp.rot90(img, n, axes=(1, 2))
    return unflatten(img, original_shape)


def shear(
    img: chex.Array,
    angles: tuple[float, float] = (0, 0),
    order: int = 0,
    mode: str = "constant",
    cval: int = 0,
) -> chex.Array:
    """
    Args:
        img (array): An image array. Shape is (..., height, width, channel).
        angles (tuple of float): Vertical and horizontal angles to shear.
        order (int): The order of the spline interpolation.
                     Only nearest neighbor (0) and linear interpolation (1) are supported.
        mode (str): The mode parameter determines how the input array is extended beyond
                    its boundaries. "reflect", "grid-mirror", "constant", "grid-contant", "nearest",
                    "mirror", "grid-wrap", and "wrap" are supported.
        cval (int): Value to fill past edges of input if mode is "constant".

    Returns:
        Transformed images.
    """
    return warp_perspective(img, shear_matrix(angles), order, mode, cval)


def translate(
    img: chex.Array,
    translation: tuple[int, int] = (0, 0),
    order: int = 0,
    mode: str = "constant",
    cval: int = 0,
) -> chex.Array:
    """
    Args:
        img (array): An image array. Shape is (..., height, width, channel).
        translation: Vertical and horizontal number of pixels to translate.
        order (int): The order of the spline interpolation.
                     Only nearest neighbor (0) and linear interpolation (1) are supported.
        mode (str): The mode parameter determines how the input array is extended beyond
                    its boundaries. "reflect", "grid-mirror", "constant", "grid-contant", "nearest",
                    "mirror", "grid-wrap", and "wrap" are supported.
        cval (float): Value to fill past edges of input if mode is "constant".

    Returns:
        Transformed images.
    """
    return warp_perspective(img, translate_matrix(translation), order, mode, cval)
