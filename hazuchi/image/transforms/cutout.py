import jax
import jax.numpy as jnp
import jax.random as jrandom
import chex

from .core import flatten, unflatten

__all__ = ["cutout"]


def _cutout_mask(rng: chex.PRNGKey, img: chex.Array, mask_size: int) -> chex.Array:
    img, original_shape = flatten(img)
    num_samples, height, width, num_channels = img.shape

    half_mask_size, offset = divmod(mask_size, 2)
    M = jnp.pad(
        jnp.ones_like(img),
        [
            [0, 0],
            [half_mask_size, half_mask_size + offset],
            [half_mask_size, half_mask_size + offset],
            [0, 0],
        ],
    )

    height_ratio, width_ratio = jrandom.uniform(rng, [2])

    start_indices = [
        0,
        (height_ratio * (height + 1)).astype(int),
        (width_ratio * (width + 1)).astype(int),
        0,
    ]

    M = jax.lax.dynamic_update_slice(
        M, jnp.zeros_like(M, shape=(num_samples, mask_size, mask_size, num_channels)), start_indices
    )

    # Crop the central patch.
    M = jax.lax.dynamic_slice(
        M,
        [0, half_mask_size, half_mask_size, 0],
        [num_samples, height, width, num_channels],
    )

    assert M.shape == img.shape
    return unflatten(M, original_shape, clip_val=False)


def cutout(rng: chex.PRNGKey, img: chex.Array, mask_size: int, cval: float = 0) -> chex.Array:
    mask = _cutout_mask(rng, img, mask_size)
    return img * mask + jnp.full_like(img, fill_value=cval) * (1.0 - mask)
