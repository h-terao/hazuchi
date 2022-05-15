from __future__ import annotations
import chex


def flatten(img: chex.Array) -> tuple[chex.Array, tuple[int, ...]]:
    """convert [..., H, W, C] -> [N, H, W, C]."""
    *_, height, width, channel = img.shape
    return img.reshape(-1, height, width, channel), img.shape


def unflatten(
    img: chex.Array, original_shape: tuple[int, ...], clip_val: bool = True
) -> chex.Array:
    """convert [N, H, W, C] -> [..., H, W, C]."""
    *batch_dims, _, _, _ = original_shape
    *_, height, width, channel = img.shape
    img = img.reshape(*batch_dims, height, width, channel)
    if clip_val:
        img = img.clip(0, 1)
    return img
