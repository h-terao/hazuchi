import jax.numpy as jnp
import chex


def autocontrast(img: chex.Array) -> chex.Array:
    low = jnp.min(img, axis=(-2, -3), keepdims=True)
    high = jnp.max(img, axis=(-2, -3), keepdims=True)
    degenerate = (img - low) / (high - low)
    return jnp.where(high > low, img, degenerate).clip(0, 1)
