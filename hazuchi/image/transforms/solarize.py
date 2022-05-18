import jax.numpy as jnp
import chex


def solarize(img: chex.Array, threshold: float = 0.5) -> chex.Array:
    return jnp.where(img < threshold, img, 1.0 - img)


def solarize_add(img: chex.Array, threshold: float = 0.5, addition: float = 0) -> chex.Array:
    return jnp.where(img < threshold, img + addition, img).clip(0, 1)
