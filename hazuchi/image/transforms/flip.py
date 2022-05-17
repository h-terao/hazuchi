import jax.numpy as jnp
import jax.random as jrandom
import chex


def hflip(img: chex.Array) -> chex.Array:
    """
    Args:
        img (array): An image array. Shape is (..., height, width, channel).

    Returns:
        Horizontally flipped images.
    """
    return img[..., :, ::-1, :]


def vflip(img: chex.Array) -> chex.Array:
    """
    Args:
        img (array): An image array. Shape is (..., height, width, channel).

    Returns:
        Vertically flipped images.
    """
    return img[..., ::-1, :, :]


def random_hflip(rng: chex.PRNGKey, img: chex.Array, p: float = 0.5):
    return jnp.where(jrandom.uniform(rng) < p, hflip(img), img)


def random_vflip(rng: chex.PRNGKey, img: chex.Array, p: float = 0.5):
    return jnp.where(jrandom.uniform(rng) < p, hflip(img), img)
