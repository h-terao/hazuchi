import jax.numpy as jnp
import chex


def charbonnier_penalty(x: chex.Array, eps: float = 0.001, alpha: float = 0.5) -> chex.Array:
    """The generalized Charbonnier penalty function."""
    return jnp.power(x**2 + eps**2, alpha)
