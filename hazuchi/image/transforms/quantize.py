import jax
import jax.numpy as jnp
import chex


def quantize(img: chex.Array) -> chex.Array:
    """Quantize pixel velues as integers.

    Compute int(x * 255) / 255, and apply STE to be differentiable.

    Args:
        img (array): An image array. Each pixel value is expected to be in [0,1].

    Returns:
        Quantized image array.
    """
    quantized = (255 * img).astype(jnp.uint8)
    quantized /= 255
    quantized = img + jax.lax.stop_gradient(quantized - img)
    return quantized
