import jax
import jax.numpy as jnp
import chex


def quantize(img: chex.Array) -> chex.Array:
    """Quantize pixel velues as integers.

    Returns int(x * 255) / 255.
    """
    quantized = (255 * img).astype(jnp.uint8)
    quantized /= 255
    quantized = img + jax.lax.stop_gradient(quantized - img)
    return quantized
