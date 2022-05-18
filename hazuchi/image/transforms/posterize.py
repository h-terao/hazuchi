import jax
import jax.numpy as jnp
import chex


def posterize(img: chex.Array, bits: int) -> chex.Array:
    assert 0 <= bits <= 8
    n = 8 - bits

    degenerate = (img * 255).astype(jnp.uint8)
    degenerate = jnp.left_shift(jnp.right_shift(degenerate, n), n)
    degenerate = degenerate.astype(img.dtype) / 255.0

    # Straight Through Estimator
    degenerate = img + jax.lax.stop_gradient(degenerate - img)
    return degenerate
