import jax
import jax.numpy as jnp
import chex

from .core import flatten, unflatten


def equalize(img: chex.Array) -> chex.Array:
    @jax.jit
    def build_lut(histo, step):
        # Compute the cumulative sum, shifting by step // 2
        # and then normalization by step.
        lut = (jnp.cumsum(histo) + (step // 2)) // step
        # Shift lut, prepending with 0.
        lut = jnp.concatenate([jnp.array([0]), lut[:-1]], 0)
        # Clip the counts to be in range.  This is done
        # in the C code for image.point.
        return jnp.clip(lut, 0, 255)

    @jax.jit
    def scale_channel(img):
        """
        Scale the data in the channel to implement equalize.
        Args:
            img: channel to scale.
        Returns:
            scaled channel
        """
        # im = im[:, :, c].astype('int32')
        img = img.astype("int32")
        # Compute the histogram of the image channel.
        histo = jnp.histogram(img, bins=255, range=(0, 255))[0]

        last_nonzero = jnp.argmax(histo[::-1] > 0)  # jnp.nonzero(histo)[0][-1]
        step = (jnp.sum(histo) - jnp.take(histo[::-1], last_nonzero)) // 255

        # If step is zero, return the original image.  Otherwise, build
        # lut from the full histogram and step and then index from it.
        return jax.lax.cond(
            step == 0,
            lambda x: x.astype("uint8"),
            lambda x: jnp.take(build_lut(histo, step), x).astype("uint8"),
            img,
        )

    img, original_shape = flatten(img)
    dtype = img.dtype

    degenerate = (img * 255).astype(jnp.uint8)
    _, height, width, channel = degenerate.shape

    degenerate = degenerate.transpose(0, 3, 1, 2).reshape(-1, height, width)
    _, degenerate = jax.lax.scan(
        lambda carry, x: (carry, scale_channel(x)),
        jnp.zeros(()),
        degenerate,
    )

    degenerate = degenerate.astype(dtype) / 255.0
    degenerate = degenerate.reshape(-1, channel, height, width).transpose(0, 2, 3, 1)
    degenerate = unflatten(degenerate, original_shape)

    # Straight Through Estimator
    degenerate = img + jax.lax.stop_gradient(degenerate - img)
    return degenerate
