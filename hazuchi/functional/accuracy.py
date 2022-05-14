import jax.numpy as jnp
import chex


def accuracy(logits: chex.Array, labels: chex.Array, k: int = 1) -> chex.Array:
    """Top-k accuracy score.
    Args:
        logits: logits. (..., num_class)
        labels: one-hot labels. (..., num_class)
    Returns:
        Array that has a shape of (...).
        Call accuracy().mean() to obtain scalar value.
    """
    assert logits.ndim in (labels.ndim, labels.ndim + 1)

    preds = jnp.argsort(logits)[..., -k:]
    if logits.ndim == labels.ndim:
        labels = jnp.argmax(labels, axis=-1, keepdims=True)
    else:
        labels = labels[..., None]
    return jnp.sum(preds == labels, axis=-1)
