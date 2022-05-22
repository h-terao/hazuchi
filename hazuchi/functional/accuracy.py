import jax.numpy as jnp
import chex


def accuracy(logits: chex.Array, labels: chex.Array, k: int = 1) -> chex.Array:
    """Top-k accuracy score.

    Args:
        logits: logits.
        labels: one-hot labels.

    Returns:
        Top-k accuracy.

    Example:
        >>> logits = jnp.array([[0.1, 0.9], [0.9, 0.1], [1.0, 0.0]])
        >>> labels = jnp.array([0, 0, 0])
        >>> accuracy(logits, labels).tolist()
        [0, 1, 0]
        >>> accuracy(logits, labels).mean()
        0.666
        >>> accuracy(logits, logits).mean()  # Use soft-target.
        1.0
    """
    assert logits.ndim in (labels.ndim, labels.ndim + 1)

    preds = jnp.argsort(logits)[..., -k:]
    if logits.ndim == labels.ndim:
        labels = jnp.argmax(labels, axis=-1, keepdims=True)
    else:
        labels = labels[..., None]
    return jnp.sum(preds == labels, axis=-1)
