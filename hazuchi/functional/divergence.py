import jax
import jax.numpy as jnp
import chex


def kl_div(logits: chex.Array, targets: chex.Array, log_targets: bool = False) -> chex.Array:
    """The Kullback-Leibler divergence.

    Args:
        logits: Logit array.
    """
    log_preds = jax.nn.log_softmax(logits)
    targets, log_targets = jax.lax.cond(log_targets, lambda t: (jnp.exp(t), t), lambda t: (t, jnp.log(t)), targets)
    return jnp.sum(targets * (log_preds - log_targets), axis=-1)


def js_div(logits: chex.Array, targets: chex.Array, log_targets: bool = False) -> chex.Array:
    """The Jensen-Shannon divergence.

    Unlike kl_div, js_div(p, q) == js_div(q, p).
    """
    preds = jax.nn.softmax(logits)
    targets, log_targets = jax.lax.cond(log_targets, lambda t: (jnp.exp(t), t), lambda t: (t, jnp.log(t)), targets)
    y = (preds + targets) / 2
    return (kl_div(logits, y) + kl_div(log_targets, y)) / 2.0


def cross_entropy(logits: chex.Array, labels: chex.Array, label_smoothing: float = 0.0) -> chex.Array:
    """The cross-entropy loss."""
    num_classes = logits.shape[-1]
    if logits.ndim == labels.ndim + 1:
        labels = jax.nn.one_hot(labels, num_classes)
        # Apply label_smoothing.
        off_value = label_smoothing / num_classes
        on_value = 1.0 - label_smoothing + off_value
        labels = jnp.where(labels > 0, on_value, off_value)
    log_preds = jax.nn.log_softmax(logits)
    return -jnp.sum(labels * log_preds, axis=-1)
