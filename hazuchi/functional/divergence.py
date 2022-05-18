import jax
import jax.numpy as jnp
import chex


def kl_div(logits: chex.Array, targets: chex.Array, log_targets: bool = False) -> chex.Array:
    """The Kullback-Leibler divergence.

    Args:
        logits: Logit array.
    """
    log_preds = jax.nn.log_softmax(logits)
    targets, log_targets = jax.lax.cond(
        log_targets, lambda t: (jnp.exp(t), t), lambda t: (t, jnp.log(t)), targets
    )
    return jnp.sum(targets * (log_preds - log_targets), axis=-1)


def js_div(logits: chex.Array, targets: chex.Array, log_targets: bool = False) -> chex.Array:
    """The Jensen-Shannon divergence.

    Unlike kl_div, js_div(p, q) == js_div(q, p).
    """
    preds = jax.nn.softmax(logits)
    targets, log_targets = jax.lax.cond(
        log_targets, lambda t: (jnp.exp(t), t), lambda t: (t, jnp.log(t)), targets
    )
    y = (preds + targets) / 2
    return (kl_div(logits, y) + kl_div(log_targets, y)) / 2.0
