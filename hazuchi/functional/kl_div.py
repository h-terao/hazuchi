from __future__ import annotations

import jax
import jax.numpy as jnp
import chex


def kl_div(logits: chex.Array, targets: chex.Array, log_target: bool = False) -> chex.Array:
    """KL divergence.

    Args:
        logits: Logits.
        targets: Target labels.
        log_target: If True, targets is considered as log targets.
    """
    if log_target:
        targets, log_targets = jnp.exp(targets), targets
    else:
        log_targets = jnp.log(targets + 1e-8)
    return jnp.sum(targets * (log_targets - jax.nn.log_softmax(logits)), axis=-1)
