from __future__ import annotations

import jax
import chex


def bind_rng_to_host_device(
    rng: chex.PRNGKey, axis_name: str, bind_to: str | None = None
) -> chex.Array:
    """Binds a rng to the host/device we are on.
    Must be called from within a pmapped function. Note that when binding to
    "device", we also bind the rng to hosts, as we fold_in the rng with axis_index
    which is unique for devices across all hosts.

    Modify from: https://github.com/google-research/scenic/blob/main/scenic/train_lib/train_utils.py

    Args:
        rng: A jax.random.PRNGKey.
        axis_name: The axis of the devices we are binding rng across.
        bind_to: Must be one of the 'host' or 'device'. None means no binding.

    Returns:
        jax.random.PRNGKey specialized to host/device.
    """
    if bind_to is None:
        return rng
    elif bind_to == "host":
        return jax.random.fold_in(rng, jax.process_index())
    elif bind_to == "device":
        return jax.random.fold_in(rng, jax.lax.axis_index(axis_name))
    else:
        raise ValueError("`bind_to` should be one of the `[None, 'host', 'device']`")
