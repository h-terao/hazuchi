# Borrowed from https://github.com/google/flax/blob/main/flax/jax_utils.py
import jax
import jax.lib.xla_bridge as xb


def _pmap_device_order():
    # match the default device assignments used in pmap:
    # for single-host, that's the XLA default device assignment
    # for multi-host, it's the order of jax.local_devices()
    if jax.process_count() == 1:
        return [
            d
            for d in xb.get_backend().get_default_device_assignment(jax.device_count())
            if d.process_index == jax.process_index()
        ]
    else:
        return jax.local_devices()


def replicate(tree, devices=None):
    """Replicates arrays to multiple devices.

    Args:
      tree: a pytree containing the arrays that should be replicated.
      devices: the devices the data is replicated to
        (default: same order as expected by `jax.pmap()`).

    Returns:
      A new pytree containing the replicated arrays.
    """
    devices = devices or _pmap_device_order()
    return jax.device_put_replicated(tree, devices)


def unreplicate(tree):
    """Returns a single instance of a replicated array."""
    return jax.tree_map(lambda x: x[0], tree)
