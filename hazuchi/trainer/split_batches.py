from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import jax_utils


def split_batches(
    iterator, limit: int | None = None, prefetch: bool = False, devices: list | None = None
):
    """
    Args:
        iterator: An iterable data loader.
        limit (int): Number of batches to yield until StopIteration is raised.
        prefetch (bool): If True, prefetch batches on the devices.
            It is useful to accelerate training when you use GPUs.
        devices: Devices.
    """
    devices = devices or jax_utils._pmap_device_order()
    num_devices = len(devices)

    def _split_batch(loader):
        for i, batch in enumerate(loader):
            leaves, treedef = jax.tree_flatten(batch)
            min_batch_size = min(len(x) for x in leaves)
            min_main_size, min_remain_size = divmod(min_batch_size, num_devices)

            main_leaves, remain_leaves = [], []
            for x in leaves:
                r = len(x) / min_batch_size
                main_size, remain_size = int(r * min_main_size), int(r * min_remain_size)
                assert main_size * num_devices + remain_size == len(x)

                if main_size > 0:
                    main_array = x[remain_size:]
                    main_leaves.append(main_array.reshape(num_devices, main_size, *x.shape[1:]))

                if remain_size > 0:
                    remain_array = x[:remain_size]
                    remain_leaves.append(jnp.stack([remain_array] * num_devices))

            if main_leaves:
                main_batch = jax.tree_unflatten(treedef, main_leaves)
                yield main_batch, min_main_size * num_devices

            if remain_leaves:
                remain_batch = jax.tree_unflatten(treedef, remain_leaves)
                yield remain_batch, min_remain_size

            if limit is not None and i + 1 == limit:
                break

    if prefetch:
        batch, size = None, None
        for next_batch, next_size in _split_batch(iterator):
            assert next_batch is not None
            next_batch = jax.tree_map(
                lambda x: jax.device_put_sharded(list(x), devices), next_batch
            )
            if batch is not None:
                yield batch, size
            batch, size = next_batch, next_size
        if batch is not None:
            yield batch, size
    else:
        yield from _split_batch(iterator)
