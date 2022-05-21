from __future__ import annotations
import functools

import jax
import jax.numpy as jnp
from flax.optim import dynamic_scale as dynamic_scale_lib
from flax.training import train_state
from flax.core import FrozenDict
import optax
import chex

__all__ = ["get_dtype_scaler", "TrainState"]


def get_dtype_scaler(precision: int) -> tuple[chex.ArrayDType, dynamic_scale_lib.DynamicScale]:
    scaler = None
    platform = jax.local_devices()[0].platform

    if precision == 16:
        dtype = jnp.float16
        if platform == "gpu":
            scaler = dynamic_scale_lib.DynamicScale()
        if platform == "tpu":
            dtype = jnp.bfloat16
    elif precision == 32:
        dtype = jnp.float32
    else:
        raise ValueError("precision must be 16 or 32.")

    return dtype, scaler


class TrainState(train_state.TrainState):
    rng: chex.PRNGKey = jax.random.PRNGKey(0)
    model_state: FrozenDict = FrozenDict()
    dynamic_scale: dynamic_scale_lib.DynamicScale | None = None

    def apply_gradients(self, *, grads, is_fin: bool = True, **kwargs):
        updates, new_opt_state = self.tx.update(
            grads,
            self.opt_state,
            self.params,
        )
        new_params = optax.apply_updates(self.params, updates)
        if self.dynamic_scale:
            new_opt_state = jax.tree_map(
                functools.partial(jnp.where, is_fin),
                new_opt_state,
                self.opt_state,
            )
            new_params = jax.tree_map(
                functools.partial(jnp.where, is_fin),
                new_params,
                self.params,
            )
        return self.replace(
            step=self.step + 1,
            params=new_params,
            opt_state=new_opt_state,
            **kwargs,
        )
