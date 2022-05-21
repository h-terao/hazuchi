from __future__ import annotations
from typing import Callable

import jax.numpy as jnp
from . import callback


class ObserveLR(callback.Callback):
    """Log learning rate."""

    priority: int = callback.PRIORITY_WRITER

    def __init__(self, learning_rate: float | Callable):
        if jnp.isscalar(learning_rate):
            self.learning_rate_fun = lambda _: learning_rate  # noqa: E731
        else:
            self.learning_rate_fun = learning_rate

    def on_fit_epoch_end(self, trainer, train_state, summary):
        lr = self.learning_rate_fun(trainer.global_step)
        summary["lr"] = lr
        return train_state, summary
