from __future__ import annotations
from typing import Callable
from dataclasses import dataclass, field
from pathlib import Path

import jax
from flax import jax_utils
from flax.serialization import register_serialization_state

from .split_batches import split_batches
from .scalars import accumulate_scalars, summarize_scalars


@dataclass
class Trainer:
    """Trainer.

    Args:
        out_dir: Output directory.
        max_epochs (int): Maximum number of epochs.
            If a negative value is specified, trainer fit models infinite times.
        train_fn
        val_fn
        test_fn
        callbacks
        val_every
        prefetch
        devices
    """

    out_dir: str
    max_epochs: int
    train_fn: Callable
    val_fn: Callable
    test_fn: Callable | None = None
    callbacks: dict = field(default_factory=dict)
    val_every: int = 1
    prefetch: bool = False
    devices: list | None = None

    axis_name: str = "batch"

    def __post_init__(self):
        self._sort_callbacks()
        self.out_dir_path = Path(self.out_dir)

        self.out_dir_path.mkdir(parents=True, exist_ok=True)

        self.global_step: int = 0
        self.current_epoch: int = 0
        self.fitted: bool = False

    def register_callback(self, key: str, callback, overwrite: bool = False):
        if not overwrite and key in self.callbacks:
            raise ValueError(
                f"{key} is already registered. Use different key to register callbacks."
            )
        self.callbacks[key] = callback
        self._sort_callbacks()

    def _sort_callbacks(self):
        self._sorted_callbacks = sorted(
            self.callbacks.values(), key=lambda x: x.priority, reverse=True
        )

    def _loop_callbacks(self, reverse: bool = False):
        """yield every registered callback.

        Args:
            reverse (bool): If False: priority 1 -> 100
                If True: priority 100 -> 1
        """
        callbacks = self._sorted_callbacks
        if reverse:
            callbacks = reversed(callbacks)
        for callback in callbacks:
            yield callback

    def _loop_epoch(self, train_state, fn, iterator, prefix, train: bool, iter_len=None):
        accum_scalars = {}
        for batch, weight in split_batches(iterator, iter_len, self.prefetch, self.devices):
            print(jax.tree_map(lambda x: x.shape, batch))
            if train:
                train_state, scalars = fn(train_state, batch)
            else:
                scalars = fn(train_state, batch)
            accum_scalars = accumulate_scalars(accum_scalars, scalars, weight)
        summary = summarize_scalars(prefix, accum_scalars)
        return train_state, summary

    def fit(
        self,
        train_state,
        train_iter,
        val_iter,
        *,
        train_len: int | None = None,
        val_len: int | None = None,
    ):
        train_state = jax_utils.replicate(train_state, self.devices)
        p_train_fn = jax.pmap(self.train_fn, axis_name=self.axis_name)
        p_val_fn = jax.pmap(self.val_fn, axis_name=self.axis_name)

        for callback in self._loop_callbacks(reverse=True):
            train_state = callback.on_fit_start(self, train_state)

        while True:
            if self.current_epoch == self.max_epochs:
                self.fitted = True

            if self.fitted:
                break

            for callback in self._loop_callbacks():
                train_state = callback.on_fit_epoch_start(self, train_state)
                self.global_step += 1

            train_state, summary = self._loop_epoch(
                train_state,
                fn=p_train_fn,
                iterator=train_iter,
                prefix="train/",
                train=True,
                iter_len=train_len,
            )
            if val_iter is not None and (self.current_epoch + 1) % self.val_every == 0:
                _, val_summary = self._loop_epoch(
                    train_state,
                    fn=p_val_fn,
                    iterator=val_iter,
                    prefix="val/",
                    train=False,
                    iter_len=val_len,
                )
                summary = dict(summary, **val_summary)
            self.current_epoch += 1

            summary["step"] = self.global_step
            summary["epoch"] = self.current_epoch
            for callback in self._loop_callbacks():
                train_state, summary = callback.on_fit_epoch_end(self, train_state, summary)

        for callback in self._loop_callbacks():
            train_state = callback.on_fit_end(self, train_state)

        train_state = jax_utils.unreplicate(train_state)
        return train_state

    def test(self, train_state, test_iter, test_len=None):
        train_state = jax_utils.replicate(train_state, self.devices)
        for callback in self._loop_callbacks():
            train_state = callback.on_test_start(self, train_state)

        for callback in self._loop_callbacks():
            train_state = callback.on_test_epoch_start(self, train_state)

        _, summary = self._loop_epoch(
            train_state,
            fn=jax.pmap(self.test_fn or self.val_fn, axis_name=self.axis_name),
            iterator=test_iter,
            prefix="test/",
            train=False,
            iter_len=test_len,
        )

        for callback in self._loop_callbacks():
            train_state, summary = callback.on_test_epoch_end(self, train_state, summary)

        for callback in self._loop_callbacks():
            train_state = callback.on_test_end(self, train_state)

        return summary

    def log_hyperparams(self, hparams):
        for callback in self._loop_callbacks():
            if hasattr(callback, "log_hyperparams"):
                callback.log_hyperparams(hparams)

    def finalize(self):
        for callback in self._loop_callbacks():
            callback.finalize()


#
#  Register trainer to flax.serialization
#
def ty_to_state_dict(trainer: Trainer):
    return {
        "global_step": trainer.global_step,
        "current_epoch": trainer.current_epoch,
        "fitted": trainer.fitted,
        "callbacks": {key: callback.to_state_dict() for key, callback in trainer.callbacks.items()},
    }


def ty_from_state_dict(trainer: Trainer, state: dict):
    trainer.global_step = state["global_step"]
    trainer.current_epoch = state["current_epoch"]
    trainer.fitted = state["fitted"]
    trainer.callbacks = {
        key: callback.from_state_dict(state["callbacks"][key])
        for key, callback in trainer.callbacks.items()
    }
    trainer._sort_callbacks()
    return trainer


register_serialization_state(Trainer, ty_to_state_dict, ty_from_state_dict)
