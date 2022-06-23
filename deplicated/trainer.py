from __future__ import annotations
from typing import Callable, Any, Mapping, Tuple

import jax
import jax.numpy as jnp
import chex

from .callbacks.callback import Callback
from . import utils

TrainState = chex.PyTreeDef
Batch = Any
Logger = Callback
TrainFun = Callable[[TrainState, Batch], Tuple[TrainState, Mapping[str, chex.Array]]]
EvalFun = Callable[[TrainState, Batch], Mapping[str, chex.Array]]


def split_and_yield(ds):
    num_devices = jax.local_device_count()
    for batch in ds:
        batch_size = len(jax.tree_leaves(batch)[0])
        remain_size = batch_size % num_devices
        main_size = batch_size - remain_size
        if main_size > 0:
            main_batch = jax.tree_map(
                lambda x: jnp.reshape(
                    x[:main_size], (num_devices, main_size // num_devices) + x.shape[1:]
                ),
                batch,
            )
            yield {"batch": main_batch, "weight": utils.replicate(main_size)}
        if remain_size > 0:
            remain_batch = jax.tree_map(
                lambda x: jnp.stack([x[main_size:] for _ in range(num_devices)], axis=0),
                batch,
            )
            yield {"batch": remain_batch, "weight": utils.replicate(remain_size)}


class Trainer:
    """Trainer class to train and evaluate neural networks.

    Args:
        train_fun (Callable): A step function that updates train_state once.
        eval_fun (Callable): A step function that computes eval metrics from a batch.
        max_epochs (int): Number of epochs. -1 is equal to inf.
        val_every (int): Interval of epochs to call eval_fun.
        callbacks (dict, optional): Callbacks.
    """

    def __init__(
        self,
        train_fun: TrainFun,
        eval_fun: EvalFun,
        max_epochs: int = -1,
        val_every: int = 1,
        callbacks: dict[str, Callback] | None = None,
    ):
        if callbacks is None:
            callbacks = {}

        self.train_fun = jax.pmap(train_fun, axis_name="batch")
        self.eval_fun = jax.pmap(eval_fun, axis_name="batch")

        self.max_epochs = max_epochs
        self.val_every = val_every

        self._callbacks = callbacks

        self.global_step = 0
        self.current_epoch = 0
        self.fitted = False

        self.train_steps_per_epoch = None
        self.val_steps_per_epoch = None
        self.test_steps_per_epoch = None

    def _callback_iterator(self, reverse: bool = False):
        for callback in sorted(
            self._callbacks.values(), key=lambda v: v.priority, reverse=not reverse
        ):
            yield callback

    def _merge_scalars(
        self, prev_scalars: Mapping[str, float], scalars: Mapping[str, float]
    ) -> Mapping[str, float]:
        updates = {}
        for key, (val, weight) in scalars.items():
            prev_val, prev_weight = prev_scalars.get(key, (0, 0))
            accum_val = float(prev_val + val)
            accum_weight = float(prev_weight + weight)
            updates[key] = (accum_val, accum_weight)
        return dict(prev_scalars, **updates)

    def _summarize_scalars(
        self, scalars: Mapping[str, float], prefix: str, **kwargs
    ) -> Mapping[str, int | float]:
        summary = {prefix + k: v / w for k, (v, w) in scalars.items()}
        summary = dict(summary, **kwargs)
        return summary

    def fit(
        self,
        train_state: chex.PyTreeDef,
        train_data: Any,
        val_data: Any | None = None,
        train_steps_per_epoch: int = -1,
        val_steps_per_epoch: int = -1,
    ) -> Any:
        """Train models.

        Args:
            train_state: Train state that holds parameters.
            train_data: Iterable object that yields batches of train data.
            val_data: Iterable object that yields batches of val data.
                If None, the model is not evaluated in fit.
            train_steps_per_epoch (int): Number of train steps per epoch.
                If -1, use len(train_data).
            val_steps_per_epoch (int): Number of val steps per epoch.
                If -1, use len(val_data).

        Returns:
            Fitted train state.
        """
        if train_steps_per_epoch == -1:
            train_steps_per_epoch = len(train_data)
        assert train_steps_per_epoch >= 0, "train_steps_per_epoch should be positive integer or -1."
        self.train_steps_per_epoch = train_steps_per_epoch

        if val_steps_per_epoch == -1 and val_data is not None:
            val_steps_per_epoch = len(val_data)
        assert val_steps_per_epoch >= 0, "val_steps_per_epoch should be positive integer or -1."
        self.val_steps_per_epoch = val_steps_per_epoch

        train_state = utils.replicate(train_state)

        for callback in self._callback_iterator():
            train_state = callback.on_fit_start(self, train_state)

        # early stopping will set fitted=True.
        while not self.fitted:
            # max_epochs=-1 is inf loop.
            if self.max_epochs >= 0 and self.current_epoch >= self.max_epochs:
                break

            for callback in self._callback_iterator():
                train_state = callback.on_fit_epoch_start(self, train_state)

            train_state, summary = self._train_loop(train_state, train_data, train_steps_per_epoch)
            if val_data and self.current_epoch % self.val_every == 0:
                train_state, val_summary = self._val_loop(
                    train_state, val_data, val_steps_per_epoch
                )
                summary = dict(summary, **val_summary)

            for callback in self._callback_iterator():
                train_state, summary = callback.on_fit_epoch_end(self, train_state, summary)

        for callback in self._callback_iterator():
            train_state = callback.on_fit_end(self, train_state)

        self.fitted = True

        train_state = utils.unreplicate(train_state)
        return train_state

    def test(
        self,
        train_state,
        test_data,
        test_fun: EvalFun | None = None,
        prefix: str | None = None,
        test_steps_per_epoch: int = -1,
    ):
        """Evaluate model.

            train_state: Train state that holds parameters.
            test_data: Iterable object that yields batches of train data.
            test_fun (Callable, optional): A step function to test models.
                If None, use eval_fun specified in Trainer.__init__.
            prefix (str, optional): A prefix string added when creates summary.
            test_steps_per_epoch (int): Number of test steps per epoch.
                If -1, len(test_data) is used.

        Returns:
            Summary of evaluate results.
        """
        if test_steps_per_epoch == -1:
            test_steps_per_epoch = len(test_data)
        assert test_steps_per_epoch >= 0, "test_steps_per_epoch should be positive integer or -1."
        self.test_steps_per_epoch = test_steps_per_epoch

        if prefix is None:
            prefix = "test/"

        train_state = utils.replicate(train_state)

        for callback in self._callback_iterator():
            train_state = callback.on_test_start(self, train_state)

        train_state, summary = self._test_loop(
            train_state=train_state,
            dataset=test_data,
            test_fun=test_fun,
            prefix=prefix,
            test_steps_per_epoch=test_steps_per_epoch,
        )

        for callback in self._callback_iterator():
            train_state = callback.on_test_end(self, train_state)

        train_state = utils.unreplicate(train_state)
        return train_state, summary

    def _train_loop(self, train_state, dataset, train_steps_per_epoch: int):
        prefix = "train/"
        for callback in self._callback_iterator():
            train_state = callback.on_train_epoch_start(self, train_state)

        scalars = {}
        for batch_idx, batch in enumerate(utils.double_buffer(split_and_yield(dataset))):
            batch, weight = batch["batch"], batch["weight"]
            weight = float(weight[0].block_until_ready())

            for callback in self._callback_iterator():
                train_state = callback.on_train_step_start(self, train_state)

            train_state, step_scalars = self.train_fun(train_state, batch)
            step_scalars = {
                k: (float(jnp.mean(v).block_until_ready()) * weight, weight)
                for k, v in step_scalars.items()
            }

            summary = self._summarize_scalars(
                step_scalars, prefix=prefix, step=self.global_step, epoch=self.current_epoch
            )
            for callback in self._callback_iterator():
                train_state, summary = callback.on_train_step_end(self, train_state, summary)

            self.global_step += 1
            scalars = self._merge_scalars(scalars, step_scalars)
            if batch_idx + 1 == train_steps_per_epoch:
                break

        summary = self._summarize_scalars(
            scalars, prefix=prefix, step=self.global_step, epoch=self.current_epoch
        )
        for callback in self._callback_iterator():
            train_state, summary = callback.on_train_epoch_end(self, train_state, summary)

        self.current_epoch += 1
        return train_state, summary

    def _val_loop(self, train_state, dataset, val_steps_per_epoch: int):
        prefix = "val/"

        for callback in self._callback_iterator():
            train_state = callback.on_val_epoch_start(self, train_state)

        scalars = {}
        for batch_idx, batch in enumerate(utils.double_buffer(split_and_yield(dataset))):
            batch, weight = batch["batch"], batch["weight"]
            weight = float(weight[0])

            for callback in self._callback_iterator():
                train_state = callback.on_val_step_start(self, train_state)

            step_scalars = self.eval_fun(train_state, batch)
            step_scalars = {
                k: (float(jnp.mean(v).block_until_ready()) * weight, weight)
                for k, v in step_scalars.items()
            }

            summary = self._summarize_scalars(
                step_scalars, prefix=prefix, step=self.global_step, epoch=self.current_epoch
            )
            for callback in self._callback_iterator():
                train_state, summary = callback.on_val_step_end(self, train_state, summary)

            scalars = self._merge_scalars(scalars, step_scalars)
            if batch_idx + 1 == val_steps_per_epoch:
                break

        summary = self._summarize_scalars(
            scalars, prefix=prefix, step=self.global_step, epoch=self.current_epoch
        )
        for callback in self._callback_iterator():
            train_state, summary = callback.on_val_epoch_end(self, train_state, summary)

        return train_state, summary

    def _test_loop(
        self, train_state, dataset, test_fun: EvalFun | None, prefix: str, test_steps_per_epoch: int
    ):
        if test_fun is None:
            test_fun = self.eval_fun

        for callback in self._callback_iterator():
            train_state = callback.on_test_epoch_start(self, train_state)

        scalars = {}
        for batch_idx, batch in enumerate(utils.double_buffer(split_and_yield(dataset))):
            batch, weight = batch["batch"], batch["weight"]
            weight = float(weight[0])

            for callback in self._callback_iterator():
                train_state = callback.on_test_step_start(self, train_state)

            step_scalars = test_fun(train_state, batch, weight)
            step_scalars = {
                k: (float(jnp.mean(v).block_until_ready()) * weight, weight)
                for k, v in step_scalars.items()
            }

            summary = self._summarize_scalars(
                step_scalars, prefix=prefix, step=self.global_step, epoch=self.current_epoch
            )
            for callback in self._callback_iterator():
                train_state, summary = callback.on_test_step_end(self, train_state, summary)

            scalars = self._merge_scalars(scalars, step_scalars)
            if batch_idx + 1 == test_steps_per_epoch:
                break

        summary = self._summarize_scalars(
            scalars, prefix=prefix, step=self.global_step, epoch=self.current_epoch
        )
        for callback in self._callback_iterator():
            train_state, summary = callback.on_test_epoch_end(self, train_state, summary)

        return train_state, summary

    def to_state_dict(self) -> dict[str, Any]:
        """Returns trainer state."""
        to_save = {
            "global_step": self.global_step,
            "current_epoch": self.current_epoch,
            "fitted": self.fitted,
            "callbacks": {},
        }
        for key, callback in self._callbacks.items():
            to_save["callbacks"][key] = callback.to_state_dict()
        return to_save

    def from_state_dict(self, state: dict[str, Any]) -> Trainer:
        """Restore trainer from the given state."""
        self.global_step = state["global_step"]
        self.current_epoch = state["current_epoch"]
        self.fitted = state["fitted"]
        for key, callback in self._callbacks.items():
            callback_state = state["callbacks"].get(key, None)
            if callback_state is None:
                # TODO: Use logger.
                print("Failed to load...")
            else:
                callback.from_state_dict(callback_state)
        return self

    def log_hyperparams(self, config):
        """Logging hyperparameters to all callbacks that have log_hyperparams method.

        Args:
            config (dict of Any): Hyperparameters.

        Return:
            None.
        """
        for callback in self._callback_iterator():
            if hasattr(callback, "log_hyperparams"):
                callback.log_hyperparams(config)