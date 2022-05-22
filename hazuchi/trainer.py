from __future__ import annotations
from typing import Callable, Any, Mapping, Sequence, Tuple
import warnings

import jax
import jax.numpy as jnp
from flax import jax_utils, traverse_util, serialization
import chex

from .observation import Observation
from .callbacks.callback import Callback

TrainState = chex.PyTreeDef
Batch = Any
Logger = Callback
TrainFun = Callable[[TrainState, Batch], Tuple[TrainState, Observation]]
EvalFun = Callable[[TrainState, Batch], Observation]


def _cycle(dataset):
    while True:
        for batch in dataset:
            yield batch


@jax.jit
def _estimate_batch_size(batch: Batch) -> int:
    """Estimate batch size.
    This function digs batch by the depth-first search,
    and returns len(x), where x denotes the firstly found array.
    """
    if isinstance(batch, Mapping):
        for v in traverse_util.flatten_dict(batch).values():
            return _estimate_batch_size(v)
    elif isinstance(batch, Sequence):
        return _estimate_batch_size(batch[0])
    else:
        return len(batch)


def _split_batch(batch: Batch, chunk_size: int = None) -> tuple[Batch | None, Batch | None]:
    if chunk_size is None:
        chunk_size = jax.device_count()

    batch_size = _estimate_batch_size(batch)
    remain_size = batch_size % chunk_size  # remain_size < chunk_size.
    main_size = batch_size - remain_size

    if main_size > 0:
        main_batch = jax.tree_map(
            lambda x: jnp.reshape(
                x[:main_size], (chunk_size, batch_size // chunk_size) + x.shape[1:]
            ),
            batch,
        )
    else:
        main_batch = None

    if remain_size > 0:
        remain_batch = jax.tree_map(
            lambda x: jnp.stack([x[main_size:] for _ in range(chunk_size)], axis=0),
            batch,
        )
    else:
        remain_batch = None

    return main_batch, remain_batch


class Trainer:
    """Trainer class to train and evaluate neural networks.

    Args:
        train_fun (Callable): A step function that updates train_state once.
        eval_fun (Callable): A step function that computes eval metrics from a batch.
        max_epochs (int): Number of epochs. -1 is equal to inf.
        val_interval (int): Interval of epochs to call eval_fun.
        callbacks (dict of str: Callback, optional): Callbacks to use.
    """

    def __init__(
        self,
        train_fun: TrainFun,
        eval_fun: EvalFun,
        max_epochs: int = -1,
        val_interval: int = 1,
        callbacks: dict[str, Callback] | None = None,
    ):
        if callbacks is None:
            callbacks = {}

        self.train_fun = jax.pmap(train_fun, axis_name="batch")
        self.eval_fun = jax.pmap(eval_fun, axis_name="batch")

        self.max_epochs = max_epochs
        self.val_interval = val_interval

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

        train_state = jax_utils.replicate(train_state)

        for callback in self._callback_iterator():
            train_state = callback.on_fit_start(self, train_state)

        # early stopping will set fitted=True.
        while not self.fitted:
            # max_epochs=-1 is inf loop.
            if self.max_epochs >= 0 and self.current_epoch >= self.max_epochs:
                break

            for callback in self._callback_iterator():
                train_state = callback.on_fit_epoch_start(self, train_state)

            train_state, summary = self.train_loop(train_state, train_data, train_steps_per_epoch)
            if val_data and self.current_epoch % self.val_interval == 0:
                train_state, val_summary = self.val_loop(train_state, val_data, val_steps_per_epoch)
                summary = dict(summary, **val_summary)

            for callback in self._callback_iterator():
                train_state, summary = callback.on_fit_epoch_end(self, train_state, summary)

        for callback in self._callback_iterator():
            train_state = callback.on_fit_end(self, train_state)

        self.fitted = True

        train_state = jax_utils.unreplicate(train_state)
        return train_state

    def test(
        self,
        train_state,
        test_data,
        test_fun: EvalFun | None = None,
        prefix: str | None = None,
        test_steps_per_epoch: int | None = None,
    ):
        """Evaluate model.

        Args:
            train_state: Train state that holds parameters.
            train_data: Iterable object that yields batches of train data.
            val_data: Iterable object that yields batches of val data. If None, the model is not evaluated in fit.
            train_steps_per_epoch (int): Number of train steps per epoch. If -1, use len(train_data).
            val_steps_per_epoch (int): Number of val steps per epoch. If -1, use len(val_data).

        Returns:
            Summary of evaluate results.
        """
        if test_steps_per_epoch == -1:
            test_steps_per_epoch = len(test_data)
        assert test_steps_per_epoch >= 0, "test_steps_per_epoch should be positive integer or -1."
        self.test_steps_per_epoch = test_steps_per_epoch

        if prefix is None:
            prefix = "test/"

        train_state = jax_utils.replicate(train_state)

        for callback in self._callback_iterator():
            train_state = callback.on_test_start(self, train_state)

        train_state, summary = self.test_loop(
            train_state=train_state,
            dataset=test_data,
            test_fun=test_fun,
            prefix=prefix,
            test_steps_per_epoch=test_steps_per_epoch,
        )

        for callback in self._callback_iterator():
            train_state = callback.on_test_end(self, train_state)

        train_state = jax_utils.unreplicate(train_state)
        return train_state, summary

    def train_loop(self, train_state, dataset, train_steps_per_epoch: int):
        prefix = "train/"
        num_devices = jax.local_device_count()

        for callback in self._callback_iterator():
            train_state = callback.on_train_epoch_start(self, train_state)

        observation = Observation()
        for batch_idx, batch in enumerate(_cycle(dataset)):
            for callback in self._callback_iterator():
                train_state = callback.on_train_step_start(self, train_state)

            main_batch, remain_batch = _split_batch(batch, num_devices)

            step_observation = Observation()
            if main_batch is not None:
                train_state, obs = self.train_fun(train_state, main_batch)
                step_observation += obs

            if remain_batch is not None:
                warnings.warn(
                    (
                        f"Batch size is not divisible by the number of devices {num_devices}. "
                        "This configuration may causes unexpected training results."
                    )
                )
                train_state, obs = self.train_fun(train_state, remain_batch)
                step_observation += obs / num_devices

            summary = step_observation.scalar_summary(
                prefix=prefix, step=self.global_step, epoch=self.current_epoch
            )
            for callback in self._callback_iterator():
                train_state, summary = callback.on_train_step_end(self, train_state, summary)

            self.global_step += 1
            observation += step_observation
            if batch_idx + 1 == train_steps_per_epoch:
                break

        summary = observation.scalar_summary(
            prefix=prefix, step=self.global_step, epoch=self.current_epoch
        )
        for callback in self._callback_iterator():
            train_state, summary = callback.on_train_epoch_end(self, train_state, summary)

        self.current_epoch += 1
        return train_state, summary

    def val_loop(self, train_state, dataset, val_steps_per_epoch: int):
        prefix = "val/"
        num_devices = jax.local_device_count()

        for callback in self._callback_iterator():
            train_state = callback.on_val_epoch_start(self, train_state)

        observation = Observation()
        for batch_idx, batch in enumerate(_cycle(dataset)):
            for callback in self._callback_iterator():
                train_state = callback.on_val_step_start(self, train_state)

            main_batch, remain_batch = _split_batch(batch, num_devices)
            step_observation = Observation()
            if main_batch is not None:
                step_observation += self.eval_fun(train_state, main_batch)
            if remain_batch is not None:
                step_observation += self.eval_fun(train_state, remain_batch) / num_devices

            summary = observation.scalar_summary(
                prefix=prefix, step=self.global_step, epoch=self.current_epoch
            )
            for callback in self._callback_iterator():
                train_state, summary = callback.on_val_step_end(self, train_state, summary)

            observation += step_observation
            if batch_idx + 1 == val_steps_per_epoch:
                break

        summary = observation.scalar_summary(
            prefix=prefix, step=self.global_step, epoch=self.current_epoch
        )
        for callback in self._callback_iterator():
            train_state, summary = callback.on_val_epoch_end(self, train_state, summary)

        return train_state, summary

    def test_loop(
        self, train_state, dataset, test_fun: EvalFun | None, prefix: str, test_steps_per_epoch: int
    ):
        if test_fun is None:
            test_fun = self.eval_fun

        num_devices = jax.local_device_count()

        for callback in self._callback_iterator():
            train_state = callback.on_test_epoch_start(self, train_state)

        observation = Observation()
        for batch_idx, batch in enumerate(_cycle(dataset)):
            for callback in self._callback_iterator():
                train_state = callback.on_test_step_start(self, train_state)

            main_batch, remain_batch = _split_batch(batch, num_devices)
            step_observation = Observation()
            if main_batch is not None:
                step_observation += test_fun(train_state, main_batch)
            if remain_batch is not None:
                step_observation += test_fun(train_state, remain_batch) / num_devices

            summary = observation.scalar_summary(
                prefix=prefix, step=self.global_step, epoch=self.current_epoch
            )
            for callback in self._callback_iterator():
                train_state, summary = callback.on_test_step_end(self, train_state, summary)

            observation += step_observation
            if batch_idx + 1 == test_steps_per_epoch:
                break

        summary = observation.scalar_summary(
            prefix=prefix, step=self.global_step, epoch=self.current_epoch
        )
        for callback in self._callback_iterator():
            train_state, summary = callback.on_test_epoch_end(self, train_state, summary)

        return train_state, summary

    def to_state_dict(self) -> dict[str, Any]:
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
        for callback in self._callback_iterator():
            if hasattr(callback, "log_hyperparams"):
                callback.log_hyperparams(config)


serialization.register_serialization_state(
    Trainer,
    ty_to_state_dict=lambda ty: ty.to_state_dict(),
    ty_from_state_dict=lambda ty, state: ty.from_state_dict(state),
)
