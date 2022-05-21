from __future__ import annotations
from tqdm.rich import tqdm
from rich.progress import (
    BarColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    SpinnerColumn,
)

from . import callback


class ProgressBar(callback.Callback):
    def __init__(self):
        self._pbar = None

    def on_fit_epoch_start(self, trainer, train_state):
        self._pbar = tqdm(
            TextColumn("{task.description}: {task.percentage:.1f}%"),
            SpinnerColumn(),
            BarColumn(),
            TextColumn(" {task.completed:d}/{task.total:d} "),
            TimeElapsedColumn(),
            TextColumn("<"),
            TimeRemainingColumn(),
            total=self.estimate_total_steps(trainer),
            leave=False,
            desc=f"[Epoch: {trainer.current_epoch}]",
        )
        return train_state

    def on_train_step_end(self, trainer, train_state, summary):
        self._pbar.update()
        return train_state, summary

    def on_val_step_end(self, trainer, train_state, summary):
        self._pbar.update()
        return train_state, summary

    def on_fit_epoch_end(self, trainer, train_state, summary):
        self._pbar.close()
        self._pbar = None
        return train_state, summary

    def estimate_total_steps(self, trainer):
        if trainer.current_epoch + 1 % trainer.val_interval == 0:
            return trainer.train_steps_per_epoch + trainer.val_steps_per_epoch
        else:
            return trainer.train_steps_per_epoch
