from __future__ import annotations

from rich.progress import (
    BarColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    SpinnerColumn,
)

from . import callback


class ProgressBar(callback.Callback):
    """TODO: Estimate epoch length."""

    def __init__(self, on_step: bool = False, on_epoch: bool = True):
        self.on_step = on_step
        self.on_epoch = on_epoch

        self._step_task = None
        self._epoch_task = None

    def on_fit_start(self, trainer, train_state):
        prog = Progress(
            TextColumn("{task.description}: {task.percentage:.1f}%"),
            SpinnerColumn(),
            BarColumn(),
            TextColumn(" {task.completed:d}/{task.total:d} "),
            TimeElapsedColumn(),
            TextColumn("<"),
            TimeRemainingColumn(),
        )

        prog.start()

        self._prog = prog

        if self.on_step:
            v = trainer.global_step % trainer.train_steps_per_epoch
            self._step_task = prog.add_task("", total=trainer.train_steps_per_epoch, completed=v)

        if self.on_epoch:
            self._epoch_task = prog.add_task("Training", total=trainer.max_epochs, completed=trainer.current_epoch)

        return train_state

    def on_train_step_end(self, trainer, train_state, summary):
        if self._step_task is not None:
            self._prog.update(self._step_task, advance=1)
        return train_state, summary

    def on_fit_epoch_end(self, trainer, train_state, summary):
        if self._step_task is not None:
            self._prog.update(self._step_task, completed=0)

        if self._epoch_task is not None:
            self._prog.update(self._epoch_task, advance=1)
        return train_state, summary
