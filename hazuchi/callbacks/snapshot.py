from __future__ import annotations
from typing import Any, Callable
import math
import operator
from pathlib import Path

from . import callback
from .. import utils

TrainState = Any


class Snapshot(callback.Callback):
    """Save the snapshot of training.

    Args:
        filename
        monitor (str | None): Name of metrics to monitor.
                              If None, save the latest checkpoint.

    Todo:
        Reprecated / not.
    """

    _priority: int = callback.PRIORITY_SNAPSHOT

    def __init__(
        self,
        save_dir,
        filename,
        monitor: str | None = None,
        mode: str = "min",
        compresslevel: int = 9,
        load_train_state: Callable[[TrainState, Any], TrainState] | None = None,
        save_train_state: Callable[[TrainState], Any] | None = None,
    ) -> None:
        self.save_dir = save_dir
        self.filename = filename
        self.monitor = monitor
        self.mode = mode
        self.compresslevel = compresslevel

        self.compare = operator.lt if mode == "min" else operator.gt
        self.best_score = math.inf if mode == "min" else -math.inf

        if load_train_state is None:
            load_train_state = lambda train_state, state: state  # noqa: E731
        if save_train_state is None:
            save_train_state = lambda train_state: train_state  # noqa: E731
        self.load_train_state = load_train_state
        self.save_train_state = save_train_state

    def on_fit_epoch_end(self, trainer, train_state, summary):
        if self.monitor is None:
            self.save(trainer, utils.unreplicate(train_state))
        elif self.monitor in summary:
            if self.compare(summary[self.monitor], self.best_score):
                self.best_score = summary[self.monitor]
                self.save(trainer, utils.unreplicate(train_state))
        return train_state, summary

    def save(self, trainer, train_state: TrainState) -> None:
        Path(self.save_dir).mkdir(parents=True, exist_ok=True)

        state = {
            "trainer": trainer.to_state_dict(),
            "train_state": self.save_train_state(train_state),
        }
        utils.serialization.save_state(self.snapshot_path, state, compresslevel=self.compresslevel)

    def load(
        self,
        trainer,
        train_state: TrainState,
        only_train_state: bool = False,
        strict: bool = False,
    ):
        if self.exists():
            snapshot = utils.serialization.load_state(self.snapshot_path)
            if not only_train_state:
                trainer = trainer.from_state_dict(snapshot["trainer"])
            train_state = self.load_train_state(train_state, snapshot["train_state"])
            return trainer, train_state
        elif not strict:
            return trainer, train_state
        else:
            raise FileNotFoundError(f"{self.snapshot_path} is not found.")

    def to_state_dict(self):
        return {"best": self.best_score}

    def from_state_dict(self, state) -> None:
        self.best_score = state["best"]

    @property
    def priority(self) -> int:
        if self.monitor is None:
            return self._priority - 100
        else:
            return self._priority

    @property
    def save_last(self) -> bool:
        return self.monitor is None

    def exists(self) -> bool:
        return self.snapshot_path.exists()

    @property
    def snapshot_path(self) -> Path:
        return Path(self.save_dir, self.filename)
