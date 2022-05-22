from __future__ import annotations
import math
import operator
from pathlib import Path

from flax.training.train_state import TrainState
from flax import jax_utils
from flax import serialization as flax_serialization

from . import callback
from ..utils import serialization


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
    ) -> None:
        self.save_dir = save_dir
        self.filename = filename
        self.monitor = monitor
        self.mode = mode
        self.compresslevel = compresslevel

        self.compare = operator.lt if mode == "min" else operator.gt
        self.best_score = math.inf if mode == "min" else -math.inf

    def on_fit_epoch_end(self, trainer, train_state, summary):
        if self.monitor is None:
            self.save(trainer, jax_utils.unreplicate(train_state))
        elif self.monitor in summary:
            # best?
            if self.compare(summary[self.monitor], self.best_score):
                self.best_score = summary[self.monitor]
                self.save(trainer, jax_utils.unreplicate(train_state))
        return train_state, summary

    def save(self, trainer, train_state: TrainState) -> None:
        Path(self.save_dir).mkdir(parents=True, exist_ok=True)

        state = {
            "trainer": flax_serialization.to_state_dict(trainer),
            "train_state": flax_serialization.to_state_dict(train_state),
        }
        serialization.save_state(self.snapshot_path, state, compresslevel=self.compresslevel)

    def load(
        self,
        trainer,
        train_state: TrainState,
        only_train_state: bool = False,
        strict: bool = False,
    ):
        if self.exists():
            snapshot = serialization.load_state(self.snapshot_path)
            if not only_train_state:
                trainer = flax_serialization.from_state_dict(trainer, snapshot["trainer"])
            train_state = flax_serialization.from_state_dict(train_state, snapshot["train_state"])
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
