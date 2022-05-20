from __future__ import annotations
import math
import operator
from pathlib import Path

from flax.training.train_state import TrainState
from flax import jax_utils
from . import callback
from ..trainer import Trainer
from ..utils.serialization import load_checkpoint, save_checkpoint


class Checkpoint(callback.Callback):
    """
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
    ) -> None:
        self.save_dir = save_dir
        self.filename = filename
        self.monitor = monitor
        self.mode = mode

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

    def save(self, trainer: Trainer, train_state: TrainState) -> None:
        """Alias of utils.serialization.save_checkpoint."""
        ckpt_path = Path(self.save_dir, self.filename)
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        save_checkpoint(ckpt_path, trainer, train_state)

    def load(
        self, trainer: Trainer, train_state: TrainState, only_train_state: bool = False, strict: bool = False
    ) -> tuple[Trainer, TrainState]:
        ckpt_path = Path(self.save_dir, self.filename)
        if ckpt_path.exists():
            return load_checkpoint(ckpt_path, trainer, train_state, only_train_state)
        elif not strict:
            return trainer, train_state
        else:
            raise FileNotFoundError(f"{ckpt_path} is not found.")

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
    def is_latest_checkpoint(self) -> bool:
        return self.monitor is None

    @property
    def checkpoint_exists(self) -> bool:
        return Path(self.save_dir, self.filename).exists()

    @property
    def ckpt_path(self):
        return Path(self.save_dir, self.filename)
