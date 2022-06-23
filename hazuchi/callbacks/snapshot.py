from __future__ import annotations
import math
import operator
from pathlib import Path
import pickle

from flax import jax_utils
from flax.serialization import from_state_dict, to_state_dict

from . import callback


class Snapshot(callback.Callback):
    """Save the snapshot of training.

    Args:
        filename (str): Filename of snapshot to dump.
        monitor (str | None): Name of metrics to monitor.
            If None, save the latest checkpoint.
        save_every (int): Interval of save the latest snapshot.
            Only used when monitor is None.
        mode (str): min or max.
        load_before_fitting (bool): Load the checkpoint before fitting if it exists.
        load_before_testing (bool): Load the checkpoint before testing if it exists.
    """

    _priority: int = callback.PRIORITY_SNAPSHOT

    def __init__(
        self,
        filename: str | Path,
        monitor: str | None = None,
        save_every: int = 1,
        mode: str = "min",
        load_before_fitting: bool = False,
        load_before_testing: bool = False,
    ) -> None:
        assert mode in ["min", "max"]

        self.filename = filename
        self.monitor = monitor
        self.save_every = save_every
        self.mode = mode

        self.load_before_fitting = load_before_fitting
        self.load_before_testing = load_before_testing

        self.compare = operator.lt if mode == "min" else operator.gt
        self.best_score = math.inf if mode == "min" else -math.inf

    def on_fit_start(self, trainer, train_state):
        if self.load_before_fitting:
            ckpt_file_path = trainer.out_dir_path / self.filename
            if ckpt_file_path.exists():
                state = pickle.loads(ckpt_file_path.read_bytes())
                trainer = from_state_dict(trainer, state["trainer"])
                train_state = from_state_dict(train_state, state["train_state"])
        return train_state

    def on_fit_epoch_end(self, trainer, train_state, summary):
        if self.monitor is None and trainer.global_step % self.save_every == 0:
            self.save(trainer, train_state)
        elif self.monitor in summary and self.compare(summary[self.monitor], self.best_score):
            self.best_score = summary[self.monitor]
            self.save(trainer, train_state)
        return train_state, summary

    def on_test_start(self, trainer, train_state):
        if self.load_before_testing:
            ckpt_file_path = trainer.out_dir_path / self.filename
            if ckpt_file_path.exists():
                state = pickle.loads(ckpt_file_path.read_bytes())
                train_state = from_state_dict(train_state, state["train_state"])
        return train_state

    def save(self, trainer, train_state):
        ckpt_file_path = trainer.out_dir_path / self.filename
        ckpt_file_path.parent.mkdir(parents=True, exist_ok=True)
        ckpt_file_path.write_bytes(
            pickle.dumps(
                {
                    "trainer": to_state_dict(trainer),
                    "train_state": to_state_dict(jax_utils.unreplicate(train_state)),
                }
            )
        )

    def load(self, filename, train_state):
        state_bytes = Path(filename).read_bytes()
        state_dict = pickle.loads(state_bytes)["train_state"]
        return from_state_dict(train_state, state_dict)

    def to_state_dict(self):
        return {"best": self.best_score}

    def from_state_dict(self, state) -> None:
        self.best_score = state["best"]
        return self

    @property
    def priority(self) -> int:
        if self.monitor is None:
            return self._priority - 100
        else:
            return self._priority

    @property
    def save_last(self) -> bool:
        return self.monitor is None
