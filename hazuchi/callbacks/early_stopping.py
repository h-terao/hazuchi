from __future__ import annotations
import math
import operator

from . import callback


class EarlyStopping(callback.Callback):
    """Early stopping."""

    priority = callback.PRIORITY_EDITOR

    def __init__(self, monitor: str, patience: int = 10, mode: str = "min"):
        assert mode in ["min", "max"]
        self.monitor = monitor
        self.patience = patience
        self.compare = operator.lt if mode == "min" else operator.gt

        self.best_score = math.inf if mode == "min" else -math.inf
        self.count = 0

    def on_fit_epoch_end(self, trainer, train_state, summary):
        if self.monitor in summary:
            if self.compare(summary[self.monitor], self.best_score):
                self.best_score = summary[self.monitor]
                self.count = 0
            else:
                self.count += 1
                if self.count > self.patience:
                    trainer._fitted = True
        return train_state, summary

    def to_state_dict(self):
        return {"best": self.best_score, "count": self.count}

    def from_state_dict(self, state):
        self.best_score = state["best"]
        self.count = state["count"]
