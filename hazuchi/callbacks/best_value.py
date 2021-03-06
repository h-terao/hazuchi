from __future__ import annotations
import math

from . import callback


class BestValue(callback.Callback):
    """Log the best value.

    Args:
        monitor (str): Metric name to monitor.
        name (str): Name of the best metrics.
            If None, use "{monitor}_best"
        mode (str): min or max.
    """

    priority: int = callback.PRIORITY_WRITER

    def __init__(self, monitor: str, name: str | None = None, mode: str = "min"):
        assert mode in ["min", "max"]
        self.monitor = monitor
        self.name = name or f"{monitor}_best"
        self.compare = min if mode == "min" else max
        self.best_score = math.inf if mode == "min" else -math.inf

    def on_fit_epoch_end(self, trainer, train_state, summary):
        self.best_score = self.compare(summary.get(self.monitor, self.best_score), self.best_score)
        summary[self.name] = self.best_score
        return train_state, summary

    def to_state_dict(self):
        return {"best": self.best_score}

    def from_state_dict(self, state) -> BestValue:
        self.best_score = state["best"]
        return self
