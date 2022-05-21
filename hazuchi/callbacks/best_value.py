from __future__ import annotations
import math
import operator

from . import callback


class BestValue(callback.Callback):
    """Log the best value.

    Args:
        name (str): Name of the best metrics.
        monitor (str): Metric name to monitor.
        mode (str): min or max.
    """

    priority: int = callback.PRIORITY_WRITER

    def __init__(self, name: str, monitor: str, mode: str = "min"):
        assert mode in ["min", "max"]
        self.name = name
        self.monitor = monitor
        self.compare = min if mode == "min" else max
        self.best_score = math.inf if mode == "min" else -math.inf

    def on_fit_epoch_end(self, trainer, train_state, summary):
        self.best_score = self.compare(summary.get(self.monitor, self.best_score), self.best_score)
        summary[self.name] = self.best_score
        return train_state, summary
