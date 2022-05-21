from __future__ import annotations
import time
from . import callback


class Timer(callback.Callback):
    """Log elapsed times of training and validation."""

    priority: int = callback.PRIORITY_WRITER

    def __init__(self):
        self.start = None

    def on_fit_epoch_start(self, trainer, train_state):
        self.start = time.time()
        return train_state

    def on_fit_epoch_end(self, trainer, train_state, summary):
        elapsed_time = time.time() - self.start
        self.start = None
        summary["elapsed_time"] = elapsed_time
        return train_state, summary
