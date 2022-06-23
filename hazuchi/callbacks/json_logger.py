from __future__ import annotations
import json
import warnings

from . import callback


class JsonLogger(callback.Callback):
    """Logging the summaries in the local.

    Args:
        filename (str): Filename of log.
        log_test_summary (bool): If True, test summary is also logged.
    """

    def __init__(
        self,
        filename: str = "log",
        log_test_summary: bool = False,
    ):
        self.filename = filename
        self.log_test_summary = log_test_summary

        self._log = []

    def on_fit_epoch_end(self, trainer, train_state, summary):
        self._log.append(summary)

        tmp_log_path = trainer.out_dir_path / f"tmp_{self.filename}"
        log_path = trainer.out_dir_path / self.filename

        tmp_log_path.write_text(json.dumps(self._log, indent=2))
        tmp_log_path.rename(log_path)

        return train_state, summary

    def on_test_epoch_end(self, trainer, train_state, summary):
        if self.log_test_summary:
            tmp_log_path = trainer.out_dir_path / "tmp_test_summary"
            log_path = trainer.out_dir_path / "test_summary"

            tmp_log_path.write_text(json.dumps(summary, indent=2))
            tmp_log_path.rename(log_path)
        return train_state, summary

    def log_hyperparams(self, params):
        warnings.warn("JsonLogger does not support log_hyperparams.")

    def to_state_dict(self):
        return {"_log": self._log}

    def from_state_dict(self, state) -> None:
        self._log = state["_log"]
        return self
