from __future__ import annotations
from pathlib import Path
import json
import warnings

from . import callback


class JsonLogger(callback.Callback):
    """Logging the summaries in the local.

    Args:
        save_dir (str): Directory to save log.
        filename (str): Filename of log.
        log_test_summary (bool): If True, test summary is also logged.
    """

    def __init__(
        self,
        save_dir: str | Path,
        filename: str = "log",
        log_test_summary: bool = False,
    ):
        self.save_dir = Path(save_dir)
        self.filename = filename
        self.log_test_summary = log_test_summary

        self._log = []

    def on_fit_epoch_end(self, trainer, train_state, summary):
        self._log.append(summary)

        tmp_log_path = self.save_dir / f"tmp_{self.filename}"
        log_path = self.save_dir / self.filename

        self.save_dir.mkdir(parents=True, exist_ok=True)
        tmp_log_path.write_text(json.dumps(self._log, indent=2))
        tmp_log_path.rename(log_path)

        return train_state, summary

    def on_test_epoch_end(self, trainer, train_state, summary):

        if self.log_test_summary:
            tmp_log_path = self.save_dir / "tmp_test_summary"
            log_path = self.save_dir / "test_summary"

            self.save_dir.mkdir(parents=True, exist_ok=True)
            tmp_log_path.write_text(json.dumps(summary, indent=2))
            tmp_log_path.rename(log_path)

        return train_state, summary

    def log_hyperparams(self, params):
        warnings.warn("JsonLogger does not support log_hyperparams.")

    def to_state_dict(self):
        return {"_log": json.dumps(self._log)}

    def from_state_dict(self, state) -> None:
        self._log = json.loads(state["_log"])
