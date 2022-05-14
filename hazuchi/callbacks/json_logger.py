from __future__ import annotations
from pathlib import Path
import json
import warnings

from . import callback


class JsonLogger(callback.Callback):
    def __init__(
        self,
        out_dir: str | Path,
        filename: str = "log",
        log_test_summary: bool = False,
    ):
        self.out_dir = Path(out_dir)
        self.filename = filename
        self.log_test_summary = log_test_summary

        self._log = []

    def on_fit_epoch_end(self, trainer, train_state, summary):
        self._log.append(summary)

        tmp_log_path = self.out_dir / f"tmp_{self.filename}"
        log_path = self.out_dir / self.filename

        self.out_dir.mkdir(parents=True, exist_ok=True)
        tmp_log_path.write_text(json.dumps(self._log, indent=2))
        tmp_log_path.rename(log_path)

        return train_state, summary

    def on_test_epoch_end(self, trainer, train_state, summary):

        if self.log_test_summary:
            tmp_log_path = self.out_dir / "tmp_test_summary"
            log_path = self.out_dir / "test_summary"

            self.out_dir.mkdir(parents=True, exist_ok=True)
            tmp_log_path.write_text(json.dumps(summary, indent=2))
            tmp_log_path.rename(log_path)

        return train_state, summary

    def log_hyperparams(self, params):
        warnings.warn("JsonLogger does not support log_hyperparams.")

    def to_state_dict(self):
        return {"_log": json.dumps(self._log)}

    def from_state_dict(self, state) -> None:
        self._log = json.loads(state["_log"])
