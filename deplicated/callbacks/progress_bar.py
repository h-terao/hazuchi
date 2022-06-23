from __future__ import annotations
import atexit

from tqdm.rich import tqdm

from . import callback


class ProgressBar(callback.Callback):
    """Show progress bar."""

    def __init__(self):
        self._pbar = None

        # Require to restore cursor on the terminal.
        # https://stackoverflow.com/questions/71143520/python-rich-restore-cursor-default-values-on-exit
        atexit.register(lambda: print("\x1b[?25h"))

    def on_fit_epoch_start(self, trainer, train_state):
        self._pbar = tqdm(
            total=self.estimate_total_steps(trainer),
            leave=False,
            desc=f"[Epoch: {trainer.current_epoch}]",
        )
        return train_state

    def on_train_step_end(self, trainer, train_state, summary):
        self._pbar.update()
        return train_state, summary

    def on_val_step_end(self, trainer, train_state, summary):
        self._pbar.update()
        return train_state, summary

    def on_fit_epoch_end(self, trainer, train_state, summary):
        self._pbar.close()
        self._pbar = None
        return train_state, summary

    def on_test_start(self, trainer, train_state):
        self._pbar = tqdm(
            total=trainer.test_steps_per_epoch,
            leave=False,
            desc="[Testing]",
        )
        return train_state

    def on_test_step_end(self, trainer, train_state, summary):
        self._pbar.update()
        return train_state, summary

    def on_test_end(self, trainer, train_state):
        self._pbar.close()
        self._pbar = None
        return train_state

    def estimate_total_steps(self, trainer):
        if (trainer.current_epoch + 1) % trainer.val_every == 0:
            return trainer.train_steps_per_epoch + trainer.val_steps_per_epoch
        else:
            return trainer.train_steps_per_epoch
