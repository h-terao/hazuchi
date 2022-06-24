from __future__ import annotations
from tqdm.rich import tqdm
from . import callback


class ProgressBar(callback.Callback):
    """Show progress bar."""

    def __init__(self) -> None:
        self._pbar = None

    def on_fit_start(self, trainer, train_state):
        self._pbar = tqdm(total=trainer.max_epochs, desc="[Training]")
        self._pbar.update(trainer.current_epoch)
        self._pbar.refresh()
        return train_state

    def on_fit_epoch_end(self, trainer, train_state, summary):
        self._pbar.update()
        return train_state, summary

    def on_fit_end(self, trainer, train_state):
        self._pbar.close()
        self._pbar = None
        return train_state
