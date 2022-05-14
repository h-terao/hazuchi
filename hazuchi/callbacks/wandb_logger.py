from __future__ import annotations
import atexit


from . import callback

WANDB_AVAILABLE = True
try:
    import wandb
except ImportError:
    WANDB_AVAILABLE = False


class WandbLogger(callback.Callback):
    def __init__(self, **kwargs):
        if not WANDB_AVAILABLE:
            raise ImportError(
                "Fail to import wandb. To use wandb_logger, install wandb before running your script."  # noqa
            )

        self._kwargs = kwargs
        self._id = None

    def init_wandb(self):
        if wandb.run is None:
            kwargs = self._kwargs.copy()
            id = kwargs.pop("id", self._id)  # kwargs > previous > random
            if id is None:
                id = wandb.util.generate_id()
            wandb.init(id=id, resume="allow", **kwargs)

            atexit.register(wandb.finish)
        else:
            id = wandb.run.id
        self._id = id

    def on_fit_start(self, trainer, train_state):
        self.init_wandb()
        return train_state

    def on_test_start(self, trainer, train_state):
        return self.on_fit_start(trainer, train_state)

    def on_fit_epoch_end(self, trainer, train_state, summary):
        wandb.log(summary, summary["step"], summary["epoch"])
        return train_state, summary

    def on_test_epoch_end(self, trainer, train_state, summary):
        return self.on_fit_epoch_end(trainer, train_state, summary)

    def log_hyperparams(self, params):
        if wandb.run is None:
            config = self._kwargs.pop("config", None)
            if config is None:
                new_config = params
            else:
                new_config = dict(config, **params)
            self._kwargs["config"] = new_config
        else:
            wandb.config.update(params, allow_val_change=True)

    def to_state_dict(self):
        return {"_id": self._id}

    def from_state_dict(self, state) -> None:
        self._id = state["_id"]
