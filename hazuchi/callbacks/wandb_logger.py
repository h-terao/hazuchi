from __future__ import annotations
import atexit

from . import callback

WANDB_AVAILABLE = True
try:
    import wandb
except ImportError:
    WANDB_AVAILABLE = False


class WandbLogger(callback.Callback):
    """Log metrics to wandb.

    Args:
        project (str): Title of the project.
        name (str, optional): Experiment name.
        code_dir (str, optional): If specified, call wandb.run.log_code(root=code_dir).
    """

    def __init__(
        self, project: str, name: str | None = None, code_dir: str | None = None, **kwargs
    ):
        if not WANDB_AVAILABLE:
            raise ImportError(
                "Fail to import wandb. To use wandb_logger, install wandb before running your script."  # noqa
            )

        self.kwargs = dict(kwargs, project=project, name=name)
        self.code_dir = code_dir
        self.id = None

    def init_wandb(self):
        if wandb.run is None:
            kwargs = self.kwargs.copy()
            id = kwargs.pop("id", self.id)  # kwargs > previous > random
            if id is None:
                id = wandb.util.generate_id()
            wandb.init(id=id, resume="allow", **kwargs)

            if self.code_dir:
                # upload python codes.
                wandb.run.log_code(root=self.code_dir)

            atexit.register(wandb.finish)
        else:
            id = wandb.run.id
        self.id = id

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
            config = self.kwargs.pop("config", None)
            if config is None:
                new_config = params
            else:
                new_config = dict(config, **params)
            self.kwargs["config"] = new_config
        else:
            wandb.config.update(params, allow_val_change=True)

    def to_state_dict(self):
        return {"id": self.id}

    def from_state_dict(self, state):
        self.id = state["id"]
        return self
