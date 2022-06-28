from __future__ import annotations
from . import callback

COMET_AVAILABLE = True
try:
    import comet_ml
except ImportError:
    COMET_AVAILABLE = False


class CometLogger(callback.Callback):
    """Log metrics comet.ml"""

    def __init__(self, project: str, name: str, experiment_id: str | None = None, **kwargs) -> None:
        if not COMET_AVAILABLE:
            raise ImportError(
                "Fail to import comet_ml. To use CometLogger, install comet_ml before running your script."  # noqa
            )
        self.project = project
        self.name = name
        self.experiment_id = experiment_id
        self._kwargs = kwargs

        self._params = dict()
        self._experiment = None

    def on_fit_epoch_end(self, trainer, train_state, summary):
        self.experiment.log_metrics(summary, step=trainer.global_step, epoch=trainer.current_epoch)
        return train_state, summary

    def on_test_epoch_end(self, trainer, train_state, summary):
        self.experiment.log_metrics(summary, step=trainer.global_step, epoch=trainer.current_epoch)
        return train_state, summary

    def log_hyperparams(self, params):
        if self._experiment is None:
            self._params = dict(self._params, **params)
        else:
            self._experiment.log_parameters(params)

    @property
    def experiment(self) -> comet_ml.Experiment | comet_ml.ExistingExperiment:
        if self._experiment is None:
            if self.experiment_id is None:
                self._experiment = comet_ml.Experiment(project_name=self.project, **self._kwargs)
                if self.name is not None:
                    self._experiment.set_name(self.name)
                self._experiment.log_parameters(self._params)
                self.experiment_id = self._experiment.get_key()
            else:
                self._experiment = comet_ml.ExistingExperiment(
                    **self._kwargs, previous_experiment=self.experiment_id
                )
                self._experiment.log_parameters(self._params)
        return self._experiment

    def to_state_dict(self):
        return {"id": self.experiment_id}

    def from_state_dict(self, state):
        self.experiment_id = state["id"]
        return self

    def finalize(self):
        if self._experiment is not None:
            self._experiment.end()
            self._experiment = None
