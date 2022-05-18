from __future__ import annotations
from flax.training.train_state import TrainState
from flax.serialization import to_state_dict, from_state_dict
import joblib
from ..trainer import Trainer


__all__ = ["save_checkpoint", "load_checkpoint"]


def save_checkpoint(file, trainer: Trainer, train_state: TrainState):
    checkpoint = {
        "trainer": trainer.to_state_dict(),
        "train_state": to_state_dict(train_state),
    }
    joblib.dump(checkpoint, file, compress=3)


def load_checkpoint(
    file, trainer: Trainer, train_state: TrainState, only_train_state: bool = False
) -> tuple[Trainer, TrainState]:
    checkpoint = joblib.load(file)
    if not only_train_state:
        trainer.from_state_dict(checkpoint["trainer"])
    train_state = from_state_dict(trainer, checkpoint["train_state"])
    return trainer, train_state
