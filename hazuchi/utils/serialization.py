from __future__ import annotations
import pickle

from flax.training.train_state import TrainState
from flax.serialization import to_state_dict, from_state_dict
from ..trainer import Trainer


__all__ = ["save_checkpoint", "load_checkpoint"]


def save_checkpoint(file, trainer: Trainer, train_state: TrainState):
    checkpoint = {
        "trainer": trainer.to_state_dict(),
        "train_state": to_state_dict(train_state),
    }
    with open(file, "wb") as fp:
        pickle.dump(checkpoint, fp)


def load_checkpoint(
    file, trainer: Trainer, train_state: TrainState, only_train_state: bool = False
) -> tuple[Trainer, TrainState]:
    with open(file, "rb") as fp:
        checkpoint = pickle.load(fp)
    if not only_train_state:
        trainer.from_state_dict(checkpoint["trainer"])
    train_state = from_state_dict(train_state, checkpoint["train_state"])
    return trainer, train_state
