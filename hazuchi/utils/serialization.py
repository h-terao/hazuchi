from __future__ import annotations
from pathlib import Path
import pickle
import gzip
import uuid

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


def load_state(file) -> dict:
    """Load the snapshot and returns the states."""
    with gzip.open(file, "rb") as f:
        content = f.read()
    return pickle.loads(content)


def dump_state(file, state, *, compresslevel: int = 9):
    """Dump the snapshot."""
    file_path = Path(file)
    tmp_path = file_path.parent / str(uuid.uuid4())[:8]
    content = pickle.dumps(state)
    with gzip.open(tmp_path, "wb", compresslevel=compresslevel) as f:
        f.write(content)
    tmp_path.rename(file_path)
