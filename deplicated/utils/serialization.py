from __future__ import annotations
from pathlib import Path
import pickle
import gzip
import uuid


__all__ = ["load_state", "save_state"]


def load_state(file) -> dict:
    """Load the snapshot and returns the states."""
    with gzip.open(file, "rb") as f:
        content = f.read()
    return pickle.loads(content)


def save_state(file, state, *, compresslevel: int = 9):
    """Dump the snapshot."""
    file_path = Path(file)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    tmp_path = file_path.parent / str(uuid.uuid4())[:8]
    content = pickle.dumps(state)
    with gzip.open(tmp_path, "wb", compresslevel=compresslevel) as f:
        f.write(content)
    tmp_path.rename(file_path)
