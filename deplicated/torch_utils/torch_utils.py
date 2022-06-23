from __future__ import annotations
from typing import Any, Mapping, Sequence

import numpy
from torch.utils import data

# import torch.multiprocessing as multiprocessing

# # https://github.com/google/jax/issues/3382
# # multiprocessing.set_start_method("spawn", force=True)


class ArrayDataset(data.Dataset):
    """Similar to TensorDataset, but returns numpy array."""

    def __init__(self, *arrays):
        self.arrays = arrays
        self.num_samples = arrays[0].shape[0]
        for array in arrays:
            assert self.num_samples == array.shape[0]

    def __getitem__(self, index):
        return [arr[index] for arr in self.arrays]

    def __len__(self):
        return len(self.num_samples)


def collate_fun(x: Any) -> Any:
    elem = x[0]
    if isinstance(elem, Sequence):
        return [collate_fun(xi) for xi in zip(*x)]
    elif isinstance(elem, Mapping):
        return {key: collate_fun([xi[key] for xi in x]) for key in elem}
    elif isinstance(elem, numpy.ndarray):
        return numpy.stack(x)
    elif numpy.isscalar(elem):
        return numpy.array(x)
    else:
        # does not stack.
        return x
