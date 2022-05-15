# flake8: noqa
try:
    import torch
    from .torch_utils import *

    del torch
except ImportError:
    pass
