# flake8: noqa

# Loss
from .absolute_error import *
from .divergence import *
from .cross_entropy import *
from .squared_error import *
from .triplet_loss import *
from .huber_loss import *
from .charbonnier_penalty import *
from .weight_decay_loss import *

# Evaluation
from .accuracy import *

# Noise
from .vat_noise import *

# Array
from .permutate import *
from .one_hot import *

__all__ = [
    "absolute_error",
    "squared_error",
    "kl_div",
    "js_div",
    "cross_entropy",
    "triplet_loss",
    "huber_loss",
    "charbonnier_penalty",
    "weight_decay_loss",
    "accuracy",
    "vat_noise",
    "permutate",
    "one_hot",
]
