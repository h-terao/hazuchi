# Loss
from .absolute_error import absolute_error
from .divergence import kl_div, js_div, cross_entropy
from .squared_error import squared_error
from .triplet_loss import triplet_loss
from .huber_loss import huber_loss
from .charbonnier_penalty import charbonnier_penalty
from .weight_decay_loss import weight_decay_loss

# Evaluation
from .accuracy import accuracy

# Noise
from .vat_noise import vat_noise

# Array
from .permute import permutate
