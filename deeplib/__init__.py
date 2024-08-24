from .tensor import *
from .ops import *
from deeplib import (
    nn as nn,
    optim as optim,
    utils as utils
)

__version__ = "0.0.3"

__all__ = ["tensor", "ops", "nn", "optim", "utils"]
