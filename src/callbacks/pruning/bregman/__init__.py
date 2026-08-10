"""Bregman learning framework for sparse neural network training.

This module implements the Bregman learning approach for neural network pruning
as described in "A Bregman Learning Framework for Sparse Neural Networks".
"""

from .bregman_optimizers import (
    AdaBreg,
    LinBreg,
    ProxSGD,
    get_bregman_optimizer,
)
from .bregman_pruner import BregmanPruner
from .bregman_regularizers import RegL1, RegL1L2, RegNone, get_regularizer

__all__ = [
    "BregmanPruner",
    "get_bregman_optimizer",
    "LinBreg",
    "AdaBreg",
    "ProxSGD",
    "get_regularizer",
    "RegL1",
    "RegL1L2",
    "RegNone",
]
