"""
Bregman learning framework for sparse neural network training.

This module implements the Bregman learning approach for neural network pruning
as described in "A Bregman Learning Framework for Sparse Neural Networks".
"""

from .bregman_pruner import BregmanPruner
from .bregman_optimizers import get_bregman_optimizer, LinBreg, AdaBreg, ProxSGD
from .bregman_regularizers import get_regularizer, RegL1, RegL1L2, RegNone

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
