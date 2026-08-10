"""Tiny image classifier for fast robustness/eval tests.

Swapped in for the real ``img_model`` backbones with::

    module.model.net._target_=tests.helpers.tiny_net.build_tiny_net
    ~module.model.net.arch
    ~module.model.net.dataset_name

The two deletions are required: this takes only the ``num_classes`` and
``in_channels`` that every ``img_model`` config shares.
"""
import torch.nn as nn


def build_tiny_net(num_classes: int, in_channels: int = 3) -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(in_channels, 8, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(4),
        nn.Flatten(),
        nn.Linear(8 * 16, num_classes),
    )
