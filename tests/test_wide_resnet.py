"""Shape/contract tests for the Wide-ResNet (WRN) CIFAR builder.

Covers the standard WRN-28-10 configuration, the CIFAR-only 32x32 input
assert (pytorchcv's CIFARWRN pools with a fixed 8x8 kernel), and that the
same type-based pruning-group matching used for the ResNet family also
routes WRN's Conv2d/BatchNorm2d/Linear parameters correctly.
"""

import pytest
import pytorch_lightning as pl
import torch

from src.callbacks.pruning.utils.pruning_manager import PruningManager
from src.modules.models.wide_resnet import build_wide_resnet


def test_output_shape_cifar10():
    net = build_wide_resnet(num_classes=10, in_channels=3)
    out = net(torch.randn(2, 3, 32, 32))
    assert tuple(out.shape) == (2, 10)


def test_output_shape_cifar100():
    net = build_wide_resnet(num_classes=100, in_channels=3)
    out = net(torch.randn(2, 3, 32, 32))
    assert tuple(out.shape) == (2, 100)


def test_grayscale_in_channels():
    net = build_wide_resnet(num_classes=10, in_channels=1)
    out = net(torch.randn(2, 1, 32, 32))
    assert tuple(out.shape) == (2, 10)


@pytest.mark.parametrize("hw", [(28, 28), (64, 64)])
def test_wrong_resolution_raises(hw):
    net = build_wide_resnet(num_classes=10, in_channels=3)
    with pytest.raises(AssertionError, match="expects 32x32"):
        net(torch.randn(2, 3, *hw))


def test_parameter_count_matches_published_wrn28_10():
    net = build_wide_resnet(depth=28, widen_factor=10, num_classes=10)
    n_params = sum(p.numel() for p in net.parameters())
    # Widely cited WRN-28-10 CIFAR-10 parameter count is ~36.5M.
    assert 36_000_000 < n_params < 37_000_000


def test_invalid_depth_raises():
    with pytest.raises(AssertionError):
        build_wide_resnet(depth=27, widen_factor=10, num_classes=10)


def test_invalid_num_classes_raises():
    with pytest.raises(AssertionError):
        build_wide_resnet(num_classes=0)


def test_invalid_in_channels_raises():
    with pytest.raises(AssertionError):
        build_wide_resnet(num_classes=10, in_channels=2)


def test_pruning_groups_route_all_parameters():
    class _FakeModule(pl.LightningModule):
        def __init__(self):
            super().__init__()
            self.net = build_wide_resnet(num_classes=10, in_channels=3)

    model = _FakeModule()
    groups = [
        {
            "name": "conv_layers",
            "layer_types": ["torch.nn.Conv2d"],
            "param_names": ["weight"],
            "optimizer_settings": {"lambda_scale": 1.0},
        },
        {
            "name": "linear_layers",
            "layer_types": ["torch.nn.Linear"],
            "param_names": ["weight"],
            "optimizer_settings": {"lambda_scale": 1.0},
        },
        {
            "name": "norm_params",
            "layer_types": ["torch.nn.BatchNorm2d"],
            "optimizer_settings": {"lambda_scale": 0.0},
        },
        {
            "name": "bias_params",
            "param_names": ["bias"],
            "optimizer_settings": {"lambda_scale": 0.0},
        },
        {
            "name": "fallback",
            "is_fallback": True,
            "module_name_patterns": [".*"],
            "param_names": ["weight"],
            "optimizer_settings": {"lambda_scale": 1.0},
        },
    ]
    manager = PruningManager(pl_module=model, group_configs=groups)
    manager.get_optimizer_param_groups()

    total = sum(len(g["params"]) for g in manager.processed_groups)
    trainable = sum(1 for p in model.parameters() if p.requires_grad)
    assert total == trainable

    group_names = {g["config"]["name"] for g in manager.processed_groups}
    assert "fallback" not in group_names


if __name__ == "__main__":
    net = build_wide_resnet(num_classes=10, in_channels=3)
    out = net(torch.randn(1, 3, 32, 32))
    print(f"WRN-28-10: {tuple(out.shape)}")
