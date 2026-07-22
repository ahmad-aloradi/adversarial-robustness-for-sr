"""Shape/contract tests for the torchvision ResNet family builder.

Covers every supported `arch`, the small-image stem replacement, and that
the existing type-based pruning-group matching (Conv2d/BatchNorm2d/Linear)
still routes every parameter correctly for the deeper Bottleneck variants —
the same generic mechanism the ResNet-18 Bregman configs already rely on.
"""

import pytest
import pytorch_lightning as pl
import torch
import torch.nn as nn

from src.callbacks.pruning.utils.pruning_manager import PruningManager
from src.modules.models.vision_resnet import _ARCHS, build_resnet

_ARCH_NAMES = sorted(_ARCHS)


@pytest.mark.parametrize("arch", _ARCH_NAMES)
def test_output_shape(arch):
    net = build_resnet(arch, num_classes=10, in_channels=3)
    out = net(torch.randn(2, 3, 32, 32))
    assert tuple(out.shape) == (2, 10)


def _count_norms(net):
    bn = sum(isinstance(m, nn.BatchNorm2d) for m in net.modules())
    ln = sum(isinstance(m, nn.LayerNorm) for m in net.modules())
    return bn, ln


def test_norm_default_is_batchnorm():
    bn, ln = _count_norms(build_resnet("resnet18", num_classes=10))
    assert (bn, ln) == (20, 0)


def test_norm_ln_last_converts_only_the_head_norm():
    net = build_resnet("resnet18", num_classes=10, norm="ln_last")
    assert _count_norms(net) == (19, 1)
    assert isinstance(net.layer4[1].bn2, nn.LayerNorm)


def test_norm_ln_all_converts_every_norm_including_downsample():
    net = build_resnet("resnet18", num_classes=10, norm="ln_all")
    assert _count_norms(net) == (0, 20)
    assert isinstance(net.layer2[0].downsample[1], nn.LayerNorm)


@pytest.mark.parametrize("norm", ["ln_last", "ln_all"])
@pytest.mark.parametrize("size", [32, 64])
def test_norm_forward_shape_is_resolution_agnostic(norm, size):
    net = build_resnet("resnet18", num_classes=10, norm=norm).eval()
    out = net(torch.randn(2, 3, size, size))
    assert tuple(out.shape) == (2, 10)


def test_invalid_norm_raises():
    with pytest.raises(AssertionError, match="norm must be one of"):
        build_resnet("resnet18", num_classes=10, norm="layernorm")


def test_grayscale_in_channels():
    net = build_resnet("resnet18", num_classes=10, in_channels=1)
    out = net(torch.randn(2, 1, 28, 28))
    assert tuple(out.shape) == (2, 10)


def test_small_image_stem_replaced():
    net = build_resnet("resnet50", num_classes=10, in_channels=3)
    assert isinstance(net.maxpool, nn.Identity)
    assert net.conv1.kernel_size == (3, 3)
    assert net.conv1.stride == (1, 1)


@pytest.mark.parametrize("bad_arch", ["resnet20", ""])
def test_unknown_arch_raises(bad_arch):
    with pytest.raises(AssertionError):
        build_resnet(bad_arch, num_classes=10)


def test_invalid_num_classes_raises():
    with pytest.raises(AssertionError):
        build_resnet("resnet18", num_classes=0)


def test_invalid_in_channels_raises():
    with pytest.raises(AssertionError):
        build_resnet("resnet18", num_classes=10, in_channels=2)


@pytest.mark.parametrize("arch", ["resnet50", "wide_resnet50_2"])
def test_pruning_groups_route_bottleneck_downsample(arch):
    """Type-based Bregman groups must still catch Bottleneck's downsample
    Conv2d/BatchNorm2d (named layerX.0.downsample.0/.1, same as BasicBlock)."""

    class _FakeModule(pl.LightningModule):
        def __init__(self):
            super().__init__()
            self.net = build_resnet(arch, num_classes=10, in_channels=3)

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

    name_to_group = {}
    for g in manager.processed_groups:
        for p in g["params"]:
            name_to_group[id(p)] = g["config"]["name"]
    for name, param in model.named_parameters():
        if "downsample.0" in name:
            assert name_to_group[id(param)] == "conv_layers"
        elif "downsample.1" in name:
            assert name_to_group[id(param)] == "norm_params"

    group_names = {g["config"]["name"] for g in manager.processed_groups}
    assert "fallback" not in group_names


if __name__ == "__main__":
    for arch in _ARCH_NAMES:
        net = build_resnet(arch, num_classes=10)
        out = net(torch.randn(1, 3, 32, 32))
        print(f"{arch}: {tuple(out.shape)}")
