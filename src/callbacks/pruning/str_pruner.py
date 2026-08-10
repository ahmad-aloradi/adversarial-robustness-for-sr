""" Implementation of: "Soft Threshold Weight Reparameterization for
Learnable Sparsity" (Kusupati et al., ICML 2020)
https://arxiv.org/pdf/2002.03231.

Idea: Every Conv2d/Linear gets one learnable scalar ``s`` and computes with
``sign(w) * relu(|w| - sigmoid(s))`` instead of ``w``. Weight decay pulls ``s``
up, the task loss pushes it down, and where they balance sets the sparsity.

**The sparsity can't be controlled!** The paper is explicit: "the
overall sparsity budget for STR is not set explicitly. Instead, it is controlled
by the weight-decay parameter, and can be further fine-tuned using s_init". So
there is no target to give it.

Run it with::

    python src/train.py experiment=img/soft_threshold datamodule=datasets/mnist

Inspect the reparameterization alone with::

    python -m src.callbacks.pruning.str_pruner
"""
import json
import os
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import Callback, LightningModule, Trainer

from src.callbacks.pruning.parameter_manager import (
    ParameterManager,
    regularizable_params,
)
from src.utils import get_pylogger

logger = get_pylogger(__name__)

THRESHOLD_ATTR = "sparse_threshold"


# ported from RAIVNLab/STR@cc74d38 utils/conv_type.py:sparseFunction
def soft_threshold(
    weight: torch.Tensor, threshold: torch.Tensor
) -> torch.Tensor:
    """Soft-threshold w at sigmoid(s). Differentiable in both, so the loss
    trains s.

    >>> w = torch.tensor([-1.0, -0.1, 0.0, 0.1, 1.0])
    >>> s = torch.zeros(1)  # sigmoid(0) = 0.5
    >>> soft_threshold(w, s)
    tensor([-0.5000, -0.0000,  0.0000,  0.0000,  0.5000])
    """
    return torch.sign(weight) * torch.relu(
        torch.abs(weight) - torch.sigmoid(threshold)
    )


class STRConv2d(nn.Conv2d):
    """Conv2d whose weight is soft-thresholded by a learnable scalar."""

    def __init__(self, *args, s_init: float = -3200.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.sparse_threshold = nn.Parameter(torch.full((1, 1), s_init))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._conv_forward(
            x, soft_threshold(self.weight, self.sparse_threshold), self.bias
        )


class STRLinear(nn.Linear):
    """Linear whose weight is soft-thresholded by a learnable scalar."""

    def __init__(self, *args, s_init: float = -3200.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.sparse_threshold = nn.Parameter(torch.full((1, 1), s_init))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(
            x, soft_threshold(self.weight, self.sparse_threshold), self.bias
        )


def apply_thresholds(
    state_dict: Dict[str, torch.Tensor]
) -> "OrderedDict[str, torch.Tensor]":
    """Apply threshold to weights and drop s.

    >>> sd = {"c.weight": torch.tensor([[0.1, 1.0]]), "c.sparse_threshold": torch.zeros(1, 1)}
    >>> apply_thresholds(sd)["c.weight"]
    tensor([[0.0000, 0.5000]])
    """
    out = OrderedDict()
    for key, value in state_dict.items():
        if key.endswith(f".{THRESHOLD_ATTR}"):
            continue
        weight_key = f"{key.rsplit('.', 1)[0]}.{THRESHOLD_ATTR}"
        if key.endswith(".weight") and weight_key in state_dict:
            value = soft_threshold(value, state_dict[weight_key])
        out[key] = value
    return out


def apply_thresholds_to_ckpt(ckpt_path: Path) -> bool:
    """Rewrite one Lightning ckpt through ``apply_thresholds``.

    Writes to a temp file and renames, so a kill mid-write cannot destroy the
    only ckpt of a save_top_k=1 run.
    """
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["state_dict"]
    if not any(k.endswith(f".{THRESHOLD_ATTR}") for k in state_dict):
        return False
    checkpoint["state_dict"] = apply_thresholds(state_dict)
    tmp_path = ckpt_path.with_suffix(".ckpt.tmp")
    torch.save(checkpoint, tmp_path)
    os.replace(tmp_path, ckpt_path)
    return True


class STRPruner(Callback):
    """Orchestrates STR pruning.

    Key parameters:
        s_init: starting value of every s. At iteration t it has decayed to
        ``s_t = s_init * exp(-sum(lr) * wd / (1 - momentum))``.
        prune_first_layer: ``False`` leaves the stem conv dense
    """

    def __init__(
        self,
        s_init: float = -3200.0,
        prune_bias: bool = False,
        prune_first_layer: bool = False,
        min_param_elements: int = 100,
        verbose: int = 1,
    ):
        self.s_init = s_init
        self.verbose = verbose
        self.min_param_elements = min_param_elements
        self.manager = ParameterManager(
            prune_bias=prune_bias,
            prune_first_layer=prune_first_layer,
            min_param_elements=min_param_elements,
        )

        # (qualified name, STR layer, the original layer it replaced)
        self._layers: List[Tuple[str, nn.Module, nn.Module]] = []
        self._substituted = False
        self._thresholds_applied = False

    # ------------------------------------------------------------------ #
    #  Lightning hooks                                                     #
    # ------------------------------------------------------------------ #

    def setup(
        self, trainer: Trainer, pl_module: LightningModule, stage: str
    ) -> None:
        # setup re-enters per stage; the thresholds are already in w by then.
        if stage != "fit" or self._substituted or self._thresholds_applied:
            return
        self._substitute(pl_module)
        self._substituted = True
        if self.verbose:
            self.manager.log_overview()
            logger.info(
                f"[STR] reparameterized {len(self._layers)} layers at "
                f"s_init={self.s_init}"
            )

    def on_train_start(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        """Refuse to start a run whose ``on_train_end`` export cannot work.

        Why: the export rewrites every ckpt but ``last.ckpt`` in place, so it
        needs a checkpoint dir, one writer, and no ``last-v*.ckpt`` -- Lightning
        writes that name when resuming into a used dir, and thresholding it
        would leave the run with no resumable checkpoint. Not ``setup``:
        ``ModelCheckpoint`` resolves ``dirpath`` in its own ``setup``, which
        Lightning orders after every other callback's.
        """
        assert trainer.world_size == 1, (
            f"the threshold export rewrites checkpoints in place through one "
            f"temp path, so world_size must be 1, got {trainer.world_size}"
        )
        assert (
            trainer.checkpoint_callback is not None
        ), "STR exports thresholded checkpoints, so checkpointing must be enabled"
        versioned = sorted(
            path.name
            for path in Path(trainer.checkpoint_callback.dirpath).glob(
                "last-v*.ckpt"
            )
        )
        assert not versioned, (
            f"the export skips 'last.ckpt' by name only, so a versioned last "
            f"checkpoint would be thresholded and stop resuming, got {versioned}"
        )

    def on_train_epoch_start(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        trainer.callback_metrics["pruning/sparsity"] = torch.tensor(
            self.sparsities()[1]
        )

    def on_train_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        collapsed = self._collapsed_layers()
        overall, pruned = self.sparsities()
        thresholds = self.thresholds()
        mean_threshold = sum(thresholds.values()) / len(thresholds)
        pl_module.log("sparsity", overall, prog_bar=False, on_epoch=True)
        pl_module.log(
            "pruning/sparsity", pruned, prog_bar=False, on_epoch=True
        )
        pl_module.log(
            "str/threshold", mean_threshold, prog_bar=False, on_epoch=True
        )
        pl_module.log(
            "str/collapsed_layers",
            float(len(collapsed)),
            prog_bar=False,
            on_epoch=True,
        )
        if collapsed:
            logger.warning(
                f"[STR] {len(collapsed)} layer(s) collapsed: "
                + ", ".join(
                    f"{n} (sigmoid(s)={t:.4g})" for n, t in collapsed.items()
                )
            )
        if len(collapsed) == len(self._layers):
            logger.warning(
                f"[STR] all {len(collapsed)} thresholded layers are collapsed; nothing "
                f"left to train. Lower module.optimizer.weight_decay."
            )
        if self.verbose:
            logger.info(
                f"[STR Monitor] Epoch {trainer.current_epoch}: "
                f"STRParams Sparsity={pruned:.2%} | "
                f"Total sparsity={overall:.2%} | "
                f"MeanThreshold={mean_threshold:.2e}"
            )

    def on_train_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        """Applies thresholds to weights in the best ckpts. Drop s and use the
        thresholded weights directly.

        Why: eval.py and the robustness scripts load the original ResNet
        (contains no s params). ``last.ckpt`` keeps its s, so resume still works.
        """
        self._write_thresholds(trainer)
        checkpoint_dir = trainer.checkpoint_callback.dirpath
        rewritten = [
            path
            for path in sorted(Path(checkpoint_dir).glob("*.ckpt"))
            if path.name != "last.ckpt" and apply_thresholds_to_ckpt(path)
        ]
        self._apply_thresholds_to_model(pl_module)
        self._thresholds_applied = True
        if self.verbose:
            logger.info(
                f"[STR] applied thresholds to {len(rewritten)} checkpoints "
                f"and to the live model"
            )

    # ------------------------------------------------------------------ #
    #  Measurement                                                         #
    # ------------------------------------------------------------------ #

    def sparsities(self) -> Tuple[float, float]:
        """(whole-model, weight-tensor) sparsity of the thresholded weights.

        w stays dense during training, so its own zeros read ~0.
        """
        assert self._layers, "no STR layers; setup(stage='fit') has not run"
        effective = {
            id(layer.weight): soft_threshold(
                layer.weight, layer.sparse_threshold
            )
            for _, layer, _ in self._layers
        }
        thresholds = {
            id(layer.sparse_threshold) for _, layer, _ in self._layers
        }

        def count(params) -> Tuple[int, int]:
            zeros = total = 0
            for param in params:
                if id(param) in thresholds:
                    continue  # a knob, not a weight
                tensor = effective.get(id(param), param)
                zeros += int((tensor.abs() <= 1e-12).sum())
                total += tensor.numel()
            return zeros, total

        overall_zeros, overall_total = count(self._model.parameters())
        weight_zeros, weight_total = count(
            getattr(module, name)
            for module, name in regularizable_params(
                self._model, self.min_param_elements
            )
        )
        return overall_zeros / overall_total, weight_zeros / weight_total

    def _collapsed_layers(self) -> Dict[str, float]:
        """Inspect layer collapse.

        Unlike Bregman, layer collapse is permanent: relu zeroes the gradient
        to both w and s once every weight < s.
        """
        collapsed = {}
        for name, layer, _ in self._layers:
            effective = soft_threshold(layer.weight, layer.sparse_threshold)
            if bool((effective == 0).all()):
                collapsed[name] = float(torch.sigmoid(layer.sparse_threshold))
        return collapsed

    def thresholds(self) -> Dict[str, float]:
        """Per-layer pruning threshold, i.e. ``sigmoid(s)``."""
        return {
            name: float(torch.sigmoid(layer.sparse_threshold))
            for name, layer, _ in self._layers
        }

    # ------------------------------------------------------------------ #
    #  Substitution                                                        #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _parent_and_attr(
        root: nn.Module, qualname: str
    ) -> Tuple[nn.Module, str]:
        parent = root
        parts = qualname.split(".")
        for part in parts[:-1]:
            parent = getattr(parent, part)
        return parent, parts[-1]

    def _substitute(self, model: nn.Module) -> None:
        """Replace every prunable Conv2d/Linear with STRConv2d/STRLinear."""
        qualnames = {id(m): name for name, m in model.named_modules()}
        self._model = model
        self._layers = []
        for module, param_name in self.manager.collect_parameters(model):
            name = qualnames[id(module)]
            if param_name != "weight":
                raise ValueError(
                    f"STR thresholds weights only, got {name}.{param_name}; set prune_bias=false"
                )
            str_layer = self._build_str_layer(module)
            parent, attr = self._parent_and_attr(model, name)
            setattr(parent, attr, str_layer)
            self._layers.append((name, str_layer, module))

    def _build_str_layer(self, layer: nn.Module) -> nn.Module:
        """Build the STR layer for one Conv2d/Linear, carrying w and bias
        over."""
        if isinstance(layer, nn.Conv2d):
            str_layer = STRConv2d(
                layer.in_channels,
                layer.out_channels,
                layer.kernel_size,
                stride=layer.stride,
                padding=layer.padding,
                dilation=layer.dilation,
                groups=layer.groups,
                bias=layer.bias is not None,
                padding_mode=layer.padding_mode,
                s_init=self.s_init,
            )
        elif isinstance(layer, nn.Linear):
            str_layer = STRLinear(
                layer.in_features,
                layer.out_features,
                bias=layer.bias is not None,
                s_init=self.s_init,
            )
        else:
            raise TypeError(
                f"STR reparameterizes Conv2d and Linear only, got {type(layer).__name__}"
            )

        str_layer = str_layer.to(layer.weight)
        with torch.no_grad():
            str_layer.weight.copy_(layer.weight)
            if layer.bias is not None:
                str_layer.bias.copy_(layer.bias)
        return str_layer

    def _apply_thresholds_to_model(self, model: nn.Module) -> None:
        """Undo the substitution: each original Conv2d/Linear comes back holding
        sign(w)*relu(|w|-sigmoid(s)).
        """
        for name, str_layer, original in self._layers:
            original = original.to(str_layer.weight)
            with torch.no_grad():
                original.weight.copy_(
                    soft_threshold(
                        str_layer.weight, str_layer.sparse_threshold
                    )
                )
                if str_layer.bias is not None:
                    original.bias.copy_(str_layer.bias)
            parent, attr = self._parent_and_attr(model, name)
            setattr(parent, attr, original)
        self._layers = []

    def _write_thresholds(self, trainer: Trainer) -> None:
        """Save the learnt per-layer thresholds before they are applied to
        w."""
        out_path = Path(trainer.default_root_dir) / "str_thresholds.json"
        out_path.write_text(json.dumps(self.thresholds(), indent=4))
        logger.info(f"Saved STR thresholds to: {out_path}")


if __name__ == "__main__":
    demo = nn.Sequential(nn.Conv2d(3, 16, 3), nn.Flatten(), nn.Linear(64, 10))
    pruner = STRPruner(s_init=-3.0, prune_first_layer=True, verbose=0)
    pruner._substitute(demo)
    print(f"layers: {[name for name, _, _ in pruner._layers]}")
    print(f"thresholds: {pruner.thresholds()}")
    overall, pruned = pruner.sparsities()
    print(f"overall={overall:.4f}  str_layers={pruned:.4f}")
