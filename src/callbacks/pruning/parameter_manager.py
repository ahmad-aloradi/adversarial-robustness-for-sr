from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from src.callbacks.pruning.shared_prune_utils import masked_name
from src.utils import get_pylogger

logger = get_pylogger(__name__)


def stem_weight(model: nn.Module) -> Tuple[nn.Module, str]:
    """The model's first weight tensor, in module-definition order.

    The single answer to "which layer is the stem", shared by ParameterManager
    (DST, magnitude, STR) and PruningManager (Bregman), so the four methods
    hold the same tensor dense at ``prune_first_layer: false``.

    Why it reads the module walk rather than a caller's target list: the size
    and dim filters run after this, so a stem below ``min_param_elements``
    would otherwise promote the second layer and hold two of them dense; and a
    group config that routes the stem to its fallback would otherwise hide it.

    >>> net = nn.Sequential(nn.BatchNorm2d(3), nn.Conv2d(3, 8, 3), nn.Linear(8, 2))
    >>> stem_weight(net)[0] is net[1]
    True
    """
    for module in model.modules():
        if isinstance(module, torch.jit.ScriptModule):
            continue
        if isinstance(module, ParameterManager.NON_PRUNABLE_LAYERS):
            continue
        for name, param in module.named_parameters(recurse=False):
            if name == "weight" and param.requires_grad:
                return module, name
    raise ValueError(
        f"{type(model).__name__} exposes no trainable weight tensor, so there "
        f"is no stem to hold dense; set prune_first_layer=true"
    )


def regularizable_params(
    model: nn.Module, min_param_elements: int = 100
) -> List[Tuple[nn.Module, str]]:
    """Every weight tensor a sparsifier could touch: all but norms and biases.

    The denominator every method reports its sparsity against. Layers a
    method holds dense still count here, at full size with no zeros.

    >>> net = nn.Sequential(nn.Conv2d(3, 8, 3), nn.BatchNorm2d(8), nn.Linear(8, 20))
    >>> [type(m).__name__ for m, _ in regularizable_params(net)]
    ['Conv2d', 'Linear']
    """
    manager = ParameterManager(
        prune_bias=False,
        prune_first_layer=True,
        min_param_elements=min_param_elements,
    )
    return manager.collect_parameters(model)


def dense_held_numel(
    model: nn.Module,
    targets: List[Tuple[nn.Module, str]],
    min_param_elements: int = 100,
) -> int:
    """Size of the weights inside the reported denominator that ``targets``
    never touches — the stem, when it is held dense.

    Keyed by (module, attribute) rather than tensor id: a pruned module
    recomputes ``weight`` every forward, so its id is not stable.

    >>> net = nn.Sequential(nn.Conv2d(3, 8, 3), nn.Conv2d(8, 8, 3))
    >>> dense_held_numel(net, [(net[1], "weight")]) == net[0].weight.numel()
    True
    >>> import torch.nn.utils.prune as prune
    >>> _ = prune.l1_unstructured(net[1], "weight", amount=0.5)
    >>> dense_held_numel(net, [(net[1], "weight")]) == net[0].weight.numel()
    True
    """
    held = {(id(module), masked_name(name)) for module, name in targets}
    return sum(
        getattr(module, masked_name(name)).numel()
        for module, name in regularizable_params(model, min_param_elements)
        if (id(module), masked_name(name)) not in held
    )


class ParameterManager:
    """Utilities for identifying, validating, and logging parameters for
    pruning.

    Uses a hybrid strategy: Explicit Allowlist + Structure Detection - Explicit Blocklist.
    """

    # 1. Allowlist: Standard layers we definitely want to prune
    PRUNABLE_LAYERS = (
        nn.Linear,
        nn.Conv1d,
        nn.Conv2d,
        nn.Conv3d,
        nn.ConvTranspose1d,
        nn.ConvTranspose2d,
        nn.ConvTranspose3d,
        nn.LSTM,
        nn.GRU,
        nn.Embedding,
    )

    # 2. Blocklist: Layers we definitely DO NOT want to prune (Normalizations)
    NON_PRUNABLE_LAYERS = (
        nn.BatchNorm1d,
        nn.BatchNorm2d,
        nn.BatchNorm3d,
        nn.SyncBatchNorm,
        nn.LayerNorm,
        nn.GroupNorm,
        nn.InstanceNorm1d,
        nn.InstanceNorm2d,
        nn.InstanceNorm3d,
        nn.LocalResponseNorm,
    )

    def __init__(
        self,
        prune_bias: bool = False,
        prune_first_layer: bool = False,
        min_param_elements: int = 100,
        pruning_dim: Optional[int] = None,
    ):
        self.prune_bias = prune_bias
        self.prune_first_layer = prune_first_layer
        self.min_param_elements = min_param_elements
        self.pruning_dim = pruning_dim
        self.prunable_params: List[Tuple[nn.Module, str]] = []
        self.skipped_params: List[Dict[str, Any]] = []

    def collect_parameters(
        self,
        model: nn.Module,
        parameters_to_prune: Optional[List[Tuple[nn.Module, str]]] = None,
    ) -> List[Tuple[nn.Module, str]]:
        self.prunable_params = []
        self.skipped_params = []
        self._model = model
        seen_params = set()

        # Strategy A: Manual List
        if parameters_to_prune:
            for module, name in parameters_to_prune:
                param = getattr(module, name, None)
                if param is None:
                    continue
                if id(param) in seen_params:
                    continue

                reason = self._check_validity(module, name, param)
                if reason is None:
                    self.prunable_params.append((module, name))
                    seen_params.add(id(param))
                else:
                    self._record_skip(module, name, f"Manual: {reason}")
            return self._keep_first_dense()

        # Strategy B: Automatic Discovery (Hybrid)
        for module in model.modules():
            if isinstance(module, torch.jit.ScriptModule):
                continue

            # 1. Safety Check: Is it a non-prunable Layer?
            if isinstance(module, self.NON_PRUNABLE_LAYERS):
                for name, _ in module.named_parameters(recurse=False):
                    self._record_skip(module, name, "Non-Prunable Layer Type")
                continue

            # 2. Support Check: Is it Standard OR Custom-but-Valid?
            is_standard = isinstance(module, self.PRUNABLE_LAYERS)

            # "Duck Typing" fallback: If it has a weight param, treat it as a custom Linear/Conv
            is_custom_valid = hasattr(module, "weight") and isinstance(
                module.weight, nn.Parameter
            )

            if not (is_standard or is_custom_valid):
                for name, _ in module.named_parameters(recurse=False):
                    self._record_skip(module, name, "Unsupported Layer Type")
                continue

            # 3. Parameter Validation
            for name, param in module.named_parameters(recurse=False):
                if id(param) in seen_params:
                    continue
                seen_params.add(id(param))

                reason = self._check_validity(module, name, param)
                if reason is None:
                    self.prunable_params.append((module, name))
                else:
                    self._record_skip(module, name, reason)

        return self._keep_first_dense()

    def _keep_first_dense(self) -> List[Tuple[nn.Module, str]]:
        """Drop the stem weight from the target set when asked."""
        if self.prune_first_layer:
            return self.prunable_params

        stem = stem_weight(self._model)
        kept = []
        for module, name in self.prunable_params:
            if (module, name) == stem:
                self._record_skip(module, name, "First Layer Kept Dense")
            else:
                kept.append((module, name))
        self.prunable_params = kept
        return self.prunable_params

    def _check_validity(
        self, module: nn.Module, name: str, param: torch.Tensor
    ) -> Optional[str]:
        if not param.requires_grad:
            return "No Gradient"

        # Skip biases if configured
        if name == "bias" and not self.prune_bias:
            return "Bias Pruning Disabled"

        # Skip small parameters
        if param.numel() < self.min_param_elements:
            return f"Too Small (<{self.min_param_elements})"

        # Structured Pruning Validations
        if self.pruning_dim is not None:
            if param.dim() <= self.pruning_dim:
                return (
                    f"Dim Mismatch (dim={param.dim()} <= {self.pruning_dim})"
                )

        return None

    def _record_skip(self, module: nn.Module, param_name: str, reason: str):
        p = getattr(module, param_name, None)
        shape = tuple(p.shape) if p is not None else "?"
        self.skipped_params.append(
            {
                "type": module.__class__.__name__,
                "param": param_name,
                "shape": shape,
                "reason": reason,
            }
        )

    def log_overview(self):
        if not self.prunable_params and not self.skipped_params:
            return

        if self.prunable_params:
            rows = []
            for mod, name in self.prunable_params:
                p = getattr(mod, name)
                rows.append(
                    (
                        mod.__class__.__name__,
                        name,
                        str(tuple(p.shape)),
                        f"{p.numel():,}",
                    )
                )
            self._print_table(
                f"PRUNABLE PARAMETERS ({len(rows)} tensors)",
                ["Layer Type", "Param", "Shape", "Elements"],
                rows,
            )

        if self.skipped_params:
            grouped = defaultdict(int)
            for entry in self.skipped_params:
                grouped[(entry["type"], entry["param"], entry["reason"])] += 1
            rows = []
            for (l_type, p_name, reason), count in sorted(
                grouped.items(), key=lambda x: -x[1]
            ):
                rows.append((l_type, p_name, reason, str(count)))
            self._print_table(
                f"SKIPPED PARAMETERS ({len(self.skipped_params)} tensors)",
                ["Layer Type", "Param", "Reason", "Count"],
                rows,
            )

    def _print_table(self, title: str, headers: List[str], rows: List[Tuple]):
        widths = [len(h) for h in headers]
        for row in rows:
            for i, col in enumerate(row):
                widths[i] = max(widths[i], len(str(col)))
        fmt = "  ".join([f"{{:<{w}}}" for w in widths])
        separator = "  ".join(["-" * w for w in widths])
        lines = [f"\n{title}", separator, fmt.format(*headers), separator]
        for row in rows:
            lines.append(fmt.format(*row))
        lines.append(separator + "\n")
        logger.info("\n".join(lines))
