import torch
import torch.nn as nn
from typing import List, Tuple, Any, Optional, Dict
from collections import defaultdict
from src.utils import get_pylogger

logger = get_pylogger(__name__)


class ParameterManager:
    """
    Utilities for identifying, validating, and logging parameters for pruning.
    Uses a hybrid strategy: Explicit Allowlist + Structure Detection - Explicit Blocklist.
    """
    # 1. Allowlist: Standard layers we definitely want to prune
    PRUNABLE_LAYERS = (
        nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d,
        nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d,
        nn.LSTM, nn.GRU, nn.Embedding
    )

    # 2. Blocklist: Layers we definitely DO NOT want to prune (Normalizations)
    NON_PRUNABLE_LAYERS = (
        nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
        nn.SyncBatchNorm, nn.LayerNorm, nn.GroupNorm,
        nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d,
        nn.LocalResponseNorm
    )

    def __init__(self, config: Any):
        self.config = config
        self.prunable_params: List[Tuple[nn.Module, str]] = []
        self.skipped_params: List[Dict[str, Any]] = []

    def collect_parameters(
        self,
        model: nn.Module,
        parameters_to_prune: Optional[List[Tuple[nn.Module, str]]] = None
    ) -> List[Tuple[nn.Module, str]]:
        self.prunable_params = []
        self.skipped_params = []
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
            return self.prunable_params

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
            is_custom_valid = hasattr(module, 'weight') and isinstance(module.weight, nn.Parameter)

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

        return self.prunable_params

    def _check_validity(
        self,
        module: nn.Module,
        name: str,
        param: torch.Tensor
    ) -> Optional[str]:
        if not param.requires_grad:
            return "No Gradient"
        
        # Skip biases if configured
        if name == 'bias' and not self.config.prune_bias:
            return "Bias Pruning Disabled"
        
        # Skip small parameters
        if param.numel() < self.config.min_param_elements:
            return f"Too Small (<{self.config.min_param_elements})"
        
        # Structured Pruning Validations
        if self.config.pruning_dim is not None:
            if param.dim() <= self.config.pruning_dim: 
                return f"Dim Mismatch (dim={param.dim()} <= {self.config.pruning_dim})"
        
        return None

    def _record_skip(self, module: nn.Module, param_name: str, reason: str):
        p = getattr(module, param_name, None)
        shape = tuple(p.shape) if p is not None else "?"
        self.skipped_params.append({
            "type": module.__class__.__name__,
            "param": param_name,
            "shape": shape,
            "reason": reason
        })

    def log_overview(self):
        if not self.prunable_params and not self.skipped_params:
            return
        
        if self.prunable_params:
            rows = []
            for mod, name in self.prunable_params:
                p = getattr(mod, name)
                rows.append((
                    mod.__class__.__name__,
                    name,
                    str(tuple(p.shape)),
                    f"{p.numel():,}"
                ))
            self._print_table(
                f"PRUNABLE PARAMETERS ({len(rows)} tensors)", 
                ["Layer Type", "Param", "Shape", "Elements"], 
                rows
            )
        
        if self.skipped_params:
            grouped = defaultdict(int)
            for entry in self.skipped_params:
                grouped[(entry["type"], entry["param"], entry["reason"])] += 1
            rows = []
            for (l_type, p_name, reason), count in sorted(grouped.items(), key=lambda x: -x[1]):
                rows.append((l_type, p_name, reason, str(count)))
            self._print_table(
                f"SKIPPED PARAMETERS ({len(self.skipped_params)} tensors)", 
                ["Layer Type", "Param", "Reason", "Count"], 
                rows
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

    def compute_sparsity(self, parameters: List[Tuple[nn.Module, str]]) -> float:
        """Computes Effective Sparsity (fraction of actual zeros)."""
        total = 0
        zeros = 0
        for module, name in parameters:
            param = getattr(module, name, None)
            if param is not None:
                total += param.numel()
                zeros += (param == 0).sum().item()
        return zeros / total if total > 0 else 0.0