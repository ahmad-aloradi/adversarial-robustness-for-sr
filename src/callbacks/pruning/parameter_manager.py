import functools
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.utils.prune as pytorch_prune

from src.utils import get_pylogger

logger = get_pylogger(__name__)


class ParameterManager:
    """
    Manages parameter collection, validation, and logging for pruning.
    """

    # Constants for prunable layer types
    DEFAULT_PRUNABLE_LAYER_TYPES = (
        nn.Linear, nn.Embedding, 
        nn.Conv1d, nn.Conv2d, nn.Conv3d,
        nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d,
        nn.LSTM, nn.GRU
    )
    
    # Constants for non-prunable layer types (e.g., normalization layers)
    DEFAULT_NON_PRUNABLE_LAYER_TYPES = (
        nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
        nn.SyncBatchNorm,
        nn.LayerNorm,
        nn.GroupNorm,
        nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d,
        nn.LocalResponseNorm,
        nn.modules.batchnorm._BatchNorm,
        nn.modules.instancenorm._InstanceNorm,
    )
    
    WEIGHT_PARAM_NAME = 'weight'
    BIAS_PARAM_NAME = 'bias'

    def __init__(self, config: Any):
        self.config = config
        self.prunable_parameters: List[Tuple[nn.Module, str]] = []
        self._non_prunable_default_parameters: List[Dict[str, str]] = []
        self._validated_params_cache: Optional[List[Tuple[nn.Module, str]]] = None
        self._pruning_fn_name: Optional[str] = None
        self._is_structured_pruning: Optional[bool] = None
        self._parameter_summary_logged: bool = False

    def invalidate_cache(self) -> None:
        """Invalidate the parameter cache."""
        self._validated_params_cache = None

    def _get_pruning_fn_name(self) -> str:
        """Get cached pruning function name."""
        if self._pruning_fn_name is None:
            pruning_fn = self.config.pruning_fn
            
            # Handle partials created by ModelPruning
            if isinstance(pruning_fn, functools.partial):
                pruning_fn = pruning_fn.func
                
            self._pruning_fn_name = (
                pruning_fn if isinstance(pruning_fn, str) 
                else getattr(pruning_fn, "__name__", "unknown_callable")
            )
        return self._pruning_fn_name

    def _requires_structured_pruning(self) -> bool:
        """Cached check for structured pruning requirement."""
        if self._is_structured_pruning is None:
            self._is_structured_pruning = "ln_structured" in self._get_pruning_fn_name()
        return self._is_structured_pruning

    def _get_parameter_tensor(self, module: nn.Module, param_name: str) -> Optional[torch.Tensor]:
        """Get parameter tensor safely without error masking."""
        # Try _orig parameter first (for pruned parameters)
        try:
            return getattr(module, f"{param_name}_orig")
        except AttributeError:
            # Fall back to regular parameter
            try:
                return getattr(module, param_name)
            except AttributeError:
                return None

    def _is_valid_tensor(self, tensor: torch.Tensor) -> bool:
        """Check if tensor meets basic validity requirements."""
        if not isinstance(tensor, torch.Tensor):
            return False
        if tensor.dim() == 0 or tensor.numel() < 2:
            return False
        return True

    def _is_valid_for_structured_pruning(self, tensor: torch.Tensor) -> bool:
        """Check if tensor is valid for structured pruning."""
        if tensor.dim() < 2:
            return False
        
        pruning_dim = self.config.pruning_dim
            
        if pruning_dim is None:
            return False
            
        return tensor.dim() > pruning_dim

    def _is_param_valid(self, module: nn.Module, param_name: str) -> bool:
        """Check if parameter is valid for pruning with reduced complexity."""
        tensor = self._get_parameter_tensor(module, param_name)
        if tensor is None:
            return False
        
        if not self._is_valid_tensor(tensor):
            return False
        
        if self._requires_structured_pruning():
            return self._is_valid_for_structured_pruning(tensor)
        
        return True

    def _log_validation_failures(self, params: List[Tuple[nn.Module, str]]) -> None:
        """Log detailed validation failures for debugging."""
        logger.error(f"Validation failed for all {len(params)} parameters.")
        
        for i, (module, param_name) in enumerate(params[:5]):  # Log first 5 failures
            tensor = self._get_parameter_tensor(module, param_name)
            if tensor is None:
                logger.error(f"  {i+1}. {module.__class__.__name__}.{param_name}: Parameter not found")
                continue
            
            if not self._is_valid_tensor(tensor):
                logger.error(f"  {i+1}. {module.__class__.__name__}.{param_name}: Invalid tensor (dim={tensor.dim()}, numel={tensor.numel()})")
                continue
            
            if self._requires_structured_pruning():
                if not self._is_valid_for_structured_pruning(tensor):
                    logger.error(
                        f"  {i+1}. {module.__class__.__name__}.{param_name}: "
                        f"Invalid for structured pruning (dim={tensor.dim()}, pruning_dim={self.config.pruning_dim})"
                    )
            else:
                logger.error(f"  {i+1}. {module.__class__.__name__}.{param_name}: Unknown validation failure")

    def get_valid_parameters(self) -> List[Tuple[nn.Module, str]]:
        """Get valid parameters with proper caching."""
        if self._validated_params_cache is not None:
            return self._validated_params_cache

        params = self.prunable_parameters
        if not params:
            raise RuntimeError("No parameters specified for pruning.")

        valid_params = [
            (module, param_name)
            for module, param_name in params
            if self._is_param_valid(module, param_name)
        ]

        if not valid_params:
            self._log_validation_failures(params)
            raise RuntimeError("No valid parameters available for pruning after validation.")

        self._validated_params_cache = valid_params
        return valid_params

    def get_actually_pruned_parameters(self) -> List[Tuple[nn.Module, str]]:
        """Get parameters that have actually been pruned."""
        if not self.prunable_parameters:
            return []

        pruned_params: List[Tuple[nn.Module, str]] = []
        for module, param_name in self.prunable_parameters:
            if (
                pytorch_prune.is_pruned(module)
                and hasattr(module, f"{param_name}_mask")
                and hasattr(module, f"{param_name}_orig")
            ):
                pruned_params.append((module, param_name))

        return pruned_params

    def _check_parameter_validity(
        self,
        param_name: str,
        param: nn.Parameter,
        is_supported: bool,
        is_non_prunable: bool,
        is_recurrent: bool,
        min_param_elements: int,
        prune_bias: bool,
    ) -> Optional[str]:
        """Check if parameter should be skipped and return reason."""
        if is_non_prunable:
            return "non_prunable_layer_type"
        if not is_supported:
            return "unsupported_layer_type"
            
        # Check structured pruning requirements
        if self._requires_structured_pruning() and not self._is_valid_for_structured_pruning(param):
            return "dimension_mismatch_for_structured_pruning"

        if is_recurrent:
            if not param_name.startswith("weight_"):
                return "recurrent_non_weight"
            if param.numel() < min_param_elements:
                return "too_small"
            return None  # Valid recurrent weight parameter
        if param_name == self.WEIGHT_PARAM_NAME:
            if param.numel() < min_param_elements:
                return "too_small"
            return None  # Valid weight parameter
        if param_name == self.BIAS_PARAM_NAME:
            if prune_bias and param.numel() >= min_param_elements:
                return None
            return "too_small" if prune_bias else "bias_disabled"
        return "unsupported_parameter"

    def collect_parameters(self, pl_module: nn.Module, manual_params: Optional[List[Tuple[nn.Module, str]]] = None) -> None:
        """
        Collects and categorizes all parameters in the model into prunable and non-prunable sets.
        """
        # 1. Identify manual selection targets (if any)
        manual_targets_ids = None
        if manual_params is not None:
            manual_targets_ids = set()
            for m, n in manual_params:
                p = getattr(m, n, None)
                if isinstance(p, torch.Tensor):
                    manual_targets_ids.add(id(p))
        
        # 2. Scan ALL parameters in the model
        prunable = []
        non_prunable = []
        seen_tensors = set()
        
        # Helper to get module names for logging
        module_names = {id(m): name for name, m in pl_module.named_modules()}

        for module in pl_module.modules():
            # Skip ScriptModules
            if isinstance(module, torch.jit.ScriptModule):
                continue

            is_supported = isinstance(module, self.DEFAULT_PRUNABLE_LAYER_TYPES)
            is_forbidden = isinstance(module, self.DEFAULT_NON_PRUNABLE_LAYER_TYPES)
            is_recurrent = isinstance(module, (nn.LSTM, nn.GRU))
            
            for name, param in module.named_parameters(recurse=False):
                # Uniqueness check: Ensure each tensor is processed exactly once
                if id(param) in seen_tensors:
                    continue
                seen_tensors.add(id(param))
                
                # Prepare entry data
                entry_tuple = (module, name)
                module_name = module_names.get(id(module), module.__class__.__name__)
                
                # Decision Logic
                should_prune = False
                reason = None

                # Case A: Manual Selection Active
                if manual_targets_ids is not None:
                    if id(param) in manual_targets_ids:
                        if is_forbidden:
                            reason = "non_prunable_layer_type"
                        else:
                            should_prune = True
                    else:
                        reason = "not_in_pruning_plan"
                
                # Case B: Auto Collection
                else:
                    if is_forbidden:
                        reason = "non_prunable_layer_type"
                    elif not is_supported:
                        reason = "unsupported_layer_type"
                    else:
                        # Check validity (size, bias config, etc.)
                        reason = self._check_parameter_validity(
                            name, param, is_supported, is_forbidden,
                            is_recurrent, self.config.min_param_elements, self.config.prune_bias
                        )
                        if reason is None:
                            should_prune = True

                # Assign to lists
                if should_prune:
                    prunable.append(entry_tuple)
                else:
                    non_prunable.append({
                        "module_type": module.__class__.__name__,
                        "module_name": module_name,
                        "param_name": name,
                        "reason": reason or "unknown",
                        "numel": int(param.numel()),
                    })

        # 3. Update state
        self.prunable_parameters = prunable
        self._non_prunable_default_parameters = non_prunable
        
        if not prunable and manual_targets_ids is None:
             self._log("Automatic parameter collection found no prunable parameters.", level=1)
        
        if manual_targets_ids is not None and not prunable:
             self._log("Manual parameter selection resulted in no prunable parameters (all filtered or invalid).", level=1)

        self._log(f"Collected {len(prunable)} prunable and {len(non_prunable)} non-prunable parameters.", level=1)

    def log_overview(self) -> None:
        """Log overview of prunable and non-prunable parameters."""
        if self._parameter_summary_logged:
            return

        if self.config.verbose >= 1:
            prunable = self.prunable_parameters
            non_prunable = self._non_prunable_default_parameters
            self._log_prunable_parameters(prunable)
            self._log_non_prunable_parameters(non_prunable)

        self._parameter_summary_logged = True

    def _log(self, message: str, level: int = 1) -> None:
        """Log messages with appropriate log level."""
        if self.config.verbose >= level:
            if level >= 2:
                logger.debug(message)
            else:
                logger.info(message)

    def _log_prunable_parameters(self, prunable: List[Tuple[nn.Module, str]]) -> None:
        """Log prunable parameters in table format."""
        type_stats = self._collect_prunable_stats(prunable)
        prunable_rows = self._format_prunable_rows(type_stats)

        self._log_table(
            "[Prunable Parameters]",
            ("Module", "Param", "Count", "Elements"),
            prunable_rows,
            level=1
        )

    def _collect_prunable_stats(self, prunable: List[Tuple[nn.Module, str]]) -> Dict:
        """Collect statistics about prunable parameters."""
        type_stats = defaultdict(lambda: defaultdict(lambda: {"count": 0, "params": 0}))
        
        for module, param_name in prunable:
            module_type = module.__class__.__name__
            tensor = getattr(module, param_name, None)
            params = tensor.numel() if isinstance(tensor, torch.Tensor) else 0
            type_stats[module_type][param_name]["count"] += 1
            type_stats[module_type][param_name]["params"] += params
        
        return type_stats

    def _format_prunable_rows(self, type_stats: Dict) -> List[Tuple[str, str, str, str]]:
        """Format prunable parameters for table display."""
        rows = []
        for module_type in sorted(type_stats.keys()):
            for param_name in sorted(type_stats[module_type].keys()):
                stats = type_stats[module_type][param_name]
                rows.append((
                    module_type,
                    param_name,
                    str(stats["count"]),
                    f"{stats['params']:,}",
                ))

        return sorted(rows, key=lambda row: (-int(row[2]), row[0], row[1]))

    def _log_non_prunable_parameters(self, non_prunable: List[Dict[str, str]]) -> None:
        """Log non-prunable parameters in table format."""
        if not non_prunable:
            return

        skipped_stats = self._collect_skipped_stats(non_prunable)
        skipped_rows = self._format_skipped_rows(skipped_stats)

        self._log_table(
            "[Non-prunable Parameters]",
            ("Module", "Param", "Count", "Elements", "Reason"),
            skipped_rows,
            level=1
        )

    def _collect_skipped_stats(self, non_prunable: List[Dict[str, str]]) -> Dict:
        """Collect statistics about non-prunable parameters."""
        skipped_stats = defaultdict(
            lambda: defaultdict(
                lambda: defaultdict(lambda: {"count": 0, "params": 0, "unknown": False})
            )
        )
        
        for entry in non_prunable:
            module_type = entry["module_type"]
            param_name = entry["param_name"]
            reason = entry["reason"]
            numel = entry["numel"]
            
            stats = skipped_stats[module_type][param_name][reason]
            stats["count"] += 1
            if isinstance(numel, int):
                stats["params"] += numel
            else:
                stats["unknown"] = True
        
        return skipped_stats

    def _format_skipped_rows(self, skipped_stats: Dict) -> List[Tuple[str, str, str, str, str]]:
        """Format skipped parameters for table display."""
        rows = []
        for module_type in sorted(skipped_stats.keys()):
            for param_name in sorted(skipped_stats[module_type].keys()):
                for reason, stats in sorted(
                    skipped_stats[module_type][param_name].items(),
                    key=lambda item: (-item[1]["count"], item[0]),
                ):
                    total_params = stats["params"]
                    if total_params > 0:
                        elements = f"{total_params:,}"
                    elif stats["unknown"]:
                        elements = "-"
                    else:
                        elements = "0"
                    
                    rows.append((
                        module_type,
                        param_name,
                        str(stats["count"]),
                        elements,
                        reason,
                    ))
        return rows

    def _log_table(self, title: str, headers: Tuple[str, ...], rows: List[Tuple], level: int = 1) -> None:
        """Log a formatted table."""
        if self.config.verbose < level:
            return

        if not rows:
            rows = [("(none)",) + ("") * (len(headers) - 1)]

        # Calculate column widths
        widths = [len(h) for h in headers]
        for row in rows:
            for idx, col in enumerate(row):
                widths[idx] = max(widths[idx], len(str(col)))

        # Format table
        def format_row(row: Tuple) -> str:
            formatted_cols = []
            for idx, col in enumerate(row):
                formatted_cols.append(str(col).ljust(widths[idx]))
            return "  " + " | ".join(formatted_cols)

        separator = "  " + "-+-".join("-" * width for width in widths)
        
        log_lines = [f"{title}", format_row(headers), separator]
        log_lines.extend(format_row(row) for row in rows)
        
        self._log("\n".join(log_lines), level)

    def compute_sparsity(self, parameters: List[Tuple[nn.Module, str]]) -> float:
        """Compute current sparsity of given parameters."""
        total_params = 0
        pruned_params = 0

        for module, param_name in parameters:
            tensor = getattr(module, param_name, None)
            if not isinstance(tensor, torch.Tensor):
                continue

            mask_name = f"{param_name}_mask"
            if pytorch_prune.is_pruned(module) and hasattr(module, mask_name):
                mask = getattr(module, mask_name)
                if isinstance(mask, torch.Tensor):
                    total_params += mask.numel()
                    pruned_params += (mask == 0).sum().item()
                    continue

            total_params += tensor.numel()
            pruned_params += (tensor == 0).sum().item()

        if total_params == 0:
            return 0.0
        return pruned_params / total_params


class ParameterSnapshotter:
    """Handles serialization/deserialization of pruning parameter selections."""

    @staticmethod
    def serialize(params: List[Tuple[nn.Module, str]], root_module: nn.Module) -> List[Dict[str, str]]:
        """Serialize pruning parameters for checkpoint storage."""
        module_name_map = {id(module): name for name, module in root_module.named_modules()}
        serialized: List[Dict[str, str]] = []
        
        for module, param_name in params:
            module_id = id(module)
            if module_id not in module_name_map:
                raise KeyError(
                    f"ParameterSnapshotter.serialize: module id {module_id} not found in root module graph"
                )
            serialized.append({
                "module_name": module_name_map[module_id],
                "param_name": param_name,
            })
        return serialized

    @staticmethod
    def restore(saved_info: List[Dict[str, str]], root_module: nn.Module) -> List[Tuple[nn.Module, str]]:
        """Restore pruning parameters from checkpoint data."""
        module_lookup = dict(root_module.named_modules())
        restored: List[Tuple[nn.Module, str]] = []

        for entry in saved_info:
            module_name = entry['module_name']
            param_name = entry['param_name']
            
            if module_name not in module_lookup:
                raise KeyError(
                    f"ParameterSnapshotter: Missing module {module_name} while restoring parameters"
                )

            module = module_lookup[module_name]
            if not hasattr(module, param_name):
                raise AttributeError(
                    f"ParameterSnapshotter: Module {module_name} missing parameter {param_name}"
                )

            restored.append((module, param_name))

        if not restored:
            raise RuntimeError("ParameterSnapshotter: No parameters restored from checkpoint metadata")

        return restored
