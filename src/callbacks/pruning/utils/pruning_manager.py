"""
PruningManager: A unified handler for parameter grouping, sparsity, and regularization.
"""

import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from typing import List, Dict, Any
import importlib
import re
import logging

from .sparsity_applier import SparsityApplier

logger = logging.getLogger(__name__)


class PruningManager:
    """
    Manages all pruning-related configuration and actions from a single source of truth.
    
    This class interprets a list of group configurations to:
    1.  Separate a model's parameters into distinct groups.
    2.  Provide these groups to an optimizer with group-specific settings (e.g., regularization).
    3.  Apply initial sparsity to each group according to its configuration.

    Args:
        pl_module (LightningModule): The model containing the parameters.
        group_configs (List[Dict[str, Any]]): A list defining the pruning groups.
            The last group should be a fallback group with `is_fallback: True`.
    """
    def __init__(self, pl_module: LightningModule, group_configs: List[Dict[str, Any]]):
        self.pl_module = pl_module
        self._raw_configs = group_configs
        self.processed_groups = self._process_configs()

    def _process_configs(self) -> List[Dict[str, Any]]:
        processed_groups = []
        for config in self._raw_configs:
            processed_groups.append({
                "params": [],
                "config": config,
                "applier": SparsityApplier(**config.get("pruning_config", {})),
            })
        
        fallback_group = next((g for g in processed_groups if g["config"].get("is_fallback")), None)
        if not fallback_group:
            raise ValueError("The `group_configs` must include one fallback group with `'is_fallback': True`.")

        for _, param in self.pl_module.named_parameters():
            if not param.requires_grad:
                continue

            assigned = False
            for group in processed_groups:
                if self._param_matches_config(param, group["config"]):
                    group["params"].append(param)
                    assigned = True
                    break
            
            if not assigned:
                fallback_group["params"].append(param)

        return [g for g in processed_groups if g["params"]]

    def _safe_re_search(self, pattern: str, string: str) -> bool:
        """A wrapper for re.search that catches re.error and returns False."""
        try:
            return bool(re.search(pattern, string))
        except re.error as e:
            logger.warning(f"Invalid regex pattern '{pattern}' ignored. Error: {e}")
            return False

    def _param_matches_config(self, param: torch.nn.Parameter, config: Dict[str, Any]) -> bool:
        """Checks if a parameter's owning module and name match the group config."""
        if config.get("is_fallback"):
            return False

        layer_type_strings = config.get("layer_types")
        if not layer_type_strings:
            return False

        resolved_types = []
        for type_str in layer_type_strings:
            try:
                module_path, class_name = type_str.rsplit('.', 1)
                module = importlib.import_module(module_path)
                resolved_types.append(getattr(module, class_name))
            except (ImportError, AttributeError, ValueError):
                continue
        
        if not resolved_types:
            return False

        layer_types_tuple = tuple(resolved_types)
        param_names = config.get("param_names")
        include_patterns = config.get("module_name_patterns")
        exclude_patterns = config.get("exclude_module_name_patterns")
        
        for mod_name, mod in self.pl_module.named_modules():
            for p_name, p_obj in mod.named_parameters(recurse=False):
                if p_obj is param:
                    # Found the parameter's direct parent module. Now perform all checks.

                    # 1. Exclusion check (highest priority)
                    if exclude_patterns and any(self._safe_re_search(pattern, mod_name) for pattern in exclude_patterns):
                        return False # Excluded. Do not consider for this group.

                    # 2. Inclusion checks
                    type_matches = isinstance(mod, layer_types_tuple)
                    param_name_matches = param_names is None or p_name in param_names

                    if type_matches and param_name_matches:
                        if include_patterns:
                            if any(self._safe_re_search(pattern, mod_name) for pattern in include_patterns):
                                return True # Include pattern matched
                        else:
                            # If no include patterns are provided, it's a match
                            return True
        return False

    def get_optimizer_param_groups(self) -> List[Dict[str, Any]]:
        optimizer_groups = []
        for group in self.processed_groups:
            opt_settings = group["config"].get("optimizer_settings", {}).copy()

            optimizer_groups.append({
                "params": group["params"],
                **opt_settings,
            })
        return optimizer_groups

    def apply_initial_sparsity(self):
        for group in self.processed_groups:
            applier = group["applier"]
            for param in group["params"]:
                applier.apply(param)

    def get_pruned_parameters(self) -> List[torch.Tensor]:
        pruned_params = []
        for group in self.processed_groups:
            if group["applier"].sparsity_rate > 0:
                pruned_params.extend(group["params"])
        return pruned_params