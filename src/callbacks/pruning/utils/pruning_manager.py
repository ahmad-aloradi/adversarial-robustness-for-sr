"""
PruningManager: A unified handler for parameter grouping, sparsity, and regularization.
"""

import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from typing import List, Dict, Any
import importlib

from .sparsity_applier import SparsityApplier


class PruningManager:
    """
    Manages all pruning-related configuration and actions from a single source of truth.

    This class interprets a list of group configurations to:
    1.  Separate a model's parameters into distinct groups.
    2.  Provide these groups to an optimizer with group-specific settings (e.g., regularization).
    3.  Apply initial sparsity to each group according to its configuration.

    This unified approach ensures consistency between initial sparsity and ongoing regularization.

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
        """
        Assigns every parameter in the model to a group based on a "first match wins"
        strategy, ensuring each parameter belongs to exactly one group.
        """
        # Initialize empty lists of parameters for each group config
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

        # Use a "first match wins" strategy to assign each parameter to a group
        for _, param in self.pl_module.named_parameters():
            if not param.requires_grad:
                continue

            assigned = False
            for group in processed_groups:
                if self._param_matches_config(param, group["config"]):
                    group["params"].append(param)
                    assigned = True
                    break # Move to the next parameter
            
            if not assigned:
                # This should not happen if a fallback is present, but is good practice
                fallback_group["params"].append(param)

        # Filter out groups that ended up with no parameters
        return [g for g in processed_groups if g["params"]]

    def _param_matches_config(self, param: torch.nn.Parameter, config: Dict[str, Any]) -> bool:
        """Checks if a parameter's owning module and name match the group config."""
        if config.get("is_fallback"):
            return False

        # **FIX**: Resolve layer type strings from the config into actual Python types.
        layer_type_strings = config.get("layer_types")
        if not layer_type_strings:
            return False # A non-fallback group must specify layer_types

        resolved_types = []
        for type_str in layer_type_strings:
            try:
                module_path, class_name = type_str.rsplit('.', 1)
                module = importlib.import_module(module_path)
                resolved_types.append(getattr(module, class_name))
            except (ImportError, AttributeError, ValueError):
                # Optionally log a warning here if a type string is invalid
                continue
        
        if not resolved_types:
            return False # Don't match if all type strings failed to resolve

        layer_types_tuple = tuple(resolved_types)
        param_names = config.get("param_names")

        for mod_name, mod in self.pl_module.named_modules():
            for p_name, p_obj in mod.named_parameters(recurse=False):
                if p_obj is param:
                    # Now check if the module's type and the parameter's name match
                    if isinstance(mod, layer_types_tuple):
                        if param_names is None or p_name in param_names:
                            return True
        return False

    def get_optimizer_param_groups(self) -> List[Dict[str, Any]]:
        """
        Returns parameter groups formatted for a PyTorch optimizer.
        """
        optimizer_groups = []
        for group in self.processed_groups:
            opt_settings = group["config"].get("optimizer_settings", {})
            optimizer_groups.append({
                "params": group["params"],
                **opt_settings,
            })
        return optimizer_groups

    def apply_initial_sparsity(self):
        """
        Applies the configured initial sparsity for each group to its parameters.
        """
        for group in self.processed_groups:
            applier = group["applier"]
            for param in group["params"]:
                applier.apply(param)

    def get_pruned_parameters(self) -> List[torch.Tensor]:
        """Returns a flat list of all parameters being pruned."""
        pruned_params = []
        for group in self.processed_groups:
            # Assume a group is "pruned" if it has a non-zero sparsity rate
            if group["applier"].sparsity_rate > 0:
                pruned_params.extend(group["params"])
        return pruned_params