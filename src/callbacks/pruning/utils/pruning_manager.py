"""
PruningManager: A unified handler for parameter grouping, sparsity, and regularization.
"""

import importlib
import logging
import re
from typing import Any, Dict, List

import torch
import torch.nn as nn
from pytorch_lightning import LightningModule

from .sparsity_applier import SparsityApplier

logger = logging.getLogger(__name__)


def _safe_re_search(pattern: str, string: str) -> bool:
    """A wrapper for re.search that catches re.error and returns False."""
    try:
        return bool(re.search(pattern, string))
    except re.error as e:
        logger.warning(
            f"Invalid regex pattern '{pattern}' ignored. Error: {e}"
        )
        return False


def _resolve_layer_types(type_strings: list) -> tuple:
    """Resolve dotted type strings to actual classes; an unresolvable entry
    raises (ImportError/AttributeError) naming the bad string."""
    resolved = []
    for type_str in type_strings or []:
        module_path, _, class_name = type_str.rpartition(".")
        assert module_path, f"layer_types entry {type_str!r} is not dotted"
        mod = importlib.import_module(module_path)
        resolved.append(getattr(mod, class_name))
    return tuple(resolved)


def module_param_matches(
    mod_name: str,
    mod: nn.Module,
    p_name: str,
    config: Dict[str, Any],
    resolved_types: tuple,
) -> bool:
    """Whether a module's own parameter matches a group config.

    Shared by PruningManager (param assignment) and the ERK shape collector so
    the ERK layer set is exactly the regularized set. ``resolved_types`` is
    the group's ``layer_types`` resolved once via ``_resolve_layer_types``.
    """
    if config.get("is_fallback"):
        return False

    exclude_patterns = config.get("exclude_module_name_patterns")
    if exclude_patterns and any(
        _safe_re_search(pattern, mod_name) for pattern in exclude_patterns
    ):
        return False

    if resolved_types and not isinstance(mod, resolved_types):
        return False

    param_names = config.get("param_names")
    if param_names is not None and p_name not in param_names:
        return False

    include_patterns = config.get("module_name_patterns")
    if include_patterns:
        return any(
            _safe_re_search(pattern, mod_name) for pattern in include_patterns
        )
    return True


class PruningManager:
    """Manages all pruning-related configuration and actions from a single
    source of truth.

    This class interprets a list of group configurations to:
    1.  Separate a model's parameters into distinct groups.
    2.  Provide these groups to an optimizer with group-specific settings (e.g., regularization).
    3.  Apply initial sparsity to each group according to its configuration.

    Args:
        pl_module (LightningModule): The model containing the parameters.
        group_configs (List[Dict[str, Any]]): A list defining the pruning groups.
            The last group should be a fallback group with `is_fallback: True`.
    """

    def __init__(
        self, pl_module: LightningModule, group_configs: List[Dict[str, Any]]
    ):
        self.pl_module = pl_module
        self._raw_configs = group_configs
        self.processed_groups = self._process_configs()

    def _process_configs(self) -> List[Dict[str, Any]]:
        processed_groups = []
        for config in self._raw_configs:
            processed_groups.append(
                {
                    "params": [],
                    "config": config,
                    "resolved_types": _resolve_layer_types(
                        config.get("layer_types")
                    ),
                    "applier": SparsityApplier(
                        **config.get("pruning_config", {})
                    ),
                }
            )

        fallback_group = next(
            (g for g in processed_groups if g["config"].get("is_fallback")),
            None,
        )
        if not fallback_group:
            raise ValueError(
                "The `group_configs` must include one fallback group with `'is_fallback': True`."
            )

        # Tied params appear under every owning module; assign them once.
        seen_param_ids = set()
        for mod_name, mod in self.pl_module.named_modules():
            for p_name, param in mod.named_parameters(recurse=False):
                if not param.requires_grad or id(param) in seen_param_ids:
                    continue
                seen_param_ids.add(id(param))

                assigned = False
                for group in processed_groups:
                    if module_param_matches(
                        mod_name,
                        mod,
                        p_name,
                        group["config"],
                        group["resolved_types"],
                    ):
                        group["params"].append(param)
                        assigned = True
                        break

                if not assigned:
                    fallback_group["params"].append(param)

        return [g for g in processed_groups if g["params"]]

    def get_optimizer_param_groups(self) -> List[Dict[str, Any]]:
        optimizer_groups = []
        for group in self.processed_groups:
            opt_settings = group["config"].get("optimizer_settings", {}).copy()

            optimizer_group = {
                "name": group["config"].get("name"),
                "params": group["params"],
                **opt_settings,
            }
            optimizer_groups.append(optimizer_group)
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
