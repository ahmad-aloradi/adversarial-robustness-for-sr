"""
PruningManager: A unified handler for parameter grouping, sparsity, and regularization.
"""

import importlib
import logging
import re
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from pytorch_lightning import LightningModule

from .sparsity_applier import SparsityApplier

logger = logging.getLogger(__name__)

_REGNONE_TARGET = "src.callbacks.pruning.bregman.bregman_regularizers.RegNone"
# Where the trainable per-layer log-scale factors s_g live on the
# LightningModule, and the anchored pattern the optimizer group uses to grab
# exactly those params. The "log_" name is load-bearing: it keys the state on
# s, so a strict load of a c-valued checkpoint fails loud (key mismatch).
SCALES_ATTR = "bregman_log_scales"


def _sanitize_scale_key(module_name: str) -> str:
    """ParameterDict keys forbid '.'; map a module name to a legal key."""
    return module_name.replace(".", "__")


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


def create_scale_params(
    pl_module: LightningModule,
    group_configs: List[Dict[str, Any]],
) -> Optional[str]:
    """Register ``pl_module.bregman_log_scales`` for trainable allocation.

    Scans ``group_configs`` for the single ``trainable_scales`` group, collects
    its prunable layer names (same predicate as the optimizer assignment), and
    registers one scalar log-scale Parameter s_g per layer in an
    ``nn.ParameterDict`` keyed by the ``.``-sanitized module name. Each is
    initialized to 0 (neutral: c_g = exp(0) = 1, λ_eff = λ_global · 1).

    Must run from ``LightningModule.setup`` (before Lightning restores model
    state on resume) so the keys exist for the strict ``load_state_dict``.
    Idempotent: a second call with a matching layer set is a no-op, so a
    restore that happens between calls is never clobbered.

    Returns the attribute name when created, else ``None`` (no trainable group).
    """
    from .layer_shapes import collect_prunable_layer_shapes

    trainable = [cfg for cfg in group_configs if cfg.get("trainable_scales")]
    if not trainable:
        return None
    if len(trainable) > 1:
        raise ValueError(
            f"At most one trainable_scales group is allowed, found "
            f"{len(trainable)}."
        )

    config = trainable[0]
    resolved_types = _resolve_layer_types(config["layer_types"])
    shapes = collect_prunable_layer_shapes(pl_module, config, resolved_types)
    if not shapes:
        raise ValueError(
            f"trainable_scales group '{config.get('name')}' matched no "
            "prunable weights; check layer_types/module_name_patterns."
        )

    keys = {s.name: _sanitize_scale_key(s.name) for s in shapes}
    if len(set(keys.values())) != len(keys):
        raise ValueError(
            "Sanitized scale keys collide: two module names map to the "
            "same '.'-free key."
        )

    if hasattr(pl_module, SCALES_ATTR):
        existing = set(getattr(pl_module, SCALES_ATTR).keys())
        if existing != set(keys.values()):
            raise ValueError(
                f"{SCALES_ATTR} already registered with a different layer "
                f"set: {sorted(existing)} vs {sorted(keys.values())}."
            )
        return SCALES_ATTR

    setattr(
        pl_module,
        SCALES_ATTR,
        nn.ParameterDict(
            {keys[s.name]: nn.Parameter(torch.zeros(())) for s in shapes}
        ),
    )
    return SCALES_ATTR


class PruningManager:
    """Manages all pruning-related configuration and actions from a single
    source of truth.

    This class interprets a list of group configurations to:
    1.  Separate a model's parameters into distinct groups.
    2.  Provide these groups to an optimizer with group-specific settings (e.g., regularization).
    3.  Apply initial sparsity to each group according to its configuration.

    A group config may carry ``trainable_scales: {scale_lr,
    initial_sparsity}``; it is expanded at construction into one group per
    matching weight (each carrying a ``trainable_scale`` marker) plus one group
    holding the per-layer log-scale Parameters.

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
        expanded_configs: List[Dict[str, Any]] = []
        for config in self._raw_configs:
            if config.get("trainable_scales"):
                expanded_configs.extend(
                    self._expand_trainable_scale_group(config)
                )
            else:
                expanded_configs.append(config)

        processed_groups = []
        for config in expanded_configs:
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

    def _expand_trainable_scale_group(
        self, config: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Expand one ``trainable_scales`` group into one config per weight.

        The global LambdaScheduler owns the sparsity level; each per-layer scalar
        scale factor c owns its share of the allocation. Each emitted config
        targets a single module by its exact name, carries the base group's
        optimizer_settings (reg + lambda_scale), a uniform initial
        ``sparsity_rate``, and a ``trainable_scale`` marker pointing at its
        scale Parameter. One extra ``bregman_log_scales`` group holds those
        params, trained by the same optimizer in a RegNone group at its own lr.
        """
        from .layer_shapes import collect_prunable_layer_shapes

        ts_cfg = config["trainable_scales"]
        scale_lr = ts_cfg.get("scale_lr", 1e-3)
        initial_sparsity = ts_cfg.get("initial_sparsity", 0.0)

        assert config.get("layer_types"), (
            f"trainable_scales group '{config.get('name')}' must declare "
            "layer_types to define the prunable weight set."
        )
        resolved_types = _resolve_layer_types(config["layer_types"])
        shapes = collect_prunable_layer_shapes(
            self.pl_module, config, resolved_types
        )
        if not shapes:
            raise ValueError(
                f"trainable_scales group '{config.get('name')}' matched no "
                "prunable weights; check layer_types/module_name_patterns."
            )

        base_name = config.get("name", "trainable")
        base_opt_settings = config.get("optimizer_settings", {})
        base_layer_types = config.get("layer_types")
        base_pruning_type = config.get("pruning_config", {}).get(
            "pruning_type", "unstructured"
        )

        expanded: List[Dict[str, Any]] = []
        for shape in shapes:
            mod_name = shape.name
            layer_config: Dict[str, Any] = {
                "name": f"{base_name}__{mod_name}",
                "param_names": ["weight"],
                # Anchored exact match so each config grabs exactly its weight.
                "module_name_patterns": ["^" + re.escape(mod_name) + "$"],
                # Shared base node so ${_bregman_lambda} resolves as it does for
                # the normal groups; sv.py instantiates a fresh reg per group.
                "optimizer_settings": base_opt_settings,
                "pruning_config": {
                    "pruning_type": base_pruning_type,
                    "sparsity_rate": initial_sparsity,
                },
                "trainable_scale": True,
                "trainable_scale_key": _sanitize_scale_key(mod_name),
            }
            if base_layer_types is not None:
                layer_config["layer_types"] = list(base_layer_types)
            expanded.append(layer_config)

        # One group holding the scale-factor params, trained by the same
        # optimizer in a RegNone (Adam-like) group at its own lr.
        expanded.append(
            {
                "name": SCALES_ATTR,
                "module_name_patterns": ["^" + SCALES_ATTR + "$"],
                "optimizer_settings": {
                    "reg": {"_target_": _REGNONE_TARGET},
                    "lambda_scale": 0.0,
                    "lr": scale_lr,
                },
            }
        )
        return expanded

    def get_optimizer_param_groups(self) -> List[Dict[str, Any]]:
        optimizer_groups = []
        for group in self.processed_groups:
            opt_settings = group["config"].get("optimizer_settings", {}).copy()

            optimizer_group = {
                "name": group["config"].get("name"),
                "params": group["params"],
                **opt_settings,
            }
            # Trainable-allocation markers ride onto the optimizer group so the
            # pruner can map a weight group to its scale Parameter and read its
            # decay/floor without re-parsing names.
            if group["config"].get("trainable_scale"):
                for key in ("trainable_scale", "trainable_scale_key"):
                    optimizer_group[key] = group["config"][key]
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
