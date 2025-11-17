import math
from typing import List, Tuple, Callable, Optional, Union, Type, Dict

from pytorch_lightning.callbacks.pruning import ModelPruning
import torch
import torch.nn as nn
import torch.nn.utils.prune as pytorch_prune
from pytorch_lightning import LightningModule, Trainer

from src import utils

# Constants are defined at the module level for clarity and reusability.
log = utils.get_pylogger(__name__)
WEIGHT_PARAM_NAME = 'weight'
BIAS_PARAM_NAME = 'bias'
DEFAULT_PRUNABLE_LAYER_TYPES: Tuple[Type[nn.Module], ...] = (
    # Core Layers
    nn.Linear,
    nn.Embedding,
    
    # Convolutional Layers
    nn.Conv1d,
    nn.Conv2d,
    nn.Conv3d,
    nn.ConvTranspose1d,
    nn.ConvTranspose2d,
    nn.ConvTranspose3d,    
)
NORMALIZATION_LAYER_TYPES: Tuple[Type[nn.Module], ...] = (
    nn.BatchNorm1d,
    nn.BatchNorm2d,
    nn.BatchNorm3d,
    nn.LayerNorm,
)
RECURRENT_PRUNABLE_LAYER_TYPES: Tuple[Type[nn.Module], ...] = (
    nn.LSTM,
    nn.GRU,
)
DEFAULT_PRUNABLE_LAYER_TYPES += RECURRENT_PRUNABLE_LAYER_TYPES

VALID_PRUNING_TRIGGERS: Tuple[str, ...] = ("pre_training", "epoch_start", "epoch_end")


class ParameterSnapshotter:
    """Handles serialization/deserialization of pruning parameter selections."""

    @staticmethod
    def serialize(params: List[Tuple[nn.Module, str]], root_module: nn.Module) -> List[Dict[str, str]]:
        module_name_map = {id(module): name for name, module in root_module.named_modules()}
        serialized: List[Dict[str, str]] = []
        for module, param_name in params:
            module_id = id(module)
            if module_id not in module_name_map:
                raise KeyError(
                    f"ParameterSnapshotter.serialize: module id {module_id} not found in root module graph"
                )
            serialized.append(
                {
                    "module_name": module_name_map[module_id],
                    "param_name": param_name,
                }
            )
        return serialized

    @staticmethod
    def restore(
        saved_info: List[Dict[str, str]],
        root_module: nn.Module,
    ) -> List[Tuple[nn.Module, str]]:
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


class ParameterSelectionMixin:
    """Encapsulates parameter collection, validation, and caching."""

    def _init_parameter_cache(self) -> None:
        self._validated_params_cache: Optional[List[Tuple[nn.Module, str]]] = None
        self._non_prunable_default_parameters: List[Dict[str, str]] = []

    def _invalidate_parameter_cache(self) -> None:
        self._validated_params_cache = None

    def _is_param_valid(self, module: nn.Module, param_name: str) -> bool:
        tensor = getattr(module, f"{param_name}_orig", None)
        if tensor is None:
            tensor = getattr(module, param_name, None)

        if tensor is None or not isinstance(tensor, torch.Tensor):
            return False
        if tensor.dim() == 0 or tensor.numel() < 2:
            return False

        if "ln_structured" in getattr(self, "_pruning_fn_name", ""):
            if tensor.dim() < 2:
                return False
            pruning_dim = getattr(self, "pruning_dim", None)
            if pruning_dim is None or tensor.dim() <= pruning_dim:
                return False

        return True

    def _get_valid_parameters(self) -> List[Tuple[nn.Module, str]]:
        if self._validated_params_cache is not None:
            return self._validated_params_cache

        params = getattr(self, "_parameters_to_prune", None)
        if not params:
            raise RuntimeError("No parameters specified for pruning. Call _collect_parameters_if_needed first.")

        valid_params = [
            (module, param_name)
            for module, param_name in params
            if self._is_param_valid(module, param_name)
        ]

        if not valid_params:
            raise RuntimeError("No valid parameters available for pruning after validation.")

        self._validated_params_cache = valid_params
        return valid_params

    def _get_invalid_parameters(self) -> List[Tuple[nn.Module, str]]:
        params = getattr(self, "_parameters_to_prune", None)
        if not params:
            raise RuntimeError("Cannot query invalid parameters before parameters_to_prune is defined.")

        return [
            (module, param_name)
            for module, param_name in params
            if not self._is_param_valid(module, param_name)
        ]

    def _discover_untracked_parameters(
        self,
        pl_module: nn.Module,
        known_params: List[Tuple[nn.Module, str]],
    ) -> List[Dict[str, str]]:
        """Fallback pass to flag any parameter not in the prunable set."""
        known_keys = {(id(module), param_name) for module, param_name in known_params}
        discovered: List[Dict[str, str]] = []
        for module_name, module in pl_module.named_modules():
            for param_name, param in module.named_parameters(recurse=False):
                if not isinstance(param, nn.Parameter):
                    continue
                if (id(module), param_name) in known_keys:
                    continue
                discovered.append(
                    {
                        "module_type": module.__class__.__name__,
                        "module_name": module_name,
                        "param_name": param_name,
                        "reason": "not_selected_for_pruning",
                        "numel": int(param.numel()) if isinstance(param, torch.Tensor) else None,
                    }
                )
        return discovered

    @staticmethod
    def _collect_default_prunable_parameters(
        module: nn.Module,
        min_param_elements: int,
        prune_bias: bool,
    ) -> Tuple[List[Tuple[nn.Module, str]], List[Dict[str, str]]]:
        prunable: List[Tuple[nn.Module, str]] = []
        skipped: List[Dict[str, str]] = []

        module_name_map = {
            id(sub_module): name or module.__class__.__name__
            for name, sub_module in module.named_modules()
        }

        def mark_skip(
            sub_module: nn.Module,
            param_name: str,
            reason: str,
            param: Optional[torch.Tensor],
        ) -> None:
            skipped.append(
                {
                    "module_type": sub_module.__class__.__name__,
                    "module_name": module_name_map[id(sub_module)],
                    "param_name": param_name,
                    "reason": reason,
                    "numel": int(param.numel()) if isinstance(param, torch.Tensor) else None,
                }
            )

        for sub_module in module.modules():
            if isinstance(sub_module, torch.jit.ScriptModule):
                raise ValueError("ScriptModules are not supported.")

            is_supported = isinstance(sub_module, DEFAULT_PRUNABLE_LAYER_TYPES)
            is_recurrent = isinstance(sub_module, RECURRENT_PRUNABLE_LAYER_TYPES)

            for param_name, param in sub_module.named_parameters(recurse=False):
                if not isinstance(param, nn.Parameter):
                    continue

                reason: Optional[str] = None

                if not is_supported:
                    reason = "unsupported_layer_type"
                elif is_recurrent:
                    if not param_name.startswith("weight_"):
                        reason = "not_weight_parameter"
                    elif param.numel() < min_param_elements:
                        reason = "too_small"
                elif param_name == WEIGHT_PARAM_NAME:
                    if param.numel() >= min_param_elements:
                        prunable.append((sub_module, param_name))
                        continue
                    reason = "too_small"
                elif param_name == BIAS_PARAM_NAME:
                    if prune_bias and param.numel() >= min_param_elements:
                        prunable.append((sub_module, param_name))
                        continue
                    reason = "too_small" if prune_bias else "bias_disabled"
                else:
                    reason = "unsupported_parameter"

                if reason:
                    mark_skip(sub_module, param_name, reason, param)

        # Preserve deterministic insertion order while removing duplicates
        seen = set()
        deduped: List[Tuple[nn.Module, str]] = []
        for m, n in prunable:
            key = (id(m), n)
            if key in seen:
                continue
            seen.add(key)
            deduped.append((m, n))

        if not deduped:
            raise RuntimeError("Automatic parameter collection produced no prunable parameters.")

        return deduped, skipped

    def _get_actually_pruned_parameters(self) -> List[Tuple[nn.Module, str]]:
        if not getattr(self, "_parameters_to_prune", None):
            return []

        pruned_params: List[Tuple[nn.Module, str]] = []
        for module, param_name in self._parameters_to_prune:
            if (
                pytorch_prune.is_pruned(module)
                and hasattr(module, f"{param_name}_mask")
                and hasattr(module, f"{param_name}_orig")
            ):
                pruned_params.append((module, param_name))

        return pruned_params


class MagnitudePruner(ParameterSelectionMixin, ModelPruning):
    """
    MagnitudePruner - Enhanced version of PyTorch Lightning's ModelPruning
    
    Adds scheduled pruning, checkpoint coordination, and optional lottery-ticket flows
    while keeping the implementation intentionally lean.
    
    Parameters
    ----------
    pruning_fn : Union[Callable, str], default="l1_unstructured"
        The pruning function or name from torch.nn.utils.prune
    amount : Union[int, float], default=0.5
        Amount of parameters to prune
    use_global_unstructured : bool, default=True
        Whether to apply pruning globally across all parameters
    apply_pruning : bool, default=True
        Whether to actually apply pruning
    make_pruning_permanent : bool, default=True
        Whether to permanently remove pruned weights after training
    use_lottery_ticket_hypothesis : bool, default=False
        Whether to reset remaining weights to original values
    resample_parameters : bool, default=False
        Used with lottery ticket hypothesis for parameter resampling
    parameters_to_prune : Optional[List[Tuple[nn.Module, str]]], default=None
        Specific parameters to prune
    pruning_dim : Optional[int], default=None
        Dimension for structured pruning
    pruning_norm : Optional[int], default=None
        Norm for structured pruning
    verbose : int, default=0
        Verbosity level (0=silent, 1=basic, 2=detailed)
    pruning_trigger : {"pre_training", "epoch_start", "epoch_end"}, default="epoch_start"
        When to run pruning: once before fitting, at each epoch start, or at each epoch end
    scheduled_pruning : bool, default=False
        Whether to gradually increase pruning over epochs
    final_amount : Optional[float], default=None
        Final pruning amount (uses amount if None)
    epochs_to_ramp : int, default=10
        Epochs to ramp linearly from 0 to `final_amount`
    collect_metrics : bool, default=False
        Reserved for backward compatibility (no-op)
    save_when_sparser_than : Optional[float], default=None
        If set, saves a checkpoint when model sparsity exceeds this value
    min_param_elements : int, default=100
        Minimum number of elements in a parameter to be considered for pruning
    prune_bias : bool, default=False
        Whether to include bias parameters in pruning
    **kwargs : Additional arguments passed to parent ModelPruning
    """
    
    def _log(self, message: str, level: int = 1) -> None:
        """Helper method to reduce verbose logging clutter."""
        if self.verbose >= level:
            if level >= 2:
                log.debug(message)
            else:
                log.info(message)
    
    def __init__(
        self,
        pruning_fn: Union[Callable, str] = "l1_unstructured",
        amount: Union[int, float] = 0.8,
        use_global_unstructured: bool = True,
        apply_pruning: bool = True,
        make_pruning_permanent: bool = True,
        use_lottery_ticket_hypothesis: bool = False,
        resample_parameters: bool = False,
        parameters_to_prune: Optional[List[Tuple[nn.Module, str]]] = None,
        pruning_dim: Optional[int] = None,
        pruning_norm: Optional[int] = None,
        verbose: int = 0,
        pruning_trigger: str = "epoch_start",
        scheduled_pruning: bool = False,
        final_amount: Optional[float] = None,
        epochs_to_ramp: int = 10,
        collect_metrics: bool = False,
        save_when_sparser_than: Optional[float] = None,
        prune_bias: bool = False,
        min_param_elements: int = 100,
        **kwargs
    ):
        # Store all key parameters first since they're used throughout the class
        self.verbose = verbose
        self._pruning_fn_name = pruning_fn if isinstance(pruning_fn, str) else getattr(pruning_fn, "__name__", "unknown_callable")
        self.use_global_unstructured = use_global_unstructured
        self._apply_pruning = apply_pruning  # Renamed to avoid conflict with method
        self.should_make_pruning_permanent = make_pruning_permanent
        self._use_lottery_ticket_hypothesis = use_lottery_ticket_hypothesis  # Renamed to avoid conflict
        self.resample_parameters = resample_parameters
        self._parameters_to_prune = parameters_to_prune
        self.pruning_dim = pruning_dim
        self.pruning_norm = pruning_norm

        assert pruning_trigger in VALID_PRUNING_TRIGGERS, f"Unexpected pruning_trigger '{pruning_trigger}'"
        self._pruning_trigger = pruning_trigger

        # Validate structured pruning requirements
        if "ln_structured" in self._pruning_fn_name and pruning_dim is None:
            raise ValueError("`pruning_dim` must be specified for structured pruning.")

        self.scheduled_pruning = scheduled_pruning

        if self._pruning_trigger == "pre_training" and self.scheduled_pruning:
            raise ValueError("Scheduled pruning requires epoch-based triggers, not pre_training.")

        if scheduled_pruning:
            # scheduled pruning: linearly increase sparsity from 0 -> final_amount
            self.final_amount = final_amount if final_amount is not None else amount
            self.epochs_to_ramp = max(1, epochs_to_ramp)
            self._validate_params(None, True, self.final_amount)
            # start with no pruning applied until the schedule advances
            effective_amount = 0.0
        else:
            # fixed mode
            self.amount = amount
            self.final_amount = None
            self.epochs_to_ramp = None
            self._validate_params(self.amount, False, None)
            effective_amount = self.amount

        super().__init__(
            pruning_fn=pruning_fn,
            amount=effective_amount,
            use_global_unstructured=use_global_unstructured,
            apply_pruning=apply_pruning,
            make_pruning_permanent=make_pruning_permanent,
            use_lottery_ticket_hypothesis=use_lottery_ticket_hypothesis,
            resample_parameters=resample_parameters,
            parameters_to_prune=parameters_to_prune,
            pruning_dim=pruning_dim,
            pruning_norm=pruning_norm,
            verbose=verbose,
            prune_on_train_epoch_end=False,
        )
        
        # Initialize caching
        self.collect_metrics = collect_metrics  # retained for backward compatibility, no-op now
        self._init_parameter_cache()
        self.save_when_sparser_than = save_when_sparser_than
        self._checkpoint_callbacks = []
        self._checkpoint_original_settings = {}
        self._best_score_has_been_reset = False
        
        # Track training state for proper resumption handling
        self._is_resuming = False
        self._pruning_completed = False
        self._saved_state = None
        self._loaded_from_checkpoint = False
        self._latest_sparsity = 0.0
        self._parameter_summary_logged = False
        self._prune_bias = prune_bias
        self._min_param_elements = min_param_elements

    def _validate_params(self, amount, scheduled_pruning, final_amount):
        """Validate initialization parameters."""
        if amount is not None and isinstance(amount, float) and not (0 <= amount <= 1):
            raise ValueError(f"Pruning amount {amount} must be between 0 and 1")
            
        if scheduled_pruning:
            if final_amount is None or not (0 <= final_amount <= 1):
                raise ValueError(f"Final amount {final_amount} must be between 0 and 1")

    def _get_sparsity_target(self, current_epoch: int) -> float:
        """Calculate the target sparsity for the current pruning step."""
        if not self.scheduled_pruning:
            return self.amount

        # treat negative epoch (pre-fit) as before schedule starts -> 0.0
        if current_epoch < 0:
            progress = 0.0
        else:
            # map epoch indices 0..(epochs_to_ramp-1) to progress 0..1
            denom = max(1, self.epochs_to_ramp - 1)
            progress = max(0.0, min(1.0, float(current_epoch) / denom))

        return progress * self.final_amount

    def _forecast_final_sparsity(self, trainer: Trainer) -> Optional[float]:
        """Predict the sparsity reachable given the configured max_epochs."""
        assert self.scheduled_pruning, "Forecasting only applies to scheduled pruning"

        max_epochs = getattr(trainer, "max_epochs")
        assert isinstance(max_epochs, (int, float)), "trainer.max_epochs must be int or float"

        max_epochs_int = int(max_epochs)
        assert max_epochs_int >= 1, "trainer.max_epochs must be at least 1"

        final_epoch_index = max_epochs_int - 1
        return self._get_sparsity_target(final_epoch_index)

    def _warn_if_schedule_cannot_reach_target(self, trainer: Trainer) -> None:
        """Emit a warning when the requested sparsity is unreachable."""
        forecast = self._forecast_final_sparsity(trainer)
        assert isinstance(forecast, (int, float)), "Forecasted sparsity must be float or int"
        assert forecast >= 0.0, "Forecasted sparsity must be non-negative" 

        tolerance = 1e-4
        if forecast + tolerance < self.final_amount:
            log.warning(
                "Scheduled pruning target %.2f%% is unreachable with max_epochs=%s and epochs_to_ramp=%s; "
                "training would top out at %.2f%% sparsity.",
                self.final_amount * 100,
                getattr(trainer, "max_epochs", None),
                self.epochs_to_ramp,
                forecast * 100,
            )
            raise RuntimeError("Pruning schedule cannot reach target sparsity with current training configuration.")

    def _determine_pruning_amount(
        self,
        current_epoch: int,
        current_sparsity: float,
    ) -> Optional[Tuple[float, float]]:
        
        target_sparsity = (
            self._get_sparsity_target(current_epoch)
            if self.scheduled_pruning
            else self.amount
        )

        # Use small tolerance to avoid floating point issues
        tolerance = 1e-4
        if current_sparsity >= target_sparsity - tolerance:
            self._log(
                f"[Epoch {current_epoch}] | Target sparsity {target_sparsity:.2%} already reached "
                f"(current {current_sparsity:.2%}), skipping."
            )
            return None

        remaining_weights = 1.0 - current_sparsity
        if remaining_weights <= 0:
            raise RuntimeError(
                f"[Epoch {current_epoch}] | No remaining weights to prune (current sparsity: {current_sparsity:.2%})."
            )

        prune_amt = (target_sparsity - current_sparsity) / remaining_weights
        prune_amt = min(max(0.0, prune_amt), 1.0)

        return prune_amt, target_sparsity

    def _maybe_restore_parameters_from_state(self, pl_module: LightningModule) -> None:
        """Populate `_parameters_to_prune` from checkpoint metadata when available."""
        if self._parameters_to_prune is not None:
            return

        state = getattr(self, '_saved_state', None)
        if not state:
            return

        if 'parameters_to_prune_info' not in state:
            raise KeyError("Checkpoint state missing 'parameters_to_prune_info'.")

        self._log("Restoring parameters_to_prune from checkpoint...")

        restored_params = ParameterSnapshotter.restore(
            state['parameters_to_prune_info'],
            pl_module,
        )
        self._parameters_to_prune = restored_params
        self._log(f"Restored {len(restored_params)} parameters for pruning")

    def _collect_parameters_if_needed(self, pl_module: LightningModule) -> None:
        """Fallback to automatic parameter collection when user input is missing."""
        if self._parameters_to_prune is not None:
            self._non_prunable_default_parameters = []
            return

        self._log("parameters_to_prune is None, using default parameter collection...")

        (
            self._parameters_to_prune,
            self._non_prunable_default_parameters,
        ) = self._collect_default_prunable_parameters(
            pl_module,
            min_param_elements=self._min_param_elements,
            prune_bias=self._prune_bias,
        )

        if not self._parameters_to_prune:
            raise RuntimeError("Automatic parameter collection failed to find prunable parameters.")

        self._log(f"Using {len(self._parameters_to_prune)} default parameters for pruning")

    def _log_parameter_overview(self, pl_module: LightningModule) -> None:
        """Log what is and is not eligible for pruning."""
        if self._parameter_summary_logged:
            return

        from collections import defaultdict
        
        prunable = self._parameters_to_prune or []
        non_prunable = getattr(self, "_non_prunable_default_parameters", [])

        # Fallback: discover any remaining parameters when nothing recorded
        if not non_prunable:
            self._log("Non-prunable list empty, running fallback detection.", level=2)
            fallback = self._discover_untracked_parameters(pl_module, prunable)
            if fallback:
                non_prunable = fallback
                self._non_prunable_default_parameters = fallback

        def format_table(title: str, headers: Tuple[str, ...], rows: List[Tuple[object, ...]]) -> str:
            safe_rows = rows or [("(none)",) + ("",) * (len(headers) - 1)]
            widths = [len(h) for h in headers]
            string_rows = []
            for row in safe_rows:
                str_row = tuple(str(col) for col in row)
                string_rows.append(str_row)
                for idx, col in enumerate(str_row):
                    widths[idx] = max(widths[idx], len(col))

            def _fmt(line: Tuple[str, ...]) -> str:
                return "  " + " | ".join(col.ljust(widths[idx]) for idx, col in enumerate(line))

            separator = "  " + "-+-".join("-" * width for width in widths)
            lines = [f"{title}", _fmt(headers), separator]
            lines.extend(_fmt(row) for row in string_rows)
            return "\n".join(lines)

        if self.verbose >= 1:
            type_stats = defaultdict(lambda: defaultdict(lambda: {"count": 0, "params": 0}))
            for module, param_name in prunable:
                module_type = module.__class__.__name__
                tensor = getattr(module, param_name, None)
                params = tensor.numel() if isinstance(tensor, torch.Tensor) else 0
                entry = type_stats[module_type][param_name]
                entry["count"] += 1
                entry["params"] += params

            prunable_rows: List[Tuple[str, str, str, str]] = []
            for module_type in sorted(type_stats.keys()):
                for param_name in sorted(type_stats[module_type].keys()):
                    stats = type_stats[module_type][param_name]
                    prunable_rows.append(
                        (
                            module_type,
                            param_name,
                            str(stats["count"]),
                            f"{stats['params']:,}",
                        )
                    )

            prunable_rows.sort(key=lambda row: (-int(row[2]), row[0], row[1]))

            self._log(
                format_table(
                    "[Prunable Parameters]",
                    ("Module", "Param", "Count", "Elements"),
                    prunable_rows,
                ),
                level=1,
            )

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

            skipped_rows: List[Tuple[str, str, str, str, str]] = []
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
                        skipped_rows.append(
                            (
                                module_type,
                                param_name,
                                str(stats["count"]),
                                elements,
                                reason,
                            )
                        )

            self._log(
                format_table(
                    "[Non-prunable Parameters]",
                    ("Module", "Param", "Count", "Elements", "Reason"),
                    skipped_rows,
                ),
                level=1,
            )

        self._parameter_summary_logged = True

    def _restore_checkpoint_settings(self, include_monitor: bool = False) -> None:
        """Restore cached checkpoint settings, optionally resetting the monitor."""
        if not self._checkpoint_callbacks:
            return

        for i, callback in enumerate(self._checkpoint_callbacks):
            if i not in self._checkpoint_original_settings:
                raise KeyError(f"Checkpoint settings for callback index {i} were not cached.")
            original = self._checkpoint_original_settings[i]

            if include_monitor:
                callback.monitor = original["monitor"]
            callback.save_top_k = original["save_top_k"]
            callback.save_last = original["save_last"]

    def _disable_best_checkpointing(self, trainer: Trainer, reason: Optional[str] = None) -> None:
        """Temporarily disable best checkpoint saving when sparsity threshold is unmet."""
        if self.save_when_sparser_than is None or not self._checkpoint_callbacks:
            return

        if trainer.is_global_zero and reason:
            log.info(reason)

        for callback in self._checkpoint_callbacks:
            callback.save_top_k = 0

    def _enable_best_checkpointing(
        self,
        trainer: Trainer,
        *,
        reason: Optional[str] = None,
        reset_monitor: bool = False,
        should_reset_best_score: bool = False,
    ) -> None:
        """Restore checkpointing behavior and optionally reset monitors/best scores."""
        if self.save_when_sparser_than is None or not self._checkpoint_callbacks:
            return

        if trainer.is_global_zero and reason:
            log.info(reason)

        self._restore_checkpoint_settings(include_monitor=reset_monitor)

        if should_reset_best_score and not self._best_score_has_been_reset:
            for cb in self._checkpoint_callbacks:
                if hasattr(cb, 'best_model_score'):
                    cb.best_model_score = (
                        torch.tensor(float("inf")) if cb.mode == "min" else torch.tensor(float("-inf"))
                    )
            self._best_score_has_been_reset = True

    def _compute_current_sparsity(self, parameters: List[Tuple[nn.Module, str]]) -> float:
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

    def make_pruning_permanent(self, pl_module):
        """
        Permanently remove pruning masks from all pruned parameters.
        """
        if not self.should_make_pruning_permanent:
            return

        self._log("Finalizing pruning: removing masks and reparametrizations...")

        # Get only parameters that are actually pruned
        pruned_params = self._get_actually_pruned_parameters()
        
        if not pruned_params:
            self._log("No pruned parameters found (nothing to finalize).")
            return

        successful_removals = 0
        for module, param_name in pruned_params:
            # Double-check the parameter is actually pruned before trying to remove
            if pytorch_prune.is_pruned(module) and hasattr(module, f"{param_name}_mask"):
                try:
                    pytorch_prune.remove(module, param_name)
                    successful_removals += 1
                    self._log(f"Mask removed: {module.__class__.__name__}.{param_name}", level=2)
                except Exception as e:
                    raise RuntimeError(
                        f"Could not remove mask for {module.__class__.__name__}.{param_name}"
                    ) from e
            else:
                raise RuntimeError(
                    f"Expected {module.__class__.__name__}.{param_name} to be pruned with an attached mask"
                )

        self._log(f"Successfully removed {successful_removals}/{len(pruned_params)} pruning masks")

    def on_fit_start(self, trainer, pl_module) -> None:
        """
        Handle pruning state restoration after checkpoint loading is complete.
        """
        self._is_resuming = bool(trainer.ckpt_path or self._loaded_from_checkpoint)
        self._maybe_restore_parameters_from_state(pl_module)
        self._collect_parameters_if_needed(pl_module)
        self._log_parameter_overview(pl_module)

        state = getattr(self, '_saved_state', None)
        if state is not None:
            self._log(f"Restoring pruning state from checkpoint: {state}")

            # Restore the scheduled pruning configuration
            self.scheduled_pruning = state["scheduled_pruning"]

            if self.scheduled_pruning:
                # For scheduled pruning, restore the original schedule parameters
                # The `_get_sparsity_target` method will calculate the right value based on current_epoch
                self.final_amount = state["final_amount"]
                self.epochs_to_ramp = state["epochs_to_ramp"]

                checkpoint_epoch = state["current_epoch"]
                checkpoint_amount = state["current_amount"]

                self._log(
                    f"Resuming scheduled pruning at epoch {checkpoint_epoch} "
                    f"(current: {checkpoint_amount:.3f}, target: {self.final_amount:.3f}, "
                    f"ramp over {self.epochs_to_ramp} epochs)"
                )
            else:
                # Fixed pruning
                self.amount = state["amount"]

            self._latest_sparsity = state["checkpoint_sparsity"]

            # Clean up
            self._saved_state = None

    def setup(self, trainer, pl_module, stage):
        """Setup for training stage - collect default parameters if needed."""
        self._log(f"MagnitudePruner: setup called with stage={stage}")

        # For training stage, ensure we have default parameters if needed
        if str(stage) in ['TrainerFn.FITTING', 'fit'] and self._parameters_to_prune is None:
            self._log("MagnitudePruner: Setting up default parameters for training")
            self._collect_parameters_if_needed(pl_module)

    def _run_pruning(
        self,
        current_epoch: int,
        pl_module: LightningModule = None,
        *,
        force_prune_before_fit: bool = False,
    ) -> bool:
        """Override to handle scheduled pruning and safety."""
        # If pruning was already completed (e.g., masks removed at end of previous training),
        # skip all further pruning operations
        if self._pruning_completed:
            if current_epoch <= 0:  # Only log once at training start
                self._log(f"[Epoch {current_epoch}] Pruning was already completed in a previous run. Skipping.")
            return False
        
        # Normalize epoch to 0 if negative (for pre-training pruning)
        display_epoch = current_epoch
        if current_epoch < 0:
            current_epoch = 0
        
        # Check pruning toggle
        should_prune = self._apply_pruning(current_epoch) if callable(self._apply_pruning) else bool(self._apply_pruning)
        if not should_prune:
            self._log(f"Epoch {display_epoch}: skip (prune={should_prune})", level=2)
            return False

        valid_params = self._get_valid_parameters()

        current_sparsity = self._compute_current_sparsity(valid_params)
        self._latest_sparsity = current_sparsity

        schedule_epoch = (
            -1 if (force_prune_before_fit and self.scheduled_pruning) else current_epoch
        )

        plan = self._determine_pruning_amount(schedule_epoch, current_sparsity)
        if plan is None:
            return False

        prune_amt, target_sparsity = plan

        self.apply_pruning(prune_amt, pl_module, prevalidated_params=valid_params)

        # lottery ticket hypothesis
        use_lth = self._use_lottery_ticket_hypothesis(current_epoch) if callable(self._use_lottery_ticket_hypothesis) else bool(self._use_lottery_ticket_hypothesis)
        if use_lth:
            self.apply_lottery_ticket_hypothesis()

        # Refresh sparsity after pruning
        self._latest_sparsity = self._compute_current_sparsity(valid_params)

        label = "[Scheduled Pruning]" if self.scheduled_pruning else "[Fixed Pruning]"
        final_target = self.final_amount if self.scheduled_pruning else self.amount
        target_msg = (
            f"epoch target {target_sparsity:.2%}, final target {final_target:.2%}"
            if self.scheduled_pruning
            else f"target {target_sparsity:.2%}"
        )
        self._log(
            f"{label} Epoch {display_epoch:>2}: pruned {prune_amt:.2%} of remaining params "
            f"({target_msg}, achieved {self._latest_sparsity:.2%})"
        )

        return True

    def apply_pruning(
        self,
        amount: Union[int, float],
        pl_module: LightningModule = None,
        prevalidated_params: Optional[List[Tuple[nn.Module, str]]] = None,
    ) -> None:
        """Apply pruning using validated parameters and parent class logic."""
        self._invalidate_parameter_cache()

        valid_params = prevalidated_params or self._get_valid_parameters()
        if not valid_params:
            raise RuntimeError("apply_pruning called without any valid parameters to prune.")

        # Temporarily override parameters_to_prune to use validated set
        original_params = self._parameters_to_prune
        self._parameters_to_prune = valid_params

        try:
            # Use parent class apply_pruning which handles global/local logic
            super().apply_pruning(amount)
            self._log(f"Applied pruning amount {amount:.2%} to {len(valid_params)} parameters", level=2)
        except Exception as exc:
            log.error(f"Pruning failed: {exc}")
            raise
        finally:
            self._parameters_to_prune = original_params
            self._invalidate_parameter_cache()

    def on_train_start(self, trainer, pl_module) -> None:
        """Handle pruning before training starts if enabled."""
        # Detect if we're resuming from a checkpoint
        self._is_resuming = bool(trainer.ckpt_path or self._loaded_from_checkpoint)
        self._loaded_from_checkpoint = False
        trigger_pre_training = self._pruning_trigger == "pre_training"
        
        if self.save_when_sparser_than is not None:
            from pytorch_lightning.callbacks import ModelCheckpoint
            
            # Handle checkpoint callbacks
            self._checkpoint_callbacks = [
                cb for cb in trainer.callbacks if isinstance(cb, ModelCheckpoint)
            ]
            if not self._checkpoint_callbacks:
                raise RuntimeError(
                    "No ModelCheckpoint callback found while `save_when_sparser_than` is set."
                )
            else:
                for i, cb in enumerate(self._checkpoint_callbacks):
                    self._checkpoint_original_settings[i] = {
                        "monitor": getattr(cb, 'monitor', None),
                        "save_top_k": getattr(cb, 'save_top_k', 1),
                        "save_last": getattr(cb, 'save_last', None),
                    }

        if self.scheduled_pruning:
            self._warn_if_schedule_cannot_reach_target(trainer)

        if not trigger_pre_training:
            return

        tolerance = 1e-4
        target_sparsity = self.final_amount if self.scheduled_pruning else self.amount

        if self._pruning_completed:
            self._log("[Training Start] Pruning already completed. Skipping pre-training pruning.")
            return

        if (
            self._is_resuming
            and isinstance(target_sparsity, float)
            and target_sparsity is not None
            and self._latest_sparsity >= target_sparsity - tolerance
        ):
            self._log(
                "[Training Resumption] Target sparsity already reached before checkpoint. Skipping pre-training pruning."
            )
            return

        self._log("[Training Start] Applying pre-training pruning")
        self._run_pruning(0, pl_module, force_prune_before_fit=True)

    def on_train_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Run pruning at the beginning of the training epoch."""
        if self._pruning_trigger != "epoch_start":
            return

        self._run_pruning(trainer.current_epoch, pl_module)

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Run pruning at the end of the training epoch when requested."""
        if self._pruning_trigger != "epoch_end":
            return

        self._run_pruning(trainer.current_epoch, pl_module)

    def on_sanity_check_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Conditionally disable checkpointing before sanity check runs."""
        if self.save_when_sparser_than is None or not self._checkpoint_callbacks:
            return

        if self.save_when_sparser_than > 0:
            self._disable_best_checkpointing(trainer, reason="Disabling checkpointing for sanity check.")

    def on_sanity_check_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Restore checkpointing settings after sanity check."""
        if self.save_when_sparser_than is None or not self._checkpoint_callbacks:
            return
        
        self._enable_best_checkpointing(trainer, reason="Restoring checkpointing settings after sanity check.")

    def on_validation_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Conditionally disable checkpointing before validation runs."""
        if self.save_when_sparser_than is None or not self._checkpoint_callbacks:
            return

        valid_params = self._get_valid_parameters()
        current_sparsity = self._compute_current_sparsity(valid_params) if valid_params else 0.0
        self._latest_sparsity = current_sparsity
        should_save = current_sparsity >= self.save_when_sparser_than - 0.01 # 1% threshold for saving

        if not should_save:
            reason = (
                f"Epoch {trainer.current_epoch}: Sparsity {current_sparsity:.2%} < "
                f"{self.save_when_sparser_than:.2%}. Disabling checkpoint saving for this validation run."
            )
            self._disable_best_checkpointing(trainer, reason=reason)
            return

        reason = (
            f"Epoch {trainer.current_epoch}: Sparsity {current_sparsity:.2%} reached threshold "
            f"{self.save_when_sparser_than:.2%}. Re-enabling checkpoint saving."
        )
        self._enable_best_checkpointing(
            trainer,
            reason=reason,
            should_reset_best_score=not self._best_score_has_been_reset,
        )

    def on_validation_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """No longer needed for restoration, but kept for potential future use."""
        pass

    def on_train_end(self, trainer, pl_module) -> None:
        """Override to safely handle the end of training with pruning."""
        # Only make pruning permanent if explicitly asked for
        if self.should_make_pruning_permanent:
            try:
                self.make_pruning_permanent(pl_module)
                # Mark that pruning has been completed and finalized
                self._pruning_completed = True
                self._log("Pruning finalized and marked as completed.")
            except Exception as e:
                log.error(f"Error during pruning finalization: {e}")


    def get_state(self) -> dict:
        """Returns information about the current sparsity of the pruned modules."""
        valid_params = self._get_valid_parameters()
        current_sparsity = self._compute_current_sparsity(valid_params)
        return {
            "current_sparsity": current_sparsity,
            "target_sparsity": self.final_amount if self.scheduled_pruning else self.amount,
            "scheduled_pruning": self.scheduled_pruning
        }

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        """Save the pruning callback state to checkpoint."""
        valid_params = self._get_valid_parameters()
        if valid_params:
            self._latest_sparsity = self._compute_current_sparsity(valid_params)

        # Save the essential state for resuming scheduled pruning
        pruning_state = {
            "scheduled_pruning": self.scheduled_pruning,
            "current_epoch": trainer.current_epoch,
            "pruning_completed": self._pruning_completed,  # Track if pruning was finalized
            "checkpoint_sparsity": self._latest_sparsity,
        }
        
        if self.scheduled_pruning:
            pruning_state.update({
                "final_amount": self.final_amount,
                "epochs_to_ramp": self.epochs_to_ramp,
                "current_amount": self._get_sparsity_target(trainer.current_epoch),
            })
        else:
            pruning_state["amount"] = self.amount
            
        # Save parameters_to_prune information for restoration
        if self._parameters_to_prune:
            pruning_state["parameters_to_prune_info"] = ParameterSnapshotter.serialize(
                self._parameters_to_prune,
                pl_module,
            )
            
        checkpoint["magnitude_pruner_state"] = pruning_state

    def on_load_checkpoint(self, trainer, pl_module, checkpoint):
        """
        Load pruning state for training resumption.
        
        Note: State dict conversion for testing is handled by PrunedCheckpointHandler
        to avoid duplication and ensure proper coordination between callbacks.
        """
        self._log("MagnitudePruner: on_load_checkpoint called")

        # Load the internal state of the pruner
        if "magnitude_pruner_state" in checkpoint:
            pruner_state = checkpoint["magnitude_pruner_state"]

            if pruner_state is None:
                raise RuntimeError("MagnitudePruner: expected non-null magnitude_pruner_state in checkpoint")

            self._saved_state = pruner_state

            # Restore the pruning completion flag
            self._pruning_completed = self._saved_state["pruning_completed"]
            self._latest_sparsity = self._saved_state["checkpoint_sparsity"]
            self._loaded_from_checkpoint = True

            self._log(f"MagnitudePruner: Loaded pruning state: {self._saved_state}")
            self._log(f"MagnitudePruner: Pruning completed flag: {self._pruning_completed}")
