import torch
import torch.nn as nn
import torch.nn.utils.prune as pytorch_prune
from typing import Dict, Any
import os
from pytorch_lightning import LightningModule, Trainer, Callback

from src import utils

log = utils.get_pylogger(__name__)


class PrunedCheckpointHandler(Callback):
    """
    Handles checkpoint compatibility issues when loading pruned models.
    
    This callback ensures that:
    1. For training resumption, the model structure is prepared to accept a pruned checkpoint,
       and optimizer states are cleared.
    2. For testing, the pruned checkpoint's state_dict is converted to a standard format
       before being loaded into the model.
    3. Automatically patches the model's load_state_dict method to handle pruned checkpoints.
    """
    
    def __init__(self, verbose: int = 1):
        self.verbose = verbose
        self._pruning_callback = None  # Will be set to reference MagnitudePruner if present
        self._patched_models = set()  # Track which models we've patched

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        """
        Prepares the model for training resumption from a pruned checkpoint.
        Also patches the model's load_state_dict method for automatic pruned checkpoint handling.
        """
        # Find and store reference to MagnitudePruner if present
        self._find_pruning_callback(trainer)
        
        # CRITICAL FIX: Patch the model's load_state_dict method to handle pruned checkpoints
        # This works for ALL scenarios including testing where callbacks are bypassed
        self._patch_model_load_state_dict(pl_module)
        
        # Only act when resuming a training run ('fit' or 'validate' stages)
        if str(stage) not in ['fit', 'validate'] or not trainer.ckpt_path:
            return

        if self._check_checkpoint_for_pruning(trainer.ckpt_path):
            if self.verbose > 0:
                log.info("PrunedCheckpointHandler: Preparing model structure for training resumption from pruned checkpoint.")
            self._prepare_model_for_pruned_checkpoint(pl_module, trainer.ckpt_path)

    def on_load_checkpoint(self, trainer: Trainer, pl_module: LightningModule, checkpoint: Dict[str, Any]) -> None:
        """
        Modifies the checkpoint in-place before it's loaded into the model.
        This is the main hook for ensuring compatibility.
        """
        state_dict = checkpoint.get("state_dict", {})
        if not state_dict:
            return

        has_pruning_artifacts = any(key.endswith('_orig') for key in state_dict.keys())
        if not has_pruning_artifacts:
            return

        if self.verbose > 0:
            log.info("PrunedCheckpointHandler: Detected pruned checkpoint. Modifying for compatibility.")

        # Check if we're in testing mode with multiple detection methods
        has_magnitude_pruner = any(
            type(cb).__name__ == 'MagnitudePruner' for cb in trainer.callbacks
        )
        
        # More robust testing scenario detection
        is_testing_scenario = (
            trainer.testing or 
            (hasattr(trainer.state, 'fn') and hasattr(trainer.state.fn, 'name') and trainer.state.fn.name == 'test') or
            not has_magnitude_pruner  # If no MagnitudePruner, we're likely testing
        )
        
        # Also check the stage if available
        if hasattr(trainer, '_current_stage'):
            is_testing_scenario = is_testing_scenario or trainer._current_stage == 'test'
            
        if self.verbose > 0:
            log.info(f"Testing scenario detected: {is_testing_scenario}")
            log.info(f"  trainer.testing: {trainer.testing}")
            log.info(f"  has_magnitude_pruner: {has_magnitude_pruner}")
            if hasattr(trainer.state, 'fn') and hasattr(trainer.state.fn, 'name'):
                log.info(f"  trainer.state.fn.name: {trainer.state.fn.name}")

        # Scenario 1: Testing. The model is clean (pruning is permanent).
        # We must convert the state_dict from pruned format to standard format.
        if is_testing_scenario:
            if self.verbose > 0:
                log.info("Testing mode: Converting state_dict from pruned to standard format.")
            self._convert_pruned_state_dict(state_dict)
        
        # Scenario 2: Resuming training. The model has been prepared by setup().
        # We must clear optimizer states to prevent shape mismatches.
        else:
            if "optimizer_states" in checkpoint and checkpoint["optimizer_states"]:
                if self.verbose > 0:
                    log.warning("Training resumption: Clearing optimizer states to prevent shape mismatches.")
                checkpoint["optimizer_states"] = []

    def _convert_pruned_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """
        Converts a state_dict with pruning artifacts (_orig, _mask)
        to a standard state_dict in-place.
        """
        orig_keys = [key for key in state_dict.keys() if key.endswith('_orig')]
        if not orig_keys:
            return

        converted_params = 0
        for orig_key in orig_keys:
            base_key = orig_key[:-5]  # Remove '_orig'
            mask_key = base_key + '_mask'
            
            if mask_key in state_dict:
                orig_param = state_dict.pop(orig_key)
                mask = state_dict.pop(mask_key)
                
                # Apply the mask to get the final pruned parameter.
                state_dict[base_key] = orig_param * mask
                converted_params += 1
                if self.verbose > 1:
                    log.debug(f"Converted {orig_key} -> {base_key}")
        
        if self.verbose > 0:
            log.info(f"Successfully converted {converted_params} pruned parameters to regular format.")

    def _check_checkpoint_for_pruning(self, ckpt_path: str) -> bool:
        """Check if a checkpoint contains pruning artifacts."""
        if not ckpt_path or not os.path.exists(ckpt_path):
            return False
        try:
            checkpoint_data = torch.load(ckpt_path, map_location='cpu')
            state_dict = checkpoint_data.get('state_dict', {})
            return any(key.endswith('_orig') for key in state_dict.keys())
        except Exception as e:
            log.warning(f"Could not analyze checkpoint {ckpt_path}: {e}")
            return False

    def _find_pruning_callback(self, trainer: Trainer) -> None:
        """Find and store reference to MagnitudePruner callback if present."""
        from .prune import MagnitudePruner
        
        for callback in trainer.callbacks:
            if isinstance(callback, MagnitudePruner):
                self._pruning_callback = callback
                if self.verbose > 1:
                    log.debug("PrunedCheckpointHandler: Found MagnitudePruner callback for coordination")
                break


    def _model_has_pruning_artifacts(self, pl_module: LightningModule) -> bool:
        """Check if the model currently has pruning artifacts."""
        for name, module in pl_module.named_modules():
            for param_name in list(module._parameters.keys()):
                if param_name.endswith('_orig'):
                    return True
        return False

    def _convert_model_pruning_to_permanent(self, pl_module: LightningModule) -> None:
        """
        Convert a model with pruning artifacts to permanent pruned format.
        
        This finds all parameters with _orig and _mask, applies the mask,
        and replaces the parameter with the final pruned version.
        """
        converted_params = 0
        
        for name, module in pl_module.named_modules():
            # Get all parameters that have _orig suffix
            orig_params = [p for p in module._parameters.keys() if p.endswith('_orig')]
            
            for orig_param_name in orig_params:
                base_param_name = orig_param_name[:-5]  # Remove '_orig'
                mask_param_name = base_param_name + '_mask'
                
                # Check if we have both _orig and _mask
                if (hasattr(module, orig_param_name) and 
                    hasattr(module, mask_param_name)):
                    
                    # Get the original parameter and mask
                    orig_param = getattr(module, orig_param_name)
                    
                    # Get mask - it might be in _buffers instead of _parameters
                    mask = None
                    if hasattr(module, mask_param_name):
                        mask = getattr(module, mask_param_name)
                    
                    if mask is not None:
                        # Apply mask to get final pruned parameter
                        final_param = orig_param * mask
                        
                        # Remove the _orig parameter and _mask
                        delattr(module, orig_param_name)
                        if hasattr(module, mask_param_name):
                            delattr(module, mask_param_name)
                        
                        # Set the final parameter
                        setattr(module, base_param_name, nn.Parameter(final_param.detach()))
                        
                        converted_params += 1
                        if self.verbose > 1:
                            log.debug(f"Converted {name}.{orig_param_name} -> {name}.{base_param_name}")

        if self.verbose > 0:
            log.info(f"Successfully converted {converted_params} pruned parameters to permanent format.")

    def _patch_model_load_state_dict(self, pl_module: LightningModule) -> None:
        """
        Patch the model's load_state_dict method to automatically handle pruned checkpoints.
        
        This is the key fix that makes testing work, because PyTorch Lightning bypasses
        callbacks during testing and loads checkpoints directly into the model.
        """
        model_id = id(pl_module)
        if model_id in self._patched_models:
            return  # Already patched
        
        # Store original method
        original_load_state_dict = pl_module.load_state_dict
        
        def patched_load_state_dict(state_dict, strict=True):
            # Check if the state_dict has pruning artifacts
            has_pruning_artifacts = any(key.endswith('_orig') for key in state_dict.keys())
            
            if has_pruning_artifacts:
                if self.verbose > 0:
                    log.info("PrunedCheckpointHandler: Auto-converting pruned state_dict during load_state_dict")
                
                # Create a copy to avoid modifying the original
                converted_state_dict = dict(state_dict)
                self._convert_pruned_state_dict(converted_state_dict)
                
                # Use the converted state_dict
                return original_load_state_dict(converted_state_dict, strict=strict)
            else:
                # No pruning artifacts, use original method
                return original_load_state_dict(state_dict, strict=strict)
        
        # Replace the method
        pl_module.load_state_dict = patched_load_state_dict
        self._patched_models.add(model_id)
        
        if self.verbose > 1:
            log.debug(f"PrunedCheckpointHandler: Patched model load_state_dict for automatic pruned checkpoint handling")

    def _prepare_model_for_pruned_checkpoint(self, pl_module: LightningModule, ckpt_path: str) -> None:
        """Applies 0% pruning to model parameters to match checkpoint structure."""
        checkpoint_data = torch.load(ckpt_path, map_location='cpu')
        state_dict = checkpoint_data.get('state_dict', {})
        mask_keys = [key for key in state_dict.keys() if key.endswith('_mask')]

        if not mask_keys:
            return

        prepared_params = 0
        for mask_key in mask_keys:
            param_path = mask_key[:-5]
            parts = param_path.split('.')
            param_name = parts[-1]
            module = pl_module
            try:
                for part in parts[:-1]:
                    module = getattr(module, part)
                if hasattr(module, param_name) and not hasattr(module, f"{param_name}_mask"):
                    pytorch_prune.l1_unstructured(module, name=param_name, amount=0.0)
                    prepared_params += 1
                    if self.verbose > 1:
                        log.debug(f"Prepared {param_path} for pruned checkpoint loading")
            except AttributeError as e:
                if self.verbose > 1:
                    log.debug(f"Could not prepare {param_path}: {e}")
                continue
        
        if self.verbose > 0:
            log.info(f"Prepared {prepared_params} parameters for pruned checkpoint loading")
