import torch
import torch.nn as nn
import torch.nn.utils.prune as pytorch_prune
from pytorch_lightning import Callback, Trainer, LightningModule
from typing import Dict, Any
from collections import OrderedDict
from src.utils import get_pylogger

logger = get_pylogger(__name__)


class PrunedCheckpointHandler(Callback):
    """
    Ensures compatibility between standard models and pruned checkpoints.
    Handles device safety and optimizer state matching.
    """

    def __init__(self, verbose: int = 1):
        self.verbose = verbose
        self._patched_models = set()

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        self._patch_model_load_state_dict(pl_module)
        if stage == 'fit' and trainer.ckpt_path:
            if self._is_pruned_checkpoint(trainer.ckpt_path):
                self.reconstruct_pruning_structure(
                    pl_module,
                    trainer.ckpt_path,
                    verbose=self.verbose
                )

    def _is_pruned_checkpoint(self, path: str) -> bool:
        try:
            ckpt = torch.load(path, map_location='cpu')
            state_dict = ckpt.get('state_dict', {})
            return any(k.endswith('_orig') for k in state_dict.keys())
        except Exception:
            return False

    @staticmethod
    def reconstruct_pruning_structure(
        pl_module: LightningModule,
        checkpoint_source: Any,
        verbose: int = 0
    ) -> None:
        if isinstance(checkpoint_source, str):
            try:
                ckpt = torch.load(checkpoint_source, map_location='cpu')
                state_dict = ckpt.get('state_dict', {})
            except Exception:
                return
        elif isinstance(checkpoint_source, dict):
            state_dict = checkpoint_source
        else:
            return

        pruned_keys = [k for k in state_dict.keys() if k.endswith('_orig')]
        if verbose and pruned_keys:
            logger.info(f"PrunedCheckpointHandler: Found {len(pruned_keys)} pruned parameters in checkpoint.")

        count = 0
        for key in pruned_keys:
            param_path = key[:-5]
            parts = param_path.split('.')
            module = pl_module
            param_name = parts[-1]
            try:
                for part in parts[:-1]:
                    module = getattr(module, part)
                if hasattr(module, param_name) and not hasattr(module, f"{param_name}_mask"):
                    pytorch_prune.identity(module, param_name)
                    PrunedCheckpointHandler._enforce_parameter_order(module, param_name)
                    count += 1
            except AttributeError:
                pass
        
        if verbose and count > 0:
            logger.info(f"PrunedCheckpointHandler: Reconstructed pruning structure for {count} parameters.")

    @staticmethod
    def _enforce_parameter_order(module: nn.Module, param_name: str):
        params = module._parameters
        orig_key = f"{param_name}_orig"
        if 'bias' in params and orig_key in params:
            keys = list(params.keys())
            if keys.index('bias') < keys.index(orig_key):
                new_params = OrderedDict()
                new_params[orig_key] = params[orig_key]
                for k, v in params.items():
                    if k != orig_key:
                        new_params[k] = v
                module._parameters = new_params

    def _patch_model_load_state_dict(self, pl_module: LightningModule):
        if id(pl_module) in self._patched_models:
            return
        original_load = pl_module.load_state_dict
        
        def patched_load(state_dict, strict=True):
            state_dict = dict(state_dict)
            keys = list(state_dict.keys())
            for key in keys:
                if key.endswith('_orig'):
                    base = key[:-5]
                    mask_key = base + '_mask'
                    parts = base.split('.')
                    module = pl_module
                    param_name = parts[-1]
                    is_model_pruned_at_loc = False
                    try:
                        for part in parts[:-1]:
                            module = getattr(module, part)
                        if hasattr(module, param_name + "_orig"):
                            is_model_pruned_at_loc = True
                    except AttributeError:
                        pass 
                    if not is_model_pruned_at_loc and mask_key in state_dict:
                        if self.verbose:
                            logger.info(f"Auto-fusing {base} for clean model loading.")
                        w_orig = state_dict.pop(key)
                        mask = state_dict.pop(mask_key)
                        state_dict[base] = w_orig * mask
            return original_load(state_dict, strict)
            
        pl_module.load_state_dict = patched_load
        self._patched_models.add(id(pl_module))
