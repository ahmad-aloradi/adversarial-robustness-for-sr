"""
PruningSelector: Unified interface for different pruning methods.

This class allows seamless switching between different pruning approaches:
- MagnitudePruner (traditional magnitude-based pruning)
- BregmanPruner (Bregman learning framework)
- Any other custom pruning methods
"""
import inspect
from typing import Union, Dict, Any, Optional
import torch
from pytorch_lightning import Callback, Trainer, LightningModule

from .prune import MagnitudePruner
from .bregman.bregman_pruner import BregmanPruner
from src import utils

log = utils.get_pylogger(__name__)


class PruningSelector(Callback):
    """
    Unified pruning interface that selects and configures different pruning methods.
    
    This class acts as a factory and wrapper for different pruning approaches,
    providing a consistent interface while allowing method-specific configurations.
    
    Parameters
    ----------
    method : str
        Pruning method to use ("magnitude", "bregman", "none")
    method_config : Dict[str, Any]
        Configuration dictionary for the selected method
    verbose : int, default=1
        Verbosity level
    """
    
    AVAILABLE_METHODS = {
        "magnitude": MagnitudePruner,
        "bregman": BregmanPruner,
        "none": None,
    }
    
    def __init__(
        self,
        method: str = "magnitude",
        method_config: Optional[Dict[str, Any]] = None,
        verbose: int = 1,
        **kwargs
    ):
        super().__init__()
        
        self.method = method
        self.method_config = method_config or {}
        self.verbose = verbose
        
        # Merge any additional kwargs into method_config
        self.method_config.update(kwargs)
        
        # Validate method
        if method not in self.AVAILABLE_METHODS:
            raise ValueError(
                f"Unknown pruning method: {method}. "
                f"Available methods: {list(self.AVAILABLE_METHODS.keys())}"
            )
        
        # Create the actual pruning callback
        self.pruning_callback = self._create_pruning_callback()
        
        if self.verbose > 0:
            if self.pruning_callback is not None:
                log.info(f"Initialized pruning method: {method}")
            else:
                log.info("No pruning method selected (method='none')")
    
    def _create_pruning_callback(self) -> Optional[Callback]:
        """Create the appropriate pruning callback based on method selection."""
        if self.method == "none":
            return None
        
        callback_class = self.AVAILABLE_METHODS[self.method]
        
        try:
            # --- START: ROBUST ARGUMENT FILTERING ---
            # Inspect the signature of the callback's constructor to find its valid arguments
            sig = inspect.signature(callback_class.__init__)
            valid_arg_keys = {p.name for p in sig.parameters.values()}

            # Prepare the full config, ensuring 'verbose' is handled correctly
            full_config = self.method_config.copy()
            if "verbose" not in full_config and "verbose" in valid_arg_keys:
                full_config["verbose"] = max(0, self.verbose - 1)

            # Filter the provided config to only include arguments accepted by the constructor
            pruner_config = {k: v for k, v in full_config.items() if k in valid_arg_keys}
            
            # Log any arguments that are being ignored, which helps debug configuration files
            ignored_args = {k:v for k,v in full_config.items() if k not in valid_arg_keys}
            if ignored_args and self.verbose > 0:
                log.warning(
                    f"Ignoring the following arguments for {callback_class.__name__} "
                    f"as they are not part of its constructor: {list(ignored_args.keys())}"
                )

            # Instantiate the callback with only the valid arguments
            callback = callback_class(**pruner_config)
            # --- END: ROBUST ARGUMENT FILTERING ---
            
            if self.verbose > 1:
                log.debug(f"Created {callback_class.__name__} with config: {pruner_config}")
            
            return callback
            
        except TypeError as e:
            log.error(f"Failed to create {callback_class.__name__} due to a configuration error: {e}")
            log.error(f"Provided config was: {self.method_config}")
            raise
        except Exception as e:
            log.error(f"An unexpected error occurred while creating {callback_class.__name__}: {e}")
            raise

    def __getattr__(self, name):
        """Delegate attribute access to the underlying pruning callback."""
        if self.pruning_callback is not None and hasattr(self.pruning_callback, name):
            return getattr(self.pruning_callback, name)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
    
    def __call__(self, *args, **kwargs):
        """Make the selector callable by delegating to the underlying callback."""
        if self.pruning_callback is not None:
            return self.pruning_callback(*args, **kwargs)
        return None
    
    def _configure_bregman_pruning_modules(self, pl_module: LightningModule):
        """
        Dynamically configures the modules to be pruned for BregmanPruner
        based on the 'parameter_selection' strategy from the config.
        This method is called from `on_fit_start`.
        """
        # Do nothing if prune_module_names is already manually set
        if getattr(self.pruning_callback, "prune_module_names", None) is not None:
            return
            
        selection_strategy = self.method_config.get("parameter_selection")
        if not selection_strategy:
            return

        module_names = []
        if selection_strategy == "weights_only":
            log.info("Selecting 'weights_only' modules for Bregman pruning (Linear and Conv layers).")
            for name, module in pl_module.named_modules():
                if isinstance(module, (torch.nn.Linear, torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d)):
                    if name:  # Exclude the root module itself
                        module_names.append(name)
            
            if module_names:
                log.info(f"Dynamically identified modules for pruning: {module_names}")
                self.pruning_callback.prune_module_names = module_names
            else:
                log.warning("Strategy 'weights_only' did not find any Linear or Conv modules to prune.")

        elif self.method_config.get("custom_parameter_fn"):
            log.warning("'custom_parameter_fn' is not supported in this version. Ignoring.")
        
        else:
             log.warning(
                 f"Unknown 'parameter_selection' strategy: '{selection_strategy}'. "
                 f"BregmanPruner will proceed with its default behavior (pruning all parameters)."
            )

    # Delegate all callback methods to the underlying pruning callback
    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: Optional[str] = None):
        if self.pruning_callback is not None:
            return self.pruning_callback.setup(trainer, pl_module, stage)
    
    def teardown(self, trainer: Trainer, pl_module: LightningModule, stage: Optional[str] = None):
        if self.pruning_callback is not None:
            return self.pruning_callback.teardown(trainer, pl_module, stage)
    
    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule):
        if self.pruning_callback is not None:
            if self.method == "bregman":
                self._configure_bregman_pruning_modules(pl_module)
            return self.pruning_callback.on_fit_start(trainer, pl_module)
    
    def on_fit_end(self, trainer: Trainer, pl_module: LightningModule):
        if self.pruning_callback is not None:
            return self.pruning_callback.on_fit_end(trainer, pl_module)
    
    def on_train_start(self, trainer: Trainer, pl_module: LightningModule):
        if self.pruning_callback is not None:
            return self.pruning_callback.on_train_start(trainer, pl_module)
    
    def on_train_end(self, trainer: Trainer, pl_module: LightningModule):
        if self.pruning_callback is not None:
            return self.pruning_callback.on_train_end(trainer, pl_module)
    
    def on_train_epoch_start(self, trainer: Trainer, pl_module: LightningModule):
        if self.pruning_callback is not None:
            return self.pruning_callback.on_train_epoch_start(trainer, pl_module)
    
    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule):
        if self.pruning_callback is not None:
            return self.pruning_callback.on_train_epoch_end(trainer, pl_module)
    
    def on_train_batch_start(self, trainer: Trainer, pl_module: LightningModule, batch: Any, batch_idx: int):
        if self.pruning_callback is not None:
            return self.pruning_callback.on_train_batch_start(trainer, pl_module, batch, batch_idx)
    
    def on_train_batch_end(self, trainer: Trainer, pl_module: LightningModule, outputs: Any, batch: Any, batch_idx: int):
        if self.pruning_callback is not None:
            return self.pruning_callback.on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)
    
    def on_validation_start(self, trainer: Trainer, pl_module: LightningModule):
        if self.pruning_callback is not None:
            return self.pruning_callback.on_validation_start(trainer, pl_module)
    
    def on_validation_end(self, trainer: Trainer, pl_module: LightningModule):
        if self.pruning_callback is not None:
            return self.pruning_callback.on_validation_end(trainer, pl_module)
    
    def state_dict(self):
        """Get state dict for checkpointing."""
        state = {
            "method": self.method,
            "method_config": self.method_config,
            "verbose": self.verbose,
        }
        
        if self.pruning_callback is not None and hasattr(self.pruning_callback, 'state_dict'):
            state["callback_state"] = self.pruning_callback.state_dict()
        
        return state
    
    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load state dict from checkpoint."""
        self.method = state_dict.get("method", self.method)
        self.method_config = state_dict.get("method_config", self.method_config)
        self.verbose = state_dict.get("verbose", self.verbose)
        
        # Recreate callback if method changed
        if self.pruning_callback is None or self.method != state_dict.get("method"):
            self.pruning_callback = self._create_pruning_callback()
        
        # Load callback state if available
        if (self.pruning_callback is not None and 
            hasattr(self.pruning_callback, 'load_state_dict') and 
            "callback_state" in state_dict):
            self.pruning_callback.load_state_dict(state_dict["callback_state"])
    
    def get_pruning_info(self) -> Dict[str, Any]:
        """Get information about the current pruning configuration and state."""
        info = {
            "method": self.method,
            "method_config": self.method_config,
            "has_callback": self.pruning_callback is not None,
        }
        
        # Add method-specific information
        if self.pruning_callback is not None:
            if hasattr(self.pruning_callback, 'get_sparsity_info'):
                info.update(self.pruning_callback.get_sparsity_info())
            elif hasattr(self.pruning_callback, '_compute_sparsity'): # For BregmanPruner
                info['current_sparsity'] = self.pruning_callback._compute_sparsity()
        
        return info
    
    @classmethod
    def from_config(cls, config: Dict[str, Any], **kwargs) -> "PruningSelector":
        """Create PruningSelector from configuration dictionary."""
        method = config.get("method", "magnitude")
        method_config = config.get("method_config", {})
        
        # Allow kwargs to override config values
        method_config.update(kwargs)
        
        return cls(
            method=method,
            method_config=method_config,
            verbose=config.get("verbose", 1)
        )


def create_pruning_callback(
    method: str = "magnitude",
    **method_config
) -> Optional[Callback]:
    """
    Convenience function to create a pruning callback.
    """
    selector = PruningSelector(method=method, method_config=method_config)
    return selector.pruning_callback


# """
# PruningSelector: Unified interface for different pruning methods.

# This class allows seamless switching between different pruning approaches:
# - MagnitudePruner (traditional magnitude-based pruning)
# - BregmanPruner (Bregman learning framework)
# - Any other custom pruning methods
# """
# from typing import Union, Dict, Any, Optional
# from pytorch_lightning import Callback

# from .prune import MagnitudePruner 
# from .bregman.bregman_pruner import BregmanPruner
# from src import utils

# log = utils.get_pylogger(__name__)


# class PruningSelector(Callback):
#     """
#     Unified pruning interface that selects and configures different pruning methods.
    
#     This class acts as a factory and wrapper for different pruning approaches,
#     providing a consistent interface while allowing method-specific configurations.
    
#     Parameters
#     ----------
#     method : str
#         Pruning method to use ("magnitude", "bregman", "none")
#     method_config : Dict[str, Any]
#         Configuration dictionary for the selected method
#     verbose : int, default=1
#         Verbosity level
    
#     Examples
#     --------
#     # Traditional magnitude-based pruning
#     pruning_selector = PruningSelector(
#         method="magnitude",
#         method_config={
#             "amount": 0.3,
#             "scheduled_pruning": True,
#             "initial_amount": 0.1,
#             "final_amount": 0.5,
#             "epochs_to_ramp": 20
#         }
#     )
    
#     # Bregman learning approach
#     pruning_selector = PruningSelector(
#         method="bregman",
#         method_config={
#             "optimizer_name": "linbreg",
#             "regularizer_name": "l1",
#             "lr": 1e-3,
#             "lamda": 1e-2,
#             "sparse_init": True,
#             "init_sparsity": 0.8
#         }
#     )
#     """
    
#     AVAILABLE_METHODS = {
#         "magnitude": MagnitudePruner,
#         "bregman": BregmanPruner,
#         "none": None,
#     }
    
#     def __init__(
#         self,
#         method: str = "magnitude",
#         method_config: Optional[Dict[str, Any]] = None,
#         verbose: int = 1,
#         **kwargs
#     ):
#         super().__init__()
        
#         self.method = method
#         self.method_config = method_config or {}
#         self.verbose = verbose
        
#         # Merge any additional kwargs into method_config
#         self.method_config.update(kwargs)
        
#         # Validate method
#         if method not in self.AVAILABLE_METHODS:
#             raise ValueError(
#                 f"Unknown pruning method: {method}. "
#                 f"Available methods: {list(self.AVAILABLE_METHODS.keys())}"
#             )
        
#         # Create the actual pruning callback
#         self.pruning_callback = self._create_pruning_callback()
        
#         if self.verbose > 0:
#             if self.pruning_callback is not None:
#                 log.info(f"Initialized pruning method: {method}")
#             else:
#                 log.info("No pruning method selected (method='none')")
    
#     def _create_pruning_callback(self) -> Optional[Callback]:
#         """Create the appropriate pruning callback based on method selection."""
#         if self.method == "none":
#             return None
        
#         callback_class = self.AVAILABLE_METHODS[self.method]
        
#         try:
#             # Add verbose parameter if not already specified
#             if "verbose" not in self.method_config:
#                 self.method_config["verbose"] = max(0, self.verbose - 1)
            
#             callback = callback_class(**self.method_config)
            
#             if self.verbose > 1:
#                 log.debug(f"Created {callback_class.__name__} with config: {self.method_config}")
            
#             return callback
            
#         except Exception as e:
#             log.error(f"Failed to create {callback_class.__name__}: {e}")
#             log.error(f"Config was: {self.method_config}")
#             raise
    
#     def __getattr__(self, name):
#         """Delegate attribute access to the underlying pruning callback."""
#         if self.pruning_callback is not None and hasattr(self.pruning_callback, name):
#             return getattr(self.pruning_callback, name)
#         raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
    
#     def __call__(self, *args, **kwargs):
#         """Make the selector callable by delegating to the underlying callback."""
#         if self.pruning_callback is not None:
#             return self.pruning_callback(*args, **kwargs)
#         return None
    
#     # Delegate all callback methods to the underlying pruning callback
#     def setup(self, trainer, pl_module, stage=None):
#         if self.pruning_callback is not None:
#             return self.pruning_callback.setup(trainer, pl_module, stage)
    
#     def teardown(self, trainer, pl_module, stage=None):
#         if self.pruning_callback is not None:
#             return self.pruning_callback.teardown(trainer, pl_module, stage)
    
#     def on_fit_start(self, trainer, pl_module):
#         if self.pruning_callback is not None:
#             return self.pruning_callback.on_fit_start(trainer, pl_module)
    
#     def on_fit_end(self, trainer, pl_module):
#         if self.pruning_callback is not None:
#             return self.pruning_callback.on_fit_end(trainer, pl_module)
    
#     def on_train_start(self, trainer, pl_module):
#         if self.pruning_callback is not None:
#             return self.pruning_callback.on_train_start(trainer, pl_module)
    
#     def on_train_end(self, trainer, pl_module):
#         if self.pruning_callback is not None:
#             return self.pruning_callback.on_train_end(trainer, pl_module)
    
#     def on_train_epoch_start(self, trainer, pl_module):
#         if self.pruning_callback is not None:
#             return self.pruning_callback.on_train_epoch_start(trainer, pl_module)
    
#     def on_train_epoch_end(self, trainer, pl_module):
#         if self.pruning_callback is not None:
#             return self.pruning_callback.on_train_epoch_end(trainer, pl_module)
    
#     def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
#         if self.pruning_callback is not None:
#             return self.pruning_callback.on_train_batch_start(trainer, pl_module, batch, batch_idx)
    
#     def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
#         if self.pruning_callback is not None:
#             return self.pruning_callback.on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)
    
#     def on_validation_start(self, trainer, pl_module):
#         if self.pruning_callback is not None:
#             return self.pruning_callback.on_validation_start(trainer, pl_module)
    
#     def on_validation_end(self, trainer, pl_module):
#         if self.pruning_callback is not None:
#             return self.pruning_callback.on_validation_end(trainer, pl_module)
    
#     def state_dict(self):
#         """Get state dict for checkpointing."""
#         state = {
#             "method": self.method,
#             "method_config": self.method_config,
#             "verbose": self.verbose,
#         }
        
#         if self.pruning_callback is not None and hasattr(self.pruning_callback, 'state_dict'):
#             state["callback_state"] = self.pruning_callback.state_dict()
        
#         return state
    
#     def load_state_dict(self, state_dict):
#         """Load state dict from checkpoint."""
#         self.method = state_dict.get("method", self.method)
#         self.method_config = state_dict.get("method_config", self.method_config)
#         self.verbose = state_dict.get("verbose", self.verbose)
        
#         # Recreate callback if method changed
#         if self.pruning_callback is None or self.method != state_dict.get("method"):
#             self.pruning_callback = self._create_pruning_callback()
        
#         # Load callback state if available
#         if (self.pruning_callback is not None and 
#             hasattr(self.pruning_callback, 'load_state_dict') and 
#             "callback_state" in state_dict):
#             self.pruning_callback.load_state_dict(state_dict["callback_state"])
    
#     def get_pruning_info(self) -> Dict[str, Any]:
#         """Get information about the current pruning configuration and state."""
#         info = {
#             "method": self.method,
#             "method_config": self.method_config,
#             "has_callback": self.pruning_callback is not None,
#         }
        
#         # Add method-specific information
#         if self.pruning_callback is not None:
#             if hasattr(self.pruning_callback, 'get_sparsity_info'):
#                 info.update(self.pruning_callback.get_sparsity_info())
#             elif hasattr(self.pruning_callback, 'metrics'):
#                 info["metrics_available"] = bool(self.pruning_callback.metrics)
        
#         return info
    
#     @classmethod
#     def from_config(cls, config: Dict[str, Any], **kwargs) -> "PruningSelector":
#         """Create PruningSelector from configuration dictionary.
        
#         Parameters
#         ----------
#         config : Dict[str, Any]
#             Configuration dictionary with 'method' key and optional method-specific configs
#         **kwargs
#             Additional arguments to override config values
        
#         Returns
#         -------
#         PruningSelector
#             Configured pruning selector instance
#         """
#         method = config.get("method", "magnitude")
#         method_config = config.get("method_config", {})
        
#         # Allow kwargs to override config values
#         method_config.update(kwargs)
        
#         return cls(
#             method=method,
#             method_config=method_config,
#             verbose=config.get("verbose", 1)
#         )


# def create_pruning_callback(
#     method: str = "magnitude",
#     **method_config
# ) -> Optional[Callback]:
#     """
#     Convenience function to create a pruning callback.
    
#     Parameters
#     ----------
#     method : str
#         Pruning method ("magnitude", "bregman", "none")
#     **method_config
#         Method-specific configuration parameters
    
#     Returns
#     -------
#     Optional[Callback]
#         Configured pruning callback or None if method="none"
    
#     Examples
#     --------
#     # Create SafeModelPruning callback
#     pruner = create_pruning_callback(
#         method="magnitude",
#         amount=0.3,
#         scheduled_pruning=True
#     )
    
#     # Create BregmanPruner callback
#     pruner = create_pruning_callback(
#         method="bregman",
#         regularizer_name="l1",
#         lamda=1e-2
#     )
#     """
#     selector = PruningSelector(method=method, method_config=method_config)
#     return selector.pruning_callback
