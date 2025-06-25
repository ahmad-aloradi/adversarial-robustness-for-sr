"""
BregmanPruner: Neural network pruning using Bregman learning framework.

Integrates the Bregman learning approach for sparse neural network training
with PyTorch Lightning callbacks for seamless integration.
"""
import torch
from pytorch_lightning import Callback, LightningModule, Trainer
from pytorch_lightning.utilities import rank_zero_only
from typing import List, Optional, Any

from .bregman_regularizers import get_regularizer
from .lambda_scheduler import LambdaScheduler
from src import utils

log = utils.get_pylogger(__name__)


class BregmanPruner(Callback):
    """
    BregmanPruner: Sparse neural network training using Bregman learning framework.
    Applies a Bregman proximal step after each optimizer step.
    """
    
    def __init__(
        self,
        sparse_init: bool = True,
        init_sparsity: float = 0.9,
        sparsity_threshold: float = 1e-30,
        prune_module_names: Optional[List[str]] = None,
        parameter_selection: Optional[str] = None,
        custom_parameter_fn: Optional[Any] = None,
        collect_metrics: bool = True,
        verbose: int = 1,
        lambda_scheduler: Optional[LambdaScheduler] = None,
        ):
        super().__init__()
        
        self.sparse_init = sparse_init
        self.init_sparsity = init_sparsity
        self.sparsity_threshold = sparsity_threshold
        self.prune_module_names = prune_module_names
        self.parameter_selection = parameter_selection
        self.custom_parameter_fn = custom_parameter_fn
        self.collect_metrics = collect_metrics
        self.verbose = verbose
        self.lambda_scheduler = lambda_scheduler
        
        self.selected_parameters = []
        self.pruned_params_ids = set()
        self._initialized = False
    
    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        # Configure modules for pruning based on parameter_selection strategy
        self._configure_pruning_modules(pl_module)
        
        # Run initialization only once at the start of training
        if not self._initialized:
            self._initialize_bregman_learning(pl_module)
            self._initialized = True
    
    def _configure_pruning_modules(self, pl_module: LightningModule) -> None:
        """
        Dynamically configures the modules to be pruned based on the 'parameter_selection' strategy.
        """
        # Do nothing if prune_module_names is already manually set
        if self.prune_module_names is not None:
            return
            
        if not self.parameter_selection:
            return

        if self.parameter_selection == "weights_only":
            log.info("Selecting modules with learnable parameters for Bregman pruning.")
            module_names = []
            for name, module in pl_module.named_modules():
                # Check if module has any learnable parameters
                if self._has_learnable_parameters(module) and name:  # Exclude root module
                    module_names.append(name)
            
            if module_names:
                log.info(f"Dynamically identified {len(module_names)} modules for pruning")
                if self.verbose > 1:
                    log.debug(f"Modules: {module_names}")
                self.prune_module_names = module_names
            else:
                log.warning("Strategy 'weights_only' did not find any modules with learnable parameters.")

        elif self.parameter_selection == "weights_no_bias":
            log.info("Selecting weight parameters only (excluding biases) for Bregman pruning.")
            weight_params = []
            for name, module in pl_module.named_modules():
                # Get weight parameters from any module type
                weight_params.extend(self._get_weight_parameters(module, name))
            
            if weight_params:
                log.info(f"Dynamically identified {len(weight_params)} weight parameters for pruning (excluding biases)")
                self.selected_parameters = weight_params
                self.pruned_params_ids = {id(p) for p in weight_params}
                return  # Skip normal module-based selection
            else:
                log.warning("Strategy 'weights_no_bias' did not find any weight parameters to prune.")

        elif self.parameter_selection == "comprehensive":
            log.info("Using comprehensive parameter selection for all modern architectures.")
            weight_params = []
            for name, module in pl_module.named_modules():
                # Get all trainable parameters that look like weights
                weight_params.extend(self._get_comprehensive_weight_parameters(module, name))
            
            if weight_params:
                log.info(f"Comprehensively identified {len(weight_params)} parameters for pruning")
                self.selected_parameters = weight_params
                self.pruned_params_ids = {id(p) for p in weight_params}
                return
            else:
                log.warning("Comprehensive selection found no parameters to prune.")

        elif self.custom_parameter_fn:
            log.warning("'custom_parameter_fn' is not supported in this version. Ignoring.")
        
        else:
            log.warning(
                f"Unknown 'parameter_selection' strategy: '{self.parameter_selection}'. "
                f"BregmanPruner will proceed with its default behavior (pruning all parameters)."
            )

    def _has_learnable_parameters(self, module: torch.nn.Module) -> bool:
        """Check if module has any learnable parameters."""
        return any(p.requires_grad for p in module.parameters(recurse=False))

    def _get_weight_parameters(self, module: torch.nn.Module, module_name: str) -> List[torch.Tensor]:
        """Get weight parameters (excluding biases) from a module."""
        weight_params = []
        
        # Common weight parameter names across different layer types
        weight_names = [
            'weight',  # Linear, Conv, etc.
            'weight_ih_l0', 'weight_hh_l0',  # LSTM/GRU layer 0
            'weight_ih_l1', 'weight_hh_l1',  # LSTM/GRU layer 1
            'weight_ih_l2', 'weight_hh_l2',  # LSTM/GRU layer 2
            'weight_ih_l3', 'weight_hh_l3',  # LSTM/GRU layer 3
            'in_proj_weight', 'out_proj.weight',  # MultiheadAttention
        ]
        
        for param_name in weight_names:
            if hasattr(module, param_name):
                param = getattr(module, param_name)
                if isinstance(param, torch.nn.Parameter) and param.requires_grad:
                    weight_params.append(param)
                    if self.verbose > 2:
                        log.debug(f"Found weight parameter: {module_name}.{param_name} {param.shape}")
        
        return weight_params

    def _get_comprehensive_weight_parameters(self, module: torch.nn.Module, module_name: str) -> List[torch.Tensor]:
        """Comprehensive parameter selection for modern architectures."""
        weight_params = []
        
        for param_name, param in module.named_parameters(recurse=False):
            if not param.requires_grad:
                continue
                
            # Skip bias parameters
            if 'bias' in param_name.lower():
                continue
                
            # Skip normalization parameters (they're usually small and important)
            if any(norm_term in param_name.lower() for norm_term in ['norm', 'bn', 'ln']):
                continue
                
            # Skip embedding position parameters (often critical for transformers)
            if 'pos' in param_name.lower() and 'embed' in param_name.lower():
                continue
                
            # Include everything else as potential weight parameters
            weight_params.append(param)
            if self.verbose > 2:
                log.debug(f"Comprehensive selection: {module_name}.{param_name} {param.shape}")
        
        return weight_params

    def _initialize_bregman_learning(self, pl_module: LightningModule) -> None:
        log.info("Initializing Bregman learning framework...")
        
        self._select_parameters(pl_module)
        if not self.selected_parameters:
            log.warning("No parameters selected for pruning. BregmanPruner will have no effect.")
            return

        if self.sparse_init: self._apply_sparse_initialization()
        
        # Get optimizer and use its regularizer
        optimizer = pl_module.optimizers()
        
        if hasattr(optimizer, 'param_groups') and optimizer.param_groups:
            opt_reg = optimizer.param_groups[0].get('reg')
            if opt_reg is not None:
                self.regularizer = opt_reg
                log.info(f"Using optimizer's regularizer: lambda={getattr(opt_reg, 'lamda', 'N/A')}")
                if getattr(opt_reg, 'lamda', 0) == 0:
                    log.warning(
                        "Regularizer's lambda is 0. No sparsity will be enforced by the Bregman proximal step. "
                        "Any observed sparsity is likely from initialization or parameters with zero gradients."
                    )
            else:
                raise ValueError("Optimizer has no regularizer! BregmanPruner requires a Bregman optimizer with regularizer.")
        else:
            raise ValueError("Invalid optimizer configuration!")
        
        # Initialize lambda scheduler if provided
        self._setup_lambda_scheduler(optimizer)
        
        log.info(f"Bregman learning initialized. Sparsity of pruned modules: {self._compute_sparsity():.2%}")
    
    def _setup_lambda_scheduler(self, optimizer) -> None:
        """
        Set up the lambda scheduler, handling both partial functions and pre-instantiated schedulers.
        
        Args:
            optimizer: The optimizer to use with the scheduler
        """
        if self.lambda_scheduler is None:
            return
            
        # Check if lambda_scheduler is a partial function (from _partial_: true)
        if callable(self.lambda_scheduler) and not hasattr(self.lambda_scheduler, 'step'):
            # It's a partial function, complete the instantiation with the optimizer
            try:
                self.lambda_scheduler = self.lambda_scheduler(optimizer=optimizer)
                log.info(f"Lambda scheduler instantiated with target sparsity: {self.lambda_scheduler.target_sparse}")
            except Exception as e:
                log.error(f"Failed to instantiate lambda scheduler: {e}")
                log.error(f"Scheduler type: {type(self.lambda_scheduler)}")
                # Disable scheduler on error
                self.lambda_scheduler = None
        else:
            # It's already an instantiated scheduler
            log.info(f"Lambda scheduler already instantiated with target sparsity: {self.lambda_scheduler.target_sparse}")

    def _select_parameters(self, pl_module: LightningModule) -> None:
        # Skip if parameters were already selected in _configure_pruning_modules
        if self.selected_parameters:
            return
            
        if not self.prune_module_names:
            log.warning("`prune_module_names` is not specified. Pruning all model parameters by default.")
            self.selected_parameters = list(pl_module.parameters())
        else:
            for name in self.prune_module_names:
                try:
                    module = pl_module.get_submodule(name)
                    self.selected_parameters.extend(list(module.parameters()))
                except AttributeError:
                    raise AttributeError(f"Module '{name}' not found in the LightningModule.")
        
        self.pruned_params_ids = {id(p) for p in self.selected_parameters}

    def _apply_sparse_initialization(self) -> None:
        log.info(f"Applying sparse initialization (sparsity={self.init_sparsity:.2%}) to selected modules...")
        for param in self.selected_parameters:
            if param.requires_grad:
                if self.init_sparsity == 1.0:
                    with torch.no_grad(): param.data.zero_()
                else:
                    keep_prob = 1 - self.init_sparsity
                    mask = torch.bernoulli(torch.full_like(param, keep_prob))
                    with torch.no_grad():
                        param.data *= mask
        log.info(f"Achieved initial sparsity on pruned modules: {self._compute_sparsity():.3%}")

    def on_train_batch_end(self, trainer: Trainer, pl_module: LightningModule, outputs: Any, batch: Any, batch_idx: int) -> None:
        if not self._initialized or not self.selected_parameters: 
            return
        
        # Update regularization strength if scheduler is configured
        current_lamda = getattr(self.regularizer, 'lamda', 1.0)  # Get current lambda from regularizer
        if self.lambda_scheduler is not None:
            current_lamda = self._update_regularization_strength(trainer)
        
        # Log metrics periodically
        if self.collect_metrics and (trainer.global_step % 100 == 0):
             self._log_metrics(trainer, lamda=current_lamda)

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if not self._initialized or not self.selected_parameters: return
        if self.verbose > 0:
            log.info(f"Epoch {trainer.current_epoch}: Sparsity of pruned modules = {self._compute_sparsity():.3%}")

    def _compute_sparsity(self) -> float:
        if not self.selected_parameters: return 0.0
        total_params = sum(p.numel() for p in self.selected_parameters if p.requires_grad)
        zero_params = sum((p.abs() <= self.sparsity_threshold).sum().item() for p in self.selected_parameters if p.requires_grad)
        return zero_params / max(1, total_params)
    
    def _update_regularization_strength(self, trainer: Trainer) -> float:
        current_sparsity = self._compute_sparsity()
        new_lamda = self.lambda_scheduler.step(current_sparsity)
        return new_lamda

    @rank_zero_only
    def _log_metrics(self, trainer: Trainer, lamda: float) -> None:
        sparsity = self._compute_sparsity()
        
        # Access lambda from the regularizer
        lamda = getattr(self.regularizer, 'lamda', 0)
        
        metrics_to_log = {
            "bregman/pruned_module_sparsity": sparsity,
            "bregman/lambda": lamda,
        }
        trainer.logger.log_metrics(metrics_to_log, step=trainer.global_step)
        
        log.info(f"Step {trainer.global_step}: Sparsity={sparsity:.3%}, lambda={lamda:.4f}")

    def get_sparsity_info(self) -> dict:
        """Returns information about the current sparsity of the pruned modules."""
        return {
            "current_sparsity": self._compute_sparsity(),
            "target_sparsity": getattr(self.lambda_scheduler, 'target_sparse', None) if self.lambda_scheduler else None,
            "lamda": getattr(self.regularizer, 'lamda', 0),
        }