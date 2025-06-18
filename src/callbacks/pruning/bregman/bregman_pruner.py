"""
BregmanPruner: Neural network pruning using Bregman learning framework.

Integrates the Bregman learning approach for sparse neural network training
with PyTorch Lightning callbacks for seamless integration.
"""
import torch
import torch.nn as nn
from pytorch_lightning import Callback, LightningModule, Trainer
from pytorch_lightning.utilities import rank_zero_only
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from collections import defaultdict
import warnings

from .bregman_regularizers import get_regularizer, BregmanRegularizer, RegNone
from .bregman_optimizers import get_bregman_optimizer, LinBreg
from src import utils

log = utils.get_pylogger(__name__)


class BregmanPruner(Callback):
    """
    BregmanPruner: Sparse neural network training using Bregman learning framework.
    
    This class implements the inverse scale space training algorithm proposed in
    "A Bregman Learning Framework for Sparse Neural Networks" by Bungert et al.
    
    The method starts with sparse initialization and gradually adds relevant
    parameters during training using linearized Bregman iterations.
    
    Parameters
    ----------
    optimizer_name : str, default="linbreg"
        Bregman optimizer type ("linbreg", "adabreg", "proxsgd")
    regularizer_name : str, default="l1"
        Regularizer type ("l1", "l1_l2", "soft_bernoulli", etc.)
    lr : float, default=1e-3
        Learning rate for Bregman optimizer
    delta : float, default=1.0
        Bregman parameter (elastic net parameter)
    lamda : float, default=1e-3
        Regularization strength
    momentum : float, default=0.0
        Momentum parameter (for LinBreg)
    betas : tuple, default=(0.9, 0.999)
        Adam-style momentum parameters (for AdaBreg)
    eps : float, default=1e-8
        Numerical stability parameter (for AdaBreg)
    sparse_init : bool, default=True
        Whether to use sparse initialization
    init_sparsity : float, default=0.9
        Initial sparsity level for sparse initialization
    target_sparsity : Optional[float], default=None
        Target sparsity level (for dynamic regularization)
    parameter_selection : str, default="all"
        Which parameters to apply Bregman learning to ("all", "weights_only", "custom")
    custom_parameter_fn : Optional[Callable], default=None
        Custom function to select parameters
    collect_metrics : bool, default=True
        Whether to collect sparsity and regularization metrics
    verbose : int, default=1
        Verbosity level (0=silent, 1=basic, 2=detailed)
    """
    
    def __init__(
        self,
        optimizer_name: str = "linbreg",
        regularizer_name: str = "l1",
        lr: float = 1e-3,
        delta: float = 1.0,
        lamda: float = 1e-3,
        momentum: float = 0.0,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        sparse_init: bool = True,
        init_sparsity: float = 0.9,
        target_sparsity: Optional[float] = None,
        parameter_selection: str = "all",
        custom_parameter_fn: Optional[Callable] = None,
        collect_metrics: bool = True,
        verbose: int = 1,
        **kwargs
    ):
        super().__init__()
        
        # Store configuration
        self.optimizer_name = optimizer_name
        self.regularizer_name = regularizer_name
        self.lr = lr
        self.delta = delta
        self.lamda = lamda
        self.momentum = momentum
        self.betas = betas
        self.eps = eps
        self.sparse_init = sparse_init
        self.init_sparsity = init_sparsity
        self.target_sparsity = target_sparsity
        self.parameter_selection = parameter_selection
        self.custom_parameter_fn = custom_parameter_fn
        self.collect_metrics = collect_metrics
        self.verbose = verbose
        
        # Initialize state
        self.bregman_optimizer = None
        self.regularizer = None
        self.original_optimizer = None
        self.selected_parameters = []
        self.metrics = defaultdict(list) if collect_metrics else None
        self._initialized = False
        
        # Validate parameters
        self._validate_config()
    
    def _validate_config(self):
        """Validate configuration parameters."""
        if not 0 < self.lr <= 1:
            warnings.warn(f"Learning rate {self.lr} might be too large for Bregman learning")
        
        if not 0 < self.delta <= 10:
            warnings.warn(f"Delta parameter {self.delta} outside typical range [0.1, 10]")
        
        if not 0 <= self.init_sparsity < 1:
            raise ValueError(f"Initial sparsity {self.init_sparsity} must be in [0, 1)")
        
        if self.target_sparsity is not None and not 0 <= self.target_sparsity < 1:
            raise ValueError(f"Target sparsity {self.target_sparsity} must be in [0, 1)")
        
        if self.parameter_selection not in ["all", "weights_only", "custom"]:
            raise ValueError(f"Invalid parameter_selection: {self.parameter_selection}")
        
        if self.parameter_selection == "custom" and self.custom_parameter_fn is None:
            raise ValueError("custom_parameter_fn must be provided when parameter_selection='custom'")
    
    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        """Setup Bregman learning components."""
        if stage == "fit" and not self._initialized:
            self._initialize_bregman_learning(pl_module)
            self._initialized = True
    
    def _initialize_bregman_learning(self, pl_module: LightningModule) -> None:
        """Initialize Bregman learning components."""
        if self.verbose > 0:
            log.info("Initializing Bregman learning framework...")
        
        # 1. Select parameters for Bregman learning
        self._select_parameters(pl_module)
        
        # 2. Apply sparse initialization if requested
        if self.sparse_init:
            self._apply_sparse_initialization()
        
        # 3. Create regularizer
        self._create_regularizer()
        
        # 4. Replace optimizer with Bregman optimizer
        self._setup_bregman_optimizer(pl_module)
        
        if self.verbose > 0:
            total_params = sum(p.numel() for p in self.selected_parameters)
            sparsity = self._compute_sparsity()
            log.info(
                f"Bregman learning initialized: {len(self.selected_parameters)} parameter groups, "
                f"{total_params:,} total parameters, {sparsity:.2%} initial sparsity"
            )
    
    def _select_parameters(self, pl_module: LightningModule) -> None:
        """Select parameters for Bregman learning."""
        if self.parameter_selection == "all":
            # All trainable parameters
            self.selected_parameters = [p for p in pl_module.parameters() if p.requires_grad]
        elif self.parameter_selection == "weights_only":
            # Only weight parameters (exclude biases)
            self.selected_parameters = []
            for name, param in pl_module.named_parameters():
                if param.requires_grad and 'bias' not in name.lower():
                    self.selected_parameters.append(param)
        elif self.parameter_selection == "custom":
            # Custom selection function
            self.selected_parameters = self.custom_parameter_fn(pl_module)
        
        if not self.selected_parameters:
            raise RuntimeError("No parameters selected for Bregman learning")
        
        if self.verbose > 1:
            log.debug(f"Selected {len(self.selected_parameters)} parameters for Bregman learning")
    
    def _apply_sparse_initialization(self) -> None:
        """Apply sparse initialization to selected parameters."""
        if self.verbose > 0:
            log.info(f"Applying sparse initialization (sparsity={self.init_sparsity:.2%})...")
        
        for param in self.selected_parameters:
            # Create sparse mask - keep (1 - sparsity) fraction of parameters
            keep_prob = 1 - self.init_sparsity
            mask = torch.bernoulli(keep_prob * torch.ones_like(param))
            
            # Apply mask
            with torch.no_grad():
                param.data *= mask
                # Rescale to maintain expected variance only if we have non-zero elements
                nonzero_count = mask.sum().item()
                if nonzero_count > 0:
                    # Scale by sqrt of the ratio to maintain variance
                    scale_factor = torch.sqrt(torch.tensor(1.0 / keep_prob, device=param.device))
                    param.data *= scale_factor
        
        # Verify initialization sparsity
        actual_sparsity = self._compute_sparsity()
        if self.verbose > 0:
            log.info(f"Achieved initial sparsity: {actual_sparsity:.3%} (target: {self.init_sparsity:.3%})")
    
    def _create_regularizer(self) -> None:
        """Create regularizer instance."""
        try:
            self.regularizer = get_regularizer(self.regularizer_name, lamda=self.lamda)
            if self.verbose > 1:
                log.debug(f"Created regularizer: {self.regularizer_name} (λ={self.lamda})")
        except Exception as e:
            log.error(f"Failed to create regularizer {self.regularizer_name}: {e}")
            self.regularizer = RegNone()
    
    def _setup_bregman_optimizer(self, pl_module: LightningModule) -> None:
        """Setup Bregman optimizer."""
        # Store original optimizer for non-Bregman parameters
        try:
            self.original_optimizer = pl_module.optimizers()
            if isinstance(self.original_optimizer, list):
                if len(self.original_optimizer) > 1:
                    log.warning("Multiple optimizers detected. Using first optimizer as reference.")
                elif len(self.original_optimizer) == 0:
                    log.warning("No optimizers found in module. This is expected during setup.")
                    self.original_optimizer = None
                else:
                    self.original_optimizer = self.original_optimizer[0]
            elif self.original_optimizer is None:
                log.warning("No optimizer found in module. This is expected during setup.")
        except Exception as e:
            log.warning(f"Could not access module optimizers: {e}. This is expected during setup.")
            self.original_optimizer = None
        
        # Create Bregman optimizer
        optimizer_kwargs = {
            "lr": self.lr,
            "reg": self.regularizer,
            "delta": self.delta,
        }
        
        if self.optimizer_name == "linbreg":
            optimizer_kwargs["momentum"] = self.momentum
        elif self.optimizer_name == "adabreg":
            optimizer_kwargs["betas"] = self.betas
            optimizer_kwargs["eps"] = self.eps
        
        try:
            BregmanOptimizerClass = get_bregman_optimizer(self.optimizer_name)
            self.bregman_optimizer = BregmanOptimizerClass(
                self.selected_parameters, **optimizer_kwargs
            )
            
            # Replace the module's optimizer
            pl_module.trainer.optimizers = [self.bregman_optimizer]
            
            if self.verbose > 1:
                log.debug(f"Created Bregman optimizer: {self.optimizer_name}")
                
        except Exception as e:
            log.error(f"Failed to create Bregman optimizer {self.optimizer_name}: {e}")
            raise
    
    def on_train_batch_end(
        self, 
        trainer: Trainer, 
        pl_module: LightningModule, 
        outputs: Any, 
        batch: Any, 
        batch_idx: int
    ) -> None:
        """Update metrics after each training batch."""
        if self.collect_metrics and self._initialized:
            # Compute current sparsity
            sparsity = self._compute_sparsity()
            self.metrics["sparsity"].append(sparsity)
            
            # Compute regularization value
            if hasattr(self.bregman_optimizer, 'evaluate_reg'):
                reg_vals = self.bregman_optimizer.evaluate_reg()
                total_reg = sum(reg_vals)
                self.metrics["regularization"].append(total_reg)
    
    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Log metrics and perform end-of-epoch operations."""
        if not self._initialized:
            return
        
        # Log metrics
        if self.collect_metrics and trainer.logger:
            self._log_metrics(trainer)
        
        # Adaptive regularization (if target sparsity is set)
        if self.target_sparsity is not None:
            self._update_regularization_strength()
        
        # Verbose logging
        if self.verbose > 0:
            current_sparsity = self._compute_sparsity()
            log.info(f"Epoch {trainer.current_epoch}: sparsity={current_sparsity:.3%}")
    
    def _compute_sparsity(self) -> float:
        """Compute current sparsity of selected parameters."""
        if not self.selected_parameters:
            return 0.0
        
        total_params = 0
        zero_params = 0
        
        # Use a more appropriate threshold for sparsity detection
        threshold = 1e-6
        
        for param in self.selected_parameters:
            total_params += param.numel()
            zero_params += (param.abs() < threshold).sum().item()
        
        return zero_params / max(1, total_params)
    
    def _update_regularization_strength(self) -> None:
        """Update regularization strength based on target sparsity."""
        if self.target_sparsity is None:
            return
        
        current_sparsity = self._compute_sparsity()
        sparsity_diff = current_sparsity - self.target_sparsity
        
        # More aggressive adaptive scheme based on distance from target
        if abs(sparsity_diff) > 0.05:  # Far from target
            if current_sparsity < self.target_sparsity:
                # Need more sparsity - increase regularization more aggressively
                self.regularizer.lamda *= 1.2
            else:
                # Too sparse - decrease regularization more aggressively
                self.regularizer.lamda *= 0.8
        elif abs(sparsity_diff) > 0.02:  # Moderately close to target
            if current_sparsity < self.target_sparsity:
                self.regularizer.lamda *= 1.1
            else:
                self.regularizer.lamda *= 0.9
        else:  # Close to target - fine-tune
            if current_sparsity < self.target_sparsity:
                self.regularizer.lamda *= 1.05
            else:
                self.regularizer.lamda *= 0.95
        
        # Clamp to reasonable bounds - wider range for better control
        self.regularizer.lamda = max(1e-8, min(10.0, self.regularizer.lamda))
        
        if self.verbose > 1:
            log.debug(
                f"Sparsity: {current_sparsity:.3%} (target: {self.target_sparsity:.3%}), "
                f"Updated λ={self.regularizer.lamda:.2e}"
            )
    
    @rank_zero_only
    def _log_metrics(self, trainer: Trainer) -> None:
        """Log metrics to available loggers."""
        if not self.metrics:
            return
        
        metrics_to_log = {}
        
        # Average metrics over the epoch
        if "sparsity" in self.metrics and self.metrics["sparsity"]:
            avg_sparsity = sum(self.metrics["sparsity"]) / len(self.metrics["sparsity"])
            metrics_to_log["bregman/sparsity"] = avg_sparsity
            # Clear for next epoch
            self.metrics["sparsity"].clear()
        
        if "regularization" in self.metrics and self.metrics["regularization"]:
            avg_reg = sum(self.metrics["regularization"]) / len(self.metrics["regularization"])
            metrics_to_log["bregman/regularization"] = avg_reg
            self.metrics["regularization"].clear()
        
        # Log regularization strength
        metrics_to_log["bregman/lambda"] = self.regularizer.lamda
        metrics_to_log["bregman/delta"] = self.delta
        
        # Log to trainer's logger
        try:
            if hasattr(trainer.logger, 'log_metrics'):
                trainer.logger.log_metrics(metrics_to_log, step=trainer.global_step)
            elif hasattr(trainer.logger, 'experiment'):
                # For loggers like TensorBoard, Wandb etc.
                for key, value in metrics_to_log.items():
                    trainer.logger.experiment.add_scalar(key, value, trainer.global_step)
        except Exception as e:
            if self.verbose > 1:
                log.debug(f"Failed to log metrics: {e}")
    
    def get_sparsity_info(self) -> Dict[str, float]:
        """Get current sparsity information."""
        if not self._initialized:
            return {"sparsity": 0.0, "total_params": 0}
        
        total_params = sum(p.numel() for p in self.selected_parameters)
        sparsity = self._compute_sparsity()
        
        return {
            "sparsity": sparsity,
            "total_params": total_params,
            "nonzero_params": int(total_params * (1 - sparsity)),
            "regularization_strength": self.regularizer.lamda if self.regularizer else 0.0,
        }
    
    def state_dict(self) -> Dict[str, Any]:
        """Get state dict for checkpointing."""
        state = {
            "optimizer_name": self.optimizer_name,
            "regularizer_name": self.regularizer_name,
            "lr": self.lr,
            "delta": self.delta,
            "lamda": self.lamda,
            "initialized": self._initialized,
        }
        
        if self.regularizer:
            state["regularizer_lamda"] = self.regularizer.lamda
        
        return state
    
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load state dict from checkpoint."""
        self.optimizer_name = state_dict.get("optimizer_name", self.optimizer_name)
        self.regularizer_name = state_dict.get("regularizer_name", self.regularizer_name)
        self.lr = state_dict.get("lr", self.lr)
        self.delta = state_dict.get("delta", self.delta)
        self.lamda = state_dict.get("lamda", self.lamda)
        self._initialized = state_dict.get("initialized", False)
        
        if self.regularizer and "regularizer_lamda" in state_dict:
            self.regularizer.lamda = state_dict["regularizer_lamda"]
