"""
BregmanPruner: A callback for orchestrating sparsity in Bregman-based training.
"""
import torch
from pytorch_lightning import Callback, LightningModule, Trainer
from pytorch_lightning.utilities import rank_zero_only
from typing import List, Any, Dict

from .lambda_scheduler import LambdaScheduler
from src.callbacks.pruning.utils.pruning_manager import PruningManager

from src import utils

log = utils.get_pylogger(__name__)

class BregmanPruner(Callback):
    """
    Orchestrates sparsity-related operations during Bregman-based training.

    This callback acts as a high-level orchestrator that hooks into the PyTorch
    Lightning training loop. It delegates all complex logic for parameter grouping,
    sparsity application, and regularization to a `PruningManager` instance.

    Its primary responsibilities are:
    - Triggering the application of initial sparsity at the start of training.
    - Logging sparsity metrics and other relevant information periodically.
    - Managing the `lambda_scheduler` to update regularization strength.

    The `PruningManager` must be instantiated and attached to the LightningModule
    (as `pl_module.pruning_manager`) within the `configure_optimizers` method to
    ensure a single, unified source of truth for all pruning configurations.

    Args:
        sparsity_threshold (float): The numerical threshold below which a weight
            is considered zero for sparsity calculations.
        collect_metrics (bool): If True, logs metrics like sparsity and lambda
            to the PyTorch Lightning logger.
        verbose (int): Controls the verbosity of logging.
            - 0: Silent.
            - 1: Logs key events like epoch-end sparsity.
            - 2: Logs detailed step-level metrics.
        lambda_scheduler (LambdaScheduler, optional): An optional scheduler for
            dynamically adjusting the regularization strength (lambda) during training.
    """
    
    def __init__(
        self,
        sparsity_threshold: float = 1e-30,
        collect_metrics: bool = True,
        verbose: int = 1,
        lambda_scheduler: LambdaScheduler = None,
        ):
        super().__init__()
        
        self.sparsity_threshold = sparsity_threshold
        self.collect_metrics = collect_metrics
        self.verbose = verbose
        self.lambda_scheduler = lambda_scheduler
        
        self.manager: PruningManager = None
        self._initialized = False
    
    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """
        Called at the beginning of training to initialize the pruner.

        This method retrieves the `PruningManager` from the LightningModule,
        triggers the application of initial sparsity, and sets up the lambda
        scheduler.
        """
        if not hasattr(pl_module, 'pruning_manager') or not isinstance(pl_module.pruning_manager, PruningManager):
            raise AttributeError(
                "The LightningModule must have a `pruning_manager` attribute of type PruningManager. "
                "Please instantiate it in `configure_optimizers`."
            )
        self.manager = pl_module.pruning_manager
        
        if not self._initialized:
            log.info("BregmanPruner: Applying initial sparsity as defined by the PruningManager...")
            self.manager.apply_initial_sparsity()
            self._initialized = True
            
            # Setup the lambda scheduler with the configured optimizer
            self._setup_lambda_scheduler(trainer.optimizers[0])
            log.info(f"Initial sparsity of pruned modules: {self._compute_sparsity():.3%}")

    def _compute_sparsity(self) -> float:
        """
        Computes the current sparsity of all parameters targeted by the PruningManager.
        """
        selected_params = self.manager.get_pruned_parameters()
        if not selected_params:
            return 0.0
        
        total_params = sum(p.numel() for p in selected_params if p.requires_grad)
        zero_params = sum(
            (p.abs() <= self.sparsity_threshold).sum().item()
            for p in selected_params if p.requires_grad
        )
        return zero_params / max(1, total_params)

    def on_train_batch_end(self, trainer: Trainer, pl_module: LightningModule, outputs: Any, batch: Any, batch_idx: int) -> None:
        """
        Hook called at the end of each training batch to update schedulers and log metrics.
        """
        if not self._initialized:
            return
        
        if self.lambda_scheduler is not None:
            self._update_regularization_strength(trainer)
        
        if self.collect_metrics and (trainer.global_step % 100 == 0):
             self._log_metrics(trainer)

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """
        Hook called at the end of each training epoch to log epoch-level sparsity.
        """
        if not self._initialized:
            return
        if self.verbose > 0:
            log.info(f"Epoch {trainer.current_epoch}: Sparsity of pruned modules = {self._compute_sparsity():.3%}")

    def _update_regularization_strength(self, trainer: Trainer):
        """
        Steps the lambda scheduler to compute and apply a new regularization strength.
        """
        current_sparsity = self._compute_sparsity()
        new_lamda = self.lambda_scheduler.step(current_sparsity)

        # Apply the new lambda to all parameter groups that have a regularizer
        for group in trainer.optimizers[0].param_groups:
            if 'reg' in group and hasattr(group['reg'], 'lamda'):
                group['reg'].lamda = new_lamda

    @rank_zero_only
    def _log_metrics(self, trainer: Trainer) -> None:
        """
        Logs the current sparsity and lambda value to the logger.
        """
        sparsity = self._compute_sparsity()
        
        # For logging purposes, find the lambda from the first group that has one
        lamda = 0
        for group in trainer.optimizers[0].param_groups:
            if 'reg' in group and hasattr(group['reg'], 'lamda'):
                lamda = group['reg'].lamda
                break
        
        metrics_to_log = {
            "bregman/pruned_module_sparsity": sparsity,
            "bregman/lambda": lamda,
        }
        trainer.logger.log_metrics(metrics_to_log, step=trainer.global_step)
        
        if self.verbose > 1:
            log.info(f"Step {trainer.global_step}: Sparsity={sparsity:.3%}, lambda={lamda:.4f}")

    def _setup_lambda_scheduler(self, optimizer) -> None:
        """
        Instantiates the lambda scheduler if it was passed as a partial function.
        """
        if self.lambda_scheduler is None:
            return
        
        # If scheduler is a partial function (common with Hydra configs), instantiate it
        if callable(self.lambda_scheduler) and not hasattr(self.lambda_scheduler, 'step'):
            try:
                self.lambda_scheduler = self.lambda_scheduler(optimizer=optimizer)
                log.info(f"Lambda scheduler instantiated with target sparsity: {getattr(self.lambda_scheduler, 'target_sparse', 'N/A')}")
            except Exception as e:
                log.error(f"Failed to instantiate lambda scheduler: {e}")
                self.lambda_scheduler = None
        else:
            log.info("Lambda scheduler already instantiated.")