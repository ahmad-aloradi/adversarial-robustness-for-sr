"""
BregmanCallback: A callback for orchestrating sparsity in Bregman-based training.

Note: Despite the legacy name "pruner", Bregman learning starts with a sparse model
and allows it to become denser during training. The lambda scheduler adjusts
regularization strength to drive sparsity toward a target level.
"""

from pytorch_lightning import Callback, LightningModule, Trainer
from pytorch_lightning.utilities import rank_zero_only
from typing import Any, Optional, List

from .lambda_scheduler import LambdaScheduler
from src.callbacks.pruning.utils.pruning_manager import PruningManager
from src import utils

log = utils.get_pylogger(__name__)


class BregmanPruner(Callback):
    """
    Orchestrates sparsity-related operations during Bregman-based training.
    
    This callback:
    - Applies initial sparsity to the model (via PruningManager)
    - Optionally updates regularization strength (lambda) per batch via LambdaScheduler
    - Logs sparsity metrics during training
    - Handles checkpointing of scheduler state
    """
    
    def __init__(
        self,
        sparsity_threshold: float = 1e-30,
        verbose: int = 1,
        lambda_scheduler: Optional[LambdaScheduler] = None,
        console_log_frequency: int = 100,
    ):
        """
        Args:
            sparsity_threshold: Threshold below which a weight is considered zero.
            verbose: Verbosity level (0=silent, 1=normal, 2=detailed).
            lambda_scheduler: Optional scheduler for dynamic lambda updates.
            console_log_frequency: How often (in steps) to print to console (verbose >= 2).
        """
        super().__init__()
        self.sparsity_threshold = sparsity_threshold
        self.verbose = verbose
        self.lambda_scheduler = lambda_scheduler
        self.console_log_frequency = console_log_frequency
        
        self.manager: Optional[PruningManager] = None
        self._initialized = False
        self._ckpt_scheduler_state: Optional[dict] = None
        self._ckpt_last_sparsity: Optional[float] = None
    
    # -------------------------------------------------------------------------
    # Lightning hooks
    # -------------------------------------------------------------------------
    
    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Initialize the callback at the start of training."""
        self._validate_module(pl_module)
        self.manager = pl_module.pruning_manager
        
        if self._initialized:
            return
            
        if not trainer.optimizers:
            raise ValueError("BregmanPruner: No optimizers found.")
        if len(trainer.optimizers) > 1:
            raise ValueError("BregmanPruner supports only a single optimizer.")
        
        optimizer = trainer.optimizers[0]
        is_resuming = trainer.ckpt_path is not None
        
        if is_resuming:
            log.info("BregmanPruner: Resuming from checkpoint.")
        else:
            log.info("BregmanPruner: Applying initial sparsity...")
            self.manager.apply_initial_sparsity()
        
        self._setup_lambda_scheduler(optimizer, is_resuming)
        
        if is_resuming and self._ckpt_scheduler_state:
            self._apply_lambda_to_groups(trainer)
            log.info("Restored lambda values to optimizer parameter groups.")
        
        self._initialized = True
        self._log_configuration(optimizer)

    def on_train_batch_end(
        self, trainer: Trainer, pl_module: LightningModule,
        outputs: Any, batch: Any, batch_idx: int
    ) -> None:
        """Update lambda scheduler and log metrics after each batch."""
        if not self._initialized:
            return
        
        if self.lambda_scheduler is not None:
            self._step_lambda_scheduler(trainer)
        
        # Log metrics via Lightning's logging system (respects logging_params)
        self._log_metrics(pl_module, trainer)
        
        # Console logging at higher verbosity with separate frequency
        if self.verbose >= 2 and trainer.global_step % self.console_log_frequency == 0:
            self._log_to_console(trainer)

    def on_validation_epoch_start(self, trainer, pl_module):
        """Log a CRITICAL warning if sparsity is less than the target."""
        current_sparsity = self._compute_overall_sparsity()
        tolerance = 5e-3

        # Log a warning and skip validation if sparsity is below target
        if self.lambda_scheduler is not None:
            target_sparsity = self.lambda_scheduler.target_sparsity
            if current_sparsity < target_sparsity - tolerance:
                log.critical(f"Validation is done for sparsity {current_sparsity} < target {target_sparsity}!!!")

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Log epoch-level sparsity."""
        if not self._initialized or self.verbose == 0:
            return
        log.info(f"Epoch {trainer.current_epoch}: Sparsity = {self._compute_overall_sparsity():.3%}")

    def on_save_checkpoint(
        self, trainer: Trainer, pl_module: LightningModule, checkpoint: dict
    ) -> None:
        """Save scheduler state to checkpoint."""
        if self.lambda_scheduler is not None:
            checkpoint['lambda_scheduler_state'] = self.lambda_scheduler.get_state()
            checkpoint['bregman_last_sparsity'] = self._compute_overall_sparsity()

    def on_load_checkpoint(
        self, trainer: Trainer, pl_module: LightningModule, checkpoint: dict
    ) -> None:
        """Load scheduler state from checkpoint (before on_fit_start)."""
        self._ckpt_scheduler_state = checkpoint.get('lambda_scheduler_state')
        self._ckpt_last_sparsity = checkpoint.get('bregman_last_sparsity')

    # -------------------------------------------------------------------------
    # Scheduler management
    # -------------------------------------------------------------------------
    
    def _setup_lambda_scheduler(self, optimizer, is_resuming: bool) -> None:
        """Instantiate and configure the lambda scheduler."""
        if self.lambda_scheduler is None:
            return
        
        # Handle Hydra partial instantiation
        if not hasattr(self.lambda_scheduler, "step"):
            if not callable(self.lambda_scheduler):
                raise TypeError(
                    f"lambda_scheduler must have a 'step' method or be callable, "
                    f"got {type(self.lambda_scheduler)}"
                )
            self.lambda_scheduler = self.lambda_scheduler()
        
        # Restore state from checkpoint
        if is_resuming and self._ckpt_scheduler_state:
            self.lambda_scheduler.load_state(self._ckpt_scheduler_state)
        
        log.info(
            f"Lambda scheduler active: target_sparsity={self.lambda_scheduler.target_sparsity}, "
            f"initial_lambda={self.lambda_scheduler.get_lambda():.4f}"
        )

    def _step_lambda_scheduler(self, trainer: Trainer) -> None:
        """Step the scheduler and update regularizer lambdas."""
        current_sparsity = self._compute_overall_sparsity()
        
        # On first step after resume, pass the cached sparsity for EMA initialization
        last_sparsity = self._ckpt_last_sparsity
        if last_sparsity is not None:
            self._ckpt_last_sparsity = None  # Use only once
        
        new_lambda = self.lambda_scheduler.step(current_sparsity, last_sparsity)
        
        for group in trainer.optimizers[0].param_groups:
            if self._group_has_regularizer(group):
                scale = group.get('lambda_scale', 1.0)
                group['reg'].lamda = new_lambda * scale

    def _apply_lambda_to_groups(self, trainer: Trainer) -> None:
        """Apply current scheduler lambda to all regularized groups."""
        if self.lambda_scheduler is None:
            return
        current_lambda = self.lambda_scheduler.get_lambda()
        for group in trainer.optimizers[0].param_groups:
            if self._group_has_regularizer(group):
                scale = group.get('lambda_scale', 1.0)
                group['reg'].lamda = current_lambda * scale

    # -------------------------------------------------------------------------
    # Sparsity computation
    # -------------------------------------------------------------------------
    
    def _compute_overall_sparsity(self) -> float:
        """Compute sparsity across all pruned parameters."""
        params = self.manager.get_pruned_parameters()
        return self._compute_sparsity(params)
    
    def _compute_group_sparsity(self, params: List) -> float:
        """Compute sparsity for a specific parameter group."""
        return self._compute_sparsity(params)
    
    def _compute_sparsity(self, params: List) -> float:
        """Compute fraction of near-zero parameters."""
        if not params:
            return 0.0
        params = [p for p in params if p.requires_grad]
        if not params:
            return 0.0
        total = sum(p.numel() for p in params)
        zeros = sum((p.abs() <= self.sparsity_threshold).sum().item() for p in params)
        return zeros / max(1, total)

    # -------------------------------------------------------------------------
    # Logging
    # -------------------------------------------------------------------------
    
    def _log_metrics(self, pl_module: LightningModule, trainer: Trainer) -> None:
        """Log sparsity and lambda metrics via Lightning's logging system.
        
        Uses pl_module.logging_params if available for consistent logging behavior.
        """
        # Use module's logging_params if available, otherwise use sensible defaults
        default_logging_params = {'on_step': False, 'on_epoch': True, 'sync_dist': True, 'prog_bar': False}
        logging_params = getattr(pl_module, 'logging_params', default_logging_params)
        
        sparsity = self._compute_overall_sparsity()
        pl_module.log("bregman/sparsity", sparsity, **logging_params)
        
        if self.lambda_scheduler:
            # Lambda changes per step, so always log on_step; override on_epoch to avoid noise
            lambda_params = {**logging_params, 'on_epoch': False,  'on_step': True}
            pl_module.log("bregman/global_lambda", self.lambda_scheduler.get_lambda(), **lambda_params)
            
            ema_sparsity = self.lambda_scheduler.get_ema_smoothed_sparsity()
            if ema_sparsity is not None:
                pl_module.log("bregman/ema_sparsity", ema_sparsity, **logging_params)

    @rank_zero_only
    def _log_to_console(self, trainer: Trainer) -> None:
        """Log sparsity info to console (controlled by console_log_frequency)."""
        sparsity = self._compute_overall_sparsity()
        msg = f"Step {trainer.global_step}: sparsity={sparsity:.3%}"
        if self.lambda_scheduler:
            msg += f", lambda={self.lambda_scheduler.get_lambda():.4f}"
        log.info(msg)

    @rank_zero_only
    def _log_configuration(self, optimizer) -> None:
        """Log the configuration of all parameter groups."""
        if self.verbose == 0:
            return
        
        log.info("=== Bregman Configuration ===")
        
        if self.lambda_scheduler:
            log.info(f"Lambda Scheduler: target_sparsity={self.lambda_scheduler.target_sparsity}, "
                     f"lambda={self.lambda_scheduler.get_lambda():.4f}")
        else:
            log.info("Lambda Scheduler: None (static lambda mode)")
        
        for group in optimizer.param_groups:
            name = group.get('name', 'unnamed')
            scale = group.get('lambda_scale', 1.0)
            
            if self._group_has_regularizer(group):
                lamda = group['reg'].lamda
                reg_type = type(group['reg']).__name__
                log.info(f"  Group '{name}': {reg_type}, lambda={lamda:.4f}, scale={scale}")
            else:
                log.info(f"  Group '{name}': No regularizer")
        
        log.info("Current sparsity by group:")
        for group in self.manager.processed_groups:
            name = group["config"].get("name", "unnamed")
            sparsity = self._compute_group_sparsity(group['params'])
            log.info(f"  {name}: {sparsity:.3%}")
        
        log.info(f"Overall sparsity: {self._compute_overall_sparsity():.3%}")
        log.info("=== End Configuration ===")

    @rank_zero_only
    def _log_group_assignments(self, pl_module: LightningModule) -> None:
        """Log detailed group assignments (for debugging)."""
        if self.verbose < 2:
            return
        
        param_to_module = {
            id(p): '.'.join(name.split('.')[:-1])
            for name, p in pl_module.named_parameters()
        }
        
        log.info("--- Parameter Group Assignments ---")
        for group in self.manager.processed_groups:
            name = group["config"].get("name", "unnamed")
            modules = {param_to_module.get(id(p)) for p in group['params']}
            modules.discard(None)
            log.info(f"  {name}: {len(modules)} modules, {len(group['params'])} params")
        log.info("-----------------------------------")

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------
    
    @staticmethod
    def _group_has_regularizer(group: dict) -> bool:
        """Check if a param group has an active regularizer."""
        return (
            'reg' in group 
            and hasattr(group['reg'], 'lamda') 
            and group.get('lambda_scale', 0.0) > 0.0
        )
    
    @staticmethod
    def _validate_module(pl_module: LightningModule) -> None:
        """Validate that the module has a pruning_manager."""
        if not hasattr(pl_module, 'pruning_manager'):
            raise AttributeError(
                "LightningModule must have a 'pruning_manager' attribute. "
                "Please instantiate it in configure_optimizers()."
            )
        if not isinstance(pl_module.pruning_manager, PruningManager):
            raise TypeError(
                f"pruning_manager must be a PruningManager, got {type(pl_module.pruning_manager)}"
            )
