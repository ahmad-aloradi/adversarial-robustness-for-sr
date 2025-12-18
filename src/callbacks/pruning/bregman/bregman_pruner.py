"""
BregmanPruner: A callback for orchestrating sparsity in Bregman-based training.
"""

from pytorch_lightning import Callback, LightningModule, Trainer
from pytorch_lightning.utilities import rank_zero_only
from typing import Any, Annotated, Optional

from .lambda_scheduler import LambdaScheduler
from src.callbacks.pruning.utils.pruning_manager import PruningManager

from src import utils

log = utils.get_pylogger(__name__)

class BregmanPruner(Callback):
    """
    Orchestrates sparsity-related operations during Bregman-based training.
    """
    
    def __init__(
        self,
        sparsity_threshold: float = 1e-30,
        collect_metrics: bool = True,
        verbose: int = 1,
        lambda_scheduler: LambdaScheduler = None,
        log_frequency: Annotated[int, "Log frequency for sparsity metrics"] = int(1e5)
        ):
        super().__init__()
        
        self.sparsity_threshold = sparsity_threshold
        self.collect_metrics = collect_metrics
        self.verbose = verbose
        self.lambda_scheduler = lambda_scheduler
        
        self.manager: PruningManager = None
        self._initialized = False
        self.log_frequency: Annotated[int, "Log frequency for metrics"] = log_frequency
        self._last_known_sparsity_from_ckpt: Optional[float] = None
        self._scheduler_state_from_ckpt: Optional[dict] = None
    
    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """
        Called at the beginning of training to initialize the pruner.
        """
        if not hasattr(pl_module, 'pruning_manager') or not isinstance(pl_module.pruning_manager, PruningManager):
            raise AttributeError(
                "The LightningModule must have a `pruning_manager` attribute of type PruningManager. "
                "Please instantiate it in `configure_optimizers`."
            )
        self.manager = pl_module.pruning_manager
        
        if not self._initialized:
            if not trainer.optimizers:
                log.warning("BregmanPruner: No optimizers found. Skipping initialization.")
                return

            if len(trainer.optimizers) > 1:
                raise ValueError(
                    f"BregmanPruner supports only a single optimizer, but found {len(trainer.optimizers)}. "
                    "Please ensure only one optimizer is configured."
                )
                
            if trainer.ckpt_path:
                log.info("BregmanPruner: Resuming from checkpoint. Skipping initial sparsity application.")
            else:
                log.info("BregmanPruner: Applying initial sparsity as defined by the PruningManager...")
                self.manager.apply_initial_sparsity()

            self._initialized = True
            
            self._setup_lambda_scheduler(trainer.optimizers[0])
            
            # If resuming, synchronize the optimizer param groups with the restored scheduler state
            if self._scheduler_state_from_ckpt:
                self._synchronize_regularization_strength(trainer)
                log.info("Restored lambda values to optimizer parameter groups.")

            # Log group assignments after all setup and restoration is complete.
            self._log_group_assignments(pl_module)

            log.info(f"Initial sparsity of pruned modules: {self._compute_overall_sparsity():.3%}")

    @rank_zero_only
    def _log_group_assignments(self, pl_module: LightningModule):
        """
        Logs the assignment of modules to pruning groups by inspecting the
        final parameter groups passed to the optimizer. This provides a definitive
        verification of the configuration's outcome.
        """
        if self.verbose == 0:
            return

        log.info("--- Pruning Group Configuration Verification ---")
        if not self.manager or not hasattr(self.manager, 'processed_groups'):
            log.warning("Could not verify pruning groups: PruningManager is not set up or has no 'processed_groups' attribute.")
            return

        # Create lookup maps for module names and types from the LightningModule.
        # This is the ground truth for linking parameters back to their parent modules.
        param_to_module_name = {
            id(p): '.'.join(name.split('.')[:-1])
            for name, p in pl_module.named_parameters()
        }
        module_name_to_type = {
            name: module.__class__.__name__
            for name, module in pl_module.named_modules()
        }

        # Create a map from a frozenset of param IDs to the corresponding optimizer group.
        # This allows us to find the optimizer group that matches a manager group.
        opt_groups_map = {
            frozenset(id(p) for p in group['params']): group
            for group in pl_module.trainer.optimizers[0].param_groups
        }

        # Use the manager's processed_groups to get the original configuration (name, etc.).
        for manager_group in self.manager.processed_groups:
            group_config = manager_group.get('config', {})
            group_name = group_config.get('name', 'Unnamed Group')
            
            # Find the corresponding group in the optimizer to get the correct lambda value.
            manager_param_ids = frozenset(id(p) for p in manager_group['params'])
            opt_group = opt_groups_map.get(manager_param_ids)

            log.info(f"Group '{group_name}':")
            
            current_group_sparsity = self._compute_group_sparsity(manager_group['params'])
            log.info(f"  - Current Sparsity: {current_group_sparsity:.3%}")

            # Get regularizer config from the manager_group for the name
            reg_config_from_manager = group_config.get('optimizer_settings', {}).get('reg')

            if opt_group and 'reg' in opt_group and hasattr(opt_group['reg'], 'lamda'):
                # Get live regularizer object from the optimizer group for the lambda value
                reg_from_opt = opt_group['reg']
                
                log.info(f"  - Regularizer: {getattr(reg_config_from_manager, '_target_', 'N/A')}")
                log.info(f"  - Initial Lambda: {reg_from_opt.lamda}")
                log.info(f"  - Lambda scale: {opt_group.get('lambda_scale', 1.0)}")
            else:
                log.info("  - Regularizer: None")

            log.info("  - Assigned Modules:")

            # Deduce unique module names from the parameters present in each group.
            assigned_modules = set()
            module_param_counts = {}
            total_group_params = 0

            for param in manager_group['params']:
                module_name = param_to_module_name.get(id(param))
                if module_name:
                    assigned_modules.add(module_name)
                    module_param_counts.setdefault(module_name, 0)
                    module_param_counts[module_name] += param.numel()
                    total_group_params += param.numel()

            if not assigned_modules:
                log.info("    - (No modules assigned to this group)")
                continue

            for module_name in sorted(list(assigned_modules)):
                module_type = module_name_to_type.get(module_name, "N/A")
                param_count = module_param_counts.get(module_name, 0)
                log.info(f"    - {module_name} (type: {module_type}, num_weights: {param_count})")

            log.info(f"  - Total number of weights in group: {total_group_params}")

        log.info("--- End of Parameters Groups Verification ---")

    def _compute_group_sparsity(self, params: list) -> float:
        """Computes sparsity for a specific list of parameters."""
        if not params:
            return 0.0
        
        params_to_consider = [p for p in params if p.requires_grad]
        if not params_to_consider:
            return 0.0
            
        total_params = sum(p.numel() for p in params_to_consider)
        zero_params = sum(
            (p.abs() <= self.sparsity_threshold).sum().item()
            for p in params_to_consider
        )
        return zero_params / max(1, total_params)

    def _compute_overall_sparsity(self) -> float:
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
        
        if self.collect_metrics and (trainer.global_step > 0 and trainer.global_step % 100 == 0):
            self._log_metrics(trainer)

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """
        Hook called at the end of each training epoch to log epoch-level sparsity.
        """
        if not self._initialized:
            return
        
        if self.verbose > 0:
            log.info(f"Epoch {trainer.current_epoch}: Sparsity of pruned modules = {self._compute_overall_sparsity():.3%}")

    def _update_regularization_strength(self, trainer: Trainer):
        """
        Steps the lambda scheduler to compute and apply a new regularization strength,
        scaled per parameter group. Only updates groups with lambda_scale > 0.
        """
        current_sparsity = self._compute_overall_sparsity()
        
        # If exists, pass the last cached sparsity ckpt on the first step after resuming
        last_sparsity = self._last_known_sparsity_from_ckpt
        if last_sparsity is not None:
            self._last_known_sparsity_from_ckpt = None # Use only once when resuming

        new_global_lamda = self.lambda_scheduler.step(current_sparsity, last_sparsity)

        for group in trainer.optimizers[0].param_groups:
            if ('reg' in group and hasattr(group['reg'], 'lamda') and 
                group.get('lambda_scale', 0.0) > 0.0):
                lambda_scale = group.get('lambda_scale', 1.0)
                group['reg'].lamda = new_global_lamda * lambda_scale

    @rank_zero_only
    def _log_metrics(self, trainer: Trainer) -> None:
        """
        Logs the current overall sparsity and the effective lambda for each relevant group.
        """
        sparsity = self._compute_overall_sparsity()
        
        metrics_to_log = {
            "bregman/total_model_sparsity": sparsity,
        }

        log_msgs = [f"Step {trainer.global_step}: Sparsity={sparsity:.3%}"]

        # Handle lambda logging based on whether scheduler is present
        if self.lambda_scheduler is not None:
            # With scheduler: log global lambda and managed group lambdas
            global_lambda = self.lambda_scheduler.get_lambda()
            metrics_to_log["bregman/global_lambda"] = global_lambda
            log_msgs[0] += f", Global Lambda={global_lambda:.4f}"

            for group in trainer.optimizers[0].param_groups:
                # Only log lambda for groups that are actively managed by the scheduler
                if ('reg' in group and hasattr(group['reg'], 'lamda') and 
                    'lambda_scale' in group and group.get('lambda_scale', 0.0) > 0.0):
                    group_name = group.get('name', 'unnamed_group')
                    effective_lambda = group['reg'].lamda
                    
                    metrics_to_log[f"bregman/lambda_{group_name}"] = effective_lambda
                    if self.verbose > 1:
                        log_msgs.append(f"| {group_name}_lambda={effective_lambda:.4f}")
        else:
            # Without scheduler: log static lambda values for regularized groups
            log_msgs[0] += ", Static Lambda Mode"
            
            for group in trainer.optimizers[0].param_groups:
                if ('reg' in group and hasattr(group['reg'], 'lamda') and 
                    group.get('lambda_scale', 0.0) > 0.0):
                    group_name = group.get('name', 'unnamed_group')
                    static_lambda = group['reg'].lamda
                    
                    metrics_to_log[f"bregman/static_lambda_{group_name}"] = static_lambda
                    if self.verbose > 1:
                        log_msgs.append(f"| {group_name}_static_lambda={static_lambda:.4f}")

        # Log metrics every self.log_frequency steps
        if self.verbose >= 1 and trainer.global_step % self.log_frequency == 0:
            trainer.logger.log_metrics(metrics_to_log, step=trainer.global_step)
        
        if self.verbose > 1 and trainer.global_step % self.log_frequency == 0:
            log.info(" ".join(log_msgs))

    def _synchronize_regularization_strength(self, trainer: Trainer):
        """
        Applies the current global lambda from the scheduler to the optimizer groups
        without stepping the scheduler. Used for synchronization after loading a checkpoint.
        """
        if self.lambda_scheduler is None:
            return
            
        current_global_lamda = self.lambda_scheduler.get_lambda()

        for group in trainer.optimizers[0].param_groups:
            if ('reg' in group and hasattr(group['reg'], 'lamda') and 
                group.get('lambda_scale', 0.0) > 0.0):
                lambda_scale = group.get('lambda_scale', 1.0)
                group['reg'].lamda = current_global_lamda * lambda_scale

    def on_save_checkpoint(self, trainer: Trainer, pl_module: LightningModule, checkpoint: dict) -> None:
        """Save lambda scheduler state to the checkpoint."""
        if self.lambda_scheduler and hasattr(self.lambda_scheduler, 'get_state'):
            checkpoint['lambda_scheduler_state'] = self.lambda_scheduler.get_state()
            # Save the current sparsity, which will be needed as `last_sparsity` on resume
            checkpoint['bregman_pruner_last_sparsity'] = self._compute_overall_sparsity()

    def on_load_checkpoint(self, trainer: Trainer, pl_module: LightningModule, checkpoint: dict) -> None:
        """
        Load lambda scheduler state from the checkpoint.
        This hook is called before `on_fit_start`, so we temporarily store the state.
        """
        if 'lambda_scheduler_state' in checkpoint:
            self._scheduler_state_from_ckpt = checkpoint['lambda_scheduler_state']

        if 'bregman_pruner_last_sparsity' in checkpoint:
            self._last_known_sparsity_from_ckpt = checkpoint['bregman_pruner_last_sparsity']

    def _setup_lambda_scheduler(self, optimizer) -> None:
        """
        Instantiates the lambda scheduler if it was passed as a partial function.
        Also validates consistency between scheduler and regularizer configurations,
        for per-batch updates.
        """
        if self.lambda_scheduler is None:
            log.info("No lambda scheduler configured - regularizer lamda values will remain static")
            self._validate_static_lambda_configuration(optimizer)
            return
        
        # Scheduler can be passed as an instance (has .step) or as a Hydra partial/callable.
        if not hasattr(self.lambda_scheduler, "step"):
            if not callable(self.lambda_scheduler):
                raise TypeError(
                    "Invalid lambda_scheduler: expected an object with a 'step' method or a callable that returns one. "
                    f"Got type={type(self.lambda_scheduler)}"
                )

            try:
                self.lambda_scheduler = self.lambda_scheduler()
            except Exception as e:
                raise RuntimeError(f"Failed to instantiate lambda scheduler: {e}") from e

            if not hasattr(self.lambda_scheduler, "step"):
                raise TypeError(
                    "Invalid lambda_scheduler instance: expected a 'step' method after instantiation. "
                    f"Got type={type(self.lambda_scheduler)}"
                )

            log.info(
                "Lambda scheduler instantiated with target sparsity: "
                f"{getattr(self.lambda_scheduler, 'target_sparsity', 'N/A')}"
            )

        # If we are resuming, load the state into the newly created scheduler
        if self._scheduler_state_from_ckpt:
            if hasattr(self.lambda_scheduler, 'load_state'):
                self.lambda_scheduler.load_state(self._scheduler_state_from_ckpt)
            else:
                log.warning("Found lambda_scheduler_state in checkpoint, but scheduler has no `load_state` method.")

        # Validate configuration consistency
        self._validate_lambda_configuration(optimizer)
        
    def _validate_lambda_configuration(self, optimizer) -> None:
        """
        Validates consistency between lambda scheduler and regularizer configurations.
        Issues warnings for potential misconfigurations.
        """
        if not self.lambda_scheduler:
            return
            
        scheduler_initial_lambda = self.lambda_scheduler.get_lambda()
        scheduler_target_sparsity = getattr(self.lambda_scheduler, 'target_sparsity', None)
        
        log.info("=== Lambda Configuration Validation ===")
        log.info(f"Lambda Scheduler - initial_lambda: {scheduler_initial_lambda:.4f}, target_sparsity: {scheduler_target_sparsity}")
        
        # Check regularizer lambda values only for groups that will be managed by the scheduler
        mismatched_lambdas = []
        for group in optimizer.param_groups:
            if 'reg' in group and hasattr(group['reg'], 'lamda'):
                group_name = group.get('name', 'unnamed_group')
                reg_lambda = group['reg'].lamda
                lambda_scale = group.get('lambda_scale', 1.0)
                
                log.info(f"Group '{group_name}' - regularizer lamda: {reg_lambda}, lambda_scale: {lambda_scale}")
                
                # Only check lambda consistency for groups that will be managed by the scheduler
                # (i.e., groups with lambda_scale > 0.0)
                if self._scheduler_state_from_ckpt is None: # Only check on initial run
                    if lambda_scale > 0.0 and abs(reg_lambda - scheduler_initial_lambda) > 1e-10:
                        mismatched_lambdas.append((group_name, reg_lambda, scheduler_initial_lambda))
        
        if mismatched_lambdas:
            log.warning("Lambda value mismatches detected:")
            for group_name, reg_lambda, sched_lambda in mismatched_lambdas:
                log.warning(f"  Group '{group_name}': regularizer lamda={reg_lambda} != scheduler initial_lambda={sched_lambda}")
            log.warning("Regularizer lamda values will be overridden by the scheduler during training.")
            
        # Report current sparsity for each group
        log.info("Current sparsity rates by group:")
        for group in self.manager.processed_groups:
            group_name = group["config"].get("name", "unnamed")
            current_sparsity = self._compute_group_sparsity(group['params'])
            log.info(f"  Group '{group_name}': {current_sparsity:.3%}")
                
        log.info("=== End Lambda Configuration Validation ===")

    def _validate_static_lambda_configuration(self, optimizer) -> None:
        """
        Validates configuration when no lambda scheduler is used.
        Provides information about static lambda values.
        """
        log.info("=== Static Lambda Configuration Validation ===")
        log.info("No lambda scheduler - regularizer lambda values will remain constant during training")
        
        # Show static lambda values for regularized groups
        regularized_groups = []
        for group in optimizer.param_groups:
            if 'reg' in group and hasattr(group['reg'], 'lamda'):
                group_name = group.get('name', 'unnamed_group')
                reg_lambda = group['reg'].lamda
                lambda_scale = group.get('lambda_scale', 1.0)
                
                if lambda_scale > 0.0:
                    regularized_groups.append((group_name, reg_lambda))
                    log.info(f"Group '{group_name}' - static lamda: {reg_lambda}")
                else:
                    log.info(f"Group '{group_name}' - no regularization (lambda_scale=0.0)")
        
        if regularized_groups:
            log.info(f"Found {len(regularized_groups)} regularized groups with static lambda values")
            log.info("These lambda values will NOT change during training!")
        else:
            log.warning("No regularized groups found - no regularization will be applied")
            
        # Report current sparsity for each group
        log.info("Current sparsity rates by group:")
        for group in self.manager.processed_groups:
            group_name = group["config"].get("name", "unnamed")
            current_sparsity = self._compute_group_sparsity(group['params'])
            log.info(f"  Group '{group_name}': {current_sparsity:.3%}")
        
        log.info("=== End Static Lambda Configuration Validation ===")
