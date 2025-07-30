"""
BregmanPruner: A callback for orchestrating sparsity in Bregman-based training.
"""

from pytorch_lightning import Callback, LightningModule, Trainer
from pytorch_lightning.utilities import rank_zero_only
from typing import Any, Annotated

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
                
            self._log_group_assignments(pl_module)
            
            log.info("BregmanPruner: Applying initial sparsity as defined by the PruningManager...")
            self.manager.apply_initial_sparsity()
            self._initialized = True
            
            self._setup_lambda_scheduler(trainer.optimizers[0])
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

        # Use the final, processed groups that are sent to the optimizer.
        for group in self.manager.processed_groups:
            group_config = group.get('config', {})
            group_name = group_config.get('name', 'Unnamed Group')
            optimizer_settings = group_config.get('optimizer_settings', {})
            reg_config = optimizer_settings.get('reg')

            log.info(f"Group '{group_name}':")

            if reg_config:
                log.info(f"  - Regularizer: {getattr(reg_config, '_target_', 'N/A')}")
                if hasattr(reg_config, 'lamda'):
                    log.info(f"  - Initial Lambda: {reg_config.lamda}")
                    log.info(f"  - Lambda scale: {optimizer_settings.lambda_scale}")
                else:
                    log.info(f"   - Regularizer has no lamda -> Initial Lambda: 0.0")
            else:
                log.info("  - Regularizer: None")

            log.info("  - Assigned Modules:")

            # Deduce unique module names from the parameters present in each group.
            assigned_modules = set()
            module_param_counts = {}
            total_group_params = 0

            for param in group['params']:
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
        new_global_lamda = self.lambda_scheduler.step(current_sparsity)

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

    def _setup_lambda_scheduler(self, optimizer) -> None:
        """
        Instantiates the lambda scheduler if it was passed as a partial function.
        Also validates consistency between scheduler and regularizer configurations.
        """
        if self.lambda_scheduler is None:
            log.info("No lambda scheduler configured - regularizer lamda values will remain static")
            self._validate_static_lambda_configuration(optimizer)
            return
        
        if callable(self.lambda_scheduler) and not hasattr(self.lambda_scheduler, 'step'):
            try:
                self.lambda_scheduler = self.lambda_scheduler(optimizer=optimizer)
                log.info(f"Lambda scheduler instantiated with target sparsity: {getattr(self.lambda_scheduler, 'target_sparsity', 'N/A')}")
            except Exception as e:
                log.error(f"Failed to instantiate lambda scheduler: {e}")
                self.lambda_scheduler = None
                self._validate_static_lambda_configuration(optimizer)
                return
        else:
            log.info("Lambda scheduler already instantiated.")
            
        # Validate configuration consistency
        self._validate_lambda_configuration(optimizer)
        
    def _validate_lambda_configuration(self, optimizer) -> None:
        """
        Validates consistency between lambda scheduler and regularizer configurations.
        Issues warnings for potential misconfigurations.
        """
        if not self.lambda_scheduler:
            return
            
        scheduler_initial_lambda = getattr(self.lambda_scheduler, 'lambda_value', None)
        scheduler_target_sparsity = getattr(self.lambda_scheduler, 'target_sparsity', None)
        
        log.info("=== Lambda Configuration Validation ===")
        log.info(f"Lambda Scheduler - initial_lambda: {scheduler_initial_lambda}, target_sparsity: {scheduler_target_sparsity}")
        
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
                if lambda_scale > 0.0 and abs(reg_lambda - scheduler_initial_lambda) > 1e-10:
                    mismatched_lambdas.append((group_name, reg_lambda, scheduler_initial_lambda))
        
        if mismatched_lambdas:
            log.warning("Lambda value mismatches detected:")
            for group_name, reg_lambda, sched_lambda in mismatched_lambdas:
                log.warning(f"  Group '{group_name}': regularizer lamda={reg_lambda} != scheduler initial_lambda={sched_lambda}")
            log.warning("Regularizer lamda values will be overridden by the scheduler during training.")
            
        # Check sparsity configuration
        initial_sparsity_rates = []
        for group in self.manager.processed_groups:
            applier = group["applier"]
            if applier.sparsity_rate > 0:
                group_name = group["config"].get("name", "unnamed")
                initial_sparsity_rates.append((group_name, applier.sparsity_rate))
                
        if initial_sparsity_rates:
            log.info("Initial sparsity rates by group:")
            for group_name, rate in initial_sparsity_rates:
                log.info(f"  Group '{group_name}': {rate:.1%}")
                
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
            
        # Show sparsity configuration
        initial_sparsity_rates = []
        for group in self.manager.processed_groups:
            applier = group["applier"]
            if applier.sparsity_rate > 0:
                group_name = group["config"].get("name", "unnamed")
                initial_sparsity_rates.append((group_name, applier.sparsity_rate))
                
        if initial_sparsity_rates:
            log.info("Initial sparsity rates by group:")
            for group_name, rate in initial_sparsity_rates:
                log.info(f"  Group '{group_name}': {rate:.1%}")
        
        log.info("=== End Static Lambda Configuration Validation ===")
