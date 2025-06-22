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
from src import utils

log = utils.get_pylogger(__name__)


class LamdaScheduler:
    """
    Scheduler for the regularization parameter multiplier (mu).
    """

    def __init__(
        self,
        warmup: int = 0,
        increment: float = 0.05,
        cooldown: int = 0,
        target_sparsity: float = 1.0,
    ):
        self.warmup = warmup
        self.increment = increment
        self.cooldown = cooldown
        self.cooldown_val = cooldown
        self.target_sparsity = target_sparsity

    def step(self, current_mu: float, sparse: float) -> float:
        if self.warmup > 0:
            self.warmup -= 1
            return current_mu
        
        if self.warmup == 0: self.warmup = -1

        if self.cooldown_val > 0:
            self.cooldown_val -= 1
            return current_mu
        
        self.cooldown_val = self.cooldown
        
        if sparse > self.target_sparsity:
            return current_mu + self.increment
        else:
            return max(current_mu - self.increment, 0.0)


class BregmanPruner(Callback):
    """
    BregmanPruner: Sparse neural network training using Bregman learning framework.
    Applies a Bregman proximal step after each optimizer step.
    """
    
    def __init__(
        self,
        regularizer_name: str = "l1",
        delta: float = 1.0,
        lamda: float = 1e-3,
        mu: float = 1.0,
        sparse_init: bool = True,
        init_sparsity: float = 0.9,
        target_sparsity: Optional[float] = None,
        sparsity_threshold: float = 1e-6,
        scheduler_warmup: int = 0,
        scheduler_increment: float = 0.05,
        scheduler_cooldown: int = 0,
        prune_module_names: Optional[List[str]] = None,
        collect_metrics: bool = True,
        verbose: int = 1,
    ):
        super().__init__()
        
        self.regularizer_name = regularizer_name
        self.delta = delta
        self.lamda = lamda
        self.mu = mu
        self.sparse_init = sparse_init
        self.init_sparsity = init_sparsity
        self.target_sparsity = target_sparsity
        self.sparsity_threshold = sparsity_threshold
        self.scheduler_warmup = scheduler_warmup
        self.scheduler_increment = scheduler_increment
        self.scheduler_cooldown = scheduler_cooldown
        self.prune_module_names = prune_module_names
        self.collect_metrics = collect_metrics
        self.verbose = verbose
        
        self.selected_parameters = []
        self.pruned_params_ids = set()
        self._initialized = False
    
    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """
        Called when fit begins.
        """
        if not self._initialized:
            self._initialize_bregman_learning(pl_module)
            self._initialized = True
    
    def _initialize_bregman_learning(self, pl_module: LightningModule) -> None:
        log.info("Initializing Bregman learning framework...")
        
        self._select_parameters(pl_module)
        if not self.selected_parameters:
            log.warning("No parameters selected for pruning. BregmanPruner will have no effect.")
            return

        if self.sparse_init: self._apply_sparse_initialization()
        
        self.regularizer = get_regularizer(self.regularizer_name, lamda=self.lamda, delta=self.delta)
        self.regularizer.mu = self.mu
        
        if self.target_sparsity is not None:
            self.lamda_scheduler = LamdaScheduler(
                warmup=self.scheduler_warmup,
                increment=self.scheduler_increment,
                cooldown=self.scheduler_cooldown,
                target_sparsity=self.target_sparsity,
            )
        
        log.info(f"Bregman learning initialized. Sparsity of pruned modules: {self._compute_sparsity():.2%}")
    
    def _select_parameters(self, pl_module: LightningModule) -> None:
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
        if not self._initialized or not self.selected_parameters: return
        
        # Apply Bregman proximal step after the optimizer's step
        with torch.no_grad():
            for opt in trainer.optimizers:
                for group in opt.param_groups:
                    lr = group['lr']
                    for p in group['params']:
                        if p.requires_grad and id(p) in self.pruned_params_ids:
                            p.data = self.regularizer.prox(p, lr).data

        if hasattr(self, 'lamda_scheduler'): self._update_regularization_strength(trainer)
        if self.collect_metrics and (trainer.global_step % 100 == 0):
             self._log_metrics(trainer)

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if not self._initialized or not self.selected_parameters: return
        if self.verbose > 0:
            log.info(f"Epoch {trainer.current_epoch}: Sparsity of pruned modules = {self._compute_sparsity():.3%}")

    def _compute_sparsity(self) -> float:
        if not self.selected_parameters: return 0.0
        total_params = sum(p.numel() for p in self.selected_parameters if p.requires_grad)
        zero_params = sum((p.abs() < self.sparsity_threshold).sum().item() for p in self.selected_parameters if p.requires_grad)
        return zero_params / max(1, total_params)
    
    def _update_regularization_strength(self, trainer: Trainer) -> None:
        current_sparsity = self._compute_sparsity()
        new_mu = self.lamda_scheduler.step(self.regularizer.mu, current_sparsity)
        if new_mu != self.regularizer.mu:
            self.regularizer.mu = new_mu

    @rank_zero_only
    def _log_metrics(self, trainer: Trainer) -> None:
        metrics_to_log = {
            "bregman/pruned_module_sparsity": self._compute_sparsity(),
            "bregman/mu": self.regularizer.mu,
            "bregman/lambda": self.regularizer.lamda,
        }
        trainer.logger.log_metrics(metrics_to_log, step=trainer.global_step)