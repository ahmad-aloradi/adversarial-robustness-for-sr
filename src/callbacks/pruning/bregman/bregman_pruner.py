"""
BregmanPruner: A callback for orchestrating sparsity in Bregman-based training.

Note: Despite the legacy name "pruner", Bregman learning starts with a sparse
model and allows it to become denser during training. The lambda scheduler
adjusts regularization strength to drive sparsity toward a target level.
"""

import csv
import os
from typing import Any, Dict, List, Literal, Optional

import torch
from pytorch_lightning import Callback, LightningModule, Trainer
from pytorch_lightning.utilities import rank_zero_only

from src import utils
from src.callbacks.pruning.shared_prune_utils import compute_sparsity
from src.callbacks.pruning.utils.pruning_manager import (
    SCALES_ATTR,
    PruningManager,
)

from .bregman_regularizers import RegNone
from .lambda_scheduler import LambdaScheduler

log = utils.get_pylogger(__name__)

# How to steer lambda: over all model parameters, or only the regularized
# ("pruned") groups. Overall is more intuitive; pruned is more principled
# (the feedback loop sees exactly the params the regularizer acts on).
WHICH_SPARSITY_PERCENTAGE: Literal["overall", "pruned"] = "overall"

# Per-layer scale evolution dump; read back by src/vis/scale_evolution.py.
SCALE_HISTORY_CSV = "trainable_scales_history.csv"


class BregmanPruner(Callback):
    """Orchestrates sparsity-related operations during Bregman-based training.

    This callback:
    - Applies initial sparsity to the model (via PruningManager)
    - Optionally updates regularization strength (lambda) per batch via
      LambdaScheduler (fixed-target feedback)
    - Optionally trains one scalar per-layer scale factor (allocation) when
      groups carry a trainable_scale marker
    - Logs sparsity metrics and checkpoints the scheduler state
    """

    def __init__(
        self,
        sparsity_threshold: float = 1e-12,
        verbose: int = 1,
        lambda_scheduler: Optional[LambdaScheduler] = None,
        target_sparsity: Optional[float] = None,
    ):
        """
        Args:
            sparsity_threshold: Threshold below which a weight is considered zero.
            verbose: Verbosity level (0=silent, 1=normal, 2=detailed).
            lambda_scheduler: Optional scheduler for dynamic lambda updates.
            target_sparsity: Sparsity setpoint the lambda controller is driven
                toward, the value the feasibility check guards, and the target
                logged each epoch. The sparsity-gated callbacks are configured
                with the same value independently. Required when a
                lambda_scheduler is set.
        """
        super().__init__()
        self.sparsity_threshold = sparsity_threshold
        self.verbose = verbose
        self.lambda_scheduler = lambda_scheduler
        if target_sparsity is None:
            self._target_sparsity: Optional[float] = None
        else:
            self._target_sparsity = float(target_sparsity)
            if not 0.0 <= self._target_sparsity <= 1.0:
                raise ValueError(
                    "target_sparsity must be in [0.0, 1.0], got "
                    f"{self._target_sparsity}"
                )

        self.manager: Optional[PruningManager] = None
        self._optimizer: Optional[torch.optim.Optimizer] = None
        self._initialized = False
        self._ckpt_scheduler_state: Optional[dict] = None

        # Per-layer scale factor c_g, keyed by weight-group name; filled at
        # on_fit_start iff groups carry a trainable_scale marker. Non-empty
        # = active.
        self._scale_params: Dict[str, torch.nn.Parameter] = {}
        self._scale_decay: float = 0.0
        # Domain floor on c_g so λ_eff = λ_global · c_g stays ≥ 0.
        self._scale_min: float = 0.0
        # One row per epoch: {"epoch", "step", <layer>: c, ...}. Dumped to
        # SCALE_HISTORY_CSV and persisted in the checkpoint (resume-safe).
        self._scale_history: List[Dict[str, float]] = []

    # -------------------------------------------------------------------------
    # Lightning hooks
    # -------------------------------------------------------------------------

    def on_fit_start(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
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
        self._optimizer = optimizer
        is_resuming = trainer.ckpt_path is not None

        self._verify_gate_target_reachable(pl_module)

        if is_resuming:
            log.info("BregmanPruner: Resuming from checkpoint.")
        else:
            log.info("BregmanPruner: Applying initial sparsity...")
            self.manager.apply_initial_sparsity()

        self._setup_lambda_scheduler(optimizer, trainer, is_resuming)

        # Resolve trainable scales before applying lambda so c (restored, else
        # neutral 1) is already in lambda_scale at batch 0.
        self._setup_trainable_scales(optimizer, pl_module)

        self._apply_lambda_to_groups(trainer)
        if is_resuming and self._ckpt_scheduler_state:
            log.info("Restored lambda values to optimizer parameter groups.")

        self._initialized = True
        self._log_configuration(optimizer)
        self._log_group_assignments(pl_module)

    def on_before_optimizer_step(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        """Inject the closed-form hypergradient into each trainable scale c_g.

        λ acts inside the no-grad prox, so autograd never sees the scale; the
        missing chain-rule link is supplied here. The per-layer threshold is
        ``t = δ·λ_global·c`` (``reg.lamda = λ_global·c``), so ``∂t/∂c = δ·λ_global``
        and for live weights ``∂θ_i/∂c = −δ·λ_global·sign(θ_i)``, giving

            ∂L/∂c = −δ · λ_global · Σ_live grad_i·sign(θ_i)  + scale_decay·(c−1)

        ``scale_decay·(c−1)`` is a soft prior toward c = 1 with finite
        equilibrium ``c* = 1 + δλ_global·signal/scale_decay``. ``λ_global`` comes
        from the scheduler (``reg.lamda`` is undefined per-``c`` once a protected
        layer floors at ``c = 0``); ``sign(0)=0`` drops dead weights.
        """
        if not self._scale_params:
            return
        lam_global = self.lambda_scheduler.get_lambda()
        with torch.no_grad():
            for group in optimizer.param_groups:
                name = group.get("name")
                if name not in self._scale_params:
                    continue
                c = self._scale_params[name]
                delta = group["delta"]
                signal = c.new_zeros(())
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    signal = signal + (p.grad * torch.sign(p)).sum()
                c.grad = -delta * lam_global * signal + self._scale_decay * (
                    c - 1.0
                )

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Update lambda scheduler and log metrics after each batch."""
        if not self._initialized:
            return

        # Floor the just-stepped scale factors and push c into lambda_scale so
        # the scheduler step below sets reg.lamda = λ · c.
        if self._scale_params:
            self._sync_trainable_scales(trainer.optimizers[0])

        if self.lambda_scheduler is not None:
            self._step_lambda_scheduler(trainer)

        # Log metrics via Lightning's logging system (respects logging_params)
        self._log_metrics(pl_module)

        # Last batch: publish overall sparsity before the validation gate reads it.
        if trainer.is_last_batch:
            trainer.callback_metrics["sparsity"] = torch.tensor(
                self._overall_sparsity()
            )

    def on_train_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        """Log epoch-level sparsity and inject into callback_metrics."""
        if not self._initialized:
            return

        sparsity = self._overall_sparsity()
        pruned_sparsity = self._pruned_sparsity()
        target = self._target_sparsity

        # Inject end-of-epoch sparsity directly into callback_metrics so that
        # ModelCheckpoint filenames and train_log.txt get the true final value
        # (not a mean over all steps).
        trainer.callback_metrics["sparsity"] = torch.tensor(sparsity)
        trainer.callback_metrics["bregman/sparsity"] = torch.tensor(sparsity)
        trainer.callback_metrics["bregman/pruned_sparsity"] = torch.tensor(
            pruned_sparsity
        )
        if target is not None:
            trainer.callback_metrics["bregman/target_sparsity"] = torch.tensor(
                target
            )

        if self.verbose > 0:
            target_str = (
                f", target = {target:.3%}" if target is not None else ""
            )
            log.info(
                f"Epoch {trainer.current_epoch}: "
                f"Sparsity = {sparsity:.3%} (pruned = {pruned_sparsity:.3%})"
                f"{target_str}"
            )

        if self._scale_params:
            self._log_scale_table()
            self._record_scale_history(trainer, pl_module)

    def on_save_checkpoint(
        self, trainer: Trainer, pl_module: LightningModule, checkpoint: dict
    ) -> None:
        """Save scheduler state to checkpoint."""
        if self.lambda_scheduler is not None:
            checkpoint[
                "bregman_lambda_scheduler_state"
            ] = self.lambda_scheduler.get_state()
        if self._scale_history:
            checkpoint["bregman_scale_history"] = self._scale_history

    def on_load_checkpoint(
        self, trainer: Trainer, pl_module: LightningModule, checkpoint: dict
    ) -> None:
        """Load scheduler state from checkpoint."""
        self._ckpt_scheduler_state = checkpoint.get(
            "bregman_lambda_scheduler_state",
            checkpoint.get("lambda_scheduler_state"),  # pre-rename compat
        )
        # Absent only for pre-feature ckpts / non-trainable runs; a present but
        # malformed value still raises (list() on a non-iterable).
        if "bregman_scale_history" in checkpoint:
            self._scale_history = list(checkpoint["bregman_scale_history"])

    # -------------------------------------------------------------------------
    # Scheduler management
    # -------------------------------------------------------------------------

    def _setup_lambda_scheduler(
        self, optimizer, trainer: Trainer, is_resuming: bool
    ) -> None:
        """Instantiate and configure the lambda scheduler."""
        if self.lambda_scheduler is None:
            return

        # Configs pass the scheduler as a Hydra partial (_partial_: true).
        if not hasattr(self.lambda_scheduler, "step"):
            self.lambda_scheduler = self.lambda_scheduler()

        if self._target_sparsity is None:
            raise ValueError(
                "BregmanPruner.target_sparsity is required when a "
                "lambda_scheduler is set; it is the controller setpoint."
            )

        # Restore state from checkpoint
        if is_resuming and self._ckpt_scheduler_state:
            self.lambda_scheduler.load_state(self._ckpt_scheduler_state)

        log.info(
            f"Lambda scheduler active: target_sparsity={self._target_sparsity}, "
            f"initial_lambda={self.lambda_scheduler.get_lambda():.4f}"
        )

    def _step_lambda_scheduler(self, trainer: Trainer) -> None:
        """Step the scheduler and update regularizer lambdas."""
        current_sparsity = (
            self._overall_sparsity()
            if WHICH_SPARSITY_PERCENTAGE == "overall"
            else self._pruned_sparsity()
        )

        new_lambda = self.lambda_scheduler.step(
            current_sparsity, self._target_sparsity, trainer.global_step
        )
        self._broadcast_lambda(trainer.optimizers[0], new_lambda)

    def _apply_lambda_to_groups(self, trainer: Trainer) -> None:
        """Apply current scheduler lambda to all regularized groups."""
        if self.lambda_scheduler is None:
            return
        current_lambda = self.lambda_scheduler.get_lambda()
        self._broadcast_lambda(trainer.optimizers[0], current_lambda)

    def _broadcast_lambda(self, optimizer, lambda_value: float) -> None:
        """Set reg.lamda = λ · lambda_scale on every regularized group."""
        for group in optimizer.param_groups:
            if self._group_regularized(group):
                group["reg"].lamda = lambda_value * self._lambda_scale(group)

    # -------------------------------------------------------------------------
    # Trainable per-layer scales (scalar, one per layer)
    # -------------------------------------------------------------------------

    def _setup_trainable_scales(
        self, optimizer, pl_module: LightningModule
    ) -> None:
        """Map each trainable weight group to its scalar scale factor c_g.

        No-op unless groups carry the ``trainable_scale`` marker. One global
        LambdaScheduler owns the sparsity level; each layer's scalar c_g owns
        its share of the allocation, so the scheduler is required.
        """
        trainable_groups = [
            g for g in optimizer.param_groups if g.get("trainable_scale")
        ]
        if not trainable_groups:
            return
        if self.lambda_scheduler is None:
            raise ValueError(
                "trainable per-layer scales require a global lambda_scheduler "
                "(it owns the sparsity level the scales allocate)."
            )
        if not hasattr(pl_module, SCALES_ATTR):
            raise AttributeError(
                f"trainable_scale groups present but pl_module has no "
                f"'{SCALES_ATTR}'; call create_scale_params in setup()."
            )
        scale_dict = getattr(pl_module, SCALES_ATTR)

        self._scale_params = {}
        for group in trainable_groups:
            key = group["trainable_scale_key"]
            if key not in scale_dict:
                raise KeyError(
                    f"trainable_scale group '{group.get('name')}' references "
                    f"missing scale key '{key}'."
                )
            c = scale_dict[key]
            assert c.dim() == 0, (
                f"trainable_scale '{group.get('name')}' expects a scalar "
                f"scale factor, got shape {tuple(c.shape)}."
            )
            self._scale_params[group["name"]] = c

        # decay/floor are uniform across trainable groups; read from one.
        first = trainable_groups[0]
        self._scale_decay = float(first["scale_decay"])
        self._scale_min = float(first["scale_min"])
        assert (
            self._scale_min >= 0.0
        ), f"scale_min must be ≥ 0 (λ_eff ≥ 0), got {self._scale_min}"

        # Sync c into lambda_scale now (before _apply_lambda_to_groups).
        self._sync_trainable_scales(optimizer)
        log.info(
            f"BregmanPruner: trainable per-layer scales active "
            f"({len(self._scale_params)} layers, decay={self._scale_decay}, "
            f"floor={self._scale_min})."
        )

    def _sync_trainable_scales(self, optimizer) -> None:
        """Floor each scale factor in place and fold c into the group's
        ``lambda_scale`` so the scheduler sets reg.lamda = λ_global · c.

        One device->host read for all layers: a single stacked tolist() instead
        of a float() each.
        """
        groups, scales = [], []
        for group in optimizer.param_groups:
            name = group.get("name")
            if name not in self._scale_params:
                continue
            c = self._scale_params[name]
            c.data.clamp_(min=self._scale_min)
            groups.append(group)
            scales.append(c)
        if scales:
            values = torch.stack([c.data for c in scales]).tolist()
            for group, value in zip(groups, values):
                group["lambda_scale"] = value

    def _trainable_scale_values(self) -> Dict[str, float]:
        """Per-layer effective scale factor c as one float."""
        return {name: float(c) for name, c in self._scale_params.items()}

    def _scale_extremes(self) -> tuple:
        """Global (min, max) of c over every trainable scale."""
        vals = self._trainable_scale_values().values()
        return (min(vals), max(vals))

    @rank_zero_only
    def _record_scale_history(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        """Append this epoch's per-layer c, log it, and rewrite the CSV.

        The logger records only the scalar min/max; the full per-layer time
        series goes to the tracker (one series per layer) and to
        SCALE_HISTORY_CSV for src/vis/scale_evolution.py.
        """
        scales = self._trainable_scale_values()
        row: Dict[str, float] = {
            "epoch": trainer.current_epoch,
            "step": trainer.global_step,
        }
        row.update(scales)
        self._scale_history.append(row)
        for name, value in scales.items():
            pl_module.log(
                f"bregman/scale/{name}",
                value,
                on_step=False,
                on_epoch=True,
                sync_dist=False,
            )
        self._write_scale_history_csv(trainer)

    def _write_scale_history_csv(self, trainer: Trainer) -> None:
        """Overwrite the run's scale-history CSV (small; rewrite is crash-
        safe)."""
        out_dir = trainer.default_root_dir
        if not out_dir:
            return
        path = os.path.join(out_dir, SCALE_HISTORY_CSV)
        fieldnames = ["epoch", "step", *self._scale_params.keys()]
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self._scale_history)

    # -------------------------------------------------------------------------
    # Sparsity
    # -------------------------------------------------------------------------

    def _overall_sparsity(self) -> float:
        """Sparsity over all model parameters (true whole-model sparsity)."""
        params = list(self.manager.pl_module.parameters())
        return compute_sparsity(params, threshold=self.sparsity_threshold)

    def _regularized_parameters(self) -> List[torch.Tensor]:
        """Parameters in optimizer groups that carry an active regularizer."""
        assert (
            self._optimizer is not None
        ), "Optimizer must be set to compute pruned sparsity."
        params: List[torch.Tensor] = []
        for group in self._optimizer.param_groups:
            if self._group_has_regularizer(group):
                params.extend(group["params"])
        return params

    def _pruned_sparsity(self) -> float:
        """Sparsity over the regularized (Bregman-pruned) groups only."""
        params = self._regularized_parameters()
        return compute_sparsity(params, threshold=self.sparsity_threshold)

    def _verify_gate_target_reachable(
        self, pl_module: LightningModule
    ) -> None:
        """Fail loud if the gate target can never reopen validation.

        Validation, checkpointing, and early stopping reopen only once overall
        sparsity reaches the target band. Non-regularized groups (norm/bias)
        stay dense, so overall sparsity is capped at prunable/total; a target
        above that ceiling leaves the band unreachable for the whole run.
        """
        if self._target_sparsity is None:
            return
        total = sum(p.numel() for p in pl_module.parameters())
        prunable = sum(p.numel() for p in self._regularized_parameters())
        ceiling = prunable / total if total else 0.0
        assert self._target_sparsity <= ceiling, (
            f"target_sparsity={self._target_sparsity} exceeds the achievable "
            f"overall-sparsity ceiling={ceiling:.4f} "
            f"({prunable}/{total} params prunable); the validation gate could "
            f"never reopen. Lower target_sparsity or regularize more groups."
        )
        if WHICH_SPARSITY_PERCENTAGE != "overall":
            log.warning(
                f"Lambda steers on '{WHICH_SPARSITY_PERCENTAGE}' sparsity "
                f"while the gates measure overall sparsity; the setpoint "
                f"may never enter the validation band (gate target "
                f"{self._target_sparsity})."
            )

    # -------------------------------------------------------------------------
    # Logging
    # -------------------------------------------------------------------------

    def _log_metrics(self, pl_module: LightningModule) -> None:
        """Log sparsity and lambda metrics via Lightning's logging system."""
        default_logging_params = {
            "on_step": False,
            "on_epoch": True,
            "sync_dist": True,
            "prog_bar": False,
        }
        logging_params = getattr(
            pl_module, "logging_params", default_logging_params
        )
        # Per-step only for TensorBoard/WandB tracking; epoch-level "sparsity"
        # is injected in on_train_epoch_end.
        per_step = {**logging_params, "on_step": True, "on_epoch": False}

        pl_module.log("bregman/sparsity", self._overall_sparsity(), **per_step)
        pl_module.log(
            "bregman/pruned_sparsity", self._pruned_sparsity(), **per_step
        )

        if self.lambda_scheduler:
            pl_module.log(
                "bregman/global_lambda",
                self.lambda_scheduler.get_lambda(),
                **per_step,
            )

        if self._scale_params:
            scale_min, scale_max = self._scale_extremes()
            pl_module.log("bregman/scale_min", scale_min, **per_step)
            pl_module.log("bregman/scale_max", scale_max, **per_step)

    @rank_zero_only
    def _log_configuration(self, optimizer) -> None:
        """Log the configuration of all parameter groups."""
        if self.verbose == 0:
            return

        log.info("=== Bregman Configuration ===")
        log.info(f"Optimizer: {type(optimizer).__name__}")

        if self.lambda_scheduler:
            sched_info = (
                f"Lambda Scheduler: target_sparsity={self._target_sparsity}, "
                f"lambda={self.lambda_scheduler.get_lambda():.4f}, "
                f"update_frequency={self.lambda_scheduler.update_frequency}"
            )
            log.info(sched_info)
        else:
            log.info("Lambda Scheduler: None (static lambda mode)")

        for group in optimizer.param_groups:
            name = group.get("name", "unnamed")
            lamda = group["reg"].lamda
            reg_type = type(group["reg"]).__name__
            if not self._group_has_regularizer(group):
                log.info(f"  Group '{name}': {reg_type} (inactive)")
                continue
            scale = self._lambda_scale(group)
            log.info(
                f"  Group '{name}': {reg_type}, lambda={lamda:.4f}, scale={scale}"
            )

            # Non-uniform scaling is the point under trainable allocation, so
            # only warn about a hand-set scale in the other modes.
            if scale != 1.0 and not self._scale_params:
                log.warning(
                    f"Group '{name}' has lambda_scale={scale} != 1.0. "
                    "Non-uniform regularization is generally not recommended."
                )

        if self._scale_params:
            self._log_scale_table()

        log.info("Current sparsity by group:")
        for group in self.manager.processed_groups:
            name = group["config"].get("name", "unnamed")
            sparsity = compute_sparsity(
                group["params"], threshold=self.sparsity_threshold
            )
            str_extras = (
                "(not pruned)" if group["applier"].sparsity_rate == 0.0 else ""
            )
            log.info(f"  {name}: {sparsity:.3%} {str_extras}")

        log.info(
            f"Overall sparsity: {self._overall_sparsity():.3%} "
            f"(pruned only: {self._pruned_sparsity():.3%})"
        )
        log.info("=== End Configuration ===")

    @rank_zero_only
    def _log_scale_table(self) -> None:
        """Log the per-layer effective scale factor c, highest first (most
        pruning pressure), so cross-layer allocation drift is visible."""
        scales = self._trainable_scale_values()
        ordered = sorted(scales.items(), key=lambda kv: kv[1], reverse=True)
        lines = [f"  {name}: {scale:.4f}" for name, scale in ordered]
        log.info(
            "Trainable per-layer scales (c, high pressure first):\n"
            + "\n".join(lines)
        )

    @rank_zero_only
    def _log_group_assignments(self, pl_module: LightningModule) -> None:
        """Log detailed group assignments (for debugging)."""
        if self.verbose < 2:
            return

        param_to_module = {
            id(p): ".".join(name.split(".")[:-1])
            for name, p in pl_module.named_parameters()
        }

        total_params = sum(
            p.numel()
            for group in self.manager.processed_groups
            for p in group["params"]
        )

        log.info("--- Parameter Group Assignments ---")
        for group in self.manager.processed_groups:
            name = group["config"].get("name", "unnamed")
            modules = {param_to_module.get(id(p)) for p in group["params"]}
            modules.discard(None)
            group_params = sum(p.numel() for p in group["params"])
            pct = group_params / total_params * 100 if total_params else 0
            log.info(40 * "-")
            log.info(
                f"  {name}: {len(modules)} modules, "
                f"{group_params:,} params ({pct:.1f}%)"
            )
            if modules:
                for m in sorted(modules):
                    log.info(f"    {m}")
        log.info(40 * "-")

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def _lambda_scale(group: dict) -> float:
        """lambda_scale of a param group; a missing key is a config bug."""
        if "lambda_scale" not in group:
            raise KeyError(
                f"Group '{group.get('name')}' has no lambda_scale; every "
                "Bregman group must set optimizer_settings.lambda_scale."
            )
        return group["lambda_scale"]

    @staticmethod
    def _group_regularized(group: dict) -> bool:
        """A group carrying a thresholding regularizer (non-RegNone with a
        lamda); its reg.lamda tracks λ_global · lambda_scale."""
        return (
            "reg" in group
            and hasattr(group["reg"], "lamda")
            and not isinstance(group["reg"], RegNone)
        )

    @staticmethod
    def _group_has_regularizer(group: dict) -> bool:
        """An actively pruning group: a thresholding regularizer with
        lambda_scale > 0."""
        return (
            BregmanPruner._group_regularized(group)
            and BregmanPruner._lambda_scale(group) > 0.0
        )

    @staticmethod
    def _validate_module(pl_module: LightningModule) -> None:
        """Validate that the module has a pruning_manager."""
        if not hasattr(pl_module, "pruning_manager"):
            raise AttributeError(
                "LightningModule must have a 'pruning_manager' attribute. "
                "Please instantiate it in configure_optimizers()."
            )
        if not isinstance(pl_module.pruning_manager, PruningManager):
            raise TypeError(
                f"pruning_manager must be a PruningManager, got {type(pl_module.pruning_manager)}"
            )
