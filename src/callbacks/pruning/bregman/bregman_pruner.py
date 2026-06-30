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

# Water-filling allocation: a survivor with 0 < |θ| ≤ SCALE_BAND·t counts toward
# the kill-rate κ_g = n_band/(SCALE_BAND·t); ρ_g = −S_g/κ_g equalizes at optimum.
SCALE_BAND: float = 0.25
# Numeric guard on the log-scale s_g (c = e^s ∈ [~1e-13, 1e13]); not a floor.
SCALE_CLAMP: float = 30.0


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
        scale_update_freq: int = 10,  # water-filling gradient period (O(params) per call)
        kill_rate: Literal["hard_band", "soft"] = "hard_band",  # κ estimator
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
            kill_rate: Per-layer kill-rate κ_g estimator for the allocation.
                "hard_band" counts survivors in (0, SCALE_BAND·t] and freezes a
                layer with none. "soft" uses κ_g = Σ_surv exp(−|θ|/h)/h with
                h = SCALE_BAND·t, so a layer freezes only when it has no
                survivor at all (avoids the spurious dead-layer freeze that
                stalls allocation at high sparsity).
        """
        super().__init__()
        if kill_rate not in ("hard_band", "soft"):
            raise ValueError(
                f"kill_rate must be 'hard_band' or 'soft', got {kill_rate!r}."
            )
        self.kill_rate = kill_rate
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

        self.scale_update_freq = scale_update_freq
        self.manager: Optional[PruningManager] = None
        self._optimizer: Optional[torch.optim.Optimizer] = None
        self._initialized = False
        self._ckpt_scheduler_state: Optional[dict] = None
        self._scales_need_sync: bool = False

        # Per-layer log-scale s_g (c_g = exp(s_g)), keyed by weight-group name;
        # filled at on_fit_start iff groups carry a trainable_scale marker.
        # Non-empty = active.
        self._scale_params: Dict[str, torch.nn.Parameter] = {}
        # Live (non-dead) scale groups from the last allocation step; the gauge
        # re-centering and the ρ̄ injection exclude dead/frozen layers.
        self._live_scale_names: set = set()
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

        if not trainer.optimizers:
            raise ValueError("BregmanPruner: No optimizers found.")
        if len(trainer.optimizers) > 1:
            raise ValueError("BregmanPruner supports only a single optimizer.")

        # Lightning rebuilds the optimizer on every fit(); refresh the reference
        # even when already initialized so sparsity reads live param tensors.
        optimizer = trainer.optimizers[0]
        self._optimizer = optimizer

        if self._initialized:
            return

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
        """Inject the water-filling allocation gradient into each log-scale s_g.

        λ acts inside the no-grad prox, so autograd never sees the scale; the
        link is supplied here. For layer g with c_g = exp(s_g) and threshold
        ``t = δ·λ_global·c_g`` (``reg.lamda = λ_global·c_g``):

            S_g = Σ_live grad_i·sign(θ_i)        (loss change per unit shrink)
            κ_g = kill_qty / (SCALE_BAND·t)       (kill-rate; see kill_rate arg)
            ρ_g = −S_g / κ_g                      (marginal loss per killed weight)
            s_g.grad = ρ_g − mean_live(ρ)         (gauge-projected, budget-neutral)

        ``sign(0)=0`` drops dead weights from S_g. A dead layer (hard_band: no
        in-band survivor; soft: no survivor at all) is frozen (``s.grad = None``)
        and excluded from the live mean, else its centered gradient diverges.
        The stationary point ρ_g = μ ∀ live g is the water-filling optimum.
        """
        if not self._scale_params:
            return
        global_step = 0 if trainer is None else trainer.global_step
        if global_step % self.scale_update_freq != 0:
            self._scales_need_sync = False
            return
        self._scales_need_sync = True
        lam_global = self.lambda_scheduler.get_lambda()
        with torch.no_grad():
            # Two passes so the per-group kill quantities read back in one
            # stacked tolist() (one device->host sync), not a float() per group.
            pending, kill_qtys, alive_counts = [], [], []
            for group in optimizer.param_groups:
                name = group.get("name")
                if name not in self._scale_params:
                    continue
                s = self._scale_params[name]
                h = SCALE_BAND * group["delta"] * lam_global * torch.exp(s)
                signal = s.new_zeros(())
                kill_qty = s.new_zeros(())  # κ_g·h: band count or kernel sum
                n_alive = s.new_zeros(())
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    signal = signal + (p.grad * torch.sign(p)).sum()
                    abs_p = p.abs()
                    alive = abs_p > self.sparsity_threshold
                    n_alive = n_alive + alive.sum()
                    if self.kill_rate == "hard_band":
                        kill_qty = kill_qty + (alive & (abs_p <= h)).sum()
                    else:  # survivor-kernel density at 0 (no spurious freeze)
                        kill_qty = kill_qty + torch.exp(-abs_p[alive] / h).sum()
                pending.append((name, s, signal, h))
                kill_qtys.append(kill_qty)
                alive_counts.append(n_alive)

            qtys = torch.stack(kill_qtys).tolist() if kill_qtys else []
            alives = torch.stack(alive_counts).tolist() if alive_counts else []
            rho: Dict[str, torch.Tensor] = {}
            for (name, s, signal, h), qty, n_alive in zip(
                pending, qtys, alives
            ):
                # Dead: hard_band has no in-band survivor; soft has none at all.
                dead = (
                    n_alive == 0.0
                    if self.kill_rate == "soft"
                    else qty == 0.0
                )
                if dead:
                    s.grad = None  # freeze, drop from the live mean
                    continue
                kappa = qty / h  # κ_g
                rho[name] = -signal / kappa

            self._live_scale_names = set(rho)
            if not rho:
                return
            rho_bar = torch.stack(list(rho.values())).mean()
            for name, rho_g in rho.items():
                self._scale_params[name].grad = rho_g - rho_bar

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

        # Re-center and fold c = exp(s) into lambda_scale only when scales were
        # actually stepped this batch (avoids a GPU→CPU sync on skipped steps).
        if self._scale_params and self._scales_need_sync:
            self._sync_trainable_scales(trainer.optimizers[0])

        # Whole-model sparsity is the dominant per-batch cost; compute once and
        # share it with the scheduler step, the metric logging, and the gate.
        overall_sparsity = self._overall_sparsity()

        if self.lambda_scheduler is not None:
            self._step_lambda_scheduler(trainer, overall_sparsity)

        # Log metrics via Lightning's logging system (respects logging_params)
        self._log_metrics(pl_module, overall_sparsity)

        # Last batch: publish overall sparsity before the validation gate reads it.
        if trainer.is_last_batch:
            trainer.callback_metrics["sparsity"] = torch.tensor(
                overall_sparsity
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
        pl_module.log(
            "bregman/pruned_sparsity",
            pruned_sparsity,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            prog_bar=False,
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
            checkpoint["bregman_lambda_scheduler_state"] = (
                self.lambda_scheduler.get_state()
            )
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

    def _step_lambda_scheduler(
        self, trainer: Trainer, overall_sparsity: float
    ) -> None:
        """Step the scheduler and update regularizer lambdas."""
        current_sparsity = (
            overall_sparsity
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

        # Populated by the first allocation step; left empty here so the eager
        # sync folds exp(s) into lambda_scale without centering on dead layers.
        self._live_scale_names = set()

        # Sync c = exp(s) into lambda_scale now (before _apply_lambda_to_groups).
        self._sync_trainable_scales(optimizer)
        log.info(
            f"BregmanPruner: trainable per-layer scales active "
            f"({len(self._scale_params)} layers, water-filling allocation, "
            f"band={SCALE_BAND})."
        )

    def _sync_trainable_scales(self, optimizer) -> None:
        """Re-center the live log-scales to zero mean (the gauge Σ_live s = 0),
        clamp s to the numeric range, and fold c = exp(s) into each group's
        ``lambda_scale`` so the scheduler sets reg.lamda = λ_global · c.

        Dead/frozen layers are excluded from the re-centering (a pinned dead s
        would otherwise offset the mean and distort every live allocation).
        """
        live_s = [
            self._scale_params[name]
            for name in self._scale_params
            if name in self._live_scale_names
        ]
        if live_s:
            mean_s = torch.stack([s.data for s in live_s]).mean()
            for s in live_s:
                s.data.sub_(mean_s)

        groups, scales = [], []
        for group in optimizer.param_groups:
            name = group.get("name")
            if name not in self._scale_params:
                continue
            s = self._scale_params[name]
            s.data.clamp_(-SCALE_CLAMP, SCALE_CLAMP)
            groups.append(group)
            scales.append(s)
        if scales:
            values = torch.stack([torch.exp(s.data) for s in scales]).tolist()
            for group, value in zip(groups, values):
                group["lambda_scale"] = value

    def _trainable_scale_values(self) -> Dict[str, float]:
        """Per-layer effective scale factor c = exp(s) as one float."""
        return {
            name: float(torch.exp(s)) for name, s in self._scale_params.items()
        }

    def _live_scale_params(self) -> List[torch.nn.Parameter]:
        """The log-scales the last allocation moved. Frozen dead layers sit at a
        stale s and would skew (min, max, std), so the scalar diagnostics read
        only these. Falls back to all before the first allocation step."""
        live = self._live_scale_names or set(self._scale_params)
        return [s for name, s in self._scale_params.items() if name in live]

    def _scale_extremes(self) -> tuple:
        """Global (min, max) of c = exp(s) over the live trainable scales."""
        vals = [float(torch.exp(s)) for s in self._live_scale_params()]
        return (min(vals), max(vals))

    def _scale_spread(self) -> torch.Tensor:
        """Std of the live log-scales s — how far the per-layer allocation has
        spread from neutral. Diagnostic only."""
        with torch.no_grad():
            scales = torch.stack(self._live_scale_params())
            if scales.numel() < 2:
                return scales.new_zeros(())
            return scales.std()

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

    def _log_metrics(
        self, pl_module: LightningModule, overall_sparsity: float
    ) -> None:
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
        # Per-step only for TensorBoard/WandB tracking; epoch-level values are
        # injected in on_train_epoch_end (pruned_sparsity there avoids a full
        # scan every batch).
        per_step = {**logging_params, "on_step": True, "on_epoch": False}

        pl_module.log("bregman/sparsity", overall_sparsity, **per_step)

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
            pl_module.log(
                "bregman/scale_spread", self._scale_spread(), **per_step
            )

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
