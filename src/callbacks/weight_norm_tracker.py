"""Tracks weight magnitude metrics per epoch for comparing optimizer behavior.

Logs per-group L2 norms, sparsity, BN gamma statistics, embedding norms,
and total model norm. Writes to both the Lightning logger and a CSV file.
"""

import csv
import math
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn as nn
from pytorch_lightning import Callback, LightningModule, Trainer
from pytorch_lightning.utilities import rank_zero_only

from src.callbacks.pruning.shared_prune_utils import compute_sparsity
from src.utils import get_pylogger

logger = get_pylogger(__name__)


def _classify_module(module: nn.Module) -> str:
    """Classify a module into a parameter group name."""
    if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        return "conv_layers"
    if isinstance(module, nn.Linear):
        return "linear_layers"
    if isinstance(
        module,
        (
            nn.BatchNorm1d,
            nn.BatchNorm2d,
            nn.BatchNorm3d,
            nn.GroupNorm,
            nn.LayerNorm,
            nn.InstanceNorm1d,
            nn.InstanceNorm2d,
            nn.InstanceNorm3d,
        ),
    ):
        return "norm_params"
    return "fallback"


class WeightNormTracker(Callback):
    """Tracks weight magnitude metrics each epoch.

    Metrics tracked:
    - Per-group weight L2 norm and sparsity
    - Classifier weight norm
    - BN gamma geometric mean, max, min
    - Embedding mean L2 norm (from validation batches)
    - Total model L2 norm
    """

    def __init__(
        self,
        filename: str = "weight_norms.csv",
        sparsity_threshold: float = 1e-12,
        embedding_sample_batches: int = 3,
    ):
        self.filename = filename
        self.sparsity_threshold = sparsity_threshold
        self.embedding_sample_batches = embedding_sample_batches

        # Accumulate embedding norms during validation
        self._embedding_norm_sum = 0.0
        self._embedding_count = 0

    def on_validation_epoch_start(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        self._embedding_norm_sum = 0.0
        self._embedding_count = 0

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if batch_idx >= self.embedding_sample_batches:
            return
        if outputs is None:
            return

        # validation_step returns {"loss": ..., "outputs": {"embeds": ...}}
        inner = (
            outputs.get("outputs")
            if isinstance(outputs, dict)
            else getattr(outputs, "outputs", None)
        )
        if inner is None:
            return
        embeds = (
            inner.get("embeds")
            if isinstance(inner, dict)
            else getattr(inner, "embeds", None)
        )
        if embeds is None:
            return

        # embeds shape: (batch, embed_dim)
        norms = embeds.detach().float().norm(dim=-1)  # (batch,)
        self._embedding_norm_sum += norms.sum().item()
        self._embedding_count += norms.numel()

    @rank_zero_only
    def on_validation_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        metrics = self._compute_metrics(pl_module)
        self._log_metrics(trainer, pl_module, metrics)
        self._write_csv(trainer, metrics)

    def _get_groups(
        self, pl_module: LightningModule
    ) -> Dict[str, List[nn.Parameter]]:
        """Get parameter groups either from pruning_manager or auto-classify."""
        manager = getattr(pl_module, "pruning_manager", None)
        if manager is not None and hasattr(manager, "processed_groups"):
            groups: Dict[str, List[nn.Parameter]] = {}
            for group in manager.processed_groups:
                name = group["config"].get("name", "unnamed")
                groups.setdefault(name, []).extend(group["params"])
            return groups

        # Auto-classify by walking named_modules
        groups = {}
        seen_param_ids = set()
        for name, module in pl_module.named_modules():
            group_name = _classify_module(module)
            for pname, param in module.named_parameters(recurse=False):
                if id(param) in seen_param_ids:
                    continue
                seen_param_ids.add(id(param))
                if pname == "bias":
                    groups.setdefault("bias_params", []).append(param)
                else:
                    groups.setdefault(group_name, []).append(param)
        return groups

    def _compute_metrics(
        self, pl_module: LightningModule
    ) -> Dict[str, float]:
        metrics: Dict[str, float] = {}

        groups = self._get_groups(pl_module)
        for group_name, params in groups.items():
            # L2 norm
            l2 = math.sqrt(
                sum(p.detach().float().pow(2).sum().item() for p in params)
            )
            metrics[f"{group_name}/l2_norm"] = l2
            # Sparsity
            sparsity = compute_sparsity(
                params, threshold=self.sparsity_threshold
            )
            metrics[f"{group_name}/sparsity"] = sparsity

        # Classifier weight norm
        classifier = getattr(pl_module, "classifier", None)
        if classifier is not None:
            w = getattr(classifier, "weight", None)
            if w is None:
                # Try nested — some classifiers wrap a Linear
                for m in classifier.modules():
                    if isinstance(m, nn.Linear) and hasattr(m, "weight"):
                        w = m.weight
                        break
            if w is not None:
                metrics["classifier/weight_l2_norm"] = (
                    w.detach().float().norm().item()
                )

        # BN gamma stats
        bn_gammas = []
        for module in pl_module.modules():
            if isinstance(
                module,
                (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d),
            ) and module.affine:
                bn_gammas.append(module.weight.detach().float())

        if bn_gammas:
            all_gammas = torch.cat([g.flatten() for g in bn_gammas])
            abs_gammas = all_gammas.abs().clamp(min=1e-30)
            metrics["bn_gamma/geo_mean"] = abs_gammas.log().mean().exp().item()
            metrics["bn_gamma/max"] = all_gammas.max().item()
            metrics["bn_gamma/min"] = all_gammas.min().item()

        # Embedding mean L2 norm
        if self._embedding_count > 0:
            metrics["embedding/mean_l2_norm"] = (
                self._embedding_norm_sum / self._embedding_count
            )

        # Total model L2 norm
        total_l2 = math.sqrt(
            sum(
                p.detach().float().pow(2).sum().item()
                for p in pl_module.parameters()
            )
        )
        metrics["model/total_l2_norm"] = total_l2

        return metrics

    def _log_metrics(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        metrics: Dict[str, float],
    ) -> None:
        prefixed = {
            f"weight_norms/{k}": v for k, v in metrics.items()
        }
        # Log directly to logger — pl_module.log_dict() is not allowed
        # inside on_validation_end per PyTorch Lightning restrictions.
        if trainer.logger:
            trainer.logger.log_metrics(prefixed, step=trainer.global_step)

    def _write_csv(
        self, trainer: Trainer, metrics: Dict[str, float]
    ) -> None:
        out_dir = Path(trainer.default_root_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / self.filename

        row = {"epoch": trainer.current_epoch, **metrics}
        file_exists = path.exists()
        with open(path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)
