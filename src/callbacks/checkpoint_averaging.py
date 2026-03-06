from __future__ import annotations

from pathlib import Path

import torch
from pytorch_lightning import Callback, LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities import rank_zero_only

from src import utils

log = utils.get_pylogger(__name__)


class CheckpointAveraging(Callback):
    """Average top-k checkpoints at the end of fit.

    This uses PyTorch Lightning's callback hooks and the ModelCheckpoint
    callback to locate the best checkpoints, then writes an averaged
    checkpoint.
    """

    def __init__(
        self,
        num_checkpoints: int | None = 10,
        output_filename: str = "averaged_top{num}.ckpt",
    ) -> None:
        super().__init__()
        self.num_checkpoints = num_checkpoints
        self.output_filename = output_filename
        self.averaged_ckpt_path: str | None = None

    @staticmethod
    def _find_model_checkpoint(trainer: Trainer) -> ModelCheckpoint | None:
        if getattr(trainer, "checkpoint_callback", None) is not None:
            return trainer.checkpoint_callback

        for cb in trainer.callbacks:
            if isinstance(cb, ModelCheckpoint):
                return cb
        return None

    @staticmethod
    def _sort_best_k(
        best_k_models: dict[str, torch.Tensor],
        mode: str,
    ) -> list[tuple[str, float]]:
        items: list[tuple[str, float]] = []
        for path, score in best_k_models.items():
            if not path:
                continue
            if score is None:
                continue
            items.append((path, float(score)))

        reverse = str(mode).lower() == "max"
        return sorted(items, key=lambda x: x[1], reverse=reverse)

    def average_checkpoints(self, ckpt_paths: list[Path], output_path: Path) -> str:
        """Average the given checkpoints and save to output_path."""
        return utils.average_checkpoints(ckpt_paths, output_path)

    @rank_zero_only
    def on_fit_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        ckpt_cb = self._find_model_checkpoint(trainer)
        if ckpt_cb is None:
            log.warning(
                "Checkpoint averaging skipped: no ModelCheckpoint callback found."
            )
            return

        if not ckpt_cb.best_k_models:
            log.warning(
                "Checkpoint averaging skipped: no best_k_models recorded."
            )
            return

        sorted_items = self._sort_best_k(ckpt_cb.best_k_models, ckpt_cb.mode)
        if not sorted_items:
            log.warning(
                "Checkpoint averaging skipped: best_k_models empty after filtering."
            )
            return

        if self.num_checkpoints is not None:
            sorted_items = sorted_items[: self.num_checkpoints]

        ckpt_paths = [
            Path(p).expanduser() for p, _ in sorted_items if Path(p).exists()
        ]
        if not ckpt_paths:
            log.warning(
                "Checkpoint averaging skipped: no checkpoint files found on disk."
            )
            return

        log.info(
            f"Averaging {len(ckpt_paths)} checkpoint(s): "
            + ", ".join(p.name for p in ckpt_paths)
        )

        # Determine output path
        output_dir = Path(ckpt_cb.dirpath or trainer.default_root_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            output_name = self.output_filename.format(num=len(ckpt_paths))
        except Exception:
            output_name = self.output_filename

        output_path = output_dir / output_name
        self.averaged_ckpt_path = self.average_checkpoints(ckpt_paths, output_path)
        log.info(f"Saved averaged checkpoint to: {output_path}")
