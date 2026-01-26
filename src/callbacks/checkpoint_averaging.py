from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple, List

import torch
from pytorch_lightning import Callback, Trainer, LightningModule
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities import rank_zero_only

from src import utils


log = utils.get_pylogger(__name__)


class CheckpointAveraging(Callback):
    """Average top-k checkpoints at the end of fit.

    This uses PyTorch Lightning's callback hooks and the ModelCheckpoint
    callback to locate the best checkpoints, then writes an averaged checkpoint.
    """

    def __init__(
        self,
        num_checkpoints: Optional[int] = 10,
        output_filename: str = "averaged_top{num}.ckpt",
    ) -> None:
        super().__init__()
        self.num_checkpoints = num_checkpoints
        self.output_filename = output_filename

    @staticmethod
    def _find_model_checkpoint(trainer: Trainer) -> Optional[ModelCheckpoint]:
        if getattr(trainer, "checkpoint_callback", None) is not None:
            return trainer.checkpoint_callback

        for cb in trainer.callbacks:
            if isinstance(cb, ModelCheckpoint):
                return cb
        return None

    @staticmethod
    def _sort_best_k(
        best_k_models: Dict[str, torch.Tensor],
        mode: str,
    ) -> List[Tuple[str, float]]:
        items: List[Tuple[str, float]] = []
        for path, score in best_k_models.items():
            if not path:
                continue
            if score is None:
                continue
            items.append((path, float(score)))

        reverse = str(mode).lower() == "max"
        return sorted(items, key=lambda x: x[1], reverse=reverse)

    @rank_zero_only
    def on_fit_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        ckpt_cb = self._find_model_checkpoint(trainer)
        if ckpt_cb is None:
            log.warning("Checkpoint averaging skipped: no ModelCheckpoint callback found.")
            return

        if not ckpt_cb.best_k_models:
            log.warning("Checkpoint averaging skipped: no best_k_models recorded.")
            return

        sorted_items = self._sort_best_k(ckpt_cb.best_k_models, ckpt_cb.mode)
        if not sorted_items:
            log.warning("Checkpoint averaging skipped: best_k_models empty after filtering.")
            return

        if self.num_checkpoints is not None:
            sorted_items = sorted_items[: self.num_checkpoints]

        ckpt_paths = [Path(p).expanduser() for p, _ in sorted_items if Path(p).exists()]
        if not ckpt_paths:
            log.warning("Checkpoint averaging skipped: no checkpoint files found on disk.")
            return

        log.info(
            f"Averaging {len(ckpt_paths)} checkpoint(s): "
            + ", ".join(p.name for p in ckpt_paths)
        )

        avg_state: Dict[str, torch.Tensor] = {}
        float_keys: List[str] = []
        num = len(ckpt_paths)

        # Load first checkpoint to initialize
        first_ckpt = torch.load(ckpt_paths[0], map_location="cpu")
        state_dict = first_ckpt.get("state_dict", {})
        for k, v in state_dict.items():
            if torch.is_tensor(v) and torch.is_floating_point(v):
                avg_state[k] = v.clone().float()
                float_keys.append(k)
            else:
                avg_state[k] = v.clone() if torch.is_tensor(v) else v

        # Accumulate remaining checkpoints
        for ckpt_path in ckpt_paths[1:]:
            ckpt = torch.load(ckpt_path, map_location="cpu")
            state_dict = ckpt.get("state_dict", {})
            for k in float_keys:
                if k in state_dict and torch.is_tensor(state_dict[k]):
                    avg_state[k] += state_dict[k].float()

        # Average
        for k in float_keys:
            avg_state[k] = avg_state[k] / float(num)

        # Save averaged checkpoint
        output_dir = Path(ckpt_cb.dirpath or trainer.default_root_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            output_name = self.output_filename.format(num=num)
        except Exception:
            output_name = self.output_filename

        output_path = output_dir / output_name
        first_ckpt["state_dict"] = avg_state
        torch.save(first_ckpt, output_path)

        log.info(f"Saved averaged checkpoint to: {output_path}")