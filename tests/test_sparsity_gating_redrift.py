"""Re-drift integration test: sparsity can leave the band AFTER reaching it.

Drives the real RampValidationGate + SparsityGatedModelCheckpoint through a full
Trainer over a scripted sparsity trajectory out -> in -> in -> OUT -> out, and
asserts the gates re-suppress on the re-drift: validation runs and a top-k slot
is won only on the in-band epochs.
"""

import os
import re
import tempfile

import torch
from pytorch_lightning import Callback, LightningModule, Trainer
from torch.utils.data import DataLoader, TensorDataset

from src.callbacks.pruning.sparsity_gating import (
    RampValidationGate,
    SparsityGatedModelCheckpoint,
)

TARGET, TOL = 0.9, 0.01
# in-band only at epochs 2,3 and 5,6; off-target (re-drift) at 4.
SEQ = [0.5, 0.92, 0.89, 0.91, 0.5, 0.9, 0.895]
IN_BAND = {2, 3, 5, 6}


class _ScriptedPruner(Callback):
    """Publishes scripted overall sparsity at the last batch.

    Mirrors the real pruner's publish-only path; registered before the gate so
    the metric is fresh when the gate reads it.
    """

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, idx):
        if trainer.is_last_batch:
            trainer.callback_metrics["sparsity"] = torch.tensor(
                SEQ[trainer.current_epoch]
            )


class _TinyModel(LightningModule):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Linear(4, 1)
        self.val_epochs = []

    def training_step(self, batch, _):
        x, y = batch
        return torch.nn.functional.mse_loss(self.net(x), y)

    def validation_step(self, batch, _):
        self.val_epochs.append(self.trainer.current_epoch)
        x, y = batch
        self.log("val_loss", torch.nn.functional.mse_loss(self.net(x), y))

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.1)


def _loader():
    data = TensorDataset(torch.randn(8, 4), torch.randn(8, 1))
    return DataLoader(data, batch_size=8)


def test_gates_resuppress_after_sparsity_leaves_band():
    with tempfile.TemporaryDirectory() as tmp:
        checkpoint = SparsityGatedModelCheckpoint(
            target_sparsity=TARGET,
            tolerance=TOL,
            dirpath=os.path.join(tmp, "ckpt"),
            monitor="val_loss",
            mode="min",
            save_top_k=5,
            save_last=True,
        )
        gate = RampValidationGate(target_sparsity=TARGET, tolerance=TOL)
        model = _TinyModel()
        trainer = Trainer(
            default_root_dir=tmp,
            max_epochs=len(SEQ),
            accelerator="cpu",
            devices=1,
            num_sanity_val_steps=0,
            enable_progress_bar=False,
            enable_model_summary=False,
            logger=False,
            callbacks=[_ScriptedPruner(), gate, checkpoint],
        )
        trainer.fit(model, _loader(), _loader())

        saved = {
            int(re.search(r"epoch=?(\d+)", p).group(1))
            for p in checkpoint.best_k_models
        }
        # Validation ran only on in-band epochs (re-suppressed at the re-drift).
        assert set(model.val_epochs) == IN_BAND, model.val_epochs
        # No off-target epoch (incl. the re-drift) won a top-k slot.
        assert saved == IN_BAND, saved
