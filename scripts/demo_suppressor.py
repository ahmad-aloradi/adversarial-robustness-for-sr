"""End-to-end integration demo for ValidationSuppressor.

Drives a real PyTorch Lightning ``Trainer`` with:
  * a tiny MLP module that records every epoch in which validation ran,
  * a FakePruner that plays the role of BregmanPruner/MagnitudePruner —
    it reports a scripted sparsity trajectory and calls the suppressor the
    same way the real pruners do (``prepare`` in ``on_fit_start``, ``gate``
    in ``on_train_epoch_start``),
  * real ``ModelCheckpoint``, ``EarlyStopping``, and a ``ReduceLROnPlateau``
    scheduler monitoring ``val/loss`` — these are exactly the sibling
    callbacks the suppressor has to keep quiet during suppressed epochs.

At the end, asserts that validation ran in exactly the epochs predicted by
the trajectory + tolerance rule.

Run: conda run -n comfort python scripts/demo_suppressor.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import torch
import torch.nn as nn
from pytorch_lightning import Callback, LightningModule, Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader, TensorDataset

from src.callbacks.pruning.shared_prune_utils import ValidationSuppressor


# Scripted per-epoch sparsity values. Target is 0.90, tolerance 0.01 → any
# value in [0.89, 0.91] permits validation; anything else keeps it suppressed.
TRAJECTORY = [
    0.50,   # epoch 0 — far from target, suppressed (case 1/3/5)
    0.70,   # epoch 1 — still suppressed (case 5)
    0.895,  # epoch 2 — within tolerance → restore (case 4)
    0.905,  # epoch 3 — still within → stays restored (case 6)
    0.70,   # epoch 4 — drifts out → re-suppress (case 7)
    0.895,  # epoch 5 — back within → restore again
    0.9009,  # epoch 6 — back within → restore again
    0.92,  # epoch 7 — still within → stays restored (case 7)
    0.88,  # epoch 8 — still within → stays restored (case 7)
]
TARGET = 0.90
TOLERANCE = 0.01


class TinyModel(LightningModule):
    """Records which epochs actually fired validation_step."""

    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(4, 2)
        self.val_epochs_run: list[int] = []
        self.sanity_hit = False

    def forward(self, x):
        return self.layer(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = nn.functional.mse_loss(self(x), y)
        self.log("train/loss", loss, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        loss = nn.functional.mse_loss(self(x), y)
        self.log("val/loss", loss, on_epoch=True)
        if self.trainer.sanity_checking:
            self.sanity_hit = True
        else:
            epoch = self.current_epoch
            if epoch not in self.val_epochs_run:
                self.val_epochs_run.append(epoch)
        return loss

    def configure_optimizers(self):
        opt = torch.optim.SGD(self.parameters(), lr=0.01)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", patience=2
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": sched,
                "monitor": "val/loss",
                "strict": True,  # prepare() flips this to False
            },
        }


class FakePruner(Callback):
    """Minimal stand-in for BregmanPruner/MagnitudePruner.

    Reports a scripted per-epoch sparsity and calls the same suppressor API
    (``prepare`` + ``gate``) the real pruners use.
    """

    def __init__(self, trajectory, target, tolerance):
        super().__init__()
        self.trajectory = trajectory
        self.target = target
        self.suppressor = ValidationSuppressor(tolerance=tolerance)

    def _current(self, epoch: int) -> float:
        return self.trajectory[min(epoch, len(self.trajectory) - 1)]

    def on_fit_start(self, trainer, pl_module):
        ValidationSuppressor.prepare(trainer)
        trainer.limit_val_batches = 0

    def on_train_epoch_start(self, trainer, pl_module):
        current = self._current(trainer.current_epoch)
        self.suppressor.gate(trainer, current, self.target)
        print(
            f"  [epoch {trainer.current_epoch}] sparsity={current:.3f}"
            f" → limit_val_batches={trainer.limit_val_batches}"
        )


def main() -> None:
    torch.manual_seed(0)

    # Deterministic toy dataset.
    xs = torch.randn(32, 4)
    ys = torch.randn(32, 2)
    loader = DataLoader(TensorDataset(xs, ys), batch_size=8, shuffle=False)

    model = TinyModel()
    pruner = FakePruner(TRAJECTORY, TARGET, TOLERANCE)

    with tempfile.TemporaryDirectory() as tmp:
        ckpt = ModelCheckpoint(
            dirpath=tmp,
            monitor="val/loss",
            mode="min",
            save_top_k=1,
            filename="best-{epoch}",
        )
        es = EarlyStopping(monitor="val/loss", mode="min", patience=10)

        trainer = Trainer(
            max_epochs=len(TRAJECTORY),
            callbacks=[pruner, ckpt, es],
            num_sanity_val_steps=2,  # prepare() should zero this
            enable_progress_bar=False,
            enable_model_summary=False,
            logger=False,
            log_every_n_steps=1,
        )
        print(f"Trajectory: {TRAJECTORY}")
        print(f"Target: {TARGET}, tolerance: {TOLERANCE}\n")
        trainer.fit(
            model=model, train_dataloaders=loader, val_dataloaders=loader
        )

        checkpoints = list(Path(tmp).glob("*.ckpt"))

    # ---- Checks ------------------------------------------------------------
    expected = sorted(
        # i for i, s in enumerate(TRAJECTORY) if abs(s - TARGET) <= TOLERANCE
        i for i, s in enumerate(TRAJECTORY) if abs(s - TARGET) <= TOLERANCE

    )
    observed = sorted(model.val_epochs_run)

    print()
    print(f"Sanity check fired:       {model.sanity_hit}   (expected False)")
    print(f"Validation-ran epochs:    {observed}")
    print(f"Expected from trajectory: {expected}")
    print(f"Checkpoint(s) written:    {[c.name for c in checkpoints]}")

    assert not model.sanity_hit, "prepare() failed to skip sanity check"
    assert observed == expected, (
        f"Suppressor gating mismatch: ran {observed}, expected {expected}"
    )
    # A best checkpoint should exist only if val ran at least once.
    if expected:
        assert checkpoints, "Expected a best-checkpoint file when val ran"
    else:
        assert not checkpoints, "Did not expect a checkpoint when val never ran"

    print("\nSuppressor behaved exactly as intended.")


if __name__ == "__main__":
    main()
