"""Top-1 accuracy over a [0,1]-image loader.

Used for the clean sanity pass and for every corruption severity. The model
normalizes internally (see ``NormalizedModel``), so loaders stay in [0,1].
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def evaluate_accuracy(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device | str,
) -> float:
    """Top-1 accuracy of ``model`` on ``loader``."""
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, targets in loader:
            assert (
                images.ndim == 4
            ), f"expected image batches, got {images.shape}"
            logits = model(images.to(device))
            preds = logits.argmax(dim=1).cpu()
            correct += int((preds == targets).sum())
            total += targets.shape[0]
    assert total > 0, "empty loader"
    return correct / total


if __name__ == "__main__":
    from torch.utils.data import TensorDataset

    torch.manual_seed(0)
    x = torch.rand(64, 3, 8, 8)
    y = torch.randint(0, 10, (64,))
    loader = DataLoader(TensorDataset(x, y), batch_size=16, shuffle=False)
    net = nn.Sequential(nn.Flatten(), nn.Linear(3 * 8 * 8, 10)).eval()
    print("accuracy:", evaluate_accuracy(net, loader, "cpu"))
