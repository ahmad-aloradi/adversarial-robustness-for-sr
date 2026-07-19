"""Generic torchattacks adapter.

Any attack class in ``torchattacks`` can be configured by name + kwargs
(``robustness.attacks.<key>.name`` / ``.kwargs``); nothing here is
AutoAttack-specific. The model must accept [0,1] images (see
``NormalizedModel``) — that is the torchattacks input contract.

Adversarial attacks need NO training: they perturb test inputs against the
frozen checkpoint at evaluation time.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

_INSTALL_HINT = (
    "torchattacks is not installed. Run: pip install torchattacks "
    "(see requirements.txt for the pinned version)."
)

# Original AutoAttack convention: APGD-T/FAB-T restrict to 9 target classes
# regardless of dataset size. torchattacks 3.5.1 instead sets
# n_target_classes = n_classes - 1 (verified: 199 on TinyImageNet), which
# would multiply the targeted stages 10-20x on CIFAR-100/TinyImageNet; the
# adapter caps it back to the published convention.
_MAX_TARGET_CLASSES = 9


def _apply_autoattack_target_cap(attack) -> None:
    multi = getattr(attack, "_autoattack", None)
    assert multi is not None and hasattr(multi, "attacks"), (
        "torchattacks AutoAttack internals changed; re-verify the "
        "n_target_classes cap against the installed version."
    )
    for sub in multi.attacks:
        if getattr(sub, "n_target_classes", None) is not None:
            sub.n_target_classes = min(
                sub.n_target_classes, _MAX_TARGET_CLASSES
            )


def evaluate_attack(
    model: nn.Module,
    loader: DataLoader,
    attack_name: str,
    attack_kwargs: dict,
    n_examples: int | None,
    device: torch.device | str,
    n_save_samples: int = 0,
) -> dict:
    """Robust accuracy of ``model`` under ``torchattacks.<attack_name>``.

    ``n_examples=None`` means the full loader; otherwise evaluation stops
    once that many examples were attacked (whole batches, truncated on the
    last one). Returns ``{"accuracy", "n_examples"}``.

    ``n_save_samples>0`` also returns ``"samples"`` — a dict of CPU tensors
    ``{"clean", "adv", "labels"}`` holding the first that-many attacked
    examples (in [0,1] pixel space) for later inspection.
    """
    assert n_save_samples >= 0
    try:
        import torchattacks
    except ImportError as exc:  # pragma: no cover - env-dependent
        raise ImportError(_INSTALL_HINT) from exc

    assert hasattr(
        torchattacks, attack_name
    ), f"torchattacks has no attack {attack_name!r}"
    assert n_examples is None or n_examples > 0

    model.eval()
    attack = getattr(torchattacks, attack_name)(model, **attack_kwargs)
    attack.set_device(device)
    if attack_name == "AutoAttack":
        _apply_autoattack_target_cap(attack)

    eps = attack_kwargs.get("eps")
    is_linf = str(attack_kwargs.get("norm", "Linf")) == "Linf"

    saved_clean, saved_adv, saved_labels = [], [], []
    n_saved = 0

    correct = total = 0
    for images, targets in loader:
        if n_examples is not None and total + images.shape[0] > n_examples:
            images = images[: n_examples - total]
            targets = targets[: n_examples - total]
        images, targets = images.to(device), targets.to(device)

        adv = attack(images, targets).detach()

        assert (
            float(adv.min()) >= -1e-6 and float(adv.max()) <= 1 + 1e-6
        ), "adversarial images left [0,1] — normalization is misplaced"
        if is_linf and eps is not None:
            max_delta = float((adv - images).abs().max())
            assert (
                max_delta <= float(eps) + 1e-6
            ), f"Linf budget violated: {max_delta} > eps={eps}"

        if n_saved < n_save_samples:
            take = min(n_save_samples - n_saved, images.shape[0])
            saved_clean.append(images[:take].cpu())
            saved_adv.append(adv[:take].cpu())
            saved_labels.append(targets[:take].cpu())
            n_saved += take

        with torch.no_grad():
            preds = model(adv).argmax(dim=1)
        correct += int((preds == targets).sum())
        total += targets.shape[0]
        if n_examples is not None and total >= n_examples:
            break

    assert total > 0, "empty loader"
    result = {"accuracy": correct / total, "n_examples": total}
    if n_save_samples > 0:
        result["samples"] = {
            "clean": torch.cat(saved_clean),
            "adv": torch.cat(saved_adv),
            "labels": torch.cat(saved_labels),
        }
    return result


if __name__ == "__main__":
    from torch.utils.data import TensorDataset

    torch.manual_seed(0)
    x = torch.rand(16, 3, 8, 8)
    y = torch.randint(0, 10, (16,))
    loader = DataLoader(TensorDataset(x, y), batch_size=8, shuffle=False)
    net = nn.Sequential(nn.Flatten(), nn.Linear(3 * 8 * 8, 10)).eval()
    out = evaluate_attack(
        net,
        loader,
        "PGD",
        {"eps": 8 / 255, "alpha": 2 / 255, "steps": 5},
        n_examples=12,
        device="cpu",
    )
    print("PGD:", out)  # {'accuracy': ..., 'n_examples': 12}
