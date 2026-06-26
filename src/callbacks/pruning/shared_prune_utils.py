"""Shared sparsity computation for pruners."""

from typing import Iterator, List, Tuple, Union

import torch
import torch.nn as nn


def compute_sparsity(
    target: Union[
        nn.Module,
        List[nn.Parameter],
        List[Tuple[nn.Module, str]],
    ],
    threshold: float = 1e-12,
) -> float:
    """Fraction of near-zero elements (|w| <= threshold) over ``target``.

    ``target`` is one of:
    - nn.Module: every parameter as the module currently exposes it — a pruned
      ``weight_orig`` is measured as its masked ``weight`` (whole-model sparsity).
    - List[Parameter]: trainable parameter tensors (Bregman style).
    - List[(Module, name)]: ``getattr(module, name)`` per pair, so a pruned
      ``weight`` is already its masked value (magnitude style).
    """
    total = zeros = 0
    for tensor in _iter_tensors(target):
        total += tensor.numel()
        zeros += int((tensor.abs() <= threshold).sum())
    return zeros / max(1, total)


def _iter_tensors(target) -> Iterator[torch.Tensor]:
    """Yield the tensor each entry contributes; for an nn.Module de-duplicated
    by identity so weight-tied layers count once."""
    if isinstance(target, nn.Module):
        seen = set()
        for module in target.modules():
            for name, param in module.named_parameters(recurse=False):
                if id(param) in seen:
                    continue
                seen.add(id(param))
                # weight_orig exposes its masked weight after pruning.
                name = name[:-5] if name.endswith("_orig") else name
                yield getattr(module, name)
    elif target and isinstance(target[0], (tuple, list)):
        for module, name in target:
            yield getattr(module, name)  # missing attr is a wiring bug
    else:
        yield from (p for p in target if p.requires_grad)


if __name__ == "__main__":
    half_zero = nn.Parameter(torch.tensor([0.0, 0.0, 1.0, 2.0]))
    print("raw list:", compute_sparsity([half_zero]))  # 0.5
    linear = nn.Linear(4, 4)
    linear.weight.data.zero_()
    print("(module, name):", compute_sparsity([(linear, "weight")]))  # 1.0
    print("whole module:", compute_sparsity(linear))  # 0.8
