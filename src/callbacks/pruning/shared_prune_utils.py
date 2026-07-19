"""Shared sparsity computation for pruners."""

from typing import Dict, Iterator, List, Tuple, Union

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


def collapse_pruning_reparam(
    state_dict: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Collapse ``torch.nn.utils.prune`` re-parametrization in a state dict.

    Magnitude-pruned checkpoints store each pruned parameter ``<p>`` as the
    pair ``<p>_orig`` / ``<p>_mask`` (the module recomputes
    ``<p> = orig * mask`` via a forward pre-hook). A freshly instantiated,
    un-pruned module expects plain ``<p>`` keys, so ``strict=True`` loading
    fails. This returns a new dict with every pair collapsed to
    ``<p> = orig * mask``; dicts without pruning keys pass through unchanged.
    """
    orig_keys = {k for k in state_dict if k.endswith("_orig")}
    mask_keys = {k for k in state_dict if k.endswith("_mask")}
    orig_stems = {k[: -len("_orig")] for k in orig_keys}
    mask_stems = {k[: -len("_mask")] for k in mask_keys}
    assert orig_stems == mask_stems, (
        f"Orphan pruning keys: _orig without _mask {orig_stems - mask_stems}"
        f" / _mask without _orig {mask_stems - orig_stems}"
    )

    collapsed = {}
    for key, value in state_dict.items():
        if key in orig_keys or key in mask_keys:
            continue
        assert key not in orig_stems, (
            f"State dict has both {key!r} and its _orig/_mask pair; "
            "refusing to guess which is authoritative."
        )
        collapsed[key] = value
    for stem in orig_stems:
        collapsed[stem] = (
            state_dict[stem + "_orig"] * state_dict[stem + "_mask"]
        )
    return collapsed


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
    import torch.nn.utils.prune

    half_zero = nn.Parameter(torch.tensor([0.0, 0.0, 1.0, 2.0]))
    print("raw list:", compute_sparsity([half_zero]))  # 0.5
    linear = nn.Linear(4, 4)    # 16 weights, 4 biases
    linear.weight.data.zero_()
    print("(module, name):", compute_sparsity([(linear, "weight")]))  # 1.0
    print("whole module:", compute_sparsity(linear))  # 0.8

    pruned = nn.Linear(4, 4)
    torch.nn.utils.prune.l1_unstructured(pruned, "weight", amount=0.5)
    plain = collapse_pruning_reparam(pruned.state_dict())
    fresh = nn.Linear(4, 4)
    fresh.load_state_dict(plain, strict=True)
    print("collapse ok:", torch.equal(fresh.weight, pruned.weight))  # True
