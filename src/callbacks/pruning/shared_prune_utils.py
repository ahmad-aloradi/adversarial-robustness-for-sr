"""Shared sparsity computation for pruners."""

from typing import Dict, Iterator, List, Tuple, Union

import torch
import torch.nn as nn


def masked_name(name: str) -> str:
    """The attribute holding a parameter's live value.

    Why: ``torch.nn.utils.prune`` renames the Parameter to ``<p>_orig`` and
    recomputes ``<p> = orig * mask`` in a forward pre-hook, so
    ``named_parameters`` reports ``weight_orig`` — whose masked entries drift
    off zero under momentum — while the value the model computes with is
    ``weight``.

    >>> masked_name("weight_orig"), masked_name("weight")
    ('weight', 'weight')
    """
    return name[: -len("_orig")] if name.endswith("_orig") else name


def pool_sparsity(sparsity: float, pool_numel: int, dense_numel: int) -> float:
    """Translate a target over every weight tensor into one over the subset a
    callback actually sparsifies.

    Weights held dense still spend their full size out of the same budget,
    which is how RigL accounts for a dense first layer. ``pool_numel`` is the
    sparsified subset, ``dense_numel`` everything else in the denominator.

    >>> round(pool_sparsity(0.9, pool_numel=9900, dense_numel=100), 4)
    0.9091
    >>> pool_sparsity(0.0, pool_numel=9900, dense_numel=100)
    0.0
    """
    kept = (1.0 - sparsity) * (pool_numel + dense_numel) - dense_numel
    assert kept > 0, (
        f"sparsity {sparsity} leaves "
        f"{(1.0 - sparsity) * (pool_numel + dense_numel):.0f} weights but the "
        f"layers held dense already take {dense_numel}; lower the target or "
        f"set prune_first_layer=true"
    )
    return 1.0 - kept / pool_numel


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
                yield getattr(module, masked_name(name))
    elif target and isinstance(target[0], (tuple, list)):
        for module, name in target:
            # missing attr is a wiring bug
            yield getattr(module, masked_name(name))
    else:
        yield from (p for p in target if p.requires_grad)


if __name__ == "__main__":
    import torch.nn.utils.prune

    half_zero = nn.Parameter(torch.tensor([0.0, 0.0, 1.0, 2.0]))
    print("raw list:", compute_sparsity([half_zero]))  # 0.5
    linear = nn.Linear(4, 4)  # 16 weights, 4 biases
    linear.weight.data.zero_()
    print("(module, name):", compute_sparsity([(linear, "weight")]))  # 1.0
    print("whole module:", compute_sparsity(linear))  # 0.8

    pruned = nn.Linear(4, 4)
    torch.nn.utils.prune.l1_unstructured(pruned, "weight", amount=0.5)
    plain = collapse_pruning_reparam(pruned.state_dict())
    fresh = nn.Linear(4, 4)
    fresh.load_state_dict(plain, strict=True)
    print("collapse ok:", torch.equal(fresh.weight, pruned.weight))  # True
