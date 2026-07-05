"""Per-input-channel activation statistics for Wanda-ordered pruning.

Collects, via forward pre-hooks, the std of each layer's input channels
from one training batch. The stats are derived state: nothing is
checkpointed; they are re-derived from the first training batch after
every fit start.

Usage:
    stats = ActivationStats({"enc.conv1": conv_module})
    stats.arm()            # capture the next training forward
    model(batch)           # hooks record per-channel std
    weights = stats.consume()  # {name: geomean-1 normalized weights}
    stats.remove()         # deregister hooks at teardown
"""
from typing import Dict

import torch
import torch.nn as nn

# Column weights are relative importances; bound their spread so the Bregman
# dual (which carries a lambda/col_weight baseline) stays well inside fp32's
# usable range. A silent channel clamps to 1/MAX_RATIO (pruned hard, not
# numerically destroyed); an unbounded floor near zero would push the dual to
# ~lambda/floor where fp32 rounding makes the recovered weight meaningless.
MAX_RATIO = 100.0


def normalize_geomean(act_std: torch.Tensor) -> torch.Tensor:
    """Normalize per-channel stds to geometric mean 1 over live channels.

    Input: (in,) stds. Output: (in,) weights clamped to
    [1/MAX_RATIO, MAX_RATIO]; a constant channel (std 0) maps to the floor.
    A non-finite input std means the encoder diverged and is rejected loud.
    """
    assert act_std.ndim == 1, f"expected (in,), got {tuple(act_std.shape)}"
    assert torch.isfinite(act_std).all(), (
        "non-finite activation std: the encoder produced NaN/Inf activations, "
        "so training has diverged upstream (check weight norms and lambda). "
        "Wanda column weights cannot be derived from a diverged model."
    )
    live = act_std > 0
    if not live.any():
        return torch.full_like(act_std, 1.0 / MAX_RATIO)
    geo_mean = torch.exp(torch.log(act_std[live]).mean())
    return (act_std / geo_mean).clamp(min=1.0 / MAX_RATIO, max=MAX_RATIO)


class ActivationStats:
    """Forward pre-hooks capturing per-input-channel std for named modules.

    Input: {group_name: module}, each an nn.Conv1d/Conv2d (groups == 1)
    or nn.Linear. ``arm()`` marks every module to record its next
    training-mode forward; each hook disarms itself after capturing, so
    validation forwards never pollute the stats. ``consume()`` returns
    {group_name: normalized weights} once all modules have captured.
    """

    def __init__(self, modules: Dict[str, nn.Module]):
        assert modules, "ActivationStats needs at least one module"
        self._in_dims: Dict[str, int] = {}
        self._armed: Dict[str, bool] = {}
        self._raw_std: Dict[str, torch.Tensor] = {}
        self._consumed = True  # nothing captured yet
        self._handles = []
        for name, module in modules.items():
            if isinstance(module, (nn.Conv1d, nn.Conv2d)):
                assert module.groups == 1, (
                    f"{name}: grouped convs are unsupported (input-channel "
                    f"to weight-column map is not identity)"
                )
                self._in_dims[name] = module.in_channels
            elif isinstance(module, nn.Linear):
                self._in_dims[name] = module.in_features
            else:
                raise TypeError(
                    f"{name}: unsupported module {type(module).__name__}; "
                    f"expected Conv1d, Conv2d, or Linear"
                )
            self._armed[name] = False
            self._handles.append(
                module.register_forward_pre_hook(self._make_hook(name))
            )
        self._names = sorted(self._in_dims)

    def _make_hook(self, name: str):
        def hook(module: nn.Module, inputs: tuple) -> None:
            if not (self._armed[name] and module.training):
                return
            with torch.no_grad():
                self._raw_std[name] = self._channel_std(module, inputs[0])
            self._armed[name] = False

        return hook

    def _channel_std(
        self, module: nn.Module, x: torch.Tensor
    ) -> torch.Tensor:
        if isinstance(module, nn.Conv1d):
            assert x.ndim == 3, f"Conv1d expects (B, C, T), got {x.shape}"
            act_std = x.float().std(dim=(0, 2), unbiased=False)
        elif isinstance(module, nn.Conv2d):
            assert x.ndim == 4, f"Conv2d expects (B, C, H, W), got {x.shape}"
            act_std = x.float().std(dim=(0, 2, 3), unbiased=False)
        elif isinstance(module, nn.Linear):
            assert x.ndim >= 2, f"Linear expects (..., F), got {x.shape}"
            act_std = (
                x.float().reshape(-1, x.shape[-1]).std(dim=0, unbiased=False)
            )
        else:
            raise TypeError(f"unsupported module {type(module).__name__}")
        return act_std

    def arm(self) -> None:
        """Record the next training-mode forward of every module."""
        for name in self._armed:
            self._armed[name] = True
        self._consumed = False

    def fresh(self) -> bool:
        """All modules captured since the last arm(), none still waiting."""
        if self._consumed:
            return False
        captured = len(self._raw_std) == len(self._in_dims)
        return captured and not any(self._armed.values())

    def consume(self) -> Dict[str, torch.Tensor]:
        """Normalized per-channel weights for every module, once fresh."""
        assert self.fresh(), "consume() before all modules captured"
        if torch.distributed.is_available() and (
            torch.distributed.is_initialized()
        ):
            # Rank-local stats would give ranks different prox thresholds
            # and desync parameters; average in deterministic name order.
            world_size = torch.distributed.get_world_size()
            for name in self._names:
                torch.distributed.all_reduce(self._raw_std[name])
                self._raw_std[name] /= world_size
        self._consumed = True
        result = {}
        for name in self._names:
            act_std = self._raw_std[name]
            assert act_std.shape[0] == self._in_dims[name], (
                f"{name}: captured {act_std.shape[0]} channels, module "
                f"has {self._in_dims[name]}"
            )
            result[name] = normalize_geomean(act_std)
        return result

    def remove(self) -> None:
        """Deregister all hooks. Idempotent."""
        for handle in self._handles:
            handle.remove()
        self._handles = []


if __name__ == "__main__":
    conv = nn.Conv1d(4, 8, 3)
    linear = nn.Linear(8, 2)
    stats = ActivationStats({"conv": conv, "fc": linear})
    stats.arm()
    x = torch.randn(2, 4, 16)
    x[:, 1] = 0.7  # constant channel -> weight at the 1/MAX_RATIO floor
    linear(conv(x).mean(dim=2))
    for name, w in stats.consume().items():
        print(f"{name}: shape={tuple(w.shape)}, min={float(w.min()):.2e}")
    stats.remove()
