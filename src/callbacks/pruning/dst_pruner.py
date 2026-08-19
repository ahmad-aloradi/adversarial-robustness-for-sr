""" Implementations of: RigL (Evci et al., ICML 2020), SET (Mocanu et al.,
Nat. Comm. 2018), Static-ERK, SNIP (Lee et al., ICLR 2019) and GraNet
(Liu et al., NeurIPS 2021).

Idea: train at the target sparsity from step 0 behind a boolean mask. The five
differ in how the first mask is built, whether and how it is updated, and
whether the target moves; ``docs/sparse_training.md`` has the per-experiment
values.

**The gradient is never masked!** Only w and the optimizer momentum are, after
each step. RigL grows on the dense gradient, so masking it would kill regrowth.

Run one of these end to end with::

    python src/train.py experiment=img/pruning_rigl datamodule=datasets/mnist

Inspect the mask builder alone with::

    python -m src.callbacks.pruning.dst_pruner
"""
import math
import os
from typing import Any, Dict, Iterator, List, Optional, Tuple

import torch
import torch.nn as nn
from pytorch_lightning import Callback, LightningModule, Trainer
from torchmetrics import Metric

from src.callbacks.pruning.dst_schedules import (
    cosine_drop_fraction,
    cubic_prune_rate,
)
from src.callbacks.pruning.parameter_manager import (
    ParameterManager,
    dense_held_numel,
)
from src.callbacks.pruning.shared_prune_utils import (
    compute_sparsity,
    pool_sparsity,
)
from src.callbacks.pruning.utils.erk_sparsity import (
    LayerShape,
    solve_erk_densities,
)
from src.utils import get_pylogger

logger = get_pylogger(__name__)


def _cycle(loader: Any) -> Iterator[Any]:
    """Endless batches, so more prune steps than batches is not a problem.

    Not ``itertools.cycle``: that caches every batch it yields, holding a whole
    epoch of images. Re-iterating the DataLoader does not.
    """
    while True:
        yield from loader


class DynamicSparsePruner(Callback):
    """Orchestrates dynamic sparse training. Masks live on the callback, not on
    the model, so the ``state_dict`` stays dense-with-zeros and loads strict.

    Key parameters:
        amount: final sparsity over every weight tensor (all but norms and
            biases), including the ones this callback holds dense.
        initial_amount: sparsity at step 0; ``None`` means "same as amount",
            i.e. constant sparsity. GraNet takes 0.0 (dense) or 0.5 (Table 4).
        mask_init: ``erk`` (RigL's Erdos-Renyi-Kernel layerwise budget) or
            ``snip`` (|w * grad| saliency, ranked globally).
        snip_iterations: prune steps SNIP takes to reach the target. ``1`` is
            the paper's one-shot ranking; higher is de Jorge et al.'s iterative
            SNIP, which rescores the masked net between steps.
        growth: ``gradient`` regrows the largest dense gradients (RigL),
            ``random`` regrows uniformly at random (SET).
        update_frequency: steps between mask updates; ``None`` never updates.
        prune_first_layer: ``False`` holds the stem conv dense; the target
            then rises on the layers that are left (``_pool_target``).
        drop_fraction / drop_end_fraction / drop_end_value: the cosine schedule
            for how much of each mask is redrawn (see ``dst_schedules``).
        final_prune_epoch: epoch GraNet's cubic ramp reaches ``amount``.
    """

    def __init__(
        self,
        amount: float = 0.9,
        initial_amount: Optional[float] = None,
        mask_init: str = "erk",
        snip_iterations: int = 1,
        growth: str = "gradient",
        update_frequency: Optional[int] = 100,
        drop_fraction: float = 0.3,
        drop_end_fraction: float = 0.75,
        drop_end_value: float = 0.0,
        final_prune_epoch: Optional[int] = None,
        prune_bias: bool = False,
        prune_first_layer: bool = True,
        min_param_elements: int = 100,
        tolerance: float = 0.005,
        verbose: int = 1,
    ):
        self.amount = amount
        self.initial_amount = initial_amount
        self.mask_init = mask_init
        self.snip_iterations = snip_iterations
        self.growth = growth
        self.update_frequency = update_frequency
        self.drop_fraction = drop_fraction
        self.drop_end_fraction = drop_end_fraction
        self.drop_end_value = drop_end_value
        self.final_prune_epoch = final_prune_epoch
        self.tolerance = tolerance
        self.verbose = verbose
        self.manager = ParameterManager(
            prune_bias=prune_bias,
            prune_first_layer=prune_first_layer,
            min_param_elements=min_param_elements,
        )

        self._targets: List[Tuple[str, nn.Module, str]] = []
        self._dense_numel: int = 0
        self._masks: Dict[str, torch.Tensor] = {}
        self._total_steps: int = 0
        self._current_target: float = amount
        # 0 leaves the ramp off: a mask update index is 1 or more, never 0.
        self._final_update: int = 0

    # ------------------------------------------------------------------ #
    #  Lightning hooks                                                     #
    # ------------------------------------------------------------------ #

    def setup(
        self, trainer: Trainer, pl_module: LightningModule, stage: str
    ) -> None:
        self._collect(pl_module)
        if self.verbose and stage == "fit":
            self.manager.log_overview()

    def on_train_start(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        assert trainer.accumulate_grad_batches == 1, (
            f"the mask update runs at the batch end, one per optimizer step, so "
            f"accumulate_grad_batches must be 1, got {trainer.accumulate_grad_batches}"
        )
        self._total_steps = int(trainer.estimated_stepping_batches)
        if self.initial_amount is not None:
            assert (
                self.update_frequency is not None
            ), "a moving target needs mask updates, got update_frequency=None"
            updates_per_epoch = (
                self._total_steps / trainer.max_epochs
            ) / self.update_frequency
            self._final_update = int(
                self.final_prune_epoch * updates_per_epoch
            )
            assert self._final_update >= 1, (
                f"the cubic ramp needs at least one mask update, got {self._final_update}: "
                f"{self._total_steps} steps over {trainer.max_epochs} epochs every "
                f"{self.update_frequency} steps is {updates_per_epoch:.3f} updates/epoch "
                f"at final_prune_epoch={self.final_prune_epoch}"
            )

        if self._masks:
            self._restore_masks_to(pl_module.device)
        else:
            self._build_masks(pl_module, trainer)
        self._apply_masks(trainer.optimizers[0])
        self._assert_on_target()

    def on_train_epoch_start(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        trainer.callback_metrics["pruning/sparsity"] = torch.tensor(
            self.pruned_sparsity()
        )

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Mask, update the topology, mask again -- all after
        ``optimizer.step()``.

        Why: a connection grown before the step is moved by the very gradient
        that selected it, so RigL's zero init would last no steps. Upstream runs
        the update under ``control_dependencies`` on the optimizer update, and
        sparselearning does ``step(); apply_mask(); pruning(); truncate_weights()``.
        """
        optimizer = trainer.optimizers[0]
        # The gradient stays dense, so the step just wrote into the masked w.
        self._apply_masks(optimizer)

        step = trainer.global_step
        if self.update_frequency is None or step % self.update_frequency != 0:
            return

        update = step // self.update_frequency
        # Past the ramp end GraNet stops pruning but keeps redistributing.
        if update <= self._final_update:
            self._current_target = cubic_prune_rate(
                update,
                self._final_update,
                1.0 - self.initial_amount,
                1.0 - self.amount,
            )
            self._global_prune(self._current_target)
        self._redistribute(step, optimizer)
        # Dropped weights leave the mask holding their old value; zero them.
        self._apply_masks(optimizer)

    def on_train_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        self._assert_on_target()
        pruned = self.pruned_sparsity()
        overall = compute_sparsity(pl_module)
        pl_module.log("sparsity", overall, prog_bar=False, on_epoch=True)
        pl_module.log(
            "pruning/sparsity", pruned, prog_bar=False, on_epoch=True
        )
        if self.verbose:
            # A static mask never redraws, so reporting a drop fraction would lie.
            drop = (
                "static"
                if self.update_frequency is None
                else f"{self._drop_fraction_at(trainer.global_step):.4f}"
            )
            logger.info(
                f"[DST Monitor] Epoch {trainer.current_epoch}: "
                f"Target={self._current_target:.2%} | "
                f"PrunedParams Sparsity={pruned:.2%} | "
                f"Total sparsity={overall:.2%} | DropFraction={drop}"
            )

    def state_dict(self) -> Dict[str, Any]:
        return {
            "masks": {k: v.cpu() for k, v in self._masks.items()},
            "current_target": self._current_target,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self._masks = dict(state_dict["masks"])
        self._current_target = state_dict["current_target"]

    # ------------------------------------------------------------------ #
    #  Measurement                                                         #
    # ------------------------------------------------------------------ #

    def pruned_sparsity(self) -> float:
        """Sparsity over every weight tensor, read from the masks.

        Masks, not w: a weight regrown at exactly 0 is active, so counting
        zeros would overstate sparsity for one step and trip the drift assert.
        """
        kept = sum(int(mask.sum()) for mask in self._masks.values())
        total = sum(mask.numel() for mask in self._masks.values())
        return 1.0 - (kept + self._dense_numel) / (total + self._dense_numel)

    def _pool_target(self, sparsity: float) -> float:
        """``sparsity`` over every weight tensor, restated over the masked
        pool."""
        pool = sum(param.numel() for _, param in self._iter_targets())
        return pool_sparsity(sparsity, pool, self._dense_numel)

    def _assert_on_target(self) -> None:
        achieved = self.pruned_sparsity()
        band = self.tolerance * self._current_target
        assert abs(achieved - self._current_target) <= band, (
            f"DST sparsity drifted: achieved {achieved:.4f}, target "
            f"{self._current_target:.4f}, band +/-{band:.4f}"
        )

    # ------------------------------------------------------------------ #
    #  Mask construction                                                   #
    # ------------------------------------------------------------------ #

    def _collect(self, model: nn.Module) -> None:
        """Resolve the prunable set and a stable name for each target."""
        params = self.manager.collect_parameters(model)
        assert params, "ParameterManager found nothing to prune"
        qualnames = {id(m): name for name, m in model.named_modules()}
        self._targets = [
            (f"{qualnames[id(module)]}.{name}", module, name)
            for module, name in params
        ]
        self._dense_numel = dense_held_numel(
            model, params, self.manager.min_param_elements
        )

    def _iter_targets(self) -> Iterator[Tuple[str, torch.Tensor]]:
        """(key, tensor) for every target, in module-definition order."""
        for key, module, name in self._targets:
            yield key, getattr(module, name)

    def _build_masks(
        self, model: nn.Module, trainer: Optional[Trainer] = None
    ) -> None:
        start_sparsity = (
            self.amount if self.initial_amount is None else self.initial_amount
        )
        self._current_target = start_sparsity
        pool_target = self._pool_target(start_sparsity)

        if self.mask_init == "snip":
            self._masks = self._snip_masks(model, trainer, pool_target)
            return
        if self.mask_init != "erk":
            raise ValueError(
                f"mask_init is 'erk' or 'snip', got {self.mask_init!r}"
            )

        densities = solve_erk_densities(
            [
                LayerShape(
                    name=key,
                    fan_in=param.shape[1] if param.dim() > 1 else 1,
                    fan_out=param.shape[0],
                    kernel_dims=tuple(param.shape[2:]),
                    n_params=param.numel(),
                )
                for key, param in self._iter_targets()
            ],
            pool_target,
            mode="erk",
        )
        generator = self._cpu_generator()
        self._masks = {}
        for key, param in self._iter_targets():
            n_keep = int(round(densities[key] * param.numel()))
            flat = torch.zeros(param.numel(), dtype=torch.bool)
            flat[
                torch.randperm(param.numel(), generator=generator)[:n_keep]
            ] = True
            self._masks[key] = flat.view_as(param).to(param.device)

    def _snip_masks(
        self, model: nn.Module, trainer: Optional[Trainer], sparsity: float
    ) -> Dict[str, torch.Tensor]:
        """|w * grad| saliency on the untrained w, ranked globally, so SNIP's
        own criterion picks the layerwise budget.

        ``snip_iterations`` > 1 walks the density down ``(1 - sparsity) ** (t / T)``
        and rescores the masked net each step. BatchNorm makes each layer's summed
        saliency roughly constant, so one ranking collapses the biggest layers.
        """
        assert trainer is not None, "mask_init='snip' needs a live trainer"
        assert trainer.world_size == 1, (
            f"SNIP scores one batch per rank, so every rank would build a "
            f"different mask; world_size must be 1, got {trainer.world_size}"
        )
        batches = _cycle(trainer.train_dataloader)
        final_density = 1.0 - sparsity
        masks = {
            key: torch.ones_like(param, dtype=torch.bool)
            for key, param in self._iter_targets()
        }
        # running_mean is None wherever a norm layer tracks no running stats.
        norm_stats = [
            (
                module,
                module.running_mean.clone(),
                module.running_var.clone(),
                module.num_batches_tracked.clone(),
            )
            for module in model.modules()
            if getattr(module, "running_mean", None) is not None
        ]

        for step in range(1, self.snip_iterations + 1):
            density = final_density ** (step / self.snip_iterations)
            scores = self._snip_scores(model, next(batches), masks)
            flat = torch.cat([s.flatten() for s in scores.values()])
            n_keep = max(int(round(density * flat.numel())), 1)
            threshold = torch.topk(flat, n_keep, sorted=True).values[-1]
            # & masks: a mask only ever shrinks, whatever the rescored ties do.
            masks = {
                key: (score >= threshold) & masks[key]
                for key, score in scores.items()
            }

        # SNIP is a measurement of the init, so undo what the scoring passes moved.
        for module, mean, var, count in norm_stats:
            module.running_mean.copy_(mean)
            module.running_var.copy_(var)
            module.num_batches_tracked.copy_(count)
        for module in model.modules():
            if isinstance(module, Metric):
                module.reset()

        # Layer collapse: nothing regrows a layer left with fewer weights than output units.
        kept = {key: int(mask.sum()) for key, mask in masks.items()}
        collapsed = {
            key: n for key, n in kept.items() if n < masks[key].shape[0]
        }
        if collapsed:
            logger.warning(
                f"SNIP collapsed {len(collapsed)} layer(s) at masked-pool sparsity "
                f"{sparsity:.4f}: "
                + ", ".join(
                    f"{key} kept {n} weights for {masks[key].shape[0]} output units"
                    for key, n in collapsed.items()
                )
                + ". Raise callbacks.model_pruning.snip_iterations or use "
                "mask_init=erk."
            )
        return masks

    def _snip_scores(
        self,
        model: nn.Module,
        batch: Any,
        masks: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """|w * grad| over one batch, with the masked weights zeroed first."""
        batch = model.transfer_batch_to_device(batch, model.device, 0)
        with torch.no_grad():
            for key, param in self._iter_targets():
                param.mul_(masks[key])
        model.zero_grad(set_to_none=True)
        # ported from namhoonlee/snip-public snip.py: saliency is |g * w|
        model.training_step(batch, 0)["loss"].backward()

        scores = {}
        for key, param in self._iter_targets():
            assert param.grad is not None, f"no gradient reached {key}"
            scores[key] = (param.grad * param).abs().detach()
        model.zero_grad(set_to_none=True)
        return scores

    def _restore_masks_to(self, device: torch.device) -> None:
        """A restored mask set must match the targets key for key and shape for
        shape: ``pruned_sparsity`` reads the masks while ``_apply_masks`` reads
        the targets, so a stale key skews the reported sparsity and the gates
        that read it without ever touching the model.
        """
        shapes = {key: param.shape for key, param in self._iter_targets()}
        assert self._masks.keys() == shapes.keys(), (
            f"restored masks must cover exactly the target weights, "
            f"got {sorted(self._masks.keys() ^ shapes.keys())} unmatched"
        )
        for key, shape in shapes.items():
            assert self._masks[key].shape == shape, (
                f"{key}: restored mask is {tuple(self._masks[key].shape)}, "
                f"weight is {tuple(shape)}"
            )
            self._masks[key] = self._masks[key].to(device)

    def _cpu_generator(self, offset: int = 0) -> torch.Generator:
        """CPU RNG seeded from the global seed, so every rank draws the same
        mask without communicating.

        ``offset`` is the step a random draw belongs to, so successive updates
        do not replay one permutation, and a resumed run redraws identically.
        """
        seed = os.environ["PL_GLOBAL_SEED"]
        return torch.Generator().manual_seed(int(seed) + offset)

    # ------------------------------------------------------------------ #
    #  Mask enforcement and updates                                        #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _state_tensors(
        optimizer: torch.optim.Optimizer, param: torch.Tensor
    ) -> Iterator[torch.Tensor]:
        """Optimizer-state tensors shaped like ``param`` -- momentum and kin.

        Shape is the only reliable marker: an optimizer may hold per-parameter
        scalars (step counters) in the same state dict.
        """
        for value in optimizer.state[param].values():
            if isinstance(value, torch.Tensor) and value.shape == param.shape:
                yield value

    @torch.no_grad()
    def _apply_masks(self, optimizer: torch.optim.Optimizer) -> None:
        """Zero the masked w and their optimizer state after the step.

        Mirrors sparselearning's apply_mask: the gradient is left dense so the
        growth criterion still sees it.
        """
        for key, param in self._iter_targets():
            mask = self._masks[key]
            param.mul_(mask)
            for buffer in self._state_tensors(optimizer, param):
                buffer.mul_(mask)

    @torch.no_grad()
    def _global_prune(self, sparsity: float) -> None:
        """Rank every target weight together and keep the largest.

        Sparsity moves here and only here; ``_redistribute`` is count-preserving.
        """
        # ported from VITA-Group/GraNet@f338a24 CIFAR/sparselearning/core.py:Masking.pruning
        flat = torch.cat(
            [param.abs().flatten() for _, param in self._iter_targets()]
        )
        n_keep = int(flat.numel() * (1.0 - self._pool_target(sparsity)))
        threshold = torch.topk(flat, n_keep, sorted=True).values[-1]
        for key, param in self._iter_targets():
            self._masks[key] = param.abs() > threshold

    def _drop_fraction_at(self, step: int) -> float:
        return cosine_drop_fraction(
            step,
            self._total_steps,
            self.drop_fraction,
            self.drop_end_fraction,
            self.drop_end_value,
        )

    @torch.no_grad()
    def _redistribute(
        self, step: int, optimizer: torch.optim.Optimizer
    ) -> None:
        """Per layer: drop the smallest weights, regrow as many elsewhere."""
        fraction = self._drop_fraction_at(step)
        if fraction <= 0.0:
            return

        generator = self._cpu_generator(step)
        for key, param in self._iter_targets():
            assert param.grad is not None, (
                f"{key} has no gradient at step {step}; the growth criterion "
                f"needs one"
            )
            mask = self._masks[key]
            n_active = int(mask.sum())
            n_inactive = mask.numel() - n_active
            n_change = min(int(fraction * n_active), n_active, n_inactive)
            if n_change == 0:
                continue

            # Death: the smallest-magnitude active weights leave the mask.
            death_scores = torch.where(
                mask, param.abs(), torch.full_like(param, math.inf)
            )
            dropped = torch.topk(
                death_scores.flatten(), n_change, largest=False
            ).indices
            mask.view(-1)[dropped] = False

            # Growth: refill the same count from the now-inactive positions.
            if self.growth == "gradient":
                growth_scores = torch.where(
                    mask, torch.full_like(param, -1.0), param.grad.abs()
                )
                grown = torch.topk(
                    growth_scores.flatten(), n_change, largest=True
                ).indices
            elif self.growth == "random":
                inactive = (~mask).flatten().nonzero().squeeze(1)
                order = torch.randperm(inactive.numel(), generator=generator)
                grown = inactive[order[:n_change].to(inactive.device)]
            else:
                raise ValueError(
                    f"growth is 'gradient' or 'random', got {self.growth!r}"
                )

            mask.view(-1)[grown] = True
            param.view(-1)[grown] = 0.0
            for buffer in self._state_tensors(optimizer, param):
                buffer.view(-1)[grown] = 0.0

            assert int(mask.sum()) == n_active, (
                f"{key}: redistribution changed the active count "
                f"{n_active} -> {int(mask.sum())}"
            )


if __name__ == "__main__":
    os.environ.setdefault("PL_GLOBAL_SEED", "42")
    demo_net = nn.Sequential(
        nn.Conv2d(3, 16, 3), nn.Conv2d(16, 32, 3), nn.Linear(32, 10)
    )
    demo = DynamicSparsePruner(amount=0.9, prune_first_layer=True, verbose=0)
    demo._collect(demo_net)
    demo._build_masks(demo_net)
    demo._apply_masks(torch.optim.SGD(demo_net.parameters(), lr=0.1))
    print("ERK layerwise sparsity at a 0.9 budget:")
    for demo_key, demo_mask in demo._masks.items():
        print(
            f"  {demo_key}: sparsity={1 - float(demo_mask.float().mean()):.4f}"
        )
    print(
        f"  overall={compute_sparsity([(m, n) for _, m, n in demo._targets]):.4f}"
    )
