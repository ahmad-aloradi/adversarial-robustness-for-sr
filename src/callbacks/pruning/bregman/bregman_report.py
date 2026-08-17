"""Everything BregmanPruner writes out: per-step metrics and the fit-start dump.

Split from the callback so the reporting can be read and changed without
touching the training lifecycle. Nothing here affects training.
"""

from pytorch_lightning import LightningModule
from pytorch_lightning.utilities import rank_zero_only

from src import utils

# Absolute (not relative) so the __main__ smoke block runs via `python <file>`.
from src.callbacks.pruning.bregman.bregman_regularizers import (
    is_regularized,
    lambda_scale,
)
from src.callbacks.pruning.bregman.lambda_scheduler import LambdaScheduler
from src.callbacks.pruning.bregman.quantile_lambda_scheduler import (
    QuantileLambdaScheduler,
)
from src.callbacks.pruning.shared_prune_utils import compute_sparsity

log = utils.get_pylogger(__name__)


def log_step_metrics(
    pl_module: LightningModule,
    lambda_scheduler,
    overall_sparsity: float,
    pruned_sparsity: float,
) -> None:
    """Log the per-step sparsity and lambda-controller series."""
    # Per-step only; the epoch-level values come from on_train_epoch_end.
    per_step = {
        **pl_module.logging_params,
        "on_step": True,
        "on_epoch": False,
    }
    pl_module.log_dict(
        {
            "bregman/sparsity": overall_sparsity,
            "bregman/pruned_sparsity": pruned_sparsity,
        },
        **per_step,
    )

    if lambda_scheduler is None:
        return

    sched = lambda_scheduler
    pl_module.log("bregman/global_lambda", sched.get_lambda(), **per_step)
    # Controller scalars are identical on every rank; skip the all-reduce.
    rank_identical = {**per_step, "sync_dist": False, "prog_bar": False}
    if isinstance(sched, LambdaScheduler):
        pl_module.log_dict(
            {
                "bregman/delta_lambda": sched.last_delta,
                "bregman/delta_lambda_over_lambda": sched.last_delta_over_lambda,
                "bregman/sparsity_diff": sched.gap,
                "bregman/sparsity_crossings": float(sched.crossings),
                "bregman/alpha": sched.alpha,
            },
            **rank_identical,
        )
    elif isinstance(sched, QuantileLambdaScheduler):
        pl_module.log_dict(
            {
                "bregman/quantile_k": float(sched.last_k),
                "bregman/achieved_sparsity_at_update": 1.0
                - sched.last_k / sched.n,
                "bregman/lambda_clamp_hits": float(sched.clamp_hits),
            },
            **rank_identical,
        )
    else:
        raise TypeError(
            "log_step_metrics supports LambdaScheduler or "
            f"QuantileLambdaScheduler, got {type(sched).__name__}"
        )


@rank_zero_only
def log_configuration(
    optimizer,
    manager,
    lambda_scheduler,
    target_sparsity: float,
    sparsity_threshold: float,
    overall_sparsity: float,
    pruned_sparsity: float,
) -> None:
    """Dump the optimizer groups, their lambdas, and per-group sparsity."""
    log.info("=== Bregman Configuration ===")
    log.info(f"Optimizer: {type(optimizer).__name__}")

    if isinstance(lambda_scheduler, LambdaScheduler):
        log.info(
            f"Lambda Scheduler: feedback controller, target_sparsity={target_sparsity}, "
            f"lambda={lambda_scheduler.get_lambda():.4f}, "
            f"update_frequency={lambda_scheduler.update_frequency}"
        )
    elif isinstance(lambda_scheduler, QuantileLambdaScheduler):
        log.info(
            f"Lambda Scheduler: quantile threshold, target_sparsity={target_sparsity}, "
            f"lambda={lambda_scheduler.get_lambda():.4f}, "
            f"update_frequency={lambda_scheduler.update_frequency}, "
            f"n={lambda_scheduler.n}"
        )
    elif lambda_scheduler is None:
        log.info("Lambda Scheduler: None (static lambda mode)")
    else:
        raise TypeError(
            "log_configuration supports LambdaScheduler, QuantileLambdaScheduler "
            f"or None, got {type(lambda_scheduler).__name__}"
        )

    for group in optimizer.param_groups:
        name = group.get("name", "unnamed")
        reg_type = type(group["reg"]).__name__
        if not is_regularized(group):
            log.info(f"  Group '{name}': {reg_type} (inactive)")
            continue
        scale = lambda_scale(group)
        log.info(
            f"  Group '{name}': {reg_type}, "
            f"lambda={group['reg'].lamda:.4f}, scale={scale}"
        )
        if scale != 1.0:
            log.warning(
                f"Group '{name}' has lambda_scale={scale} != 1.0. "
                "Non-uniform regularization is generally not recommended."
            )

    log.info("Current sparsity by group:")
    for group in manager.processed_groups:
        name = group["config"].get("name", "unnamed")
        sparsity = compute_sparsity(
            group["params"], threshold=sparsity_threshold
        )
        unpruned = (
            "(not pruned)" if group["applier"].sparsity_rate == 0.0 else ""
        )
        log.info(f"  {name}: {sparsity:.3%} {unpruned}")

    log.info(
        f"Overall sparsity: {overall_sparsity:.3%} "
        f"(pruned only: {pruned_sparsity:.3%})"
    )
    log.info("=== End Configuration ===")


@rank_zero_only
def log_group_assignments(pl_module: LightningModule, manager) -> None:
    """Dump which modules landed in which pruning group, for debugging."""
    param_to_module = {
        id(p): ".".join(name.split(".")[:-1])
        for name, p in pl_module.named_parameters()
    }
    total_params = sum(
        p.numel()
        for group in manager.processed_groups
        for p in group["params"]
    )

    log.info("--- Parameter Group Assignments ---")
    for group in manager.processed_groups:
        name = group["config"].get("name", "unnamed")
        modules = {param_to_module.get(id(p)) for p in group["params"]}
        modules.discard(None)
        group_params = sum(p.numel() for p in group["params"])
        pct = group_params / total_params * 100 if total_params else 0
        log.info(40 * "-")
        log.info(
            f"  {name}: {len(modules)} modules, "
            f"{group_params:,} params ({pct:.1f}%)"
        )
        for module in sorted(modules):
            log.info(
                f"    {module} ({type(pl_module.get_submodule(module)).__name__})"
            )
    log.info(40 * "-")


if __name__ == "__main__":
    # Smoke: the group vocabulary these reports key off, on fake param groups.
    from src.callbacks.pruning.bregman.bregman_regularizers import (
        RegL1,
        RegNone,
    )

    groups = [
        {"name": "conv", "reg": RegL1(lamda=0.25), "lambda_scale": 1.0},
        {"name": "norm", "reg": RegNone(), "lambda_scale": 0.0},
    ]
    for g in groups:
        print(
            f"{g['name']}: regularized={is_regularized(g)} scale={lambda_scale(g)}"
        )
