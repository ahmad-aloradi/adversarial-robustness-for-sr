#!/usr/bin/env python3
"""Post-hoc robustness benchmarks for a finished image-classification run.

Evaluates the exact checkpoint whose clean ``test_accuracy`` is recorded in
the run's ``results.json`` under (1) the published {dataset}-C corruption
sets (gaussian_noise by default, severities 1-5) and (2) adversarial attacks
via a generic torchattacks adapter (AutoAttack by default). Attacks need NO
training: they perturb test inputs against the frozen checkpoint. Results are
merged into ``results.json`` and appended to the run's own loggers (wandb
resumed by id, csv/tensorboard as version_1). Each attack also dumps its first
``robustness.save_samples`` clean/adversarial pairs (default 5) as PNGs under
``{exp_dir}/adv_attacks/<name>/<norm>_e_<eps>/`` for visual inspection.

Usage::

    python scripts/eval_robustness.py exp_dir=/path/to/run robustness=default
    python scripts/eval_robustness.py exp_dir=... robustness=default \\
        robustness.attacks.autoattack.n_examples=1000 \\
        robustness.save_samples=10 \\
        robustness.loggers=[csv,tensorboard]

The script reuses the eval-side machinery for loading the training-time
config (same structure as ``scripts/quantize_ptq.py``).
"""

from __future__ import annotations

import datetime
import sys
from pathlib import Path

import hydra
import pyrootutils
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import (
    LightningDataModule,
    LightningModule,
    seed_everything,
)

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".env", "setup.py", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

_HYDRA_PARAMS = {
    "version_base": "1.3",
    "config_path": str(root / "configs"),
    "config_name": "eval.yaml",
}

# Reuse shared exp_dir preprocessing (the same helpers eval.py uses).
# NOTE: ensure_use_avg_ckpt_default is deliberately NOT used — averaged
# weights are not the checkpoint whose test_accuracy results.json records.
from src import utils  # noqa: E402
from src.callbacks.pruning.shared_prune_utils import (  # noqa: E402
    collapse_pruning_reparam,
)
from src.robustness import (  # noqa: E402
    NormalizedModel,
    append_metrics_to_run,
    attack_sample_dir,
    build_corruption_loader,
    build_run_loggers,
    evaluate_accuracy,
    evaluate_attack,
    extract_normalization,
    list_corruptions,
    merge_robustness_results,
    read_reference_test_accuracy,
    resolve_best_checkpoint,
    save_adversarial_samples,
)
from src.utils.exp_dir import (  # noqa: E402
    apply_eval_adjustments,
    prepare_argv_for_exp_dir,
    resolve_checkpoint,
)

log = utils.get_pylogger(__name__)

# Recomputed clean accuracy must reproduce the recorded test_accuracy —
# anything larger means the wrong checkpoint or transform pipeline.
_CLEAN_ACCURACY_TOLERANCE = 2e-3


def _resolve_device(requested: str) -> torch.device:
    if requested.startswith("cuda") and not torch.cuda.is_available():
        log.warning("CUDA requested but unavailable; falling back to CPU.")
        return torch.device("cpu")
    return torch.device(requested)


def _eval_loader_kwargs(cfg: DictConfig) -> dict:
    """The run's eval loader settings, reusable for corruption loaders."""
    kwargs = OmegaConf.to_container(cfg.datamodule.loaders.valid, resolve=True)
    kwargs["shuffle"] = False
    return kwargs


def _run_corruption(cfg, model, datamodule, device, metrics, block) -> None:
    c_cfg = cfg.robustness.corruption
    dataset_name = str(cfg.datamodule.name)
    severities = [int(s) for s in c_cfg.severities]
    corruptions = list_corruptions(dataset_name, str(c_cfg.data_dir))
    log.info(
        f"Corruption benchmark: reading {dataset_name}-C from "
        f"{c_cfg.data_dir} — {len(corruptions)} types x "
        f"{len(severities)} severities: {corruptions}"
    )

    class_to_idx = getattr(datamodule.test_data, "class_to_idx", None)
    per_type = {}
    for corruption in corruptions:
        per_type[corruption] = {}
        for severity in severities:
            loader = build_corruption_loader(
                dataset_name=dataset_name,
                corruption=corruption,
                severity=severity,
                data_dir=str(c_cfg.data_dir),
                loader_kwargs=_eval_loader_kwargs(cfg),
                class_to_idx=class_to_idx,
            )
            acc = evaluate_accuracy(model, loader, device)
            log.info(f"{corruption} severity {severity}: accuracy={acc:.4f}")
            per_type[corruption][f"severity_{severity}"] = acc
            metrics[
                f"robust/corruption/{corruption}/severity_{severity}"
            ] = acc

    # Mean over every corruption type evaluated; n_types travels with it so
    # the CIFAR (19) and Tiny-ImageNet (15) means are never confused.
    mean = {}
    for severity in severities:
        key = f"severity_{severity}"
        accs = [per_type[c][key] for c in corruptions]
        mean[key] = sum(accs) / len(accs)
        log.info(
            f"mean over {len(accs)} corruptions, severity {severity}: "
            f"accuracy={mean[key]:.4f}"
        )
        metrics[f"robust/corruption/mean/{key}"] = mean[key]

    block["corruption"] = {
        "types": per_type,
        "mean": mean,
        "n_types": len(corruptions),
    }


def _save_samples(exp_dir, model, device, name, kwargs, samples) -> None:
    """Write the attack's first N clean/adv pairs under adv_attacks/."""
    clean, adv, labels = samples["clean"], samples["adv"], samples["labels"]
    with torch.no_grad():
        clean_preds = model(clean.to(device)).argmax(dim=1).cpu()
        adv_preds = model(adv.to(device)).argmax(dim=1).cpu()
    out_dir = attack_sample_dir(exp_dir, name, kwargs)
    save_adversarial_samples(
        out_dir, clean, adv, labels, clean_preds, adv_preds
    )
    log.info(f"Saved {clean.shape[0]} adversarial samples to {out_dir}")


def _run_attacks(
    cfg, model, datamodule, device, metrics, block, exp_dir
) -> None:
    n_save_samples = int(cfg.robustness.save_samples)
    attacks = {}
    for key, spec in (cfg.robustness.get("attacks") or {}).items():
        name = str(spec.name)
        kwargs = OmegaConf.to_container(spec.kwargs, resolve=True)
        n_examples = spec.get("n_examples")
        n_examples = int(n_examples) if n_examples is not None else None
        log.info(f"Running attack {name} ({key}) with kwargs={kwargs}")
        result = evaluate_attack(
            model=model,
            loader=datamodule.test_dataloader(),
            attack_name=name,
            attack_kwargs=kwargs,
            n_examples=n_examples,
            device=device,
            n_save_samples=n_save_samples,
        )
        log.info(
            f"{name}: robust accuracy={result['accuracy']:.4f} "
            f"on {result['n_examples']} examples"
        )
        # Raw sample tensors go to disk, never into the JSON results block.
        samples = result.pop("samples", None)
        if samples is not None:
            _save_samples(exp_dir, model, device, name, kwargs, samples)
        # Keyed by config key, not class name: two specs of the same class
        # (e.g. AutoAttack at two budgets) must not overwrite each other.
        attacks[key] = {"name": name, **result, "kwargs": kwargs}
        # Budget embedded in the key so re-runs at other budgets cannot
        # silently overwrite each other in the logger history.
        metric_key = f"robust/attack/{name}"
        if kwargs.get("eps") is not None:
            metric_key += (
                f"_{kwargs.get('norm', '')}_eps{float(kwargs['eps']):.5f}"
            )
        metrics[metric_key] = result["accuracy"]
    block["attacks"] = attacks


@utils.task_wrapper
def robustness_eval(cfg: DictConfig):
    """Run both robustness benchmarks end-to-end on one finished run."""
    if not cfg.get("exp_dir"):
        raise ValueError(
            "robustness eval requires exp_dir=<training run path>"
        )
    if cfg.get("robustness") is None or cfg.robustness.get("spec") in (
        None,
        "none",
    ):
        raise ValueError(
            "robustness eval requires a robustness config "
            "(e.g. robustness=default)"
        )

    apply_eval_adjustments(cfg)
    exp_dir = Path(str(cfg.exp_dir)).expanduser().resolve()
    if cfg.get("seed"):
        seed_everything(cfg.seed, workers=True)

    # Default to the exact best checkpoint results.json records; an explicit
    # ckpt_path / use_avg_ckpt override skips the clean-accuracy assert.
    ckpt_was_defaulted = not cfg.get("ckpt_path") and not cfg.get(
        "use_avg_ckpt"
    )
    if ckpt_was_defaulted:
        cfg.ckpt_path = str(resolve_best_checkpoint(exp_dir))
    else:
        cfg.ckpt_path = resolve_checkpoint(cfg)
    log.info(f"Evaluating checkpoint: {cfg.ckpt_path}")

    if cfg.robustness.get("batch_size"):
        # loaders.test interpolates ${.valid}, so valid is the real knob.
        cfg.datamodule.loaders.valid.batch_size = int(
            cfg.robustness.batch_size
        )

    # Move normalization out of the data pipeline and into the model so
    # attacks and corruption data operate in [0,1] pixel space.
    eval_specs, mean, std = extract_normalization(
        cfg.datamodule.transforms.eval
    )
    cfg.datamodule.transforms.eval = eval_specs

    log.info(f"Instantiating datamodule <{cfg.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(
        cfg.datamodule, _recursive_=False
    )
    datamodule.prepare_data()
    datamodule.setup(stage="test")

    log.info(f"Instantiating model <{cfg.module._target_}>")
    module: LightningModule = hydra.utils.instantiate(
        cfg.module, _recursive_=False
    )

    ckpt = torch.load(cfg.ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    # Magnitude-pruned ckpts store weight_orig/weight_mask pairs; collapse
    # them so a plain (un-pruned) module loads with strict=True.
    state = collapse_pruning_reparam(state)
    module.load_state_dict(state, strict=True)

    device = _resolve_device(str(cfg.robustness.device))
    model = NormalizedModel(module.net, mean, std).to(device).eval()
    # Attacks need gradients w.r.t. inputs only; freeze all parameters.
    model.requires_grad_(False)

    # Clean sanity check: recomputing the recorded test accuracy proves the
    # right checkpoint met the right transform pipeline.
    clean_accuracy = evaluate_accuracy(
        model, datamodule.test_dataloader(), device
    )
    reference = read_reference_test_accuracy(exp_dir)
    log.info(
        f"Clean accuracy: {clean_accuracy:.4f} (recorded: {reference:.4f})"
    )
    if ckpt_was_defaulted:
        assert abs(clean_accuracy - reference) <= _CLEAN_ACCURACY_TOLERANCE, (
            f"clean accuracy {clean_accuracy:.4f} does not reproduce the "
            f"recorded test_accuracy {reference:.4f} — wrong checkpoint or "
            "transform pipeline."
        )

    metrics = {"robust/clean": clean_accuracy}
    block = {}
    _run_corruption(cfg, model, datamodule, device, metrics, block)
    _run_attacks(cfg, model, datamodule, device, metrics, block, exp_dir)

    try:
        from importlib.metadata import version

        torchattacks_version = version("torchattacks")
    except Exception:
        torchattacks_version = None
    block["metadata"] = {
        "checkpoint_path": str(cfg.ckpt_path),
        "ckpt_epoch": ckpt.get("epoch"),
        "ckpt_global_step": ckpt.get("global_step"),
        "clean_accuracy": clean_accuracy,
        "clean_accuracy_reference": reference,
        "batch_size": int(cfg.datamodule.loaders.valid.batch_size),
        "torchattacks_version": torchattacks_version,
        "date": datetime.datetime.now().astimezone().isoformat(),
    }

    merge_robustness_results(exp_dir, block)
    log.info(f"Merged robustness block into {exp_dir / 'results.json'}")

    step = int(ckpt.get("global_step") or 0)
    loggers = build_run_loggers(exp_dir, list(cfg.robustness.loggers))
    append_metrics_to_run(loggers, metrics, step=step)
    log.info(
        f"Appended {len(metrics)} metrics at step {step} to: "
        f"{list(cfg.robustness.loggers)}"
    )

    object_dict = {"cfg": cfg, "model": model, "datamodule": datamodule}
    return metrics, object_dict


# Pre-Hydra hook: parse `exp_dir=` from CLI, swap Hydra's config search
# path to the training snapshot, inject sanitized training overrides.
# Idempotent (no-op if already prepared).
prepare_argv_for_exp_dir(_HYDRA_PARAMS)


@utils.register_custom_resolvers(**_HYDRA_PARAMS | {"overrides": sys.argv[1:]})
@hydra.main(**_HYDRA_PARAMS)
def main(cfg: DictConfig) -> None:
    robustness_eval(cfg)


if __name__ == "__main__":
    main()
