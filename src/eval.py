import re
import sys
from pathlib import Path
from typing import List, Tuple

import hydra
import pyrootutils
import yaml
from omegaconf import DictConfig, ListConfig, OmegaConf, open_dict
from pytorch_lightning import (
    LightningDataModule,
    LightningModule,
    Trainer,
    seed_everything,
)
from pytorch_lightning.loggers.logger import Logger as PLLogger

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
from src import utils  # noqa: E402

log = utils.get_pylogger(__name__)


# ---------------------------------------------------------------------------
# Cluster path utilities
# ---------------------------------------------------------------------------

_CLUSTER_PATH_RE = re.compile(r".*/dsnf101h/.*?/(train|eval)/runs/[^/]+/[^/]+")


def _fix_cluster_paths_in_config(
    cfg: DictConfig,
    exp_dir: Path,
    prefix_remaps: list[tuple[str, str]] | None = None,
) -> None:
    """Fix stale cluster paths throughout a config tree.

    Applies: (1) prefix remaps (longest-first), then (2) cluster
    run-dir regex -> exp_dir.
    """
    remaps = sorted(
        prefix_remaps or [],
        key=lambda x: len(x[0]),
        reverse=True,
    )
    exp_dir_s = str(exp_dir)

    def _fix(value: str) -> str:
        for old, new in remaps:
            if old in value:
                value = value.replace(old, new)
        if "dsnf101h" in value:
            value = _CLUSTER_PATH_RE.sub(exp_dir_s, value)
        return value

    def _recurse(node):
        if isinstance(node, DictConfig):
            items = ((k, node[k]) for k in node)
        elif isinstance(node, ListConfig):
            items = enumerate(node)
        else:
            return
        for key, value in items:
            try:
                if isinstance(value, str):
                    fixed = _fix(value)
                    if fixed != value:
                        log.info(f"Fixed path: {value} -> {fixed}")
                        node[key] = fixed
                elif isinstance(value, (DictConfig, ListConfig)):
                    _recurse(value)
            except Exception:
                pass  # read-only / interpolated nodes

    _recurse(cfg)


def _infer_prefix_remap(
    broken_path: str,
    reference_path: Path,
) -> tuple[str, str] | None:
    """Infer a path remap from a broken path and a known-good reference.

    Tries two strategies:

    1. **Prefix swap** — find >=2 consecutive common path components
       (e.g. ``dsnf/dsnf101h``) and swap the divergent prefix
       (e.g. ``/home/hpc`` → ``/home/woody``).

    2. **Ancestor search** — look for the dataset directory name
       (last component of *broken_path*, e.g. ``cnceleb``) under
       ``datasets/`` or ``data/`` beneath ancestors of *reference_path*.
       Handles the case where the directory layout differs across
       filesystems (e.g. ``repo/data/cnceleb`` on compute node vs
       ``/home/woody/.../datasets/cnceleb`` on shared storage).

    Returns ``(old_parent, new_parent)`` or ``None``.
    """
    b_parts = Path(broken_path).parts
    r_parts = reference_path.parts

    # Strategy 1: filesystem prefix swap.
    for bi in range(1, len(b_parts) - 1):
        for ri in range(1, len(r_parts) - 1):
            if (
                b_parts[bi] == r_parts[ri]
                and b_parts[bi + 1] == r_parts[ri + 1]
            ):
                old_pre = str(Path(*b_parts[:bi]))
                new_pre = str(Path(*r_parts[:ri]))
                if old_pre == new_pre:
                    continue
                candidate = broken_path.replace(old_pre, new_pre, 1)
                if Path(candidate).is_dir():
                    return (old_pre, new_pre)

    # Strategy 2: search for the dataset dir under reference ancestors.
    dataset_name = Path(broken_path).name
    broken_parent = str(Path(broken_path).parent)
    # Don't search above /home/<fs>/<group> (depth 4).
    min_depth = min(4, len(r_parts) - 1)
    for depth in range(len(r_parts) - 1, min_depth, -1):
        ancestor = Path(*r_parts[:depth])
        for sub in ("datasets", "data"):
            candidate = ancestor / sub / dataset_name
            if candidate.is_dir():
                new_parent = str(ancestor / sub)
                if new_parent != broken_parent:
                    return (broken_parent, new_parent)
    return None


# ---------------------------------------------------------------------------
# Pre-Hydra: resolve exp_dir and inject training overrides
# ---------------------------------------------------------------------------


def _resolve_exp_dir_from_argv() -> Path | None:
    """Extract exp_dir from CLI, or derive from ckpt_path."""
    for arg in sys.argv[1:]:
        if arg.startswith("exp_dir="):
            return Path(arg.split("=", 1)[1]).expanduser().resolve()
        if arg.startswith("ckpt_path="):
            return Path(arg.split("=", 1)[1]).resolve().parents[1]
    return None


def _load_exp_overrides(exp_dir: Path) -> list[str]:
    """Load training-time Hydra overrides from exp_dir."""
    path = exp_dir / ".hydra" / "overrides.yaml"
    if not path.exists():
        return []
    with path.open("r") as f:
        data = yaml.safe_load(f) or []
    return [str(item) for item in data] if isinstance(data, list) else []


def _sanitize_overrides_for_eval(
    overrides: list[str],
    exp_dir: Path,
    keep_logger: bool,
) -> list[str]:
    """Filter training-only overrides and fix cluster paths."""
    training_only_prefixes = (
        "trainer",
        "datamodule.loaders.train.",
        "datamodule.loaders.val",
        "ckpt_path=",
        "name=",
        "use_training_configs=",
    )

    skipped_training: list[str] = []
    skipped_logger: list[str] = []
    sanitized: list[str] = []

    for s in overrides:
        if s.startswith(training_only_prefixes):
            skipped_training.append(s)
            continue

        # Skip bare-path overrides (not valid Hydra overrides).
        if "=" not in s:
            if s.startswith("/") or (
                not s.startswith("~") and not s.startswith("+")
            ):
                log.info(f"Skipping invalid override: {s}")
                continue

        if not keep_logger and (
            s.startswith("logger=") or s.startswith("logger.")
        ):
            skipped_logger.append(s)
            continue

        # Fix or drop cluster paths.
        if "dsnf101h" in s:
            key = s.split("=", 1)[0]
            if key.startswith("paths.") or "data_dir" in key:
                log.info(f"Skipping cluster path override: {s}")
                continue
            s = _CLUSTER_PATH_RE.sub(str(exp_dir), s)

        if s.startswith("experiment="):
            sanitized.append("+" + s)
        else:
            sanitized.append(s)

    if skipped_training:
        log.info(
            f"Dropped {len(skipped_training)} training-only "
            f"override(s): {skipped_training}"
        )
    if skipped_logger:
        log.info(
            f"Dropped {len(skipped_logger)} logger override(s) "
            f"(pass keep_logger=true to retain): {skipped_logger}"
        )

    # Point to the local artifacts directory.
    artifacts_dirs = [
        p
        for p in exp_dir.glob("*_artifacts")
        if not p.name.startswith("_") and not p.name.startswith("test")
    ]
    assert (
        len(artifacts_dirs) == 1
    ), f"Expected 1 artifacts dir, found {len(artifacts_dirs)}"
    sanitized.append(
        "datamodule.dataset.artifacts_dir=" f"{artifacts_dirs[0].as_posix()}"
    )
    return sanitized


def inject_exp_dir_overrides() -> None:
    """Pre-Hydra hook: resolve exp_dir, inject training overrides.

    When exp_dir is available (from CLI or derived from ckpt_path):
    1. Ensure exp_dir is in sys.argv for Hydra
    2. Optionally switch config_path to training-time configs
    3. Load + sanitize training overrides, append to sys.argv
    """
    exp_dir = _resolve_exp_dir_from_argv()
    if exp_dir is None:
        return
    if not exp_dir.exists():
        raise FileNotFoundError(f"exp_dir does not exist: {exp_dir}")

    if not any(arg.startswith("exp_dir=") for arg in sys.argv[1:]):
        sys.argv.append(f"exp_dir={exp_dir.as_posix()}")

    # Optionally use training-time configs from metadata/.
    use_training_configs = not any(
        a.startswith("use_training_configs=")
        and a.split("=", 1)[1].lower() in ("0", "false")
        for a in sys.argv[1:]
    )
    metadata_configs = exp_dir / "metadata" / "configs"

    if use_training_configs:
        if not metadata_configs.is_dir():
            raise FileNotFoundError(
                f"use_training_configs=True but "
                f"{metadata_configs} not found"
            )
        log.info(f"Using training-time configs from {metadata_configs}")
        # Copy repo eval.yaml so Hydra can find it in metadata dir.
        repo_eval = (
            Path(_HYDRA_PARAMS["config_path"]) / _HYDRA_PARAMS["config_name"]
        )
        utils.utils.copy_yaml(
            repo_eval,
            metadata_configs / _HYDRA_PARAMS["config_name"],
        )
        _HYDRA_PARAMS["config_path"] = str(metadata_configs)

    # Load training overrides (user CLI args take precedence).
    overrides = _load_exp_overrides(exp_dir)
    keep_logger = any(
        a.startswith("keep_logger=")
        and a.split("=", 1)[1].lower() in ("1", "true")
        for a in sys.argv[1:]
    )
    new_overrides = _sanitize_overrides_for_eval(
        overrides, exp_dir, keep_logger
    )

    skipped: list[str] = []
    for ov in new_overrides:
        key = ov.lstrip("+~").split("=")[0]
        user_has_key = any(
            arg.lstrip("+~").split("=")[0] == key
            for arg in sys.argv[1:]
            if "=" in arg
        )
        if user_has_key:
            skipped.append(ov)
        else:
            sys.argv.append(ov)

    if skipped:
        log.info(
            f"Skipped {len(skipped)} training override(s) "
            f"superseded by CLI: {skipped}"
        )


# ---------------------------------------------------------------------------
# Post-Hydra: replace composed config with training config + eval patches
# ---------------------------------------------------------------------------


def _apply_exp_dir_config(cfg: DictConfig) -> DictConfig:
    """Replace Hydra-composed config with training config + eval patches.

    Loads .hydra/config.yaml from exp_dir, then layers eval-specific
    defaults and user CLI overrides on top.

    Order: (1) transfer eval keys, (2) fix stale paths,
    (3) eval adjustments, (4) re-apply user CLI overrides.
    """
    exp_dir_value = cfg.get("exp_dir")
    if not exp_dir_value:
        return cfg

    exp_dir = Path(str(exp_dir_value)).expanduser().resolve()
    exp_cfg_path = exp_dir / ".hydra" / "config.yaml"
    if not exp_cfg_path.exists():
        raise FileNotFoundError(f"Missing experiment config: {exp_cfg_path}")

    exp_cfg = OmegaConf.load(exp_cfg_path)

    # Capture old paths before overlaying local eval paths.
    old_paths = {
        k: OmegaConf.select(exp_cfg, f"paths.{k}")
        for k in ("data_dir", "root_dir")
        if OmegaConf.select(exp_cfg, f"paths.{k}") is not None
    }

    # 1. Transfer eval-specific keys from Hydra-composed config.
    eval_keys = [
        "predict",
        "use_avg_ckpt",
        "ckpt_path",
        "ckpt_dir",
        "ckpt_avg_num",
        "ckpt_avg_min",
        "extras",
        "paths",
        "tags",
    ]
    preserved = [k for k in eval_keys if cfg.get(k) is not None]
    for key in preserved:
        OmegaConf.update(exp_cfg, key, cfg.get(key), merge=True)
    if preserved:
        log.info(f"Preserved eval-specific keys: {preserved}")

    # Clear ckpt_path if not explicitly provided — prevents stale
    # training-time values (e.g. from resumed training) from leaking
    # through and overriding use_avg_ckpt.
    if "ckpt_path" not in preserved:
        with open_dict(exp_cfg):
            exp_cfg.ckpt_path = None

    # 2. Fix stale paths (prefix remaps + cluster run-dir regex).
    # Skip remaps where the old path still exists — the training paths are
    # already valid on the current machine (e.g. running eval on the cluster
    # where /home/woody/... is mounted).
    prefix_remaps = [
        (
            str(old_val),
            str(OmegaConf.select(exp_cfg, f"paths.{k}")),
        )
        for k, old_val in old_paths.items()
        if str(old_val) != str(OmegaConf.select(exp_cfg, f"paths.{k}") or "")
        and not Path(str(old_val)).exists()
    ]
    if prefix_remaps:
        log.info(f"Path prefix remaps: {prefix_remaps}")
    _fix_cluster_paths_in_config(exp_cfg, exp_dir, prefix_remaps)

    # 2b. Auto-fix HPC filesystem mismatches.
    # After standard remaps, check whether data_dir actually exists.
    # If not, infer the correct filesystem prefix by comparing the
    # broken data_dir against exp_dir (which is known to be valid).
    data_dir_str = str(
        OmegaConf.select(exp_cfg, "datamodule.dataset.data_dir") or ""
    )
    if data_dir_str and not Path(data_dir_str).is_dir():
        remap = _infer_prefix_remap(data_dir_str, exp_dir)
        if remap:
            old_pre, new_pre = remap
            log.info(
                f"HPC auto-remap: {old_pre} -> {new_pre} "
                f"(data_dir {data_dir_str!r} did not exist)"
            )
            _fix_cluster_paths_in_config(exp_cfg, exp_dir, [remap])
        else:
            log.warning(
                f"data_dir does not exist and no HPC prefix "
                f"remap could be inferred: {data_dir_str}"
            )

    # 3. Eval-specific adjustments.
    exp_cfg.task_name = "eval"

    if not cfg.get("keep_logger"):
        exp_cfg.logger = None

    if exp_cfg.get("use_avg_ckpt") and not exp_cfg.get("ckpt_dir"):
        exp_cfg.ckpt_dir = str(exp_dir / "checkpoints")

    # Resolve artifacts directory.
    artifacts_dirs = [
        p
        for p in sorted(exp_dir.glob("*_artifacts"))
        if not p.name.startswith("test")
    ]
    if len(artifacts_dirs) > 1:
        raise NotImplementedError(
            f"Multiple artifacts dirs in {exp_dir}: "
            f"{[p.name for p in artifacts_dirs]}. "
            f"multi_sv evaluation is not supported yet."
        )
    assert artifacts_dirs, f"No artifacts dir found in {exp_dir}"
    OmegaConf.update(
        exp_cfg,
        "datamodule.dataset.artifacts_dir",
        artifacts_dirs[0].as_posix(),
        merge=True,
    )

    # Strip data_augmentation (training-only). Resolve num_classes
    # first since it may reference data_augmentation via interpolation.
    if OmegaConf.select(exp_cfg, "module.data_augmentation") is not None:
        try:
            num_classes = OmegaConf.select(
                exp_cfg,
                "datamodule.num_classes",
                throw_on_missing=True,
            )
            if num_classes is not None:
                with open_dict(exp_cfg):
                    OmegaConf.update(
                        exp_cfg,
                        "datamodule.num_classes",
                        num_classes,
                    )
        except Exception:
            pass  # already concrete or doesn't exist

        log.info("Removing module.data_augmentation (training-only)")
        with open_dict(exp_cfg):
            del exp_cfg.module["data_augmentation"]

    # 4. Re-apply user CLI overrides (always win).
    applied_cli: list[str] = []
    for arg in _USER_ARGV:
        if "=" not in arg or arg.startswith("exp_dir="):
            continue
        key = arg.lstrip("+~").split("=")[0]
        # Defaults-list overrides handled during Hydra composition.
        if "/" in key:
            continue
        value = OmegaConf.select(cfg, key)
        if value is not None:
            OmegaConf.update(exp_cfg, key, value, merge=True)
            applied_cli.append(arg)
    if applied_cli:
        log.info(f"Applied CLI override(s): {applied_cli}")

    return exp_cfg


# ---------------------------------------------------------------------------
# TEMPORARY: backward-compatibility for cnceleb concat changes
# ---------------------------------------------------------------------------


def _strip_stale_concat_rows(cfg: DictConfig) -> None:
    """Remove concatenated utterance rows from cnceleb train.csv.

    WARNING — TEMPORARY WORKAROUND
    The naming convention generated by concat_short_utterances.py was
    changed from a global 6-digit index (``concat_013457.wav``) to a
    per-(speaker, genre) 4-digit index (``concat_0000.wav``).  Old
    experiment artifacts still contain CSVs that reference the old
    filenames, which no longer exist on disk.

    This function strips those rows so evaluation can proceed without
    re-preparing the dataset.  It should be removed once all legacy
    experiments have been re-prepared or archived.
    """
    import pandas as pd

    artifacts_dir_str = OmegaConf.select(
        cfg, "datamodule.dataset.artifacts_dir"
    )
    if not artifacts_dir_str:
        return

    artifacts_dir = Path(str(artifacts_dir_str))
    if "cnceleb" not in artifacts_dir.name.lower():
        return

    train_csv = artifacts_dir / "train.csv"
    if not train_csv.exists():
        return

    df = pd.read_csv(train_csv, sep="|")

    if "is_concatenated" not in df.columns:
        return

    n_concat = df["is_concatenated"].sum()
    if n_concat == 0:
        return

    log.warning(
        f"TEMPORARY: Stripping {n_concat} concatenated rows from {train_csv} "
        "(backward-incompatible naming change in concat_short_utterances.py)"
    )

    df = df[~df["is_concatenated"]].reset_index(drop=True)
    df.to_csv(train_csv, sep="|", index=False)

    log.warning(
        f"TEMPORARY: train.csv rewritten with {len(df)} rows "
        "(concat rows removed). Remove this workaround once "
        "legacy experiments are re-prepared."
    )


# ---------------------------------------------------------------------------
# Checkpoint resolution
# ---------------------------------------------------------------------------


def _extract_metric_from_checkpoint(
    ckpt_path: Path,
) -> float | None:
    """Extract metric_valid value from checkpoint filename."""
    match = re.search(r"metric_valid([0-9]+(?:\.[0-9]+)?)", ckpt_path.name)
    return float(match.group(1)) if match else None


def _resolve_checkpoint(cfg: DictConfig) -> str:
    """Resolve checkpoint: explicit path or averaged from directory."""
    ckpt_path = cfg.get("ckpt_path")
    if ckpt_path:
        ckpt_path = Path(str(ckpt_path)).expanduser()
        if ckpt_path.exists():
            if cfg.get("use_avg_ckpt"):
                log.warning(
                    "Both ckpt_path and use_avg_ckpt=True; " "using ckpt_path"
                )
            log.info(f"Using checkpoint: {ckpt_path}")
            return str(ckpt_path)
        if not cfg.get("use_avg_ckpt"):
            raise ValueError(f"Checkpoint not found: {ckpt_path}")
        log.warning(
            f"ckpt_path not found ({ckpt_path}); "
            "falling back to use_avg_ckpt"
        )

    if cfg.get("use_avg_ckpt"):
        return _resolve_averaged_checkpoint(cfg)

    raise ValueError("ckpt_path is required when use_avg_ckpt is False")


def _resolve_averaged_checkpoint(cfg: DictConfig) -> str:
    """Average top N checkpoints or reuse existing averaged one."""
    ckpt_dir = cfg.get("ckpt_dir")
    if not ckpt_dir:
        raise ValueError("use_avg_ckpt=True requires ckpt_dir")

    ckpt_dir = Path(str(ckpt_dir)).expanduser()
    if not ckpt_dir.exists():
        raise ValueError(f"Checkpoint directory not found: {ckpt_dir}")

    avg_num = cfg.get("ckpt_avg_num")  # None = all available
    avg_min = cfg.get("ckpt_avg_min", 2)

    # Reuse existing averaged checkpoint if available.
    existing = sorted(
        ckpt_dir.glob("averaged_top*.ckpt"),
        key=lambda p: int(p.stem.split("top")[1]),
        reverse=True,
    )
    if existing:
        log.info(f"Reusing averaged checkpoint: {existing[0]}")
        return str(existing[0])

    # Find candidates (exclude averaged and last.ckpt).
    log.warning(f"No averaged checkpoint in {ckpt_dir}; creating one.")
    candidates = [
        p
        for p in ckpt_dir.glob("*.ckpt")
        if not p.name.startswith("averaged_") and p.name != "last.ckpt"
    ]

    # Sort by metric from filename, fall back to mtime.
    assert cfg.callbacks.get("model_checkpoint") is not None
    ckpt_mode = cfg.callbacks.model_checkpoint.get("mode")
    assert ckpt_mode in ("min", "max")

    with_metric = [
        (p, m)
        for p in candidates
        if (m := _extract_metric_from_checkpoint(p)) is not None
    ]
    if with_metric:
        with_metric.sort(key=lambda x: x[1], reverse=(ckpt_mode == "max"))
        candidates = [p for p, _ in with_metric]
        log.info(
            f"Sorted by metric (mode={ckpt_mode}), "
            f"best: {with_metric[0][1]}"
        )
    else:
        log.warning(
            "Cannot extract metrics from filenames; "
            "falling back to modification time"
        )
        candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)

    if len(candidates) < avg_min:
        raise ValueError(
            f"Not enough checkpoints: " f"{len(candidates)} < {avg_min}"
        )

    if avg_num is not None:
        candidates = candidates[:avg_num]
    output_path = ckpt_dir / f"averaged_top{len(candidates)}.ckpt"

    log.info(
        f"Averaging {len(candidates)} checkpoints: "
        f"{[p.name for p in candidates]}"
    )
    result = utils.average_checkpoints(candidates, output_path)
    if not result:
        raise ValueError("Failed to create averaged checkpoint")
    return str(result)


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------


@utils.task_wrapper
def evaluate(cfg: DictConfig) -> Tuple[dict, dict]:
    """Evaluate a checkpoint on a datamodule's test set."""

    ckpt_path_val = cfg.get("ckpt_path")
    exp_dir_val = cfg.get("exp_dir")

    if not ckpt_path_val and not exp_dir_val:
        raise ValueError(
            "Either 'ckpt_path' or 'exp_dir' must be provided.\n"
            "  ckpt_path: path to a specific checkpoint\n"
            "  exp_dir:   path to a training run directory"
        )

    # Derive exp_dir from ckpt_path if not provided.
    if ckpt_path_val and not exp_dir_val:
        ckpt = Path(str(ckpt_path_val)).expanduser().resolve()
        if not ckpt.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
        derived = ckpt.parents[1]
        if not (derived / ".hydra" / "config.yaml").exists():
            raise ValueError(f"No training config at {derived / '.hydra'}")
        cfg.exp_dir = str(derived)
        log.info(f"Derived exp_dir from ckpt_path: {cfg.exp_dir}")

    # Default to averaged checkpoint when only exp_dir is given.
    if exp_dir_val and not ckpt_path_val:
        if not cfg.get("use_avg_ckpt"):
            cfg.use_avg_ckpt = True
            log.warning("No ckpt_path; defaulting to use_avg_ckpt=True")

    cfg = _apply_exp_dir_config(cfg)
    _strip_stale_concat_rows(cfg)  # TEMPORARY — see docstring
    utils.log_gpu_memory_metadata()

    if cfg.get("seed"):
        log.info(f"Seed: {cfg.seed}")
        seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(
        cfg.datamodule, _recursive_=False
    )

    log.info(f"Instantiating model <{cfg.module._target_}>")
    model: LightningModule = hydra.utils.instantiate(
        cfg.module, _recursive_=False
    )

    log.info("Instantiating loggers...")
    logger: List[PLLogger] = utils.instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, logger=logger)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        utils.log_hyperparameters(object_dict)

    cfg.ckpt_path = _resolve_checkpoint(cfg)
    log.info(f"Checkpoint: {cfg.ckpt_path}")

    if cfg.get("predict"):
        log.info("Starting prediction!")
        predictions = trainer.predict(
            model=model,
            datamodule=datamodule,
            ckpt_path=cfg.ckpt_path,
        )
        utils.save_predictions(
            predictions=predictions,
            dirname=cfg.paths.output_dir,
            **cfg.extras.predictions_saving_params,
        )
    else:
        log.info("Starting testing!")
        trainer.test(
            model=model,
            datamodule=datamodule,
            ckpt_path=cfg.ckpt_path,
        )

    return trainer.callback_metrics, object_dict


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

# Snapshot user CLI args before inject_exp_dir_overrides appends
# training overrides — used in _apply_exp_dir_config to distinguish
# genuine CLI overrides from injected ones.
_USER_ARGV = list(sys.argv[1:])
inject_exp_dir_overrides()


@utils.register_custom_resolvers(**_HYDRA_PARAMS | {"overrides": sys.argv[1:]})
@hydra.main(**_HYDRA_PARAMS)
def main(cfg: DictConfig) -> None:
    evaluate(cfg)


if __name__ == "__main__":
    main()
