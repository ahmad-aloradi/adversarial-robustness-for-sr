"""Shared preprocessing helpers for loading a prior training run by `exp_dir=`.

Used by `src/eval.py` (eval entry point) and `scripts/quantize_ptq.py`
(PTQ entry point). The flow is single-stage:

  1. Pre-Hydra (`prepare_argv_for_exp_dir`): parse `exp_dir=` from sys.argv,
     swap Hydra's `config_path` to the training-time snapshot at
     `{exp_dir}/metadata/configs/`, mirror config groups added after that
     snapshot was taken (e.g. `quantization/`), load + sanitize the saved
     overrides, append them to sys.argv. Idempotent.

  2. Hydra composes normally. The composed cfg IS the final cfg — there
     is no separate post-Hydra disk load.

  3. Post-Hydra (`apply_eval_adjustments`): in-place adjustments only —
     cluster-path remaps, HPC auto-remap, eval-specific cfg edits,
     artifacts-dir injection (idempotent; covers the in-process test
     path where step 1 was bypassed).
"""

from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path

import yaml
from omegaconf import DictConfig, ListConfig, OmegaConf, open_dict
from omegaconf.errors import OmegaConfBaseException

from src.utils.pylogger import get_pylogger
from src.utils.utils import average_checkpoints, copy_yaml

log = get_pylogger(__name__)


# Config groups present in the live repo that did not exist at the time
# older training snapshots were taken. Mirrored into the metadata snapshot
# so Hydra can compose the defaults list without MissingConfigException.
_MIRRORED_GROUPS: tuple[str, ...] = ("quantization",)

# Idempotency guard. `prepare_argv_for_exp_dir` is called explicitly from
# each entry point; if it fires twice (e.g. a script imports another
# script that also calls it), the second call short-circuits.
_PREPARED: bool = False

_CLUSTER_PATH_RE = re.compile(r".*/dsnf101h/.*?/(train|eval)/runs/[^/]+/[^/]+")


# ---------------------------------------------------------------------------
# Cluster / HPC path remap
# ---------------------------------------------------------------------------


def _fix_cluster_paths_in_config(
    cfg: DictConfig,
    exp_dir: Path,
    prefix_remaps: list[tuple[str, str]] | None = None,
) -> None:
    """Fix stale cluster paths throughout a config tree, in place.

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
            except OmegaConfBaseException as exc:
                # Read-only nodes, unresolved interpolations, missing
                # mandatory values — none of these should abort the
                # whole sweep, but we log so they aren't truly silent.
                log.debug(f"Skipping {key!r} during path remap: {exc}")

    _recurse(cfg)


def _infer_prefix_remap(
    broken_path: str,
    reference_path: Path,
) -> tuple[str, str] | None:
    """Infer a path remap from a broken path and a known-good reference.

    Strategies:
      1. Prefix swap — find >=2 consecutive common path components and
         swap the divergent prefix.
      2. Ancestor search — look for the dataset directory name under
         ``datasets/`` or ``data/`` beneath ancestors of the reference.
    """
    b_parts = Path(broken_path).parts
    r_parts = reference_path.parts

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

    dataset_name = Path(broken_path).name
    broken_parent = str(Path(broken_path).parent)
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
# Pre-Hydra: argv preparation
# ---------------------------------------------------------------------------


def _resolve_exp_dir_from_argv() -> Path | None:
    for arg in sys.argv[1:]:
        if arg.startswith("exp_dir="):
            return Path(arg.split("=", 1)[1]).expanduser().resolve()
        if arg.startswith("ckpt_path="):
            return Path(arg.split("=", 1)[1]).resolve().parents[1]
    return None


def _load_exp_overrides(exp_dir: Path) -> list[str]:
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
    """Filter training-only overrides, fix cluster paths, inject artifacts dirs."""
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

    artifacts_dirs = [
        p
        for p in exp_dir.glob("*_artifacts")
        if not p.name.startswith("_") and not p.name.startswith("test")
    ]
    assert artifacts_dirs, f"No artifacts dirs found in {exp_dir}"
    if len(artifacts_dirs) == 1:
        sanitized.append(
            "datamodule.dataset.artifacts_dir="
            f"{artifacts_dirs[0].as_posix()}"
        )
    else:
        for d in artifacts_dirs:
            ds_name = d.name[: -len("_artifacts")]
            sanitized.append(
                f"datamodule.datasets.{ds_name}.datamodule.dataset.artifacts_dir={d.as_posix()}"
            )
    return sanitized


def prepare_argv_for_exp_dir(hydra_params: dict) -> Path | None:
    """Pre-Hydra hook. Idempotent.

    If `exp_dir=` (or `ckpt_path=` from which exp_dir can be derived) is
    present in sys.argv, swap `hydra_params["config_path"]` to the
    training-time `{exp_dir}/metadata/configs/` snapshot (unless
    `use_training_configs=False` is passed), mirror missing config
    groups into that snapshot, and append sanitized training overrides
    to sys.argv. Returns the resolved exp_dir or None.
    """
    global _PREPARED
    if _PREPARED:
        return _resolve_exp_dir_from_argv()
    _PREPARED = True

    exp_dir = _resolve_exp_dir_from_argv()
    if exp_dir is None:
        return None
    if not exp_dir.exists():
        raise FileNotFoundError(f"exp_dir does not exist: {exp_dir}")

    if not any(arg.startswith("exp_dir=") for arg in sys.argv[1:]):
        sys.argv.append(f"exp_dir={exp_dir.as_posix()}")

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
        repo_configs = Path(hydra_params["config_path"])
        repo_eval = repo_configs / hydra_params["config_name"]
        copy_yaml(repo_eval, metadata_configs / hydra_params["config_name"])
        for group in _MIRRORED_GROUPS:
            src_group = repo_configs / group
            if src_group.is_dir():
                shutil.copytree(
                    src_group,
                    metadata_configs / group,
                    dirs_exist_ok=True,
                )
        hydra_params["config_path"] = str(metadata_configs)

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

    return exp_dir


# ---------------------------------------------------------------------------
# Post-Hydra: in-place cfg adjustments
# ---------------------------------------------------------------------------


def ensure_use_avg_ckpt_default(cfg: DictConfig) -> None:
    """If exp_dir is set and ckpt_path is unset, force use_avg_ckpt=True."""
    if cfg.get("exp_dir") and not cfg.get("ckpt_path"):
        if not cfg.get("use_avg_ckpt"):
            cfg.use_avg_ckpt = True
            log.warning("No ckpt_path; defaulting to use_avg_ckpt=True")


def apply_eval_adjustments(cfg: DictConfig) -> None:
    """Post-Hydra, in-place. Does NOT replace cfg.

    Performs: cluster-path remaps, HPC auto-remap for missing data_dir,
    `task_name=eval`, drop `module.data_augmentation`, logger null-out
    unless `keep_logger`, default `ckpt_dir` when `use_avg_ckpt`, and
    artifacts-dir injection (idempotent — covers the in-process path
    where `prepare_argv_for_exp_dir` was bypassed).
    """
    exp_dir_value = cfg.get("exp_dir")
    if not exp_dir_value:
        return

    exp_dir = Path(str(exp_dir_value)).expanduser().resolve()

    # Cluster regex sweep on the already-composed cfg.
    _fix_cluster_paths_in_config(cfg, exp_dir, prefix_remaps=None)

    # HPC auto-remap when data_dir is missing on this host.
    data_dir_str = str(
        OmegaConf.select(cfg, "datamodule.dataset.data_dir") or ""
    )
    if data_dir_str and not Path(data_dir_str).is_dir():
        remap = _infer_prefix_remap(data_dir_str, exp_dir)
        if remap:
            old_pre, new_pre = remap
            log.info(
                f"HPC auto-remap: {old_pre} -> {new_pre} "
                f"(data_dir {data_dir_str!r} did not exist)"
            )
            _fix_cluster_paths_in_config(cfg, exp_dir, [remap])
        else:
            log.warning(
                f"data_dir does not exist and no HPC prefix "
                f"remap could be inferred: {data_dir_str}"
            )

    cfg.task_name = "eval"

    if not cfg.get("keep_logger"):
        cfg.logger = None

    if cfg.get("use_avg_ckpt") and not cfg.get("ckpt_dir"):
        cfg.ckpt_dir = str(exp_dir / "checkpoints")

    # Idempotent artifacts-dir injection. Covers the in-process test path
    # where `prepare_argv_for_exp_dir` did not run. When the same value
    # is already present (CLI path), this is a harmless no-op merge.
    artifacts_dirs = [
        p
        for p in sorted(exp_dir.glob("*_artifacts"))
        if not p.name.startswith("test")
    ]
    if artifacts_dirs:
        if len(artifacts_dirs) == 1:
            OmegaConf.update(
                cfg,
                "datamodule.dataset.artifacts_dir",
                artifacts_dirs[0].as_posix(),
                merge=True,
            )
        else:
            for d in artifacts_dirs:
                ds_name = d.name[: -len("_artifacts")]
                OmegaConf.update(
                    cfg,
                    f"datamodule.datasets.{ds_name}.datamodule.dataset.artifacts_dir",
                    d.as_posix(),
                    merge=True,
                )

    # Strip data_augmentation (training-only). Resolve num_classes first
    # since it may reference data_augmentation via interpolation.
    if OmegaConf.select(cfg, "module.data_augmentation") is not None:
        try:
            num_classes = OmegaConf.select(
                cfg,
                "datamodule.num_classes",
                throw_on_missing=True,
            )
            if num_classes is not None:
                with open_dict(cfg):
                    OmegaConf.update(
                        cfg,
                        "datamodule.num_classes",
                        num_classes,
                    )
        except OmegaConfBaseException as exc:
            # Either num_classes is already concrete (no interpolation to
            # resolve) or genuinely missing — both are fine; this block is
            # a best-effort cache of an interpolated value.
            log.debug(f"num_classes resolution skipped: {exc}")

        log.info("Removing module.data_augmentation (training-only)")
        with open_dict(cfg):
            del cfg.module["data_augmentation"]


# ---------------------------------------------------------------------------
# Checkpoint resolution
# ---------------------------------------------------------------------------


def _extract_metric_from_checkpoint(ckpt_path: Path) -> float | None:
    match = re.search(r"metric_valid([0-9]+(?:\.[0-9]+)?)", ckpt_path.name)
    return float(match.group(1)) if match else None


def resolve_checkpoint(cfg: DictConfig) -> str:
    """Resolve a checkpoint: explicit `cfg.ckpt_path` or averaged from `ckpt_dir`."""
    ckpt_path = cfg.get("ckpt_path")
    if ckpt_path:
        ckpt_path = Path(str(ckpt_path)).expanduser()
        if ckpt_path.exists():
            if cfg.get("use_avg_ckpt"):
                log.warning(
                    "Both ckpt_path and use_avg_ckpt=True; using ckpt_path"
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
    ckpt_dir = cfg.get("ckpt_dir")
    if not ckpt_dir:
        raise ValueError("use_avg_ckpt=True requires ckpt_dir")

    ckpt_dir = Path(str(ckpt_dir)).expanduser()
    if not ckpt_dir.exists():
        raise ValueError(f"Checkpoint directory not found: {ckpt_dir}")

    avg_num = cfg.get("ckpt_avg_num")
    avg_min = cfg.get("ckpt_avg_min", 2)

    existing = sorted(
        ckpt_dir.glob("averaged_top*.ckpt"),
        key=lambda p: int(p.stem.split("top")[1]),
        reverse=True,
    )
    if existing:
        log.info(f"Reusing averaged checkpoint: {existing[0]}")
        return str(existing[0])

    log.warning(f"No averaged checkpoint in {ckpt_dir}; creating one.")
    candidates = [
        p
        for p in ckpt_dir.glob("*.ckpt")
        if not p.name.startswith("averaged_") and p.name != "last.ckpt"
    ]

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
            f"Not enough checkpoints: {len(candidates)} < {avg_min}"
        )

    if avg_num is not None:
        candidates = candidates[:avg_num]
    output_path = ckpt_dir / f"averaged_top{len(candidates)}.ckpt"

    log.info(
        f"Averaging {len(candidates)} checkpoints: "
        f"{[p.name for p in candidates]}"
    )
    result = average_checkpoints(candidates, output_path)
    if not result:
        raise ValueError("Failed to create averaged checkpoint")
    return str(result)
