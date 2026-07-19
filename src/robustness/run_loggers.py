"""Rebuild a finished training run's loggers to append post-hoc metrics.

``apply_eval_adjustments`` nulls ``cfg.logger`` during eval reloads, so the
robustness eval reconstructs loggers straight from the run's own
``.hydra/config.yaml`` logger block: wandb resumes the original dashboard
entry by parsed id (same convention as ``scripts/fabfile.py``), csv and
tensorboard append as a fresh ``version_1`` under the run dir (which
``scripts/visualize.py`` already globs via ``version_*``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import yaml
from pytorch_lightning.loggers import CSVLogger

from src.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


def find_wandb_run_id(exp_dir: Path) -> str | None:
    """Newest wandb run id under ``{exp_dir}/wandb``, or None.

    WandbLogger names each run dir ``run-{timestamp}-{id}``
    (``offline-run-...`` when offline); the id is the token after the final
    ``-``. Local port of ``scripts/fabfile.py:_find_wandb_run_id``.
    """
    wandb_dir = Path(exp_dir) / "wandb"
    if not wandb_dir.is_dir():
        return None
    candidates = [
        p for p in wandb_dir.iterdir() if p.is_dir() and "run-" in p.name
    ]
    if not candidates:
        return None
    newest = max(candidates, key=lambda p: p.stat().st_mtime)
    if newest.name.startswith("offline-run-"):
        log.warning(
            f"Newest wandb dir is offline ({newest.name}); resuming its id "
            "online may create a new dashboard entry."
        )
    return newest.name.rsplit("-", 1)[-1]


def _load_logger_block(exp_dir: Path) -> dict:
    config_path = Path(exp_dir) / ".hydra" / "config.yaml"
    assert config_path.exists(), f"{config_path} not found"
    # safe_load, not OmegaConf: values like ${paths.output_dir} are
    # unresolved interpolations we are about to overwrite anyway.
    config = yaml.safe_load(config_path.read_text())
    block = (config or {}).get("logger")
    assert block, f"{config_path} has no logger block to rebuild from"
    return block


def build_run_loggers(exp_dir: Path, selected: Sequence[str]) -> list:
    """Instantiate the run's loggers with save dirs re-anchored to exp_dir.

    ``selected`` names must exist in the run's logger block (KeyError
    otherwise) — a run trained without wandb cannot have wandb appended.
    """
    import hydra

    exp_dir = Path(exp_dir).resolve()
    block = _load_logger_block(exp_dir)

    loggers = []
    for name in selected:
        if name not in block:
            raise KeyError(
                f"logger {name!r} not in the run's logger block "
                f"({sorted(block)}); it cannot be appended to."
            )
        spec = dict(block[name])
        if name == "csv":
            spec["save_dir"] = str(exp_dir)  # -> {exp_dir}/csv/version_1/
        elif name == "tensorboard":
            spec["save_dir"] = str(exp_dir / "tensorboard")
        elif name == "wandb":
            run_id = find_wandb_run_id(exp_dir)
            assert run_id is not None, (
                f"no wandb run dir under {exp_dir}/wandb — cannot resume; "
                "drop 'wandb' from robustness.loggers."
            )
            spec["save_dir"] = str(exp_dir)
            spec["id"] = run_id
            spec["name"] = None  # keep the original run name
            spec["resume"] = "allow"
        else:
            spec["save_dir"] = str(exp_dir)
        loggers.append(hydra.utils.instantiate(spec))
    return loggers


def _summary_metrics(metrics: dict[str, float]) -> dict[str, float]:
    """Drop the per-corruption-type series, keep the per-severity mean.

    The full breakdown is 19x5 (CIFAR) series; only the ``.../mean/...`` rows,
    plus clean and attack accuracy, belong on the dashboards.
    """
    return {
        k: v
        for k, v in metrics.items()
        if not k.startswith("robust/corruption/")
        or k.startswith("robust/corruption/mean/")
    }


def append_metrics_to_run(
    loggers: list, metrics: dict[str, float], step: int
) -> None:
    """Log ``metrics`` at ``step`` on every logger, then flush and close.

    Only CSV gets the full per-type corruption breakdown; every other logger
    (wandb, tensorboard) gets the summary (clean, per-severity mean, attacks).
    """
    assert metrics, "no metrics to append"
    summary = _summary_metrics(metrics)
    for logger in loggers:
        payload = metrics if isinstance(logger, CSVLogger) else summary
        logger.log_metrics(payload, step=step)
        logger.save()
        logger.finalize("success")


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        exp = Path(tmp)
        (exp / "wandb" / "run-20260101_000000-abc123").mkdir(parents=True)
        (exp / "wandb" / "run-20260102_000000-def456").mkdir()
        print("id:", find_wandb_run_id(exp))  # def456 (newest by mtime)
        print("empty:", find_wandb_run_id(Path(tmp) / "nope"))  # None
