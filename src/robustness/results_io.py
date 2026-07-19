"""Read/merge access to a finished run's ``results.json``.

``src.modules.img._write_results`` overwrites results.json wholesale at
train/test end, so the robustness block must be merged read-modify-write —
and a re-train correctly invalidates any previously merged block.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path


def _read_results(exp_dir: Path) -> dict:
    path = exp_dir / "results.json"
    assert path.exists(), (
        f"{path} not found — robustness eval only runs on finished training "
        "runs (it never creates results.json)."
    )
    with path.open("r") as f:
        results = json.load(f)
    assert "best_checkpoint" in results, f"{path} has no best_checkpoint block"
    return results


def resolve_best_checkpoint(exp_dir: Path) -> Path:
    """The exact ckpt whose ``test_accuracy`` is recorded in results.json.

    The stored ``checkpoint_path`` may carry a stale absolute prefix
    (cluster runs evaluated elsewhere), so only its basename is trusted,
    re-anchored under ``{exp_dir}/checkpoints/``.
    """
    exp_dir = Path(exp_dir)
    best = _read_results(exp_dir)["best_checkpoint"]
    assert "test_accuracy" in best, (
        f"{exp_dir}/results.json best_checkpoint has no test_accuracy — "
        "the run never completed its test pass."
    )
    stored = best.get("checkpoint_path")
    assert (
        stored
    ), f"{exp_dir}/results.json best_checkpoint has no checkpoint_path"
    ckpt = exp_dir / "checkpoints" / Path(stored).name
    assert ckpt.exists(), f"checkpoint not found: {ckpt}"
    return ckpt


def read_reference_test_accuracy(exp_dir: Path) -> float:
    """The recorded clean test accuracy the robustness eval must reproduce."""
    best = _read_results(Path(exp_dir))["best_checkpoint"]
    assert "test_accuracy" in best, "run has no recorded test_accuracy"
    return float(best["test_accuracy"])


def merge_robustness_results(exp_dir: Path, block: dict) -> Path:
    """Set ``results['robustness'] = block``, preserving everything else.

    Atomic (tmp file + os.replace) so a crash mid-write cannot corrupt the
    training results.
    """
    exp_dir = Path(exp_dir)
    results = _read_results(exp_dir)
    results["robustness"] = block

    path = exp_dir / "results.json"
    fd, tmp_path = tempfile.mkstemp(dir=exp_dir, suffix=".results.json.tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(results, f, indent=4)
        os.replace(tmp_path, path)
    except BaseException:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise
    return path


if __name__ == "__main__":
    import tempfile as _tf

    with _tf.TemporaryDirectory() as tmp:
        exp = Path(tmp)
        (exp / "checkpoints").mkdir()
        (exp / "checkpoints" / "epoch_001.ckpt").touch()
        (exp / "results.json").write_text(
            json.dumps(
                {
                    "best_checkpoint": {
                        "test_accuracy": 0.97,
                        "checkpoint_path": "/stale/prefix/checkpoints/epoch_001.ckpt",
                    }
                }
            )
        )
        print("ckpt:", resolve_best_checkpoint(exp).name)  # epoch_001.ckpt
        print("ref acc:", read_reference_test_accuracy(exp))  # 0.97
        merge_robustness_results(exp, {"gaussian": {"severity_1": 0.9}})
        merged = json.loads((exp / "results.json").read_text())
        print(
            "merged:", "robustness" in merged and "best_checkpoint" in merged
        )
