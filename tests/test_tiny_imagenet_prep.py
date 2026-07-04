"""Synthetic-tree tests for the TinyImageNet prep (no network).

Exercises the non-destructive val restructuring and the contract check on a
small hand-built tree; the real download/extract path is not tested here.
"""

from pathlib import Path

import pytest

from src.datamodules.components.vision.tiny_imagenet_prep import (
    restructure_val,
    validate,
)


def _make_synthetic(base: Path, wnids, imgs_per=3) -> None:
    """Build a minimal TinyImageNet-shaped tree under `base`."""
    for wnid in wnids:
        images = base / "train" / wnid / "images"
        images.mkdir(parents=True)
        for i in range(imgs_per):
            (images / f"{wnid}_{i}.JPEG").write_bytes(b"x")

    val_images = base / "val" / "images"
    val_images.mkdir(parents=True)
    lines = []
    for wnid in wnids:
        for i in range(imgs_per):
            filename = f"val_{wnid}_{i}.JPEG"
            (val_images / filename).write_bytes(b"x")
            lines.append(f"{filename}\t{wnid}\t0\t0\t0\t0")
    (base / "val" / "val_annotations.txt").write_text("\n".join(lines))


def test_restructure_layout(tmp_path):
    base = tmp_path / "tiny-imagenet-200"
    wnids = ["n01", "n02", "n03"]
    _make_synthetic(base, wnids, imgs_per=3)

    out = restructure_val(base)
    for wnid in wnids:
        assert (out / wnid).is_dir()
        assert len(list((out / wnid).glob("*.JPEG"))) == 3

    # val/ is copied from, never modified.
    assert len(list((base / "val" / "images").glob("*.JPEG"))) == 9


def test_restructure_idempotent(tmp_path):
    base = tmp_path / "tiny-imagenet-200"
    _make_synthetic(base, ["n01", "n02"], imgs_per=3)

    restructure_val(base)
    out = restructure_val(base)  # returns early, no error
    assert sum(1 for _ in out.rglob("*.JPEG")) == 6


def test_missing_annotations_raises(tmp_path):
    base = tmp_path / "tiny-imagenet-200"
    (base / "val").mkdir(parents=True)
    with pytest.raises(AssertionError):
        restructure_val(base)


def test_validate_rejects_incomplete(tmp_path):
    base = tmp_path / "tiny-imagenet-200"
    _make_synthetic(base, ["n01", "n02", "n03"], imgs_per=3)
    restructure_val(base)
    # Only 3 wnids / 9 val images — nowhere near the 200 / 10000 contract.
    with pytest.raises(AssertionError):
        validate(base)
