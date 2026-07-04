"""Download + prepare TinyImageNet-200 for torchvision ImageFolder.

The val split ships as a flat ``images/`` dir plus ``val_annotations.txt``,
which ImageFolder cannot label. This builds ``val_structured/<wnid>/<file>`` by
copying (``val/`` is left untouched) so that train and val yield the *same*
``class_to_idx``. The train split is already ImageFolder-shaped.

    python src/datamodules/components/vision/tiny_imagenet_prep.py \
        --data-dir data
"""

import argparse
import shutil
import urllib.request
import zipfile
from pathlib import Path

URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"


def download_and_extract(data_dir: Path) -> Path:
    """Fetch + unzip the archive; return the extracted base dir. Idempotent."""
    base = data_dir / "tiny-imagenet-200"
    if base.is_dir():
        print(f"{base} exists; skipping download")
        return base
    data_dir.mkdir(parents=True, exist_ok=True)
    zip_path = data_dir / "tiny-imagenet-200.zip"
    if not zip_path.exists():
        print(f"Downloading {URL} ...")
        urllib.request.urlretrieve(URL, zip_path)
    print(f"Extracting {zip_path} ...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(data_dir)
    return base


def _read_val_annotations(base: Path) -> list:
    """Return [(filename, wnid), ...] from val/val_annotations.txt."""
    ann = base / "val" / "val_annotations.txt"
    assert ann.exists(), f"missing {ann}"
    entries = []
    for line in ann.read_text().splitlines():
        parts = line.split("\t")
        entries.append((parts[0], parts[1]))
    return entries


def restructure_val(base: Path) -> Path:
    """Build base/val_structured/<wnid>/<file> by copying from val/images.

    Non-destructive (val/ untouched) and idempotent: returns early when the
    output already holds one image per annotation.
    """
    out = base / "val_structured"
    entries = _read_val_annotations(base)

    if out.is_dir() and sum(1 for _ in out.rglob("*.JPEG")) == len(entries):
        print(f"{out} already complete ({len(entries)} images)")
        return out

    src_dir = base / "val" / "images"
    for filename, wnid in entries:
        dst_dir = out / wnid
        dst_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(src_dir / filename, dst_dir / filename)
    print(f"Wrote {len(entries)} images to {out}")
    return out


def validate(base: Path) -> None:
    """Assert the tree matches the TinyImageNet-200 contract."""
    train_wnids = {p.name for p in (base / "train").iterdir() if p.is_dir()}
    val_wnids = {
        p.name for p in (base / "val_structured").iterdir() if p.is_dir()
    }
    assert (
        len(train_wnids) == 200
    ), f"expected 200 train wnids, got {len(train_wnids)}"
    assert train_wnids == val_wnids, "train and val wnid sets differ"
    n_val = sum(1 for _ in (base / "val_structured").rglob("*.JPEG"))
    assert n_val == 200 * 50, f"expected 10000 val images, got {n_val}"
    print(f"validate OK: 200 wnids, {n_val} val images")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data")
    args = parser.parse_args()

    base = download_and_extract(Path(args.data_dir))
    restructure_val(base)
    validate(base)


if __name__ == "__main__":
    main()
