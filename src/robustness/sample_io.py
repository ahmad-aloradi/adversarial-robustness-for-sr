"""Save a few adversarial examples per attack for visual inspection.

The attack loop keeps the first N clean/adversarial pairs; this writes them
to disk so a run's ``adv_attacks/`` tree can be browsed after the fact.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import Tensor
from torchvision.utils import save_image

_PIXEL_TOL = 1e-6  # [0,1] compliance slack, same as the attack-side check


def attack_sample_dir(exp_dir, attack_name: str, attack_kwargs: dict) -> Path:
    """Per-attack sample directory: ``<exp_dir>/adv_attacks/<name>/<budget>``.

    ``budget`` is ``<norm>_e_<eps>`` (eps at 4 decimals, trailing zeros
    trimmed, e.g. ``Linf_e_0.0314``); attacks without an ``eps`` use the
    norm alone.
    """
    norm = str(attack_kwargs.get("norm", "Linf"))
    eps = attack_kwargs.get("eps")
    if eps is None:
        budget = norm
    else:
        eps_str = f"{float(eps):.4f}".rstrip("0").rstrip(".")
        budget = f"{norm}_e_{eps_str}"
    return Path(exp_dir) / "adv_attacks" / attack_name / budget


def save_adversarial_samples(
    out_dir,
    clean: Tensor,
    adv: Tensor,
    labels: Tensor,
    clean_preds: Tensor,
    adv_preds: Tensor,
) -> Path:
    """Write clean/adv/delta PNGs plus ``samples.json`` for N examples.

    Inputs: ``clean``/``adv`` are ``(N, C, H, W)`` images in [0,1];
    ``labels``/``clean_preds``/``adv_preds`` are ``(N,)`` class ids. The
    delta PNG is the absolute perturbation scaled to its own max so the
    (tiny) difference is visible. Returns the directory written to.
    """
    assert clean.shape == adv.shape, f"clean {clean.shape} != adv {adv.shape}"
    assert clean.ndim == 4, f"expected (N, C, H, W), got {clean.shape}"
    n_samples = clean.shape[0]
    assert (
        labels.shape[0] == n_samples
        and clean_preds.shape[0] == n_samples
        and adv_preds.shape[0] == n_samples
    ), "labels/preds count must match the number of images"
    assert (
        float(clean.min()) >= -_PIXEL_TOL
        and float(clean.max()) <= 1 + _PIXEL_TOL
    ), "clean images left [0,1] — save expects pixel-space inputs"
    assert (
        float(adv.min()) >= -_PIXEL_TOL and float(adv.max()) <= 1 + _PIXEL_TOL
    ), "adversarial images left [0,1] — save expects pixel-space inputs"

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    delta = (adv - clean).abs()
    manifest = []
    for i in range(n_samples):
        save_image(clean[i], out_dir / f"sample{i}_clean.png")
        save_image(adv[i], out_dir / f"sample{i}_adv.png")
        delta_max = float(delta[i].max())
        delta_vis = delta[i] / delta_max if delta_max > 0 else delta[i]
        save_image(delta_vis, out_dir / f"sample{i}_delta.png")
        manifest.append(
            {
                "index": i,
                "true_label": int(labels[i]),
                "clean_pred": int(clean_preds[i]),
                "adv_pred": int(adv_preds[i]),
                "attack_flipped": int(adv_preds[i]) != int(labels[i]),
                "delta_linf": delta_max,
            }
        )
    (out_dir / "samples.json").write_text(json.dumps(manifest, indent=2))
    return out_dir


if __name__ == "__main__":
    import tempfile

    torch.manual_seed(0)
    clean = torch.rand(3, 3, 8, 8)
    adv = (clean + 0.03).clamp(0, 1)
    labels = torch.tensor([1, 2, 3])
    clean_preds = torch.tensor([1, 2, 3])
    adv_preds = torch.tensor([1, 0, 3])
    out = save_adversarial_samples(
        Path(tempfile.mkdtemp()) / "PGD" / "Linf_e_0.03",
        clean,
        adv,
        labels,
        clean_preds,
        adv_preds,
    )
    written = sorted(p.name for p in out.iterdir())
    print(f"wrote {len(written)} files to {out}: {written}")
