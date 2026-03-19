"""Official ASVSpoof5 Track 1 evaluation metrics wrapped as a Lightning Metric.

The core computation functions (compute_eer, compute_det_curve, compute_mindcf,
compute_actDCF, calculate_CLLR) are vendored from the official ASVSpoof5
evaluation package:
    https://github.com/asvspoof-challenge/asvspoof5/tree/main/evaluation-package

Official cost model for Track 1:
    Pspoof = 0.05, Cmiss = 1, Cfa = 10

Primary metric: minDCF (normalized)
Secondary metrics: EER, actDCF, CLLR
"""

from typing import Any, Dict, Optional

import numpy as np
import torch
from torchmetrics import Metric

# Official ASVSpoof5 Track 1 cost model
ASVSPOOF5_COST_MODEL = {
    "Pspoof": 0.05,
    "Cmiss": 1,
    "Cfa": 10,
}

# ---------------------------------------------------------------------------
#  Vendored from asvspoof5/evaluation-package/calculate_modules.py
# ---------------------------------------------------------------------------


def compute_det_curve(target_scores, nontarget_scores):
    """Compute DET curve.

    Returns (frr, far, thresholds) as numpy arrays.
    """
    n_scores = target_scores.size + nontarget_scores.size
    all_scores = np.concatenate((target_scores, nontarget_scores))
    labels = np.concatenate(
        (np.ones(target_scores.size), np.zeros(nontarget_scores.size))
    )

    indices = np.argsort(all_scores, kind="mergesort")
    labels = labels[indices]

    tar_trial_sums = np.cumsum(labels)
    nontarget_trial_sums = nontarget_scores.size - (
        np.arange(1, n_scores + 1) - tar_trial_sums
    )

    frr = np.concatenate((np.atleast_1d(0), tar_trial_sums / target_scores.size))
    far = np.concatenate(
        (np.atleast_1d(1), nontarget_trial_sums / nontarget_scores.size)
    )
    thresholds = np.concatenate(
        (np.atleast_1d(all_scores[indices[0]] - 0.001), all_scores[indices])
    )

    return frr, far, thresholds


def compute_eer(target_scores, nontarget_scores):
    """Returns (eer, frr, far, thresholds, eer_threshold)."""
    frr, far, thresholds = compute_det_curve(target_scores, nontarget_scores)
    abs_diffs = np.abs(frr - far)
    min_index = np.argmin(abs_diffs)
    eer = np.mean((frr[min_index], far[min_index]))
    return eer, frr, far, thresholds, thresholds[min_index]


def compute_mindcf(frr, far, thresholds, Pspoof, Cmiss, Cfa):
    """Compute normalized minimum Detection Cost Function."""
    min_c_det = float("inf")
    min_c_det_threshold = thresholds[0] if len(thresholds) > 0 else 0.0

    p_target = 1 - Pspoof
    for i in range(len(frr)):
        c_det = Cmiss * frr[i] * p_target + Cfa * far[i] * (1 - p_target)
        if c_det < min_c_det:
            min_c_det = c_det
            min_c_det_threshold = thresholds[i]

    c_def = min(Cmiss * p_target, Cfa * (1 - p_target))
    min_dcf = min_c_det / c_def
    return min_dcf, min_c_det_threshold


def compute_actDCF(bonafide_scores, spoof_scores, Pspoof, Cmiss, Cfa):
    """Compute actual (calibration-sensitive) DCF at Bayes-optimal threshold."""
    beta = Cmiss * (1 - Pspoof) / (Cfa * Pspoof)
    threshold = -np.log(beta)

    rate_miss = np.sum(bonafide_scores < threshold) / bonafide_scores.size
    rate_fa = np.sum(spoof_scores >= threshold) / spoof_scores.size

    act_dcf = Cmiss * (1 - Pspoof) * rate_miss + Cfa * Pspoof * rate_fa
    act_dcf = act_dcf / np.min([Cfa * Pspoof, Cmiss * (1 - Pspoof)])

    return act_dcf, threshold


def calculate_CLLR(target_llrs, nontarget_llrs):
    """Calculate log-likelihood ratio cost (CLLR) in bits."""

    def negative_log_sigmoid(lodds):
        return np.log1p(np.exp(-lodds))

    target_llrs = np.array(target_llrs)
    nontarget_llrs = np.array(nontarget_llrs)

    cllr = 0.5 * (
        np.mean(negative_log_sigmoid(target_llrs))
        + np.mean(negative_log_sigmoid(-nontarget_llrs))
    ) / np.log(2)

    return cllr


# ---------------------------------------------------------------------------
#  Lightning Metric wrapper
# ---------------------------------------------------------------------------

class ASVSpoof5Metrics(Metric):
    """Official ASVSpoof5 Track 1 metrics as a Lightning Metric.

    Accumulates (score, label) pairs across batches, then computes all
    official metrics at epoch end via ``compute()``.

    Labels: bonafide=1 (positive/target), spoof=0 (negative/nontarget).
    Scores: higher = more likely bonafide (e.g. log-softmax of bonafide class).

    Returned dict keys: eer, minDCF, actDCF, cllr, eer_threshold, minDCF_threshold
    """

    def __init__(
        self,
        Pspoof: float = ASVSPOOF5_COST_MODEL["Pspoof"],
        Cmiss: int = ASVSPOOF5_COST_MODEL["Cmiss"],
        Cfa: int = ASVSPOOF5_COST_MODEL["Cfa"],
        compute_on_step: bool = False,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
    ):
        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
        )
        self.Pspoof = Pspoof
        self.Cmiss = Cmiss
        self.Cfa = Cfa

        self.add_state("scores", default=torch.tensor([]), dist_reduce_fx="cat")
        self.add_state("labels", default=torch.tensor([]), dist_reduce_fx="cat")

    def update(self, scores: torch.Tensor, labels: torch.Tensor) -> None:
        scores = scores.detach().to(self.scores.device)
        labels = labels.detach().to(self.labels.device)
        assert scores.shape == labels.shape, "Scores and labels must have the same shape"
        assert scores.ndim == 1, "Scores and labels must be 1D tensors"
        self.scores = torch.cat([self.scores, scores])
        self.labels = torch.cat([self.labels, labels])

    def compute(self) -> Dict[str, torch.Tensor]:
        if self.scores.numel() == 0:
            raise ValueError("No data to compute metrics.")

        scores_np = self.scores.cpu().numpy()
        labels_np = self.labels.cpu().numpy()

        bonafide_scores = scores_np[labels_np == 1]
        spoof_scores = scores_np[labels_np == 0]

        if bonafide_scores.size == 0 or spoof_scores.size == 0:
            raise ValueError(
                "Both bonafide and spoof samples are required. "
                f"Got {bonafide_scores.size} bonafide, {spoof_scores.size} spoof."
            )

        # EER (+ DET curve for minDCF)
        eer, frr, far, thresholds, eer_threshold = compute_eer(
            bonafide_scores, spoof_scores
        )

        # minDCF (normalized, primary metric)
        mindcf, mindcf_threshold = compute_mindcf(
            frr, far, thresholds, self.Pspoof, self.Cmiss, self.Cfa
        )

        # actDCF (calibration-sensitive)
        actdcf, actdcf_threshold = compute_actDCF(
            bonafide_scores, spoof_scores, self.Pspoof, self.Cmiss, self.Cfa
        )

        # CLLR
        cllr = calculate_CLLR(bonafide_scores, spoof_scores)

        return {
            "eer": torch.tensor(eer, dtype=torch.float32),
            "minDCF": torch.tensor(mindcf, dtype=torch.float32),
            "actDCF": torch.tensor(actdcf, dtype=torch.float32),
            "cllr": torch.tensor(cllr, dtype=torch.float32),
            "eer_threshold": torch.tensor(eer_threshold, dtype=torch.float32),
            "minDCF_threshold": torch.tensor(mindcf_threshold, dtype=torch.float32),
        }
