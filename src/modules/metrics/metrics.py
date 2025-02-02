from typing import Tuple, Dict, Optional

import hydra
import torch
from omegaconf import DictConfig
from torchmetrics import Metric, MetricCollection, MinMetric
from torchmetrics.utilities.data import dim_zero_cat

class SpeakerVerificationMetrics(Metric):
    """Computes EER and minDCF for speaker verification tasks."""
    def __init__(self, Cfa: float = 1.0, Cmiss: float = 1.0, P_target: float = 0.01, **kwargs):
        super().__init__(**kwargs)
        self.Cfa = Cfa
        self.Cmiss = Cmiss
        self.P_target = P_target
        self.eps = 1e-8

        # States to accumulate scores and targets
        self.add_state("scores", default=[], dist_reduce_fx="cat")
        self.add_state("targets", default=[], dist_reduce_fx="cat")

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """Update metric states with predictions and targets.
        
        Args:
            preds: Predicted similarity scores
            target: Ground truth labels (0 for non-target, 1 for target)
        """
        self.scores.append(preds)
        self.targets.append(target)

    def compute(self) -> Dict[str, torch.Tensor]:
        """Compute EER and minDCF from accumulated scores and targets."""
        scores = dim_zero_cat(self.scores)
        targets = dim_zero_cat(self.targets)

        assert len(scores) > 0, "Input is empty. No scores or targets provided."
        assert len(scores) == len(targets), "Number of scores and targets must be equal."

        # Adaptive s-norm (Speaker Verification standard)
        scores = self._adaptive_s_norm(scores, targets)

        # Separate target and non-target scores
        target_scores = scores[targets == 1]
        non_target_scores = scores[targets == 0]

        # Assert presence of both target and non-target scores
        assert len(target_scores) > 0, "No target scores found. Cannot compute EER or minDCF."
        assert len(non_target_scores) > 0, "No non-target scores found. Cannot compute EER or minDCF."

        # Optimize number of thresholds for large datasets
        sorted_scores, _ = torch.sort(scores, descending=True)
        num_thresholds = min(len(scores), 1000)  # Cap at 1000 thresholds
        step = max(1, len(sorted_scores) // num_thresholds)
        thresholds = sorted_scores[::step]

        # Initialize arrays for FAR and FRR
        far = torch.zeros(len(thresholds), device=scores.device)
        frr = torch.zeros(len(thresholds), device=scores.device)

        # Compute FAR and FRR for each threshold
        for i, threshold in enumerate(thresholds):
            far[i] = (non_target_scores >= threshold).float().mean() + self.eps
            frr[i] = (target_scores < threshold).float().mean() + self.eps

        # Compute EER
        eer, eer_threshold = self._compute_eer(far, frr, thresholds)

        # Compute minDCF
        min_dcf, min_dcf_threshold = self._compute_min_dcf(far, frr, thresholds)

        return {
            f"EER": eer,
            f"EER_threshold": eer_threshold,
            f"minDCF": min_dcf,
            f"minDCF_threshold": min_dcf_threshold
        }
            
    def _compute_eer(self, far: torch.Tensor, frr: torch.Tensor, thresholds: torch.Tensor
                     ) -> Tuple[torch.Tensor, torch.Tensor]:

        abs_diff = torch.abs(far - frr)
        min_diff_idx = torch.argmin(abs_diff)

        if min_diff_idx > 0:
            # Interpolate for more precise EER
            idx1, idx2 = min_diff_idx - 1, min_diff_idx
            t1, t2 = thresholds[idx1], thresholds[idx2]
            far1, far2 = far[idx1], far[idx2]
            frr1, frr2 = frr[idx1], frr[idx2]

            # Linear interpolation
            delta = (far2 - frr2) - (far1 - frr1)
            if abs(delta) > self.eps:
                alpha = (far1 - frr1) / delta
                eer = far1 + alpha * (far2 - far1)
                eer_threshold = t1 + alpha * (t2 - t1)
            else:
                eer = (far2 + frr2) / 2
                eer_threshold = t2
        else:
            eer = (far[min_diff_idx] + frr[min_diff_idx]) / 2
            eer_threshold = thresholds[min_diff_idx]

        return eer, eer_threshold

    def _compute_min_dcf(self, far: torch.Tensor, frr: torch.Tensor, thresholds: torch.Tensor) -> torch.Tensor:
        dcf = self.Cfa * far * (1 - self.P_target) + self.Cmiss * frr * self.P_target
        min_dcf_idx = torch.argmin(dcf)
        min_dcf = dcf[min_dcf_idx]
        min_dcf_threshold = thresholds[min_dcf_idx]
        return min_dcf, min_dcf_threshold

    def _adaptive_s_norm(self, scores: torch.Tensor, targets: torch.Tensor, 
                        cohort_size: int = 200) -> torch.Tensor:
        """Apply Adaptive S-normalization to scores.
        
        Args:
            scores: Raw similarity scores
            targets: Ground truth labels
            cohort_size: Size of cohort for normalization (default: 200)
        
        Returns:
            Normalized scores
        """
        # Get non-target scores for cohort
        non_target_scores = scores[targets == 0]
        
        if len(non_target_scores) > cohort_size:
            # Randomly sample cohort_size scores for efficiency
            indices = torch.randperm(len(non_target_scores))[:cohort_size]
            cohort_scores = non_target_scores[indices]
        else:
            cohort_scores = non_target_scores

        # Compute mean and std for each score using cohort
        means = []
        stds = []
        
        for score in scores:
            # Get top-K closest cohort scores
            diffs = torch.abs(cohort_scores - score)
            _, topk_indices = torch.topk(diffs, k=min(50, len(cohort_scores)), largest=False)
            topk_scores = cohort_scores[topk_indices]
            
            # Compute adaptive statistics
            mean = topk_scores.mean()
            std = topk_scores.std() + self.eps
            
            means.append(mean)
            stds.append(std)

        # Convert lists to tensors
        means = torch.stack(means)
        stds = torch.stack(stds)

        # Apply normalization
        normalized_scores = (scores - means) / stds
        
        return normalized_scores

class AutoSyncDictMinMetric(Metric):
    """Automatically configures dist_sync_on_step based on execution context."""
    
    def __init__(self, target_key: str, compute_on_step=False):
        super().__init__(compute_on_step=compute_on_step)
        
        self.target_key = target_key
        self._min_metric: Optional[MinMetric] = None
        self._best_values = {}
        self._current_min = torch.tensor(float('inf'))
        
        # Deferred initialization
        self.add_state("_initialized", default=torch.tensor(0), dist_reduce_fx="mean")

    def _lazy_init(self):
        """Initialize MinMetric after device placement is known."""
        if self._min_metric is None:
            # Auto-determine dist_sync_on_step based on Lightning context
            dist_sync = self._should_sync_on_step()
            self._min_metric = MinMetric(dist_sync_on_step=dist_sync).to(self.device)

    def _should_sync_on_step(self) -> bool:
        """Determine if synchronization should happen on step."""
        try:
            # Check if we're in a distributed environment
            if self.trainer and self.trainer.num_devices > 1:
                # Only sync on step during training, not validation/test
                return self.trainer.training
            return False
        except AttributeError:
            # Fallback if not used with Lightning
            return False

    def update(self, metric_dict: dict):
        self._lazy_init()
        current_target = metric_dict[self.target_key].detach()
        
        # Update min metric
        self._min_metric.update(current_target)
        
        # Track best values
        if current_target < self._current_min:
            self._current_min = current_target.clone()
            self._best_values = {
                k: v.detach().clone() 
                for k, v in metric_dict.items() 
                if k != self.target_key
            }

    def compute(self) -> dict:
        self._lazy_init()
        return {
            self.target_key: self._min_metric.compute(),
            **self._best_values
        }

    def reset(self):
        if self._min_metric:
            self._min_metric.reset()
        self._current_min = torch.tensor(float('inf'))
        self._best_values = {}


def load_metrics(
    metrics_cfg: DictConfig,
) -> Tuple[Metric, Metric, MetricCollection]:
    """Load main metric, `best` metric tracker, MetricCollection of additional
    metrics.

    Args:
        metrics_cfg (DictConfig): Metrics config.

    Returns:
        Tuple[Metric, Metric, ModuleList]: Main metric, `best` metric tracker,
            MetricCollection of additional metrics.
    """

    main_metric_name = metrics_cfg.main if metrics_cfg.get("main") else metrics_cfg.valid
    main_metric = hydra.utils.instantiate(main_metric_name)

    if not metrics_cfg.get("valid_best"):
        raise RuntimeError(
            "Requires valid_best metric that would track best state of "
            "Main Metric. Usually it can be MaxMetric or MinMetric."
        )
    valid_metric_best = hydra.utils.instantiate(metrics_cfg.valid_best)

    additional_metrics = []
    if metrics_cfg.get("additional"):
        for _, metric_cfg in metrics_cfg.additional.items():
            additional_metrics.append(hydra.utils.instantiate(metric_cfg))

    return main_metric, valid_metric_best, MetricCollection(additional_metrics)
