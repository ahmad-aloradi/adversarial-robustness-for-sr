from typing import Tuple, Dict, Optional, Any

import hydra
import torch
from omegaconf import DictConfig
from torchmetrics import Metric, MetricCollection, MinMetric
from torchmetrics.utilities.data import dim_zero_cat

import matplotlib.pyplot as plt
import numpy as np

class VerificationMetrics(Metric):
    """
    Verification metrics with plotting capabilities and minDCF support.
    
    Args:
        positive_label: Value representing positive class (default: 1)
        beta: Weight for F-score computation (default: 1.0)
        threshold: Fixed threshold. If None, EER threshold is used (default: None)
        Cfa: Cost of false acceptance (default: 1.0)
        Cfr: Cost of false rejection (default: 1.0)
        P_target: Prior probability of target trials (default: 0.5)
        eps: Small value to prevent division by zero (default: 1e-8)
    """
    
    def __init__(
        self,
        positive_label: int = 1,
        beta: float = 1.0,
        threshold: Optional[float] = None,
        Cfa: float = 1.0,
        Cfr: float = 1.0,
        P_target: float = 0.5,
        eps: float = 1e-8,
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
    ):
        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
        )
        
        self.positive_label = positive_label
        self.beta = torch.tensor(beta)
        self.fixed_threshold = torch.tensor(threshold) if threshold is not None else None
        self.Cfa = torch.tensor(Cfa)
        self.Cfr = torch.tensor(Cfr)
        self.P_target = torch.tensor(P_target)
        self.eps = torch.tensor(eps)
        
        self.add_state("scores", default=torch.tensor([]), dist_reduce_fx='cat')
        self.add_state("labels", default=torch.tensor([]), dist_reduce_fx='cat')
        self._curve_data = None

    def update(self, scores: torch.Tensor, labels: torch.Tensor) -> None:
        """
        Update state with new predictions and targets.
        
        Args:
            scores: Predicted scores/probabilities
            labels: Ground truth labels
        """
        scores = scores.detach().to(self.scores.device)
        labels = labels.detach().to(self.labels.device)

        assert scores.shape == labels.shape, "Scores and labels must have the same shape"
        assert len(scores.shape) == 1, "Scores and labels must be 1D tensors"

        self.scores = torch.cat([self.scores, scores])
        self.labels = torch.cat([self.labels, labels])

    def _compute_confusion_matrix(
        self,
        scores: torch.Tensor,
        labels: torch.Tensor,
        threshold: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute confusion matrix elements ensuring correct behavior at boundary conditions.
        """
        if not torch.all(torch.logical_or(labels == 0, labels == 1)):
            raise ValueError("Labels must be binary (0 or 1)")
            
        predictions = (scores > threshold).float()
        exact_matches = (scores == threshold)
        
        if exact_matches.any():
            predictions_with = predictions.clone()
            predictions_with[exact_matches] = 1.0
            
            predictions_without = predictions.clone()
            predictions_without[exact_matches] = 0.0
            
            cost_with = self._compute_detection_cost(predictions_with, labels)
            cost_without = self._compute_detection_cost(predictions_without, labels)
            
            predictions = predictions_with if cost_with <= cost_without else predictions_without

        positives = (labels == 1)
        negatives = (labels == 0)
        
        TP = (predictions * positives).sum()
        TN = ((1 - predictions) * negatives).sum()
        FP = (predictions * negatives).sum()
        FN = ((1 - predictions) * positives).sum()
        
        return TP, TN, FP, FN

    def _compute_detection_cost(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute detection cost for given predictions.
        """
        positives = (labels == 1)
        negatives = (labels == 0)
        
        far = (predictions * negatives).sum() / (negatives.sum() + self.eps)
        frr = ((1 - predictions) * positives).sum() / (positives.sum() + self.eps)
        
        return self.Cfa * self.P_target * far + self.Cfr * (1 - self.P_target) * frr

    def _compute_eer(
        self,
        scores: torch.Tensor,
        labels: torch.Tensor,
        thresholds: torch.Tensor,
        num_points: int = 1000
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute EER with improved interpolation and error handling.
        """
        if len(scores) < 2:
            raise ValueError("Need at least 2 samples to compute EER")
            
        min_t = thresholds.min()
        max_t = thresholds.max()
        fine_thresholds = torch.linspace(min_t, max_t, num_points, device=scores.device)
        
        fars = []
        frrs = []
        for t in fine_thresholds:
            TP, TN, FP, FN = self._compute_confusion_matrix(scores, labels, t)
            
            far = FP / (FP + TN + self.eps)
            frr = FN / (TP + FN + self.eps)
            
            fars.append(far)
            frrs.append(frr)
            
        fars = torch.stack(fars)
        frrs = torch.stack(frrs)
        
        abs_diffs = torch.abs(fars - frrs)
        
        if torch.min(abs_diffs) == 0:
            idx = torch.argmin(abs_diffs)
            eer = fars[idx]
            threshold = fine_thresholds[idx]
        else:
            idx = torch.argmin(abs_diffs)
            
            if idx > 0:
                t0, t1 = fine_thresholds[idx-1], fine_thresholds[idx]
                far0, far1 = fars[idx-1], fars[idx]
                frr0, frr1 = frrs[idx-1], frrs[idx]
                
                delta_t = t1 - t0
                delta_far = far1 - far0
                delta_frr = frr1 - frr0
                
                if torch.abs(delta_far - delta_frr) > self.eps:
                    alpha = (frr0 - far0) / (delta_far - delta_frr)
                    threshold = t0 + alpha * delta_t
                    eer = far0 + alpha * delta_far
                else:
                    threshold = (t0 + t1) / 2
                    eer = (far0 + frr0) / 2
            else:
                threshold = fine_thresholds[idx]
                eer = (fars[idx] + frrs[idx]) / 2
                
        return eer, threshold

    def _compute_stats_at_threshold(
        self, 
        scores: torch.Tensor,
        labels: torch.Tensor,
        threshold: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute classification statistics at a given threshold.
        """
        TP, TN, FP, FN = self._compute_confusion_matrix(scores, labels, threshold)

        precision = TP / (TP + FP + self.eps)
        recall = TP / (TP + FN + self.eps)
        far = FP / (FP + TN + self.eps)
        frr = FN / (TP + FN + self.eps)
        
        f_score = ((1.0 + self.beta**2.0) * TP) / (
            (1.0 + self.beta**2.0) * TP + 
            self.beta**2.0 * FN + 
            FP + 
            self.eps
        )
        
        detection_cost = self.Cfa * self.P_target * far + self.Cfr * (1 - self.P_target) * frr
        
        return {
            'threshold': threshold,
            'TP': TP,
            'TN': TN,
            'FP': FP,
            'FN': FN,
            'precision': precision,
            'recall': recall,
            'far': far,
            'frr': frr,
            'f_score': f_score,
            'detection_cost': detection_cost
        }

    def compute(self) -> Dict[str, torch.Tensor]:
        """
        Compute final metrics.
        """
        scores = self.scores
        labels = self.labels
        
        if scores.numel() == 0 or labels.numel() == 0:
            raise ValueError("No data to compute metrics. Please call `update` with valid inputs first.")
        
        assert labels.shape == scores.shape, "Scores and labels must have the same shape"
        assert len(labels.shape) == 1, "Scores and labels must be 1D tensors"

        pos_mask = (labels == self.positive_label)
        pos_scores = scores[pos_mask]
        neg_scores = scores[~pos_mask]
        
        if pos_scores.numel() == 0 or neg_scores.numel() == 0:
            raise ValueError("Both positive and negative samples are required to compute verification metrics.")
        
        thresholds = torch.sort(torch.unique(scores))[0]
        
        if thresholds.numel() > 1:
            intermediate = (thresholds[:-1] + thresholds[1:]) / 2
            thresholds = torch.sort(torch.cat([thresholds, intermediate]))[0]
        
        if self.fixed_threshold is not None:
            thresholds = torch.cat([
                thresholds,
                self.fixed_threshold.unsqueeze(0).to(thresholds.device)
            ])
            thresholds = torch.unique(thresholds)
            thresholds = torch.sort(thresholds)[0]
        
        self._curve_data = {
            'threshold': [],
            'precision': [],
            'recall': [],
            'far': [],
            'frr': [],
            'f_score': [],
            'detection_cost': [],
            'TP': [],
            'FP': [],
            'FN': [],
            'TN': []
        }
        
        min_dcf = torch.tensor(float('inf'), device=scores.device)
        min_dcf_threshold = None
        
        for t in thresholds:
            stats = self._compute_stats_at_threshold(scores, labels, t)
            
            for key in self._curve_data.keys():
                if key in stats:
                    self._curve_data[key].append(stats[key])
            
            if stats['detection_cost'] < min_dcf:
                min_dcf = stats['detection_cost']
                min_dcf_threshold = stats['threshold']
        
        # Convert lists to tensors
        self._curve_data = {k: torch.stack(v) if v else torch.tensor([]) for k, v in self._curve_data.items()}
        
        eer, eer_threshold = self._compute_eer(scores, labels, thresholds)
        
        threshold = self.fixed_threshold if self.fixed_threshold is not None else eer_threshold
        final_stats = self._compute_stats_at_threshold(scores, labels, threshold)
        
        final_stats['eer'] = eer
        final_stats['eer_threshold'] = eer_threshold
        final_stats['minDCF'] = min_dcf
        final_stats['minDCF_threshold'] = min_dcf_threshold
        
        return final_stats

    def plot_curves(self) -> Dict[str, plt.Figure]:
        """
        Plot verification curves with professional styling.
        
        Returns:
            Dictionary containing matplotlib figures for different plots
        """
        if self._curve_data is None:
            _ = self.compute()
            
        # Convert tensors to numpy for plotting
        curve_data_np = {
            k: v.detach().cpu().numpy() 
            for k, v in self._curve_data.items()
        }
        
        metrics = self.compute()
        metrics_np = {
            k: v.detach().cpu().numpy() if torch.is_tensor(v) else v
            for k, v in metrics.items()
        }
        
        # Set up professional plotting style
        plt.style.use('seaborn-v0_8-paper')
        plt.rcParams.update({
            'font.family': 'serif',
            'font.size': 10,
            'axes.labelsize': 11,
            'axes.titlesize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'figure.titlesize': 12,
            'axes.grid': True,
            'grid.alpha': 0.3
        })
        
        figures = {}
        
        # DET Curve
        fig_det, ax_det = plt.subplots(figsize=(8, 8))
        far = np.maximum(curve_data_np['far'], self.eps.item())
        frr = np.maximum(curve_data_np['frr'], self.eps.item())
        
        ax_det.plot(far, frr, 'b-', label='DET curve', linewidth=1.5)
        ax_det.plot(metrics_np['eer'], metrics_np['eer'], 'ro', 
                   label=f'EER: {metrics_np["eer"]:.3f}', markersize=6)
        
        ax_det.plot([self.eps.item(), 1], [self.eps.item(), 1], 'k--', alpha=0.3, linewidth=1)
        
        ax_det.set_xscale('log')
        ax_det.set_yscale('log')
        ax_det.set_xlabel('False Acceptance Rate (FAR)')
        ax_det.set_ylabel('False Rejection Rate (FRR)')
        ax_det.set_title('Detection Error Tradeoff (DET) Curve')
        ax_det.grid(True, which='both', linestyle='--', alpha=0.3)
        ax_det.legend(frameon=True, fancybox=False, edgecolor='black')
        figures['det'] = fig_det
        
        # Detection Cost Curve
        fig_dcf, ax_dcf = plt.subplots(figsize=(8, 6))
        ax_dcf.plot(
            curve_data_np['threshold'],
            curve_data_np['detection_cost'],
            'b-',
            label='Detection Cost',
            linewidth=1.5
        )
        ax_dcf.axvline(
            metrics_np['minDCF_threshold'],
            color='r',
            linestyle='--',
            label=f'minDCF: {metrics_np["minDCF"]:.3f}'
        )
        ax_dcf.set_xlabel('Threshold (θ)')
        ax_dcf.set_ylabel('Detection Cost')
        ax_dcf.set_title('Detection Cost Function')
        ax_dcf.legend(frameon=True, fancybox=False, edgecolor='black')
        ax_dcf.grid(True, linestyle='--', alpha=0.3)
        figures['dcf'] = fig_dcf
        
        # Precision-Recall Curve
        fig_pr, ax_pr = plt.subplots(figsize=(8, 8))
        ax_pr.plot(
            curve_data_np['recall'],
            curve_data_np['precision'],
            'b-',
            linewidth=1.5
        )
        ax_pr.set_xlabel('Recall')
        ax_pr.set_ylabel('Precision')
        ax_pr.set_title('Precision-Recall Curve')
        ax_pr.grid(True, linestyle='--', alpha=0.3)
        figures['pr'] = fig_pr
        
        # Threshold vs Metrics plot
        fig_rates, ax_rates = plt.subplots(figsize=(10, 6))
        metrics_to_plot = {
            'TP': ('True Positives', 'b-'),
            'FP': ('False Positives', 'r--'),
            'FN': ('False Negatives', 'g:'),
            'TN': ('True Negatives', 'm-.')
        }
        
        for metric, (label, style) in metrics_to_plot.items():
            ax_rates.plot(
                curve_data_np['threshold'],
                curve_data_np[metric],
                style,
                label=label,
                linewidth=1.5
            )
            
        ax_rates.set_xlabel('Threshold (θ)')
        ax_rates.set_ylabel('Count')
        ax_rates.set_title('Classification Metrics vs Threshold')
        ax_rates.legend(frameon=True, fancybox=False, edgecolor='black')
        ax_rates.grid(True, linestyle='--', alpha=0.3)
        figures['rates'] = fig_rates
        
        # Adjust layout for all figures
        for fig in figures.values():
            fig.tight_layout()
        
        return figures

    def reset(self) -> None:
        """Reset metric states."""
        super().reset()
        self._curve_data = None

    def __str__(self) -> str:
        """String representation of metrics."""
        try:
            metrics = self.compute()
            return (
                f"VerificationMetrics(EER={metrics['eer'].item():.3f}, "
                f"minDCF={metrics['minDCF'].item():.3f}, "
                f"F{self.beta.item()}={metrics['f_score'].item():.3f})"
            )
        except:
            return "VerificationMetrics(no data)"

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
        current_target = metric_dict[self.target_key].detach() if torch.is_tensor(metric_dict[self.target_key]) else torch.tensor(metric_dict[self.target_key])
        
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


def AS_norm(score: float, 
            enroll_embedding: torch.Tensor, 
            test_embedding: torch.Tensor, 
            cohort_embeddings: torch.Tensor, 
            topk: int = 3000,
            min_cohort_size: int = 300) -> float:
    """
    Adaptive Symmetric Normalization (AS-Norm) for speaker verification
    
    Args:
        score: Raw cosine similarity score between enrollment and test embeddings
        enroll_embedding: Enrollment utterance embedding (1D tensor)
        test_embedding: Test utterance embedding (1D tensor)
        cohort_embeddings: Impostor cohort embeddings (2D tensor: [num_cohorts, embedding_dim])
        topk: Number of top scores to consider for normalization
        min_cohort_size: Minimum number of cohort speakers required for reliable normalization
    
    Returns:
        Normalized score
    """
    EPS = 1e-9
    # Cohort size validation
    if cohort_embeddings.shape[0] < min_cohort_size:
        raise ValueError(f"Cohort size ({cohort_embeddings.shape[0]}) is smaller than recommended minimum ({min_cohort_size}). "
                        f"This may lead to unreliable normalization. Consider using a larger cohort.")
    
    # Input validation and shape checking
    assert isinstance(score, (float, int)) or (isinstance(score, torch.Tensor) and score.numel() == 1), \
        "Score must be a scalar value"
    assert len(enroll_embedding.shape) == 1, "Enrollment embedding must be 1D"
    assert len(test_embedding.shape) == 1, "Test embedding must be 1D"
    assert len(cohort_embeddings.shape) == 2, "Cohort embeddings must be 2D"
    assert enroll_embedding.shape[0] == test_embedding.shape[0] == cohort_embeddings.shape[1], \
        "Embedding dimensions must match"

    # Convert score to tensor if it's not already
    if not isinstance(score, torch.Tensor):
        score = torch.tensor(score, dtype=torch.float32)

    # Ensure all tensors are on the same device
    device = enroll_embedding.device
    score = score.to(device)

    if topk > cohort_embeddings.shape[0]:
        print(f"WARNING: topk ({topk}) is larger than number of cohort embeddings ({cohort_embeddings.size(0)}). "
              f"Setting topk to maximum available: {cohort_embeddings.size(0)}")
        topk = cohort_embeddings.shape[0]

    # Compute enrollment vs cohort scores
    with torch.no_grad():  # Add no_grad for efficiency
        enroll_scores = torch.nn.functional.cosine_similarity(enroll_embedding, cohort_embeddings, dim=-1)
        enroll_top_scores, _ = torch.topk(enroll_scores, topk, dim=0)
        enroll_mean = enroll_top_scores.mean()
        enroll_std = enroll_top_scores.std(unbiased=True)  # Use unbiased standard deviation

        # Compute similarity scores between test and cohort
        test_scores = torch.nn.functional.cosine_similarity(test_embedding, cohort_embeddings, dim=-1)
        test_top_scores, _ = torch.topk(test_scores, topk, dim=0)
        test_mean = test_top_scores.mean()
        test_std = test_top_scores.std(unbiased=True)  # Use unbiased standard deviation

        # Symmetric normalization
        normalized_score = 0.5 * ((score - enroll_mean) / (enroll_std + EPS) + 
                                  (score - test_mean) / (test_std + EPS))
        
        # Final numerical stability check
        if torch.isnan(normalized_score) or torch.isinf(normalized_score):
            print("WARNING: Invalid value detected in final normalized score. Returning original score.")
            return score.item() if isinstance(score, torch.Tensor) else score
            
        return normalized_score.item()


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
