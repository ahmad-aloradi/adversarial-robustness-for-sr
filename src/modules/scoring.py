"""Unified scoring pipeline for speaker verification.

This module provides a coherent abstraction for:
1. Enrollment embedding aggregation (multi-utterance → single embedding)
2. Mean centering (subtract cohort mean from embeddings)
3. Score normalization (AS-Norm)

The pipeline ensures consistent application order and prevents common mistakes
like test-set data leakage in mean computation.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F


class EnrollmentAggregation(str, Enum):
    """Methods for aggregating multi-utterance enrollments."""

    MEAN = "mean"  # Simple average (most common, no length tracking needed)
    LENGTH_WEIGHTED = "length_weighted"  # Weight by utterance duration (longer = more weight)


class MeanSource(str, Enum):
    """Source for computing the centering mean vector."""

    COHORT = "cohort"  # Use cohort embeddings (recommended)
    NONE = "none"  # No centering


class NormMethod(str, Enum):
    """Score normalization methods."""

    NONE = "none"
    AS_NORM = "as_norm"  # Adaptive Symmetric Normalization


@dataclass
class CohortConfig:
    """Configuration for cohort construction and usage.

    The cohort is used for both mean centering and score normalization. Proper
    cohort handling is critical for reliable speaker verification.
    """

    # Whether to average embeddings per speaker (recommended for AS-Norm)
    speaker_level: bool = True

    # For AS-Norm: number of top speakers (not utterances) to use
    topk_speakers: int = 300

    # Minimum cohort size (speakers if speaker_level=True, else utterances)
    min_size: int = 100

    # Maximum cohort size (for memory efficiency)
    max_size: Optional[int] = None


@dataclass
class ScoringConfig:
    """Configuration for the complete scoring pipeline.

    Example usage in YAML config:
        scoring:
          enrollment_aggregation: mean
          mean_source: cohort
          norm_method: as_norm
          cohort:
            speaker_level: true
            topk_speakers: 300
    """

    # Enrollment aggregation: "mean" or "length_weighted"
    enrollment_aggregation: str = "mean"

    # Mean centering: "cohort" (recommended) or "none"
    mean_source: str = "cohort"

    # Score normalization: "as_norm" (recommended) or "none"
    norm_method: str = "as_norm"

    # Cohort configuration (used for both centering and AS-Norm)
    cohort: CohortConfig = field(default_factory=CohortConfig)


class ScoringPipeline:
    """Unified scoring pipeline for speaker verification.

    Handles the complete test-time scoring flow:
    1. Enrollment aggregation: Combine multi-utterance enrollments
    2. Mean centering: Subtract cohort mean from embeddings
    3. Score computation: Cosine similarity
    4. Score normalization: AS-Norm (Adaptive Symmetric Normalization)

    Usage:
        pipeline = ScoringPipeline(config)
        pipeline.set_cohort(cohort_embeddings, speaker_ids)

        # For each trial:
        score = pipeline.score(enroll_emb, test_emb)

    Attributes:
        config: ScoringConfig with all pipeline settings
        mean_vector: Computed cohort mean for centering (if enabled)
        cohort_embeddings: Speaker-level or utterance-level cohort
    """

    EPS = 1e-9

    def __init__(self, config: ScoringConfig):
        self.config = config
        self.mean_vector: Optional[torch.Tensor] = None
        self.cohort_embeddings: Optional[torch.Tensor] = None
        self._cohort_speaker_ids: Optional[torch.Tensor] = None
        self._speaker_cohort: Optional[
            torch.Tensor
        ] = None  # Per-speaker averaged

    def set_cohort(
        self,
        embeddings: torch.Tensor,
        speaker_ids: Optional[torch.Tensor] = None,
    ) -> None:
        """Set cohort embeddings for normalization and mean computation.

        Args:
            embeddings: Cohort embeddings [N, embed_dim]
            speaker_ids: Speaker ID for each embedding [N]. Required if
                        config.cohort.speaker_level is True.
        """
        self.cohort_embeddings = embeddings
        self._cohort_speaker_ids = speaker_ids

        if self.config.cohort.speaker_level:
            if speaker_ids is None:
                raise ValueError(
                    "speaker_ids required when cohort.speaker_level=True. "
                    "Either provide speaker_ids or set cohort.speaker_level=False."
                )
            self._speaker_cohort = self._aggregate_by_speaker(
                embeddings, speaker_ids
            )
        else:
            self._speaker_cohort = embeddings

        # Compute mean vector if using cohort-based centering
        if self.config.mean_source == MeanSource.COHORT.value:
            self.mean_vector = self._speaker_cohort.mean(dim=0)

    def _aggregate_by_speaker(
        self, embeddings: torch.Tensor, speaker_ids: torch.Tensor
    ) -> torch.Tensor:
        """Average embeddings per speaker.

        Args:
            embeddings: [N, embed_dim]
            speaker_ids: [N] integer speaker IDs

        Returns:
            Speaker-averaged embeddings [num_speakers, embed_dim]
        """
        unique_speakers = torch.unique(speaker_ids)
        speaker_embeds = []

        for spk_id in unique_speakers:
            mask = speaker_ids == spk_id
            spk_embed = embeddings[mask].mean(dim=0)
            speaker_embeds.append(spk_embed)

        return torch.stack(speaker_embeds)

    def aggregate_enrollment(
        self,
        embeddings: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Aggregate multi-utterance enrollment embeddings.

        Args:
            embeddings: Enrollment utterance embeddings [num_utts, embed_dim]
            lengths: Utterance lengths for length-weighted aggregation [num_utts]

        Returns:
            Single aggregated embedding [embed_dim]
        """
        method = EnrollmentAggregation(self.config.enrollment_aggregation)

        if method == EnrollmentAggregation.MEAN:
            return embeddings.mean(dim=0)

        elif method == EnrollmentAggregation.LENGTH_WEIGHTED:
            if lengths is None:
                raise ValueError(
                    "lengths required for length_weighted aggregation"
                )
            weights = lengths.float() / lengths.sum()
            return (embeddings * weights.unsqueeze(-1)).sum(dim=0)

        else:
            raise ValueError(f"Unknown aggregation method: {method}")

    def center(self, embedding: torch.Tensor) -> torch.Tensor:
        """Apply mean centering to embedding(s).

        Args:
            embedding: Single embedding [embed_dim] or batch [N, embed_dim]

        Returns:
            Centered embedding(s), same shape as input
        """
        if self.config.mean_source == MeanSource.NONE.value:
            return embedding

        if self.mean_vector is None:
            raise RuntimeError(
                f"Mean vector not set. Call set_cohort() first when using "
                f"mean_source='{self.config.mean_source}'"
            )

        return embedding - self.mean_vector.to(embedding.device)

    def compute_raw_score(
        self, enroll_embedding: torch.Tensor, test_embedding: torch.Tensor
    ) -> torch.Tensor:
        """Compute raw cosine similarity score.

        Args:
            enroll_embedding: [embed_dim] or [N, embed_dim]
            test_embedding: [embed_dim] or [N, embed_dim]

        Returns:
            Cosine similarity score(s)
        """
        return F.cosine_similarity(enroll_embedding, test_embedding, dim=-1)

    def normalize_score(
        self,
        raw_score: torch.Tensor,
        enroll_embedding: torch.Tensor,
        test_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """Apply score normalization (AS-Norm).

        Args:
            raw_score: Raw cosine similarity score (scalar or batch)
            enroll_embedding: Enrollment embedding(s) [embed_dim] or [N, embed_dim]
            test_embedding: Test embedding(s) [embed_dim] or [N, embed_dim]

        Returns:
            Normalized score(s)
        """
        method = NormMethod(self.config.norm_method)

        if method == NormMethod.NONE:
            return raw_score

        if self._speaker_cohort is None:
            raise RuntimeError(
                "Cohort not set. Call set_cohort() before normalizing scores."
            )

        if method == NormMethod.AS_NORM:
            return self._as_norm(raw_score, enroll_embedding, test_embedding)
        else:
            raise ValueError(f"Unknown normalization method: {method}")

    def _as_norm(
        self,
        score: torch.Tensor,
        enroll_embedding: torch.Tensor,
        test_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """Adaptive Symmetric Normalization.

        Computes normalization statistics from top-K most similar speakers in
        the cohort, separately for enrollment and test embeddings.

        This implementation operates at the speaker level (if configured),
        computing mean similarity per cohort speaker before selecting top-K.
        """
        topk = min(
            self.config.cohort.topk_speakers, self._speaker_cohort.shape[0]
        )
        cohort = self._speaker_cohort.to(enroll_embedding.device)

        # Handle batched vs single embeddings
        if enroll_embedding.dim() == 1:
            return self._as_norm_single(
                score, enroll_embedding, test_embedding, cohort, topk
            )
        else:
            # Batched processing
            normalized = []
            for i in range(enroll_embedding.shape[0]):
                norm_score = self._as_norm_single(
                    score[i],
                    enroll_embedding[i],
                    test_embedding[i],
                    cohort,
                    topk,
                )
                normalized.append(norm_score)
            return torch.stack(normalized)

    def _as_norm_single(
        self,
        score: torch.Tensor,
        enroll_embedding: torch.Tensor,
        test_embedding: torch.Tensor,
        cohort: torch.Tensor,
        topk: int,
    ) -> torch.Tensor:
        """AS-Norm for a single trial."""
        # Enrollment vs cohort
        enroll_scores = F.cosine_similarity(
            enroll_embedding.unsqueeze(0), cohort, dim=-1
        )
        enroll_topk, _ = torch.topk(enroll_scores, topk)
        enroll_mean = enroll_topk.mean()
        enroll_std = enroll_topk.std(unbiased=True) + self.EPS

        # Test vs cohort
        test_scores = F.cosine_similarity(
            test_embedding.unsqueeze(0), cohort, dim=-1
        )
        test_topk, _ = torch.topk(test_scores, topk)
        test_mean = test_topk.mean()
        test_std = test_topk.std(unbiased=True) + self.EPS

        # Symmetric normalization
        normalized = 0.5 * (
            (score - enroll_mean) / enroll_std + (score - test_mean) / test_std
        )

        return normalized

    def score(
        self,
        enroll_embedding: torch.Tensor,
        test_embedding: torch.Tensor,
        return_raw: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """Complete scoring pipeline: center → raw score → normalize.

        This is the main entry point for scoring trials.

        Args:
            enroll_embedding: Enrollment embedding [embed_dim] or [N, embed_dim]
            test_embedding: Test embedding [embed_dim] or [N, embed_dim]
            return_raw: If True, also return the raw (pre-normalization) score

        Returns:
            Normalized score, or (normalized_score, raw_score) if return_raw=True
        """
        # Apply centering
        centered_enroll = self.center(enroll_embedding)
        centered_test = self.center(test_embedding)

        # Compute raw score
        raw_score = self.compute_raw_score(centered_enroll, centered_test)

        # Apply normalization
        normalized_score = self.normalize_score(
            raw_score, centered_enroll, centered_test
        )

        if return_raw:
            return normalized_score, raw_score
        return normalized_score

    def score_batch(
        self,
        enroll_embeddings: torch.Tensor,
        test_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Score a batch of trials, returning both raw and normalized scores.

        Optimized for batch processing during evaluation.

        Args:
            enroll_embeddings: [batch_size, embed_dim]
            test_embeddings: [batch_size, embed_dim]

        Returns:
            (normalized_scores, raw_scores) both [batch_size]
        """
        return self.score(enroll_embeddings, test_embeddings, return_raw=True)


def build_scoring_pipeline(config: Optional[Dict] = None) -> ScoringPipeline:
    """Factory function to build ScoringPipeline from config dict.

    Args:
        config: Config dict with scoring settings (enrollment_aggregation,
                mean_source, norm_method, cohort). Falls back to defaults if None.

    Returns:
        Configured ScoringPipeline instance
    """
    config = config or {}
    cohort_cfg = CohortConfig(**config.get("cohort", {}))
    scoring_cfg = ScoringConfig(
        enrollment_aggregation=config.get("enrollment_aggregation", "mean"),
        mean_source=config.get("mean_source", "cohort"),
        norm_method=config.get("norm_method", "as_norm"),
        cohort=cohort_cfg,
    )
    return ScoringPipeline(scoring_cfg)
