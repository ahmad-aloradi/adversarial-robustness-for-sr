"""Scoring pipeline: enrollment aggregation, cosine scoring, AS-Norm.

Pipeline:
    raw embs → L2 → aggregate → center (L2 → subtract mean) → cosine → AS-Norm

L2-normalization is applied explicitly where needed: before aggregation
(equal utterance contribution), before centering (unit-sphere alignment),
and at cohort construction output.

Diverges from WeSpeaker, which centers raw embeddings before any L2-norm:
    raw embs → subtract mean(raw) → cosine (L2-norms internally)

Our approach is robust to variable embedding scales (e.g. AdaBreg 469×
vs LinBreg 1×) at the cost of not exactly reproducing WeSpeaker numbers.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from src import utils

log = utils.get_pylogger(__name__)

EPS = 1e-9


def l2_normalize(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x, p=2, dim=-1, eps=EPS)


# ── Configuration ─────────────────────────────────────────────────


@dataclass
class CohortConfig:
    speaker_level: bool = True
    topk_speakers: int = 300
    # These are only placeholders for backward compatiblity
    min_size: int = None
    max_size: int = None


@dataclass
class ScoringConfig:
    enrollment_aggregation: Optional[str] = None  # "mean" or "length_weighted"
    mean_source: Optional[str] = None  # "cohort" or "none"
    norm_method: Optional[str] = None  # "as_norm" or "none"
    cohort: CohortConfig = field(default_factory=CohortConfig)


# ── 1. Enrollment Aggregation ─────────────────────────────────────


def aggregate_embeddings(
    embeddings: torch.Tensor,
    method: str = "mean",
    lengths: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Aggregate multi-utterance embeddings into one.

    [N, D] → [D].
    """
    embeddings = l2_normalize(embeddings)
    if method == "mean":
        return embeddings.mean(dim=0)
    if method == "length_weighted":
        if lengths is None:
            raise ValueError(
                "lengths required for length_weighted aggregation"
            )
        weights = (lengths.float() / lengths.sum()).to(embeddings.device)
        return (embeddings * weights.unsqueeze(-1)).sum(dim=0)
    raise ValueError(f"Unknown aggregation method: {method}")


# ── 2. Cosine Scorer ─────────────────────────────────────────────


class CosineScorer:
    """L2-norm → mean-center → cosine similarity.

    We L2-normalize before centering so that all operations happen in unit-
    sphere space regardless of raw embedding scale.

    F.cosine_similarity L2-normalizes internally, so no explicit norm is needed
    after centering.
    """

    def __init__(self, mean_vector: Optional[torch.Tensor] = None):
        self.mean_vector = mean_vector

    def center(self, embedding: torch.Tensor) -> torch.Tensor:
        """L2-norm → subtract mean vector.

        The L2-norm ensures centering happens in the same unit-sphere space
        where the mean vector was computed (from L2-normalized cohort
        embeddings).
        """
        embedding = l2_normalize(embedding)
        if self.mean_vector is not None:
            embedding = embedding - self.mean_vector.to(embedding.device)
        return embedding

    def score(
        self, enroll: torch.Tensor, test: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (raw_score, centered_enroll, centered_test)."""
        enroll, test = self.center(enroll), self.center(test)
        return F.cosine_similarity(enroll, test, dim=-1), enroll, test

    def score_multi_enroll(
        self,
        enroll_utts: torch.Tensor,
        test: torch.Tensor,
        method: str = "mean",
        lengths: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Aggregate (L2-norm → avg), then center (L2-norm → subtract mean),
        then cosine."""
        enroll = aggregate_embeddings(
            enroll_utts, method=method, lengths=lengths
        )
        enroll = self.center(enroll)
        test = self.center(test)
        return F.cosine_similarity(enroll, test, dim=-1), enroll, test


# ── 3. Score Normalization (AS-Norm + cohort handling) ────────────


def build_speaker_cohort(
    embeddings: torch.Tensor,
    speaker_ids: Optional[torch.Tensor],
    speaker_level: bool,
) -> torch.Tensor:
    """Build (optionally speaker-averaged) cohort.

    [N, D] → [S, D], all unit-norm.
    """
    embeddings = l2_normalize(embeddings)
    if speaker_level:
        if speaker_ids is None:
            raise ValueError("speaker_ids required when speaker_level=True")
        unique = torch.unique(speaker_ids)
        embeddings = torch.stack(
            [embeddings[speaker_ids == s].mean(dim=0) for s in unique]
        )
    return l2_normalize(embeddings)


class ScoreNormalizer:
    """Batched AS-Norm using F.cosine_similarity (normalization-agnostic)."""

    def __init__(
        self,
        speaker_cohort: torch.Tensor,
        topk_speakers: int,
        mean_vector: Optional[torch.Tensor] = None,
    ):
        if mean_vector is not None:
            speaker_cohort = speaker_cohort - mean_vector
        self._speaker_cohort = speaker_cohort
        self.k = min(topk_speakers, speaker_cohort.shape[0])

    def __call__(
        self,
        raw_score: torch.Tensor,
        enroll: torch.Tensor,
        test: torch.Tensor,
    ) -> torch.Tensor:
        """AS-Norm for a batch of trials.

        enroll/test: [N, D], raw_score: [N].
        """
        cohort = self._speaker_cohort.to(enroll.device)
        if enroll.dim() == 1:
            enroll = enroll.unsqueeze(0)
        if test.dim() == 1:
            test = test.unsqueeze(0)

        # Compute cohort scores
        e_scores = F.cosine_similarity(
            enroll.unsqueeze(1), cohort.unsqueeze(0), dim=-1
        )  # [N, S]
        t_scores = F.cosine_similarity(
            test.unsqueeze(1), cohort.unsqueeze(0), dim=-1
        )  # [N, S]

        # Top-K scores
        e_topk, _ = torch.topk(e_scores, self.k, dim=-1)
        t_topk, _ = torch.topk(t_scores, self.k, dim=-1)

        # compute cohorts stats: (mean_{enroll}, std_{enroll}) and (mean_{test}, std_{test})
        e_mean, e_std = (
            e_topk.mean(dim=-1),
            e_topk.std(dim=-1, unbiased=True) + EPS,
        )
        t_mean, t_std = (
            t_topk.mean(dim=-1),
            t_topk.std(dim=-1, unbiased=True) + EPS,
        )

        # AS-norm (variant 1)
        return 0.5 * (
            (raw_score - e_mean) / e_std + (raw_score - t_mean) / t_std
        )


# ── API for sv.py ──────────────────────────────────────────────


class ScoringPipeline:
    """Thin API composing aggregation, scoring, and normalization.

    Fully configured at construction — no mutable state after __init__.
    """

    def __init__(
        self,
        config: ScoringConfig,
        cohort_embeddings: Optional[torch.Tensor] = None,
        cohort_speaker_ids: Optional[torch.Tensor] = None,
    ):
        if config.enrollment_aggregation is None:
            log.warning("enrollment_aggregation not set; defaulting to 'mean'")
            config.enrollment_aggregation = "mean"
        if config.mean_source is None:
            config.mean_source = "none"
        if config.norm_method is None:
            config.norm_method = "none"
        self.config = config
        self.scorer = CosineScorer()
        self.normalizer: Optional[ScoreNormalizer] = None

        if cohort_embeddings is not None:
            speaker_cohort = build_speaker_cohort(
                cohort_embeddings,
                cohort_speaker_ids,
                config.cohort.speaker_level,
            )
            mean_vector = None
            if config.mean_source == "cohort":
                mean_vector = speaker_cohort.mean(dim=0)
                self.scorer.mean_vector = mean_vector
            if config.norm_method != "none":
                self.normalizer = ScoreNormalizer(
                    speaker_cohort,
                    config.cohort.topk_speakers,
                    mean_vector=mean_vector,
                )

    def score(
        self,
        enroll: torch.Tensor,
        test: torch.Tensor,
        enroll_multi: Optional[torch.Tensor] = None,
        enroll_lengths: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if enroll_multi is not None:
            raw, enroll, test = self.scorer.score_multi_enroll(
                enroll_multi,
                test,
                method=self.config.enrollment_aggregation,
                lengths=enroll_lengths,
            )
        else:
            raw, enroll, test = self.scorer.score(enroll, test)

        norm = self.normalizer(raw, enroll, test) if self.normalizer else raw
        return norm, raw


def build_scoring_pipeline(
    config: Optional[Dict] = None,
    cohort_embeddings: Optional[torch.Tensor] = None,
    cohort_speaker_ids: Optional[torch.Tensor] = None,
) -> ScoringPipeline:
    config = config or {}
    cohort_cfg = CohortConfig(**(config.get("cohort") or {}))
    scoring_cfg = ScoringConfig(
        enrollment_aggregation=config.get("enrollment_aggregation"),
        mean_source=config.get("mean_source"),
        norm_method=config.get("norm_method"),
        cohort=cohort_cfg,
    )
    return ScoringPipeline(scoring_cfg, cohort_embeddings, cohort_speaker_ids)
