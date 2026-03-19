"""Scoring pipeline: enrollment aggregation, cosine scoring, AS-Norm."""

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
    """Aggregate multi-utterance embeddings into one. [N, D] → [D]."""
    embeddings = l2_normalize(embeddings)
    if method == "mean":
        return l2_normalize(embeddings.mean(dim=0))
    if method == "length_weighted":
        if lengths is None:
            raise ValueError("lengths required for length_weighted aggregation")
        weights = (lengths.float() / lengths.sum()).to(embeddings.device)
        return l2_normalize((embeddings * weights.unsqueeze(-1)).sum(dim=0))
    raise ValueError(f"Unknown aggregation method: {method}")


# ── 2. Cosine Scorer ─────────────────────────────────────────────


class CosineScorer:
    """L2-norm → center → L2-norm → cosine similarity."""

    def __init__(self, mean_vector: Optional[torch.Tensor] = None):
        self.mean_vector = mean_vector

    def prepare(self, embedding: torch.Tensor) -> torch.Tensor:
        """L2-norm → center → L2-norm."""
        embedding = l2_normalize(embedding)
        if self.mean_vector is not None:
            embedding = embedding - self.mean_vector.to(embedding.device)
        return l2_normalize(embedding)

    def score(
        self, enroll: torch.Tensor, test: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (raw_score, prepared_enroll, prepared_test)."""
        enroll, test = self.prepare(enroll), self.prepare(test)
        return F.cosine_similarity(enroll, test, dim=-1), enroll, test

    def score_multi_enroll(
        self,
        enroll_utts: torch.Tensor,
        test: torch.Tensor,
        method: str = "mean",
        lengths: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Center each utterance before aggregation. Returns (raw_score, agg_enroll, prepared_test)."""
        enroll = aggregate_embeddings(self.prepare(enroll_utts), method=method, lengths=lengths)
        test = self.prepare(test)
        return F.cosine_similarity(enroll, test, dim=-1), enroll, test


# ── 3. Score Normalization (AS-Norm + cohort handling) ────────────


def build_speaker_cohort(
    embeddings: torch.Tensor,
    speaker_ids: Optional[torch.Tensor],
    speaker_level: bool,
) -> torch.Tensor:
    """Build (optionally speaker-averaged) cohort from raw embeddings. [N, D] → [S, D]."""
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
    """Batched AS-Norm. Expects [N, D] enroll/test from score."""

    def __init__(self, speaker_cohort: torch.Tensor, topk_speakers: int):
        self._speaker_cohort = speaker_cohort
        self._topk = min(topk_speakers, speaker_cohort.shape[0])

    def __call__(
        self,
        raw_score: torch.Tensor,
        enroll: torch.Tensor,
        test: torch.Tensor,
    ) -> torch.Tensor:
        """AS-Norm for a batch of trials. enroll/test: [N, D], raw_score: [N]."""
        cohort = self._speaker_cohort.to(enroll.device)

        # [N, D] @ [D, S] → [N, S]
        e_scores = enroll @ cohort.T
        t_scores = test @ cohort.T

        # Top-K per trial: [N, K]
        e_topk, _ = torch.topk(e_scores, self._topk, dim=-1)
        t_topk, _ = torch.topk(t_scores, self._topk, dim=-1)

        e_mean, e_std = e_topk.mean(dim=-1), e_topk.std(dim=-1, unbiased=True) + EPS
        t_mean, t_std = t_topk.mean(dim=-1), t_topk.std(dim=-1, unbiased=True) + EPS

        return 0.5 * ((raw_score - e_mean) / e_std + (raw_score - t_mean) / t_std)


# ── API for sv.py ──────────────────────────────────────────────


class ScoringPipeline:
    """Thin API composing aggregation, scoring, and normalization."""

    def __init__(self, config: ScoringConfig):
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

    def set_cohort(
        self,
        embeddings: torch.Tensor,
        speaker_ids: Optional[torch.Tensor] = None,
    ) -> None:
        speaker_cohort = build_speaker_cohort(
            embeddings, speaker_ids, self.config.cohort.speaker_level
        )
        if self.config.mean_source == "cohort":
            self.scorer.mean_vector = speaker_cohort.mean(dim=0)
        if self.config.norm_method != "none":
            self.normalizer = ScoreNormalizer(speaker_cohort, self.config.cohort.topk_speakers)

    def score(
        self,
        enroll: torch.Tensor,
        test: torch.Tensor,
        enroll_multi: Optional[torch.Tensor] = None,
        enroll_lengths: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if enroll_multi is not None:
            raw, enroll, test = self.scorer.score_multi_enroll(
                enroll_multi, test,
                method=self.config.enrollment_aggregation,
                lengths=enroll_lengths,
            )
        else:
            raw, enroll, test = self.scorer.score(enroll, test)

        norm = self.normalizer(raw, enroll, test) if self.normalizer else raw
        return norm, raw


def build_scoring_pipeline(config: Optional[Dict] = None) -> ScoringPipeline:
    config = config or {}
    cohort_cfg = CohortConfig(**(config.get("cohort") or {}))
    scoring_cfg = ScoringConfig(
        enrollment_aggregation=config.get("enrollment_aggregation"),
        mean_source=config.get("mean_source"),
        norm_method=config.get("norm_method"),
        cohort=cohort_cfg,
    )
    return ScoringPipeline(scoring_cfg)
