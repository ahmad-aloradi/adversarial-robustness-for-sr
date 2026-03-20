"""Correctness tests for scoring pipeline.

Compares refactored batched AS-Norm against a known-correct reference
implementation (per-trial loop with F.cosine_similarity).
"""

import torch
import torch.nn.functional as F

from src.modules.scoring import (
    EPS,
    CohortConfig,
    CosineScorer,
    ScoreNormalizer,
    ScoringConfig,
    ScoringPipeline,
    aggregate_embeddings,
    build_scoring_pipeline,
    build_speaker_cohort,
    l2_normalize,
)

ATOL = 1e-6


# ── Reference implementation (old loop-based AS-Norm) ─────────────


def _reference_as_norm_single(
    score, enroll, test, cohort, topk, mean_vector=None
):
    """Per-trial AS-Norm using F.cosine_similarity (the old code)."""
    if mean_vector is not None:
        cohort = cohort - mean_vector
    e_scores = F.cosine_similarity(enroll.unsqueeze(0), cohort, dim=-1)
    e_topk, _ = torch.topk(e_scores, topk)
    e_mean = e_topk.mean()
    e_std = e_topk.std(unbiased=True) + EPS

    t_scores = F.cosine_similarity(test.unsqueeze(0), cohort, dim=-1)
    t_topk, _ = torch.topk(t_scores, topk)
    t_mean = t_topk.mean()
    t_std = t_topk.std(unbiased=True) + EPS

    return 0.5 * ((score - e_mean) / e_std + (score - t_mean) / t_std)


def reference_as_norm_batch(
    raw_scores, enroll_batch, test_batch, cohort, topk, mean_vector=None
):
    """Loop-based batch AS-Norm (the old code's approach)."""
    return torch.stack(
        [
            _reference_as_norm_single(
                raw_scores[i],
                enroll_batch[i],
                test_batch[i],
                cohort,
                topk,
                mean_vector=mean_vector,
            )
            for i in range(enroll_batch.shape[0])
        ]
    )


# ── Tests ─────────────────────────────────────────────────────────


class TestScoreNormalizerVsReference:
    """Verify batched matmul AS-Norm matches loop-based F.cosine_similarity."""

    def _make_data(
        self, batch_size=16, embed_dim=192, n_cohort_spk=50, topk=20
    ):
        torch.manual_seed(42)
        enroll = l2_normalize(torch.randn(batch_size, embed_dim))
        test = l2_normalize(torch.randn(batch_size, embed_dim))
        raw_scores = F.cosine_similarity(enroll, test, dim=-1)
        # Cohort: average of unit vectors → NOT unit norm (like real speaker cohort)
        raw_cohort = l2_normalize(torch.randn(n_cohort_spk * 3, embed_dim))
        speaker_ids = torch.arange(n_cohort_spk).repeat_interleave(3)
        cohort = build_speaker_cohort(
            raw_cohort, speaker_ids, speaker_level=True
        )
        mean_vector = cohort.mean(dim=0)
        return enroll, test, raw_scores, cohort, topk, mean_vector

    def test_batched_matches_reference(self):
        enroll, test, raw_scores, cohort, topk, mean_vector = self._make_data()
        normalizer = ScoreNormalizer(cohort, topk, mean_vector=mean_vector)

        new_result = normalizer(raw_scores, enroll, test)
        ref_result = reference_as_norm_batch(
            raw_scores, enroll, test, cohort, topk, mean_vector=mean_vector
        )

        assert torch.allclose(
            new_result, ref_result, atol=ATOL
        ), f"Max diff: {(new_result - ref_result).abs().max().item():.2e}"

    def test_single_trial_matches_reference(self):
        """Single trial (batch_size=1) must also work."""
        enroll, test, raw_scores, cohort, topk, mean_vector = self._make_data(
            batch_size=1
        )
        normalizer = ScoreNormalizer(cohort, topk, mean_vector=mean_vector)

        new_result = normalizer(raw_scores, enroll, test)
        ref_result = reference_as_norm_batch(
            raw_scores, enroll, test, cohort, topk, mean_vector=mean_vector
        )

        assert torch.allclose(
            new_result, ref_result, atol=ATOL
        ), f"Max diff: {(new_result - ref_result).abs().max().item():.2e}"

    def test_topk_larger_than_cohort(self):
        """Topk > cohort size should clamp without error."""
        enroll, test, raw_scores, cohort, topk, mean_vector = self._make_data(
            n_cohort_spk=10, topk=50
        )
        normalizer = ScoreNormalizer(cohort, topk, mean_vector=mean_vector)
        # Should use all 10, not crash
        new_result = normalizer(raw_scores, enroll, test)
        ref_result = reference_as_norm_batch(
            raw_scores,
            enroll,
            test,
            cohort,
            min(50, cohort.shape[0]),
            mean_vector=mean_vector,
        )
        assert torch.allclose(new_result, ref_result, atol=ATOL)


class TestCosineScorer:
    def test_center_projects_to_unit_sphere_before_subtracting(self):
        """Center() L2-normalizes before subtracting mean, so centering happens
        in the same unit-sphere space as the cohort mean vector."""
        torch.manual_seed(0)
        mean_vec = l2_normalize(torch.randn(192))
        scorer = CosineScorer(mean_vector=mean_vec)
        emb = torch.randn(192)

        centered = scorer.center(emb)
        # Manual: l2norm → subtract mean (no final l2norm)
        expected = l2_normalize(emb) - mean_vec
        assert torch.allclose(centered, expected, atol=ATOL)

    def test_score_returns_cosine(self):
        torch.manual_seed(1)
        scorer = CosineScorer()
        enroll = torch.randn(8, 192)
        test = torch.randn(8, 192)
        raw, _, _ = scorer.score(enroll, test)

        expected = F.cosine_similarity(
            l2_normalize(enroll), l2_normalize(test), dim=-1
        )
        assert torch.allclose(raw, expected, atol=ATOL)

    def test_center_without_mean_is_just_l2norm(self):
        """Without mean vector, center() is just L2-normalization."""
        torch.manual_seed(2)
        scorer = CosineScorer()
        emb = torch.randn(16, 192)
        centered = scorer.center(emb)
        norms = centered.norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=ATOL)


class TestAggregateEmbeddings:
    def test_mean_output_norm_leq_one(self):
        """Mean of L2-normalized vectors has norm ≤ 1 (not exactly 1)."""
        torch.manual_seed(3)
        embs = torch.randn(5, 192)
        agg = aggregate_embeddings(embs, method="mean")
        assert agg.norm().item() <= 1.0 + ATOL

    def test_length_weighted(self):
        torch.manual_seed(4)
        embs = torch.randn(3, 192)
        lengths = torch.tensor([100, 200, 300])
        agg = aggregate_embeddings(
            embs, method="length_weighted", lengths=lengths
        )
        assert agg.norm().item() <= 1.0 + ATOL

    def test_single_utterance_preserves_direction(self):
        torch.manual_seed(5)
        emb = torch.randn(1, 192)
        agg = aggregate_embeddings(emb, method="mean")
        # Single utterance: aggregated should equal l2_normalize(emb)
        expected = l2_normalize(emb).squeeze(0)
        assert torch.allclose(agg, expected, atol=ATOL)


class TestBuildSpeakerCohort:
    def test_speaker_averaging(self):
        torch.manual_seed(6)
        # 3 speakers, 4 utterances each
        embs = torch.randn(12, 64)
        spk_ids = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
        cohort = build_speaker_cohort(embs, spk_ids, speaker_level=True)
        assert cohort.shape == (3, 64)

    def test_cohort_embeddings_are_unit_norm(self):
        torch.manual_seed(7)
        embs = torch.randn(12, 64)
        spk_ids = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
        cohort = build_speaker_cohort(embs, spk_ids, speaker_level=True)
        norms = cohort.norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=ATOL)


class TestFullPipeline:
    """End-to-end: build_scoring_pipeline → score."""

    def test_with_as_norm(self):
        torch.manual_seed(8)
        config = {
            "enrollment_aggregation": "mean",
            "mean_source": "cohort",
            "norm_method": "as_norm",
            "cohort": {"speaker_level": True, "topk_speakers": 20},
        }

        # Build cohort
        cohort_embs = torch.randn(150, 128)
        spk_ids = torch.arange(50).repeat_interleave(3)
        pipeline = build_scoring_pipeline(
            config, cohort_embeddings=cohort_embs, cohort_speaker_ids=spk_ids
        )

        # Score batch
        enroll = torch.randn(32, 128)
        test = torch.randn(32, 128)
        norm_scores, raw_scores = pipeline.score(enroll, test)

        assert norm_scores.shape == (32,)
        assert raw_scores.shape == (32,)
        # raw scores should be in [-1, 1] (cosine similarity)
        assert raw_scores.min() >= -1.0 - ATOL
        assert raw_scores.max() <= 1.0 + ATOL

    def test_without_normalization(self):
        torch.manual_seed(9)
        pipeline = build_scoring_pipeline(
            {"norm_method": "none", "mean_source": "none"}
        )
        enroll = torch.randn(8, 64)
        test = torch.randn(8, 64)
        norm_scores, raw_scores = pipeline.score(enroll, test)
        # Without normalization, norm == raw
        assert torch.allclose(norm_scores, raw_scores, atol=ATOL)

    def test_centering_only_no_as_norm(self):
        torch.manual_seed(10)
        config = {
            "mean_source": "cohort",
            "norm_method": "none",
            "cohort": {"speaker_level": False},
        }
        cohort_embs = torch.randn(100, 64)
        pipeline = build_scoring_pipeline(
            config, cohort_embeddings=cohort_embs
        )

        enroll = torch.randn(8, 64)
        test = torch.randn(8, 64)
        norm_scores, raw_scores = pipeline.score(enroll, test)
        # No AS-Norm → norm == raw, but centering should have changed scores
        assert torch.allclose(norm_scores, raw_scores, atol=ATOL)
        assert pipeline.scorer.mean_vector is not None

    def test_multi_enroll_with_as_norm(self):
        """Multi-enrollment + AS-Norm end-to-end."""
        torch.manual_seed(11)
        config = {
            "enrollment_aggregation": "mean",
            "mean_source": "cohort",
            "norm_method": "as_norm",
            "cohort": {"speaker_level": True, "topk_speakers": 10},
        }

        cohort_embs = torch.randn(60, 64)
        spk_ids = torch.arange(20).repeat_interleave(3)
        pipeline = build_scoring_pipeline(
            config, cohort_embeddings=cohort_embs, cohort_speaker_ids=spk_ids
        )

        # Multi-enrollment: 5 utterances per speaker, 1 test
        enroll_utts = torch.randn(5, 64)
        test = torch.randn(1, 64)
        norm_score, raw_score = pipeline.score(
            enroll=torch.empty(0),  # unused when enroll_multi provided
            test=test,
            enroll_multi=enroll_utts,
        )
        assert norm_score.shape == (1,)
        assert raw_score.shape == (1,)

    def test_multi_enroll_without_normalization(self):
        """Multi-enrollment without AS-Norm: norm == raw."""
        torch.manual_seed(12)
        pipeline = build_scoring_pipeline(
            {"norm_method": "none", "mean_source": "none"}
        )

        enroll_utts = torch.randn(4, 64)
        test = torch.randn(1, 64)
        norm_score, raw_score = pipeline.score(
            enroll=torch.empty(0),
            test=test,
            enroll_multi=enroll_utts,
        )
        assert torch.allclose(norm_score, raw_score, atol=ATOL)

    def test_multi_enroll_length_weighted(self):
        """Length-weighted multi-enrollment aggregation."""
        torch.manual_seed(13)
        config = {
            "enrollment_aggregation": "length_weighted",
            "mean_source": "none",
            "norm_method": "none",
        }
        pipeline = build_scoring_pipeline(config)

        enroll_utts = torch.randn(3, 64)
        lengths = torch.tensor([100, 200, 300])
        test = torch.randn(1, 64)
        norm_score, raw_score = pipeline.score(
            enroll=torch.empty(0),
            test=test,
            enroll_multi=enroll_utts,
            enroll_lengths=lengths,
        )
        assert -1.0 - ATOL <= raw_score.item() <= 1.0 + ATOL


class TestScaleMismatchProtection:
    """Verify the fix for the multi-enrollment collapse caused by embedding
    scale mismatch (norm-1 enrollment vs large-norm cohort mean).

    See memory: scoring_scale_mismatch_fix.md
    """

    def test_large_norm_cohort_does_not_collapse_multi_enroll(self):
        """If cohort embeddings have large norms (like AdaBreg ~469), multi-
        enrollment scores should still discriminate target vs non-target."""
        torch.manual_seed(42)
        D = 128
        scale = 469.0  # AdaBreg-like raw embedding scale

        config = {
            "enrollment_aggregation": "mean",
            "mean_source": "cohort",
            "norm_method": "as_norm",
            "cohort": {"speaker_level": True, "topk_speakers": 10},
        }

        # Cohort with large norms (simulating raw encoder output)
        cohort_embs = torch.randn(60, D) * scale
        spk_ids = torch.arange(20).repeat_interleave(3)
        pipeline = build_scoring_pipeline(
            config, cohort_embeddings=cohort_embs, cohort_speaker_ids=spk_ids
        )

        # Mean vector should be small after L2-normalization inside build_speaker_cohort
        assert pipeline.scorer.mean_vector is not None
        mean_norm = pipeline.scorer.mean_vector.norm().item()
        assert (
            mean_norm < 1.0
        ), f"Mean vector norm {mean_norm:.2f} too large — cohort not L2-normalized before averaging"

    def test_multi_enroll_diversity_after_centering(self):
        """After centering, different enrollment sets should produce different
        aggregated embeddings (not collapse to -mean_vector).

        Uses low-D (16) to avoid high-dimensional concentration masking
        collapse.
        """
        torch.manual_seed(43)
        D = 16
        scale = 500.0

        config = {
            "enrollment_aggregation": "mean",
            "mean_source": "cohort",
            "norm_method": "none",
        }

        cohort_embs = torch.randn(60, D) * scale
        spk_ids = torch.arange(20).repeat_interleave(3)
        pipeline = build_scoring_pipeline(
            config, cohort_embeddings=cohort_embs, cohort_speaker_ids=spk_ids
        )

        # Two maximally different enrollment sets: opposite directions
        enroll_a = torch.randn(5, D) * scale
        enroll_b = (
            -enroll_a
        )  # opposite direction guarantees divergent aggregation
        test = torch.randn(1, D) * scale

        score_a, _ = pipeline.score(
            torch.empty(0), test, enroll_multi=enroll_a
        )
        score_b, _ = pipeline.score(
            torch.empty(0), test, enroll_multi=enroll_b
        )

        # Opposite enrollments must produce clearly different scores
        assert not torch.allclose(
            score_a, score_b, atol=1e-2
        ), f"Scores collapsed: score_a={score_a.item():.6f}, score_b={score_b.item():.6f}"

    def test_single_enroll_scale_invariance(self):
        """Single-enrollment scoring should produce same scores regardless of
        input embedding scale (since we L2-normalize)."""
        torch.manual_seed(44)
        D = 64

        pipeline_cfg = {"norm_method": "none", "mean_source": "none"}

        # Score with unit-scale embeddings
        pipeline1 = build_scoring_pipeline(pipeline_cfg)
        enroll = torch.randn(8, D)
        test = torch.randn(8, D)
        _, raw1 = pipeline1.score(enroll, test)

        # Score with large-scale embeddings (same direction)
        pipeline2 = build_scoring_pipeline(pipeline_cfg)
        _, raw2 = pipeline2.score(enroll * 469.0, test * 469.0)

        assert torch.allclose(
            raw1, raw2, atol=ATOL
        ), f"Scores differ across scales: max diff={( raw1 - raw2).abs().max():.2e}"

    def test_cohort_centering_scale_invariance(self):
        """With cohort centering, scores should be the same whether cohort
        embeddings are unit-norm or large-norm (the fix)."""
        torch.manual_seed(45)
        D = 64

        config = {
            "mean_source": "cohort",
            "norm_method": "none",
            "cohort": {"speaker_level": False},
        }

        # Small-scale cohort
        cohort_small = torch.randn(50, D)
        pipeline1 = build_scoring_pipeline(
            config, cohort_embeddings=cohort_small
        )

        # Large-scale cohort (same directions, 469x larger)
        pipeline2 = build_scoring_pipeline(
            config, cohort_embeddings=cohort_small * 469.0
        )

        enroll = torch.randn(8, D)
        test = torch.randn(8, D)
        _, raw1 = pipeline1.score(enroll, test)
        _, raw2 = pipeline2.score(enroll, test)

        assert torch.allclose(
            raw1, raw2, atol=ATOL
        ), f"Cohort scale changes scores: max diff={(raw1 - raw2).abs().max():.2e}"
