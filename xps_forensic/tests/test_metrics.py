"""Tests for core metrics and statistical utilities."""
import numpy as np
import pytest

from xps_forensic.utils.metrics import (
    compute_eer,
    compute_segment_eer,
    compute_segment_f1,
    compute_tFNR,
    compute_tFDR,
    compute_tIoU,
)
from xps_forensic.utils.stats import (
    bootstrap_ci,
    binomial_coverage_test,
    friedman_nemenyi,
    holm_bonferroni,
)


# ── EER tests ────────────────────────────────────────────────────────────────


class TestEER:
    """Tests for Equal Error Rate computation."""

    def test_perfect_separation(self):
        """Perfectly separated scores should yield EER near 0."""
        scores = np.array([0.0, 0.1, 0.2, 0.3, 0.8, 0.9, 0.95, 1.0])
        labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        eer, threshold = compute_eer(scores, labels)
        assert eer < 0.01, f"EER should be < 0.01 for perfect separation, got {eer}"

    def test_random_scores(self, rng):
        """Random scores should yield EER around 0.5."""
        n = 2000
        labels = np.concatenate([np.zeros(n), np.ones(n)])
        scores = rng.uniform(0, 1, size=2 * n)
        eer, _ = compute_eer(scores, labels)
        assert 0.3 < eer < 0.7, f"EER for random scores should be ~0.5, got {eer}"

    def test_returns_threshold(self):
        """EER function must return a valid threshold."""
        scores = np.array([0.1, 0.2, 0.8, 0.9])
        labels = np.array([0, 0, 1, 1])
        eer, threshold = compute_eer(scores, labels)
        assert isinstance(threshold, float)


# ── Segment-level metrics tests ──────────────────────────────────────────────


class TestSegmentMetrics:
    """Tests for temporal localization metrics (tFNR, tFDR, tIoU)."""

    def test_perfect_localization(self):
        """Perfect predictions should give tFNR=0, tFDR=0, tIoU=1."""
        true = np.array([0, 0, 1, 1, 1, 0, 0])
        pred = np.array([0, 0, 1, 1, 1, 0, 0])
        assert compute_tFNR(pred, true) == 0.0
        assert compute_tFDR(pred, true) == 0.0
        assert compute_tIoU(pred, true) == 1.0

    def test_missed_detection(self):
        """Predicting all-zero when there IS spoof: tFNR=1, tIoU=0."""
        true = np.array([0, 0, 1, 1, 1, 0, 0])
        pred = np.array([0, 0, 0, 0, 0, 0, 0])
        assert compute_tFNR(pred, true) == 1.0
        assert compute_tIoU(pred, true) == 0.0

    def test_all_real(self):
        """When ground truth is all-real, tFNR should be 0."""
        true = np.zeros(100, dtype=int)
        pred = np.zeros(100, dtype=int)
        assert compute_tFNR(pred, true) == 0.0
        assert compute_tIoU(pred, true) == 1.0

    def test_false_alarm(self):
        """Predicting spoof on real frames: tFDR should be > 0."""
        true = np.array([0, 0, 0, 0, 0])
        pred = np.array([1, 1, 0, 0, 0])
        assert compute_tFDR(pred, true) == 1.0
        assert compute_tIoU(pred, true) == 0.0

    def test_partial_overlap(self):
        """Partial overlap: tIoU should be between 0 and 1."""
        true = np.array([0, 1, 1, 1, 0])
        pred = np.array([0, 0, 1, 1, 1])
        tiou = compute_tIoU(pred, true)
        assert 0.0 < tiou < 1.0

    def test_segment_f1_perfect(self):
        """Perfect frame-level predictions should give F1 = 1."""
        labels = np.array([0, 0, 1, 1, 0])
        preds = np.array([0, 0, 1, 1, 0])
        assert compute_segment_f1(preds, labels) == 1.0

    def test_segment_f1_all_real(self):
        """All-zero predictions and labels should give F1 = 1."""
        labels = np.zeros(50, dtype=int)
        preds = np.zeros(50, dtype=int)
        assert compute_segment_f1(preds, labels) == 1.0

    def test_segment_eer(self, rng):
        """Segment EER should be computable at different resolutions."""
        n = 1000
        frame_labels = np.zeros(n, dtype=int)
        frame_labels[300:700] = 1
        frame_scores = rng.uniform(0.1, 0.3, size=n)
        frame_scores[300:700] = rng.uniform(0.7, 0.95, size=400)

        eer_20, _ = compute_segment_eer(frame_scores, frame_labels, 20)
        eer_160, _ = compute_segment_eer(frame_scores, frame_labels, 160)
        # Both should be low for well-separated data
        assert eer_20 < 0.15
        assert eer_160 < 0.15


# ── Bootstrap CI tests ───────────────────────────────────────────────────────


class TestBootstrap:
    """Tests for bootstrap confidence intervals."""

    def test_ci_contains_mean(self, rng):
        """The 95% CI should contain the true mean for normal data."""
        data = rng.normal(loc=5.0, scale=1.0, size=1000)
        lo, hi = bootstrap_ci(data, statistic="mean", confidence=0.95)
        sample_mean = np.mean(data)
        assert lo <= sample_mean <= hi

    def test_ci_width(self, rng):
        """CI width should be reasonable (< 0.3) for n=1000 standard normal."""
        data = rng.normal(loc=0.0, scale=1.0, size=1000)
        lo, hi = bootstrap_ci(data, statistic="mean", confidence=0.95)
        width = hi - lo
        assert width < 0.3, f"CI width {width} is too large"

    def test_median_ci(self, rng):
        """Median CI should also contain the sample median."""
        data = rng.normal(loc=3.0, scale=2.0, size=500)
        lo, hi = bootstrap_ci(data, statistic="median", confidence=0.95)
        sample_median = np.median(data)
        assert lo <= sample_median <= hi


# ── Binomial coverage tests ─────────────────────────────────────────────────


class TestBinomialCoverage:
    """Tests for binomial coverage hypothesis testing."""

    def test_good_coverage(self):
        """96/100 covered at alpha=0.05 (target 95%): should NOT reject."""
        p = binomial_coverage_test(96, 100, alpha=0.05)
        assert p > 0.05, f"p={p}: good coverage should not be rejected"

    def test_bad_coverage(self):
        """80/100 covered at alpha=0.05 (target 95%): should reject."""
        p = binomial_coverage_test(80, 100, alpha=0.05)
        assert p < 0.05, f"p={p}: bad coverage should be rejected"

    def test_perfect_coverage(self):
        """100/100 covered: p-value should be large."""
        p = binomial_coverage_test(100, 100, alpha=0.05)
        assert p > 0.5


# ── Friedman-Nemenyi tests ───────────────────────────────────────────────────


class TestFriedmanNemenyi:
    """Tests for Friedman test with Nemenyi post-hoc."""

    def test_clear_ranking(self):
        """When one method always wins, its mean rank should be 1."""
        # 5 datasets, 3 algorithms; col 0 always highest
        results = np.array([
            [0.9, 0.7, 0.5],
            [0.85, 0.65, 0.45],
            [0.88, 0.68, 0.48],
            [0.92, 0.72, 0.52],
            [0.87, 0.67, 0.47],
        ])
        out = friedman_nemenyi(results)
        assert out["mean_ranks"][0] == 1.0
        assert out["mean_ranks"][2] == 3.0
        assert out["friedman_p"] < 0.05

    def test_returns_cd(self):
        """Critical difference should be a positive number."""
        results = np.array([
            [0.9, 0.8, 0.7],
            [0.85, 0.75, 0.65],
            [0.88, 0.78, 0.68],
        ])
        out = friedman_nemenyi(results)
        assert out["critical_difference"] > 0


# ── Holm-Bonferroni tests ───────────────────────────────────────────────────


class TestHolmBonferroni:
    """Tests for Holm-Bonferroni multiple testing correction."""

    def test_all_significant(self):
        """Very small p-values should all be rejected."""
        p_values = [0.001, 0.002, 0.003]
        rejected = holm_bonferroni(p_values, alpha=0.05)
        assert all(rejected)

    def test_none_significant(self):
        """Large p-values should not be rejected."""
        p_values = [0.5, 0.6, 0.7]
        rejected = holm_bonferroni(p_values, alpha=0.05)
        assert not any(rejected)

    def test_partial_rejection(self):
        """Mixed p-values: only small ones should be rejected."""
        p_values = [0.001, 0.04, 0.5]
        rejected = holm_bonferroni(p_values, alpha=0.05)
        assert rejected[0] is True
        # p=0.04 with correction: alpha/(3-1) = 0.025, so 0.04 > 0.025 → not rejected
        assert rejected[1] is False
        assert rejected[2] is False
