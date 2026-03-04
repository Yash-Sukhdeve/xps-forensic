"""Tests for CPSL conformal prediction components.

Tests cover all four sub-modules:
- nonconformity: score aggregation functions (max, logsumexp)
- scp_aps: Split Conformal Prediction + Adaptive Prediction Sets (Stage 1)
- crc: Conformal Risk Control on tFNR (Stage 2)
- composed: Full CPSL pipeline (Stage 1 + Stage 2)
"""
import numpy as np
import pytest
from xps_forensic.cpsl.nonconformity import (
    max_score,
    logsumexp_score,
    compute_nonconformity,
)


class TestNonconformityScores:
    def test_max_score(self):
        frame_scores = np.array([0.1, 0.3, 0.9, 0.2])
        assert max_score(frame_scores) == pytest.approx(0.9)

    def test_logsumexp_score(self):
        frame_scores = np.array([0.1, 0.3, 0.9, 0.2])
        lse = logsumexp_score(frame_scores, beta=10.0)
        assert lse > 0.0
        assert np.isfinite(lse)

    def test_logsumexp_beta_sensitivity(self):
        frame_scores = np.array([0.5, 0.5, 0.5])
        lse_low = logsumexp_score(frame_scores, beta=1.0)
        lse_high = logsumexp_score(frame_scores, beta=20.0)
        assert lse_high > lse_low or abs(lse_high - lse_low) < 0.01

    def test_compute_nonconformity_batch(self, dummy_frame_scores):
        scores = compute_nonconformity(dummy_frame_scores, method="max")
        assert len(scores) == len(dummy_frame_scores)
        assert all(0 <= s <= 1 for s in scores)


from xps_forensic.cpsl.scp_aps import SCPAPS


class TestSCPAPS:
    def test_calibrate_and_predict(self, rng):
        n_cal = 200
        cal_scores = rng.uniform(0, 1, n_cal)
        cal_labels = rng.integers(0, 3, n_cal)
        scp = SCPAPS(alpha=0.10, classes=["real", "partially_fake", "fully_fake"])
        scp.calibrate(cal_scores, cal_labels)
        test_scores = rng.uniform(0, 1, 50)
        pred_sets = scp.predict(test_scores)
        assert len(pred_sets) == 50
        for ps in pred_sets:
            assert isinstance(ps, set)
            assert len(ps) >= 1
            assert ps.issubset({0, 1, 2})

    def test_coverage_guarantee(self, rng):
        n = 1000
        alpha = 0.10
        scores = rng.uniform(0, 1, n)
        labels = rng.integers(0, 3, n)
        cal_scores, test_scores = scores[:800], scores[800:]
        cal_labels, test_labels = labels[:800], labels[800:]
        scp = SCPAPS(alpha=alpha, classes=["real", "partially_fake", "fully_fake"])
        scp.calibrate(cal_scores, cal_labels)
        pred_sets = scp.predict(test_scores)
        covered = sum(1 for ps, y in zip(pred_sets, test_labels) if y in ps)
        coverage = covered / len(test_labels)
        assert coverage >= 1 - alpha - 0.10

    def test_prediction_set_size(self, rng):
        n = 500
        scores = rng.uniform(0, 1, n)
        labels = rng.integers(0, 3, n)
        scp_loose = SCPAPS(alpha=0.20, classes=["real", "partially_fake", "fully_fake"])
        scp_loose.calibrate(scores, labels)
        sets_loose = scp_loose.predict(scores[:50])
        scp_tight = SCPAPS(alpha=0.01, classes=["real", "partially_fake", "fully_fake"])
        scp_tight.calibrate(scores, labels)
        sets_tight = scp_tight.predict(scores[:50])
        avg_loose = np.mean([len(s) for s in sets_loose])
        avg_tight = np.mean([len(s) for s in sets_tight])
        assert avg_tight >= avg_loose


from xps_forensic.cpsl.crc import ConformalRiskControl


class TestCRC:
    def test_calibrate_threshold(self, dummy_frame_scores, dummy_segment_labels):
        crc = ConformalRiskControl(alpha=0.10, risk_metric="tFNR")
        crc.calibrate(
            frame_scores=dummy_frame_scores[:8],
            frame_labels=dummy_segment_labels[:8],
        )
        assert crc.threshold is not None
        assert 0 <= crc.threshold <= 1

    def test_predict(self, dummy_frame_scores, dummy_segment_labels):
        crc = ConformalRiskControl(alpha=0.10, risk_metric="tFNR")
        crc.calibrate(
            frame_scores=dummy_frame_scores[:8],
            frame_labels=dummy_segment_labels[:8],
        )
        preds = crc.predict(dummy_frame_scores[8:])
        assert len(preds) == 2
        for p in preds:
            assert p.dtype == int
            assert set(np.unique(p)).issubset({0, 1})

    def test_tFNR_controlled(self, rng):
        """Verify that CRC controls tFNR on test set.

        Calibration and evaluation are both restricted to utterances
        containing spoofed segments, consistent with the composed
        pipeline where CRC Stage 2 is applied only to partial spoofs.
        """
        alpha = 0.20
        n = 100
        frame_scores = []
        frame_labels = []
        for i in range(n):
            n_frames = 200
            scores = rng.uniform(0.1, 0.4, n_frames)
            labels = np.zeros(n_frames, dtype=int)
            # All utterances have spoofed regions for consistent calibration
            start, end = 50, 150
            scores[start:end] = rng.uniform(0.6, 0.95, end - start)
            labels[start:end] = 1
            frame_scores.append(scores)
            frame_labels.append(labels)

        crc = ConformalRiskControl(alpha=alpha, risk_metric="tFNR")
        crc.calibrate(frame_scores[:80], frame_labels[:80])

        from xps_forensic.utils.metrics import compute_tFNR
        test_tfnrs = []
        for scores, labels in zip(frame_scores[80:], frame_labels[80:]):
            pred = crc.predict([scores])[0]
            test_tfnrs.append(compute_tFNR(pred, labels))

        if test_tfnrs:
            mean_tfnr = np.mean(test_tfnrs)
            assert mean_tfnr <= alpha + 0.15


from xps_forensic.cpsl.composed import CPSLPipeline


class TestCPSLPipeline:
    def test_composed_guarantee(self):
        alpha1 = 0.05
        alpha2 = 0.10
        pipeline = CPSLPipeline(alpha_utterance=alpha1, alpha_segment=alpha2)
        assert pipeline.composed_guarantee == pytest.approx(0.855)

    def test_end_to_end(self, rng):
        n = 200
        frame_scores_list = []
        utt_labels = []
        frame_labels_list = []

        for i in range(n):
            n_frames = 150
            scores = rng.uniform(0.1, 0.3, n_frames)
            labels = np.zeros(n_frames, dtype=int)
            if i % 3 == 0:
                scores[40:100] = rng.uniform(0.7, 0.95, 60)
                labels[40:100] = 1
                utt_labels.append(1)
            elif i % 3 == 1:
                scores[:] = rng.uniform(0.8, 0.99, n_frames)
                labels[:] = 1
                utt_labels.append(2)
            else:
                utt_labels.append(0)
            frame_scores_list.append(scores)
            frame_labels_list.append(labels)

        pipeline = CPSLPipeline(alpha_utterance=0.10, alpha_segment=0.10)
        pipeline.calibrate(
            frame_scores_list[:160],
            np.array(utt_labels[:160]),
            frame_labels_list[:160],
        )

        results = pipeline.predict(frame_scores_list[160:])
        assert len(results) == 40
        for r in results:
            assert "prediction_set" in r
            assert "segment_predictions" in r
