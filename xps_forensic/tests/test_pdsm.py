"""Tests for PDSM-PS components."""
import numpy as np
import pytest
from xps_forensic.pdsm_ps.alignment import PhonemeSegment, align_phonemes_mock


class TestPhonemeAlignment:
    def test_phoneme_segment(self):
        seg = PhonemeSegment(phoneme="AH", start_sec=0.1, end_sec=0.2, confidence=0.95)
        assert seg.duration_sec == pytest.approx(0.1)
        assert seg.start_frame(frame_shift_ms=20) == 5
        assert seg.end_frame(frame_shift_ms=20) == 10

    def test_mock_alignment(self):
        segments = align_phonemes_mock(duration_sec=1.0, n_phonemes=10)
        assert len(segments) == 10
        assert segments[0].start_sec == pytest.approx(0.0)
        assert segments[-1].end_sec == pytest.approx(1.0)
        for seg in segments:
            assert seg.duration_sec > 0


from xps_forensic.pdsm_ps.saliency import compute_saliency_mock


class TestSaliency:
    def test_mock_saliency(self):
        n_frames = 100
        saliency = compute_saliency_mock(n_frames)
        assert saliency.shape == (n_frames,)
        assert np.all(saliency >= 0)


from xps_forensic.pdsm_ps.discretize import (
    discretize_by_phonemes,
    discretize_by_fixed_window,
    PhonemeSaliency,
)


class TestDiscretize:
    def test_phoneme_discretization(self):
        frame_saliency = np.random.uniform(0, 1, 50)
        phonemes = align_phonemes_mock(duration_sec=1.0, n_phonemes=10)
        result = discretize_by_phonemes(frame_saliency, phonemes, frame_shift_ms=20)
        assert len(result) == 10
        for ps in result:
            assert isinstance(ps, PhonemeSaliency)
            assert ps.mean_saliency >= 0

    def test_fixed_window_discretization(self):
        frame_saliency = np.random.uniform(0, 1, 100)
        result = discretize_by_fixed_window(frame_saliency, window_ms=100, frame_shift_ms=20)
        assert len(result) == 20  # 100 frames * 20ms / 100ms = 20 windows
        for ws in result:
            assert ws.mean_saliency >= 0

    def test_discretized_vs_raw(self):
        """Discretized saliency should have same total as raw."""
        frame_saliency = np.random.uniform(0, 1, 50)
        phonemes = align_phonemes_mock(duration_sec=1.0, n_phonemes=10)
        result = discretize_by_phonemes(frame_saliency, phonemes, frame_shift_ms=20)
        total_disc = sum(ps.mean_saliency * ps.n_frames for ps in result)
        total_raw = frame_saliency.sum()
        assert abs(total_disc - total_raw) < total_raw * 0.2


from xps_forensic.pdsm_ps.faithfulness import (
    normalized_aopc,
    comprehensiveness,
    sufficiency,
    phoneme_iou,
)


class TestFaithfulness:
    def test_normalized_aopc(self):
        original_score = 0.9
        perturbed_scores = [0.8, 0.6, 0.3, 0.1]
        aopc = normalized_aopc(original_score, perturbed_scores)
        assert 0 <= aopc <= 1

    def test_comprehensiveness(self):
        original = 0.9
        without_top = 0.3
        comp = comprehensiveness(original, without_top)
        assert comp == pytest.approx(0.6)

    def test_sufficiency(self):
        original = 0.9
        with_only_top = 0.85
        suff = sufficiency(original, with_only_top)
        assert suff == pytest.approx(0.05)

    def test_phoneme_iou(self):
        salient_indices = {2, 3, 4, 5, 6}
        gt_indices = {3, 4, 5, 6, 7}
        iou = phoneme_iou(salient_indices, gt_indices)
        assert iou == pytest.approx(4 / 6)


from xps_forensic.pdsm_ps import PDSMPSPipeline


class TestPDSMPSPipeline:
    def test_run_mock(self):
        pipeline = PDSMPSPipeline(aligner="mock", saliency_method="mock")
        result = pipeline.run(
            frame_saliency=np.random.uniform(0, 1, 50),
            duration_sec=1.0,
            spoofed_frame_mask=np.concatenate([np.zeros(20), np.ones(15), np.zeros(15)]),
        )
        assert "phoneme_saliencies" in result
        assert "top_k_phonemes" in result
        assert "phoneme_iou" in result
        assert len(result["phoneme_saliencies"]) > 0
