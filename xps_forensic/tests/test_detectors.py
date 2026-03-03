"""Tests for detector interfaces."""
import numpy as np
import pytest
from xps_forensic.detectors.base import DetectorOutput, BaseDetector
from xps_forensic.detectors.bam import BAMDetector
from xps_forensic.detectors.sal import SALDetector
from xps_forensic.detectors.cfprf import CFPRFDetector
from xps_forensic.detectors.mrm import MRMDetector


class TestDetectorOutput:
    def test_output_fields(self):
        out = DetectorOutput(
            utterance_id="utt_001",
            frame_scores=np.linspace(0, 1, 100),
            utterance_score=0.7,
            frame_shift_ms=20,
            detector_name="test",
        )
        assert out.n_frames == 100
        assert out.duration_ms == 2000
        assert out.utterance_score == pytest.approx(0.7)

    def test_binarize(self):
        scores = np.array([0.1, 0.3, 0.6, 0.9, 0.2])
        out = DetectorOutput(
            utterance_id="utt",
            frame_scores=scores,
            utterance_score=0.5,
            frame_shift_ms=20,
            detector_name="test",
        )
        binary = out.binarize(threshold=0.5)
        np.testing.assert_array_equal(binary, [0, 0, 1, 1, 0])

    def test_binarize_default_threshold(self):
        scores = np.array([0.4, 0.5, 0.6])
        out = DetectorOutput(
            utterance_id="utt",
            frame_scores=scores,
            utterance_score=0.5,
            frame_shift_ms=20,
            detector_name="test",
        )
        binary = out.binarize()
        np.testing.assert_array_equal(binary, [0, 1, 1])

    def test_scores_at_resolution(self):
        # 10 frames at 20ms = 200ms total, request 40ms resolution -> 5 segments
        scores = np.arange(10, dtype=float)
        out = DetectorOutput(
            utterance_id="utt",
            frame_scores=scores,
            utterance_score=0.5,
            frame_shift_ms=20,
            detector_name="test",
        )
        coarse = out.scores_at_resolution(40)
        assert len(coarse) == 5
        np.testing.assert_allclose(coarse[0], np.mean([0.0, 1.0]))
        np.testing.assert_allclose(coarse[4], np.mean([8.0, 9.0]))

    def test_scores_at_resolution_same(self):
        scores = np.array([0.1, 0.5, 0.9])
        out = DetectorOutput(
            utterance_id="utt",
            frame_scores=scores,
            utterance_score=0.5,
            frame_shift_ms=20,
            detector_name="test",
        )
        coarse = out.scores_at_resolution(20)
        np.testing.assert_array_equal(coarse, scores)

    def test_n_frames_empty(self):
        out = DetectorOutput(
            utterance_id="utt",
            frame_scores=np.array([]),
            utterance_score=0.0,
            frame_shift_ms=20,
            detector_name="test",
        )
        assert out.n_frames == 0
        assert out.duration_ms == 0


class TestBaseDetectorInterface:
    """Test that BaseDetector cannot be instantiated directly."""

    def test_abstract(self):
        with pytest.raises(TypeError):
            BaseDetector()


class TestBAMDetector:
    def test_instantiation(self):
        det = BAMDetector(device="cpu")
        assert det.name == "BAM"
        assert det.frame_shift_ms == 20

    def test_predict_shape(self):
        det = BAMDetector(device="cpu")
        assert hasattr(det, "predict")
        assert hasattr(det, "load_model")

    def test_load_model_requires_external_dir(self):
        det = BAMDetector(device="cpu")
        with pytest.raises(ValueError, match="external_dir"):
            det.load_model()

    def test_predict_requires_loaded_model(self):
        det = BAMDetector(device="cpu")
        with pytest.raises(RuntimeError, match="load_model"):
            det.predict(np.zeros(16000))


class TestSALDetector:
    def test_instantiation(self):
        det = SALDetector(device="cpu")
        assert det.name == "SAL"

    def test_load_model_requires_external_dir(self):
        det = SALDetector(device="cpu")
        with pytest.raises(ValueError, match="external_dir"):
            det.load_model()

    def test_predict_requires_loaded_model(self):
        det = SALDetector(device="cpu")
        with pytest.raises(RuntimeError, match="load_model"):
            det.predict(np.zeros(16000))


class TestCFPRFDetector:
    def test_instantiation(self):
        det = CFPRFDetector(device="cpu")
        assert det.name == "CFPRF"

    def test_load_model_requires_external_dir(self):
        det = CFPRFDetector(device="cpu")
        with pytest.raises(ValueError, match="external_dir"):
            det.load_model()

    def test_predict_requires_loaded_model(self):
        det = CFPRFDetector(device="cpu")
        with pytest.raises(RuntimeError, match="load_model"):
            det.predict(np.zeros(16000))


class TestMRMDetector:
    def test_instantiation(self):
        det = MRMDetector(device="cpu")
        assert det.name == "MRM"

    def test_load_model_requires_external_dir(self):
        det = MRMDetector(device="cpu")
        with pytest.raises(ValueError, match="external_dir"):
            det.load_model()

    def test_predict_requires_loaded_model(self):
        det = MRMDetector(device="cpu")
        with pytest.raises(RuntimeError, match="load_model"):
            det.predict(np.zeros(16000))
