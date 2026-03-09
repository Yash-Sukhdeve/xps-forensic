"""Tests for detector interfaces."""
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

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
        assert det.frame_shift_ms == 160

    def test_instantiation_custom_params(self):
        det = BAMDetector(
            device="cpu",
            external_dir="/tmp/bam",
            ssl_ckpt="/tmp/wavlm.pt",
            resolution=0.32,
        )
        assert det.external_dir.name == "bam"
        assert det.ssl_ckpt == "/tmp/wavlm.pt"
        assert det.resolution == 0.32

    def test_predict_shape(self):
        det = BAMDetector(device="cpu")
        assert hasattr(det, "predict")
        assert hasattr(det, "load_model")

    def test_load_model_requires_external_dir(self):
        det = BAMDetector(device="cpu")
        with pytest.raises(ValueError, match="external_dir"):
            det.load_model()

    def test_load_model_requires_existing_dir(self):
        det = BAMDetector(external_dir="/nonexistent/path", device="cpu")
        with pytest.raises(FileNotFoundError, match="not found"):
            det.load_model()

    def test_predict_requires_loaded_model(self):
        det = BAMDetector(device="cpu")
        with pytest.raises(RuntimeError, match="load_model"):
            det.predict(np.zeros(16000))

    def test_predict_with_mock_model(self):
        """Test predict with a mock BAM model returning (output, b_pred)."""
        det = BAMDetector(device="cpu")
        n_segments = 10
        # BAM returns (batch, n_segments, 2) logits and (batch, n_segments) boundary
        mock_output = torch.randn(1, n_segments, 2)
        mock_b_pred = torch.sigmoid(torch.randn(1, n_segments))
        mock_model = MagicMock()
        mock_model.return_value = (mock_output, mock_b_pred)
        det.model = mock_model

        waveform = np.random.randn(16000).astype(np.float32)
        result = det.predict(waveform, utterance_id="test_001")

        assert isinstance(result, DetectorOutput)
        assert result.utterance_id == "test_001"
        assert result.detector_name == "BAM"
        assert result.frame_shift_ms == 160
        assert result.n_frames == n_segments
        assert result.frame_scores.shape == (n_segments,)
        # Scores should be valid probabilities (softmax output)
        assert np.all(result.frame_scores >= 0.0)
        assert np.all(result.frame_scores <= 1.0)
        assert result.utterance_score == pytest.approx(
            float(np.max(result.frame_scores))
        )

    def test_predict_empty_output(self):
        """Test predict handles zero-segment output gracefully."""
        det = BAMDetector(device="cpu")
        mock_output = torch.zeros(1, 0, 2)
        mock_b_pred = torch.zeros(1, 0)
        mock_model = MagicMock()
        mock_model.return_value = (mock_output, mock_b_pred)
        det.model = mock_model

        result = det.predict(np.zeros(160), utterance_id="empty")
        assert result.n_frames == 0
        assert result.utterance_score == 0.0

    def test_load_checkpoint_missing_file(self, tmp_path):
        """Test that load_model raises when checkpoint file is missing."""
        ext_dir = tmp_path / "BAM"
        ext_dir.mkdir()
        det = BAMDetector(
            checkpoint="/nonexistent/model.ckpt",
            external_dir=str(ext_dir),
            device="cpu",
        )
        with patch.dict("sys.modules", {"models.bam": MagicMock()}):
            with patch("sys.path"):
                with pytest.raises(FileNotFoundError, match="checkpoint"):
                    det.load_model()


class TestSALDetector:
    def test_instantiation(self):
        det = SALDetector(device="cpu")
        assert det.name == "SAL"
        assert det.frame_shift_ms == 160

    def test_instantiation_custom_params(self):
        det = SALDetector(
            device="cpu",
            external_dir="/tmp/sal",
            ssl_ckpt="/tmp/xlsr.pt",
            resolution=0.32,
        )
        assert det.external_dir.name == "sal"
        assert det.ssl_ckpt == "/tmp/xlsr.pt"
        assert det.resolution == 0.32

    def test_predict_shape(self):
        det = SALDetector(device="cpu")
        assert hasattr(det, "predict")
        assert hasattr(det, "load_model")

    def test_load_model_requires_external_dir(self):
        det = SALDetector(device="cpu")
        with pytest.raises(ValueError, match="external_dir"):
            det.load_model()

    def test_load_model_requires_existing_dir(self):
        det = SALDetector(external_dir="/nonexistent/path", device="cpu")
        with pytest.raises(FileNotFoundError, match="not found"):
            det.load_model()

    def test_predict_requires_loaded_model(self):
        det = SALDetector(device="cpu")
        with pytest.raises(RuntimeError, match="load_model"):
            det.predict(np.zeros(16000))

    def test_predict_with_mock_model(self):
        """Test predict with a mock SAL model returning (out1, out2)."""
        det = SALDetector(device="cpu")
        n_segments = 10
        # SAL returns (batch, n_segments, 8) position logits and
        # (batch, n_segments, 2) binary logits
        mock_out1 = torch.randn(1, n_segments, 8)
        mock_out2 = torch.randn(1, n_segments, 2)
        mock_model = MagicMock()
        mock_model.return_value = (mock_out1, mock_out2)
        det.model = mock_model

        waveform = np.random.randn(16000).astype(np.float32)
        result = det.predict(waveform, utterance_id="test_sal_001")

        assert isinstance(result, DetectorOutput)
        assert result.utterance_id == "test_sal_001"
        assert result.detector_name == "SAL"
        assert result.frame_shift_ms == 160
        assert result.n_frames == n_segments
        assert result.frame_scores.shape == (n_segments,)
        # Scores should be valid probabilities (softmax output)
        assert np.all(result.frame_scores >= 0.0)
        assert np.all(result.frame_scores <= 1.0)
        assert result.utterance_score == pytest.approx(
            float(np.max(result.frame_scores))
        )

    def test_predict_empty_output(self):
        """Test predict handles zero-segment output gracefully."""
        det = SALDetector(device="cpu")
        mock_out1 = torch.zeros(1, 0, 8)
        mock_out2 = torch.zeros(1, 0, 2)
        mock_model = MagicMock()
        mock_model.return_value = (mock_out1, mock_out2)
        det.model = mock_model

        result = det.predict(np.zeros(160), utterance_id="empty")
        assert result.n_frames == 0
        assert result.utterance_score == 0.0

    def test_load_checkpoint_missing_file(self, tmp_path):
        """Test that load_model raises when checkpoint file is missing."""
        ext_dir = tmp_path / "SAL"
        ext_dir.mkdir()
        det = SALDetector(
            checkpoint="/nonexistent/model.ckpt",
            external_dir=str(ext_dir),
            device="cpu",
        )
        with patch.dict("sys.modules", {"src.models.net.model": MagicMock()}):
            with patch("sys.path"):
                with pytest.raises(FileNotFoundError, match="checkpoint"):
                    det.load_model()


class TestCFPRFDetector:
    def test_instantiation(self):
        det = CFPRFDetector(device="cpu")
        assert det.name == "CFPRF"
        assert det.frame_shift_ms == 20

    def test_instantiation_custom_params(self):
        det = CFPRFDetector(
            device="cpu",
            external_dir="/tmp/cfprf",
            ssl_path="/tmp/ssl_dir",
            seq_len=500,
            gmlp_layers=2,
        )
        assert det.external_dir.name == "cfprf"
        assert det.ssl_path == "/tmp/ssl_dir"
        assert det.seq_len == 500
        assert det.gmlp_layers == 2

    def test_load_model_requires_external_dir(self):
        det = CFPRFDetector(device="cpu")
        with pytest.raises(ValueError, match="external_dir"):
            det.load_model()

    def test_load_model_requires_existing_dir(self):
        det = CFPRFDetector(external_dir="/nonexistent/path", device="cpu")
        with pytest.raises(FileNotFoundError, match="not found"):
            det.load_model()

    def test_predict_requires_loaded_model(self):
        det = CFPRFDetector(device="cpu")
        with pytest.raises(RuntimeError, match="load_model"):
            det.predict(np.zeros(16000))

    def test_predict_with_mock_model(self):
        """Test predict with a mock FDN model returning (seg_scores, bd_scores, emb_T, F_BA)."""
        det = CFPRFDetector(device="cpu")
        n_frames = 50
        # FDN returns (batch, n_frames, 2) seg logits, (batch, n_frames, 2) boundary,
        # (batch, n_frames, 128) emb_T, (batch, n_frames, 128) F_BA
        mock_seg_scores = torch.randn(1, n_frames, 2)
        mock_bd_scores = torch.randn(1, n_frames, 2)
        mock_emb_T = torch.randn(1, n_frames, 128)
        mock_F_BA = torch.randn(1, n_frames, 128)
        mock_model = MagicMock()
        mock_model.return_value = (mock_seg_scores, mock_bd_scores, mock_emb_T, mock_F_BA)
        det.model = mock_model

        waveform = np.random.randn(16000).astype(np.float32)
        result = det.predict(waveform, utterance_id="test_cfprf_001")

        assert isinstance(result, DetectorOutput)
        assert result.utterance_id == "test_cfprf_001"
        assert result.detector_name == "CFPRF"
        assert result.frame_shift_ms == 20
        assert result.n_frames == n_frames
        assert result.frame_scores.shape == (n_frames,)
        # Scores should be valid probabilities (softmax output)
        assert np.all(result.frame_scores >= 0.0)
        assert np.all(result.frame_scores <= 1.0)
        assert result.utterance_score == pytest.approx(
            float(np.max(result.frame_scores))
        )

    def test_predict_empty_output(self):
        """Test predict handles zero-frame output gracefully."""
        det = CFPRFDetector(device="cpu")
        mock_seg_scores = torch.zeros(1, 0, 2)
        mock_bd_scores = torch.zeros(1, 0, 2)
        mock_emb_T = torch.zeros(1, 0, 128)
        mock_F_BA = torch.zeros(1, 0, 128)
        mock_model = MagicMock()
        mock_model.return_value = (mock_seg_scores, mock_bd_scores, mock_emb_T, mock_F_BA)
        det.model = mock_model

        result = det.predict(np.zeros(160), utterance_id="empty")
        assert result.n_frames == 0
        assert result.utterance_score == 0.0

    def test_load_checkpoint_missing_file(self, tmp_path):
        """Test that load_model raises when checkpoint file is missing."""
        ext_dir = tmp_path / "CFPRF"
        ext_dir.mkdir()
        det = CFPRFDetector(
            checkpoint="/nonexistent/model.pth",
            external_dir=str(ext_dir),
            device="cpu",
        )
        with patch.dict("sys.modules", {"models.FDN": MagicMock()}):
            with patch("sys.path"):
                with pytest.raises(FileNotFoundError, match="checkpoint"):
                    det.load_model()


class TestMRMDetector:
    def test_instantiation(self):
        det = MRMDetector(device="cpu")
        assert det.name == "MRM"
        assert det.frame_shift_ms == 20

    def test_instantiation_custom_params(self):
        det = MRMDetector(
            device="cpu",
            external_dir="/tmp/mrm",
            ssl_path="/tmp/xlsr.pt",
            num_scales=3,
            include_utt=False,
            use_mask=False,
            max_seq_len=1000,
        )
        assert det.external_dir.name == "mrm"
        assert det.ssl_path == "/tmp/xlsr.pt"
        assert det.num_scales == 3
        assert det.include_utt is False
        assert det.use_mask is False
        assert det.max_seq_len == 1000

    def test_load_model_requires_external_dir(self):
        det = MRMDetector(device="cpu")
        with pytest.raises(ValueError, match="external_dir"):
            det.load_model()

    def test_load_model_requires_existing_dir(self):
        det = MRMDetector(external_dir="/nonexistent/path", device="cpu")
        with pytest.raises(FileNotFoundError, match="not found"):
            det.load_model()

    def test_predict_requires_loaded_model(self):
        det = MRMDetector(device="cpu")
        with pytest.raises(RuntimeError, match="load_model"):
            det.predict(np.zeros(16000))

    def test_predict_with_mock_model(self):
        """Test predict with a mock MRM model returning (logits, masks).

        MRM forward returns (logits, masks) where:
        - logits[0]: (batch*n_segments, 2) cosine similarities at scale 0
        - logits[-1]: (batch, 2) utterance-level (if include_utt=True)
        - masks: list of boolean tensors
        """
        det = MRMDetector(device="cpu", num_scales=1, include_utt=True)
        n_segments = 50

        # Scale 0 logits: (batch*n_segments, 2) cosine similarity in [-1, 1]
        scale0_logits = torch.rand(n_segments, 2) * 2 - 1  # [-1, 1]
        # Utterance logits: (batch, 2) cosine similarity in [-1, 1]
        utt_logits = torch.rand(1, 2) * 2 - 1

        mock_logits = [scale0_logits, utt_logits]
        mock_masks = []  # no mask when use_mask=False

        mock_model = MagicMock()
        mock_model.return_value = (mock_logits, mock_masks)
        det.model = mock_model
        det.use_mask = False

        waveform = np.random.randn(16000).astype(np.float32)
        result = det.predict(waveform, utterance_id="test_mrm_001")

        assert isinstance(result, DetectorOutput)
        assert result.utterance_id == "test_mrm_001"
        assert result.detector_name == "MRM"
        assert result.frame_shift_ms == 20
        assert result.n_frames == n_segments
        assert result.frame_scores.shape == (n_segments,)
        # Scores mapped from [-1,1] to [0,1]
        assert np.all(result.frame_scores >= 0.0)
        assert np.all(result.frame_scores <= 1.0)
        # Utterance score also in [0, 1]
        assert 0.0 <= result.utterance_score <= 1.0

    def test_predict_with_mask(self):
        """Test predict applies mask correctly to filter valid segments."""
        det = MRMDetector(device="cpu", num_scales=1, include_utt=False)
        n_total = 50
        n_valid = 40

        # Scale 0 logits: (n_total, 2)
        scale0_logits = torch.rand(n_total, 2) * 2 - 1
        # Mask: first n_valid are True
        mask = torch.zeros(n_total, dtype=torch.bool)
        mask[:n_valid] = True

        mock_logits = [scale0_logits]
        mock_masks = [mask]

        mock_model = MagicMock()
        mock_model.return_value = (mock_logits, mock_masks)
        det.model = mock_model
        det.use_mask = True

        waveform = np.random.randn(16000).astype(np.float32)
        result = det.predict(waveform, utterance_id="test_mask")

        assert result.n_frames == n_valid
        assert np.all(result.frame_scores >= 0.0)
        assert np.all(result.frame_scores <= 1.0)

    def test_predict_empty_output(self):
        """Test predict handles zero-segment output gracefully."""
        det = MRMDetector(device="cpu", num_scales=1, include_utt=False)

        scale0_logits = torch.zeros(0, 2)
        mock_logits = [scale0_logits]
        mock_masks = []

        mock_model = MagicMock()
        mock_model.return_value = (mock_logits, mock_masks)
        det.model = mock_model
        det.use_mask = False

        result = det.predict(np.zeros(160), utterance_id="empty")
        assert result.n_frames == 0
        assert result.utterance_score == 0.0

    def test_predict_input_shape(self):
        """Test that predict adds channel dim (batch=1, channels=1, samples)."""
        det = MRMDetector(device="cpu", num_scales=1, include_utt=False)

        scale0_logits = torch.rand(10, 2) * 2 - 1
        mock_logits = [scale0_logits]
        mock_masks = []

        mock_model = MagicMock()
        mock_model.return_value = (mock_logits, mock_masks)
        det.model = mock_model
        det.use_mask = False

        waveform = np.random.randn(16000).astype(np.float32)
        det.predict(waveform, utterance_id="shape_test")

        # Check the model was called with (1, 1, 16000) tensor
        call_args = mock_model.call_args[0][0]
        assert call_args.shape == (1, 1, 16000)

    def test_predict_extracts_spoof_column(self):
        """Regression test: P2SActivation column 1 is spoof, column 0 is bonafide.

        P2SGrad one-hot encoding assigns target=0 for bonafide (column 0) and
        target=1 for spoof (column 1). The wrapper must extract column 1 for
        spoof scores, not column 0.
        """
        det = MRMDetector(device="cpu", num_scales=1, include_utt=True)

        n_segments = 5
        # Set DISTINCT values per column so we can tell which was extracted.
        # Column 0 (bonafide) = -0.8, Column 1 (spoof) = +0.6
        scale0_logits = torch.full((n_segments, 2), -0.8)
        scale0_logits[:, 1] = 0.6  # spoof column

        # Utterance-level: column 0 (bonafide) = -0.4, column 1 (spoof) = +0.8
        utt_logits = torch.tensor([[-0.4, 0.8]])

        mock_logits = [scale0_logits, utt_logits]
        mock_masks = []

        mock_model = MagicMock()
        mock_model.return_value = (mock_logits, mock_masks)
        det.model = mock_model
        det.use_mask = False

        waveform = np.random.randn(16000).astype(np.float32)
        result = det.predict(waveform, utterance_id="col_test")

        # Expected frame scores: (0.6 + 1) / 2 = 0.8
        expected_frame = (0.6 + 1.0) / 2.0  # 0.8
        np.testing.assert_allclose(result.frame_scores, expected_frame, atol=1e-6)

        # Expected utterance score: (0.8 + 1) / 2 = 0.9
        expected_utt = (0.8 + 1.0) / 2.0  # 0.9
        assert result.utterance_score == pytest.approx(expected_utt, abs=1e-6)

        # Verify it did NOT use bonafide column values
        wrong_frame = (-0.8 + 1.0) / 2.0  # 0.1
        assert not np.allclose(result.frame_scores, wrong_frame)

    def test_load_checkpoint_missing_file(self, tmp_path):
        """Test that load_model raises when checkpoint file is missing."""
        ext_dir = tmp_path / "MRM"
        ext_dir.mkdir()
        det = MRMDetector(
            checkpoint="/nonexistent/model.pt",
            external_dir=str(ext_dir),
            device="cpu",
        )
        with patch.dict("sys.modules", {"modules.multiresomodel": MagicMock()}):
            with patch("sys.path"):
                with pytest.raises(FileNotFoundError, match="checkpoint"):
                    det.load_model()
