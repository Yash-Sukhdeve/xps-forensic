"""Tests for data loading."""
import numpy as np
import pytest

from xps_forensic.data.base import AudioSegmentSample, BasePartialSpoofDataset
from xps_forensic.data.partialspoof import PartialSpoofDataset
from xps_forensic.data.partialedit import PartialEditDataset
from xps_forensic.data.hqmpsd import HQMPSDDataset
from xps_forensic.data.llamapartialspoof import LlamaPartialSpoofDataset


class TestAudioSegmentSample:
    def test_sample_fields(self):
        waveform = np.zeros(16000, dtype=np.float32)
        sample = AudioSegmentSample(
            utterance_id="utt_001",
            waveform=waveform,
            sample_rate=16000,
            utterance_label=1,
            frame_labels=np.array([0, 0, 1, 1, 0]),
            dataset="partialspoof",
        )
        assert sample.utterance_id == "utt_001"
        assert sample.duration_sec == pytest.approx(1.0)
        assert sample.is_partially_fake is True

    def test_real_sample(self):
        waveform = np.zeros(32000, dtype=np.float32)
        sample = AudioSegmentSample(
            utterance_id="utt_002",
            waveform=waveform,
            sample_rate=16000,
            utterance_label=0,
            frame_labels=np.zeros(10),
            dataset="partialspoof",
        )
        assert sample.is_partially_fake is False
        assert sample.duration_sec == pytest.approx(2.0)


class TestPartialSpoofDataset:
    def test_manifest_structure(self, tmp_path):
        (tmp_path / "eval" / "con_wav").mkdir(parents=True)
        (tmp_path / "eval" / "label").mkdir(parents=True)
        proto = tmp_path / "eval" / "protocol.txt"
        proto.write_text("CON_E_00001 0\nCON_E_00002 1\n")
        label_file = tmp_path / "eval" / "label" / "CON_E_00002.txt"
        label_file.write_text("0 0 0 1 1 1 0 0\n")
        ds = PartialSpoofDataset(root=tmp_path, split="eval")
        assert len(ds.manifest) == 2
        assert ds.manifest[1]["utterance_label"] == 1


class TestPartialEditDataset:
    def test_init_missing_dir(self, tmp_path):
        ds = PartialEditDataset(root=tmp_path / "nonexistent")
        assert len(ds) == 0


class TestHQMPSD:
    def test_init_missing(self, tmp_path):
        ds = HQMPSDDataset(root=tmp_path / "nonexistent", language="en")
        assert len(ds) == 0


class TestLlamaPartialSpoof:
    def test_init_missing(self, tmp_path):
        ds = LlamaPartialSpoofDataset(root=tmp_path / "nonexistent")
        assert len(ds) == 0


class TestBaseDatasetSplit:
    def test_get_split(self):
        class MockDS(BasePartialSpoofDataset):
            def _load_manifest(self):
                return [{"id": i} for i in range(100)]

            def _load_sample(self, entry):
                pass

        ds = MockDS(root="/tmp", split="eval")
        ds.manifest = [{"id": i} for i in range(100)]
        cal, ver = ds.get_split(ratio=0.8, seed=42)
        assert len(cal) == 80
        assert len(ver) == 20
        assert set(cal + ver) == set(range(100))
