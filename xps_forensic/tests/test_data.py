"""Tests for data loading."""
import numpy as np
import pytest
import soundfile as sf

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
    @staticmethod
    def _create_dataset(tmp_path, split="eval"):
        """Create a minimal PartialSpoof directory structure for testing.

        Mimics the actual layout under database/ with:
        - .lst file for utterance enumeration
        - Protocol file for utterance-level labels
        - .npy segment labels with correct polarity ('0'=spoof, '1'=bona)
        - Dummy wav files
        """
        db = tmp_path / "database"
        split_dir = db / split
        wav_dir = split_dir / "con_wav"
        wav_dir.mkdir(parents=True)

        proto_dir = db / "protocols" / "PartialSpoof_LA_cm_protocols"
        proto_dir.mkdir(parents=True)

        seg_dir = db / "segment_labels"
        seg_dir.mkdir(parents=True)

        # Utterances: 1 bonafide, 1 partial, 1 fully fake
        utt_ids = ["LA_E_0000001", "CON_E_0000001", "CON_E_0000002"]

        # .lst file
        lst_path = split_dir / f"{split}.lst"
        lst_path.write_text("\n".join(utt_ids) + "\n")

        # Protocol file
        proto_path = proto_dir / f"PartialSpoof.LA.cm.{split}.trl.txt"
        proto_path.write_text(
            "LA_0001 LA_E_0000001 - - bonafide\n"
            "LA_0001 CON_E_0000001 - CON spoof\n"
            "LA_0001 CON_E_0000002 - CON spoof\n"
        )

        # Segment labels (.npy) — dataset polarity: '0'=spoof, '1'=bonafide
        seg_labels = {
            "LA_E_0000001": np.array(["1", "1", "1", "1", "1"]),       # all bona
            "CON_E_0000001": np.array(["1", "1", "0", "0", "1"]),      # partial
            "CON_E_0000002": np.array(["0", "0", "0", "0", "0"]),      # all spoof
        }
        npy_path = seg_dir / f"{split}_seglab_0.01.npy"
        np.save(npy_path, seg_labels)

        # Dummy wav files (0.5s each = 5 frames at 10ms)
        for utt_id in utt_ids:
            sf.write(str(wav_dir / f"{utt_id}.wav"), np.zeros(8000), 16000)

        return tmp_path

    def test_init_missing(self, tmp_path):
        ds = PartialSpoofDataset(root=tmp_path / "nonexistent")
        assert len(ds) == 0

    def test_manifest_structure(self, tmp_path):
        root = self._create_dataset(tmp_path)
        ds = PartialSpoofDataset(root=root, split="eval", seg_resolution="0.01")
        assert len(ds.manifest) == 3
        # Check ternary labels
        assert ds.manifest[0]["utterance_label"] == 0  # bonafide
        assert ds.manifest[1]["utterance_label"] == 1  # partial
        assert ds.manifest[2]["utterance_label"] == 2  # fully fake

    def test_label_polarity_flip(self, tmp_path):
        """Verify dataset '0'(spoof) is flipped to internal 1(fake)."""
        root = self._create_dataset(tmp_path)
        ds = PartialSpoofDataset(root=root, split="eval", seg_resolution="0.01")

        # Partial spoof sample: dataset labels '1','1','0','0','1'
        # After flip: internal labels 0,0,1,1,0
        sample = ds[1]  # CON_E_0000001
        assert sample.utterance_label == 1  # partial
        np.testing.assert_array_equal(
            sample.frame_labels[:5], [0, 0, 1, 1, 0]
        )

    def test_bonafide_all_zeros(self, tmp_path):
        """Bonafide utterance should have all-zero frame labels."""
        root = self._create_dataset(tmp_path)
        ds = PartialSpoofDataset(root=root, split="eval", seg_resolution="0.01")
        sample = ds[0]  # LA_E_0000001
        assert sample.utterance_label == 0
        assert sample.frame_labels.sum() == 0

    def test_fully_fake_all_ones(self, tmp_path):
        """Fully fake utterance should have all-one frame labels."""
        root = self._create_dataset(tmp_path)
        ds = PartialSpoofDataset(root=root, split="eval", seg_resolution="0.01")
        sample = ds[2]  # CON_E_0000002
        assert sample.utterance_label == 2
        assert sample.frame_labels[:5].sum() == 5

    def test_protocol_parsing(self, tmp_path):
        """Verify protocol file parsing extracts correct labels."""
        root = self._create_dataset(tmp_path)
        labels = PartialSpoofDataset._load_protocol(
            root / "database", "eval"
        )
        assert labels["LA_E_0000001"] == "bonafide"
        assert labels["CON_E_0000001"] == "spoof"
        assert labels["CON_E_0000002"] == "spoof"


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

    def test_parse_label_line_bonafide(self, tmp_path):
        line = "dev-clean_1462_170145_000020_000002 0.9600 bonafide 0.0000-0.9600-bonafide"
        entry = LlamaPartialSpoofDataset._parse_label_line(line, tmp_path)
        assert entry is not None
        assert entry["utterance_id"] == "dev-clean_1462_170145_000020_000002"
        assert entry["duration_sec"] == pytest.approx(0.96)
        assert entry["utterance_label"] == 0  # real
        assert len(entry["segments"]) == 1
        assert entry["segments"][0][2] is False  # bonafide

    def test_parse_label_line_fully_fake(self, tmp_path):
        line = "dev01-cosyvoice-full_1462_170145_000020_000002 1.4629 spoof 0.0000-1.4629-spoof"
        entry = LlamaPartialSpoofDataset._parse_label_line(line, tmp_path)
        assert entry is not None
        assert entry["utterance_label"] == 2  # fully fake

    def test_parse_label_line_partial(self, tmp_path):
        line = (
            "dev01-cosyvoice-partial-cf_1272_128104_000007_000001 3.9090 spoof "
            "0.0000-0.0970-bonafide 0.0970-2.7070-spoof 2.7070-3.9090-bonafide"
        )
        entry = LlamaPartialSpoofDataset._parse_label_line(line, tmp_path)
        assert entry is not None
        assert entry["utterance_label"] == 1  # partial
        assert len(entry["segments"]) == 3

    def test_segments_to_frame_labels(self):
        # 1 second audio at 16 kHz, 10 ms frames = 100 frames
        segments = [(0.2, 0.5, True), (0.7, 0.9, True)]
        labels = LlamaPartialSpoofDataset._segments_to_frame_labels(
            segments, duration_sec=1.0, n_samples=16000, sample_rate=16000
        )
        assert labels.shape == (100,)
        assert labels[20:50].sum() == 30  # 0.2s-0.5s all spoof
        assert labels[70:90].sum() == 20  # 0.7s-0.9s all spoof
        assert labels[50:70].sum() == 0   # 0.5s-0.7s all bonafide
        assert labels[0:20].sum() == 0    # 0.0s-0.2s all bonafide

    def test_manifest_from_files(self, tmp_path):
        """Test manifest loading from actual file structure."""
        audio_dir = tmp_path / "R01TTS.0.a"
        audio_dir.mkdir()
        label_file = tmp_path / "label_R01TTS.0.a.txt"
        label_file.write_text(
            "utt_001 1.0000 bonafide 0.0000-1.0000-bonafide\n"
            "utt_002 2.0000 spoof 0.0000-1.0000-bonafide 1.0000-2.0000-spoof\n"
            "utt_003 1.5000 spoof 0.0000-1.5000-spoof\n"
        )
        # Create dummy wav files matching declared durations
        durations = {"utt_001": 1.0, "utt_002": 2.0, "utt_003": 1.5}
        for utt_id, dur in durations.items():
            n = int(dur * 16000)
            sf.write(str(audio_dir / f"{utt_id}.wav"), np.zeros(n), 16000)

        ds = LlamaPartialSpoofDataset(root=tmp_path, parts=["R01TTS.0.a"])
        assert len(ds) == 3
        assert ds.manifest[0]["utterance_label"] == 0  # bonafide
        assert ds.manifest[1]["utterance_label"] == 1  # partial
        assert ds.manifest[2]["utterance_label"] == 2  # fully fake

        # Load a sample and check frame labels
        sample = ds[1]  # partial spoof
        assert sample.utterance_id == "utt_002"
        assert sample.utterance_label == 1
        # Second half should be spoof
        mid = len(sample.frame_labels) // 2
        assert sample.frame_labels[:mid].sum() == 0
        assert sample.frame_labels[mid:].sum() > 0


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
