"""LlamaPartialSpoof dataset loader.

Expected directory layout:
    root/
    ├── wav/              # waveform files
    ├── labels/           # per-utterance frame-level label files
    └── protocol.txt      # "utt_id label" per line

Reference:
    LlamaPartialSpoof dataset documentation.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import soundfile as sf

from xps_forensic.data.base import AudioSegmentSample, BasePartialSpoofDataset

logger = logging.getLogger(__name__)

FRAME_SHIFT_MS: int = 10


class LlamaPartialSpoofDataset(BasePartialSpoofDataset):
    """Loader for the LlamaPartialSpoof corpus."""

    # ------------------------------------------------------------------
    # Manifest
    # ------------------------------------------------------------------

    def _load_manifest(self) -> list[dict]:
        """Parse ``protocol.txt`` and build manifest entries."""
        protocol_path = self.root / "protocol.txt"
        if not protocol_path.exists():
            logger.warning("Protocol file not found: %s", protocol_path)
            return []

        manifest: list[dict] = []
        with open(protocol_path, "r") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 2:
                    continue
                utt_id = parts[0]
                utt_label_raw = int(parts[1])

                wav_path = self.root / "wav" / f"{utt_id}.wav"
                label_path = self.root / "labels" / f"{utt_id}.txt"

                utterance_label = self._determine_utterance_label(
                    utt_label_raw, label_path
                )

                manifest.append(
                    {
                        "utterance_id": utt_id,
                        "wav_path": str(wav_path),
                        "label_path": str(label_path),
                        "utterance_label": utterance_label,
                    }
                )
        return manifest

    # ------------------------------------------------------------------
    # Sample loading
    # ------------------------------------------------------------------

    def _load_sample(self, entry: dict) -> AudioSegmentSample:
        """Load waveform and frame labels for one utterance."""
        wav_path = Path(entry["wav_path"])
        label_path = Path(entry["label_path"])

        waveform, sr = sf.read(wav_path, dtype="float32")
        if sr != self.sample_rate:
            waveform = self._resample(waveform, sr, self.sample_rate)

        frame_labels = self._load_frame_labels(label_path, waveform, sr)

        return AudioSegmentSample(
            utterance_id=entry["utterance_id"],
            waveform=waveform,
            sample_rate=self.sample_rate,
            utterance_label=entry["utterance_label"],
            frame_labels=frame_labels,
            dataset="llamapartialspoof",
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_frame_labels(
        label_path: Path,
        waveform: np.ndarray,
        sample_rate: int,
    ) -> np.ndarray:
        """Load frame-level binary labels from text file.

        Returns all-zeros if the label file is missing (bonafide utterance).
        """
        n_frames = int(
            np.ceil(len(waveform) / (sample_rate * FRAME_SHIFT_MS / 1000))
        )
        if not label_path.exists():
            return np.zeros(n_frames, dtype=np.int32)
        return np.loadtxt(label_path, dtype=np.int32).ravel()

    @staticmethod
    def _determine_utterance_label(raw_label: int, label_path: Path) -> int:
        """Map raw protocol label to ternary: 0=real, 1=partial, 2=fully_fake.

        Uses frame-level label file when available:
        - All zeros      → 0 (genuine)
        - fake_ratio > 0.95 → 2 (fully fake)
        - Otherwise       → 1 (partially fake)
        """
        if not label_path.exists():
            return 0 if raw_label == 0 else 2

        labels = np.loadtxt(label_path, dtype=np.int32).ravel()
        if len(labels) == 0:
            return 0 if raw_label == 0 else 2

        fake_ratio = labels.sum() / len(labels)
        if fake_ratio == 0.0:
            return 0
        if fake_ratio > 0.95:
            return 2
        return 1

    @staticmethod
    def _resample(
        waveform: np.ndarray,
        orig_sr: int,
        target_sr: int,
    ) -> np.ndarray:
        """Resample waveform using torchaudio (imported lazily)."""
        import torch
        import torchaudio.functional as F

        wav_t = torch.from_numpy(waveform).unsqueeze(0)
        wav_t = F.resample(wav_t, orig_sr, target_sr)
        return wav_t.squeeze(0).numpy()
