"""HQ-MPSD (High-Quality Multi-lingual Partial Spoof Dataset) loader.

Expected directory layout:
    root/
    ├── en/
    │   ├── audio/         # wav files
    │   └── labels/        # per-utterance frame-level label files
    └── metadata.csv       # id,audio_path,label_path,language,utterance_label

Label convention in source data (ternary per-frame):
    0 = genuine
    1 = deepfake
    2 = transition

Binarization for XPS-Forensic: 0 → 0 (real), {1, 2} → 1 (fake).

Reference:
    HQ-MPSD dataset documentation.
"""
from __future__ import annotations

import csv
import logging
from pathlib import Path

import numpy as np
import soundfile as sf

from xps_forensic.data.base import AudioSegmentSample, BasePartialSpoofDataset

logger = logging.getLogger(__name__)

FRAME_SHIFT_MS: int = 10


class HQMPSDDataset(BasePartialSpoofDataset):
    """Loader for the HQ-MPSD corpus (English subset by default)."""

    def __init__(
        self,
        root: str | Path,
        split: str = "eval",
        sample_rate: int = 16000,
        language: str = "en",
    ):
        self.language = language
        # Call super().__init__ which triggers _load_manifest if root exists
        super().__init__(root=root, split=split, sample_rate=sample_rate)

    # ------------------------------------------------------------------
    # Manifest
    # ------------------------------------------------------------------

    def _load_manifest(self) -> list[dict]:
        """Parse ``metadata.csv`` and filter by language."""
        meta_path = self.root / "metadata.csv"
        if not meta_path.exists():
            logger.warning("metadata.csv not found: %s", meta_path)
            return []

        manifest: list[dict] = []
        with open(meta_path, "r", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                if row.get("language", "") != self.language:
                    continue
                wav_path = self.root / row["audio_path"]
                label_path = self.root / row["label_path"]
                manifest.append(
                    {
                        "utterance_id": row["id"],
                        "wav_path": str(wav_path),
                        "label_path": str(label_path),
                        "utterance_label_raw": row.get("utterance_label", ""),
                    }
                )
        return manifest

    # ------------------------------------------------------------------
    # Sample loading
    # ------------------------------------------------------------------

    def _load_sample(self, entry: dict) -> AudioSegmentSample:
        """Load waveform and binarized frame labels."""
        wav_path = Path(entry["wav_path"])
        label_path = Path(entry["label_path"])

        waveform, sr = sf.read(wav_path, dtype="float32")
        if sr != self.sample_rate:
            waveform = self._resample(waveform, sr, self.sample_rate)

        frame_labels = self._load_and_binarize_labels(label_path, waveform, sr)

        # Determine ternary utterance label from binarized frames
        fake_ratio = float(np.mean(frame_labels)) if len(frame_labels) > 0 else 0.0
        if fake_ratio == 0.0:
            utterance_label = 0
        elif fake_ratio > 0.95:
            utterance_label = 2
        else:
            utterance_label = 1

        return AudioSegmentSample(
            utterance_id=entry["utterance_id"],
            waveform=waveform,
            sample_rate=self.sample_rate,
            utterance_label=utterance_label,
            frame_labels=frame_labels,
            dataset="hqmpsd",
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_and_binarize_labels(
        label_path: Path,
        waveform: np.ndarray,
        sample_rate: int,
    ) -> np.ndarray:
        """Load ternary labels and binarize: 0 → 0, {1,2} → 1.

        Falls back to all-zeros if the label file does not exist.
        """
        n_frames = int(
            np.ceil(len(waveform) / (sample_rate * FRAME_SHIFT_MS / 1000))
        )
        if not label_path.exists():
            return np.zeros(n_frames, dtype=np.int32)

        raw = np.loadtxt(label_path, dtype=np.int32).ravel()
        # Binarize: genuine (0) stays 0; deepfake (1) and transition (2) → 1
        binary = (raw > 0).astype(np.int32)
        return binary

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
