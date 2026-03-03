"""PartialEdit dataset loader.

Expected directory layout:
    root/
    ├── audio/            # edited waveforms
    ├── original/         # (optional) original waveforms
    └── metadata.json     # list of entries with edit regions

metadata.json schema (each entry):
    {
        "id": "PE_00001",
        "filename": "PE_00001.wav",
        "edit_regions": [
            {"start_sec": 1.2, "end_sec": 2.5},
            ...
        ]
    }
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import soundfile as sf

from xps_forensic.data.base import AudioSegmentSample, BasePartialSpoofDataset

logger = logging.getLogger(__name__)

# Frame resolution (seconds per frame).
FRAME_SHIFT_MS: int = 10


class PartialEditDataset(BasePartialSpoofDataset):
    """Loader for the PartialEdit corpus."""

    # ------------------------------------------------------------------
    # Manifest
    # ------------------------------------------------------------------

    def _load_manifest(self) -> list[dict]:
        """Parse ``metadata.json`` and build manifest entries."""
        meta_path = self.root / "metadata.json"
        if not meta_path.exists():
            logger.warning("metadata.json not found: %s", meta_path)
            return []

        with open(meta_path, "r") as fh:
            raw_entries = json.load(fh)

        manifest: list[dict] = []
        for entry in raw_entries:
            wav_path = self.root / "audio" / entry["filename"]
            edit_regions = entry.get("edit_regions", [])
            manifest.append(
                {
                    "utterance_id": entry["id"],
                    "wav_path": str(wav_path),
                    "edit_regions": edit_regions,
                }
            )
        return manifest

    # ------------------------------------------------------------------
    # Sample loading
    # ------------------------------------------------------------------

    def _load_sample(self, entry: dict) -> AudioSegmentSample:
        """Load waveform and derive frame labels from edit regions."""
        wav_path = Path(entry["wav_path"])
        waveform, sr = sf.read(wav_path, dtype="float32")

        if sr != self.sample_rate:
            waveform = self._resample(waveform, sr, self.sample_rate)
            sr = self.sample_rate

        frame_labels = self._regions_to_frame_labels(
            entry["edit_regions"], waveform, sr
        )

        # Determine utterance label
        fake_ratio = frame_labels.sum() / max(len(frame_labels), 1)
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
            dataset="partialedit",
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _regions_to_frame_labels(
        edit_regions: list[dict],
        waveform: np.ndarray,
        sample_rate: int,
    ) -> np.ndarray:
        """Convert edit regions (start_sec, end_sec) to binary frame labels.

        Uses 10 ms frames: frame *i* covers [i*0.01, (i+1)*0.01) seconds.
        A frame is marked fake (1) if it overlaps any edit region.
        """
        frame_shift_sec = FRAME_SHIFT_MS / 1000.0
        duration_sec = len(waveform) / sample_rate
        n_frames = int(np.ceil(duration_sec / frame_shift_sec))
        labels = np.zeros(n_frames, dtype=np.int32)

        for region in edit_regions:
            start_frame = int(region["start_sec"] / frame_shift_sec)
            end_frame = int(np.ceil(region["end_sec"] / frame_shift_sec))
            start_frame = max(0, start_frame)
            end_frame = min(n_frames, end_frame)
            labels[start_frame:end_frame] = 1

        return labels

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
