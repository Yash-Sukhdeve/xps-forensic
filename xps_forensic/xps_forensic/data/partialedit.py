"""PartialEdit dataset loader.

Supports two layouts:

1. **CSV layout** (actual PartialEdit distribution):
    root/
    ├── PartialEdit_E1E2.csv    # "rel_path,edit_start,edit_end,duration"
    ├── E1/p225/*.wav           # edited waveforms (subset E1)
    └── E2/p225/*.wav           # edited waveforms (subset E2)

2. **JSON layout** (legacy/synthetic):
    root/
    ├── metadata.json           # [{"id", "filename", "edit_regions": [...]}]
    └── audio/*.wav

Reference: Zhang et al., "PartialEdit: A Dataset for Neural Speech Editing
Evaluation", 2025.
"""
from __future__ import annotations

import csv
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
    """Loader for the PartialEdit corpus.

    Auto-detects between CSV layout (PartialEdit_E1E2.csv) and
    JSON layout (metadata.json).
    """

    # ------------------------------------------------------------------
    # Manifest
    # ------------------------------------------------------------------

    def _load_manifest(self) -> list[dict]:
        """Load manifest from CSV or JSON, whichever exists."""
        csv_path = self.root / "PartialEdit_E1E2.csv"
        json_path = self.root / "metadata.json"

        if csv_path.exists():
            return self._load_csv_manifest(csv_path)
        elif json_path.exists():
            return self._load_json_manifest(json_path)
        else:
            logger.warning(
                "Neither PartialEdit_E1E2.csv nor metadata.json found in %s",
                self.root,
            )
            return []

    def _load_csv_manifest(self, csv_path: Path) -> list[dict]:
        """Parse PartialEdit_E1E2.csv.

        Format: relative_path, edit_start_sec, edit_end_sec, duration_sec
        Example: E1/p237/p237_321_edited_partial_16k.wav,1.038,1.46,3.24

        Each row represents one edited utterance with a single edit region.
        """
        manifest: list[dict] = []
        n_missing = 0

        with open(csv_path, "r") as fh:
            reader = csv.reader(fh)
            for row in reader:
                if len(row) < 4:
                    continue

                rel_path = row[0].strip()
                edit_start = float(row[1])
                edit_end = float(row[2])
                duration = float(row[3])

                wav_path = self.root / rel_path
                if not wav_path.exists():
                    n_missing += 1
                    if n_missing <= 5:
                        logger.warning("Missing: %s", wav_path)
                    continue

                # Derive utterance ID from filename
                utt_id = Path(rel_path).stem

                manifest.append(
                    {
                        "utterance_id": utt_id,
                        "wav_path": str(wav_path),
                        "edit_regions": [
                            {"start_sec": edit_start, "end_sec": edit_end}
                        ],
                        "utterance_label_raw": 1,  # all entries are edited
                    }
                )

        if n_missing > 0:
            logger.warning(
                "PartialEdit: %d/%d files missing on disk",
                n_missing,
                n_missing + len(manifest),
            )

        logger.info(
            "PartialEdit [CSV]: loaded %d utterances from %s",
            len(manifest),
            csv_path.name,
        )
        return manifest

    def _load_json_manifest(self, json_path: Path) -> list[dict]:
        """Parse legacy metadata.json format."""
        with open(json_path, "r") as fh:
            raw_entries = json.load(fh)

        manifest: list[dict] = []
        for entry in raw_entries:
            wav_path = self.root / "audio" / entry["filename"]
            edit_regions = entry.get("edit_regions", [])
            utterance_label_raw = 1 if edit_regions else 0
            manifest.append(
                {
                    "utterance_id": entry["id"],
                    "wav_path": str(wav_path),
                    "edit_regions": edit_regions,
                    "utterance_label_raw": utterance_label_raw,
                }
            )

        logger.info(
            "PartialEdit [JSON]: loaded %d utterances", len(manifest)
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
