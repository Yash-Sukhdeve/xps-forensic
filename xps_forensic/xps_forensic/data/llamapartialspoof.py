"""LlamaPartialSpoof dataset loader.

Actual directory layout (v1.0.b, Zenodo DOI: 10.5281/zenodo.14214149):
    root/
    ├── R01TTS.0.a/                    # bonafide + fully fake + partial (crossfade)
    │   └── <utt_id>.wav
    ├── R01TTS.0.b/                    # partial (cut/paste, overlap/add)
    │   └── <utt_id>.wav
    ├── label_R01TTS.0.a.txt           # labels for part a
    ├── label_R01TTS.0.b.txt           # labels for part b
    └── metadata_crossfade.csv         # crossfade function metadata

Label file format (one line per utterance):
    <utt_id> <duration_sec> <utterance_label> <seg1> <seg2> ... <segN>
    where each segment = <start_sec>-<end_sec>-<bonafide|spoof>

Reference:
    Luong et al., "LlamaPartialSpoof: An LLM-based Partial Spoof Speech
    Dataset", ICASSP 2025.
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

# Dataset parts: (audio_dir, label_file)
_PARTS = [
    ("R01TTS.0.a", "label_R01TTS.0.a.txt"),
    ("R01TTS.0.b", "label_R01TTS.0.b.txt"),
]


class LlamaPartialSpoofDataset(BasePartialSpoofDataset):
    """Loader for the LlamaPartialSpoof corpus (v1.0.b).

    Parses label files with segment-level timestamps and converts to
    frame-level binary labels at 10 ms resolution.
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "eval",
        sample_rate: int = 16000,
        parts: list[str] | None = None,
    ):
        """
        Args:
            root: Path to LlamaPartialSpoof directory.
            split: Unused (dataset has no predefined splits), kept for API
                compatibility with BasePartialSpoofDataset.
            sample_rate: Target sample rate.
            parts: Which parts to load, e.g. ["R01TTS.0.a"]. Default: all.
        """
        self._parts_filter = parts
        self._crossfade_meta: dict[str, str] = {}
        super().__init__(root=root, split=split, sample_rate=sample_rate)

    # ------------------------------------------------------------------
    # Manifest
    # ------------------------------------------------------------------

    def _load_manifest(self) -> list[dict]:
        """Parse label files and build manifest entries."""
        self._crossfade_meta = self._load_crossfade_metadata()

        manifest: list[dict] = []
        for audio_dir, label_file in _PARTS:
            if self._parts_filter and audio_dir not in self._parts_filter:
                continue

            label_path = self.root / label_file
            if not label_path.exists():
                logger.warning("Label file not found: %s", label_path)
                continue

            audio_root = self.root / audio_dir
            if not audio_root.exists():
                logger.warning("Audio directory not found: %s", audio_root)
                continue

            with open(label_path, "r") as fh:
                for line in fh:
                    entry = self._parse_label_line(line, audio_root)
                    if entry is not None:
                        manifest.append(entry)

        logger.info(
            "LlamaPartialSpoof: loaded %d utterances from %s",
            len(manifest),
            self.root,
        )
        return manifest

    # ------------------------------------------------------------------
    # Sample loading
    # ------------------------------------------------------------------

    def _load_sample(self, entry: dict) -> AudioSegmentSample:
        """Load waveform and compute frame-level labels from segments."""
        wav_path = Path(entry["wav_path"])

        waveform, sr = sf.read(wav_path, dtype="float32")
        if sr != self.sample_rate:
            waveform = self._resample(waveform, sr, self.sample_rate)

        frame_labels = self._segments_to_frame_labels(
            entry["segments"],
            entry["duration_sec"],
            len(waveform),
            self.sample_rate,
        )

        metadata = {"part": entry["part"]}
        if entry["utterance_id"] in self._crossfade_meta:
            metadata["crossfade_fn"] = self._crossfade_meta[entry["utterance_id"]]

        return AudioSegmentSample(
            utterance_id=entry["utterance_id"],
            waveform=waveform,
            sample_rate=self.sample_rate,
            utterance_label=entry["utterance_label"],
            frame_labels=frame_labels,
            dataset="llamapartialspoof",
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    # Parsing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_label_line(line: str, audio_root: Path) -> dict | None:
        """Parse one label line into a manifest entry.

        Format: <utt_id> <duration> <utt_label> <seg1> <seg2> ... <segN>
        Segment: <start>-<end>-<bonafide|spoof>

        Returns None if the line is malformed.
        """
        line = line.strip()
        if not line:
            return None

        parts = line.split()
        if len(parts) < 4:
            return None

        utt_id = parts[0]
        try:
            duration_sec = float(parts[1])
        except ValueError:
            return None

        utt_label_raw = parts[2]  # "bonafide" or "spoof"
        segment_strs = parts[3:]

        segments = []
        has_spoof = False
        has_bonafide = False
        for seg_str in segment_strs:
            seg_parts = seg_str.rsplit("-", 1)
            if len(seg_parts) != 2:
                continue
            time_part, label = seg_parts[0], seg_parts[1]
            time_fields = time_part.split("-")
            if len(time_fields) != 2:
                continue
            try:
                start = float(time_fields[0])
                end = float(time_fields[1])
            except ValueError:
                continue
            is_spoof = label == "spoof"
            segments.append((start, end, is_spoof))
            if is_spoof:
                has_spoof = True
            else:
                has_bonafide = True

        # Ternary utterance label: 0=real, 1=partial, 2=fully_fake
        if utt_label_raw == "bonafide":
            utterance_label = 0
        elif has_spoof and has_bonafide:
            utterance_label = 1  # partially fake
        elif has_spoof:
            utterance_label = 2  # fully fake
        else:
            utterance_label = 0  # all bonafide segments

        wav_path = audio_root / f"{utt_id}.wav"

        return {
            "utterance_id": utt_id,
            "wav_path": str(wav_path),
            "duration_sec": duration_sec,
            "utterance_label": utterance_label,
            "segments": segments,
            "part": audio_root.name,
        }

    @staticmethod
    def _segments_to_frame_labels(
        segments: list[tuple[float, float, bool]],
        duration_sec: float,
        n_samples: int,
        sample_rate: int,
    ) -> np.ndarray:
        """Convert timestamp segments to binary frame labels at 10 ms shift.

        Args:
            segments: List of (start_sec, end_sec, is_spoof).
            duration_sec: Total utterance duration from label file.
            n_samples: Number of audio samples (used for frame count).
            sample_rate: Audio sample rate.

        Returns:
            Binary array of shape (n_frames,), 1 = spoof.
        """
        frame_shift_s = FRAME_SHIFT_MS / 1000.0
        n_frames = int(np.ceil(n_samples / (sample_rate * frame_shift_s)))
        labels = np.zeros(n_frames, dtype=np.int32)

        for start, end, is_spoof in segments:
            if not is_spoof:
                continue
            frame_start = int(start / frame_shift_s)
            frame_end = int(np.ceil(end / frame_shift_s))
            frame_end = min(frame_end, n_frames)
            if frame_start < frame_end:
                labels[frame_start:frame_end] = 1

        return labels

    def _load_crossfade_metadata(self) -> dict[str, str]:
        """Load crossfade function metadata if available.

        Returns mapping from partial-cf utterance ID to function code
        (t=triangle, q=quarter sine, h=half sine, l=logarithmic,
        p=inverted parabola).
        """
        csv_path = self.root / "metadata_crossfade.csv"
        if not csv_path.exists():
            return {}

        meta: dict[str, str] = {}
        with open(csv_path, "r") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                meta[row["id"]] = row["function"]
        return meta

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
