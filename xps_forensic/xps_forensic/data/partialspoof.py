"""PartialSpoof dataset loader (v1.2).

Actual directory layout (Zhang et al., 2022 — arXiv:2204.05177):
    root/
    ├── database/
    │   ├── train/
    │   │   ├── con_wav/        # LA_*.wav (bonafide) + CON_*.wav (spoof)
    │   │   └── train.lst       # utterance IDs, one per line
    │   ├── dev/
    │   │   ├── con_wav/
    │   │   └── dev.lst
    │   ├── eval/
    │   │   ├── con_wav/
    │   │   └── eval.lst
    │   ├── protocols/
    │   │   └── PartialSpoof_LA_cm_protocols/
    │   │       ├── PartialSpoof.LA.cm.train.trl.txt
    │   │       ├── PartialSpoof.LA.cm.dev.trl.txt
    │   │       └── PartialSpoof.LA.cm.eval.trl.txt
    │   ├── segment_labels/
    │   │   └── {split}_seglab_{resolution}.npy   # 0.01 .. 0.64
    │   └── vad/
    │       └── {split}/{utt_id}.vad

Label polarity in the dataset files:
    '0' = spoof,  '1' = bonafide   (README_v1.2, line 256)

This loader **flips** labels on load so that the internal convention is:
    0 = real (bonafide),  1 = fake (spoof)

This matches AudioSegmentSample.frame_labels semantics used throughout the
XPS-Forensic pipeline.

References:
    [1] L. Zhang et al., "An Initial Investigation for Detecting Partially
        Spoofed Audio," Proc. Interspeech 2021.  arXiv:2104.02518
    [2] L. Zhang et al., "Multi-task Learning in Utterance-level and
        Segmental-level Spoof Detection," Proc. ASVspoof 2021. arXiv:2107.14132
    [3] L. Zhang et al., "The PartialSpoof Database and Countermeasures for the
        Detection of Short Generated Audio Segments Embedded in a Speech
        Utterance," IEEE/ACM TASLP, 2023.  arXiv:2204.05177
    [4] Zenodo: https://zenodo.org/record/5766198
    [5] GitHub: https://github.com/nii-yamagishilab/PartialSpoof
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import soundfile as sf

from xps_forensic.data.base import AudioSegmentSample, BasePartialSpoofDataset

logger = logging.getLogger(__name__)

# Finest frame resolution in the dataset (10 ms).
FRAME_SHIFT_MS: int = 10

# Multi-resolution windows available in segment_labels (seconds).
AVAILABLE_RESOLUTIONS_S: list[str] = [
    "0.01", "0.02", "0.04", "0.08", "0.16", "0.32", "0.64",
]

# Corresponding ms values used in XPS-Forensic experiments.
RESOLUTIONS_MS: list[int] = [10, 20, 40, 80, 160, 320, 640]

# Split name mapping for .lst files (split -> lst filename stem).
_SPLIT_MAP = {"train": "train", "dev": "dev", "eval": "eval"}


class PartialSpoofDataset(BasePartialSpoofDataset):
    """Loader for the PartialSpoof corpus (v1.2).

    Loads utterances from .lst files, utterance-level labels from protocol
    files, and frame-level labels from .npy segment label dictionaries.

    Label polarity is flipped on load: dataset '0' (spoof) → internal 1 (fake),
    dataset '1' (bonafide) → internal 0 (real).

    Args:
        root: Path to the PartialSpoof root (parent of ``database/``).
        split: One of 'train', 'dev', 'eval'.
        sample_rate: Target sample rate (default 16000).
        seg_resolution: Segment label resolution in seconds (default '0.01').
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "eval",
        sample_rate: int = 16000,
        seg_resolution: str = "0.01",
    ):
        self.seg_resolution = seg_resolution
        self._seg_labels: dict[str, np.ndarray] = {}
        self._utt_labels: dict[str, str] = {}
        super().__init__(root=root, split=split, sample_rate=sample_rate)

    # ------------------------------------------------------------------
    # Manifest
    # ------------------------------------------------------------------

    def _load_manifest(self) -> list[dict]:
        """Build manifest from .lst file, protocol, and segment labels."""
        db = self.root / "database"
        if not db.exists():
            logger.warning("database/ directory not found under %s", self.root)
            return []

        # 1. Load utterance IDs from .lst
        split_dir = db / self.split
        lst_path = split_dir / f"{self.split}.lst"
        if not lst_path.exists():
            logger.warning("List file not found: %s", lst_path)
            return []
        utt_ids = self._load_lst(lst_path)

        # 2. Load utterance-level labels from protocol
        self._utt_labels = self._load_protocol(db, self.split)

        # 3. Load segment-level labels from .npy
        self._seg_labels = self._load_segment_labels(db, self.split, self.seg_resolution)

        # 4. Build manifest
        manifest: list[dict] = []
        wav_dir = split_dir / "con_wav"
        for utt_id in utt_ids:
            wav_path = wav_dir / f"{utt_id}.wav"
            utt_key = self._utt_labels.get(utt_id, "bonafide")

            # Ternary utterance label: 0=real, 1=partial, 2=fully_fake
            utterance_label = self._determine_utterance_label(
                utt_key, utt_id, self._seg_labels
            )

            manifest.append(
                {
                    "utterance_id": utt_id,
                    "wav_path": str(wav_path),
                    "utterance_label": utterance_label,
                }
            )

        logger.info(
            "PartialSpoof [%s]: loaded %d utterances (resolution=%s)",
            self.split, len(manifest), self.seg_resolution,
        )
        return manifest

    # ------------------------------------------------------------------
    # Sample loading
    # ------------------------------------------------------------------

    def _load_sample(self, entry: dict) -> AudioSegmentSample:
        """Load waveform and frame-level labels for one utterance."""
        wav_path = Path(entry["wav_path"])

        waveform, sr = sf.read(wav_path, dtype="float32")
        if sr != self.sample_rate:
            waveform = self._resample(waveform, sr, self.sample_rate)

        frame_labels = self._get_frame_labels(
            entry["utterance_id"], waveform, self.sample_rate
        )

        return AudioSegmentSample(
            utterance_id=entry["utterance_id"],
            waveform=waveform,
            sample_rate=self.sample_rate,
            utterance_label=entry["utterance_label"],
            frame_labels=frame_labels,
            dataset="partialspoof",
        )

    # ------------------------------------------------------------------
    # Parsing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_lst(lst_path: Path) -> list[str]:
        """Load utterance IDs from a .lst file (one ID per line)."""
        ids = []
        with open(lst_path, "r") as fh:
            for line in fh:
                utt_id = line.strip()
                if utt_id:
                    ids.append(utt_id)
        return ids

    @staticmethod
    def _load_protocol(db_root: Path, split: str) -> dict[str, str]:
        """Parse CM protocol file into {utt_id: 'bonafide'|'spoof'}.

        Protocol format (README_v1.2, lines 201-212):
            SPEAKER_ID  AUDIO_FILE_NAME  -  SYSTEM_ID  KEY

        Column 2 (0-indexed: 1) = utterance ID.
        Column 5 (0-indexed: 4) = 'bonafide' or 'spoof'.
        """
        proto_path = (
            db_root / "protocols" / "PartialSpoof_LA_cm_protocols"
            / f"PartialSpoof.LA.cm.{split}.trl.txt"
        )
        labels: dict[str, str] = {}
        if not proto_path.exists():
            logger.warning("Protocol file not found: %s", proto_path)
            return labels

        with open(proto_path, "r") as fh:
            for line in fh:
                parts = line.strip().split()
                if len(parts) >= 5:
                    utt_id = parts[1]
                    key = parts[4]
                    labels[utt_id] = key
        return labels

    @staticmethod
    def _load_segment_labels(
        db_root: Path, split: str, resolution: str
    ) -> dict[str, np.ndarray]:
        """Load segment labels from .npy dictionary.

        The .npy file stores a pickled defaultdict mapping
        utt_id → np.ndarray of string '0'/'1'.

        Dataset convention: '0' = spoof, '1' = bonafide.
        We flip to internal convention: 0 = real, 1 = fake.
        """
        npy_path = db_root / "segment_labels" / f"{split}_seglab_{resolution}.npy"
        if not npy_path.exists():
            logger.warning("Segment labels not found: %s", npy_path)
            return {}

        raw = np.load(npy_path, allow_pickle=True).item()
        flipped: dict[str, np.ndarray] = {}
        for utt_id, label_arr in raw.items():
            # Convert string array to int and flip: '0'(spoof)→1, '1'(bona)→0
            int_arr = np.array([int(x) for x in label_arr], dtype=np.int32)
            flipped[utt_id] = 1 - int_arr
        return flipped

    @staticmethod
    def _determine_utterance_label(
        protocol_key: str,
        utt_id: str,
        seg_labels: dict[str, np.ndarray],
    ) -> int:
        """Determine ternary utterance label: 0=real, 1=partial, 2=fully_fake.

        Uses segment labels when available (preferred), falls back to protocol.
        After polarity flip, frame_labels: 0=real, 1=fake.

        - All zeros → 0 (genuine)
        - Mixed 0s and 1s → 1 (partially fake)
        - All ones → 2 (fully fake)

        For protocol-only: 'bonafide' → 0, 'spoof' → 2.
        """
        if utt_id in seg_labels:
            labels = seg_labels[utt_id]
            fake_ratio = labels.sum() / len(labels) if len(labels) > 0 else 0.0
            if fake_ratio == 0.0:
                return 0  # all real
            if fake_ratio == 1.0:
                return 2  # all fake
            return 1  # partial
        # Fallback to protocol
        return 0 if protocol_key == "bonafide" else 2

    def _get_frame_labels(
        self,
        utt_id: str,
        waveform: np.ndarray,
        sample_rate: int,
    ) -> np.ndarray:
        """Return frame-level labels (0=real, 1=fake) at 10 ms resolution.

        Uses pre-loaded segment labels if available; otherwise returns
        all-zeros for bonafide or all-ones for spoof (per protocol).
        """
        n_frames = int(
            np.ceil(len(waveform) / (sample_rate * FRAME_SHIFT_MS / 1000))
        )

        if utt_id in self._seg_labels:
            labels = self._seg_labels[utt_id]
            # Handle length mismatch between label and audio
            if len(labels) >= n_frames:
                return labels[:n_frames]
            # Pad with last value if labels are shorter
            padded = np.zeros(n_frames, dtype=np.int32)
            padded[: len(labels)] = labels
            if len(labels) > 0:
                padded[len(labels):] = labels[-1]
            return padded

        # No segment labels: fall back to protocol
        key = self._utt_labels.get(utt_id, "bonafide")
        if key == "bonafide":
            return np.zeros(n_frames, dtype=np.int32)
        return np.ones(n_frames, dtype=np.int32)

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
