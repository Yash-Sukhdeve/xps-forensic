"""Base data structures for XPS-Forensic datasets."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np


@dataclass
class AudioSegmentSample:
    """Single utterance with frame-level labels.

    Attributes:
        utterance_id: Unique identifier for the utterance.
        waveform: Raw audio waveform, shape (n_samples,).
        sample_rate: Sampling rate in Hz.
        utterance_label: Ternary label — 0=real, 1=partially_fake, 2=fully_fake.
        frame_labels: Binary frame-level labels, shape (n_frames,), 0=real/1=fake.
        dataset: Name of the source dataset.
        metadata: Additional key-value metadata (e.g., codec, SNR).
    """

    utterance_id: str
    waveform: np.ndarray  # shape: (n_samples,)
    sample_rate: int
    utterance_label: int  # 0=real, 1=partially_fake, 2=fully_fake
    frame_labels: np.ndarray  # shape: (n_frames,), binary 0/1
    dataset: str
    metadata: dict = field(default_factory=dict)

    @property
    def duration_sec(self) -> float:
        """Duration of the waveform in seconds."""
        return len(self.waveform) / self.sample_rate

    @property
    def is_partially_fake(self) -> bool:
        """True if the utterance is partially spoofed."""
        return self.utterance_label == 1


class BasePartialSpoofDataset:
    """Abstract base for partial spoof datasets.

    Subclasses must implement ``_load_manifest()`` and ``_load_sample()``.
    Handles graceful degradation when the data root does not exist
    (manifest stays empty, len == 0).
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "eval",
        sample_rate: int = 16000,
    ):
        self.root = Path(root)
        self.split = split
        self.sample_rate = sample_rate
        self.manifest: list[dict] = []
        if self.root.exists():
            self.manifest = self._load_manifest()

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    def _load_manifest(self) -> list[dict]:
        """Return a list of dicts describing every sample in this split."""
        raise NotImplementedError

    def _load_sample(self, entry: dict) -> AudioSegmentSample:
        """Load a single sample from a manifest entry."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Sequence protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, idx: int) -> AudioSegmentSample:
        return self._load_sample(self.manifest[idx])

    def __iter__(self) -> Iterator[AudioSegmentSample]:
        for entry in self.manifest:
            yield self._load_sample(entry)

    # ------------------------------------------------------------------
    # Splitting helpers
    # ------------------------------------------------------------------

    def get_split(
        self,
        ratio: float = 0.8,
        seed: int = 42,
    ) -> tuple[list[int], list[int]]:
        """Split indices into calibration / verification sets.

        Args:
            ratio: Fraction of samples assigned to the calibration set.
            seed: RNG seed for reproducibility.

        Returns:
            Tuple of (calibration_indices, verification_indices).
        """
        rng = np.random.default_rng(seed)
        n = len(self.manifest)
        perm = rng.permutation(n)
        split_point = int(n * ratio)
        return perm[:split_point].tolist(), perm[split_point:].tolist()
