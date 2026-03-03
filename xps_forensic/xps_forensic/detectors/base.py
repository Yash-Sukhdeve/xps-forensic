"""Base detector interface for XPS-Forensic.

Provides the abstract base class and output dataclass that all detector
wrappers must implement. Every detector produces frame-level spoof scores
at a fixed temporal resolution, enabling downstream calibration and
explainability modules to operate detector-agnostically.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass
class DetectorOutput:
    """Output from a frame-level deepfake detector.

    Attributes:
        utterance_id: Unique identifier for the utterance.
        frame_scores: Array of shape (n_frames,) with values in [0, 1],
            where higher values indicate higher probability of spoofing.
        utterance_score: Aggregated utterance-level spoof score.
        frame_shift_ms: Temporal resolution in milliseconds per frame.
        detector_name: Name of the detector that produced this output.
    """

    utterance_id: str
    frame_scores: np.ndarray
    utterance_score: float
    frame_shift_ms: int
    detector_name: str

    @property
    def n_frames(self) -> int:
        """Number of frames in the output."""
        return len(self.frame_scores)

    @property
    def duration_ms(self) -> int:
        """Total duration covered by the frame scores in milliseconds."""
        return self.n_frames * self.frame_shift_ms

    def binarize(self, threshold: float = 0.5) -> np.ndarray:
        """Convert frame scores to binary predictions.

        Args:
            threshold: Decision threshold; frames >= threshold are marked 1.

        Returns:
            Integer array of shape (n_frames,) with values in {0, 1}.
        """
        return (self.frame_scores >= threshold).astype(int)

    def scores_at_resolution(self, resolution_ms: int) -> np.ndarray:
        """Average frame scores at a coarser temporal resolution.

        Args:
            resolution_ms: Desired resolution in milliseconds. Must be >= frame_shift_ms.

        Returns:
            Array of averaged scores at the requested resolution.
        """
        frames_per_seg = max(1, resolution_ms // self.frame_shift_ms)
        n = len(self.frame_scores)
        segments = []
        for i in range(0, n, frames_per_seg):
            segments.append(np.mean(self.frame_scores[i : i + frames_per_seg]))
        return np.array(segments)


class BaseDetector(ABC):
    """Abstract base class for frame-level spoof detectors.

    All detector wrappers inherit from this class and implement:
      - ``load_model()``: Load pretrained weights from a checkpoint.
      - ``predict()``: Run inference on a single waveform.

    The ``predict_batch()`` method provides a default sequential
    implementation that subclasses may override for GPU-batched inference.

    Attributes:
        name: Human-readable detector name (e.g. "BAM", "SAL").
        frame_shift_ms: Temporal resolution of frame-level outputs in ms.
    """

    name: str = "base"
    frame_shift_ms: int = 20

    def __init__(self, checkpoint: str | Path | None = None, device: str = "cpu"):
        self.device = torch.device(device)
        self.checkpoint = checkpoint
        self.model = None

    @abstractmethod
    def load_model(self) -> None:
        """Load model weights from checkpoint."""

    @abstractmethod
    def predict(
        self,
        waveform: np.ndarray,
        sample_rate: int = 16000,
        utterance_id: str = "",
    ) -> DetectorOutput:
        """Run inference on a single waveform.

        Args:
            waveform: 1-D float array of raw audio samples.
            sample_rate: Sample rate in Hz (default 16000).
            utterance_id: Identifier for the utterance.

        Returns:
            DetectorOutput with frame-level scores.
        """

    def predict_batch(
        self,
        waveforms: list[np.ndarray],
        utterance_ids: list[str],
        sample_rate: int = 16000,
    ) -> list[DetectorOutput]:
        """Run inference on a batch of waveforms.

        Default implementation processes sequentially. Subclasses may
        override for GPU-batched inference.

        Args:
            waveforms: List of 1-D float arrays.
            utterance_ids: Corresponding utterance identifiers.
            sample_rate: Sample rate in Hz.

        Returns:
            List of DetectorOutput, one per waveform.
        """
        return [
            self.predict(wav, sample_rate, utterance_id=uid)
            for wav, uid in zip(waveforms, utterance_ids)
        ]
