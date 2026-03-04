"""Phoneme-level saliency discretization for PDSM-PS.

Aggregates frame-level saliency attributions to phoneme boundaries,
following Gupta et al. (Interspeech 2024) extended to partial spoof segments.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .alignment import PhonemeSegment


@dataclass
class PhonemeSaliency:
    """Saliency aggregated at phoneme level."""
    phoneme: str
    start_sec: float
    end_sec: float
    mean_saliency: float
    max_saliency: float
    n_frames: int
    alignment_confidence: float = 1.0

    @property
    def duration_sec(self) -> float:
        return self.end_sec - self.start_sec


def discretize_by_phonemes(
    frame_saliency: np.ndarray,
    phoneme_segments: list[PhonemeSegment],
    frame_shift_ms: int = 20,
) -> list[PhonemeSaliency]:
    """Aggregate frame saliency to phoneme boundaries.

    Args:
        frame_saliency: Shape (n_frames,).
        phoneme_segments: List of phoneme boundary segments.
        frame_shift_ms: Frame shift in milliseconds.

    Returns:
        List of PhonemeSaliency, one per phoneme.
    """
    results = []
    for seg in phoneme_segments:
        start_frame = seg.start_frame(frame_shift_ms)
        end_frame = seg.end_frame(frame_shift_ms)

        start_frame = max(0, min(start_frame, len(frame_saliency)))
        end_frame = max(start_frame, min(end_frame, len(frame_saliency)))

        if end_frame > start_frame:
            segment_sal = frame_saliency[start_frame:end_frame]
            results.append(PhonemeSaliency(
                phoneme=seg.phoneme,
                start_sec=seg.start_sec,
                end_sec=seg.end_sec,
                mean_saliency=float(np.mean(segment_sal)),
                max_saliency=float(np.max(segment_sal)),
                n_frames=end_frame - start_frame,
                alignment_confidence=seg.confidence,
            ))
        else:
            results.append(PhonemeSaliency(
                phoneme=seg.phoneme,
                start_sec=seg.start_sec,
                end_sec=seg.end_sec,
                mean_saliency=0.0,
                max_saliency=0.0,
                n_frames=0,
                alignment_confidence=seg.confidence,
            ))

    return results


def discretize_by_fixed_window(
    frame_saliency: np.ndarray,
    window_ms: int = 100,
    frame_shift_ms: int = 20,
) -> list[PhonemeSaliency]:
    """Aggregate frame saliency to fixed-width windows (baseline).

    Args:
        frame_saliency: Shape (n_frames,).
        window_ms: Window width in milliseconds.
        frame_shift_ms: Frame shift in milliseconds.

    Returns:
        List of PhonemeSaliency with synthetic phoneme labels.
    """
    frames_per_window = max(1, window_ms // frame_shift_ms)
    n_frames = len(frame_saliency)
    results = []

    for i in range(0, n_frames, frames_per_window):
        end = min(i + frames_per_window, n_frames)
        segment = frame_saliency[i:end]
        start_sec = i * frame_shift_ms / 1000
        end_sec = end * frame_shift_ms / 1000

        results.append(PhonemeSaliency(
            phoneme=f"W{i // frames_per_window}",
            start_sec=start_sec,
            end_sec=end_sec,
            mean_saliency=float(np.mean(segment)),
            max_saliency=float(np.max(segment)),
            n_frames=end - i,
        ))

    return results
