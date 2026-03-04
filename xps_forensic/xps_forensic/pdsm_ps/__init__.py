"""PDSM-PS: Phoneme-Discretized Saliency Maps for Partial Spoofs.

Extension of PDSM (Gupta et al., Interspeech 2024) to segment-level
partial spoof localization. Applied to CPSL-flagged segments only.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .alignment import PhonemeSegment, align_phonemes_mock
from .discretize import (
    PhonemeSaliency,
    discretize_by_phonemes,
    discretize_by_fixed_window,
)
from .faithfulness import phoneme_iou


@dataclass
class PDSMPSResult:
    """Result of PDSM-PS analysis on a single utterance."""
    phoneme_saliencies: list[PhonemeSaliency]
    top_k_phonemes: list[PhonemeSaliency]
    phoneme_iou_score: float
    aligner_used: str


class PDSMPSPipeline:
    """End-to-end PDSM-PS pipeline."""

    def __init__(
        self,
        aligner: str = "mock",
        saliency_method: str = "mock",
        top_k: int = 5,
        frame_shift_ms: int = 20,
    ):
        self.aligner = aligner
        self.saliency_method = saliency_method
        self.top_k = top_k
        self.frame_shift_ms = frame_shift_ms

    def run(
        self,
        frame_saliency: np.ndarray,
        duration_sec: float,
        spoofed_frame_mask: np.ndarray | None = None,
        wav_path: str | None = None,
        phoneme_segments: list[PhonemeSegment] | None = None,
    ) -> dict:
        """Run PDSM-PS analysis.

        Args:
            frame_saliency: Frame-level saliency, shape (n_frames,).
            duration_sec: Audio duration in seconds.
            spoofed_frame_mask: Ground truth binary mask for IoU computation.
            wav_path: Path to audio for MFA/WhisperX alignment.
            phoneme_segments: Pre-computed phoneme segments (optional).

        Returns:
            Dict with phoneme_saliencies, top_k_phonemes, phoneme_iou.
        """
        # Step 1: Get phoneme boundaries
        if phoneme_segments is None:
            if self.aligner == "mock":
                n_phonemes = max(1, int(duration_sec * 10))
                phoneme_segments = align_phonemes_mock(duration_sec, n_phonemes)
            elif self.aligner == "mfa" and wav_path:
                from .alignment import align_with_mfa
                phoneme_segments = align_with_mfa(wav_path)
            elif self.aligner == "whisperx" and wav_path:
                from .alignment import align_with_whisperx
                phoneme_segments = align_with_whisperx(wav_path)
            else:
                phoneme_segments = align_phonemes_mock(duration_sec)

        # Step 2: Discretize saliency to phoneme level
        phoneme_saliencies = discretize_by_phonemes(
            frame_saliency, phoneme_segments, self.frame_shift_ms
        )

        # Step 3: Rank and select top-K salient phonemes
        sorted_by_saliency = sorted(
            enumerate(phoneme_saliencies),
            key=lambda x: x[1].mean_saliency,
            reverse=True,
        )
        top_k_indices = {idx for idx, _ in sorted_by_saliency[:self.top_k]}
        top_k_phonemes = [ps for idx, ps in sorted_by_saliency[:self.top_k]]

        # Step 4: Compute Phoneme-IoU if ground truth available
        iou_score = 0.0
        if spoofed_frame_mask is not None:
            gt_indices = set()
            for i, seg in enumerate(phoneme_segments):
                start_f = seg.start_frame(self.frame_shift_ms)
                end_f = min(seg.end_frame(self.frame_shift_ms), len(spoofed_frame_mask))
                if start_f < end_f:
                    overlap = spoofed_frame_mask[start_f:end_f].mean()
                    if overlap > 0.5:
                        gt_indices.add(i)
            iou_score = phoneme_iou(top_k_indices, gt_indices)

        return {
            "phoneme_saliencies": phoneme_saliencies,
            "top_k_phonemes": top_k_phonemes,
            "phoneme_iou": iou_score,
            "aligner": self.aligner,
        }
