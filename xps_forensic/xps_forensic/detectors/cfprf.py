"""CFPRF detector wrapper.

Reference: Wu et al. "Coarse-to-Fine Proposal Refinement Framework for Audio
Temporal Forgery Detection and Localization." ACM MM 2024. arXiv:2407.16554

Wraps the official CFPRF implementation from:
https://github.com/ItzJuny/CFPRF

IMPORTANT: This is a READ-ONLY wrapper. The external CFPRF source code must
NOT be modified. All adaptation logic lives in this wrapper.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

from .base import BaseDetector, DetectorOutput


class CFPRFDetector(BaseDetector):
    """Wrapper for CFPRF (Coarse-to-Fine Proposal Refinement) detector.

    CFPRF uses a proposal-based approach for audio temporal forgery
    detection and localization. The model outputs temporal proposals
    (start, end, confidence) which this wrapper converts to frame-level
    scores for compatibility with the XPS-Forensic pipeline.

    Args:
        checkpoint: Path to pretrained model weights (.pt/.pth).
        external_dir: Path to the cloned CFPRF repository root.
        device: Torch device string (default "cpu").
    """

    name = "CFPRF"
    frame_shift_ms = 20

    def __init__(
        self,
        checkpoint: str | Path | None = None,
        external_dir: str | Path | None = None,
        device: str = "cpu",
    ):
        super().__init__(checkpoint, device)
        self.external_dir = Path(external_dir) if external_dir else None

    def load_model(self) -> None:
        """Load CFPRF model from external repo + checkpoint.

        Inserts the external CFPRF repo into sys.path and imports the
        model class. The external source code is used as-is without
        modification.

        Raises:
            ValueError: If external_dir is not set.
        """
        if self.external_dir is None:
            raise ValueError("external_dir must point to cloned CFPRF repo")
        cfprf_path = str(self.external_dir)
        if cfprf_path not in sys.path:
            sys.path.insert(0, cfprf_path)

        from models import CFPRF as CFPRFModel  # noqa: E402

        self.model = CFPRFModel()
        if self.checkpoint:
            state = torch.load(
                self.checkpoint, map_location=self.device, weights_only=True
            )
            self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()

    def _proposals_to_frame_scores(
        self,
        proposals: list[tuple[float, float, float]],
        n_frames: int,
    ) -> np.ndarray:
        """Convert temporal proposals to frame-level scores.

        Each proposal is (start_sec, end_sec, confidence). When proposals
        overlap, the maximum confidence is taken for each frame.

        Args:
            proposals: List of (start, end, score) tuples.
            n_frames: Total number of frames in the output.

        Returns:
            Array of shape (n_frames,) with scores in [0, 1].
        """
        frame_scores = np.zeros(n_frames)
        if not proposals:
            return frame_scores

        for start, end, score in proposals:
            s_frame = int(start * 1000 / self.frame_shift_ms)
            e_frame = int(end * 1000 / self.frame_shift_ms)
            # Clamp to valid range
            s_frame = max(0, s_frame)
            e_frame = min(n_frames, e_frame)
            if s_frame < e_frame:
                frame_scores[s_frame:e_frame] = np.maximum(
                    frame_scores[s_frame:e_frame], score
                )
        return frame_scores

    def predict(self, waveform: np.ndarray, sample_rate: int = 16000) -> DetectorOutput:
        """Run CFPRF inference on a single waveform.

        Handles two output formats from the CFPRF model:
        1. dict with 'proposals' key: list of (start, end, score) tuples
           that are converted to frame-level scores.
        2. raw tensor logits: sigmoid-activated to produce frame scores.

        Args:
            waveform: 1-D float array of raw audio samples at sample_rate.
            sample_rate: Sample rate in Hz (default 16000).

        Returns:
            DetectorOutput with frame-level spoof probabilities.

        Raises:
            RuntimeError: If load_model() has not been called.
        """
        if self.model is None:
            raise RuntimeError("Call load_model() before predict()")

        with torch.no_grad():
            x = torch.from_numpy(waveform).float().unsqueeze(0).to(self.device)
            output = self.model(x)

            if isinstance(output, dict) and "proposals" in output:
                proposals = output["proposals"]
                n_frames = max(
                    1, int(len(waveform) / sample_rate * 1000 / self.frame_shift_ms)
                )
                frame_scores = self._proposals_to_frame_scores(proposals, n_frames)
            else:
                logits = output[0] if isinstance(output, tuple) else output
                frame_scores = torch.sigmoid(logits).cpu().numpy().flatten()

        utterance_score = float(np.max(frame_scores)) if len(frame_scores) > 0 else 0.0

        return DetectorOutput(
            utterance_id="",
            frame_scores=frame_scores,
            utterance_score=utterance_score,
            frame_shift_ms=self.frame_shift_ms,
            detector_name=self.name,
        )
