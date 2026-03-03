"""SAL detector wrapper.

Reference: Mao, Huang, Qian. "Localizing Speech Deepfakes Beyond Transitions
via Segment-Aware Learning." arXiv:2601.21925, 2026.

Wraps the official SAL implementation from:
https://github.com/SentryMao/SAL

IMPORTANT: This is a READ-ONLY wrapper. The external SAL source code must NOT
be modified. All adaptation logic lives in this wrapper.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .base import BaseDetector, DetectorOutput


class SALDetector(BaseDetector):
    """Wrapper for SAL (Segment-Aware Learning) detector.

    SAL localizes speech deepfakes beyond transition boundaries using
    segment-aware learning. It operates on raw waveform input and
    produces frame-level spoof scores at 20ms resolution.

    Args:
        checkpoint: Path to pretrained model weights (.pt/.pth).
        external_dir: Path to the cloned SAL repository root.
        device: Torch device string (default "cpu").
    """

    name = "SAL"
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
        """Load SAL model from external repo + checkpoint.

        Inserts the external SAL repo into sys.path and imports the
        model class. The external source code is used as-is without
        modification.

        Raises:
            ValueError: If external_dir is not set.
        """
        if self.external_dir is None:
            raise ValueError("external_dir must point to cloned SAL repo")
        sal_path = str(self.external_dir)
        if sal_path not in sys.path:
            sys.path.insert(0, sal_path)

        from model import SAL as SALModel  # noqa: E402

        self.model = SALModel()
        if self.checkpoint:
            state = torch.load(
                self.checkpoint, map_location=self.device, weights_only=True
            )
            self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()

    def predict(self, waveform: np.ndarray, sample_rate: int = 16000) -> DetectorOutput:
        """Run SAL inference on a single waveform.

        Handles multiple output formats from the SAL model:
        - dict with 'frame_logits' or 'logits' key
        - tuple (logits, ...)
        - raw tensor

        For 3-D logits (batch, frames, classes), applies softmax and
        takes the spoof class probability (index 1). For 2-D or 1-D,
        applies sigmoid.

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

            # Handle different output formats from the SAL model
            if isinstance(output, dict):
                logits = output.get("frame_logits", output.get("logits"))
            elif isinstance(output, tuple):
                logits = output[0]
            else:
                logits = output

            # Convert logits to probabilities based on tensor dimensions
            if logits.dim() == 3:
                probs = F.softmax(logits, dim=-1)
                frame_scores = probs[0, :, 1].cpu().numpy()
            elif logits.dim() == 2:
                frame_scores = torch.sigmoid(logits[0]).cpu().numpy()
            else:
                frame_scores = torch.sigmoid(logits).cpu().numpy().flatten()

        return DetectorOutput(
            utterance_id="",
            frame_scores=frame_scores,
            utterance_score=float(np.max(frame_scores)),
            frame_shift_ms=self.frame_shift_ms,
            detector_name=self.name,
        )
