"""BAM detector wrapper.

Reference: Zhong, Li, Yi. "Enhancing Partially Spoofed Audio Localization with
Boundary-aware Attention Mechanism." Interspeech 2024. arXiv:2407.21611

Wraps the official BAM implementation from:
https://github.com/media-sec-lab/BAM

IMPORTANT: This is a READ-ONLY wrapper. The external BAM source code must NOT
be modified. All adaptation logic lives in this wrapper.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .base import BaseDetector, DetectorOutput


class BAMDetector(BaseDetector):
    """Wrapper for BAM (Boundary-aware Attention Mechanism) detector.

    BAM uses WavLM features with a boundary-aware attention mechanism
    for frame-level partial spoof localization. The model outputs
    per-frame logits at 20ms resolution (WavLM frame shift).

    Args:
        checkpoint: Path to pretrained model weights (.pt/.pth).
        external_dir: Path to the cloned BAM repository root.
        device: Torch device string (default "cpu").
    """

    name = "BAM"
    frame_shift_ms = 20  # BAM outputs at 20ms resolution (WavLM frame shift)

    def __init__(
        self,
        checkpoint: str | Path | None = None,
        external_dir: str | Path | None = None,
        device: str = "cpu",
    ):
        super().__init__(checkpoint, device)
        self.external_dir = Path(external_dir) if external_dir else None

    def load_model(self) -> None:
        """Load BAM model from external repo + checkpoint.

        Inserts the external BAM repo into sys.path and imports the
        model class. The external source code is used as-is without
        modification.

        Raises:
            ValueError: If external_dir is not set.
        """
        if self.external_dir is None:
            raise ValueError("external_dir must point to cloned BAM repo")

        bam_path = str(self.external_dir)
        if bam_path not in sys.path:
            sys.path.insert(0, bam_path)

        from model import BAM as BAMModel  # noqa: E402

        self.model = BAMModel()
        if self.checkpoint:
            state = torch.load(
                self.checkpoint, map_location=self.device, weights_only=True
            )
            self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()

    def predict(self, waveform: np.ndarray, sample_rate: int = 16000) -> DetectorOutput:
        """Run BAM inference on a single waveform.

        Handles multiple output formats from the BAM model:
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

            # Handle different output formats from the BAM model
            if isinstance(output, dict):
                logits = output.get("frame_logits", output.get("logits"))
            elif isinstance(output, tuple):
                logits = output[0]
            else:
                logits = output

            # Convert logits to probabilities based on tensor dimensions
            if logits.dim() == 3:
                # (batch, n_frames, n_classes) -> softmax over classes
                probs = F.softmax(logits, dim=-1)
                frame_scores = probs[0, :, 1].cpu().numpy()
            elif logits.dim() == 2:
                # (batch, n_frames) -> sigmoid
                frame_scores = torch.sigmoid(logits[0]).cpu().numpy()
            else:
                # (n_frames,) -> sigmoid
                frame_scores = torch.sigmoid(logits).cpu().numpy().flatten()

        utterance_score = float(np.max(frame_scores))

        return DetectorOutput(
            utterance_id="",
            frame_scores=frame_scores,
            utterance_score=utterance_score,
            frame_shift_ms=self.frame_shift_ms,
            detector_name=self.name,
        )
