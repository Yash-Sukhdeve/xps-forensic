"""MRM (Multi-Resolution Model) detector wrapper.

Reference: Zhang et al. "The PartialSpoof Database and Countermeasures for the
Detection of Short Generated Speech Segments Embedded in Natural Speech."
IEEE/ACM TASLP 2023. arXiv:2204.05177

Wraps the official MRM implementation from:
https://github.com/hieuthi/MultiResoModel-Simple

IMPORTANT: This is a READ-ONLY wrapper. The external MRM source code must NOT
be modified. All adaptation logic lives in this wrapper.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

from .base import BaseDetector, DetectorOutput


class MRMDetector(BaseDetector):
    """Wrapper for MRM (Multi-Resolution Model) baseline detector.

    MRM is the baseline detector from the PartialSpoof paper. It uses
    multi-resolution features for frame-level spoof detection, producing
    sigmoid-activated frame scores at 20ms resolution.

    Args:
        checkpoint: Path to pretrained model weights (.pt/.pth).
        external_dir: Path to the cloned MRM repository root.
        device: Torch device string (default "cpu").
    """

    name = "MRM"
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
        """Load MRM model from external repo + checkpoint.

        Inserts the external MRM repo into sys.path and imports the
        model class. The external source code is used as-is without
        modification.

        Raises:
            ValueError: If external_dir is not set.
        """
        if self.external_dir is None:
            raise ValueError("external_dir must point to cloned MRM repo")
        mrm_path = str(self.external_dir)
        if mrm_path not in sys.path:
            sys.path.insert(0, mrm_path)

        from model import MultiResoModel  # noqa: E402

        self.model = MultiResoModel()
        if self.checkpoint:
            state = torch.load(
                self.checkpoint, map_location=self.device, weights_only=True
            )
            self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()

    def predict(
        self,
        waveform: np.ndarray,
        sample_rate: int = 16000,
        utterance_id: str = "",
    ) -> DetectorOutput:
        """Run MRM inference on a single waveform.

        Handles multiple output formats from the MRM model:
        - dict with 'frame_logits' or 'logits' key
        - tuple (logits, ...)
        - raw tensor

        All logits are sigmoid-activated to produce frame scores.

        Args:
            waveform: 1-D float array of raw audio samples at sample_rate.
            sample_rate: Sample rate in Hz (default 16000).
            utterance_id: Identifier for the utterance.

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

            # Handle different output formats from the MRM model
            if isinstance(output, dict):
                logits = output.get("frame_logits", output.get("logits"))
            elif isinstance(output, tuple):
                logits = output[0]
            else:
                logits = output

            frame_scores = torch.sigmoid(logits).cpu().numpy().flatten()

        return DetectorOutput(
            utterance_id=utterance_id,
            frame_scores=frame_scores,
            utterance_score=float(np.max(frame_scores)) if len(frame_scores) > 0 else 0.0,
            frame_shift_ms=self.frame_shift_ms,
            detector_name=self.name,
        )
