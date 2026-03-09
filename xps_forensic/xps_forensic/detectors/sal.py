"""SAL detector wrapper.

Reference: Mao, Huang, Qian. "Localizing Speech Deepfakes Beyond Transitions
via Segment-Aware Learning." arXiv:2601.21925, 2026.

Wraps the official SAL implementation from:
https://github.com/SentryMao/SAL

IMPORTANT: This is a READ-ONLY wrapper. The external SAL source code must NOT
be modified. All adaptation logic lives in this wrapper.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .base import BaseDetector, DetectorOutput

logger = logging.getLogger(__name__)

# Default SAL (SSLSeq8Bin) constructor kwargs derived from the SAL repo
# config files and model definition.
_DEFAULT_SAL_KWARGS = {
    "ssl_encoder": "xlsr",
    "mode": "s3prl",
    "hid_dim": 1024,
    "resolution_train": 0.16,
    "resolution_test": 0.16,
    "pool": "att",
    "pool_head_num": 1,
    "num_outputs": 2,
    "seq_model": "cf",
    "num_layers": 2,
    "num_heads": 4,
}


class SALDetector(BaseDetector):
    """Wrapper for SAL (Segment-Aware Learning) detector.

    SAL localizes speech deepfakes beyond transition boundaries using
    segment-aware learning with an SSL encoder and conformer sequence model.
    It operates on raw waveform input and produces segment-level spoof
    scores at 160ms resolution (8 pooled SSL frames x 20ms).

    The model class is ``SSLSeq8Bin`` from ``src.models.net.model``.
    Its forward pass returns ``(out1, out2)`` where:
    - ``out1``: ``(batch, n_segments, 8)`` positional logits (SPL head)
    - ``out2``: ``(batch, n_segments, 2)`` binary logits (bonafide/spoof)

    We use ``out2`` for binary spoof detection, applying softmax and
    taking the spoof class (index 1) as the per-segment score.

    Checkpoint format: PyTorch Lightning ``.ckpt`` with state dict keys
    prefixed by ``net.`` (from ``SALTrainer``).

    Args:
        checkpoint: Path to pretrained model weights (.ckpt).
        external_dir: Path to the cloned SAL repository root.
        device: Torch device string (default "cpu").
        ssl_ckpt: Path to SSL encoder checkpoint (e.g. xlsr). If None,
            uses the default from SAL config.
        resolution: Segment resolution in seconds (default 0.16).
    """

    name = "SAL"
    frame_shift_ms = 160  # SAL pools 8 SSL frames (8 * 20ms = 160ms)

    def __init__(
        self,
        checkpoint: str | Path | None = None,
        external_dir: str | Path | None = None,
        device: str = "cpu",
        ssl_ckpt: str | Path | None = None,
        resolution: float = 0.16,
    ):
        super().__init__(checkpoint, device)
        self.external_dir = Path(external_dir) if external_dir else None
        self.ssl_ckpt = str(ssl_ckpt) if ssl_ckpt else None
        self.resolution = resolution

    def load_model(self) -> None:
        """Load SAL model from external repo + checkpoint.

        Inserts the external SAL repo into sys.path and imports the
        ``SSLSeq8Bin`` model class. Constructs the model with default
        kwargs, optionally overriding the SSL checkpoint path.

        For Lightning checkpoints, strips the ``net.`` prefix from
        state dict keys before loading into the SSLSeq8Bin module.

        Raises:
            ValueError: If external_dir is not set.
            FileNotFoundError: If external_dir or checkpoint does not exist.
        """
        if self.external_dir is None:
            raise ValueError("external_dir must point to cloned SAL repo")
        if not self.external_dir.is_dir():
            raise FileNotFoundError(
                f"SAL external directory not found: {self.external_dir}"
            )

        sal_path = str(self.external_dir)
        if sal_path not in sys.path:
            sys.path.insert(0, sal_path)

        from src.models.net.model import SSLSeq8Bin  # noqa: E402

        # Build kwargs from defaults, optionally override SSL checkpoint
        kwargs = dict(_DEFAULT_SAL_KWARGS)
        kwargs["resolution_train"] = self.resolution
        kwargs["resolution_test"] = self.resolution
        if self.ssl_ckpt is not None:
            kwargs["ckpt"] = self.ssl_ckpt

        self.model = SSLSeq8Bin(**kwargs)

        if self.checkpoint:
            ckpt_path = Path(self.checkpoint)
            if not ckpt_path.is_file():
                raise FileNotFoundError(
                    f"SAL checkpoint not found: {ckpt_path}"
                )
            state = torch.load(
                str(ckpt_path), map_location=self.device, weights_only=False
            )
            # Lightning .ckpt files store weights under 'state_dict' key
            # with keys prefixed by 'net.' (from SALTrainer)
            if "state_dict" in state:
                raw_sd = state["state_dict"]
                cleaned_sd = {}
                prefix = "net."
                for k, v in raw_sd.items():
                    if k.startswith(prefix):
                        cleaned_sd[k[len(prefix):]] = v
                    else:
                        cleaned_sd[k] = v
                self.model.load_state_dict(cleaned_sd)
                logger.info(
                    "Loaded SAL weights from Lightning checkpoint "
                    "(%d parameters)", len(cleaned_sd)
                )
            else:
                # Plain state dict (non-Lightning)
                self.model.load_state_dict(state)
                logger.info("Loaded SAL weights from plain state dict")

        self.model.to(self.device)
        self.model.eval()

    def predict(
        self,
        waveform: np.ndarray,
        sample_rate: int = 16000,
        utterance_id: str = "",
    ) -> DetectorOutput:
        """Run SAL inference on a single waveform.

        The SSLSeq8Bin forward pass returns ``(out1, out2)`` where
        ``out2`` has shape ``(batch, n_segments, 2)``. We apply softmax
        over the last dimension and take class index 1 (spoof probability)
        as the per-segment score.

        Note: The SAL SSL encoder internally pads the input by 256 samples,
        so no external padding is needed.

        Args:
            waveform: 1-D float array of raw audio samples at sample_rate.
            sample_rate: Sample rate in Hz (default 16000).
            utterance_id: Identifier for the utterance.

        Returns:
            DetectorOutput with segment-level spoof probabilities at
            160ms resolution.

        Raises:
            RuntimeError: If load_model() has not been called.
        """
        if self.model is None:
            raise RuntimeError("Call load_model() before predict()")

        with torch.no_grad():
            x = torch.from_numpy(waveform).float().unsqueeze(0).to(self.device)
            _out1, out2 = self.model(x)

            # out2 shape: (1, n_segments, 2) — apply softmax for probs
            probs = F.softmax(out2, dim=-1)
            # Class 1 = spoof probability
            frame_scores = probs[0, :, 1].cpu().numpy()

        utterance_score = (
            float(np.max(frame_scores)) if len(frame_scores) > 0 else 0.0
        )

        return DetectorOutput(
            utterance_id=utterance_id,
            frame_scores=frame_scores,
            utterance_score=utterance_score,
            frame_shift_ms=self.frame_shift_ms,
            detector_name=self.name,
        )
