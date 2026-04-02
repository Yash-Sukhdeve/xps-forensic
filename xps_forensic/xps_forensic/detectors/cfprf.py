"""CFPRF detector wrapper (FDN stage only).

Reference: Wu et al. "Coarse-to-Fine Proposal Refinement Framework for Audio
Temporal Forgery Detection and Localization." ACM MM 2024. arXiv:2407.16554

Wraps the official CFPRF implementation from:
https://github.com/ItzJuny/CFPRF

Architecture note: CFPRF is a two-stage pipeline:
  - Stage 1 (FDN): Frame-level Detection Network — frame-level seg/boundary
    scores at XLSR-native 20ms resolution. This is what we wrap.
  - Stage 2 (PRN): Proposal Refinement Network — refines boundary proposals
    using embeddings from FDN. Available for future use but NOT required for
    our core metrics (segment EER, CPSL).

IMPORTANT: This is a READ-ONLY wrapper. The external CFPRF source code must
NOT be modified. All adaptation logic lives in this wrapper.
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .base import BaseDetector, DetectorOutput

logger = logging.getLogger(__name__)

# Default FDN constructor kwargs from CFPRF repo
_DEFAULT_FDN_KWARGS = {
    "seq_len": 1070,
    "gmlp_layers": 1,
}


class CFPRFDetector(BaseDetector):
    """Wrapper for CFPRF Frame-level Detection Network (FDN).

    The FDN uses XLSR-300M features with a boundary-aware gMLP architecture
    for frame-level partial spoof localization. The model outputs per-frame
    2-class logits at 20ms resolution (XLSR native frame rate).

    The FDN forward pass returns ``(seg_scores, bd_scores, emb_T, F_BA)``
    where:
    - ``seg_scores``: ``(batch, n_frames, 2)`` logits (class 0=real, 1=spoof)
    - ``bd_scores``: ``(batch, n_frames, 2)`` boundary logits
    - ``emb_T``: ``(batch, n_frames, 128)`` temporal embeddings
    - ``F_BA``: ``(batch, n_frames, 128)`` boundary-aware features

    We use ``seg_scores`` for binary spoof detection, applying softmax and
    taking the spoof class (index 1) as the per-frame score.

    The FDN internally loads XLSR-300M via fairseq from a hardcoded relative
    path ``./pretrain_models/xlsr2_300m.pt``. The ``ssl_path`` parameter
    controls the working directory used during model construction so that
    this relative path resolves correctly.

    Checkpoint format: Plain ``.pth`` state dict files.

    Args:
        checkpoint: Path to pretrained FDN weights (.pth).
        external_dir: Path to the cloned CFPRF repository root.
        device: Torch device string (default "cpu").
        ssl_path: Directory containing ``pretrain_models/xlsr2_300m.pt``.
            If None, defaults to ``external_dir`` (the CFPRF repo root,
            which is where the original code expects the pretrain_models/
            directory to be).
        seq_len: Maximum sequence length for gMLP (default 1070).
        gmlp_layers: Number of gMLP layers (default 1).
    """

    name = "CFPRF"
    frame_shift_ms = 20  # FDN operates at XLSR native 20ms resolution

    def __init__(
        self,
        checkpoint: str | Path | None = None,
        external_dir: str | Path | None = None,
        device: str = "cpu",
        ssl_path: str | Path | None = None,
        seq_len: int = 1070,
        gmlp_layers: int = 1,
    ):
        super().__init__(checkpoint, device)
        self.external_dir = Path(external_dir) if external_dir else None
        self.ssl_path = str(ssl_path) if ssl_path else None
        self.seq_len = seq_len
        self.gmlp_layers = gmlp_layers

    def load_model(self) -> None:
        """Load CFPRF FDN model from external repo + checkpoint.

        Inserts the external CFPRF repo into sys.path and imports the
        ``CFPRF_FDN`` model class. Temporarily changes the working
        directory so that the hardcoded XLSR checkpoint path in
        ``ASRModel.__init__`` resolves correctly.

        For plain ``.pth`` checkpoint files, loads the state dict
        directly into the FDN module.

        Raises:
            ValueError: If external_dir is not set.
            FileNotFoundError: If external_dir or checkpoint does not exist.
        """
        if self.external_dir is None:
            raise ValueError("external_dir must point to cloned CFPRF repo")
        if not self.external_dir.is_dir():
            raise FileNotFoundError(
                f"CFPRF external directory not found: {self.external_dir}"
            )

        # Resolve to absolute path to avoid cwd-dependent import failures
        cfprf_path = str(self.external_dir.resolve())
        if cfprf_path not in sys.path:
            sys.path.insert(0, cfprf_path)

        from models.FDN import CFPRF_FDN  # noqa: E402

        # The ASRModel class in FDN.py loads XLSR from a hardcoded
        # relative path: ./pretrain_models/xlsr2_300m.pt
        # We temporarily change cwd so the path resolves correctly.
        ssl_dir = self.ssl_path or cfprf_path
        orig_cwd = os.getcwd()
        try:
            os.chdir(ssl_dir)
            self.model = CFPRF_FDN(
                seq_len=self.seq_len,
                gmlp_layers=self.gmlp_layers,
            )
        finally:
            os.chdir(orig_cwd)

        if self.checkpoint:
            ckpt_path = Path(self.checkpoint)
            if not ckpt_path.is_file():
                raise FileNotFoundError(
                    f"CFPRF checkpoint not found: {ckpt_path}"
                )
            state = torch.load(
                str(ckpt_path), map_location=self.device, weights_only=False
            )
            # Handle both plain state dict and wrapped formats
            if "state_dict" in state:
                state = state["state_dict"]
            self.model.load_state_dict(state)
            logger.info(
                "Loaded CFPRF FDN weights from checkpoint (%d keys)",
                len(state),
            )

        self.model.to(self.device)
        self.model.eval()

    def predict(
        self,
        waveform: np.ndarray,
        sample_rate: int = 16000,
        utterance_id: str = "",
    ) -> DetectorOutput:
        """Run CFPRF FDN inference on a single waveform.

        The FDN forward pass returns ``(seg_scores, bd_scores, emb_T, F_BA)``
        where ``seg_scores`` has shape ``(batch, n_frames, 2)``. We apply
        softmax over the last dimension and take class index 1 (spoof
        probability) as the per-frame score.

        Args:
            waveform: 1-D float array of raw audio samples at sample_rate.
            sample_rate: Sample rate in Hz (default 16000).
            utterance_id: Identifier for the utterance.

        Returns:
            DetectorOutput with frame-level spoof probabilities at 20ms
            resolution.

        Raises:
            RuntimeError: If load_model() has not been called.
        """
        if self.model is None:
            raise RuntimeError("Call load_model() before predict()")

        with torch.no_grad():
            x = torch.from_numpy(waveform).float().unsqueeze(0).to(self.device)
            seg_scores, _bd_scores, _emb_T, _F_BA = self.model(x)

            # seg_scores shape: (1, n_frames, 2) — apply softmax for probs
            probs = F.softmax(seg_scores, dim=-1)
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
