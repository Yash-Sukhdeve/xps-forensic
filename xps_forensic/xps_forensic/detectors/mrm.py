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

import logging
import sys
from pathlib import Path

import numpy as np
import torch

from .base import BaseDetector, DetectorOutput

logger = logging.getLogger(__name__)

# Default config values from configs/baseline.toml
_DEFAULT_MRM_CONFIG = {
    "num_scales": 6,
    "num_gmlp_layers": 5,
    "include_utt": True,
    "flag_pool": "mean",
    "use_mask": True,
    "ssl_dim": 1024,
    "ssl_path": "pretrained/w2v_large_lv_fsh_swbd_cv_fixed.pt",
    "ssl_tuning": True,
    "max_seq_len": 2001,
}


class MRMDetector(BaseDetector):
    """Wrapper for MRM (Multi-Resolution Model) baseline detector.

    MRM is the baseline detector from the PartialSpoof paper. It analyses
    audio at multiple temporal resolutions simultaneously, producing
    cosine-similarity scores via P2SActivationLayer at each scale.

    Scale 0 operates at 20ms resolution (wav2vec 2.0 native frame rate),
    which is the finest scale and the one we extract for our pipeline.

    The P2SActivationLayer outputs ``(n, 2)`` cosine similarities where:
    - Column 0 = cosine similarity to the **bonafide** class weight vector
    - Column 1 = cosine similarity to the **spoof** class weight vector
    Values lie in [-1, 1]. We map spoof similarity to [0, 1] for downstream use.

    Evidence: P2SGrad one-hot encoding assigns target=0 for bonafide (column 0)
    and target=1 for spoof (column 1).

    If ``include_utt=True``, the last element of the logits list is an
    utterance-level prediction of shape ``(batch, 2)``.

    Checkpoint format (from ``utils.save_checkpoint``):
        ``{"modules": {"model": state_dict, ...}, "meta": {...}}``

    Args:
        checkpoint: Path to pretrained model weights (.pt).
        external_dir: Path to the cloned MRM repository root.
        device: Torch device string (default "cpu").
        ssl_path: Path to wav2vec2/XLSR .pt file (fairseq format).
            If None, uses the default from baseline.toml.
        num_scales: Number of temporal scales (default 6).
        include_utt: Whether model includes utterance-level head (default True).
        use_mask: Whether model uses mask pooling (default True).
        max_seq_len: Maximum SSL feature sequence length (default 2001).
    """

    name = "MRM"
    frame_shift_ms = 20  # Finest scale (scale 0) = 20ms per frame

    def __init__(
        self,
        checkpoint: str | Path | None = None,
        external_dir: str | Path | None = None,
        device: str = "cpu",
        ssl_path: str | Path | None = None,
        num_scales: int = 6,
        include_utt: bool = True,
        use_mask: bool = True,
        max_seq_len: int = 2001,
    ):
        super().__init__(checkpoint, device)
        self.external_dir = Path(external_dir) if external_dir else None
        self.ssl_path = str(ssl_path) if ssl_path else None
        self.num_scales = num_scales
        self.include_utt = include_utt
        self.use_mask = use_mask
        self.max_seq_len = max_seq_len

    def load_model(self) -> None:
        """Load MRM model from external repo + checkpoint.

        Inserts the external MRM repo into sys.path and imports
        ``MultiResoModel`` from ``modules.multiresomodel``. The external
        source code is used as-is without modification.

        For checkpoints saved via ``utils.save_checkpoint``, extracts the
        model state dict from ``states["modules"]["model"]``.

        Raises:
            ValueError: If external_dir is not set.
            FileNotFoundError: If external_dir or checkpoint does not exist.
        """
        if self.external_dir is None:
            raise ValueError("external_dir must point to cloned MRM repo")
        if not self.external_dir.is_dir():
            raise FileNotFoundError(
                f"MRM external directory not found: {self.external_dir}"
            )

        mrm_path = str(self.external_dir)
        if mrm_path not in sys.path:
            sys.path.insert(0, mrm_path)

        from modules.multiresomodel import MultiResoModel  # noqa: E402

        # Build constructor kwargs from defaults + overrides
        config = dict(_DEFAULT_MRM_CONFIG)
        if self.ssl_path is not None:
            config["ssl_path"] = self.ssl_path
        config["num_scales"] = self.num_scales
        config["include_utt"] = self.include_utt
        config["use_mask"] = self.use_mask
        config["max_seq_len"] = self.max_seq_len
        config["device"] = str(self.device)

        self.model = MultiResoModel(**config)

        if self.checkpoint:
            ckpt_path = Path(self.checkpoint)
            if not ckpt_path.is_file():
                raise FileNotFoundError(
                    f"MRM checkpoint not found: {ckpt_path}"
                )
            states = torch.load(
                str(ckpt_path), map_location=self.device, weights_only=False
            )
            # MRM uses utils.save_checkpoint format:
            # {"modules": {"model": state_dict, ...}, "meta": {...}}
            if "modules" in states and "model" in states["modules"]:
                model_sd = states["modules"]["model"]
                self.model.load_state_dict(model_sd)
                logger.info(
                    "Loaded MRM weights from checkpoint (%d parameters)",
                    len(model_sd),
                )
            else:
                # Fallback: plain state dict
                self.model.load_state_dict(states)
                logger.info("Loaded MRM weights from plain state dict")

        self.model.to(self.device)
        self.model.eval()

    def predict(
        self,
        waveform: np.ndarray,
        sample_rate: int = 16000,
        utterance_id: str = "",
    ) -> DetectorOutput:
        """Run MRM inference on a single waveform.

        The input waveform is reshaped to ``(1, 1, samples)`` — the extra
        channel dimension is expected by MRM's SSL feature extractor
        (which calls ``x.squeeze(1)`` internally).

        The forward pass returns ``(logits, masks)`` where ``logits`` is
        a list of tensors. At scale 0 (20ms), logits are flattened to
        ``(batch * n_segments, 2)``. We reshape back to ``(1, n_segments, 2)``
        and extract spoof scores from column 1.

        P2SActivationLayer outputs cosine similarities in [-1, 1]:
        - Column 0 = bonafide similarity (higher → more bonafide-like)
        - Column 1 = spoof similarity (higher → more spoof-like)

        We map spoof cosine similarity from [-1, 1] to [0, 1] via
        ``(cos_sim + 1) / 2`` for compatibility with downstream calibration.

        If ``include_utt=True``, the utterance-level score is extracted from
        ``logits[-1]`` (shape ``(batch, 2)``).

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
            # MRM expects (batch, 1, samples) — the channel dim is squeezed
            # inside SSLModel.extract_feat via x.squeeze(1)
            x = (
                torch.from_numpy(waveform)
                .float()
                .unsqueeze(0)  # batch dim
                .unsqueeze(0)  # channel dim
                .to(self.device)
            )
            logits, masks = self.model(x)

            # Scale 0 (finest, 20ms): logits[0] is (batch*n_segments, 2)
            # flattened by torch.flatten(cl_block(hidd), start_dim=0, end_dim=1)
            scale0_logits = logits[0]  # (1*n_segments, 2)

            if scale0_logits.numel() == 0:
                frame_scores = np.array([])
                utterance_score = 0.0
            else:
                # Reshape: since batch=1, n_segments = total_rows
                n_segments = scale0_logits.shape[0]

                # Apply mask if available to get valid segment count
                if self.use_mask and len(masks) > 0:
                    mask = masks[0]  # boolean mask for scale 0
                    scale0_valid = scale0_logits[mask]
                else:
                    scale0_valid = scale0_logits

                # P2SActivation outputs cosine similarity in [-1, 1]
                # Column 0 = bonafide similarity, Column 1 = spoof similarity
                # Map to [0, 1]: (cos_sim + 1) / 2
                spoof_cos = scale0_valid[:, 1]  # spoof cosine similarity
                frame_scores = ((spoof_cos + 1.0) / 2.0).cpu().numpy()

                # Utterance-level score
                if self.include_utt and len(logits) > self.num_scales:
                    # Last element is utterance-level (batch, 2)
                    utt_logits = logits[-1]  # (1, 2)
                    utt_spoof_cos = utt_logits[0, 1]  # spoof similarity
                    utterance_score = float((utt_spoof_cos + 1.0) / 2.0)
                else:
                    utterance_score = (
                        float(np.max(frame_scores))
                        if len(frame_scores) > 0
                        else 0.0
                    )

        return DetectorOutput(
            utterance_id=utterance_id,
            frame_scores=frame_scores,
            utterance_score=utterance_score,
            frame_shift_ms=self.frame_shift_ms,
            detector_name=self.name,
        )
