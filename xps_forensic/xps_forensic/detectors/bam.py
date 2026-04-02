"""BAM detector wrapper.

Reference: Zhong, J., Li, B., & Yi, J. (2024). "Enhancing Partially Spoofed
Audio Localization with Boundary-aware Attention Mechanism." In Interspeech
2024, pp. 4838-4842. doi:10.21437/Interspeech.2024-587

Wraps the official BAM implementation from:
https://github.com/media-sec-lab/BAM

IMPORTANT: This is a READ-ONLY wrapper. The external BAM source code must NOT
be modified. All adaptation logic lives in this wrapper.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

from .base import BaseDetector, DetectorOutput

logger = logging.getLogger(__name__)

# Default BAM config values from config/bam_wavlm.yaml
_DEFAULT_BAM_CONFIG = {
    "ssl_name": "wavlm_local",
    "ssl_ckpt": "wavlm_large.pt",
    "ssl_feat_dim": 1024,
    "embed_dim": 1024,
    "pool_head_num": 1,
    "local_channel_dim": 32,
    "gap_head_num": 1,
    "gap_layer_num": 2,
}


class BAMDetector(BaseDetector):
    """Wrapper for BAM (Boundary-aware Attention Mechanism) detector.

    BAM uses WavLM features with a boundary-aware attention mechanism
    for frame-level partial spoof localization. The model outputs
    per-segment 2-class logits at 160ms resolution (8 WavLM frames pooled).

    The forward pass returns ``(output, b_pred)`` where:
    - ``output``: ``(batch, n_segments, 2)`` logits (class 0=real, 1=spoof)
    - ``b_pred``: ``(batch, n_segments)`` boundary sigmoid scores

    Checkpoint format: PyTorch Lightning ``.ckpt`` with state dict keys
    prefixed by ``model.`` (from ``LightingModelWrapper``).

    Args:
        checkpoint: Path to pretrained model weights (.ckpt).
        external_dir: Path to the cloned BAM repository root.
        device: Torch device string (default "cpu").
        ssl_ckpt: Path to WavLM checkpoint (wavlm_large.pt). If None,
            uses the default path from BAM config.
        resolution: Segment resolution in seconds (default 0.16).
    """

    name = "BAM"
    frame_shift_ms = 160  # BAM pools 8 SSL frames (8 * 20ms = 160ms)

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
        """Load BAM model from external repo + checkpoint.

        Inserts the external BAM repo into sys.path and imports the
        model class. Constructs the required ``args`` and ``config``
        namespace objects per the BAM constructor signature.

        For Lightning checkpoints, strips the ``model.`` prefix from
        state dict keys before loading into the BAM module.

        Raises:
            ValueError: If external_dir is not set or does not exist.
            FileNotFoundError: If checkpoint path does not exist.
        """
        if self.external_dir is None:
            raise ValueError("external_dir must point to cloned BAM repo")
        if not self.external_dir.is_dir():
            raise FileNotFoundError(
                f"BAM external directory not found: {self.external_dir}"
            )

        bam_path = str(self.external_dir)
        if bam_path not in sys.path:
            sys.path.insert(0, bam_path)

        from models.bam import BAM as BAMModel  # noqa: E402

        # Build args namespace (BAM constructor reads args.resolution)
        args = SimpleNamespace(resolution=self.resolution)

        # Build config namespace from defaults
        config_dict = dict(_DEFAULT_BAM_CONFIG)
        if self.ssl_ckpt is not None:
            config_dict["ssl_ckpt"] = self.ssl_ckpt
        config = SimpleNamespace(**config_dict)

        self.model = BAMModel(args, config)

        if self.checkpoint:
            ckpt_path = Path(self.checkpoint)
            if not ckpt_path.is_file():
                raise FileNotFoundError(
                    f"BAM checkpoint not found: {ckpt_path}"
                )
            # Lightning checkpoints may be nested zip archives
            # (outer zip contains hparams.yaml + model.ckpt inner file).
            # PyTorch 2.6+ rejects the outer zip; extract inner file first.
            import zipfile
            import io
            try:
                state = torch.load(
                    str(ckpt_path), map_location=self.device, weights_only=False
                )
            except RuntimeError:
                logger.info("Extracting inner checkpoint from Lightning zip archive")
                with zipfile.ZipFile(str(ckpt_path), "r") as zf:
                    inner_names = [n for n in zf.namelist() if n.endswith(".ckpt")]
                    inner_name = inner_names[0] if inner_names else zf.namelist()[0]
                    with zf.open(inner_name) as f:
                        buf = io.BytesIO(f.read())
                state = torch.load(
                    buf, map_location=self.device, weights_only=False
                )
            # Lightning .ckpt files store weights under 'state_dict' key
            # with keys prefixed by 'model.'
            if "state_dict" in state:
                raw_sd = state["state_dict"]
                cleaned_sd = {}
                prefix = "model."
                for k, v in raw_sd.items():
                    if k.startswith(prefix):
                        cleaned_sd[k[len(prefix):]] = v
                    else:
                        cleaned_sd[k] = v
                self.model.load_state_dict(cleaned_sd)
                logger.info(
                    "Loaded BAM weights from Lightning checkpoint "
                    "(%d parameters)", len(cleaned_sd)
                )
            else:
                # Plain state dict (non-Lightning)
                self.model.load_state_dict(state)
                logger.info("Loaded BAM weights from plain state dict")

        self.model.to(self.device)
        self.model.eval()

    def predict(
        self,
        waveform: np.ndarray,
        sample_rate: int = 16000,
        utterance_id: str = "",
    ) -> DetectorOutput:
        """Run BAM inference on a single waveform.

        The BAM forward pass returns ``(output, b_pred)`` where output
        has shape ``(batch, n_segments, 2)``. We apply softmax over the
        last dimension and take class index 1 (spoof probability) as
        the per-segment score.

        Note: BAM internally pads the input by 256 samples, so no
        external padding is needed.

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

            # BAM pools SSL features in groups of pool_frame_num (8 for 160ms).
            # If the SSL output length isn't divisible by 8, the model crashes.
            # Probe the SSL layer to get exact feature count, then pad if needed.
            pool_n = int(self.resolution / 0.02)  # 8
            x_probe = F.pad(x, (0, 256), mode='constant', value=0)
            ssl_out = self.model.ssl_layer(x_probe)["hidden_states"][-1]
            n_frames = ssl_out.shape[1]
            remainder = n_frames % pool_n
            if remainder != 0:
                # Pad waveform so SSL produces pool_n-aligned frames
                pad_frames = pool_n - remainder
                pad_samples = pad_frames * 320  # SSL stride = 320
                x = F.pad(x, (0, pad_samples))

            output, _b_pred = self.model(x)

            # output shape: (1, n_segments, 2) — apply softmax for probs
            probs = F.softmax(output, dim=-1)
            # BAM label convention: class 0 = spoof, class 1 = bonafide
            # (PartialSpoof raw labels: '0' = spoof, '1' = bonafide,
            #  used directly as CrossEntropyLoss targets by BAM)
            # XPS convention: higher score = more likely fake
            frame_scores = probs[0, :, 0].cpu().numpy()

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
