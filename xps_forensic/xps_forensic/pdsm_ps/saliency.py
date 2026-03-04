"""Saliency computation for PDSM-PS.

Computes frame-level saliency attributions using:
- Integrated Gradients (Sundararajan et al., ICML 2017)
- GradSHAP (Lundberg & Lee, NeurIPS 2017)

Applied to WavLM/wav2vec2 intermediate features for frame-level attribution.

Reference: Gupta et al. (Interspeech 2024) for PDSM methodology.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import torch


def compute_saliency_mock(n_frames: int, seed: int = 42) -> np.ndarray:
    """Mock saliency for testing without a model."""
    rng = np.random.default_rng(seed)
    return rng.uniform(0, 1, n_frames).astype(np.float32)


def compute_integrated_gradients(
    model: torch.nn.Module,
    waveform: torch.Tensor,
    target_class: int = 1,
    n_steps: int = 50,
    baseline: torch.Tensor | None = None,
) -> np.ndarray:
    """Compute Integrated Gradients attribution.

    Args:
        model: Detector model with frame-level output.
        waveform: Input tensor, shape (1, n_samples).
        target_class: Class to attribute (1 = fake).
        n_steps: Number of interpolation steps.
        baseline: Reference input (default: zeros).

    Returns:
        Frame-level saliency, shape (n_frames,).
    """
    if baseline is None:
        baseline = torch.zeros_like(waveform)

    waveform.requires_grad_(True)

    scaled_inputs = [
        baseline + (float(k) / n_steps) * (waveform - baseline)
        for k in range(1, n_steps + 1)
    ]

    gradients = []
    for scaled in scaled_inputs:
        scaled = scaled.detach().requires_grad_(True)
        output = model(scaled)

        if isinstance(output, dict):
            logits = output.get("frame_logits", output.get("logits"))
        elif isinstance(output, tuple):
            logits = output[0]
        else:
            logits = output

        if logits.dim() == 3:
            target_sum = logits[0, :, target_class].sum()
        else:
            target_sum = logits[0].sum()

        target_sum.backward()
        gradients.append(scaled.grad.detach().clone())
        scaled.grad = None

    avg_grad = torch.stack(gradients).mean(dim=0)
    ig = (waveform - baseline) * avg_grad

    samples_per_frame = 320
    n_samples = ig.shape[-1]
    n_frames = n_samples // samples_per_frame

    ig_flat = ig.squeeze().detach().cpu().numpy()
    frame_saliency = np.array([
        np.mean(np.abs(ig_flat[i * samples_per_frame:(i + 1) * samples_per_frame]))
        for i in range(n_frames)
    ])

    return frame_saliency


def compute_gradshap(
    model: torch.nn.Module,
    waveform: torch.Tensor,
    target_class: int = 1,
    n_samples: int = 25,
    seed: int = 42,
) -> np.ndarray:
    """Compute GradSHAP attribution.

    GradSHAP ≈ expected value of Integrated Gradients over random baselines.

    Args:
        model: Detector model.
        waveform: Input tensor, shape (1, n_samples).
        target_class: Class to attribute.
        n_samples: Number of random baseline samples.
        seed: Random seed.

    Returns:
        Frame-level saliency, shape (n_frames,).
    """
    torch.manual_seed(seed)

    attributions = []
    for _ in range(n_samples):
        baseline = torch.randn_like(waveform) * 0.01
        ig = compute_integrated_gradients(
            model, waveform, target_class, n_steps=10, baseline=baseline
        )
        attributions.append(ig)

    return np.mean(attributions, axis=0)
