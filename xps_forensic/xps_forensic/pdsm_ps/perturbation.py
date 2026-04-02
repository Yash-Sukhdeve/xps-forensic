"""Perturbation utilities for faithfulness evaluation.

Bridges detector wrappers with PDSM-PS faithfulness metrics by:
1. Computing saliency from a detector + waveform
2. Perturbing phoneme regions and re-running inference
3. Collecting score drops for N-AOPC/comprehensiveness/sufficiency

Reference: DeYoung et al., "ERASER: A Benchmark to Evaluate Rationalized
NLP Models", ACL 2020.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from xps_forensic.detectors.base import BaseDetector
    from xps_forensic.pdsm_ps.alignment import PhonemeSegment

logger = logging.getLogger(__name__)


def compute_saliency_from_detector(
    detector: "BaseDetector",
    waveform: np.ndarray,
    sample_rate: int = 16000,
    method: str = "ig",
    n_steps: int = 50,
) -> np.ndarray:
    """Compute frame-level saliency from a detector wrapper.

    Uses the hand-rolled IG/GradSHAP from saliency.py. Attempts Captum
    LayerIntegratedGradients first; falls back to manual implementation.

    Args:
        detector: A loaded BaseDetector instance.
        waveform: 1D numpy array of audio samples.
        sample_rate: Audio sample rate.
        method: "ig" for Integrated Gradients, "gradshap" for GradSHAP.
        n_steps: Interpolation steps for IG.

    Returns:
        Frame-level saliency array, shape (n_frames,).
    """
    import torch

    if not hasattr(detector, "_model") or detector._model is None:
        raise RuntimeError("Detector model not loaded. Call load_model() first.")

    wav_tensor = torch.from_numpy(waveform).float().unsqueeze(0)
    device = next(detector._model.parameters()).device
    wav_tensor = wav_tensor.to(device)

    # Try Captum first
    try:
        from captum.attr import IntegratedGradients

        # Wrap detector model for Captum: input → spoof score
        class _DetectorForward(torch.nn.Module):
            def __init__(self, det):
                super().__init__()
                self.det = det

            def forward(self, x):
                output = self.det._model(x)
                if isinstance(output, tuple):
                    logits = output[0]
                elif isinstance(output, dict):
                    logits = output.get("frame_logits", output.get("logits", output))
                else:
                    logits = output
                if logits.dim() == 3:
                    return logits[0, :, 1].sum().unsqueeze(0)
                return logits.sum().unsqueeze(0)

        wrapper = _DetectorForward(detector)
        ig = IntegratedGradients(wrapper)
        baseline = torch.zeros_like(wav_tensor)
        attr = ig.attribute(wav_tensor, baseline, n_steps=n_steps)

        # Aggregate to frames
        attr_np = attr.squeeze().detach().cpu().numpy()
        samples_per_frame = sample_rate * detector.frame_shift_ms // 1000
        n_frames = len(attr_np) // samples_per_frame
        frame_saliency = np.array([
            np.mean(np.abs(attr_np[i * samples_per_frame:(i + 1) * samples_per_frame]))
            for i in range(n_frames)
        ])
        logger.info("Saliency computed via Captum IG (%d frames)", n_frames)
        return frame_saliency

    except (ImportError, Exception) as exc:
        logger.info("Captum unavailable (%s), using manual IG", exc)

    # Fallback: manual IG from saliency.py
    from xps_forensic.pdsm_ps.saliency import compute_integrated_gradients, compute_gradshap

    if method == "gradshap":
        return compute_gradshap(
            detector._model, wav_tensor, target_class=1, n_samples=25
        )
    return compute_integrated_gradients(
        detector._model, wav_tensor, target_class=1,
        n_steps=n_steps, sample_rate=sample_rate,
        frame_shift_ms=detector.frame_shift_ms,
    )


def perturb_and_score(
    detector: "BaseDetector",
    waveform: np.ndarray,
    sample_rate: int,
    phoneme_segments: list["PhonemeSegment"],
    ranked_indices: list[int],
    perturbation: str = "silence",
) -> list[float]:
    """Perturb top-k phonemes cumulatively and collect scores.

    Args:
        detector: Loaded detector.
        waveform: Original audio (1D numpy).
        sample_rate: Audio sample rate.
        phoneme_segments: List of PhonemeSegment with start_sec/end_sec.
        ranked_indices: Phoneme indices sorted by saliency (most salient first).
        perturbation: "silence" (zeros) or "noise" (Gaussian).

    Returns:
        List of utterance scores after removing top-1, top-2, ..., top-K phonemes.
    """
    scores = []
    perturbed = waveform.copy()

    for k, idx in enumerate(ranked_indices):
        seg = phoneme_segments[idx]
        start_sample = int(seg.start_sec * sample_rate)
        end_sample = min(int(seg.end_sec * sample_rate), len(perturbed))

        if perturbation == "silence":
            perturbed[start_sample:end_sample] = 0.0
        elif perturbation == "noise":
            perturbed[start_sample:end_sample] = np.random.randn(
                end_sample - start_sample
            ) * 0.001

        output = detector.predict(perturbed, sample_rate)
        scores.append(output.utterance_score)

    return scores


def compute_faithfulness_suite(
    detector: "BaseDetector",
    waveform: np.ndarray,
    sample_rate: int,
    frame_saliency: np.ndarray,
    phoneme_segments: list["PhonemeSegment"],
    phoneme_saliencies: list[float],
    ground_truth_mask: np.ndarray | None = None,
    top_k: int = 10,
) -> dict[str, float]:
    """Compute full faithfulness metrics for one utterance.

    Args:
        detector: Loaded detector.
        waveform: Original audio (1D numpy).
        sample_rate: Audio sample rate.
        frame_saliency: Frame-level saliency from compute_saliency_from_detector.
        phoneme_segments: Phoneme boundaries.
        phoneme_saliencies: Per-phoneme mean saliency values.
        ground_truth_mask: Binary frame mask for Phoneme-IoU (optional).
        top_k: Number of top phonemes to evaluate.

    Returns:
        Dict with n_aopc, comprehensiveness, sufficiency, phoneme_iou.
    """
    from xps_forensic.pdsm_ps.faithfulness import (
        normalized_aopc,
        comprehensiveness as comp_fn,
        sufficiency as suff_fn,
        phoneme_iou,
    )

    # Rank phonemes by saliency (most salient first)
    ranked = np.argsort(phoneme_saliencies)[::-1].tolist()
    top_indices = ranked[:top_k]

    # Get original score
    original_output = detector.predict(waveform, sample_rate)
    original_score = original_output.utterance_score

    # Perturb top-k cumulatively
    perturbed_scores = perturb_and_score(
        detector, waveform, sample_rate,
        phoneme_segments, top_indices, perturbation="silence",
    )

    result = {
        "n_aopc": normalized_aopc(original_score, perturbed_scores),
        "comprehensiveness": comp_fn(original_score, perturbed_scores[-1]) if perturbed_scores else 0.0,
    }

    # Sufficiency: keep ONLY top-k, silence everything else
    sufficient_wav = np.zeros_like(waveform)
    for idx in top_indices:
        seg = phoneme_segments[idx]
        s = int(seg.start_sec * sample_rate)
        e = min(int(seg.end_sec * sample_rate), len(waveform))
        sufficient_wav[s:e] = waveform[s:e]

    suff_output = detector.predict(sufficient_wav, sample_rate)
    result["sufficiency"] = suff_fn(original_score, suff_output.utterance_score)

    # Phoneme-IoU if ground truth available
    if ground_truth_mask is not None:
        result["phoneme_iou"] = phoneme_iou(
            phoneme_segments, top_indices, ground_truth_mask,
            frame_shift_ms=detector.frame_shift_ms,
        )

    return result
