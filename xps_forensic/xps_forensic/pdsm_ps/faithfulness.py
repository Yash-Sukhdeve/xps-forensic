"""Faithfulness metrics for saliency explanations.

Implements:
- Normalized AOPC (Edin et al., ACL 2025)
- Comprehensiveness/Sufficiency (DeYoung et al., ACL 2020)
- Phoneme-IoU (novel metric for PDSM-PS alignment with ground truth)
"""
from __future__ import annotations

import numpy as np


def normalized_aopc(
    original_score: float,
    perturbed_scores: list[float],
) -> float:
    """Normalized Area Over Perturbation Curve (N-AOPC).

    Measures how much the model prediction drops when top-k features
    are removed. Higher = more faithful explanation.

    Reference: Edin et al., "Are Saliency Maps Faithful?", ACL 2025.

    Args:
        original_score: Model score on original input.
        perturbed_scores: Scores after removing top-1, top-2, ... features.

    Returns:
        N-AOPC in [0, 1].
    """
    if not perturbed_scores:
        return 0.0

    K = len(perturbed_scores)
    drops = [original_score - ps for ps in perturbed_scores]
    aopc = sum(drops) / K

    max_possible = original_score
    if max_possible == 0:
        return 0.0
    return float(np.clip(aopc / max_possible, 0, 1))


def comprehensiveness(
    original_score: float,
    score_without_top_features: float,
) -> float:
    """Comprehensiveness: drop in prediction when top features removed.

    Higher = explanation captures important features.

    Reference: DeYoung et al., "ERASER: A Benchmark to Evaluate
    Rationalized NLP Models," ACL 2020.
    """
    return float(max(0, original_score - score_without_top_features))


def sufficiency(
    original_score: float,
    score_with_only_top_features: float,
) -> float:
    """Sufficiency: how much prediction is retained with only top features.

    Lower = explanation is sufficient (top features alone explain prediction).
    """
    return float(max(0, original_score - score_with_only_top_features))


def phoneme_iou(
    salient_indices: set[int],
    ground_truth_indices: set[int],
) -> float:
    """Phoneme-level Intersection over Union.

    Measures alignment between top-K salient phonemes and ground-truth
    manipulated phonemes.

    Args:
        salient_indices: Set of phoneme indices marked as salient.
        ground_truth_indices: Set of phoneme indices that are actually spoofed.

    Returns:
        IoU in [0, 1].
    """
    if not salient_indices and not ground_truth_indices:
        return 1.0
    if not salient_indices or not ground_truth_indices:
        return 0.0

    intersection = len(salient_indices & ground_truth_indices)
    union = len(salient_indices | ground_truth_indices)
    return float(intersection / union)
