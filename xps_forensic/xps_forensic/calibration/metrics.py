"""Calibration evaluation metrics.

Reference: Guo et al., "On Calibration of Modern Neural Networks,"
ICML 2017; Dimitri et al., 2025.
"""
from __future__ import annotations

import numpy as np


def expected_calibration_error(
    scores: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
) -> float:
    """Expected Calibration Error (ECE).

    Args:
        scores: Predicted probabilities, shape (n,).
        labels: Binary labels, shape (n,).
        n_bins: Number of equal-width bins.

    Returns:
        ECE value in [0, 1].
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(scores)

    for i in range(n_bins):
        mask = (scores > bin_edges[i]) & (scores <= bin_edges[i + 1])
        if i == 0:
            mask = (scores >= bin_edges[i]) & (scores <= bin_edges[i + 1])
        count = mask.sum()
        if count == 0:
            continue
        avg_conf = scores[mask].mean()
        avg_acc = labels[mask].mean()
        ece += (count / n) * abs(avg_acc - avg_conf)

    return float(ece)


def brier_score(scores: np.ndarray, labels: np.ndarray) -> float:
    """Brier score: mean squared error of probability estimates."""
    return float(np.mean((scores - labels) ** 2))


def negative_log_likelihood(
    scores: np.ndarray, labels: np.ndarray
) -> float:
    """Negative log-likelihood (cross-entropy loss)."""
    eps = 1e-7
    clipped = np.clip(scores, eps, 1 - eps)
    nll = -np.mean(
        labels * np.log(clipped) + (1 - labels) * np.log(1 - clipped)
    )
    return float(nll)


def reliability_diagram_data(
    scores: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute data for reliability diagram.

    Returns:
        (bin_midpoints, accuracies, confidences, counts)
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    midpoints = (bin_edges[:-1] + bin_edges[1:]) / 2
    accs = np.zeros(n_bins)
    confs = np.zeros(n_bins)
    counts = np.zeros(n_bins, dtype=int)

    for i in range(n_bins):
        if i == 0:
            mask = (scores >= bin_edges[i]) & (scores <= bin_edges[i + 1])
        else:
            mask = (scores > bin_edges[i]) & (scores <= bin_edges[i + 1])
        count = mask.sum()
        counts[i] = count
        if count > 0:
            accs[i] = labels[mask].mean()
            confs[i] = scores[mask].mean()

    return midpoints, accs, confs, counts
