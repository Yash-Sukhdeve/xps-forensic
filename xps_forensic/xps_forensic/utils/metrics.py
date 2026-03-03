"""Core evaluation metrics for spoof detection and localization.

Implements:
- Equal Error Rate (EER) computation via ROC interpolation
- Segment-level EER at configurable temporal resolution
- Segment F1 score
- Temporal False Negative Rate (tFNR)
- Temporal False Discovery Rate (tFDR)
- Temporal Intersection over Union (tIoU)

References
----------
Kinnunen et al., "t-DCF: a Detection Cost Function for the Tandem
Assessment of Spoofing Countermeasures and Automatic Speaker Verification",
Proc. Odyssey 2018.

Zhang et al., "The PartialSpoof Database and Countermeasures for the
Detection of Short Fake Speech Segments Embedded in Real Speech",
IEEE/ACM TASLP, 2023.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve, f1_score


def compute_eer(
    scores: np.ndarray,
    labels: np.ndarray,
) -> tuple[float, float]:
    """Compute the Equal Error Rate and the corresponding threshold.

    Parameters
    ----------
    scores : array-like, shape (n_samples,)
        Detector scores.  Higher values indicate *spoof* (positive class).
    labels : array-like, shape (n_samples,)
        Binary ground truth: 1 = spoof, 0 = bonafide.

    Returns
    -------
    eer : float
        Equal error rate in [0, 1].
    threshold : float
        Score threshold at the EER operating point.
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)

    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1.0 - tpr

    # Interpolate to find the crossing point where FPR == FNR
    try:
        eer = brentq(lambda x: interp1d(fpr, fnr)(x) - x, 0.0, 1.0)
    except ValueError:
        # Fallback: pick the point closest to FPR == FNR
        idx = np.nanargmin(np.abs(fpr - fnr))
        eer = float((fpr[idx] + fnr[idx]) / 2.0)

    # Threshold at EER: find the threshold closest to the EER operating point
    diff = np.abs(fpr - fnr)
    idx_best = np.argmin(diff)
    threshold = float(thresholds[idx_best])

    return float(eer), threshold


def compute_segment_eer(
    frame_scores: np.ndarray,
    frame_labels: np.ndarray,
    resolution_ms: float,
    frame_shift_ms: float = 20.0,
) -> tuple[float, float]:
    """Compute segment-level EER at a given temporal resolution.

    Frames are aggregated into non-overlapping segments of
    ``resolution_ms / frame_shift_ms`` frames, using mean pooling for
    scores and majority vote for labels.

    Parameters
    ----------
    frame_scores : array-like, shape (n_frames,)
        Per-frame spoof scores.
    frame_labels : array-like, shape (n_frames,)
        Per-frame binary labels (1 = spoof).
    resolution_ms : float
        Segment duration in milliseconds.
    frame_shift_ms : float
        Frame shift in milliseconds (default 10 ms).

    Returns
    -------
    eer : float
    threshold : float
    """
    frame_scores = np.asarray(frame_scores, dtype=float)
    frame_labels = np.asarray(frame_labels, dtype=int)

    seg_len = max(1, int(round(resolution_ms / frame_shift_ms)))
    n_frames = len(frame_scores)
    n_segments = n_frames // seg_len

    if n_segments == 0:
        return compute_eer(frame_scores, frame_labels)

    # Truncate to exact multiples
    scores_trunc = frame_scores[: n_segments * seg_len].reshape(n_segments, seg_len)
    labels_trunc = frame_labels[: n_segments * seg_len].reshape(n_segments, seg_len)

    seg_scores = scores_trunc.mean(axis=1)
    seg_labels = (labels_trunc.mean(axis=1) >= 0.5).astype(int)

    return compute_eer(seg_scores, seg_labels)


def compute_segment_f1(
    frame_preds: np.ndarray,
    frame_labels: np.ndarray,
) -> float:
    """Compute frame-level F1 score for spoof localization.

    Parameters
    ----------
    frame_preds : array-like, shape (n_frames,)
        Binary frame-level predictions (1 = spoof).
    frame_labels : array-like, shape (n_frames,)
        Binary frame-level ground truth (1 = spoof).

    Returns
    -------
    float
        F1 score.  Returns 1.0 when both predictions and labels are all-zero
        (no spoof to detect and none predicted).
    """
    frame_preds = np.asarray(frame_preds, dtype=int)
    frame_labels = np.asarray(frame_labels, dtype=int)

    if frame_preds.sum() == 0 and frame_labels.sum() == 0:
        return 1.0

    return float(f1_score(frame_labels, frame_preds, zero_division=0.0))


def compute_tFNR(
    pred_binary: np.ndarray,
    true_binary: np.ndarray,
) -> float:
    """Temporal False Negative Rate.

    Fraction of truly spoofed frames that are *not* detected.

    Parameters
    ----------
    pred_binary : array-like, shape (n_frames,)
        Binary predictions (1 = predicted spoof).
    true_binary : array-like, shape (n_frames,)
        Binary ground truth (1 = actual spoof).

    Returns
    -------
    float
        tFNR in [0, 1].  Returns 0.0 if there are no spoofed frames
        in the ground truth.
    """
    pred_binary = np.asarray(pred_binary, dtype=int)
    true_binary = np.asarray(true_binary, dtype=int)

    total_spoof = true_binary.sum()
    if total_spoof == 0:
        return 0.0

    missed = np.sum((true_binary == 1) & (pred_binary == 0))
    return float(missed / total_spoof)


def compute_tFDR(
    pred_binary: np.ndarray,
    true_binary: np.ndarray,
) -> float:
    """Temporal False Discovery Rate.

    Fraction of predicted-spoof frames that are actually bonafide.

    Parameters
    ----------
    pred_binary : array-like, shape (n_frames,)
        Binary predictions (1 = predicted spoof).
    true_binary : array-like, shape (n_frames,)
        Binary ground truth (1 = actual spoof).

    Returns
    -------
    float
        tFDR in [0, 1].  Returns 0.0 if no frames are predicted as spoof.
    """
    pred_binary = np.asarray(pred_binary, dtype=int)
    true_binary = np.asarray(true_binary, dtype=int)

    total_pred = pred_binary.sum()
    if total_pred == 0:
        return 0.0

    false_alarms = np.sum((pred_binary == 1) & (true_binary == 0))
    return float(false_alarms / total_pred)


def compute_tIoU(
    pred_binary: np.ndarray,
    true_binary: np.ndarray,
) -> float:
    """Temporal Intersection over Union.

    Measures the overlap between predicted and true spoof regions.

    Parameters
    ----------
    pred_binary : array-like, shape (n_frames,)
        Binary predictions (1 = predicted spoof).
    true_binary : array-like, shape (n_frames,)
        Binary ground truth (1 = actual spoof).

    Returns
    -------
    float
        tIoU in [0, 1].  Returns 1.0 if both are all-zero (no spoof present
        and none predicted).  Returns 0.0 if union is non-empty but
        intersection is empty.
    """
    pred_binary = np.asarray(pred_binary, dtype=int)
    true_binary = np.asarray(true_binary, dtype=int)

    intersection = np.sum((pred_binary == 1) & (true_binary == 1))
    union = np.sum((pred_binary == 1) | (true_binary == 1))

    if union == 0:
        return 1.0

    return float(intersection / union)
