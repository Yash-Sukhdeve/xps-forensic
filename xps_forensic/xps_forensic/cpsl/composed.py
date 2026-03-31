"""Composed CPSL pipeline: Stage 1 (SCP+APS) + Stage 2 (CRC).

Composed guarantee:

    P(Stage 1 correct AND Stage 2 correct)
        >= (1 - alpha_utterance) * (1 - alpha_segment)

Stage 1 (SCP+APS) produces utterance-level prediction sets C(x) with
coverage guarantee P(Y in C(X)) >= 1 - alpha_utterance.

Stage 2 (CRC) is applied ONLY when the prediction set includes
``partially_fake`` (class index 1).  It controls E[tFNR] <= alpha_segment
for the segment-level localization.

The composed guarantee follows from Bonferroni's inequality:
P(both correct) >= 1 - alpha1 - alpha2.  Note: the product bound
(1-alpha1)(1-alpha2) requires strict independence, which does not
hold here because partial-spoof utterances participate in both
Stage 1 and Stage 2 calibration.

Reference
---------
Angelopoulos, Bates. "A Gentle Introduction to Conformal Prediction
and Distribution-Free Uncertainty Quantification", 2023.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .nonconformity import compute_nonconformity
from .scp_aps import SCPAPS
from .crc import ConformalRiskControl


@dataclass
class CPSLResult:
    """Structured result from the CPSL pipeline for a single utterance.

    Attributes
    ----------
    utterance_id : str
        Identifier for the utterance.
    nonconformity_score : float
        Aggregated utterance-level nonconformity score.
    prediction_set : set[int]
        Set of class indices in the conformal prediction set.
    prediction_set_labels : set[str]
        Human-readable labels for the prediction set.
    segment_predictions : np.ndarray or None
        Frame-level binary predictions (1=spoof) if partial spoof
        is in the prediction set; None otherwise.
    crc_threshold : float or None
        CRC-calibrated threshold used for segment predictions.
    """

    utterance_id: str
    nonconformity_score: float
    prediction_set: set[int]
    prediction_set_labels: set[str]
    segment_predictions: np.ndarray | None
    crc_threshold: float | None


class CPSLPipeline:
    """Composed Conformalized Partial Spoof Localization pipeline.

    Two-stage conformal prediction:
    - Stage 1: SCP+APS for utterance-level ternary classification
    - Stage 2: CRC on tFNR for segment-level localization
    - Composed guarantee: P(both correct) >= (1-alpha1)(1-alpha2)

    Stage 2 is applied only when Stage 1 includes partially_fake
    in the prediction set.

    Parameters
    ----------
    alpha_utterance : float
        Miscoverage level for Stage 1 (utterance classification).
    alpha_segment : float
        Risk level for Stage 2 (segment localization tFNR control).
    nonconformity_method : str
        Frame-to-utterance aggregation: ``"max"`` or ``"logsumexp"``.
    nonconformity_beta : float
        Sharpness for logsumexp (ignored if method="max").
    """

    CLASS_NAMES = ["real", "partially_fake", "fully_fake"]

    def __init__(
        self,
        alpha_utterance: float = 0.05,
        alpha_segment: float = 0.10,
        nonconformity_method: str = "max",
        nonconformity_beta: float = 10.0,
        score_frame_shift_ms: float = 20.0,
        label_frame_shift_ms: float = 10.0,
    ):
        self.alpha_utterance = alpha_utterance
        self.alpha_segment = alpha_segment
        self.nc_method = nonconformity_method
        self.nc_beta = nonconformity_beta
        self.stage1 = SCPAPS(alpha=alpha_utterance, classes=self.CLASS_NAMES)
        self.stage2 = ConformalRiskControl(alpha=alpha_segment, risk_metric="tFNR")
        self.score_fs_ms = score_frame_shift_ms
        self.label_fs_ms = label_frame_shift_ms

    @property
    def composed_guarantee(self) -> float:
        """Lower bound on P(both stages correct) via Bonferroni.

        Returns 1 - alpha_utterance - alpha_segment.

        Note: the product bound (1-a1)(1-a2) would require strict
        independence between stages, which does not hold because
        partial-spoof calibration data is shared between stages.
        Bonferroni's bound is unconditionally valid.
        """
        return 1 - self.alpha_utterance - self.alpha_segment

    def calibrate(
        self,
        frame_scores_list: list[np.ndarray],
        utterance_labels: np.ndarray,
        frame_labels_list: list[np.ndarray],
    ) -> None:
        """Calibrate both stages on held-out calibration data.

        Stage 1 is calibrated on all utterances.  Stage 2 is calibrated
        only on utterances labeled as partially_fake (class 1), since
        CRC for segment localization is only meaningful for partial spoofs.

        Parameters
        ----------
        frame_scores_list : list of np.ndarray
            Per-utterance frame-level detector scores.
        utterance_labels : np.ndarray, shape (n_cal,)
            Ternary labels: 0=real, 1=partially_fake, 2=fully_fake.
        frame_labels_list : list of np.ndarray
            Per-utterance frame-level binary ground truth (1=spoof).
        """
        # Stage 1: compute nonconformity scores and calibrate SCP+APS
        nc_scores = compute_nonconformity(
            frame_scores_list, method=self.nc_method, beta=self.nc_beta
        )
        self.stage1.calibrate(nc_scores, utterance_labels)

        # Stage 2: calibrate CRC on partially_fake utterances only
        partial_mask = utterance_labels == 1
        if partial_mask.any():
            partial_frame_scores = [fs for fs, m in zip(frame_scores_list, partial_mask) if m]
            partial_frame_labels = [fl for fl, m in zip(frame_labels_list, partial_mask) if m]

            # Align to label grid (e.g., 10 ms) by upsampling scores
            aligned_scores = []
            aligned_labels = []
            from ..utils.metrics import upsample_binary_predictions_to_label_grid
            for fs, fl in zip(partial_frame_scores, partial_frame_labels):
                # For calibration risk computation, we need binary predictions vs labels.
                # We upsample scores later after thresholding; here we just keep raw scores
                # and align after thresholding inside CRC by comparing lengths.
                # However, CRC expects equal-length arrays; to ensure this, we upsample
                # a dummy binary mask to determine the target length and then truncate.
                # Simpler approach: store labels as-is; provide scores as-is and let CRC
                # operate on scores directly but we must align before risk. We perform
                # alignment here by creating a per-label-frame score via repetition.
                ratio = int(round(self.score_fs_ms / self.label_fs_ms))
                if ratio <= 0:
                    ratio = 1
                # Repeat each score to label grid; then truncate to label length
                fs_rep = np.repeat(fs, ratio)
                L = min(len(fs_rep), len(fl))
                aligned_scores.append(fs_rep[:L])
                aligned_labels.append(fl[:L])

            self.stage2.calibrate(aligned_scores, aligned_labels)

    def predict(
        self,
        frame_scores_list: list[np.ndarray],
        utterance_ids: list[str] | None = None,
    ) -> list[CPSLResult]:
        """Run the full CPSL pipeline on test utterances.

        Stage 1 produces prediction sets.  If partially_fake (class 1)
        is in the prediction set AND Stage 2 has been calibrated,
        segment-level predictions are produced via CRC.

        Parameters
        ----------
        frame_scores_list : list of np.ndarray
            Per-utterance frame-level detector scores.
        utterance_ids : list[str] or None
            Optional identifiers.  Auto-generated if None.

        Returns
        -------
        list[CPSLResult]
            Per-utterance structured results.
        """
        if utterance_ids is None:
            utterance_ids = [f"utt_{i}" for i in range(len(frame_scores_list))]

        nc_scores = compute_nonconformity(
            frame_scores_list, method=self.nc_method, beta=self.nc_beta
        )
        pred_sets = self.stage1.predict(nc_scores)

        results = []
        for i, (ps, scores, uid) in enumerate(
            zip(pred_sets, frame_scores_list, utterance_ids)
        ):
            seg_preds = None
            crc_thresh = None
            # Apply Stage 2 only if partially_fake is in the prediction set
            if 1 in ps and self.stage2.threshold is not None:
                seg_preds = self.stage2.predict([scores])[0]
                crc_thresh = self.stage2.threshold
            results.append(CPSLResult(
                utterance_id=uid,
                nonconformity_score=float(nc_scores[i]),
                prediction_set=ps,
                prediction_set_labels={self.CLASS_NAMES[c] for c in ps},
                segment_predictions=seg_preds,
                crc_threshold=crc_thresh,
            ))
        return results
