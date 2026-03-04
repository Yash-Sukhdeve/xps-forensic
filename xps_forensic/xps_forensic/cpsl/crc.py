"""Conformal Risk Control (CRC) for segment-level localization.

Stage 2 of CPSL: controls temporal false negative rate (tFNR)
at the segment level with guarantee:

    E[tFNR] <= alpha_segment

The threshold lambda is calibrated by searching over a grid of candidate
thresholds and selecting the largest lambda whose average risk on the
calibration set (with a finite-sample correction term) does not exceed
alpha_segment.

The finite-sample correction follows the "learn then test" framework:

    R_hat(lambda) = (Sum_i R_i(lambda) + 1) / (n + 1)

This ensures the coverage guarantee holds marginally over the calibration
set randomness.

Reference
---------
Angelopoulos, Bates, Fisch, Lei, Schuster. "Conformal Risk Control",
ICLR 2024.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..utils.metrics import compute_tFNR, compute_tFDR


@dataclass
class ConformalRiskControl:
    """Conformal Risk Control for segment-level spoof localization.

    Parameters
    ----------
    alpha : float
        Target risk level.  E[risk_metric] <= alpha is guaranteed.
    risk_metric : str
        Risk function to control: ``"tFNR"`` (temporal false negative rate)
        or ``"tFDR"`` (temporal false discovery rate).
    threshold : float or None
        Calibrated binarization threshold.  Set by ``calibrate()``.
    """

    alpha: float = 0.10
    risk_metric: str = "tFNR"
    threshold: float | None = None
    _lambda_grid: np.ndarray = field(
        default_factory=lambda: np.linspace(0, 1, 1001), repr=False
    )

    def calibrate(
        self,
        frame_scores: list[np.ndarray],
        frame_labels: list[np.ndarray],
    ) -> None:
        """Calibrate the binarization threshold on a held-out set.

        Searches over a grid of thresholds and selects the largest
        threshold whose finite-sample-corrected average risk does not
        exceed alpha.

        Parameters
        ----------
        frame_scores : list of np.ndarray
            Per-utterance frame-level detector scores.
        frame_labels : list of np.ndarray
            Per-utterance frame-level binary ground truth (1 = spoof).
        """
        n = len(frame_scores)
        # Validate: tFNR is undefined for all-bonafide utterances.
        # Including them silently returns 0.0 and deflates risk estimates.
        for i, labels in enumerate(frame_labels):
            if self.risk_metric == "tFNR" and labels.sum() == 0:
                raise ValueError(
                    f"frame_labels[{i}] has no spoofed frames. "
                    f"tFNR is undefined for all-bonafide utterances. "
                    f"Filter calibration data to utterances with spoof regions."
                )
            if self.risk_metric == "tFDR" and labels.sum() == len(labels):
                raise ValueError(
                    f"frame_labels[{i}] has no bonafide frames. "
                    f"tFDR is undefined for all-spoofed utterances. "
                    f"Filter calibration data appropriately."
                )
        risk_fn = compute_tFNR if self.risk_metric == "tFNR" else compute_tFDR

        risks = []
        for lam in self._lambda_grid:
            risk_values = []
            for scores, labels in zip(frame_scores, frame_labels):
                pred = (scores >= lam).astype(int)
                risk_values.append(risk_fn(pred, labels))
            # Finite-sample correction: (sum + 1) / (n + 1)
            avg_risk = (sum(risk_values) + 1) / (n + 1)
            risks.append(avg_risk)

        risks = np.array(risks)
        valid = np.where(risks <= self.alpha)[0]
        if len(valid) > 0:
            # Select the largest valid threshold (most conservative binarization)
            self.threshold = float(self._lambda_grid[valid[-1]])
        else:
            # Fallback: use threshold 0 (predict everything as spoof)
            self.threshold = 0.0

    def predict(self, frame_scores: list[np.ndarray]) -> list[np.ndarray]:
        """Produce binary segment predictions using the calibrated threshold.

        Parameters
        ----------
        frame_scores : list of np.ndarray
            Per-utterance frame-level detector scores.

        Returns
        -------
        list of np.ndarray
            Per-utterance binary predictions (1 = predicted spoof).

        Raises
        ------
        RuntimeError
            If calibrate() has not been called.
        """
        if self.threshold is None:
            raise RuntimeError("Call calibrate() before predict()")
        return [(scores >= self.threshold).astype(int) for scores in frame_scores]

    def compute_empirical_risk(
        self,
        frame_scores: list[np.ndarray],
        frame_labels: list[np.ndarray],
    ) -> dict[str, float]:
        """Compute empirical risk metrics on a test set.

        Parameters
        ----------
        frame_scores : list of np.ndarray
            Per-utterance frame-level detector scores.
        frame_labels : list of np.ndarray
            Per-utterance frame-level binary ground truth.

        Returns
        -------
        dict[str, float]
            Dictionary with mean_tFNR, mean_tFDR, mean_tIoU, and threshold.
        """
        from ..utils.metrics import compute_tIoU

        preds = self.predict(frame_scores)
        tfnrs, tfdrs, tious = [], [], []
        for pred, labels in zip(preds, frame_labels):
            tfnrs.append(compute_tFNR(pred, labels))
            tfdrs.append(compute_tFDR(pred, labels))
            tious.append(compute_tIoU(pred, labels))
        return {
            "mean_tFNR": float(np.mean(tfnrs)),
            "mean_tFDR": float(np.mean(tfdrs)),
            "mean_tIoU": float(np.mean(tious)),
            "threshold": self.threshold,
        }
