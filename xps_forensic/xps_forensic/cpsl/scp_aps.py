"""Split Conformal Prediction (Mondrian/class-conditional) for Stage 1.

This module implements class-conditional ("Mondrian") split conformal
prediction for utterance-level ternary classification with labels in
{real, partially_fake, fully_fake}. Given a single nonconformity score
per utterance s(x), it computes class-conditional quantiles on a
calibration set and produces prediction sets C(x) satisfying the
marginal coverage guarantee:

    P(Y in C(X)) >= 1 - alpha.

Notes
-----
- This is NOT Adaptive Prediction Sets (APS) of Romano, Sesia, Candès
  (NeurIPS 2020), which constructs adaptive sets from per-class scores.
  We instead apply standard split conformal with class-conditional
  thresholds (Mondrian CP). See Angelopoulos & Bates (2023) for a
  tutorial on conformal prediction and Mondrian conditioning.

- Conformal prediction is applied at the utterance level ONLY. Frame-level
  CP would violate the exchangeability assumption due to temporal
  autocorrelation in audio. Stage 2 handles localization via CRC.

References
----------
- Angelopoulos, Bates. "A Gentle Introduction to Conformal Prediction and
  Distribution-Free Uncertainty Quantification", 2023.
  https://arxiv.org/abs/2302.08112
- Romano, Sesia, Candès. "Classification with Valid and Adaptive Coverage",
  NeurIPS 2020. https://arxiv.org/abs/2009.14193
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class SCPAPS:
    """Split Conformal Prediction (Mondrian/class-conditional).

    Parameters
    ----------
    alpha : float
        Miscoverage level.  Prediction sets satisfy P(Y in C(X)) >= 1-alpha.
    classes : list[str]
        Class names for the ternary classification.
    """

    alpha: float = 0.05
    classes: list[str] = field(
        default_factory=lambda: ["real", "partially_fake", "fully_fake"]
    )
    _quantiles: dict[int, float] = field(default_factory=dict, repr=False)
    _class_conditional: bool = True
    enforce_contiguity: bool = True

    @property
    def n_classes(self) -> int:
        """Number of classes."""
        return len(self.classes)

    def calibrate(
        self, nonconformity_scores: np.ndarray, labels: np.ndarray
    ) -> None:
        """Compute conformal quantiles from calibration data.

        Uses class-conditional calibration: for each class c, computes
        the ceil((n_c+1)(1-alpha))/n_c quantile of scores in that class.

        Parameters
        ----------
        nonconformity_scores : np.ndarray, shape (n_cal,)
            Nonconformity scores for calibration examples.
        labels : np.ndarray, shape (n_cal,)
            Integer class labels (0, 1, 2) for calibration examples.
        """
        unique = set(np.unique(labels))
        valid = set(range(self.n_classes))
        if not unique.issubset(valid):
            raise ValueError(
                f"Labels must be in {valid}, got {unique}"
            )

        if self._class_conditional:
            for c in range(self.n_classes):
                mask = labels == c
                if mask.sum() == 0:
                    self._quantiles[c] = 1.0
                    continue
                class_scores = nonconformity_scores[mask]
                n_c = len(class_scores)
                # Finite-sample correction: ceil((n+1)(1-alpha))/n
                q_level = np.ceil((n_c + 1) * (1 - self.alpha)) / n_c
                q_level = min(q_level, 1.0)
                self._quantiles[c] = float(np.quantile(class_scores, q_level))
        else:
            n = len(nonconformity_scores)
            q_level = np.ceil((n + 1) * (1 - self.alpha)) / n
            q_level = min(q_level, 1.0)
            q = float(np.quantile(nonconformity_scores, q_level))
            for c in range(self.n_classes):
                self._quantiles[c] = q

    def predict(self, nonconformity_scores: np.ndarray) -> list[set[int]]:
        """Produce prediction sets for test examples.

        For each test score, includes class c in the prediction set if
        score <= quantile[c].  If the resulting set is empty (extreme
        score), includes the class with the highest quantile as a
        fallback to guarantee non-empty sets.

        Parameters
        ----------
        nonconformity_scores : np.ndarray, shape (n_test,)
            Nonconformity scores for test examples.

        Returns
        -------
        list[set[int]]
            Prediction sets, each a set of integer class indices.

        Raises
        ------
        RuntimeError
            If calibrate() has not been called.
        """
        if not self._quantiles:
            raise RuntimeError("Call calibrate() before predict()")

        prediction_sets = []
        for score in nonconformity_scores:
            ps = set()
            for c in range(self.n_classes):
                if score <= self._quantiles[c]:
                    ps.add(c)
            # Guarantee non-empty prediction sets
            if not ps:
                best_class = max(self._quantiles, key=self._quantiles.get)
                ps.add(best_class)
            # Ordinal contiguity constraint for classes [0(real),1(partial),2(full)].
            # If {0,2} are both present but 1 is not, add 1. This enlarges sets
            # and cannot reduce marginal coverage.
            if self.enforce_contiguity and 0 in ps and 2 in ps and 1 not in ps:
                ps.add(1)
            prediction_sets.append(ps)
        return prediction_sets

    def get_quantiles(self) -> dict[str, float]:
        """Return calibrated quantiles with human-readable class names.

        Returns
        -------
        dict[str, float]
            Mapping from class name to conformal quantile threshold.
        """
        return {self.classes[c]: q for c, q in self._quantiles.items()}
