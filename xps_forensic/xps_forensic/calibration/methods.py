"""Post-hoc calibration methods for frame-level detector scores.

Implements systematic comparison of:
- Platt scaling (Platt, 1999)
- Temperature scaling (Guo et al., ICML 2017)
- Isotonic regression (Zadrozny & Elkan, 2002)

Applied to audio CM scores following Wang et al. (Interspeech 2024) and
Pascu et al. (Interspeech 2024) methodology.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


class BaseCalibrator(ABC):
    """Base class for score calibrators."""

    @abstractmethod
    def fit(self, scores: np.ndarray, labels: np.ndarray) -> None:
        """Fit calibrator on calibration data."""

    @abstractmethod
    def transform(self, scores: np.ndarray) -> np.ndarray:
        """Apply calibration to scores."""

    def fit_transform(self, scores: np.ndarray, labels: np.ndarray) -> np.ndarray:
        self.fit(scores, labels)
        return self.transform(scores)


class PlattScaling(BaseCalibrator):
    """Platt scaling: logistic regression on raw scores."""

    def __init__(self):
        self._lr = LogisticRegression(solver="lbfgs", max_iter=1000)

    def fit(self, scores: np.ndarray, labels: np.ndarray) -> None:
        self._lr.fit(scores.reshape(-1, 1), labels)

    def transform(self, scores: np.ndarray) -> np.ndarray:
        return self._lr.predict_proba(scores.reshape(-1, 1))[:, 1]


class TemperatureScaling(BaseCalibrator):
    """Temperature scaling: scale logits by learned temperature T.

    P_calibrated = sigmoid(logit(score) / T)
    """

    def __init__(self):
        self.temperature: float = 1.0

    def fit(self, scores: np.ndarray, labels: np.ndarray) -> None:
        eps = 1e-7
        clipped = np.clip(scores, eps, 1 - eps)
        logits = np.log(clipped / (1 - clipped))

        def nll(t):
            scaled = 1.0 / (1.0 + np.exp(-logits / t))
            scaled = np.clip(scaled, eps, 1 - eps)
            return -np.mean(
                labels * np.log(scaled) + (1 - labels) * np.log(1 - scaled)
            )

        result = minimize_scalar(nll, bounds=(0.01, 10.0), method="bounded")
        self.temperature = result.x

    def transform(self, scores: np.ndarray) -> np.ndarray:
        eps = 1e-7
        clipped = np.clip(scores, eps, 1 - eps)
        logits = np.log(clipped / (1 - clipped))
        return 1.0 / (1.0 + np.exp(-logits / self.temperature))


class IsotonicCalibrator(BaseCalibrator):
    """Isotonic regression calibration."""

    def __init__(self):
        self._ir = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)

    def fit(self, scores: np.ndarray, labels: np.ndarray) -> None:
        self._ir.fit(scores, labels)

    def transform(self, scores: np.ndarray) -> np.ndarray:
        return self._ir.predict(scores)


def calibrate_scores(
    scores: np.ndarray,
    labels: np.ndarray,
    methods: list[str] | None = None,
) -> dict[str, np.ndarray]:
    """Apply all calibration methods and return results.

    Args:
        scores: Raw detector scores, shape (n,).
        labels: Binary labels, shape (n,).
        methods: List of method names. Default: all methods.

    Returns:
        Dict mapping method name to calibrated scores.
    """
    if methods is None:
        methods = ["platt", "temperature", "isotonic"]

    calibrators = {
        "platt": PlattScaling,
        "temperature": TemperatureScaling,
        "isotonic": IsotonicCalibrator,
    }

    results = {"uncalibrated": scores.copy()}

    for name in methods:
        cal = calibrators[name]()
        cal.fit(scores, labels)
        results[name] = cal.transform(scores)

    return results
