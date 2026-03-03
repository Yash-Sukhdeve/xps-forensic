"""Tests for post-hoc calibration methods and metrics."""
import numpy as np
import pytest
from xps_forensic.calibration.methods import (
    PlattScaling,
    TemperatureScaling,
    IsotonicCalibrator,
    calibrate_scores,
)


@pytest.fixture
def calibration_data(rng):
    """Simulated uncalibrated scores and labels."""
    n = 500
    real_scores = rng.beta(2, 5, size=n // 2)
    fake_scores = rng.beta(5, 2, size=n // 2)
    scores = np.concatenate([real_scores, fake_scores])
    labels = np.concatenate([np.zeros(n // 2), np.ones(n // 2)])
    perm = rng.permutation(n)
    return scores[perm], labels[perm].astype(int)


class TestPlattScaling:
    def test_fit_transform(self, calibration_data):
        scores, labels = calibration_data
        platt = PlattScaling()
        platt.fit(scores, labels)
        calibrated = platt.transform(scores)
        assert calibrated.shape == scores.shape
        assert np.all(calibrated >= 0) and np.all(calibrated <= 1)

    def test_calibrated_mean_closer_to_prevalence(self, calibration_data):
        scores, labels = calibration_data
        platt = PlattScaling()
        platt.fit(scores, labels)
        calibrated = platt.transform(scores)
        prevalence = labels.mean()
        assert abs(calibrated.mean() - prevalence) < abs(scores.mean() - prevalence) + 0.1


class TestTemperatureScaling:
    def test_fit_transform(self, calibration_data):
        scores, labels = calibration_data
        temp = TemperatureScaling()
        temp.fit(scores, labels)
        calibrated = temp.transform(scores)
        assert calibrated.shape == scores.shape
        assert temp.temperature > 0


class TestIsotonicCalibrator:
    def test_fit_transform(self, calibration_data):
        scores, labels = calibration_data
        iso = IsotonicCalibrator()
        iso.fit(scores, labels)
        calibrated = iso.transform(scores)
        assert calibrated.shape == scores.shape
        sorted_idx = np.argsort(scores)
        cal_sorted = calibrated[sorted_idx]
        assert np.all(np.diff(cal_sorted) >= -1e-10)


class TestCalibrateScores:
    def test_all_methods(self, calibration_data):
        scores, labels = calibration_data
        results = calibrate_scores(scores, labels)
        assert "uncalibrated" in results
        assert "platt" in results
        assert "temperature" in results
        assert "isotonic" in results
        for name, cal_scores in results.items():
            assert cal_scores.shape == scores.shape


from xps_forensic.calibration.metrics import (
    expected_calibration_error,
    brier_score,
    negative_log_likelihood,
    reliability_diagram_data,
)


class TestCalibrationMetrics:
    def test_ece_perfect(self):
        scores = np.array([0.1, 0.2, 0.8, 0.9])
        labels = np.array([0, 0, 1, 1])
        ece = expected_calibration_error(scores, labels, n_bins=2)
        assert ece < 0.2

    def test_ece_terrible(self):
        scores = np.array([0.9, 0.9, 0.9, 0.9])
        labels = np.array([0, 0, 0, 0])
        ece = expected_calibration_error(scores, labels, n_bins=2)
        assert ece > 0.5

    def test_brier_perfect(self):
        scores = np.array([0.0, 0.0, 1.0, 1.0])
        labels = np.array([0, 0, 1, 1])
        assert brier_score(scores, labels) == pytest.approx(0.0)

    def test_nll_finite(self):
        scores = np.array([0.1, 0.2, 0.8, 0.9])
        labels = np.array([0, 0, 1, 1])
        nll = negative_log_likelihood(scores, labels)
        assert np.isfinite(nll)
        assert nll > 0

    def test_reliability_diagram(self):
        rng = np.random.default_rng(42)
        scores = rng.uniform(0, 1, 100)
        labels = (scores > 0.5).astype(int)
        bins, accs, confs, counts = reliability_diagram_data(scores, labels)
        assert len(bins) == len(accs) == len(confs) == len(counts)
