"""E2: Post-hoc Calibration Comparison.

Applies Platt/temperature/isotonic calibration to all 4 detectors.
Includes uncalibrated baseline. Reports ECE, Brier, NLL with bootstrap CIs.
Uses utterance-stratified cross-validation.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from xps_forensic.utils.config import load_config
from xps_forensic.calibration.methods import calibrate_scores, PlattScaling, TemperatureScaling, IsotonicCalibrator
from xps_forensic.calibration.metrics import (
    expected_calibration_error,
    brier_score,
    negative_log_likelihood,
    reliability_diagram_data,
)
from xps_forensic.utils.stats import bootstrap_ci, friedman_nemenyi


def run_e2(cfg=None, precomputed_scores=None):
    """Run E2 calibration comparison.

    Args:
        cfg: Config dict.
        precomputed_scores: Dict of {detector_name: (scores, labels)} from E1.
    """
    if cfg is None:
        cfg = load_config()

    output_dir = Path(cfg.experiments.output_dir) / "e2_calibration"
    output_dir.mkdir(parents=True, exist_ok=True)

    methods = ["uncalibrated", "platt", "temperature", "isotonic"]
    calibrator_classes = {
        "platt": PlattScaling,
        "temperature": TemperatureScaling,
        "isotonic": IsotonicCalibrator,
    }

    results = {}

    for det_name, (scores, labels) in (precomputed_scores or {}).items():
        print(f"\n{'='*60}")
        print(f"Detector: {det_name.upper()}")

        det_results = {}

        # Stratified K-fold cross-validation
        skf = StratifiedKFold(n_splits=cfg.calibration.cv_folds, shuffle=True, random_state=42)

        for method in methods:
            eces, briers, nlls = [], [], []

            for train_idx, test_idx in skf.split(scores, labels):
                train_scores, train_labels = scores[train_idx], labels[train_idx]
                test_scores, test_labels = scores[test_idx], labels[test_idx]

                if method == "uncalibrated":
                    cal_scores = test_scores
                else:
                    cal = calibrator_classes[method]()
                    cal.fit(train_scores, train_labels)
                    cal_scores = cal.transform(test_scores)

                eces.append(expected_calibration_error(cal_scores, test_labels))
                briers.append(brier_score(cal_scores, test_labels))
                nlls.append(negative_log_likelihood(cal_scores, test_labels))

            det_results[method] = {
                "ece_mean": float(np.mean(eces)),
                "ece_ci": bootstrap_ci(np.array(eces)),
                "brier_mean": float(np.mean(briers)),
                "brier_ci": bootstrap_ci(np.array(briers)),
                "nll_mean": float(np.mean(nlls)),
                "nll_ci": bootstrap_ci(np.array(nlls)),
            }
            print(f"  {method:15s} ECE={np.mean(eces):.4f} Brier={np.mean(briers):.4f} NLL={np.mean(nlls):.4f}")

        results[det_name] = det_results

    # Friedman test across detectors for best calibration method
    if len(results) >= 3:
        ece_matrix = np.array([
            [results[d][m]["ece_mean"] for m in methods]
            for d in results
        ])
        friedman = friedman_nemenyi(ece_matrix)
        results["statistical_tests"] = {"friedman_ece": friedman}

    # Save
    output_file = output_dir / "results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=lambda x: list(x) if isinstance(x, tuple) else float(x) if isinstance(x, np.floating) else x)
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    run_e2()
