"""E2: Post-hoc Calibration Comparison.

Applies Platt/temperature/isotonic calibration to all 4 detectors.
Includes uncalibrated baseline. Reports ECE, Brier, NLL with bootstrap CIs.
Uses utterance-stratified cross-validation.

Input format (from E1):
    precomputed_scores = {
        "detector_name": {
            "utt_scores": np.ndarray,   # (N,) float
            "utt_labels": np.ndarray,   # (N,) int  (0=real, 1 or 2=spoof)
            "frame_scores": [...],       # list of per-utterance arrays
            "frame_labels": [...],
            "utt_ids": [...],
            "frame_shift_ms": float,
        },
        ...
    }

If precomputed_scores is None, loads from E1 disk output at
results/e1_baseline/{det}_utt_scores.npy etc.

References
----------
Platt, "Probabilistic Outputs for SVMs", 1999.
Guo et al., "On Calibration of Modern Neural Networks", ICML 2017.
Zadrozny & Elkan, "Transforming Classifier Scores into Accurate
Multiclass Probability Estimates", KDD 2002.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from xps_forensic.utils.config import load_config
from xps_forensic.calibration.methods import (
    PlattScaling,
    TemperatureScaling,
    IsotonicCalibrator,
)
from xps_forensic.calibration.metrics import (
    expected_calibration_error,
    brier_score,
    negative_log_likelihood,
    reliability_diagram_data,
)
from xps_forensic.utils.stats import bootstrap_ci, friedman_nemenyi

logger = logging.getLogger(__name__)

# Detectors we expect from E1 (used for disk loading fallback).
_DETECTOR_NAMES = ["bam", "sal", "cfprf", "mrm"]


def _binarize_labels(labels: np.ndarray) -> np.ndarray:
    """Binarize ternary labels: 0 -> 0 (real), {1, 2} -> 1 (spoof).

    E1 already binarizes at save time (min(label, 1)), but we guard
    against raw ternary labels from in-memory pipelines that may skip
    that step.
    """
    binary = np.where(labels >= 1, 1, 0)
    return binary.astype(int)


def _load_e1_from_disk(e1_dir: Path) -> dict:
    """Load E1 per-detector results from .npy files on disk.

    Returns dict in the same format as E1's in-memory ``precomputed``.
    """
    precomputed: dict = {}
    for det in _DETECTOR_NAMES:
        scores_path = e1_dir / f"{det}_utt_scores.npy"
        labels_path = e1_dir / f"{det}_utt_labels.npy"
        if not scores_path.exists() or not labels_path.exists():
            logger.info(
                "E1 disk output not found for %s at %s — skipping.",
                det.upper(),
                e1_dir,
            )
            continue
        utt_scores = np.load(scores_path)
        utt_labels = np.load(labels_path)

        # Optional arrays — load if available
        frame_scores_path = e1_dir / f"{det}_frame_scores.npy"
        frame_labels_path = e1_dir / f"{det}_frame_labels.npy"
        utt_ids_path = e1_dir / f"{det}_utt_ids.npy"

        entry: dict = {
            "utt_scores": utt_scores,
            "utt_labels": utt_labels,
        }
        if frame_scores_path.exists():
            entry["frame_scores"] = list(
                np.load(frame_scores_path, allow_pickle=True)
            )
        if frame_labels_path.exists():
            entry["frame_labels"] = list(
                np.load(frame_labels_path, allow_pickle=True)
            )
        if utt_ids_path.exists():
            entry["utt_ids"] = list(
                np.load(utt_ids_path, allow_pickle=True)
            )

        precomputed[det] = entry
        logger.info("Loaded E1 disk output for %s: %d utterances.",
                     det.upper(), len(utt_scores))

    return precomputed


def _json_serializer(obj):
    """JSON default serializer for numpy / tuple types."""
    if isinstance(obj, tuple):
        return list(obj)
    if isinstance(obj, (np.floating, float)):
        return float(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def run_e2(cfg=None, precomputed_scores=None):
    """Run E2 calibration comparison.

    Parameters
    ----------
    cfg : optional
        Config dict.  Loaded from default.yaml if None.
    precomputed_scores : dict or None
        Per-detector dict from E1's ``run_e1()`` return value.
        Each value is a dict with at least ``utt_scores`` and ``utt_labels``.
        If None, attempts to load from ``results/e1_baseline/`` on disk.

    Returns
    -------
    dict
        Results dict keyed by detector name, each containing per-method
        calibration metrics.  Also includes ``statistical_tests`` if >= 3
        detectors are present.
    """
    if cfg is None:
        cfg = load_config()

    output_dir = Path(cfg.experiments.output_dir) / "e2_calibration"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Resolve input scores ──────────────────────────────────────
    if precomputed_scores is None:
        e1_dir = Path(cfg.experiments.output_dir) / "e1_baseline"
        print(f"No precomputed scores passed; loading from disk: {e1_dir}")
        precomputed_scores = _load_e1_from_disk(e1_dir)

    if not precomputed_scores:
        logger.warning("No detector scores available — nothing to calibrate.")
        return {}

    methods = ["uncalibrated", "platt", "temperature", "isotonic"]
    calibrator_classes = {
        "platt": PlattScaling,
        "temperature": TemperatureScaling,
        "isotonic": IsotonicCalibrator,
    }

    n_folds = cfg.calibration.cv_folds
    results: dict = {}

    for det_name, det_data in precomputed_scores.items():
        # ── Extract & binarize ────────────────────────────────────
        scores = np.asarray(det_data["utt_scores"], dtype=float)
        labels = _binarize_labels(np.asarray(det_data["utt_labels"]))

        if len(scores) == 0:
            logger.warning("Detector %s has 0 utterances — skipping.", det_name)
            continue

        print(f"\n{'='*60}")
        print(f"Detector: {det_name.upper()} ({len(scores)} utterances)")
        print(f"  Label distribution: real={int((labels==0).sum())}, "
              f"spoof={int((labels==1).sum())}")

        det_results: dict = {
            "n_utterances": int(len(scores)),
        }

        # ── Stratified K-fold CV ──────────────────────────────────
        skf = StratifiedKFold(
            n_splits=n_folds, shuffle=True, random_state=42,
        )

        for method in methods:
            fold_eces: list[float] = []
            fold_briers: list[float] = []
            fold_nlls: list[float] = []

            for fold_idx, (train_idx, test_idx) in enumerate(
                skf.split(scores, labels)
            ):
                train_scores, train_labels = scores[train_idx], labels[train_idx]
                test_scores, test_labels = scores[test_idx], labels[test_idx]

                if method == "uncalibrated":
                    cal_scores = test_scores
                else:
                    cal = calibrator_classes[method]()
                    cal.fit(train_scores, train_labels)
                    cal_scores = cal.transform(test_scores)

                fold_eces.append(
                    expected_calibration_error(cal_scores, test_labels)
                )
                fold_briers.append(brier_score(cal_scores, test_labels))
                fold_nlls.append(
                    negative_log_likelihood(cal_scores, test_labels)
                )

            det_results[method] = {
                "ece_mean": float(np.mean(fold_eces)),
                "ece_std": float(np.std(fold_eces)),
                "ece_ci": bootstrap_ci(np.array(fold_eces)),
                "ece_folds": [float(v) for v in fold_eces],
                "brier_mean": float(np.mean(fold_briers)),
                "brier_std": float(np.std(fold_briers)),
                "brier_ci": bootstrap_ci(np.array(fold_briers)),
                "brier_folds": [float(v) for v in fold_briers],
                "nll_mean": float(np.mean(fold_nlls)),
                "nll_std": float(np.std(fold_nlls)),
                "nll_ci": bootstrap_ci(np.array(fold_nlls)),
                "nll_folds": [float(v) for v in fold_nlls],
            }
            print(
                f"  {method:15s}  ECE={np.mean(fold_eces):.4f}  "
                f"Brier={np.mean(fold_briers):.4f}  "
                f"NLL={np.mean(fold_nlls):.4f}"
            )

        results[det_name] = det_results

    # ── Friedman-Nemenyi tests ────────────────────────────────────
    # Each row = one detector, each column = one calibration method.
    # We need >= 3 rows (detectors) for Friedman to be valid.
    det_keys = [k for k in results if k != "statistical_tests"]

    if len(det_keys) >= 3:
        stat_tests: dict = {}
        for metric_name in ["ece", "brier", "nll"]:
            metric_matrix = np.array([
                [results[d][m][f"{metric_name}_mean"] for m in methods]
                for d in det_keys
            ])
            friedman = friedman_nemenyi(
                metric_matrix, higher_is_better=False,
            )
            stat_tests[f"friedman_{metric_name}"] = friedman
            print(
                f"\nFriedman test ({metric_name.upper()}): "
                f"stat={friedman['friedman_stat']:.3f}, "
                f"p={friedman['friedman_p']:.4f}"
            )
        results["statistical_tests"] = stat_tests
    else:
        print(
            f"\nSkipping Friedman test: need >= 3 detectors, "
            f"got {len(det_keys)}."
        )

    # ── Save results ──────────────────────────────────────────────
    output_file = output_dir / "results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=_json_serializer)
    print(f"\nResults saved to {output_file}")

    return results


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    run_e2()
