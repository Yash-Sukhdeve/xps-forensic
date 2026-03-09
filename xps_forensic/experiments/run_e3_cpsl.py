"""E3: CPSL Coverage & Efficiency.

Stage 1: SCP+APS coverage at alpha={0.01, 0.05, 0.10}, prediction set sizes.
Stage 2: CRC on tFNR, empirical tIoU, tFNR, tFDR.
Verify on held-out 20% of PartialSpoof eval.
Statistical test: one-sided binomial for coverage verification.

Input format (from E1):
    precomputed = {
        "detector_name": {
            "utt_scores": np.ndarray,   # (N,) float
            "utt_labels": np.ndarray,   # (N,) int  (0/1/2)
            "frame_scores": [...],       # list of per-utterance arrays
            "frame_labels": [...],       # list of per-utterance label arrays
            "utt_ids": [...],
            "frame_shift_ms": float,     # detector-specific (e.g. 160)
        },
        ...
    }

If precomputed is None, loads from E1 disk output at
results/e1_baseline/{det}_*.npy.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from xps_forensic.utils.config import load_config
from xps_forensic.utils.stats import binomial_coverage_test, bootstrap_ci
from xps_forensic.cpsl.composed import CPSLPipeline
from xps_forensic.cpsl.nonconformity import compute_nonconformity
from xps_forensic.utils.metrics import (
    compute_tFNR,
    compute_tFDR,
    compute_tIoU,
    upsample_binary_predictions_to_label_grid,
)

logger = logging.getLogger(__name__)

# Label frame shift in PartialSpoof dataset (10 ms at finest resolution).
LABEL_FRAME_SHIFT_MS = 10.0

# Detectors we expect from E1 (used for disk loading fallback).
_DETECTOR_NAMES = ["bam", "sal", "cfprf", "mrm"]

# Default frame shift per detector (used when E1 results.json is unavailable).
_DEFAULT_FRAME_SHIFT_MS = {
    "bam": 160.0,
    "sal": 160.0,
    "cfprf": 20.0,
    "mrm": 20.0,
}


def _load_e1_from_disk(e1_dir: Path) -> dict:
    """Load E1 per-detector results from .npy files on disk.

    Returns dict in the same format as E1's in-memory ``precomputed``.
    Reads frame_shift_ms from E1's results.json if available; otherwise
    falls back to detector-specific defaults.
    """
    # Try to load frame_shift_ms from E1 results.json
    e1_results: dict = {}
    results_json = e1_dir / "results.json"
    if results_json.exists():
        with open(results_json) as f:
            e1_results = json.load(f)

    precomputed: dict = {}
    for det in _DETECTOR_NAMES:
        scores_path = e1_dir / f"{det}_utt_scores.npy"
        labels_path = e1_dir / f"{det}_utt_labels.npy"
        frame_scores_path = e1_dir / f"{det}_frame_scores.npy"
        frame_labels_path = e1_dir / f"{det}_frame_labels.npy"
        utt_ids_path = e1_dir / f"{det}_utt_ids.npy"

        # E3 requires frame scores — skip detectors without them
        if not frame_scores_path.exists() or not frame_labels_path.exists():
            logger.info(
                "E1 disk output not found for %s at %s — skipping.",
                det.upper(),
                e1_dir,
            )
            continue
        if not scores_path.exists() or not labels_path.exists():
            logger.info(
                "E1 utterance scores/labels not found for %s — skipping.",
                det.upper(),
                e1_dir,
            )
            continue

        utt_scores = np.load(scores_path)
        utt_labels = np.load(labels_path)
        frame_scores = list(np.load(frame_scores_path, allow_pickle=True))
        frame_labels = list(np.load(frame_labels_path, allow_pickle=True))

        entry: dict = {
            "utt_scores": utt_scores,
            "utt_labels": utt_labels,
            "frame_scores": frame_scores,
            "frame_labels": frame_labels,
        }

        if utt_ids_path.exists():
            entry["utt_ids"] = list(
                np.load(utt_ids_path, allow_pickle=True)
            )

        # Resolve frame_shift_ms: prefer E1 results.json, then defaults
        if det in e1_results and "frame_shift_ms" in e1_results[det]:
            entry["frame_shift_ms"] = float(e1_results[det]["frame_shift_ms"])
        else:
            entry["frame_shift_ms"] = _DEFAULT_FRAME_SHIFT_MS.get(det, 20.0)

        precomputed[det] = entry
        logger.info(
            "Loaded E1 disk output for %s: %d utterances, frame_shift=%.1f ms.",
            det.upper(),
            len(utt_scores),
            entry["frame_shift_ms"],
        )

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


def run_e3(cfg=None, precomputed=None):
    """Run E3 CPSL experiment.

    For each detector in the precomputed dict:
      1. Split data 80/20 into calibration/verification sets
      2. Stage 1 (SCP+APS): calibrate, predict, test coverage
      3. Stage 2 (CRC): calibrate tFNR, evaluate temporal metrics
      4. Run binomial coverage test
      5. Run nonconformity score ablation

    Parameters
    ----------
    cfg : optional
        Config dict. Loaded from default.yaml if None.
    precomputed : dict or None
        Per-detector dict from E1's ``run_e1()`` return value.
        If None, attempts to load from ``results/e1_baseline/`` on disk.

    Returns
    -------
    dict
        Results dict keyed by detector name.
    """
    if cfg is None:
        cfg = load_config()

    output_dir = Path(cfg.experiments.output_dir) / "e3_cpsl"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Resolve input scores ──────────────────────────────────────
    if precomputed is None:
        e1_dir = Path(cfg.experiments.output_dir) / "e1_baseline"
        print(f"No precomputed scores passed; loading from disk: {e1_dir}")
        precomputed = _load_e1_from_disk(e1_dir)

    if not precomputed:
        logger.warning(
            "No detector scores available — cannot run E3 CPSL. Exiting."
        )
        return {}

    all_results: dict = {}

    for det_name, det_data in precomputed.items():
        print(f"\n{'='*60}")
        print(f"Detector: {det_name.upper()}")
        print(f"{'='*60}")

        frame_scores = det_data["frame_scores"]
        utt_labels = np.asarray(det_data["utt_labels"])
        frame_labels = det_data["frame_labels"]
        score_frame_shift_ms = float(det_data.get(
            "frame_shift_ms",
            _DEFAULT_FRAME_SHIFT_MS.get(det_name, 20.0),
        ))

        print(f"  Frame shift: {score_frame_shift_ms} ms")
        print(f"  N utterances: {len(frame_scores)}")

        # Split: 80% calibration, 20% verification
        n = len(frame_scores)
        rng = np.random.default_rng(cfg.project.seed)
        perm = rng.permutation(n)
        split = int(n * cfg.data.partialspoof.eval_split_ratio)

        cal_idx, ver_idx = perm[:split], perm[split:]
        cal_fs = [frame_scores[i] for i in cal_idx]
        cal_ul = utt_labels[cal_idx]
        cal_fl = [frame_labels[i] for i in cal_idx]
        ver_fs = [frame_scores[i] for i in ver_idx]
        ver_ul = utt_labels[ver_idx]
        ver_fl = [frame_labels[i] for i in ver_idx]

        det_results: dict = {
            "frame_shift_ms": score_frame_shift_ms,
            "n_utterances": n,
            "n_cal": len(cal_idx),
            "n_ver": len(ver_idx),
        }

        # ── Alpha sweep ───────────────────────────────────────────
        alpha_results: dict = {}
        for alpha_utt in cfg.cpsl.alpha_sweep:
            for alpha_seg in cfg.cpsl.alpha_sweep:
                key = f"a_utt={alpha_utt}_a_seg={alpha_seg}"
                print(f"\n  --- {key} ---")

                pipeline = CPSLPipeline(
                    alpha_utterance=alpha_utt,
                    alpha_segment=alpha_seg,
                    nonconformity_method=cfg.cpsl.nonconformity,
                    score_frame_shift_ms=score_frame_shift_ms,
                    label_frame_shift_ms=LABEL_FRAME_SHIFT_MS,
                )
                pipeline.calibrate(cal_fs, cal_ul, cal_fl)

                # Verify on held-out set
                predictions = pipeline.predict(ver_fs)

                # Stage 1: coverage
                covered = sum(
                    1 for pred, true_label in zip(predictions, ver_ul)
                    if true_label in pred.prediction_set
                )
                n_ver = len(ver_ul)
                coverage = covered / n_ver
                p_val = binomial_coverage_test(covered, n_ver, alpha_utt)

                # Prediction set sizes
                set_sizes = [len(p.prediction_set) for p in predictions]

                # Stage 2: temporal metrics on partial spoofs
                tfnrs, tfdrs, tious = [], [], []
                for pred, fl in zip(predictions, ver_fl):
                    if pred.segment_predictions is not None:
                        # Align predictions to label grid using detector's
                        # actual frame shift
                        seg_pred_aligned = upsample_binary_predictions_to_label_grid(
                            pred.segment_predictions,
                            score_frame_shift_ms,
                            LABEL_FRAME_SHIFT_MS,
                        )
                        min_len = min(len(seg_pred_aligned), len(fl))
                        tfnrs.append(
                            compute_tFNR(
                                seg_pred_aligned[:min_len], fl[:min_len]
                            )
                        )
                        tfdrs.append(
                            compute_tFDR(
                                seg_pred_aligned[:min_len], fl[:min_len]
                            )
                        )
                        tious.append(
                            compute_tIoU(
                                seg_pred_aligned[:min_len], fl[:min_len]
                            )
                        )

                alpha_results[key] = {
                    "alpha_utterance": alpha_utt,
                    "alpha_segment": alpha_seg,
                    "composed_guarantee": pipeline.composed_guarantee,
                    "stage1": {
                        "coverage": coverage,
                        "coverage_target": 1 - alpha_utt,
                        "binomial_p_value": p_val,
                        "coverage_verified": p_val > 0.05,
                        "avg_set_size": float(np.mean(set_sizes)),
                        "set_size_ci": bootstrap_ci(np.array(set_sizes)),
                    },
                    "stage2": {
                        "mean_tFNR": float(np.mean(tfnrs))
                        if tfnrs
                        else None,
                        "mean_tFDR": float(np.mean(tfdrs))
                        if tfdrs
                        else None,
                        "mean_tIoU": float(np.mean(tious))
                        if tious
                        else None,
                        "tFNR_ci": bootstrap_ci(np.array(tfnrs))
                        if tfnrs
                        else None,
                        "crc_threshold": pipeline.stage2.threshold,
                    },
                    "quantiles": pipeline.stage1.get_quantiles(),
                }

                print(
                    f"    Coverage: {coverage:.3f} "
                    f"(target: {1-alpha_utt:.3f}), p={p_val:.4f}"
                )
                print(f"    Avg set size: {np.mean(set_sizes):.2f}")
                if tfnrs:
                    print(
                        f"    Mean tFNR: {np.mean(tfnrs):.4f}, "
                        f"tIoU: {np.mean(tious):.4f}"
                    )

        det_results["alpha_sweep"] = alpha_results

        # ── Nonconformity score ablation ──────────────────────────
        print(f"\n  --- Nonconformity ablation ({det_name.upper()}) ---")
        nc_results: dict = {}
        for method in ["max", "logsumexp"]:
            betas = [None] if method == "max" else cfg.cpsl.logsumexp_beta
            for beta in betas:
                nc_key = f"{method}" + (f"_b{beta}" if beta else "")
                nc_scores = compute_nonconformity(
                    cal_fs, method=method, beta=beta or 10.0
                )
                nc_results[nc_key] = {
                    "mean": float(np.mean(nc_scores)),
                    "std": float(np.std(nc_scores)),
                }
        det_results["nonconformity_ablation"] = nc_results

        all_results[det_name] = det_results

    # ── Save ──────────────────────────────────────────────────────
    output_file = output_dir / "results.json"
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2, default=_json_serializer)
    print(f"\nResults saved to {output_file}")

    return all_results


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    run_e3()
