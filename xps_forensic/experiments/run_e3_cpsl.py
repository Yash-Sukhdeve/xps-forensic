"""E3: CPSL Coverage & Efficiency.

Stage 1: SCP+APS coverage at alpha={0.01, 0.05, 0.10}, prediction set sizes.
Stage 2: CRC on tFNR, empirical tIoU, tFNR, tFDR.
Verify on held-out 20% of PartialSpoof eval.
Statistical test: one-sided binomial for coverage verification.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from xps_forensic.utils.config import load_config
from xps_forensic.utils.stats import binomial_coverage_test, bootstrap_ci
from xps_forensic.cpsl.composed import CPSLPipeline
from xps_forensic.cpsl.nonconformity import compute_nonconformity


def run_e3(cfg=None, precomputed=None):
    """Run E3 CPSL experiment.

    Args:
        cfg: Config.
        precomputed: Dict with 'frame_scores', 'utt_labels', 'frame_labels'
            as lists from E1 inference.
    """
    if cfg is None:
        cfg = load_config()

    output_dir = Path(cfg.experiments.output_dir) / "e3_cpsl"
    output_dir.mkdir(parents=True, exist_ok=True)

    if precomputed is None:
        print("E3 requires precomputed frame scores from E1. Exiting.")
        return

    frame_scores = precomputed["frame_scores"]
    utt_labels = np.array(precomputed["utt_labels"])
    frame_labels = precomputed["frame_labels"]

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

    results = {}

    # Sweep alpha values
    for alpha_utt in cfg.cpsl.alpha_sweep:
        for alpha_seg in cfg.cpsl.alpha_sweep:
            key = f"a_utt={alpha_utt}_a_seg={alpha_seg}"
            print(f"\n--- {key} ---")

            pipeline = CPSLPipeline(
                alpha_utterance=alpha_utt,
                alpha_segment=alpha_seg,
                nonconformity_method=cfg.cpsl.nonconformity,
            )
            pipeline.calibrate(cal_fs, cal_ul, cal_fl)

            # Verify on held-out set
            predictions = pipeline.predict(ver_fs)

            # Stage 1: coverage (CPSLResult dataclass access)
            covered = sum(
                1 for pred, true_label in zip(predictions, ver_ul)
                if true_label in pred.prediction_set
            )
            n_ver = len(ver_ul)
            coverage = covered / n_ver
            p_val = binomial_coverage_test(covered, n_ver, alpha_utt)

            # Prediction set sizes (CPSLResult dataclass access)
            set_sizes = [len(p.prediction_set) for p in predictions]

            # Stage 2: tFNR on partial spoofs (CPSLResult dataclass access)
            from xps_forensic.utils.metrics import compute_tFNR, compute_tFDR, compute_tIoU
            tfnrs, tfdrs, tious = [], [], []
            for pred, fl in zip(predictions, ver_fl):
                if pred.segment_predictions is not None:
                    seg_pred = pred.segment_predictions
                    min_len = min(len(seg_pred), len(fl))
                    tfnrs.append(compute_tFNR(seg_pred[:min_len], fl[:min_len]))
                    tfdrs.append(compute_tFDR(seg_pred[:min_len], fl[:min_len]))
                    tious.append(compute_tIoU(seg_pred[:min_len], fl[:min_len]))

            results[key] = {
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
                    "mean_tFNR": float(np.mean(tfnrs)) if tfnrs else None,
                    "mean_tFDR": float(np.mean(tfdrs)) if tfdrs else None,
                    "mean_tIoU": float(np.mean(tious)) if tious else None,
                    "tFNR_ci": bootstrap_ci(np.array(tfnrs)) if tfnrs else None,
                    "crc_threshold": pipeline.stage2.threshold,
                },
                "quantiles": pipeline.stage1.get_quantiles(),
            }

            print(f"  Coverage: {coverage:.3f} (target: {1-alpha_utt:.3f}), p={p_val:.4f}")
            print(f"  Avg set size: {np.mean(set_sizes):.2f}")
            if tfnrs:
                print(f"  Mean tFNR: {np.mean(tfnrs):.4f}, tIoU: {np.mean(tious):.4f}")

    # Nonconformity score ablation
    print("\n--- Nonconformity ablation ---")
    nc_results = {}
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
    results["nonconformity_ablation"] = nc_results

    # Save
    output_file = output_dir / "results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=lambda x: list(x) if isinstance(x, tuple) else float(x) if isinstance(x, np.floating) else x)
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    run_e3()
