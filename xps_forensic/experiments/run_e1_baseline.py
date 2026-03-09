"""E1: Baseline Detection & Localization.

Reproduces published results for all 4 detectors on PartialSpoof eval set.
Reports Utt-EER, Seg-EER at multiple resolutions, Seg-F1 with bootstrap CIs.

Saves raw scores per-utterance so that E2 (calibration) and E3 (CPSL) can
consume them without re-running inference.

Output layout (under results/e1_baseline/):
    results.json          — all computed metrics
    {det}_utt_scores.npy  — utterance-level scores, shape (N,)
    {det}_utt_labels.npy  — utterance-level binary labels, shape (N,)
    {det}_frame_scores.npy  — list of per-utterance frame score arrays (object array)
    {det}_frame_labels.npy  — list of per-utterance frame label arrays (object array)
    {det}_utt_ids.npy       — utterance IDs (object array of strings)
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from xps_forensic.utils.config import load_config
from xps_forensic.utils.metrics import (
    compute_eer,
    compute_segment_eer_mixed,
    compute_segment_f1,
    upsample_binary_predictions_to_label_grid,
)
from xps_forensic.utils.stats import bootstrap_ci
from xps_forensic.data.partialspoof import PartialSpoofDataset
from xps_forensic.detectors.bam import BAMDetector
from xps_forensic.detectors.sal import SALDetector
from xps_forensic.detectors.cfprf import CFPRFDetector
from xps_forensic.detectors.mrm import MRMDetector

logger = logging.getLogger(__name__)

DETECTOR_MAP = {
    "bam": BAMDetector,
    "sal": SALDetector,
    "cfprf": CFPRFDetector,
    "mrm": MRMDetector,
}

# Label frame shift in PartialSpoof dataset (10 ms at finest resolution).
LABEL_FRAME_SHIFT_MS = 10.0


def _build_detector(det_name: str, det_cfg: dict, device: str):
    """Instantiate a detector from config, returning None if checkpoint missing.

    Each detector wrapper accepts slightly different constructor kwargs.
    We map config keys to the constructor signatures defined in the wrappers.
    """
    DetClass = DETECTOR_MAP[det_name]
    checkpoint = det_cfg.get("checkpoint")

    # If checkpoint is explicitly null / None, we cannot run inference.
    if checkpoint is None:
        logger.warning(
            "Detector %s: checkpoint is null in config — skipping.",
            det_name.upper(),
        )
        return None

    ckpt_path = Path(checkpoint)
    if not ckpt_path.is_file():
        logger.warning(
            "Detector %s: checkpoint not found at %s — skipping.",
            det_name.upper(),
            ckpt_path,
        )
        return None

    # Common kwargs shared by all detectors
    kwargs: dict = {
        "checkpoint": str(ckpt_path),
        "device": device,
    }

    # Per-detector optional kwargs from config
    external_dir = det_cfg.get("external_dir")
    if external_dir is not None:
        kwargs["external_dir"] = external_dir

    ssl_ckpt = det_cfg.get("ssl_ckpt")
    ssl_path = det_cfg.get("ssl_path")

    if det_name == "bam":
        if ssl_ckpt:
            kwargs["ssl_ckpt"] = ssl_ckpt
    elif det_name == "sal":
        if ssl_ckpt:
            kwargs["ssl_ckpt"] = ssl_ckpt
    elif det_name == "cfprf":
        if ssl_path:
            kwargs["ssl_path"] = ssl_path
    elif det_name == "mrm":
        if ssl_path:
            kwargs["ssl_path"] = ssl_path

    try:
        detector = DetClass(**kwargs)
        detector.load_model()
        return detector
    except Exception as exc:
        logger.warning(
            "Detector %s: failed to load model — %s. Skipping.",
            det_name.upper(),
            exc,
        )
        return None


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


def run_e1(cfg=None):
    """Run E1 baseline experiment.

    For each available detector (skips if checkpoint missing):
      1. Load model checkpoint.
      2. Run inference on PartialSpoof eval set.
      3. Compute utterance-level EER with bootstrap CI.
      4. Compute segment-level EER at multiple resolutions with bootstrap CI.
      5. Compute segment F1 at the detector's native resolution.
      6. Save all scores and results.

    Returns:
        dict: Per-detector results including raw score arrays, suitable for
        passing directly to E2/E3.
    """
    if cfg is None:
        cfg = load_config()

    output_dir = Path(cfg.experiments.output_dir) / "e1_baseline"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    dataset = PartialSpoofDataset(
        root=cfg.data.partialspoof.path,
        split="eval",
        sample_rate=cfg.data.partialspoof.sample_rate,
    )
    n_utterances = len(dataset)
    print(f"Loaded PartialSpoof eval: {n_utterances} utterances")

    if n_utterances == 0:
        logger.warning("Dataset is empty — nothing to evaluate.")
        return {}

    resolutions = cfg.experiments.resolutions_ms
    results: dict = {}
    # Structured output for downstream experiments (E2, E3).
    precomputed: dict = {}

    for det_name in ["bam", "sal", "cfprf", "mrm"]:
        print(f"\n{'=' * 60}")
        print(f"Detector: {det_name.upper()}")
        print(f"{'=' * 60}")

        det_cfg = cfg.detectors.get(det_name, {})
        if not det_cfg:
            logger.warning("No config entry for detector %s — skipping.", det_name)
            continue

        detector = _build_detector(det_name, det_cfg, cfg.device)
        if detector is None:
            print(f"  [SKIP] Detector {det_name.upper()} not available.")
            continue

        det_frame_shift_ms = detector.frame_shift_ms
        print(f"  Frame shift: {det_frame_shift_ms} ms")

        # ── Inference ────────────────────────────────────────────────
        all_utt_ids: list[str] = []
        all_utt_scores: list[float] = []
        all_utt_labels: list[int] = []
        all_frame_scores: list[np.ndarray] = []
        all_frame_labels: list[np.ndarray] = []

        t0 = time.time()
        for idx, sample in enumerate(dataset):
            output = detector.predict(
                sample.waveform, sample.sample_rate,
                utterance_id=sample.utterance_id,
            )

            all_utt_ids.append(sample.utterance_id)
            all_utt_scores.append(output.utterance_score)
            # Binary utterance label: 0=real, >=1 means contains spoof
            all_utt_labels.append(min(sample.utterance_label, 1))
            all_frame_scores.append(output.frame_scores)
            all_frame_labels.append(sample.frame_labels)

            if (idx + 1) % 500 == 0:
                elapsed = time.time() - t0
                print(f"  Processed {idx + 1}/{n_utterances} "
                      f"({elapsed:.1f}s elapsed)")

        elapsed = time.time() - t0
        print(f"  Inference complete: {n_utterances} utterances in {elapsed:.1f}s")

        utt_scores = np.array(all_utt_scores)
        utt_labels = np.array(all_utt_labels)

        # ── Save raw scores for E2/E3 ───────────────────────────────
        np.save(output_dir / f"{det_name}_utt_scores.npy", utt_scores)
        np.save(output_dir / f"{det_name}_utt_labels.npy", utt_labels)
        np.save(
            output_dir / f"{det_name}_frame_scores.npy",
            np.array(all_frame_scores, dtype=object),
            allow_pickle=True,
        )
        np.save(
            output_dir / f"{det_name}_frame_labels.npy",
            np.array(all_frame_labels, dtype=object),
            allow_pickle=True,
        )
        np.save(
            output_dir / f"{det_name}_utt_ids.npy",
            np.array(all_utt_ids, dtype=object),
            allow_pickle=True,
        )

        # Store for in-memory downstream use
        precomputed[det_name] = {
            "utt_scores": utt_scores,
            "utt_labels": utt_labels,
            "frame_scores": all_frame_scores,
            "frame_labels": all_frame_labels,
            "utt_ids": all_utt_ids,
            "frame_shift_ms": det_frame_shift_ms,
        }

        # ── Utterance-level EER ──────────────────────────────────────
        utt_eer, utt_thresh = compute_eer(utt_scores, utt_labels)
        # Bootstrap CI on the error indicator
        utt_errors = (utt_scores > utt_thresh).astype(int) != utt_labels
        utt_eer_ci = bootstrap_ci(utt_errors.astype(float), n_bootstrap=1000)

        det_results: dict = {
            "frame_shift_ms": det_frame_shift_ms,
            "n_utterances": n_utterances,
            "utt_eer": float(utt_eer),
            "utt_eer_threshold": float(utt_thresh),
            "utt_eer_ci": utt_eer_ci,
        }
        print(f"  Utt-EER: {utt_eer:.4f} "
              f"(95% CI: {utt_eer_ci[0]:.4f}–{utt_eer_ci[1]:.4f})")

        # ── Segment-level EER at each resolution ─────────────────────
        # Uses compute_segment_eer_mixed which handles different frame shifts
        # for scores (det_frame_shift_ms) and labels (10 ms).
        for res in resolutions:
            seg_eers: list[float] = []
            for fs, fl in zip(all_frame_scores, all_frame_labels):
                if len(fl) == 0 or not fl.any():
                    continue
                if len(fs) == 0:
                    continue
                eer_val, _ = compute_segment_eer_mixed(
                    frame_scores=fs,
                    score_frame_shift_ms=float(det_frame_shift_ms),
                    frame_labels=fl,
                    label_frame_shift_ms=LABEL_FRAME_SHIFT_MS,
                    resolution_ms=float(res),
                )
                seg_eers.append(eer_val)

            if seg_eers:
                mean_seg_eer = float(np.mean(seg_eers))
                seg_ci = bootstrap_ci(np.array(seg_eers), n_bootstrap=1000)
                det_results[f"seg_eer_{res}ms"] = mean_seg_eer
                det_results[f"seg_eer_{res}ms_ci"] = seg_ci
                print(f"  Seg-EER@{res}ms: {mean_seg_eer:.4f} "
                      f"(95% CI: {seg_ci[0]:.4f}–{seg_ci[1]:.4f})")
            else:
                print(f"  Seg-EER@{res}ms: N/A (no spoofed utterances)")

        # ── Segment F1 at detector's native resolution ───────────────
        # Binarize frame scores using the utterance EER threshold, then
        # upsample to the 10 ms label grid for fair comparison.
        all_preds_aligned: list[int] = []
        all_gts_aligned: list[int] = []
        for fs, fl in zip(all_frame_scores, all_frame_labels):
            if len(fs) == 0 or len(fl) == 0:
                continue
            pred_binary = (fs > utt_thresh).astype(int)
            pred_at_label_grid = upsample_binary_predictions_to_label_grid(
                pred_binary,
                pred_frame_shift_ms=float(det_frame_shift_ms),
                label_frame_shift_ms=LABEL_FRAME_SHIFT_MS,
            )
            min_len = min(len(pred_at_label_grid), len(fl))
            all_preds_aligned.extend(pred_at_label_grid[:min_len].tolist())
            all_gts_aligned.extend(fl[:min_len].tolist())

        if all_preds_aligned:
            seg_f1 = compute_segment_f1(
                np.array(all_preds_aligned), np.array(all_gts_aligned)
            )
            det_results[f"seg_f1_{det_frame_shift_ms}ms"] = float(seg_f1)
            print(f"  Seg-F1@{det_frame_shift_ms}ms (native): {seg_f1:.4f}")

        results[det_name] = det_results

    # ── Persist results JSON ─────────────────────────────────────────
    output_file = output_dir / "results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=_json_serializer)
    print(f"\nResults saved to {output_file}")

    return precomputed


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    run_e1()
