"""E5: Cross-Dataset Generalization.

Run available detectors on cross-domain datasets (LlamaPartialSpoof,
PartialEdit, HQ-MPSD-EN) to evaluate generalization under domain shift.

Reports: Utt-EER, Seg-EER at multiple resolutions, ECE (uncalibrated and
calibrated if calibrators provided), CPSL coverage validity.

Datasets are loaded opportunistically — only those with data on disk are
evaluated. PartialEdit and HQ-MPSD-EN are expected to be unavailable in
the current setup; LlamaPartialSpoof (~140K utterances) is always available.

Reference: Tibshirani et al. (NeurIPS 2019) for covariate-shift CP context.

Output layout (under results/e5_cross_dataset/):
    results.json              — all computed metrics
    {det}_{ds}_utt_scores.npy — utterance-level scores per detector+dataset
    {det}_{ds}_utt_labels.npy — utterance-level labels per detector+dataset
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
from xps_forensic.calibration.metrics import expected_calibration_error
from xps_forensic.data.partialedit import PartialEditDataset
from xps_forensic.data.hqmpsd import HQMPSDDataset
from xps_forensic.data.llamapartialspoof import LlamaPartialSpoofDataset
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

# Label frame shift in all cross-datasets (10 ms).
LABEL_FRAME_SHIFT_MS = 10.0

# Default frame shifts per detector (used when building from config).
_DEFAULT_FRAME_SHIFT_MS = {
    "bam": 160.0,
    "sal": 160.0,
    "cfprf": 20.0,
    "mrm": 20.0,
}

# Cross-dataset registry: (config_key, DatasetClass, constructor_kwargs_fn)
_CROSS_DATASETS = [
    (
        "llamapartialspoof",
        LlamaPartialSpoofDataset,
        lambda cfg: {"root": cfg.data.llamapartialspoof.path},
    ),
    (
        "partialedit",
        PartialEditDataset,
        lambda cfg: {"root": cfg.data.partialedit.path},
    ),
    (
        "hqmpsd",
        HQMPSDDataset,
        lambda cfg: {
            "root": cfg.data.hqmpsd.path,
            "language": cfg.data.hqmpsd.get("language", "en"),
        },
    ),
]


def _build_detector(det_name: str, det_cfg: dict, device: str):
    """Instantiate a detector from config, returning None if checkpoint missing.

    Mirrors the E1 pattern for detector construction.
    """
    DetClass = DETECTOR_MAP[det_name]
    checkpoint = det_cfg.get("checkpoint")

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

    kwargs: dict = {
        "checkpoint": str(ckpt_path),
        "device": device,
    }

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


def _load_cross_datasets(cfg) -> dict:
    """Load available cross-domain datasets, skipping those without data.

    Returns dict of {name: dataset_instance} for datasets that have
    at least one utterance on disk.
    """
    datasets: dict = {}
    for ds_name, DatasetClass, kwargs_fn in _CROSS_DATASETS:
        try:
            kwargs = kwargs_fn(cfg)
        except AttributeError:
            logger.info(
                "Dataset %s: config entry missing — skipping.", ds_name
            )
            continue

        root = Path(kwargs["root"])
        if not root.exists():
            logger.info(
                "Dataset %s: path %s does not exist — skipping.",
                ds_name,
                root,
            )
            continue

        try:
            ds = DatasetClass(**kwargs)
        except Exception as exc:
            logger.warning(
                "Dataset %s: failed to instantiate — %s. Skipping.",
                ds_name,
                exc,
            )
            continue

        n = len(ds)
        if n == 0:
            logger.info(
                "Dataset %s: loaded but contains 0 utterances — skipping.",
                ds_name,
            )
            continue

        datasets[ds_name] = ds
        print(f"  Loaded {ds_name}: {n} utterances")

    return datasets


def _resolve_detectors(cfg, detectors: dict | None) -> dict:
    """Resolve detector dict: use provided or build from config.

    Returns dict of {name: detector_instance} for available detectors.
    """
    if detectors:
        return detectors

    print("No detectors passed; building from config...")
    built: dict = {}
    for det_name in ["bam", "sal", "cfprf", "mrm"]:
        det_cfg = cfg.detectors.get(det_name, {})
        if not det_cfg:
            continue
        detector = _build_detector(det_name, det_cfg, cfg.device)
        if detector is not None:
            built[det_name] = detector
    return built


def run_e5(
    cfg=None,
    detectors: dict | None = None,
    calibrators: dict | None = None,
    cpsl_pipeline=None,
    max_utterances: int | None = None,
):
    """Run E5 cross-dataset generalization experiment.

    Parameters
    ----------
    cfg : optional
        Config dict. Loaded from default.yaml if None.
    detectors : dict or None
        Dict of {name: loaded_detector} from E1, or None to build from config.
    calibrators : dict or None
        Dict of {name: fitted_calibrator} from E2, for ECE evaluation.
        If None, only uncalibrated ECE is reported.
    cpsl_pipeline : optional
        Fitted CPSLPipeline from E3, for coverage testing under domain shift.
        If None, CPSL coverage is not evaluated.
    max_utterances : int or None
        Maximum number of utterances to process per dataset. Useful for
        testing/debugging with the full 140K LlamaPartialSpoof corpus.
        If None, uses value from config (experiments.max_utterances_e5),
        or processes all utterances if that config key is absent.

    Returns
    -------
    dict
        Results dict keyed by dataset name, then detector name.
    """
    if cfg is None:
        cfg = load_config()

    output_dir = Path(cfg.experiments.output_dir) / "e5_cross_dataset"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve max_utterances from argument, config, or default (unlimited)
    if max_utterances is None:
        max_utterances = cfg.experiments.get("max_utterances_e5", None)

    resolutions = cfg.experiments.resolutions_ms

    # ── Load cross-domain datasets ────────────────────────────────
    print("Loading cross-domain datasets...")
    datasets = _load_cross_datasets(cfg)

    if not datasets:
        logger.warning("No cross-domain datasets available — nothing to evaluate.")
        # Save empty results to record which datasets were checked
        result = {"available_datasets": [], "note": "no data found"}
        output_file = output_dir / "results.json"
        with open(output_file, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Empty results saved to {output_file}")
        return result

    # ── Resolve detectors ─────────────────────────────────────────
    det_dict = _resolve_detectors(cfg, detectors)

    if not det_dict:
        logger.warning(
            "No detectors available — recording dataset availability only."
        )
        result = {
            "available_datasets": list(datasets.keys()),
            "available_detectors": [],
            "note": "no detectors available (checkpoints missing?)",
        }
        output_file = output_dir / "results.json"
        with open(output_file, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Results saved to {output_file}")
        return result

    # ── Main evaluation loop ──────────────────────────────────────
    all_results: dict = {
        "available_datasets": list(datasets.keys()),
        "available_detectors": list(det_dict.keys()),
    }

    for ds_name, dataset in datasets.items():
        print(f"\n{'='*60}")
        n_total = len(dataset)
        n_eval = min(n_total, max_utterances) if max_utterances else n_total
        print(f"Dataset: {ds_name} ({n_eval}/{n_total} utterances)")
        print(f"{'='*60}")

        ds_results: dict = {"n_utterances_total": n_total, "n_utterances_evaluated": n_eval}

        for det_name, detector in det_dict.items():
            print(f"\n  Detector: {det_name.upper()}")

            det_frame_shift_ms = float(detector.frame_shift_ms)
            print(f"    Frame shift: {det_frame_shift_ms} ms")

            # ── Inference ────────────────────────────────────────
            all_utt_scores: list[float] = []
            all_utt_labels: list[int] = []
            all_frame_scores: list[np.ndarray] = []
            all_frame_labels: list[np.ndarray] = []

            t0 = time.time()
            for idx in range(n_eval):
                sample = dataset[idx]
                output = detector.predict(
                    sample.waveform,
                    sample.sample_rate,
                    utterance_id=sample.utterance_id,
                )

                all_utt_scores.append(output.utterance_score)
                # Binary utterance label: 0=real, >=1 means contains spoof
                all_utt_labels.append(min(sample.utterance_label, 1))
                all_frame_scores.append(output.frame_scores)
                all_frame_labels.append(sample.frame_labels)

                if (idx + 1) % 500 == 0:
                    elapsed = time.time() - t0
                    print(
                        f"    Processed {idx + 1}/{n_eval} "
                        f"({elapsed:.1f}s elapsed)"
                    )

            elapsed = time.time() - t0
            print(f"    Inference complete: {n_eval} utterances in {elapsed:.1f}s")

            utt_scores = np.array(all_utt_scores)
            utt_labels = np.array(all_utt_labels)

            # ── Save raw scores ──────────────────────────────────
            np.save(
                output_dir / f"{det_name}_{ds_name}_utt_scores.npy",
                utt_scores,
            )
            np.save(
                output_dir / f"{det_name}_{ds_name}_utt_labels.npy",
                utt_labels,
            )

            # ── Utterance-level EER ──────────────────────────────
            det_result: dict = {
                "frame_shift_ms": det_frame_shift_ms,
                "n_utterances": n_eval,
            }

            if len(np.unique(utt_labels)) < 2:
                print(f"    [WARN] Only one class present — cannot compute EER.")
                det_result["utt_eer"] = None
                det_result["note"] = "single class in labels"
            else:
                utt_eer, utt_thresh = compute_eer(utt_scores, utt_labels)
                utt_errors = (utt_scores > utt_thresh).astype(int) != utt_labels
                utt_eer_ci = bootstrap_ci(utt_errors.astype(float), n_bootstrap=1000)
                det_result["utt_eer"] = float(utt_eer)
                det_result["utt_eer_threshold"] = float(utt_thresh)
                det_result["utt_eer_ci"] = utt_eer_ci
                print(
                    f"    Utt-EER: {utt_eer:.4f} "
                    f"(95% CI: {utt_eer_ci[0]:.4f}-{utt_eer_ci[1]:.4f})"
                )

                # ── Segment-level EER at each resolution ─────────
                for res in resolutions:
                    seg_eers: list[float] = []
                    for fs, fl in zip(all_frame_scores, all_frame_labels):
                        if len(fl) == 0 or not fl.any():
                            continue
                        if len(fs) == 0:
                            continue
                        eer_val, _ = compute_segment_eer_mixed(
                            frame_scores=fs,
                            score_frame_shift_ms=det_frame_shift_ms,
                            frame_labels=fl,
                            label_frame_shift_ms=LABEL_FRAME_SHIFT_MS,
                            resolution_ms=float(res),
                        )
                        seg_eers.append(eer_val)

                    if seg_eers:
                        mean_seg_eer = float(np.mean(seg_eers))
                        seg_ci = bootstrap_ci(np.array(seg_eers), n_bootstrap=1000)
                        det_result[f"seg_eer_{res}ms"] = mean_seg_eer
                        det_result[f"seg_eer_{res}ms_ci"] = seg_ci
                        print(
                            f"    Seg-EER@{res}ms: {mean_seg_eer:.4f} "
                            f"(95% CI: {seg_ci[0]:.4f}-{seg_ci[1]:.4f})"
                        )
                    else:
                        print(f"    Seg-EER@{res}ms: N/A (no spoofed utterances)")

                # ── Segment F1 at native resolution ──────────────
                all_preds_aligned: list[int] = []
                all_gts_aligned: list[int] = []
                for fs, fl in zip(all_frame_scores, all_frame_labels):
                    if len(fs) == 0 or len(fl) == 0:
                        continue
                    pred_binary = (fs > utt_thresh).astype(int)
                    pred_at_label_grid = upsample_binary_predictions_to_label_grid(
                        pred_binary,
                        pred_frame_shift_ms=det_frame_shift_ms,
                        label_frame_shift_ms=LABEL_FRAME_SHIFT_MS,
                    )
                    min_len = min(len(pred_at_label_grid), len(fl))
                    all_preds_aligned.extend(pred_at_label_grid[:min_len].tolist())
                    all_gts_aligned.extend(fl[:min_len].tolist())

                if all_preds_aligned:
                    seg_f1 = compute_segment_f1(
                        np.array(all_preds_aligned), np.array(all_gts_aligned)
                    )
                    det_result[f"seg_f1_{det_frame_shift_ms}ms"] = float(seg_f1)
                    print(
                        f"    Seg-F1@{det_frame_shift_ms}ms (native): {seg_f1:.4f}"
                    )

            # ── ECE (uncalibrated) ───────────────────────────────
            ece_uncal = expected_calibration_error(utt_scores, utt_labels)
            det_result["ece_uncalibrated"] = float(ece_uncal)
            print(f"    ECE (uncalibrated): {ece_uncal:.4f}")

            # ── ECE (calibrated, if calibrators provided) ────────
            if calibrators and det_name in calibrators:
                calibrator = calibrators[det_name]
                try:
                    cal_scores = calibrator.transform(utt_scores)
                    ece_cal = expected_calibration_error(cal_scores, utt_labels)
                    det_result["ece_calibrated"] = float(ece_cal)
                    print(f"    ECE (calibrated): {ece_cal:.4f}")
                except Exception as exc:
                    logger.warning(
                        "Calibrator for %s failed: %s", det_name, exc
                    )
                    det_result["ece_calibrated"] = None

            # ── CPSL coverage (if pipeline provided) ─────────────
            if cpsl_pipeline is not None:
                try:
                    preds = cpsl_pipeline.predict(all_frame_scores)
                    covered = sum(
                        1
                        for p, y in zip(preds, utt_labels)
                        if y in p.prediction_set
                    )
                    cpsl_coverage = covered / len(utt_labels)
                    det_result["cpsl_coverage"] = float(cpsl_coverage)
                    det_result["cpsl_n_covered"] = covered
                    det_result["cpsl_n_total"] = len(utt_labels)
                    print(f"    CPSL coverage: {cpsl_coverage:.4f}")
                except Exception as exc:
                    logger.warning(
                        "CPSL prediction failed for %s on %s: %s",
                        det_name,
                        ds_name,
                        exc,
                    )
                    det_result["cpsl_coverage"] = None

            ds_results[det_name] = det_result

        all_results[ds_name] = ds_results

    # ── Save results ──────────────────────────────────────────────
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
    run_e5()
