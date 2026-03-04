"""E1: Baseline Detection & Localization.

Reproduces published results for all 4 detectors on PartialSpoof eval set.
Reports Utt-EER, Seg-EER at multiple resolutions, Seg-F1 with bootstrap CIs.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from xps_forensic.utils.config import load_config
from xps_forensic.utils.metrics import (
    compute_eer,
    compute_segment_eer,
    compute_segment_f1,
)
from xps_forensic.utils.stats import bootstrap_ci
from xps_forensic.data.partialspoof import PartialSpoofDataset
from xps_forensic.detectors.bam import BAMDetector
from xps_forensic.detectors.sal import SALDetector
from xps_forensic.detectors.cfprf import CFPRFDetector
from xps_forensic.detectors.mrm import MRMDetector


DETECTOR_MAP = {
    "bam": BAMDetector,
    "sal": SALDetector,
    "cfprf": CFPRFDetector,
    "mrm": MRMDetector,
}


def run_e1(cfg=None):
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
    print(f"Loaded PartialSpoof eval: {len(dataset)} utterances")

    results = {}
    resolutions = cfg.experiments.resolutions_ms

    for det_name in ["bam", "sal", "cfprf", "mrm"]:
        print(f"\n{'='*60}")
        print(f"Detector: {det_name.upper()}")
        print(f"{'='*60}")

        DetClass = DETECTOR_MAP[det_name]
        detector = DetClass(
            checkpoint=cfg.detectors[det_name].get("checkpoint"),
            external_dir=f"external/{det_name.upper()}",
            device=cfg.device,
        )
        detector.load_model()

        all_utt_scores = []
        all_utt_labels = []
        all_frame_scores = []
        all_frame_labels = []

        for sample in dataset:
            output = detector.predict(sample.waveform, sample.sample_rate)
            output.utterance_id = sample.utterance_id

            all_utt_scores.append(output.utterance_score)
            all_utt_labels.append(min(sample.utterance_label, 1))  # binary
            all_frame_scores.append(output.frame_scores)
            all_frame_labels.append(sample.frame_labels)

        utt_scores = np.array(all_utt_scores)
        utt_labels = np.array(all_utt_labels)

        # Utterance EER
        utt_eer, utt_thresh = compute_eer(utt_scores, utt_labels)
        utt_eer_ci = bootstrap_ci(
            (utt_scores > utt_thresh).astype(int) != utt_labels,
            n_bootstrap=1000,
        )

        det_results = {
            "utt_eer": utt_eer,
            "utt_eer_ci": utt_eer_ci,
        }

        # Segment EER at each resolution
        for res in resolutions:
            seg_eers = []
            for fs, fl in zip(all_frame_scores, all_frame_labels):
                if len(fl) > 0 and fl.any():
                    eer, _ = compute_segment_eer(fs, fl, resolution_ms=res)
                    seg_eers.append(eer)
            if seg_eers:
                mean_seg_eer = np.mean(seg_eers)
                ci = bootstrap_ci(np.array(seg_eers), n_bootstrap=1000)
                det_results[f"seg_eer_{res}ms"] = mean_seg_eer
                det_results[f"seg_eer_{res}ms_ci"] = ci

        # Segment F1 at 160ms
        all_preds = []
        all_gts = []
        for fs, fl in zip(all_frame_scores, all_frame_labels):
            pred = (fs > utt_thresh).astype(int)
            min_len = min(len(pred), len(fl))
            all_preds.extend(pred[:min_len].tolist())
            all_gts.extend(fl[:min_len].tolist())
        det_results["seg_f1_160ms"] = compute_segment_f1(
            np.array(all_preds), np.array(all_gts)
        )

        results[det_name] = det_results
        print(f"  Utt-EER: {utt_eer:.4f} ({utt_eer_ci[0]:.4f}-{utt_eer_ci[1]:.4f})")
        for res in resolutions:
            key = f"seg_eer_{res}ms"
            if key in det_results:
                print(f"  Seg-EER@{res}ms: {det_results[key]:.4f}")

    # Save results
    output_file = output_dir / "results.json"

    # Convert tuples to lists for JSON serialization
    def serialize(obj):
        if isinstance(obj, tuple):
            return list(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return obj

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=serialize)
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    run_e1()
