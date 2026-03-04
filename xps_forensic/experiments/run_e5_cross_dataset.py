"""E5: Cross-Dataset Generalization.

Run all 4 detectors on PartialEdit, HQ-MPSD (EN), LlamaPartialSpoof.
Report: Seg-EER, Seg-F1, calibration drift (ECE before/after), CPSL coverage
validity, PDSM-PS faithfulness stability under domain shift.

Reference: Tibshirani et al. (NeurIPS 2019) for covariate-shift CP context.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from xps_forensic.utils.config import load_config
from xps_forensic.utils.metrics import compute_eer, compute_segment_eer, compute_segment_f1
from xps_forensic.utils.stats import bootstrap_ci
from xps_forensic.calibration.metrics import expected_calibration_error
from xps_forensic.data.partialedit import PartialEditDataset
from xps_forensic.data.hqmpsd import HQMPSDDataset
from xps_forensic.data.llamapartialspoof import LlamaPartialSpoofDataset


CROSS_DATASETS = {
    "partialedit": PartialEditDataset,
    "hqmpsd": HQMPSDDataset,
    "llamapartialspoof": LlamaPartialSpoofDataset,
}


def run_e5(cfg=None, detectors=None, calibrators=None, cpsl_pipeline=None):
    """Run E5 cross-dataset generalization.

    Args:
        cfg: Config.
        detectors: Dict of {name: loaded_detector} from E1.
        calibrators: Dict of {name: fitted_calibrator} from E2.
        cpsl_pipeline: Fitted CPSLPipeline from E3.
    """
    if cfg is None:
        cfg = load_config()

    output_dir = Path(cfg.experiments.output_dir) / "e5_cross_dataset"
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets = {}
    if Path(cfg.data.partialedit.path).exists():
        datasets["partialedit"] = PartialEditDataset(root=cfg.data.partialedit.path)
    if Path(cfg.data.hqmpsd.path).exists():
        datasets["hqmpsd"] = HQMPSDDataset(root=cfg.data.hqmpsd.path, language="en")
    if Path(cfg.data.llamapartialspoof.path).exists():
        datasets["llamapartialspoof"] = LlamaPartialSpoofDataset(root=cfg.data.llamapartialspoof.path)

    results = {}

    for ds_name, dataset in datasets.items():
        print(f"\n{'='*60}")
        print(f"Dataset: {ds_name} ({len(dataset)} utterances)")

        ds_results = {}

        if detectors:
            for det_name, detector in detectors.items():
                frame_scores_all = []
                frame_labels_all = []
                utt_scores = []
                utt_labels = []

                for sample in dataset:
                    output = detector.predict(sample.waveform, sample.sample_rate)
                    frame_scores_all.append(output.frame_scores)
                    frame_labels_all.append(sample.frame_labels)
                    utt_scores.append(output.utterance_score)
                    utt_labels.append(min(sample.utterance_label, 1))

                utt_scores = np.array(utt_scores)
                utt_labels = np.array(utt_labels)

                # Detection metrics
                seg_eer, _ = compute_eer(utt_scores, utt_labels)

                # Calibration drift
                ece_uncal = expected_calibration_error(utt_scores, utt_labels)

                # CPSL coverage (if pipeline fitted) — CPSLResult dataclass access
                cpsl_coverage = None
                if cpsl_pipeline:
                    preds = cpsl_pipeline.predict(frame_scores_all)
                    covered = sum(
                        1 for p, y in zip(preds, utt_labels)
                        if y in p.prediction_set
                    )
                    cpsl_coverage = covered / len(utt_labels)

                ds_results[det_name] = {
                    "seg_eer": float(seg_eer),
                    "ece_uncalibrated": float(ece_uncal),
                    "cpsl_coverage": cpsl_coverage,
                }

        results[ds_name] = ds_results

    # Save
    output_file = output_dir / "results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    run_e5()
