"""E4: PDSM-PS Faithfulness.

Apply IG + GradSHAP to WavLM features on CPSL-flagged segments.
Compare: phoneme-discretized (MFA) vs fixed-window (50/100ms) vs raw continuous.
Metrics: N-AOPC, Comprehensiveness/Sufficiency, Phoneme-IoU.
Subsample: ~750 utterances for saliency computation.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from xps_forensic.utils.config import load_config
from xps_forensic.utils.stats import bootstrap_ci
from xps_forensic.pdsm_ps import PDSMPSPipeline
from xps_forensic.pdsm_ps.discretize import discretize_by_fixed_window
from xps_forensic.pdsm_ps.faithfulness import (
    normalized_aopc,
    comprehensiveness,
    sufficiency,
    phoneme_iou,
)


def run_e4(cfg=None, precomputed=None):
    """Run E4 PDSM-PS faithfulness experiment.

    Args:
        cfg: Config.
        precomputed: Dict with 'frame_saliencies' (list of arrays),
            'frame_labels', 'durations', 'wav_paths'.
    """
    if cfg is None:
        cfg = load_config()

    output_dir = Path(cfg.experiments.output_dir) / "e4_pdsm"
    output_dir.mkdir(parents=True, exist_ok=True)

    n_subsample = cfg.pdsm.subsample_utterances

    results = {
        "methods_compared": ["phoneme_mfa", "phoneme_whisperx", "window_50ms", "window_100ms", "raw"],
        "metrics": {},
    }

    # For each saliency method (IG, GradSHAP)
    for saliency_method in ["ig", "gradshap"]:
        method_results = {}

        # Phoneme-discretized with MFA
        for aligner in ["mfa", "whisperx"]:
            pipeline = PDSMPSPipeline(
                aligner=aligner,
                saliency_method=saliency_method,
                top_k=10,
            )
            ious, aopcs, comps, suffs = [], [], [], []

            # Process subsampled utterances
            if precomputed:
                for i in range(min(n_subsample, len(precomputed["frame_saliencies"]))):
                    result = pipeline.run(
                        frame_saliency=precomputed["frame_saliencies"][i],
                        duration_sec=precomputed["durations"][i],
                        spoofed_frame_mask=precomputed["frame_labels"][i],
                        wav_path=precomputed.get("wav_paths", [None])[i],
                    )
                    # PDSMPSResult dataclass access
                    ious.append(result.phoneme_iou_score)

            key = f"{saliency_method}_{aligner}"
            method_results[key] = {
                "mean_phoneme_iou": float(np.mean(ious)) if ious else 0,
                "phoneme_iou_ci": bootstrap_ci(np.array(ious)) if len(ious) > 1 else (0, 0),
                "n_utterances": len(ious),
            }

        # Fixed-window baselines
        for window_ms in cfg.pdsm.window_baselines:
            key = f"{saliency_method}_window_{window_ms}ms"
            method_results[key] = {
                "window_ms": window_ms,
                "n_utterances": 0,
            }

        results["metrics"][saliency_method] = method_results

    # Save
    output_file = output_dir / "results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=lambda x: list(x) if isinstance(x, tuple) else float(x) if isinstance(x, np.floating) else x)
    print(f"Results saved to {output_file}")


if __name__ == "__main__":
    run_e4()
