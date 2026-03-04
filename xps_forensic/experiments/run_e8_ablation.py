"""E8: Ablation Studies.

- CPSL: +-calibration pre-step; max vs logsumexp; frame vs segment conformal
- PDSM-PS: IG vs GradSHAP; phoneme vs word aggregation; MFA vs WhisperX
- Detectors: BAM vs SAL saliency (boundary-focused vs boundary-debiased)
- Pipeline: single detector vs ensemble agreement
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from xps_forensic.utils.config import load_config
from xps_forensic.utils.stats import bootstrap_ci


ABLATION_CONFIGS = {
    "cpsl_no_calibration": {
        "description": "CPSL without calibration pre-step",
        "layer1": False,
    },
    "cpsl_max_vs_logsumexp": {
        "description": "Compare max and logsumexp nonconformity scores",
        "methods": ["max", "logsumexp"],
    },
    "pdsm_ig_vs_gradshap": {
        "description": "Compare IG and GradSHAP saliency methods",
        "methods": ["ig", "gradshap"],
    },
    "pdsm_phoneme_vs_word": {
        "description": "Compare phoneme-level and word-level aggregation",
        "levels": ["phoneme", "word"],
    },
    "detector_bam_vs_sal_saliency": {
        "description": "BAM (boundary-aware) vs SAL (boundary-debiased) saliency patterns",
        "detectors": ["bam", "sal"],
    },
}


def run_e8(cfg=None):
    if cfg is None:
        cfg = load_config()

    output_dir = Path(cfg.experiments.output_dir) / "e8_ablation"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("E8: Ablation Studies")
    for name, config in ABLATION_CONFIGS.items():
        print(f"  - {name}: {config['description']}")

    results = {
        "ablations": ABLATION_CONFIGS,
        "status": "ready_to_run",
    }

    output_file = output_dir / "results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Config saved to {output_file}")


if __name__ == "__main__":
    run_e8()
