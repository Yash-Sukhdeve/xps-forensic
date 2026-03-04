"""E7: MFA vs WhisperX Alignment Quality.

Compare phoneme boundaries from MFA and WhisperX.
Quantify alignment error on bona fide vs synthesized segments.
Report impact on PDSM-PS faithfulness (Phoneme-IoU delta).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from xps_forensic.utils.config import load_config
from xps_forensic.utils.stats import bootstrap_ci


def run_e7(cfg=None):
    if cfg is None:
        cfg = load_config()

    output_dir = Path(cfg.experiments.output_dir) / "e7_alignment"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("E7: MFA vs WhisperX Alignment Quality")
    print("Steps:")
    print("  1. Run MFA on PartialSpoof eval set")
    print("  2. Run WhisperX on same data")
    print("  3. Compare phoneme boundaries")
    print("  4. Measure boundary accuracy on bona fide vs synthesized")
    print("  5. Compute PDSM-PS faithfulness with each aligner")

    results = {
        "aligners": ["mfa", "whisperx"],
        "status": "ready_to_run",
    }

    output_file = output_dir / "results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Config saved to {output_file}")


if __name__ == "__main__":
    run_e7()
