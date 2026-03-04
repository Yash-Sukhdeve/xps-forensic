#!/bin/bash
# Run all XPS-Forensic experiments in sequence
# Usage: bash scripts/run_all.sh [GPU_ID]

set -euo pipefail

GPU="${1:-0}"
export CUDA_VISIBLE_DEVICES="$GPU"

echo "============================================"
echo "  XPS-Forensic: Full Experiment Pipeline"
echo "  GPU: $GPU"
echo "============================================"

RESULTS="./results"
mkdir -p "$RESULTS"

echo ""
echo ">>> E1: Baseline Detection & Localization"
python experiments/run_e1_baseline.py

echo ""
echo ">>> E2: Post-hoc Calibration Comparison"
python experiments/run_e2_calibration.py

echo ""
echo ">>> E3: CPSL Coverage & Efficiency"
python experiments/run_e3_cpsl.py

echo ""
echo ">>> E4: PDSM-PS Faithfulness"
python experiments/run_e4_pdsm.py

echo ""
echo ">>> E5: Cross-Dataset Generalization"
python experiments/run_e5_cross_dataset.py

echo ""
echo ">>> E6: Codec Stress Test"
python experiments/run_e6_codec.py

echo ""
echo ">>> E7: MFA vs WhisperX Alignment"
python experiments/run_e7_alignment.py

echo ""
echo ">>> E8: Ablation Studies"
python experiments/run_e8_ablation.py

echo ""
echo "============================================"
echo "  All experiments complete!"
echo "  Results in: $RESULTS/"
echo "============================================"
