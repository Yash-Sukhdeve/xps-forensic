#!/bin/bash
# Train SAL (Segment-Aware Learning) detector on PartialSpoof.
# Produces a checkpoint for E1 evaluation.
#
# SAL repo: external/SAL/ (never modified)
# Override data path via Hydra CLI override.
#
# Usage:
#   bash scripts/train_sal.sh              # default 60 epochs
#   bash scripts/train_sal.sh --wait-for-e1  # wait for E1 to finish first
set -euo pipefail

SAL_DIR="/media/lab2208/ssd/Explainablility/xps_forensic/external/SAL"
PS_ROOT="/media/lab2208/ssd/datasets/PartialSpoof"
XPS_DIR="/media/lab2208/ssd/Explainablility/xps_forensic"

# Wait for E1 if requested
if [[ "${1:-}" == "--wait-for-e1" ]]; then
    echo "Waiting for E1 to finish..."
    while pgrep -f "run_e1_baseline" > /dev/null 2>&1; do
        sleep 60
    done
    echo "E1 finished. Starting SAL training."
fi

echo "=== SAL Training ==="
echo "SAL dir: ${SAL_DIR}"
echo "Data: ${PS_ROOT}"

# Check dependencies
if ! python -c "import lightning" 2>/dev/null; then
    echo "ERROR: PyTorch Lightning not installed"
    exit 1
fi

cd "${SAL_DIR}"

# Train SAL with WavLM on PartialSpoof
# Override data root via Hydra
python src/train.py \
    experiment=SAL_WavLM_PS \
    data.root="${PS_ROOT}" \
    data.batch_size=16 \
    data.num_workers=8 \
    trainer.max_epochs=60 \
    trainer.devices=1

echo "=== SAL Training Complete ==="
echo "Checkpoint should be in: ${SAL_DIR}/logs/train/runs/"
ls -la "${SAL_DIR}/logs/train/runs/" 2>/dev/null | tail -5
