#!/usr/bin/env bash
# ===========================================================================
# download_datasets.sh — Dataset and detector-repo setup for XPS-Forensic
#
# Checks whether each required dataset directory exists under the shared
# dataset directory and prints download instructions if not.  Clones
# detector model repositories into ./external/.
# ===========================================================================
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA_DIR="/media/lab2208/ssd/datasets"
EXT_DIR="${PROJECT_ROOT}/external"

echo "=== XPS-Forensic: Dataset & Detector Setup ==="
echo "Project root : ${PROJECT_ROOT}"
echo "Data dir     : ${DATA_DIR}"
echo "External dir : ${EXT_DIR}"
echo ""

# -------------------------------------------------------------------
# 1. Datasets
# -------------------------------------------------------------------
check_dataset() {
    local name="$1"
    local subdir="$2"
    local instructions="$3"
    if [ -d "${DATA_DIR}/${subdir}" ]; then
        echo "[OK]   ${name} found at ${DATA_DIR}/${subdir}"
    else
        echo "[MISS] ${name} NOT found at ${DATA_DIR}/${subdir}"
        echo "       ${instructions}"
        echo ""
    fi
}

mkdir -p "${DATA_DIR}"

echo "--- Checking datasets ---"
echo ""

check_dataset "PartialSpoof" "PartialSpoof" \
    "Download from: https://zenodo.org/record/5766198  (Zhang et al., Interspeech 2022). Extract into ${DATA_DIR}/PartialSpoof/"

check_dataset "PartialEdit" "PartialEdit" \
    "Prepare PartialEdit corpus with audio/ and metadata.json. Place into ${DATA_DIR}/PartialEdit/"

check_dataset "HQ-MPSD (English)" "HQ-MPSD-EN" \
    "Download HQ-MPSD and extract the English subset into ${DATA_DIR}/HQ-MPSD-EN/"

check_dataset "LlamaPartialSpoof" "LlamaPartialSpoof" \
    "Download LlamaPartialSpoof corpus. Place into ${DATA_DIR}/LlamaPartialSpoof/"

echo ""

# -------------------------------------------------------------------
# 2. Detector repositories
# -------------------------------------------------------------------
clone_if_missing() {
    local name="$1"
    local repo_url="$2"
    local target="${EXT_DIR}/${name}"
    if [ -d "${target}" ]; then
        echo "[OK]   ${name} already cloned at ${target}"
    else
        echo "[CLONE] Cloning ${name} from ${repo_url} ..."
        git clone --depth 1 "${repo_url}" "${target}"
        echo "[OK]   ${name} cloned to ${target}"
    fi
}

mkdir -p "${EXT_DIR}"

echo "--- Cloning detector repositories ---"
echo ""

clone_if_missing "BAM"   "https://github.com/media-sec-lab/BAM"
clone_if_missing "SAL"   "https://github.com/SentryMao/SAL"
clone_if_missing "CFPRF" "https://github.com/ItzJuny/CFPRF"
clone_if_missing "MRM"   "https://github.com/hieuthi/MultiResoModel-Simple"

echo ""
echo "=== Setup complete ==="
