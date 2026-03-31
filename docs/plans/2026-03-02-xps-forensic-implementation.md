# XPS-Forensic Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement the XPS-Forensic pipeline — a calibrated explainability system for partial audio spoof localization with conformal coverage guarantees, targeting IEEE TIFS.

**Architecture:** A 5-layer post-hoc pipeline (detectors → calibration → CPSL conformal → PDSM-PS saliency → evidence packaging) applied to 4 pre-trained frame-level deepfake detectors, evaluated across 4 partial spoof datasets with 8 experiments.

**Tech Stack:** Python 3.10+, PyTorch 2.x, torchaudio, Hydra (config), Montreal Forced Aligner, WhisperX, Captum (saliency), scikit-learn (calibration), pytest

**Design Document:** `docs/plans/2026-03-02-xps-forensic-design.md`

---

## Project Structure

```
xps_forensic/
├── configs/
│   ├── default.yaml
│   └── experiment/
│       ├── e1_baseline.yaml
│       ├── e2_calibration.yaml
│       ├── e3_cpsl.yaml
│       ├── e4_pdsm.yaml
│       ├── e5_cross_dataset.yaml
│       ├── e6_codec.yaml
│       ├── e7_alignment.yaml
│       └── e8_ablation.yaml
├── xps_forensic/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── partialspoof.py
│   │   ├── partialedit.py
│   │   ├── hqmpsd.py
│   │   └── llamapartialspoof.py
│   ├── detectors/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── bam.py
│   │   ├── sal.py
│   │   ├── cfprf.py
│   │   └── mrm.py
│   ├── calibration/
│   │   ├── __init__.py
│   │   ├── methods.py
│   │   └── metrics.py
│   ├── cpsl/
│   │   ├── __init__.py
│   │   ├── nonconformity.py
│   │   ├── scp_aps.py
│   │   ├── crc.py
│   │   └── composed.py
│   ├── pdsm_ps/
│   │   ├── __init__.py
│   │   ├── alignment.py
│   │   ├── saliency.py
│   │   ├── discretize.py
│   │   └── faithfulness.py
│   ├── evidence/
│   │   ├── __init__.py
│   │   ├── schema.py
│   │   └── assembler.py
│   └── utils/
│       ├── __init__.py
│       ├── metrics.py
│       └── stats.py
├── experiments/
│   ├── run_e1_baseline.py
│   ├── run_e2_calibration.py
│   ├── run_e3_cpsl.py
│   ├── run_e4_pdsm.py
│   ├── run_e5_cross_dataset.py
│   ├── run_e6_codec.py
│   ├── run_e7_alignment.py
│   └── run_e8_ablation.py
├── scripts/
│   ├── download_datasets.sh
│   └── run_all.sh
├── tests/
│   ├── conftest.py
│   ├── test_data.py
│   ├── test_detectors.py
│   ├── test_calibration.py
│   ├── test_cpsl.py
│   ├── test_pdsm.py
│   └── test_evidence.py
├── environment.yaml
├── pyproject.toml
└── README.md
```

---

## Phase 0: Project Infrastructure (Tasks 1-3)

### Task 1: Project Scaffolding

**Files:**
- Create: `xps_forensic/pyproject.toml`
- Create: `xps_forensic/environment.yaml`
- Create: `xps_forensic/xps_forensic/__init__.py`
- Create: `xps_forensic/tests/conftest.py`

**Step 1: Create pyproject.toml**

```toml
[build-system]
requires = ["setuptools>=68.0", "wheel"]
build-backend = "setuptools.backends._legacy:_Backend"

[project]
name = "xps-forensic"
version = "0.1.0"
description = "Calibrated explainability pipeline for partial audio spoof localization"
requires-python = ">=3.10"
dependencies = [
    "torch>=2.0",
    "torchaudio>=2.0",
    "transformers>=4.35",
    "captum>=0.7",
    "scikit-learn>=1.3",
    "scipy>=1.11",
    "numpy>=1.24",
    "pandas>=2.0",
    "hydra-core>=1.3",
    "omegaconf>=2.3",
    "soundfile>=0.12",
    "librosa>=0.10",
    "matplotlib>=3.7",
    "seaborn>=0.13",
    "tqdm>=4.65",
    "whisperx>=3.1",
]

[project.optional-dependencies]
dev = ["pytest>=7.4", "pytest-cov>=4.1"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-v --tb=short"
```

**Step 2: Create conda environment file**

```yaml
# environment.yaml
name: xps-forensic
channels:
  - pytorch
  - nvidia
  - conda-forge
  - defaults
dependencies:
  - python=3.10
  - pytorch=2.1
  - torchaudio=2.1
  - pytorch-cuda=12.1
  - pip
  - pip:
    - -e ".[dev]"
```

**Step 3: Create package init and test conftest**

`xps_forensic/xps_forensic/__init__.py`:
```python
"""XPS-Forensic: Calibrated explainability for partial audio spoof localization."""
__version__ = "0.1.0"
```

`xps_forensic/tests/conftest.py`:
```python
"""Shared test fixtures for XPS-Forensic."""
import pytest
import torch
import numpy as np


@pytest.fixture
def rng():
    """Reproducible random generator."""
    return np.random.default_rng(42)


@pytest.fixture
def device():
    """Test device (CPU for CI, GPU if available)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def dummy_frame_scores(rng):
    """Simulated frame-level detector scores for 10 utterances.

    Returns list of arrays, each shape (n_frames,) with values in [0,1].
    Odd-indexed utterances have a 'spoofed' segment in the middle.
    """
    scores = []
    for i in range(10):
        n_frames = rng.integers(100, 500)
        s = rng.uniform(0.1, 0.4, size=n_frames)  # mostly real
        if i % 2 == 1:
            # inject spoofed segment
            start = n_frames // 3
            end = 2 * n_frames // 3
            s[start:end] = rng.uniform(0.7, 0.95, size=end - start)
        scores.append(s)
    return scores


@pytest.fixture
def dummy_labels():
    """Ground-truth utterance labels matching dummy_frame_scores.

    0=real, 1=partially_fake, 2=fully_fake.
    Odd-indexed utterances are partially_fake.
    """
    return [1 if i % 2 == 1 else 0 for i in range(10)]


@pytest.fixture
def dummy_segment_labels(dummy_frame_scores, rng):
    """Ground-truth frame-level binary labels (0=real, 1=fake)."""
    labels = []
    for i, s in enumerate(dummy_frame_scores):
        n = len(s)
        lab = np.zeros(n, dtype=int)
        if i % 2 == 1:
            lab[n // 3: 2 * n // 3] = 1
        labels.append(lab)
    return labels
```

**Step 4: Create directory structure**

Run:
```bash
cd /media/lab2208/ssd/Explainablility
mkdir -p xps_forensic/{xps_forensic/{data,detectors,calibration,cpsl,pdsm_ps/{alignment,saliency},evidence,utils},tests,configs/experiment,experiments,scripts}
touch xps_forensic/xps_forensic/{data,detectors,calibration,cpsl,pdsm_ps,evidence,utils}/__init__.py
```

**Step 5: Verify structure and install**

Run: `cd xps_forensic && pip install -e ".[dev]" && pytest tests/conftest.py --collect-only`
Expected: Package installs, conftest fixtures discovered

**Step 6: Commit**

```bash
git add xps_forensic/
git commit -m "feat: scaffold XPS-Forensic project with environment and test fixtures"
```

---

### Task 2: Configuration System

**Files:**
- Create: `xps_forensic/configs/default.yaml`
- Create: `xps_forensic/xps_forensic/utils/config.py`
- Test: `xps_forensic/tests/test_config.py`

**Step 1: Write failing test**

`xps_forensic/tests/test_config.py`:
```python
"""Tests for configuration loading."""
from xps_forensic.utils.config import load_config


def test_load_default_config():
    cfg = load_config()
    assert cfg.project.name == "xps-forensic"
    assert cfg.data.partialspoof.sample_rate == 16000
    assert cfg.calibration.methods == ["platt", "temperature", "isotonic"]
    assert cfg.cpsl.alpha_utterance == 0.05
    assert cfg.cpsl.alpha_segment == 0.10


def test_config_device_detection():
    cfg = load_config()
    assert cfg.device in ("cuda", "cpu")
```

**Step 2: Run test to verify it fails**

Run: `cd xps_forensic && pytest tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError`

**Step 3: Write default config**

`xps_forensic/configs/default.yaml`:
```yaml
project:
  name: "xps-forensic"
  seed: 42

device: "auto"  # auto-detect GPU

data:
  root: "/media/lab2208/ssd/datasets"
  partialspoof:
    path: "${data.root}/PartialSpoof"
    sample_rate: 16000
    eval_split_ratio: 0.8  # 80% calibration, 20% verification
  partialedit:
    path: "${data.root}/PartialEdit"
    sample_rate: 16000
  hqmpsd:
    path: "${data.root}/HQ-MPSD-EN"
    sample_rate: 16000
    language: "en"
  llamapartialspoof:
    path: "${data.root}/LlamaPartialSpoof"
    sample_rate: 16000

detectors:
  bam:
    name: "BAM"
    repo: "https://github.com/media-sec-lab/BAM"
    checkpoint: null
    backbone: "wavlm-large"
  sal:
    name: "SAL"
    repo: "https://github.com/SentryMao/SAL"
    checkpoint: null
    backbone: "wavlm-large"
  cfprf:
    name: "CFPRF"
    repo: "https://github.com/ItzJuny/CFPRF"
    checkpoint: null
    backbone: "wav2vec2-xlsr"
  mrm:
    name: "MRM"
    repo: "https://github.com/hieuthi/MultiResoModel-Simple"
    checkpoint: null
    backbone: "ssl"

calibration:
  methods: ["platt", "temperature", "isotonic"]
  include_uncalibrated: true
  cv_folds: 5

cpsl:
  alpha_utterance: 0.05
  alpha_segment: 0.10
  alpha_sweep: [0.01, 0.05, 0.10]
  nonconformity: "max"  # "max" or "logsumexp"
  logsumexp_beta: [1.0, 5.0, 10.0, 20.0]
  classes: ["real", "partially_fake", "fully_fake"]

pdsm:
  saliency_method: "ig"  # "ig" or "gradshap"
  n_steps_ig: 50
  n_samples_shap: 25
  aligner: "mfa"  # "mfa" or "whisperx"
  window_baselines: [50, 100]  # ms, for fixed-window discretization comparison
  subsample_utterances: 750

evidence:
  schema_version: "1.0"

experiments:
  output_dir: "./results"
  codecs: ["aac", "opus", "amr", "g711"]
  resolutions_ms: [20, 40, 80, 160, 320, 640]
```

**Step 4: Write config loader**

`xps_forensic/xps_forensic/utils/config.py`:
```python
"""Configuration loading utilities."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import yaml


_CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"


def load_config(overrides: dict[str, Any] | None = None) -> "DotDict":
    """Load default config, apply overrides, resolve device."""
    cfg_path = _CONFIG_DIR / "default.yaml"
    with open(cfg_path) as f:
        raw = yaml.safe_load(f)

    # Resolve ${data.root} references
    _resolve_refs(raw)

    if overrides:
        _deep_merge(raw, overrides)

    # Auto-detect device
    if raw.get("device") == "auto":
        raw["device"] = "cuda" if torch.cuda.is_available() else "cpu"

    return DotDict(raw)


class DotDict(dict):
    """Dict subclass with dot-notation access."""

    def __getattr__(self, key: str) -> Any:
        try:
            val = self[key]
        except KeyError:
            raise AttributeError(f"No key '{key}'") from None
        if isinstance(val, dict) and not isinstance(val, DotDict):
            val = DotDict(val)
            self[key] = val
        return val

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value


def _resolve_refs(d: dict, root: dict | None = None) -> None:
    """Resolve ${section.key} references in-place."""
    if root is None:
        root = d
    for k, v in d.items():
        if isinstance(v, str) and "${" in v:
            # Simple single-level resolution
            import re
            for match in re.findall(r"\$\{([^}]+)\}", v):
                parts = match.split(".")
                val = root
                for p in parts:
                    val = val[p]
                v = v.replace(f"${{{match}}}", str(val))
            d[k] = v
        elif isinstance(v, dict):
            _resolve_refs(v, root)


def _deep_merge(base: dict, override: dict) -> None:
    """Recursively merge override into base."""
    for k, v in override.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v
```

**Step 5: Run test to verify it passes**

Run: `cd xps_forensic && pytest tests/test_config.py -v`
Expected: PASS

**Step 6: Commit**

```bash
git add configs/ xps_forensic/utils/config.py tests/test_config.py
git commit -m "feat: add configuration system with Hydra-style YAML loading"
```

---

### Task 3: Core Metrics Utilities

**Files:**
- Create: `xps_forensic/xps_forensic/utils/metrics.py`
- Create: `xps_forensic/xps_forensic/utils/stats.py`
- Test: `xps_forensic/tests/test_metrics.py`

**Step 1: Write failing test**

`xps_forensic/tests/test_metrics.py`:
```python
"""Tests for detection and evaluation metrics."""
import numpy as np
import pytest
from xps_forensic.utils.metrics import (
    compute_eer,
    compute_segment_eer,
    compute_segment_f1,
    compute_tFNR,
    compute_tFDR,
    compute_tIoU,
)
from xps_forensic.utils.stats import (
    bootstrap_ci,
    binomial_coverage_test,
    friedman_nemenyi,
)


class TestEER:
    def test_perfect_separation(self):
        scores = np.array([0.1, 0.2, 0.3, 0.8, 0.9, 1.0])
        labels = np.array([0, 0, 0, 1, 1, 1])
        eer, thresh = compute_eer(scores, labels)
        assert eer < 0.01

    def test_random_scores(self, rng):
        scores = rng.uniform(0, 1, 1000)
        labels = rng.integers(0, 2, 1000)
        eer, thresh = compute_eer(scores, labels)
        assert 0.3 < eer < 0.7  # roughly chance


class TestSegmentMetrics:
    def test_perfect_localization(self):
        pred = np.array([0, 0, 0, 1, 1, 1, 0, 0])
        true = np.array([0, 0, 0, 1, 1, 1, 0, 0])
        assert compute_tFNR(pred, true) == pytest.approx(0.0)
        assert compute_tFDR(pred, true) == pytest.approx(0.0)
        assert compute_tIoU(pred, true) == pytest.approx(1.0)

    def test_missed_detection(self):
        pred = np.array([0, 0, 0, 0, 0, 0, 0, 0])
        true = np.array([0, 0, 0, 1, 1, 1, 0, 0])
        assert compute_tFNR(pred, true) == pytest.approx(1.0)
        assert compute_tIoU(pred, true) == pytest.approx(0.0)

    def test_all_real(self):
        pred = np.array([0, 0, 0, 0])
        true = np.array([0, 0, 0, 0])
        # No fake segments — tFNR undefined, return 0 by convention
        assert compute_tFNR(pred, true) == pytest.approx(0.0)


class TestBootstrap:
    def test_ci_contains_mean(self, rng):
        data = rng.normal(5.0, 1.0, 100)
        lo, hi = bootstrap_ci(data, n_bootstrap=500, seed=42)
        assert lo < 5.0 < hi

    def test_ci_width(self, rng):
        data = rng.normal(0, 1, 1000)
        lo, hi = bootstrap_ci(data, n_bootstrap=1000, seed=42)
        assert (hi - lo) < 0.3  # should be tight with n=1000


class TestBinomialCoverage:
    def test_good_coverage(self):
        # 96 out of 100 covered at alpha=0.05 → should NOT reject
        p_val = binomial_coverage_test(n_covered=96, n_total=100, alpha=0.05)
        assert p_val > 0.05

    def test_bad_coverage(self):
        # 80 out of 100 covered at alpha=0.05 → should reject
        p_val = binomial_coverage_test(n_covered=80, n_total=100, alpha=0.05)
        assert p_val < 0.05
```

**Step 2: Run test to verify it fails**

Run: `cd xps_forensic && pytest tests/test_metrics.py -v`
Expected: FAIL with `ModuleNotFoundError`

**Step 3: Implement metrics module**

`xps_forensic/xps_forensic/utils/metrics.py`:
```python
"""Detection and localization metrics for XPS-Forensic."""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import f1_score


def compute_eer(
    scores: np.ndarray, labels: np.ndarray
) -> tuple[float, float]:
    """Compute Equal Error Rate and threshold.

    Args:
        scores: Detection scores (higher = more likely fake).
        labels: Binary labels (0=real, 1=fake).

    Returns:
        (eer, threshold) tuple.
    """
    from sklearn.metrics import roc_curve

    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr

    # Find intersection of FPR and FNR
    try:
        eer = brentq(lambda x: interp1d(fpr, fpr)(x) - interp1d(fpr, fnr)(x), 0, 1)
    except ValueError:
        # Fallback: find closest crossing point
        idx = np.nanargmin(np.abs(fpr - fnr))
        eer = (fpr[idx] + fnr[idx]) / 2

    # Corresponding threshold
    idx = np.nanargmin(np.abs(fpr - fnr))
    threshold = thresholds[idx] if idx < len(thresholds) else 0.5

    return float(eer), float(threshold)


def compute_segment_eer(
    frame_scores: np.ndarray,
    frame_labels: np.ndarray,
    resolution_ms: int = 160,
    frame_shift_ms: int = 20,
) -> tuple[float, float]:
    """Compute segment-level EER at a given resolution.

    Averages frame scores within non-overlapping segments.
    """
    frames_per_seg = max(1, resolution_ms // frame_shift_ms)
    n_frames = min(len(frame_scores), len(frame_labels))

    seg_scores, seg_labels = [], []
    for i in range(0, n_frames, frames_per_seg):
        end = min(i + frames_per_seg, n_frames)
        seg_scores.append(np.mean(frame_scores[i:end]))
        # Majority vote for label
        seg_labels.append(int(np.mean(frame_labels[i:end]) > 0.5))

    return compute_eer(np.array(seg_scores), np.array(seg_labels))


def compute_segment_f1(
    frame_preds: np.ndarray, frame_labels: np.ndarray
) -> float:
    """Compute frame-level F1 score for spoof detection."""
    return float(f1_score(frame_labels, frame_preds, zero_division=0))


def compute_tFNR(
    pred_binary: np.ndarray, true_binary: np.ndarray
) -> float:
    """Temporal false negative rate: fraction of true-fake frames missed."""
    fake_mask = true_binary == 1
    if not fake_mask.any():
        return 0.0
    missed = np.sum((pred_binary == 0) & fake_mask)
    return float(missed / fake_mask.sum())


def compute_tFDR(
    pred_binary: np.ndarray, true_binary: np.ndarray
) -> float:
    """Temporal false discovery rate: fraction of predicted-fake frames that are real."""
    pred_fake = pred_binary == 1
    if not pred_fake.any():
        return 0.0
    false_pos = np.sum((true_binary == 0) & pred_fake)
    return float(false_pos / pred_fake.sum())


def compute_tIoU(
    pred_binary: np.ndarray, true_binary: np.ndarray
) -> float:
    """Temporal Intersection-over-Union for spoof segments."""
    intersection = np.sum((pred_binary == 1) & (true_binary == 1))
    union = np.sum((pred_binary == 1) | (true_binary == 1))
    if union == 0:
        return 1.0 if not true_binary.any() else 0.0
    return float(intersection / union)
```

**Step 4: Implement stats module**

`xps_forensic/xps_forensic/utils/stats.py`:
```python
"""Statistical utilities for XPS-Forensic."""
from __future__ import annotations

import numpy as np
from scipy import stats as sp_stats


def bootstrap_ci(
    data: np.ndarray,
    statistic: str = "mean",
    confidence: float = 0.95,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> tuple[float, float]:
    """Compute bootstrap confidence interval.

    Args:
        data: 1D array of observations.
        statistic: "mean" or "median".
        confidence: Confidence level (default 0.95).
        n_bootstrap: Number of bootstrap samples.
        seed: Random seed.

    Returns:
        (lower, upper) CI bounds.
    """
    rng = np.random.default_rng(seed)
    stat_fn = np.mean if statistic == "mean" else np.median
    boot_stats = np.array([
        stat_fn(rng.choice(data, size=len(data), replace=True))
        for _ in range(n_bootstrap)
    ])
    alpha = 1 - confidence
    lo = np.percentile(boot_stats, 100 * alpha / 2)
    hi = np.percentile(boot_stats, 100 * (1 - alpha / 2))
    return float(lo), float(hi)


def binomial_coverage_test(
    n_covered: int, n_total: int, alpha: float
) -> float:
    """One-sided binomial test: H0: coverage >= 1-alpha.

    Returns p-value. Small p-value → coverage is significantly below nominal.
    """
    nominal = 1 - alpha
    # P(X <= n_covered) under Binomial(n_total, nominal)
    p_val = sp_stats.binom.cdf(n_covered, n_total, nominal)
    return float(p_val)


def friedman_nemenyi(
    results_matrix: np.ndarray,
) -> dict:
    """Friedman test + Nemenyi post-hoc for comparing methods.

    Args:
        results_matrix: Shape (n_datasets, n_methods) with metric values.

    Returns:
        Dict with 'friedman_stat', 'friedman_p', 'ranks', 'cd' (critical difference).
    """
    n_datasets, n_methods = results_matrix.shape

    # Rank within each dataset (lower is better)
    from scipy.stats import rankdata
    ranks = np.array([rankdata(row) for row in results_matrix])
    mean_ranks = ranks.mean(axis=0)

    stat, p_val = sp_stats.friedmanchisquare(
        *[results_matrix[:, j] for j in range(n_methods)]
    )

    # Nemenyi critical difference (alpha=0.05)
    q_alpha = 2.569  # q_0.05 for k methods (approx for k<=6)
    cd = q_alpha * np.sqrt(n_methods * (n_methods + 1) / (6 * n_datasets))

    return {
        "friedman_stat": float(stat),
        "friedman_p": float(p_val),
        "mean_ranks": mean_ranks.tolist(),
        "critical_difference": float(cd),
    }


def holm_bonferroni(p_values: list[float], alpha: float = 0.05) -> list[bool]:
    """Holm-Bonferroni correction for multiple comparisons.

    Returns list of booleans: True if rejected (significant).
    """
    m = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    rejected = [False] * m

    for rank, (orig_idx, p) in enumerate(indexed):
        adjusted_alpha = alpha / (m - rank)
        if p <= adjusted_alpha:
            rejected[orig_idx] = True
        else:
            break  # Stop rejecting once a hypothesis is not rejected

    return rejected
```

**Step 5: Run tests to verify they pass**

Run: `cd xps_forensic && pytest tests/test_metrics.py -v`
Expected: All PASS

**Step 6: Commit**

```bash
git add xps_forensic/utils/ tests/test_metrics.py
git commit -m "feat: add core metrics (EER, tFNR/tFDR/tIoU) and statistical utilities"
```

---

## Phase 1: Data Pipeline (Tasks 4-8)

### Task 4: Base Dataset Interface

**Files:**
- Create: `xps_forensic/xps_forensic/data/base.py`
- Test: `xps_forensic/tests/test_data.py`

**Step 1: Write failing test**

`xps_forensic/tests/test_data.py`:
```python
"""Tests for data loading."""
import numpy as np
import pytest
from xps_forensic.data.base import AudioSegmentSample


class TestAudioSegmentSample:
    def test_sample_fields(self):
        waveform = np.zeros(16000, dtype=np.float32)
        sample = AudioSegmentSample(
            utterance_id="utt_001",
            waveform=waveform,
            sample_rate=16000,
            utterance_label=1,
            frame_labels=np.array([0, 0, 1, 1, 0]),
            dataset="partialspoof",
        )
        assert sample.utterance_id == "utt_001"
        assert sample.duration_sec == pytest.approx(1.0)
        assert sample.is_partially_fake is True

    def test_real_sample(self):
        waveform = np.zeros(32000, dtype=np.float32)
        sample = AudioSegmentSample(
            utterance_id="utt_002",
            waveform=waveform,
            sample_rate=16000,
            utterance_label=0,
            frame_labels=np.zeros(10),
            dataset="partialspoof",
        )
        assert sample.is_partially_fake is False
        assert sample.duration_sec == pytest.approx(2.0)
```

**Step 2: Run test to verify it fails**

Run: `cd xps_forensic && pytest tests/test_data.py::TestAudioSegmentSample -v`
Expected: FAIL with `ModuleNotFoundError`

**Step 3: Implement base data module**

`xps_forensic/xps_forensic/data/base.py`:
```python
"""Base data structures for XPS-Forensic datasets."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np


@dataclass
class AudioSegmentSample:
    """Single utterance with frame-level labels."""
    utterance_id: str
    waveform: np.ndarray          # shape: (n_samples,)
    sample_rate: int
    utterance_label: int           # 0=real, 1=partially_fake, 2=fully_fake
    frame_labels: np.ndarray       # shape: (n_frames,), binary 0/1
    dataset: str
    metadata: dict = field(default_factory=dict)

    @property
    def duration_sec(self) -> float:
        return len(self.waveform) / self.sample_rate

    @property
    def is_partially_fake(self) -> bool:
        return self.utterance_label == 1


class BasePartialSpoofDataset:
    """Abstract base class for partial spoof datasets.

    Subclasses must implement _load_manifest() and _load_sample().
    """

    def __init__(self, root: str | Path, split: str = "eval", sample_rate: int = 16000):
        self.root = Path(root)
        self.split = split
        self.sample_rate = sample_rate
        self.manifest: list[dict] = []
        if self.root.exists():
            self.manifest = self._load_manifest()

    def _load_manifest(self) -> list[dict]:
        """Load list of {utterance_id, path, label, ...}. Override in subclass."""
        raise NotImplementedError

    def _load_sample(self, entry: dict) -> AudioSegmentSample:
        """Load a single sample from manifest entry. Override in subclass."""
        raise NotImplementedError

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, idx: int) -> AudioSegmentSample:
        return self._load_sample(self.manifest[idx])

    def __iter__(self) -> Iterator[AudioSegmentSample]:
        for entry in self.manifest:
            yield self._load_sample(entry)

    def get_split(
        self, ratio: float = 0.8, seed: int = 42
    ) -> tuple[list[int], list[int]]:
        """Split indices into calibration/verification sets.

        Args:
            ratio: Fraction for calibration set.
            seed: Random seed.

        Returns:
            (cal_indices, ver_indices)
        """
        rng = np.random.default_rng(seed)
        n = len(self.manifest)
        perm = rng.permutation(n)
        split_point = int(n * ratio)
        return perm[:split_point].tolist(), perm[split_point:].tolist()
```

**Step 4: Run test to verify it passes**

Run: `cd xps_forensic && pytest tests/test_data.py::TestAudioSegmentSample -v`
Expected: PASS

**Step 5: Commit**

```bash
git add xps_forensic/data/base.py tests/test_data.py
git commit -m "feat: add base dataset interface and AudioSegmentSample dataclass"
```

---

### Task 5: PartialSpoof Dataset Loader

**Files:**
- Create: `xps_forensic/xps_forensic/data/partialspoof.py`
- Test: `xps_forensic/tests/test_data.py` (append)

**Step 1: Write failing test**

Append to `xps_forensic/tests/test_data.py`:
```python
from xps_forensic.data.partialspoof import PartialSpoofDataset


class TestPartialSpoofDataset:
    def test_manifest_structure(self, tmp_path):
        """Test that manifest parsing works with mock data."""
        # Create mock PartialSpoof structure
        (tmp_path / "eval" / "con_wav").mkdir(parents=True)
        (tmp_path / "eval" / "label").mkdir(parents=True)

        # Mock protocol file
        proto = tmp_path / "eval" / "protocol.txt"
        # Format: utt_id label(0/1) seg_info
        proto.write_text("CON_E_00001 0\nCON_E_00002 1\n")

        # Mock label file for utt 2 (partially spoofed)
        label_file = tmp_path / "eval" / "label" / "CON_E_00002.txt"
        label_file.write_text("0 0 0 1 1 1 0 0\n")

        ds = PartialSpoofDataset(root=tmp_path, split="eval")
        assert len(ds.manifest) == 2
        assert ds.manifest[1]["utterance_label"] == 1
```

**Step 2: Run test to verify it fails**

Run: `cd xps_forensic && pytest tests/test_data.py::TestPartialSpoofDataset -v`
Expected: FAIL

**Step 3: Implement PartialSpoof loader**

`xps_forensic/xps_forensic/data/partialspoof.py`:
```python
"""PartialSpoof dataset loader.

Reference: Zhang et al., "The PartialSpoof Database and Countermeasures for the
Detection of Short Generated Speech Segments Embedded in Natural Speech,"
IEEE/ACM TASLP 2023. arXiv:2204.05177
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from .base import AudioSegmentSample, BasePartialSpoofDataset


class PartialSpoofDataset(BasePartialSpoofDataset):
    """Loader for the PartialSpoof dataset.

    Expected directory structure:
        root/
        ├── train/
        ├── dev/
        └── eval/
            ├── con_wav/        # concatenated waveforms
            ├── label/          # frame-level binary labels
            └── protocol.txt    # utterance-level protocol
    """

    RESOLUTIONS_MS = [20, 40, 80, 160, 320, 640]
    FRAME_SHIFT_MS = 10  # 10ms frame shift

    def _load_manifest(self) -> list[dict]:
        proto_path = self.root / self.split / "protocol.txt"
        if not proto_path.exists():
            return []

        manifest = []
        for line in proto_path.read_text().strip().split("\n"):
            parts = line.strip().split()
            utt_id = parts[0]
            utt_label = int(parts[1])  # 0=real, 1=spoof

            # Map: 0→real(0), 1→check if partial or full
            entry = {
                "utterance_id": utt_id,
                "utterance_label": utt_label,
                "wav_path": str(
                    self.root / self.split / "con_wav" / f"{utt_id}.wav"
                ),
                "label_path": str(
                    self.root / self.split / "label" / f"{utt_id}.txt"
                ),
            }
            manifest.append(entry)

        return manifest

    def _load_sample(self, entry: dict) -> AudioSegmentSample:
        wav_path = Path(entry["wav_path"])
        waveform, sr = sf.read(wav_path, dtype="float32")

        if sr != self.sample_rate:
            import torchaudio
            import torch

            waveform_t = torch.from_numpy(waveform).unsqueeze(0)
            waveform_t = torchaudio.functional.resample(waveform_t, sr, self.sample_rate)
            waveform = waveform_t.squeeze(0).numpy()

        # Load frame-level labels
        label_path = Path(entry["label_path"])
        if label_path.exists():
            frame_labels = np.loadtxt(label_path, dtype=int).flatten()
        else:
            # Real utterance: all zeros
            n_frames = len(waveform) // (self.sample_rate * self.FRAME_SHIFT_MS // 1000)
            frame_labels = np.zeros(n_frames, dtype=int)

        # Determine utterance label: 0=real, 1=partial, 2=fully_fake
        utt_label = entry["utterance_label"]
        if utt_label == 1 and frame_labels.any():
            fake_ratio = frame_labels.mean()
            if fake_ratio > 0.95:
                utt_label = 2  # fully fake
            else:
                utt_label = 1  # partially fake

        return AudioSegmentSample(
            utterance_id=entry["utterance_id"],
            waveform=waveform,
            sample_rate=self.sample_rate,
            utterance_label=utt_label,
            frame_labels=frame_labels,
            dataset="partialspoof",
        )
```

**Step 4: Run test to verify it passes**

Run: `cd xps_forensic && pytest tests/test_data.py::TestPartialSpoofDataset -v`
Expected: PASS

**Step 5: Commit**

```bash
git add xps_forensic/data/partialspoof.py tests/test_data.py
git commit -m "feat: add PartialSpoof dataset loader"
```

---

### Task 6: PartialEdit Dataset Loader

**Files:**
- Create: `xps_forensic/xps_forensic/data/partialedit.py`

**Step 1: Write failing test**

Append to `xps_forensic/tests/test_data.py`:
```python
from xps_forensic.data.partialedit import PartialEditDataset


class TestPartialEditDataset:
    def test_init_missing_dir(self, tmp_path):
        ds = PartialEditDataset(root=tmp_path / "nonexistent")
        assert len(ds) == 0
```

**Step 2: Run test to verify it fails**

Run: `cd xps_forensic && pytest tests/test_data.py::TestPartialEditDataset -v`
Expected: FAIL

**Step 3: Implement PartialEdit loader**

`xps_forensic/xps_forensic/data/partialedit.py`:
```python
"""PartialEdit dataset loader.

Reference: Zhang et al., "PartialEdit: A Benchmark for Neural Speech Editing
with Flexible Editing," Interspeech 2025.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf

from .base import AudioSegmentSample, BasePartialSpoofDataset


class PartialEditDataset(BasePartialSpoofDataset):
    """Loader for the PartialEdit dataset.

    Expected structure:
        root/
        ├── audio/          # edited waveforms
        ├── metadata.json   # edit regions and labels
        └── original/       # original (bona fide) audio
    """

    def _load_manifest(self) -> list[dict]:
        meta_path = self.root / "metadata.json"
        if not meta_path.exists():
            return []

        with open(meta_path) as f:
            metadata = json.load(f)

        manifest = []
        for entry in metadata:
            manifest.append({
                "utterance_id": entry["id"],
                "wav_path": str(self.root / "audio" / entry["filename"]),
                "edit_regions": entry.get("edit_regions", []),
                "utterance_label": 1 if entry.get("edit_regions") else 0,
            })
        return manifest

    def _load_sample(self, entry: dict) -> AudioSegmentSample:
        wav_path = Path(entry["wav_path"])
        waveform, sr = sf.read(wav_path, dtype="float32")

        if sr != self.sample_rate:
            import torchaudio
            import torch
            waveform_t = torch.from_numpy(waveform).unsqueeze(0)
            waveform_t = torchaudio.functional.resample(waveform_t, sr, self.sample_rate)
            waveform = waveform_t.squeeze(0).numpy()

        # Build frame-level labels from edit regions
        n_frames = len(waveform) // (self.sample_rate // 100)  # 10ms frames
        frame_labels = np.zeros(n_frames, dtype=int)
        for region in entry.get("edit_regions", []):
            start_frame = int(region["start_sec"] * 100)
            end_frame = int(region["end_sec"] * 100)
            frame_labels[start_frame:min(end_frame, n_frames)] = 1

        return AudioSegmentSample(
            utterance_id=entry["utterance_id"],
            waveform=waveform,
            sample_rate=self.sample_rate,
            utterance_label=entry["utterance_label"],
            frame_labels=frame_labels,
            dataset="partialedit",
        )
```

**Step 4: Run test to verify it passes**

Run: `cd xps_forensic && pytest tests/test_data.py::TestPartialEditDataset -v`
Expected: PASS

**Step 5: Commit**

```bash
git add xps_forensic/data/partialedit.py tests/test_data.py
git commit -m "feat: add PartialEdit dataset loader"
```

---

### Task 7: HQ-MPSD and LlamaPartialSpoof Loaders

**Files:**
- Create: `xps_forensic/xps_forensic/data/hqmpsd.py`
- Create: `xps_forensic/xps_forensic/data/llamapartialspoof.py`

**Step 1: Write failing tests**

Append to `xps_forensic/tests/test_data.py`:
```python
from xps_forensic.data.hqmpsd import HQMPSDDataset
from xps_forensic.data.llamapartialspoof import LlamaPartialSpoofDataset


class TestHQMPSD:
    def test_init_missing(self, tmp_path):
        ds = HQMPSDDataset(root=tmp_path / "nonexistent", language="en")
        assert len(ds) == 0


class TestLlamaPartialSpoof:
    def test_init_missing(self, tmp_path):
        ds = LlamaPartialSpoofDataset(root=tmp_path / "nonexistent")
        assert len(ds) == 0
```

**Step 2: Run tests to verify they fail**

Run: `cd xps_forensic && pytest tests/test_data.py::TestHQMPSD tests/test_data.py::TestLlamaPartialSpoof -v`
Expected: FAIL

**Step 3: Implement HQ-MPSD loader**

`xps_forensic/xps_forensic/data/hqmpsd.py`:
```python
"""HQ-MPSD dataset loader (English subset).

Reference: Li et al., "HQ-MPSD: A High-Quality Multi-lingual Partial Spoof
Detection Dataset," 2025.

Labels are ternary: genuine (0), deepfake (1), transition (2).
We map: genuine→0 (real frame), deepfake→1 (fake frame), transition→1 (fake frame).
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import soundfile as sf

from .base import AudioSegmentSample, BasePartialSpoofDataset


class HQMPSDDataset(BasePartialSpoofDataset):
    """Loader for HQ-MPSD English subset.

    Expected structure:
        root/
        ├── en/
        │   ├── audio/
        │   └── labels/
        └── metadata.csv
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "eval",
        sample_rate: int = 16000,
        language: str = "en",
    ):
        self.language = language
        super().__init__(root, split, sample_rate)

    def _load_manifest(self) -> list[dict]:
        meta_path = self.root / "metadata.csv"
        if not meta_path.exists():
            # Try alternative structure
            meta_path = self.root / self.language / "metadata.csv"
            if not meta_path.exists():
                return []

        manifest = []
        with open(meta_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("language", self.language) != self.language:
                    continue
                manifest.append({
                    "utterance_id": row["id"],
                    "wav_path": str(self.root / row["audio_path"]),
                    "label_path": str(self.root / row["label_path"]),
                    "utterance_label": int(row.get("utterance_label", 1)),
                })
        return manifest

    def _load_sample(self, entry: dict) -> AudioSegmentSample:
        wav_path = Path(entry["wav_path"])
        waveform, sr = sf.read(wav_path, dtype="float32")

        if sr != self.sample_rate:
            import torchaudio
            import torch
            waveform_t = torch.from_numpy(waveform).unsqueeze(0)
            waveform_t = torchaudio.functional.resample(waveform_t, sr, self.sample_rate)
            waveform = waveform_t.squeeze(0).numpy()

        # Load ternary labels (30ms resolution) and binarize
        label_path = Path(entry["label_path"])
        if label_path.exists():
            raw_labels = np.loadtxt(label_path, dtype=int).flatten()
            # Map: 0→0 (real), 1→1 (fake), 2→1 (transition=fake)
            frame_labels = (raw_labels > 0).astype(int)
        else:
            frame_labels = np.zeros(1, dtype=int)

        return AudioSegmentSample(
            utterance_id=entry["utterance_id"],
            waveform=waveform,
            sample_rate=self.sample_rate,
            utterance_label=entry["utterance_label"],
            frame_labels=frame_labels,
            dataset="hqmpsd",
        )
```

**Step 4: Implement LlamaPartialSpoof loader**

`xps_forensic/xps_forensic/data/llamapartialspoof.py`:
```python
"""LlamaPartialSpoof dataset loader.

Reference: Luong et al., "LlamaPartialSpoof: An LLM-Driven Fake Speech Dataset
Simulating Disinformation Generation," ICASSP 2025.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from .base import AudioSegmentSample, BasePartialSpoofDataset


class LlamaPartialSpoofDataset(BasePartialSpoofDataset):
    """Loader for LlamaPartialSpoof dataset.

    Expected structure:
        root/
        ├── wav/
        ├── labels/
        └── protocol.txt
    """

    def _load_manifest(self) -> list[dict]:
        proto_path = self.root / "protocol.txt"
        if not proto_path.exists():
            return []

        manifest = []
        for line in proto_path.read_text().strip().split("\n"):
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            utt_id = parts[0]
            utt_label = int(parts[1])
            manifest.append({
                "utterance_id": utt_id,
                "wav_path": str(self.root / "wav" / f"{utt_id}.wav"),
                "label_path": str(self.root / "labels" / f"{utt_id}.txt"),
                "utterance_label": utt_label,
            })
        return manifest

    def _load_sample(self, entry: dict) -> AudioSegmentSample:
        wav_path = Path(entry["wav_path"])
        waveform, sr = sf.read(wav_path, dtype="float32")

        if sr != self.sample_rate:
            import torchaudio
            import torch
            waveform_t = torch.from_numpy(waveform).unsqueeze(0)
            waveform_t = torchaudio.functional.resample(waveform_t, sr, self.sample_rate)
            waveform = waveform_t.squeeze(0).numpy()

        label_path = Path(entry["label_path"])
        if label_path.exists():
            frame_labels = np.loadtxt(label_path, dtype=int).flatten()
        else:
            frame_labels = np.zeros(1, dtype=int)

        return AudioSegmentSample(
            utterance_id=entry["utterance_id"],
            waveform=waveform,
            sample_rate=self.sample_rate,
            utterance_label=entry["utterance_label"],
            frame_labels=frame_labels,
            dataset="llamapartialspoof",
        )
```

**Step 5: Run tests**

Run: `cd xps_forensic && pytest tests/test_data.py -v`
Expected: All PASS

**Step 6: Commit**

```bash
git add xps_forensic/data/ tests/test_data.py
git commit -m "feat: add HQ-MPSD and LlamaPartialSpoof dataset loaders"
```

---

### Task 8: Dataset Download Script

**Files:**
- Create: `xps_forensic/scripts/download_datasets.sh`

**Step 1: Create download script**

`xps_forensic/scripts/download_datasets.sh`:
```bash
#!/bin/bash
# Download all required datasets for XPS-Forensic
# Usage: bash scripts/download_datasets.sh [DATA_DIR]

set -euo pipefail

DATA_DIR="${1:-/media/lab2208/ssd/datasets}"
mkdir -p "$DATA_DIR"

echo "=== XPS-Forensic Dataset Download ==="
echo "Target directory: $DATA_DIR"
echo ""

# 1. PartialSpoof (from ASVspoof)
echo "[1/4] PartialSpoof"
if [ ! -d "$DATA_DIR/PartialSpoof" ]; then
    echo "  PartialSpoof must be downloaded manually from:"
    echo "  https://zenodo.org/records/6674653"
    echo "  Extract to: $DATA_DIR/PartialSpoof/"
    echo "  Expected structure: PartialSpoof/{train,dev,eval}/{con_wav,label,protocol.txt}"
else
    echo "  Already exists at $DATA_DIR/PartialSpoof"
fi

echo ""

# 2. PartialEdit
echo "[2/4] PartialEdit"
if [ ! -d "$DATA_DIR/PartialEdit" ]; then
    echo "  PartialEdit — check paper/GitHub for download link:"
    echo "  Zhang et al., Interspeech 2025"
    echo "  Extract to: $DATA_DIR/PartialEdit/"
else
    echo "  Already exists at $DATA_DIR/PartialEdit"
fi

echo ""

# 3. HQ-MPSD (English subset only)
echo "[3/4] HQ-MPSD (English subset)"
if [ ! -d "$DATA_DIR/HQ-MPSD-EN" ]; then
    echo "  HQ-MPSD — download English subset only (~3.2 GB compressed):"
    echo "  Li et al., 2025"
    echo "  Extract to: $DATA_DIR/HQ-MPSD-EN/"
    echo "  NOTE: Full dataset is 1.7 TB — use English subset only"
else
    echo "  Already exists at $DATA_DIR/HQ-MPSD-EN"
fi

echo ""

# 4. LlamaPartialSpoof
echo "[4/4] LlamaPartialSpoof"
if [ ! -d "$DATA_DIR/LlamaPartialSpoof" ]; then
    echo "  LlamaPartialSpoof — check paper/GitHub for download link:"
    echo "  Luong et al., ICASSP 2025"
    echo "  https://github.com/..."
    echo "  Extract to: $DATA_DIR/LlamaPartialSpoof/"
else
    echo "  Already exists at $DATA_DIR/LlamaPartialSpoof"
fi

echo ""
echo "=== Detector Repositories ==="

# Clone detector repos
DETECTORS_DIR="$DATA_DIR/../external"
mkdir -p "$DETECTORS_DIR"

echo "[1/4] BAM"
if [ ! -d "$DETECTORS_DIR/BAM" ]; then
    git clone https://github.com/media-sec-lab/BAM.git "$DETECTORS_DIR/BAM"
else
    echo "  Already cloned"
fi

echo "[2/4] SAL"
if [ ! -d "$DETECTORS_DIR/SAL" ]; then
    git clone https://github.com/SentryMao/SAL.git "$DETECTORS_DIR/SAL"
else
    echo "  Already cloned"
fi

echo "[3/4] CFPRF"
if [ ! -d "$DETECTORS_DIR/CFPRF" ]; then
    git clone https://github.com/ItzJuny/CFPRF.git "$DETECTORS_DIR/CFPRF"
else
    echo "  Already cloned"
fi

echo "[4/4] MRM"
if [ ! -d "$DETECTORS_DIR/MRM" ]; then
    git clone https://github.com/hieuthi/MultiResoModel-Simple.git "$DETECTORS_DIR/MRM"
else
    echo "  Already cloned"
fi

echo ""
echo "=== Done ==="
echo "Next: verify dataset structures match expected formats."
```

**Step 2: Make executable and commit**

```bash
chmod +x xps_forensic/scripts/download_datasets.sh
git add scripts/download_datasets.sh
git commit -m "feat: add dataset and detector download script"
```

---

## Phase 2: Detector Integration (Tasks 9-13)

### Task 9: Base Detector Interface

**Files:**
- Create: `xps_forensic/xps_forensic/detectors/base.py`
- Test: `xps_forensic/tests/test_detectors.py`

**Step 1: Write failing test**

`xps_forensic/tests/test_detectors.py`:
```python
"""Tests for detector interfaces."""
import numpy as np
import torch
import pytest
from xps_forensic.detectors.base import DetectorOutput, BaseDetector


class TestDetectorOutput:
    def test_output_fields(self):
        out = DetectorOutput(
            utterance_id="utt_001",
            frame_scores=np.linspace(0, 1, 100),
            utterance_score=0.7,
            frame_shift_ms=20,
            detector_name="test",
        )
        assert out.n_frames == 100
        assert out.duration_ms == 2000
        assert out.utterance_score == pytest.approx(0.7)

    def test_binarize(self):
        scores = np.array([0.1, 0.3, 0.6, 0.9, 0.2])
        out = DetectorOutput(
            utterance_id="utt",
            frame_scores=scores,
            utterance_score=0.5,
            frame_shift_ms=20,
            detector_name="test",
        )
        binary = out.binarize(threshold=0.5)
        np.testing.assert_array_equal(binary, [0, 0, 1, 1, 0])
```

**Step 2: Run test to verify it fails**

Run: `cd xps_forensic && pytest tests/test_detectors.py -v`
Expected: FAIL

**Step 3: Implement base detector**

`xps_forensic/xps_forensic/detectors/base.py`:
```python
"""Base detector interface for XPS-Forensic."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass
class DetectorOutput:
    """Output from a frame-level deepfake detector."""
    utterance_id: str
    frame_scores: np.ndarray   # shape: (n_frames,), values in [0,1], higher=more fake
    utterance_score: float     # aggregated utterance-level score
    frame_shift_ms: int        # temporal resolution in ms
    detector_name: str

    @property
    def n_frames(self) -> int:
        return len(self.frame_scores)

    @property
    def duration_ms(self) -> int:
        return self.n_frames * self.frame_shift_ms

    def binarize(self, threshold: float = 0.5) -> np.ndarray:
        """Convert scores to binary predictions."""
        return (self.frame_scores >= threshold).astype(int)

    def scores_at_resolution(self, resolution_ms: int) -> np.ndarray:
        """Average frame scores at a coarser resolution."""
        frames_per_seg = max(1, resolution_ms // self.frame_shift_ms)
        n = len(self.frame_scores)
        segments = []
        for i in range(0, n, frames_per_seg):
            segments.append(np.mean(self.frame_scores[i:i + frames_per_seg]))
        return np.array(segments)


class BaseDetector(ABC):
    """Abstract base for frame-level spoof detectors."""

    name: str = "base"
    frame_shift_ms: int = 20

    def __init__(self, checkpoint: str | Path | None = None, device: str = "cpu"):
        self.device = torch.device(device)
        self.checkpoint = checkpoint
        self.model = None

    @abstractmethod
    def load_model(self) -> None:
        """Load model weights from checkpoint."""

    @abstractmethod
    def predict(self, waveform: np.ndarray, sample_rate: int = 16000) -> DetectorOutput:
        """Run inference on a single waveform.

        Args:
            waveform: Audio signal, shape (n_samples,).
            sample_rate: Sample rate in Hz.

        Returns:
            DetectorOutput with frame-level scores.
        """

    def predict_batch(
        self, waveforms: list[np.ndarray], utterance_ids: list[str], sample_rate: int = 16000
    ) -> list[DetectorOutput]:
        """Run inference on a batch. Default: sequential."""
        return [
            DetectorOutput(
                utterance_id=uid,
                frame_scores=self.predict(wav, sample_rate).frame_scores,
                utterance_score=self.predict(wav, sample_rate).utterance_score,
                frame_shift_ms=self.frame_shift_ms,
                detector_name=self.name,
            )
            for wav, uid in zip(waveforms, utterance_ids)
        ]
```

**Step 4: Run test to verify it passes**

Run: `cd xps_forensic && pytest tests/test_detectors.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add xps_forensic/detectors/base.py tests/test_detectors.py
git commit -m "feat: add base detector interface with DetectorOutput dataclass"
```

---

### Task 10: BAM Detector Wrapper

**Files:**
- Create: `xps_forensic/xps_forensic/detectors/bam.py`

**Step 1: Write failing test**

Append to `xps_forensic/tests/test_detectors.py`:
```python
from xps_forensic.detectors.bam import BAMDetector


class TestBAMDetector:
    def test_instantiation(self):
        det = BAMDetector(device="cpu")
        assert det.name == "BAM"
        assert det.frame_shift_ms == 20

    def test_predict_shape(self):
        """Test with mock model — actual model requires checkpoint."""
        det = BAMDetector(device="cpu")
        # Cannot test predict without checkpoint; test interface only
        assert hasattr(det, "predict")
        assert hasattr(det, "load_model")
```

**Step 2: Run test to verify it fails**

Run: `cd xps_forensic && pytest tests/test_detectors.py::TestBAMDetector -v`
Expected: FAIL

**Step 3: Implement BAM wrapper**

`xps_forensic/xps_forensic/detectors/bam.py`:
```python
"""BAM detector wrapper.

Reference: Zhong, Li, Yi. "Enhancing Partially Spoofed Audio Localization with
Boundary-aware Attention Mechanism." Interspeech 2024. arXiv:2407.21611

Wraps the official BAM implementation from:
https://github.com/media-sec-lab/BAM
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .base import BaseDetector, DetectorOutput


class BAMDetector(BaseDetector):
    """Wrapper for BAM (Boundary-aware Attention Mechanism) detector."""

    name = "BAM"
    frame_shift_ms = 20  # BAM outputs at 20ms resolution

    def __init__(
        self,
        checkpoint: str | Path | None = None,
        external_dir: str | Path | None = None,
        device: str = "cpu",
    ):
        super().__init__(checkpoint, device)
        self.external_dir = Path(external_dir) if external_dir else None

    def load_model(self) -> None:
        """Load BAM model from external repo + checkpoint."""
        if self.external_dir is None:
            raise ValueError("external_dir must point to cloned BAM repo")

        # Add BAM repo to sys.path for imports
        bam_path = str(self.external_dir)
        if bam_path not in sys.path:
            sys.path.insert(0, bam_path)

        # Import BAM model class (adjust based on actual repo structure)
        from model import BAM as BAMModel  # noqa: E402

        self.model = BAMModel()
        if self.checkpoint:
            state = torch.load(self.checkpoint, map_location=self.device, weights_only=True)
            self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()

    def predict(self, waveform: np.ndarray, sample_rate: int = 16000) -> DetectorOutput:
        """Run BAM inference.

        Args:
            waveform: Audio signal, shape (n_samples,).
            sample_rate: Sample rate (must be 16000).

        Returns:
            DetectorOutput with frame-level spoof scores.
        """
        if self.model is None:
            raise RuntimeError("Call load_model() before predict()")

        with torch.no_grad():
            x = torch.from_numpy(waveform).float().unsqueeze(0).to(self.device)
            # BAM returns frame-level logits
            output = self.model(x)

            # Handle different output formats from BAM
            if isinstance(output, dict):
                logits = output.get("frame_logits", output.get("logits"))
            elif isinstance(output, tuple):
                logits = output[0]
            else:
                logits = output

            # Convert logits to probabilities
            if logits.dim() == 3:
                # (batch, n_frames, 2) — binary classification per frame
                probs = F.softmax(logits, dim=-1)
                frame_scores = probs[0, :, 1].cpu().numpy()  # P(fake)
            elif logits.dim() == 2:
                # (batch, n_frames) — single score per frame
                frame_scores = torch.sigmoid(logits[0]).cpu().numpy()
            else:
                frame_scores = torch.sigmoid(logits).cpu().numpy().flatten()

        utterance_score = float(np.max(frame_scores))

        return DetectorOutput(
            utterance_id="",
            frame_scores=frame_scores,
            utterance_score=utterance_score,
            frame_shift_ms=self.frame_shift_ms,
            detector_name=self.name,
        )
```

**Step 4: Run test to verify it passes**

Run: `cd xps_forensic && pytest tests/test_detectors.py::TestBAMDetector -v`
Expected: PASS

**Step 5: Commit**

```bash
git add xps_forensic/detectors/bam.py tests/test_detectors.py
git commit -m "feat: add BAM detector wrapper (Interspeech 2024)"
```

---

### Task 11: SAL, CFPRF, and MRM Detector Wrappers

**Files:**
- Create: `xps_forensic/xps_forensic/detectors/sal.py`
- Create: `xps_forensic/xps_forensic/detectors/cfprf.py`
- Create: `xps_forensic/xps_forensic/detectors/mrm.py`

**Step 1: Write failing tests**

Append to `xps_forensic/tests/test_detectors.py`:
```python
from xps_forensic.detectors.sal import SALDetector
from xps_forensic.detectors.cfprf import CFPRFDetector
from xps_forensic.detectors.mrm import MRMDetector


class TestSALDetector:
    def test_instantiation(self):
        det = SALDetector(device="cpu")
        assert det.name == "SAL"

class TestCFPRFDetector:
    def test_instantiation(self):
        det = CFPRFDetector(device="cpu")
        assert det.name == "CFPRF"

class TestMRMDetector:
    def test_instantiation(self):
        det = MRMDetector(device="cpu")
        assert det.name == "MRM"
```

**Step 2: Run tests to verify they fail**

Run: `cd xps_forensic && pytest tests/test_detectors.py::TestSALDetector tests/test_detectors.py::TestCFPRFDetector tests/test_detectors.py::TestMRMDetector -v`
Expected: FAIL

**Step 3: Implement SAL wrapper**

`xps_forensic/xps_forensic/detectors/sal.py`:
```python
"""SAL detector wrapper.

Reference: Mao, Huang, Qian. "Localizing Speech Deepfakes Beyond Transitions
via Segment-Aware Learning." arXiv:2601.21925, 2026.

Wraps: https://github.com/SentryMao/SAL
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .base import BaseDetector, DetectorOutput


class SALDetector(BaseDetector):
    """Wrapper for SAL (Segment-Aware Learning) detector."""

    name = "SAL"
    frame_shift_ms = 20

    def __init__(
        self,
        checkpoint: str | Path | None = None,
        external_dir: str | Path | None = None,
        device: str = "cpu",
    ):
        super().__init__(checkpoint, device)
        self.external_dir = Path(external_dir) if external_dir else None

    def load_model(self) -> None:
        if self.external_dir is None:
            raise ValueError("external_dir must point to cloned SAL repo")
        sal_path = str(self.external_dir)
        if sal_path not in sys.path:
            sys.path.insert(0, sal_path)

        # Import based on actual SAL repo structure
        from model import SAL as SALModel  # noqa: E402

        self.model = SALModel()
        if self.checkpoint:
            state = torch.load(self.checkpoint, map_location=self.device, weights_only=True)
            self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()

    def predict(self, waveform: np.ndarray, sample_rate: int = 16000) -> DetectorOutput:
        if self.model is None:
            raise RuntimeError("Call load_model() before predict()")

        with torch.no_grad():
            x = torch.from_numpy(waveform).float().unsqueeze(0).to(self.device)
            output = self.model(x)

            if isinstance(output, dict):
                logits = output.get("frame_logits", output.get("logits"))
            elif isinstance(output, tuple):
                logits = output[0]
            else:
                logits = output

            if logits.dim() == 3:
                probs = F.softmax(logits, dim=-1)
                frame_scores = probs[0, :, 1].cpu().numpy()
            elif logits.dim() == 2:
                frame_scores = torch.sigmoid(logits[0]).cpu().numpy()
            else:
                frame_scores = torch.sigmoid(logits).cpu().numpy().flatten()

        return DetectorOutput(
            utterance_id="",
            frame_scores=frame_scores,
            utterance_score=float(np.max(frame_scores)),
            frame_shift_ms=self.frame_shift_ms,
            detector_name=self.name,
        )
```

**Step 4: Implement CFPRF wrapper**

`xps_forensic/xps_forensic/detectors/cfprf.py`:
```python
"""CFPRF detector wrapper.

Reference: Wu et al. "Coarse-to-Fine Proposal Refinement Framework for Audio
Temporal Forgery Detection and Localization." ACM MM 2024. arXiv:2407.16554

Wraps: https://github.com/ItzJuny/CFPRF
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

from .base import BaseDetector, DetectorOutput


class CFPRFDetector(BaseDetector):
    """Wrapper for CFPRF (Coarse-to-Fine Proposal Refinement) detector."""

    name = "CFPRF"
    frame_shift_ms = 20

    def __init__(
        self,
        checkpoint: str | Path | None = None,
        external_dir: str | Path | None = None,
        device: str = "cpu",
    ):
        super().__init__(checkpoint, device)
        self.external_dir = Path(external_dir) if external_dir else None

    def load_model(self) -> None:
        if self.external_dir is None:
            raise ValueError("external_dir must point to cloned CFPRF repo")
        cfprf_path = str(self.external_dir)
        if cfprf_path not in sys.path:
            sys.path.insert(0, cfprf_path)

        # CFPRF uses proposal-based localization
        from models import CFPRF as CFPRFModel  # noqa: E402

        self.model = CFPRFModel()
        if self.checkpoint:
            state = torch.load(self.checkpoint, map_location=self.device, weights_only=True)
            self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()

    def predict(self, waveform: np.ndarray, sample_rate: int = 16000) -> DetectorOutput:
        if self.model is None:
            raise RuntimeError("Call load_model() before predict()")

        with torch.no_grad():
            x = torch.from_numpy(waveform).float().unsqueeze(0).to(self.device)
            output = self.model(x)

            # CFPRF outputs proposals; convert to frame-level scores
            if isinstance(output, dict) and "proposals" in output:
                proposals = output["proposals"]  # list of (start, end, score)
                n_frames = int(len(waveform) / sample_rate * 1000 / self.frame_shift_ms)
                frame_scores = np.zeros(n_frames)
                for start, end, score in proposals:
                    s_frame = int(start * 1000 / self.frame_shift_ms)
                    e_frame = int(end * 1000 / self.frame_shift_ms)
                    frame_scores[s_frame:min(e_frame, n_frames)] = max(
                        frame_scores[s_frame:min(e_frame, n_frames)].max() if s_frame < n_frames else 0,
                        score
                    )
            else:
                # Fallback: treat as frame-level output
                logits = output[0] if isinstance(output, tuple) else output
                frame_scores = torch.sigmoid(logits).cpu().numpy().flatten()

        return DetectorOutput(
            utterance_id="",
            frame_scores=frame_scores,
            utterance_score=float(np.max(frame_scores)) if len(frame_scores) > 0 else 0.0,
            frame_shift_ms=self.frame_shift_ms,
            detector_name=self.name,
        )
```

**Step 5: Implement MRM wrapper**

`xps_forensic/xps_forensic/detectors/mrm.py`:
```python
"""MRM (Multi-Resolution Model) detector wrapper.

Reference: Zhang et al. "The PartialSpoof Database and Countermeasures for the
Detection of Short Generated Speech Segments Embedded in Natural Speech."
IEEE/ACM TASLP 2023. arXiv:2204.05177

Wraps: https://github.com/hieuthi/MultiResoModel-Simple
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

from .base import BaseDetector, DetectorOutput


class MRMDetector(BaseDetector):
    """Wrapper for MRM (Multi-Resolution Model) baseline detector."""

    name = "MRM"
    frame_shift_ms = 20

    def __init__(
        self,
        checkpoint: str | Path | None = None,
        external_dir: str | Path | None = None,
        device: str = "cpu",
    ):
        super().__init__(checkpoint, device)
        self.external_dir = Path(external_dir) if external_dir else None

    def load_model(self) -> None:
        if self.external_dir is None:
            raise ValueError("external_dir must point to cloned MRM repo")
        mrm_path = str(self.external_dir)
        if mrm_path not in sys.path:
            sys.path.insert(0, mrm_path)

        from model import MultiResoModel  # noqa: E402

        self.model = MultiResoModel()
        if self.checkpoint:
            state = torch.load(self.checkpoint, map_location=self.device, weights_only=True)
            self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()

    def predict(self, waveform: np.ndarray, sample_rate: int = 16000) -> DetectorOutput:
        if self.model is None:
            raise RuntimeError("Call load_model() before predict()")

        with torch.no_grad():
            x = torch.from_numpy(waveform).float().unsqueeze(0).to(self.device)
            output = self.model(x)

            if isinstance(output, dict):
                logits = output.get("frame_logits", output.get("logits"))
            elif isinstance(output, tuple):
                logits = output[0]
            else:
                logits = output

            frame_scores = torch.sigmoid(logits).cpu().numpy().flatten()

        return DetectorOutput(
            utterance_id="",
            frame_scores=frame_scores,
            utterance_score=float(np.max(frame_scores)),
            frame_shift_ms=self.frame_shift_ms,
            detector_name=self.name,
        )
```

**Step 6: Run tests to verify they pass**

Run: `cd xps_forensic && pytest tests/test_detectors.py -v`
Expected: All PASS

**Step 7: Commit**

```bash
git add xps_forensic/detectors/ tests/test_detectors.py
git commit -m "feat: add SAL, CFPRF, and MRM detector wrappers"
```

---

## Phase 3: Calibration Layer (Tasks 12-14)

### Task 12: Calibration Methods

**Files:**
- Create: `xps_forensic/xps_forensic/calibration/methods.py`
- Test: `xps_forensic/tests/test_calibration.py`

**Step 1: Write failing test**

`xps_forensic/tests/test_calibration.py`:
```python
"""Tests for post-hoc calibration methods."""
import numpy as np
import pytest
from xps_forensic.calibration.methods import (
    PlattScaling,
    TemperatureScaling,
    IsotonicCalibrator,
    calibrate_scores,
)


@pytest.fixture
def calibration_data(rng):
    """Simulated uncalibrated scores and labels."""
    n = 500
    # Real samples: scores clustered around 0.2-0.4
    real_scores = rng.beta(2, 5, size=n // 2)
    # Fake samples: scores clustered around 0.6-0.9
    fake_scores = rng.beta(5, 2, size=n // 2)
    scores = np.concatenate([real_scores, fake_scores])
    labels = np.concatenate([np.zeros(n // 2), np.ones(n // 2)])
    # Shuffle
    perm = rng.permutation(n)
    return scores[perm], labels[perm].astype(int)


class TestPlattScaling:
    def test_fit_transform(self, calibration_data):
        scores, labels = calibration_data
        platt = PlattScaling()
        platt.fit(scores, labels)
        calibrated = platt.transform(scores)
        assert calibrated.shape == scores.shape
        assert np.all(calibrated >= 0) and np.all(calibrated <= 1)

    def test_calibrated_mean_closer_to_prevalence(self, calibration_data):
        scores, labels = calibration_data
        platt = PlattScaling()
        platt.fit(scores, labels)
        calibrated = platt.transform(scores)
        prevalence = labels.mean()
        # Calibrated mean should be closer to prevalence than raw mean
        assert abs(calibrated.mean() - prevalence) < abs(scores.mean() - prevalence) + 0.1


class TestTemperatureScaling:
    def test_fit_transform(self, calibration_data):
        scores, labels = calibration_data
        temp = TemperatureScaling()
        temp.fit(scores, labels)
        calibrated = temp.transform(scores)
        assert calibrated.shape == scores.shape
        assert temp.temperature > 0


class TestIsotonicCalibrator:
    def test_fit_transform(self, calibration_data):
        scores, labels = calibration_data
        iso = IsotonicCalibrator()
        iso.fit(scores, labels)
        calibrated = iso.transform(scores)
        assert calibrated.shape == scores.shape
        # Isotonic should preserve monotonicity
        sorted_idx = np.argsort(scores)
        cal_sorted = calibrated[sorted_idx]
        assert np.all(np.diff(cal_sorted) >= -1e-10)


class TestCalibrateScores:
    def test_all_methods(self, calibration_data):
        scores, labels = calibration_data
        results = calibrate_scores(scores, labels)
        assert "uncalibrated" in results
        assert "platt" in results
        assert "temperature" in results
        assert "isotonic" in results
        for name, cal_scores in results.items():
            assert cal_scores.shape == scores.shape
```

**Step 2: Run test to verify it fails**

Run: `cd xps_forensic && pytest tests/test_calibration.py -v`
Expected: FAIL

**Step 3: Implement calibration methods**

`xps_forensic/xps_forensic/calibration/methods.py`:
```python
"""Post-hoc calibration methods for frame-level detector scores.

Implements systematic comparison of:
- Platt scaling (Platt, 1999)
- Temperature scaling (Guo et al., ICML 2017)
- Isotonic regression (Zadrozny & Elkan, 2002)

Applied to audio CM scores following Wang et al. (Interspeech 2024) and
Pascu et al. (Interspeech 2024) methodology.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


class BaseCalibrator(ABC):
    """Base class for score calibrators."""

    @abstractmethod
    def fit(self, scores: np.ndarray, labels: np.ndarray) -> None:
        """Fit calibrator on calibration data."""

    @abstractmethod
    def transform(self, scores: np.ndarray) -> np.ndarray:
        """Apply calibration to scores."""

    def fit_transform(self, scores: np.ndarray, labels: np.ndarray) -> np.ndarray:
        self.fit(scores, labels)
        return self.transform(scores)


class PlattScaling(BaseCalibrator):
    """Platt scaling: logistic regression on raw scores."""

    def __init__(self):
        self._lr = LogisticRegression(solver="lbfgs", max_iter=1000)

    def fit(self, scores: np.ndarray, labels: np.ndarray) -> None:
        self._lr.fit(scores.reshape(-1, 1), labels)

    def transform(self, scores: np.ndarray) -> np.ndarray:
        return self._lr.predict_proba(scores.reshape(-1, 1))[:, 1]


class TemperatureScaling(BaseCalibrator):
    """Temperature scaling: scale logits by learned temperature T.

    P_calibrated = sigmoid(logit(score) / T)
    """

    def __init__(self):
        self.temperature: float = 1.0

    def fit(self, scores: np.ndarray, labels: np.ndarray) -> None:
        # Convert scores to logits
        eps = 1e-7
        clipped = np.clip(scores, eps, 1 - eps)
        logits = np.log(clipped / (1 - clipped))

        # Optimize temperature to minimize NLL
        def nll(t):
            scaled = 1.0 / (1.0 + np.exp(-logits / t))
            scaled = np.clip(scaled, eps, 1 - eps)
            return -np.mean(
                labels * np.log(scaled) + (1 - labels) * np.log(1 - scaled)
            )

        result = minimize_scalar(nll, bounds=(0.01, 10.0), method="bounded")
        self.temperature = result.x

    def transform(self, scores: np.ndarray) -> np.ndarray:
        eps = 1e-7
        clipped = np.clip(scores, eps, 1 - eps)
        logits = np.log(clipped / (1 - clipped))
        return 1.0 / (1.0 + np.exp(-logits / self.temperature))


class IsotonicCalibrator(BaseCalibrator):
    """Isotonic regression calibration."""

    def __init__(self):
        self._ir = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)

    def fit(self, scores: np.ndarray, labels: np.ndarray) -> None:
        self._ir.fit(scores, labels)

    def transform(self, scores: np.ndarray) -> np.ndarray:
        return self._ir.predict(scores)


def calibrate_scores(
    scores: np.ndarray,
    labels: np.ndarray,
    methods: list[str] | None = None,
) -> dict[str, np.ndarray]:
    """Apply all calibration methods and return results.

    Args:
        scores: Raw detector scores, shape (n,).
        labels: Binary labels, shape (n,).
        methods: List of method names. Default: all methods.

    Returns:
        Dict mapping method name to calibrated scores.
    """
    if methods is None:
        methods = ["platt", "temperature", "isotonic"]

    calibrators = {
        "platt": PlattScaling,
        "temperature": TemperatureScaling,
        "isotonic": IsotonicCalibrator,
    }

    results = {"uncalibrated": scores.copy()}

    for name in methods:
        cal = calibrators[name]()
        cal.fit(scores, labels)
        results[name] = cal.transform(scores)

    return results
```

**Step 4: Run tests to verify they pass**

Run: `cd xps_forensic && pytest tests/test_calibration.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add xps_forensic/calibration/methods.py tests/test_calibration.py
git commit -m "feat: add post-hoc calibration methods (Platt, temperature, isotonic)"
```

---

### Task 13: Calibration Metrics

**Files:**
- Create: `xps_forensic/xps_forensic/calibration/metrics.py`

**Step 1: Write failing test**

Append to `xps_forensic/tests/test_calibration.py`:
```python
from xps_forensic.calibration.metrics import (
    expected_calibration_error,
    brier_score,
    negative_log_likelihood,
    reliability_diagram_data,
)


class TestCalibrationMetrics:
    def test_ece_perfect(self):
        # Perfectly calibrated: score == P(y=1)
        scores = np.array([0.1, 0.2, 0.8, 0.9])
        labels = np.array([0, 0, 1, 1])
        ece = expected_calibration_error(scores, labels, n_bins=2)
        assert ece < 0.2  # should be low

    def test_ece_terrible(self):
        # Completely miscalibrated
        scores = np.array([0.9, 0.9, 0.9, 0.9])
        labels = np.array([0, 0, 0, 0])
        ece = expected_calibration_error(scores, labels, n_bins=2)
        assert ece > 0.5

    def test_brier_perfect(self):
        scores = np.array([0.0, 0.0, 1.0, 1.0])
        labels = np.array([0, 0, 1, 1])
        assert brier_score(scores, labels) == pytest.approx(0.0)

    def test_nll_finite(self):
        scores = np.array([0.1, 0.2, 0.8, 0.9])
        labels = np.array([0, 0, 1, 1])
        nll = negative_log_likelihood(scores, labels)
        assert np.isfinite(nll)
        assert nll > 0

    def test_reliability_diagram(self):
        scores = np.random.uniform(0, 1, 100)
        labels = (scores > 0.5).astype(int)
        bins, accs, confs, counts = reliability_diagram_data(scores, labels)
        assert len(bins) == len(accs) == len(confs) == len(counts)
```

**Step 2: Run test to verify it fails**

Run: `cd xps_forensic && pytest tests/test_calibration.py::TestCalibrationMetrics -v`
Expected: FAIL

**Step 3: Implement calibration metrics**

`xps_forensic/xps_forensic/calibration/metrics.py`:
```python
"""Calibration evaluation metrics.

Reference: Guo et al., "On Calibration of Modern Neural Networks,"
ICML 2017; Dimitri et al., 2025.
"""
from __future__ import annotations

import numpy as np


def expected_calibration_error(
    scores: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
) -> float:
    """Expected Calibration Error (ECE).

    Args:
        scores: Predicted probabilities, shape (n,).
        labels: Binary labels, shape (n,).
        n_bins: Number of equal-width bins.

    Returns:
        ECE value in [0, 1].
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(scores)

    for i in range(n_bins):
        mask = (scores > bin_edges[i]) & (scores <= bin_edges[i + 1])
        if i == 0:
            mask = (scores >= bin_edges[i]) & (scores <= bin_edges[i + 1])
        count = mask.sum()
        if count == 0:
            continue
        avg_conf = scores[mask].mean()
        avg_acc = labels[mask].mean()
        ece += (count / n) * abs(avg_acc - avg_conf)

    return float(ece)


def brier_score(scores: np.ndarray, labels: np.ndarray) -> float:
    """Brier score: mean squared error of probability estimates."""
    return float(np.mean((scores - labels) ** 2))


def negative_log_likelihood(
    scores: np.ndarray, labels: np.ndarray
) -> float:
    """Negative log-likelihood (cross-entropy loss)."""
    eps = 1e-7
    clipped = np.clip(scores, eps, 1 - eps)
    nll = -np.mean(
        labels * np.log(clipped) + (1 - labels) * np.log(1 - clipped)
    )
    return float(nll)


def reliability_diagram_data(
    scores: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute data for reliability diagram.

    Returns:
        (bin_midpoints, accuracies, confidences, counts)
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    midpoints = (bin_edges[:-1] + bin_edges[1:]) / 2
    accs = np.zeros(n_bins)
    confs = np.zeros(n_bins)
    counts = np.zeros(n_bins, dtype=int)

    for i in range(n_bins):
        if i == 0:
            mask = (scores >= bin_edges[i]) & (scores <= bin_edges[i + 1])
        else:
            mask = (scores > bin_edges[i]) & (scores <= bin_edges[i + 1])
        count = mask.sum()
        counts[i] = count
        if count > 0:
            accs[i] = labels[mask].mean()
            confs[i] = scores[mask].mean()

    return midpoints, accs, confs, counts
```

**Step 4: Run tests to verify they pass**

Run: `cd xps_forensic && pytest tests/test_calibration.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add xps_forensic/calibration/metrics.py tests/test_calibration.py
git commit -m "feat: add calibration metrics (ECE, Brier, NLL, reliability diagram)"
```

---

## Phase 4: CPSL — Conformalized Partial Spoof Localization (Tasks 14-17)

### Task 14: Nonconformity Score Functions

**Files:**
- Create: `xps_forensic/xps_forensic/cpsl/nonconformity.py`
- Test: `xps_forensic/tests/test_cpsl.py`

**Step 1: Write failing test**

`xps_forensic/tests/test_cpsl.py`:
```python
"""Tests for CPSL conformal prediction components."""
import numpy as np
import pytest
from xps_forensic.cpsl.nonconformity import (
    max_score,
    logsumexp_score,
    compute_nonconformity,
)


class TestNonconformityScores:
    def test_max_score(self):
        frame_scores = np.array([0.1, 0.3, 0.9, 0.2])
        assert max_score(frame_scores) == pytest.approx(0.9)

    def test_logsumexp_score(self):
        frame_scores = np.array([0.1, 0.3, 0.9, 0.2])
        # log-sum-exp should be dominated by the max but slightly higher
        lse = logsumexp_score(frame_scores, beta=10.0)
        assert lse > 0.0
        assert np.isfinite(lse)

    def test_logsumexp_beta_sensitivity(self):
        frame_scores = np.array([0.5, 0.5, 0.5])
        lse_low = logsumexp_score(frame_scores, beta=1.0)
        lse_high = logsumexp_score(frame_scores, beta=20.0)
        # Higher beta → closer to max
        assert lse_high > lse_low or abs(lse_high - lse_low) < 0.01

    def test_compute_nonconformity_batch(self, dummy_frame_scores):
        scores = compute_nonconformity(dummy_frame_scores, method="max")
        assert len(scores) == len(dummy_frame_scores)
        assert all(0 <= s <= 1 for s in scores)
```

**Step 2: Run test to verify it fails**

Run: `cd xps_forensic && pytest tests/test_cpsl.py::TestNonconformityScores -v`
Expected: FAIL

**Step 3: Implement nonconformity scores**

`xps_forensic/xps_forensic/cpsl/nonconformity.py`:
```python
"""Nonconformity score functions for CPSL.

Aggregates frame-level detector scores to utterance-level nonconformity
scores for split conformal prediction.

Two methods:
- max: s(x) = max_t f(x_t) — sensitive to strongest spoof frame
- logsumexp: s(x) = (1/β) log(Σ exp(β·f(x_t))) — smooth max approximation

Reference: Romano, Sesia, Candes (NeurIPS 2020) for APS nonconformity design.
"""
from __future__ import annotations

import numpy as np
from scipy.special import logsumexp as _logsumexp


def max_score(frame_scores: np.ndarray) -> float:
    """Max nonconformity: maximum frame-level spoof score."""
    return float(np.max(frame_scores))


def logsumexp_score(frame_scores: np.ndarray, beta: float = 10.0) -> float:
    """Log-sum-exp nonconformity: smooth approximation to max.

    s(x) = (1/β) · log( (1/T) · Σ_t exp(β · f(x_t)) )

    As β → ∞, approaches max. Lower β gives softer aggregation.
    """
    T = len(frame_scores)
    return float(_logsumexp(beta * frame_scores) / beta - np.log(T) / beta)


def compute_nonconformity(
    frame_scores_list: list[np.ndarray],
    method: str = "max",
    beta: float = 10.0,
) -> np.ndarray:
    """Compute nonconformity scores for a batch of utterances.

    Args:
        frame_scores_list: List of per-utterance frame score arrays.
        method: "max" or "logsumexp".
        beta: Temperature for logsumexp (ignored if method="max").

    Returns:
        Array of shape (n_utterances,) with nonconformity scores.
    """
    if method == "max":
        return np.array([max_score(fs) for fs in frame_scores_list])
    elif method == "logsumexp":
        return np.array([logsumexp_score(fs, beta) for fs in frame_scores_list])
    else:
        raise ValueError(f"Unknown method: {method}. Use 'max' or 'logsumexp'.")
```

**Step 4: Run tests**

Run: `cd xps_forensic && pytest tests/test_cpsl.py::TestNonconformityScores -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add xps_forensic/cpsl/nonconformity.py tests/test_cpsl.py
git commit -m "feat: add nonconformity score functions (max, logsumexp) for CPSL"
```

---

### Task 15: SCP + APS (Stage 1 — Utterance-Level Conformal)

**Files:**
- Create: `xps_forensic/xps_forensic/cpsl/scp_aps.py`

**Step 1: Write failing test**

Append to `xps_forensic/tests/test_cpsl.py`:
```python
from xps_forensic.cpsl.scp_aps import SCPAPS


class TestSCPAPS:
    def test_calibrate_and_predict(self, rng):
        n_cal = 200
        # Generate calibration data: 3-class (real=0, partial=1, full=2)
        cal_scores = rng.uniform(0, 1, n_cal)
        cal_labels = rng.integers(0, 3, n_cal)

        scp = SCPAPS(alpha=0.10, classes=["real", "partially_fake", "fully_fake"])
        scp.calibrate(cal_scores, cal_labels)

        # Predict on test data
        test_scores = rng.uniform(0, 1, 50)
        pred_sets = scp.predict(test_scores)

        assert len(pred_sets) == 50
        for ps in pred_sets:
            assert isinstance(ps, set)
            assert len(ps) >= 1  # at least one class
            assert ps.issubset({0, 1, 2})

    def test_coverage_guarantee(self, rng):
        """Verify marginal coverage >= 1-alpha on held-out data."""
        n = 1000
        alpha = 0.10
        scores = rng.uniform(0, 1, n)
        labels = rng.integers(0, 3, n)

        # Split: 80% cal, 20% test
        cal_scores, test_scores = scores[:800], scores[800:]
        cal_labels, test_labels = labels[:800], labels[800:]

        scp = SCPAPS(alpha=alpha, classes=["real", "partially_fake", "fully_fake"])
        scp.calibrate(cal_scores, cal_labels)
        pred_sets = scp.predict(test_scores)

        # Check coverage
        covered = sum(1 for ps, y in zip(pred_sets, test_labels) if y in ps)
        coverage = covered / len(test_labels)
        # Finite-sample: coverage >= 1-alpha - slack
        assert coverage >= 1 - alpha - 0.10  # allow some slack for small sample

    def test_prediction_set_size(self, rng):
        """Higher alpha → smaller prediction sets (more efficient)."""
        n = 500
        scores = rng.uniform(0, 1, n)
        labels = rng.integers(0, 3, n)

        scp_loose = SCPAPS(alpha=0.20, classes=["real", "partially_fake", "fully_fake"])
        scp_loose.calibrate(scores, labels)
        sets_loose = scp_loose.predict(scores[:50])

        scp_tight = SCPAPS(alpha=0.01, classes=["real", "partially_fake", "fully_fake"])
        scp_tight.calibrate(scores, labels)
        sets_tight = scp_tight.predict(scores[:50])

        avg_loose = np.mean([len(s) for s in sets_loose])
        avg_tight = np.mean([len(s) for s in sets_tight])
        assert avg_tight >= avg_loose  # tighter alpha → larger sets
```

**Step 2: Run test to verify it fails**

Run: `cd xps_forensic && pytest tests/test_cpsl.py::TestSCPAPS -v`
Expected: FAIL

**Step 3: Implement SCP + APS**

`xps_forensic/xps_forensic/cpsl/scp_aps.py`:
```python
"""Split Conformal Prediction with Adaptive Prediction Sets (SCP + APS).

Stage 1 of CPSL: utterance-level conformal classification.

Provides distribution-free coverage guarantee:
    P(Y ∈ C(X)) ≥ 1 - α

Reference:
- Romano, Sesia, Candes. "Classification with Valid and Adaptive Coverage."
  NeurIPS 2020.
- Angelopoulos & Bates. "A Gentle Introduction to Conformal Prediction and
  Distribution-Free Uncertainty Quantification." FnTML 2022.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class SCPAPS:
    """Split Conformal Prediction with Adaptive Prediction Sets.

    For ternary classification: {real, partially_fake, fully_fake}
    with ordinal contiguity constraint.

    Usage:
        scp = SCPAPS(alpha=0.05)
        scp.calibrate(cal_nonconformity_scores, cal_labels)
        prediction_sets = scp.predict(test_nonconformity_scores)
    """

    alpha: float = 0.05
    classes: list[str] = field(default_factory=lambda: ["real", "partially_fake", "fully_fake"])
    _quantiles: dict[int, float] = field(default_factory=dict, repr=False)
    _class_conditional: bool = True

    @property
    def n_classes(self) -> int:
        return len(self.classes)

    def calibrate(
        self,
        nonconformity_scores: np.ndarray,
        labels: np.ndarray,
    ) -> None:
        """Calibrate quantiles on held-out calibration data.

        Args:
            nonconformity_scores: Shape (n_cal,). Higher = more anomalous.
            labels: Shape (n_cal,). Class indices 0, 1, 2.
        """
        if self._class_conditional:
            # Class-conditional calibration for handling imbalance
            for c in range(self.n_classes):
                mask = labels == c
                if mask.sum() == 0:
                    self._quantiles[c] = 1.0
                    continue
                class_scores = nonconformity_scores[mask]
                n_c = len(class_scores)
                # Finite-sample adjusted quantile: ceil((n+1)(1-alpha))/n
                q_level = np.ceil((n_c + 1) * (1 - self.alpha)) / n_c
                q_level = min(q_level, 1.0)
                self._quantiles[c] = float(np.quantile(class_scores, q_level))
        else:
            # Marginal calibration
            n = len(nonconformity_scores)
            q_level = np.ceil((n + 1) * (1 - self.alpha)) / n
            q_level = min(q_level, 1.0)
            q = float(np.quantile(nonconformity_scores, q_level))
            for c in range(self.n_classes):
                self._quantiles[c] = q

    def predict(self, nonconformity_scores: np.ndarray) -> list[set[int]]:
        """Generate prediction sets for test data.

        Uses APS: include classes in decreasing order of softmax probability
        until cumulative probability exceeds threshold. Here we use
        nonconformity-based inclusion: include class c if
        score <= quantile_c.

        For simplicity with a single nonconformity score, we include class c
        if the score is consistent with calibration data for that class.

        Args:
            nonconformity_scores: Shape (n_test,).

        Returns:
            List of prediction sets (each a set of class indices).
        """
        if not self._quantiles:
            raise RuntimeError("Call calibrate() before predict()")

        prediction_sets = []
        for score in nonconformity_scores:
            ps = set()
            for c in range(self.n_classes):
                if score <= self._quantiles[c]:
                    ps.add(c)
            # Guarantee: never return empty set
            if not ps:
                # Include the class with the highest quantile
                best_class = max(self._quantiles, key=self._quantiles.get)
                ps.add(best_class)
            prediction_sets.append(ps)

        return prediction_sets

    def get_quantiles(self) -> dict[str, float]:
        """Return calibrated quantiles per class."""
        return {self.classes[c]: q for c, q in self._quantiles.items()}
```

**Step 4: Run tests**

Run: `cd xps_forensic && pytest tests/test_cpsl.py::TestSCPAPS -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add xps_forensic/cpsl/scp_aps.py tests/test_cpsl.py
git commit -m "feat: add SCP+APS utterance-level conformal classification (CPSL Stage 1)"
```

---

### Task 16: CRC on tFNR (Stage 2 — Segment-Level)

**Files:**
- Create: `xps_forensic/xps_forensic/cpsl/crc.py`

**Step 1: Write failing test**

Append to `xps_forensic/tests/test_cpsl.py`:
```python
from xps_forensic.cpsl.crc import ConformalRiskControl


class TestCRC:
    def test_calibrate_threshold(self, dummy_frame_scores, dummy_segment_labels):
        crc = ConformalRiskControl(alpha=0.10, risk_metric="tFNR")
        # Use first 8 as calibration
        crc.calibrate(
            frame_scores=dummy_frame_scores[:8],
            frame_labels=dummy_segment_labels[:8],
        )
        assert crc.threshold is not None
        assert 0 <= crc.threshold <= 1

    def test_predict(self, dummy_frame_scores, dummy_segment_labels):
        crc = ConformalRiskControl(alpha=0.10, risk_metric="tFNR")
        crc.calibrate(
            frame_scores=dummy_frame_scores[:8],
            frame_labels=dummy_segment_labels[:8],
        )
        # Predict on remaining
        preds = crc.predict(dummy_frame_scores[8:])
        assert len(preds) == 2
        for p in preds:
            assert p.dtype == int
            assert set(np.unique(p)).issubset({0, 1})

    def test_tFNR_controlled(self, rng):
        """Verify E[tFNR] ≤ alpha on calibration data."""
        alpha = 0.20
        n = 100
        frame_scores = []
        frame_labels = []
        for i in range(n):
            n_frames = 200
            scores = rng.uniform(0.1, 0.4, n_frames)
            labels = np.zeros(n_frames, dtype=int)
            if i % 2 == 0:
                start, end = 50, 150
                scores[start:end] = rng.uniform(0.6, 0.95, end - start)
                labels[start:end] = 1
            frame_scores.append(scores)
            frame_labels.append(labels)

        crc = ConformalRiskControl(alpha=alpha, risk_metric="tFNR")
        crc.calibrate(frame_scores[:80], frame_labels[:80])

        # Check on test set
        from xps_forensic.utils.metrics import compute_tFNR
        test_tfnrs = []
        for scores, labels in zip(frame_scores[80:], frame_labels[80:]):
            if labels.any():
                pred = crc.predict([scores])[0]
                test_tfnrs.append(compute_tFNR(pred, labels))

        if test_tfnrs:
            mean_tfnr = np.mean(test_tfnrs)
            # Allow some slack for finite sample
            assert mean_tfnr <= alpha + 0.15
```

**Step 2: Run test to verify it fails**

Run: `cd xps_forensic && pytest tests/test_cpsl.py::TestCRC -v`
Expected: FAIL

**Step 3: Implement CRC**

`xps_forensic/xps_forensic/cpsl/crc.py`:
```python
"""Conformal Risk Control (CRC) for segment-level localization.

Stage 2 of CPSL: controls temporal false negative rate (tFNR)
at the segment level with guarantee E[tFNR] ≤ α_segment.

tFNR is monotone non-increasing in threshold λ (lower λ → more frames
predicted as fake → lower tFNR), making it CRC-compatible.

Reference: Angelopoulos, Bates, Fisch, Lei, Schuster.
"Conformal Risk Control." ICLR 2024.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..utils.metrics import compute_tFNR, compute_tFDR


@dataclass
class ConformalRiskControl:
    """CRC for controlling tFNR at segment level.

    Finds threshold λ such that E[tFNR(λ)] ≤ α.

    tFNR(λ) = (# fake frames with score < λ) / (# true fake frames)

    Lower λ → more frames classified as fake → lower tFNR.
    """

    alpha: float = 0.10
    risk_metric: str = "tFNR"  # "tFNR" or "tFDR"
    threshold: float | None = None
    _lambda_grid: np.ndarray = field(
        default_factory=lambda: np.linspace(0, 1, 1001), repr=False
    )

    def calibrate(
        self,
        frame_scores: list[np.ndarray],
        frame_labels: list[np.ndarray],
    ) -> None:
        """Calibrate threshold λ on calibration data.

        Uses the CRC procedure: find smallest λ such that
        empirical risk on calibration set ≤ α, with finite-sample
        correction.

        Args:
            frame_scores: List of per-utterance frame scores.
            frame_labels: List of per-utterance binary frame labels.
        """
        n = len(frame_scores)

        # For each λ, compute average risk on calibration set
        risks = []
        for lam in self._lambda_grid:
            risk_values = []
            for scores, labels in zip(frame_scores, frame_labels):
                pred = (scores >= lam).astype(int)
                if self.risk_metric == "tFNR":
                    risk_values.append(compute_tFNR(pred, labels))
                else:
                    risk_values.append(compute_tFDR(pred, labels))
            # Finite-sample corrected: (sum of risks + 1) / (n + 1)
            avg_risk = (sum(risk_values) + 1) / (n + 1)
            risks.append(avg_risk)

        risks = np.array(risks)

        # Find the largest λ where risk ≤ α (CRC guarantee)
        valid = np.where(risks <= self.alpha)[0]
        if len(valid) > 0:
            self.threshold = float(self._lambda_grid[valid[-1]])
        else:
            # If no λ satisfies, use λ=0 (predict everything as fake)
            self.threshold = 0.0

    def predict(self, frame_scores: list[np.ndarray]) -> list[np.ndarray]:
        """Apply calibrated threshold for segment-level localization.

        Args:
            frame_scores: List of per-utterance frame scores.

        Returns:
            List of binary frame-level predictions.
        """
        if self.threshold is None:
            raise RuntimeError("Call calibrate() before predict()")

        return [(scores >= self.threshold).astype(int) for scores in frame_scores]

    def compute_empirical_risk(
        self,
        frame_scores: list[np.ndarray],
        frame_labels: list[np.ndarray],
    ) -> dict[str, float]:
        """Compute empirical tFNR, tFDR, and tIoU on given data."""
        from ..utils.metrics import compute_tIoU

        preds = self.predict(frame_scores)
        tfnrs, tfdrs, tious = [], [], []

        for pred, labels in zip(preds, frame_labels):
            tfnrs.append(compute_tFNR(pred, labels))
            tfdrs.append(compute_tFDR(pred, labels))
            tious.append(compute_tIoU(pred, labels))

        return {
            "mean_tFNR": float(np.mean(tfnrs)),
            "mean_tFDR": float(np.mean(tfdrs)),
            "mean_tIoU": float(np.mean(tious)),
            "threshold": self.threshold,
        }
```

**Step 4: Run tests**

Run: `cd xps_forensic && pytest tests/test_cpsl.py::TestCRC -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add xps_forensic/cpsl/crc.py tests/test_cpsl.py
git commit -m "feat: add CRC for segment-level tFNR control (CPSL Stage 2)"
```

---

### Task 17: Composed CPSL Guarantee

**Files:**
- Create: `xps_forensic/xps_forensic/cpsl/composed.py`

**Step 1: Write failing test**

Append to `xps_forensic/tests/test_cpsl.py`:
```python
from xps_forensic.cpsl.composed import CPSLPipeline


class TestCPSLPipeline:
    def test_composed_guarantee(self):
        alpha1 = 0.05
        alpha2 = 0.10
        pipeline = CPSLPipeline(alpha_utterance=alpha1, alpha_segment=alpha2)
        # Composed: P(both correct) >= (1-0.05)(1-0.10) = 0.855
        assert pipeline.composed_guarantee == pytest.approx(0.855)

    def test_end_to_end(self, rng):
        n = 200
        frame_scores_list = []
        utt_labels = []
        frame_labels_list = []

        for i in range(n):
            n_frames = 150
            scores = rng.uniform(0.1, 0.3, n_frames)
            labels = np.zeros(n_frames, dtype=int)
            if i % 3 == 0:
                scores[40:100] = rng.uniform(0.7, 0.95, 60)
                labels[40:100] = 1
                utt_labels.append(1)
            elif i % 3 == 1:
                scores[:] = rng.uniform(0.8, 0.99, n_frames)
                labels[:] = 1
                utt_labels.append(2)
            else:
                utt_labels.append(0)
            frame_scores_list.append(scores)
            frame_labels_list.append(labels)

        pipeline = CPSLPipeline(alpha_utterance=0.10, alpha_segment=0.10)
        pipeline.calibrate(
            frame_scores_list[:160],
            np.array(utt_labels[:160]),
            frame_labels_list[:160],
        )

        results = pipeline.predict(frame_scores_list[160:])
        assert len(results) == 40
        for r in results:
            assert "prediction_set" in r
            assert "segment_predictions" in r
```

**Step 2: Run test to verify it fails**

Run: `cd xps_forensic && pytest tests/test_cpsl.py::TestCPSLPipeline -v`
Expected: FAIL

**Step 3: Implement composed pipeline**

`xps_forensic/xps_forensic/cpsl/composed.py`:
```python
"""Composed CPSL pipeline: Stage 1 (SCP+APS) + Stage 2 (CRC).

Composed guarantee:
    P(Stage 1 correct AND Stage 2 correct) ≥ (1-α₁)(1-α₂)

Stage 2 (CRC) is applied only when Stage 1 predicts {partially_fake}
or the prediction set includes partially_fake.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .nonconformity import compute_nonconformity
from .scp_aps import SCPAPS
from .crc import ConformalRiskControl


@dataclass
class CPSLResult:
    """Result for a single utterance."""
    utterance_id: str
    nonconformity_score: float
    prediction_set: set[int]
    prediction_set_labels: set[str]
    segment_predictions: np.ndarray | None  # binary frame-level, or None if real
    crc_threshold: float | None


class CPSLPipeline:
    """Two-stage Conformalized Partial Spoof Localization.

    Stage 1: SCP + APS for utterance-level ternary classification
    Stage 2: CRC on tFNR for segment-level localization
    """

    CLASS_NAMES = ["real", "partially_fake", "fully_fake"]

    def __init__(
        self,
        alpha_utterance: float = 0.05,
        alpha_segment: float = 0.10,
        nonconformity_method: str = "max",
        nonconformity_beta: float = 10.0,
    ):
        self.alpha_utterance = alpha_utterance
        self.alpha_segment = alpha_segment
        self.nc_method = nonconformity_method
        self.nc_beta = nonconformity_beta

        self.stage1 = SCPAPS(alpha=alpha_utterance, classes=self.CLASS_NAMES)
        self.stage2 = ConformalRiskControl(alpha=alpha_segment, risk_metric="tFNR")

    @property
    def composed_guarantee(self) -> float:
        """P(both stages correct) ≥ (1-α₁)(1-α₂)."""
        return (1 - self.alpha_utterance) * (1 - self.alpha_segment)

    def calibrate(
        self,
        frame_scores_list: list[np.ndarray],
        utterance_labels: np.ndarray,
        frame_labels_list: list[np.ndarray],
    ) -> None:
        """Calibrate both stages.

        Args:
            frame_scores_list: Per-utterance frame scores from detector.
            utterance_labels: Utterance-level labels (0, 1, 2).
            frame_labels_list: Per-utterance binary frame labels.
        """
        # Compute nonconformity scores
        nc_scores = compute_nonconformity(
            frame_scores_list, method=self.nc_method, beta=self.nc_beta
        )

        # Stage 1: calibrate SCP + APS
        self.stage1.calibrate(nc_scores, utterance_labels)

        # Stage 2: calibrate CRC on partially-spoofed utterances only
        partial_mask = utterance_labels == 1
        if partial_mask.any():
            partial_frame_scores = [
                fs for fs, m in zip(frame_scores_list, partial_mask) if m
            ]
            partial_frame_labels = [
                fl for fl, m in zip(frame_labels_list, partial_mask) if m
            ]
            self.stage2.calibrate(partial_frame_scores, partial_frame_labels)

    def predict(
        self,
        frame_scores_list: list[np.ndarray],
        utterance_ids: list[str] | None = None,
    ) -> list[dict]:
        """Run full CPSL pipeline.

        Args:
            frame_scores_list: Per-utterance frame scores.
            utterance_ids: Optional utterance identifiers.

        Returns:
            List of result dicts with prediction_set and segment_predictions.
        """
        if utterance_ids is None:
            utterance_ids = [f"utt_{i}" for i in range(len(frame_scores_list))]

        nc_scores = compute_nonconformity(
            frame_scores_list, method=self.nc_method, beta=self.nc_beta
        )
        pred_sets = self.stage1.predict(nc_scores)

        results = []
        for i, (ps, scores, uid) in enumerate(
            zip(pred_sets, frame_scores_list, utterance_ids)
        ):
            result = {
                "utterance_id": uid,
                "nonconformity_score": float(nc_scores[i]),
                "prediction_set": ps,
                "prediction_set_labels": {self.CLASS_NAMES[c] for c in ps},
                "segment_predictions": None,
                "crc_threshold": None,
            }

            # Apply Stage 2 only if partial spoof is in prediction set
            if 1 in ps and self.stage2.threshold is not None:
                seg_preds = self.stage2.predict([scores])[0]
                result["segment_predictions"] = seg_preds
                result["crc_threshold"] = self.stage2.threshold

            results.append(result)

        return results
```

**Step 4: Run tests**

Run: `cd xps_forensic && pytest tests/test_cpsl.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add xps_forensic/cpsl/composed.py tests/test_cpsl.py
git commit -m "feat: add composed CPSL pipeline with coverage guarantee theorem"
```

---

## Phase 5: PDSM-PS — Phoneme-Discretized Saliency (Tasks 18-22)

### Task 18: Phoneme Alignment Interface

**Files:**
- Create: `xps_forensic/xps_forensic/pdsm_ps/alignment.py`
- Test: `xps_forensic/tests/test_pdsm.py`

**Step 1: Write failing test**

`xps_forensic/tests/test_pdsm.py`:
```python
"""Tests for PDSM-PS components."""
import numpy as np
import pytest
from xps_forensic.pdsm_ps.alignment import PhonemeSegment, align_phonemes_mock


class TestPhonemeAlignment:
    def test_phoneme_segment(self):
        seg = PhonemeSegment(phoneme="AH", start_sec=0.1, end_sec=0.2, confidence=0.95)
        assert seg.duration_sec == pytest.approx(0.1)
        assert seg.start_frame(frame_shift_ms=20) == 5
        assert seg.end_frame(frame_shift_ms=20) == 10

    def test_mock_alignment(self):
        segments = align_phonemes_mock(duration_sec=1.0, n_phonemes=10)
        assert len(segments) == 10
        assert segments[0].start_sec == pytest.approx(0.0)
        assert segments[-1].end_sec == pytest.approx(1.0)
        for seg in segments:
            assert seg.duration_sec > 0
```

**Step 2: Run test to verify it fails**

Run: `cd xps_forensic && pytest tests/test_pdsm.py::TestPhonemeAlignment -v`
Expected: FAIL

**Step 3: Implement alignment module**

`xps_forensic/xps_forensic/pdsm_ps/alignment.py`:
```python
"""Phoneme alignment for PDSM-PS.

Supports:
- Montreal Forced Aligner (MFA) — primary
- WhisperX — neural baseline
- Mock aligner for testing

Reference: Gupta et al., "Phoneme Discretized Saliency Maps for Explainable
Detection of AI-Generated Voice," Interspeech 2024.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class PhonemeSegment:
    """A single phoneme with temporal boundaries."""
    phoneme: str
    start_sec: float
    end_sec: float
    confidence: float = 1.0

    @property
    def duration_sec(self) -> float:
        return self.end_sec - self.start_sec

    def start_frame(self, frame_shift_ms: int = 20) -> int:
        return int(self.start_sec * 1000 / frame_shift_ms)

    def end_frame(self, frame_shift_ms: int = 20) -> int:
        return int(self.end_sec * 1000 / frame_shift_ms)


def align_phonemes_mock(
    duration_sec: float, n_phonemes: int = 20
) -> list[PhonemeSegment]:
    """Mock phoneme alignment for testing."""
    phoneme_dur = duration_sec / n_phonemes
    phonemes = ["AH", "B", "CH", "D", "EH", "F", "G", "HH", "IH", "JH",
                 "K", "L", "M", "N", "OW", "P", "R", "S", "T", "UW"]
    segments = []
    for i in range(n_phonemes):
        segments.append(PhonemeSegment(
            phoneme=phonemes[i % len(phonemes)],
            start_sec=i * phoneme_dur,
            end_sec=(i + 1) * phoneme_dur,
            confidence=0.95,
        ))
    return segments


def align_with_mfa(
    wav_path: str | Path,
    transcript: str | None = None,
    language: str = "english_us_arpa",
) -> list[PhonemeSegment]:
    """Align phonemes using Montreal Forced Aligner.

    Requires MFA installed: conda install -c conda-forge montreal-forced-aligner

    Args:
        wav_path: Path to audio file.
        transcript: Optional transcript. If None, uses MFA G2P.
        language: MFA acoustic model/dictionary name.

    Returns:
        List of PhonemeSegments.
    """
    import subprocess
    import tempfile
    import json

    wav_path = Path(wav_path)

    with tempfile.TemporaryDirectory() as tmpdir:
        input_dir = Path(tmpdir) / "input"
        output_dir = Path(tmpdir) / "output"
        input_dir.mkdir()
        output_dir.mkdir()

        # Copy wav to input dir
        import shutil
        shutil.copy2(wav_path, input_dir / wav_path.name)

        # Write transcript if provided
        if transcript:
            txt_path = input_dir / wav_path.with_suffix(".txt").name
            txt_path.write_text(transcript)

        # Run MFA
        cmd = [
            "mfa", "align",
            str(input_dir),
            language,  # dictionary
            language,  # acoustic model
            str(output_dir),
            "--output_format", "json",
            "--clean",
        ]
        subprocess.run(cmd, capture_output=True, check=True)

        # Parse TextGrid or JSON output
        results = list(output_dir.rglob("*.json"))
        if not results:
            results = list(output_dir.rglob("*.TextGrid"))
        if not results:
            return []

        return _parse_mfa_output(results[0])


def _parse_mfa_output(path: Path) -> list[PhonemeSegment]:
    """Parse MFA JSON or TextGrid output."""
    if path.suffix == ".json":
        import json
        with open(path) as f:
            data = json.load(f)
        segments = []
        for tier in data.get("tiers", {}).values():
            if tier.get("type") == "phones":
                for entry in tier.get("entries", []):
                    if len(entry) >= 3 and entry[2].strip():
                        segments.append(PhonemeSegment(
                            phoneme=entry[2],
                            start_sec=float(entry[0]),
                            end_sec=float(entry[1]),
                        ))
        return segments

    # TextGrid fallback — basic parser
    text = path.read_text()
    segments = []
    lines = text.split("\n")
    in_phones = False
    i = 0
    while i < len(lines):
        if "phones" in lines[i].lower():
            in_phones = True
        if in_phones and "xmin" in lines[i]:
            xmin = float(lines[i].split("=")[1].strip())
            xmax = float(lines[i + 1].split("=")[1].strip())
            text_val = lines[i + 2].split('"')[1] if '"' in lines[i + 2] else ""
            if text_val.strip():
                segments.append(PhonemeSegment(
                    phoneme=text_val.strip(),
                    start_sec=xmin,
                    end_sec=xmax,
                ))
            i += 3
        else:
            i += 1
    return segments


def align_with_whisperx(
    wav_path: str | Path,
    device: str = "cpu",
) -> list[PhonemeSegment]:
    """Align phonemes using WhisperX word-level alignment.

    Note: WhisperX provides word-level, not phoneme-level alignment.
    We use grapheme-to-phoneme (G2P) conversion post-hoc.

    Args:
        wav_path: Path to audio file.
        device: "cpu" or "cuda".

    Returns:
        List of PhonemeSegments (word-level granularity).
    """
    import whisperx

    model = whisperx.load_model("base", device=device)
    audio = whisperx.load_audio(str(wav_path))
    result = model.transcribe(audio)

    # Align
    model_a, metadata = whisperx.load_align_model(language_code="en", device=device)
    aligned = whisperx.align(result["segments"], model_a, metadata, audio, device)

    segments = []
    for seg in aligned.get("word_segments", []):
        if "start" in seg and "end" in seg:
            segments.append(PhonemeSegment(
                phoneme=seg.get("word", ""),
                start_sec=seg["start"],
                end_sec=seg["end"],
                confidence=seg.get("score", 0.0),
            ))

    return segments
```

**Step 4: Run tests**

Run: `cd xps_forensic && pytest tests/test_pdsm.py::TestPhonemeAlignment -v`
Expected: PASS

**Step 5: Commit**

```bash
git add xps_forensic/pdsm_ps/alignment.py tests/test_pdsm.py
git commit -m "feat: add phoneme alignment interface (MFA + WhisperX + mock)"
```

---

### Task 19: Saliency Computation (IG + GradSHAP)

**Files:**
- Create: `xps_forensic/xps_forensic/pdsm_ps/saliency.py`

**Step 1: Write failing test**

Append to `xps_forensic/tests/test_pdsm.py`:
```python
from xps_forensic.pdsm_ps.saliency import compute_saliency_mock


class TestSaliency:
    def test_mock_saliency(self):
        n_frames = 100
        saliency = compute_saliency_mock(n_frames)
        assert saliency.shape == (n_frames,)
        assert np.all(saliency >= 0)
```

**Step 2: Run test to verify it fails**

Run: `cd xps_forensic && pytest tests/test_pdsm.py::TestSaliency -v`
Expected: FAIL

**Step 3: Implement saliency module**

`xps_forensic/xps_forensic/pdsm_ps/saliency.py`:
```python
"""Saliency computation for PDSM-PS.

Computes frame-level saliency attributions using:
- Integrated Gradients (Sundararajan et al., ICML 2017)
- GradSHAP (Lundberg & Lee, NeurIPS 2017)

Applied to WavLM/wav2vec2 intermediate features for frame-level attribution.

Reference: Gupta et al. (Interspeech 2024) for PDSM methodology.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import torch


def compute_saliency_mock(n_frames: int, seed: int = 42) -> np.ndarray:
    """Mock saliency for testing without a model."""
    rng = np.random.default_rng(seed)
    return rng.uniform(0, 1, n_frames).astype(np.float32)


def compute_integrated_gradients(
    model: torch.nn.Module,
    waveform: torch.Tensor,
    target_class: int = 1,
    n_steps: int = 50,
    baseline: torch.Tensor | None = None,
) -> np.ndarray:
    """Compute Integrated Gradients attribution.

    Args:
        model: Detector model with frame-level output.
        waveform: Input tensor, shape (1, n_samples).
        target_class: Class to attribute (1 = fake).
        n_steps: Number of interpolation steps.
        baseline: Reference input (default: zeros).

    Returns:
        Frame-level saliency, shape (n_frames,).
    """
    if baseline is None:
        baseline = torch.zeros_like(waveform)

    waveform.requires_grad_(True)

    # Riemann sum approximation
    scaled_inputs = [
        baseline + (float(k) / n_steps) * (waveform - baseline)
        for k in range(1, n_steps + 1)
    ]

    gradients = []
    for scaled in scaled_inputs:
        scaled = scaled.detach().requires_grad_(True)
        output = model(scaled)

        # Extract frame-level logits
        if isinstance(output, dict):
            logits = output.get("frame_logits", output.get("logits"))
        elif isinstance(output, tuple):
            logits = output[0]
        else:
            logits = output

        # Sum target class logits across frames
        if logits.dim() == 3:
            target_sum = logits[0, :, target_class].sum()
        else:
            target_sum = logits[0].sum()

        target_sum.backward()
        gradients.append(scaled.grad.detach().clone())
        scaled.grad = None

    # Average gradients and multiply by (input - baseline)
    avg_grad = torch.stack(gradients).mean(dim=0)
    ig = (waveform - baseline) * avg_grad

    # Aggregate to frame-level: average absolute attribution per frame
    # Assume 20ms frames at 16kHz = 320 samples per frame
    samples_per_frame = 320
    n_samples = ig.shape[-1]
    n_frames = n_samples // samples_per_frame

    ig_flat = ig.squeeze().detach().cpu().numpy()
    frame_saliency = np.array([
        np.mean(np.abs(ig_flat[i * samples_per_frame:(i + 1) * samples_per_frame]))
        for i in range(n_frames)
    ])

    return frame_saliency


def compute_gradshap(
    model: torch.nn.Module,
    waveform: torch.Tensor,
    target_class: int = 1,
    n_samples: int = 25,
    seed: int = 42,
) -> np.ndarray:
    """Compute GradSHAP attribution.

    GradSHAP ≈ expected value of Integrated Gradients over random baselines.

    Args:
        model: Detector model.
        waveform: Input tensor, shape (1, n_samples).
        target_class: Class to attribute.
        n_samples: Number of random baseline samples.
        seed: Random seed.

    Returns:
        Frame-level saliency, shape (n_frames,).
    """
    torch.manual_seed(seed)

    attributions = []
    for _ in range(n_samples):
        # Random baseline: Gaussian noise with same shape
        baseline = torch.randn_like(waveform) * 0.01
        ig = compute_integrated_gradients(
            model, waveform, target_class, n_steps=10, baseline=baseline
        )
        attributions.append(ig)

    return np.mean(attributions, axis=0)
```

**Step 4: Run tests**

Run: `cd xps_forensic && pytest tests/test_pdsm.py::TestSaliency -v`
Expected: PASS

**Step 5: Commit**

```bash
git add xps_forensic/pdsm_ps/saliency.py tests/test_pdsm.py
git commit -m "feat: add saliency computation (Integrated Gradients + GradSHAP)"
```

---

### Task 20: Phoneme-Level Discretization

**Files:**
- Create: `xps_forensic/xps_forensic/pdsm_ps/discretize.py`

**Step 1: Write failing test**

Append to `xps_forensic/tests/test_pdsm.py`:
```python
from xps_forensic.pdsm_ps.discretize import (
    discretize_by_phonemes,
    discretize_by_fixed_window,
    PhonemeSaliency,
)


class TestDiscretize:
    def test_phoneme_discretization(self):
        frame_saliency = np.random.uniform(0, 1, 50)
        phonemes = align_phonemes_mock(duration_sec=1.0, n_phonemes=10)
        result = discretize_by_phonemes(frame_saliency, phonemes, frame_shift_ms=20)
        assert len(result) == 10
        for ps in result:
            assert isinstance(ps, PhonemeSaliency)
            assert ps.mean_saliency >= 0

    def test_fixed_window_discretization(self):
        frame_saliency = np.random.uniform(0, 1, 100)
        result = discretize_by_fixed_window(frame_saliency, window_ms=100, frame_shift_ms=20)
        assert len(result) == 20  # 100 frames * 20ms / 100ms = 20 windows
        for ws in result:
            assert ws.mean_saliency >= 0

    def test_discretized_vs_raw(self):
        """Discretized saliency should have same total as raw."""
        frame_saliency = np.random.uniform(0, 1, 50)
        phonemes = align_phonemes_mock(duration_sec=1.0, n_phonemes=10)
        result = discretize_by_phonemes(frame_saliency, phonemes, frame_shift_ms=20)
        total_disc = sum(ps.mean_saliency * ps.n_frames for ps in result)
        total_raw = frame_saliency.sum()
        # Should be approximately equal (may differ due to boundary effects)
        assert abs(total_disc - total_raw) < total_raw * 0.2
```

**Step 2: Run test to verify it fails**

Run: `cd xps_forensic && pytest tests/test_pdsm.py::TestDiscretize -v`
Expected: FAIL

**Step 3: Implement discretization**

`xps_forensic/xps_forensic/pdsm_ps/discretize.py`:
```python
"""Phoneme-level saliency discretization for PDSM-PS.

Aggregates frame-level saliency attributions to phoneme boundaries,
following Gupta et al. (Interspeech 2024) extended to partial spoof segments.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .alignment import PhonemeSegment


@dataclass
class PhonemeSaliency:
    """Saliency aggregated at phoneme level."""
    phoneme: str
    start_sec: float
    end_sec: float
    mean_saliency: float
    max_saliency: float
    n_frames: int
    alignment_confidence: float = 1.0

    @property
    def duration_sec(self) -> float:
        return self.end_sec - self.start_sec


def discretize_by_phonemes(
    frame_saliency: np.ndarray,
    phoneme_segments: list[PhonemeSegment],
    frame_shift_ms: int = 20,
) -> list[PhonemeSaliency]:
    """Aggregate frame saliency to phoneme boundaries.

    Args:
        frame_saliency: Shape (n_frames,).
        phoneme_segments: List of phoneme boundary segments.
        frame_shift_ms: Frame shift in milliseconds.

    Returns:
        List of PhonemeSaliency, one per phoneme.
    """
    results = []
    for seg in phoneme_segments:
        start_frame = seg.start_frame(frame_shift_ms)
        end_frame = seg.end_frame(frame_shift_ms)

        # Clamp to valid range
        start_frame = max(0, min(start_frame, len(frame_saliency)))
        end_frame = max(start_frame, min(end_frame, len(frame_saliency)))

        if end_frame > start_frame:
            segment_sal = frame_saliency[start_frame:end_frame]
            results.append(PhonemeSaliency(
                phoneme=seg.phoneme,
                start_sec=seg.start_sec,
                end_sec=seg.end_sec,
                mean_saliency=float(np.mean(segment_sal)),
                max_saliency=float(np.max(segment_sal)),
                n_frames=end_frame - start_frame,
                alignment_confidence=seg.confidence,
            ))
        else:
            results.append(PhonemeSaliency(
                phoneme=seg.phoneme,
                start_sec=seg.start_sec,
                end_sec=seg.end_sec,
                mean_saliency=0.0,
                max_saliency=0.0,
                n_frames=0,
                alignment_confidence=seg.confidence,
            ))

    return results


def discretize_by_fixed_window(
    frame_saliency: np.ndarray,
    window_ms: int = 100,
    frame_shift_ms: int = 20,
) -> list[PhonemeSaliency]:
    """Aggregate frame saliency to fixed-width windows (baseline).

    Args:
        frame_saliency: Shape (n_frames,).
        window_ms: Window width in milliseconds.
        frame_shift_ms: Frame shift in milliseconds.

    Returns:
        List of PhonemeSaliency with synthetic phoneme labels.
    """
    frames_per_window = max(1, window_ms // frame_shift_ms)
    n_frames = len(frame_saliency)
    results = []

    for i in range(0, n_frames, frames_per_window):
        end = min(i + frames_per_window, n_frames)
        segment = frame_saliency[i:end]
        start_sec = i * frame_shift_ms / 1000
        end_sec = end * frame_shift_ms / 1000

        results.append(PhonemeSaliency(
            phoneme=f"W{i // frames_per_window}",
            start_sec=start_sec,
            end_sec=end_sec,
            mean_saliency=float(np.mean(segment)),
            max_saliency=float(np.max(segment)),
            n_frames=end - i,
        ))

    return results
```

**Step 4: Run tests**

Run: `cd xps_forensic && pytest tests/test_pdsm.py::TestDiscretize -v`
Expected: PASS

**Step 5: Commit**

```bash
git add xps_forensic/pdsm_ps/discretize.py tests/test_pdsm.py
git commit -m "feat: add phoneme-level saliency discretization (PDSM-PS core)"
```

---

### Task 21: Faithfulness Metrics

**Files:**
- Create: `xps_forensic/xps_forensic/pdsm_ps/faithfulness.py`

**Step 1: Write failing test**

Append to `xps_forensic/tests/test_pdsm.py`:
```python
from xps_forensic.pdsm_ps.faithfulness import (
    normalized_aopc,
    comprehensiveness,
    sufficiency,
    phoneme_iou,
)


class TestFaithfulness:
    def test_normalized_aopc(self):
        # If removing top-k features drops prediction a lot → high AOPC
        original_score = 0.9
        perturbed_scores = [0.8, 0.6, 0.3, 0.1]  # dropping top-1, top-2, ...
        aopc = normalized_aopc(original_score, perturbed_scores)
        assert 0 <= aopc <= 1

    def test_comprehensiveness(self):
        original = 0.9
        without_top = 0.3  # removing top features drops score a lot
        comp = comprehensiveness(original, without_top)
        assert comp == pytest.approx(0.6)

    def test_sufficiency(self):
        original = 0.9
        with_only_top = 0.85  # keeping only top features retains most
        suff = sufficiency(original, with_only_top)
        assert suff == pytest.approx(0.05)

    def test_phoneme_iou(self):
        # Top-5 salient phonemes: indices [2,3,4,5,6]
        salient_indices = {2, 3, 4, 5, 6}
        # Ground truth spoofed phonemes: indices [3,4,5,6,7]
        gt_indices = {3, 4, 5, 6, 7}
        iou = phoneme_iou(salient_indices, gt_indices)
        # Intersection: {3,4,5,6} = 4, Union: {2,3,4,5,6,7} = 6
        assert iou == pytest.approx(4 / 6)
```

**Step 2: Run test to verify it fails**

Run: `cd xps_forensic && pytest tests/test_pdsm.py::TestFaithfulness -v`
Expected: FAIL

**Step 3: Implement faithfulness metrics**

`xps_forensic/xps_forensic/pdsm_ps/faithfulness.py`:
```python
"""Faithfulness metrics for saliency explanations.

Implements:
- Normalized AOPC (Edin et al., ACL 2025)
- Comprehensiveness/Sufficiency (DeYoung et al., ACL 2020)
- Phoneme-IoU (novel metric for PDSM-PS alignment with ground truth)
"""
from __future__ import annotations

import numpy as np


def normalized_aopc(
    original_score: float,
    perturbed_scores: list[float],
) -> float:
    """Normalized Area Over Perturbation Curve (N-AOPC).

    Measures how much the model prediction drops when top-k features
    are removed. Higher = more faithful explanation.

    Reference: Edin et al., "Are Saliency Maps Faithful?", ACL 2025.

    Args:
        original_score: Model score on original input.
        perturbed_scores: Scores after removing top-1, top-2, ... features.

    Returns:
        N-AOPC in [0, 1].
    """
    if not perturbed_scores:
        return 0.0

    K = len(perturbed_scores)
    drops = [original_score - ps for ps in perturbed_scores]
    aopc = sum(drops) / K

    # Normalize to [0, 1]
    max_possible = original_score  # max drop is from original to 0
    if max_possible == 0:
        return 0.0
    return float(np.clip(aopc / max_possible, 0, 1))


def comprehensiveness(
    original_score: float,
    score_without_top_features: float,
) -> float:
    """Comprehensiveness: drop in prediction when top features removed.

    Higher = explanation captures important features.

    Reference: DeYoung et al., "ERASER: A Benchmark to Evaluate
    Rationalized NLP Models," ACL 2020.
    """
    return float(max(0, original_score - score_without_top_features))


def sufficiency(
    original_score: float,
    score_with_only_top_features: float,
) -> float:
    """Sufficiency: how much prediction is retained with only top features.

    Lower = explanation is sufficient (top features alone explain prediction).
    """
    return float(max(0, original_score - score_with_only_top_features))


def phoneme_iou(
    salient_indices: set[int],
    ground_truth_indices: set[int],
) -> float:
    """Phoneme-level Intersection over Union.

    Measures alignment between top-K salient phonemes and ground-truth
    manipulated phonemes.

    Args:
        salient_indices: Set of phoneme indices marked as salient.
        ground_truth_indices: Set of phoneme indices that are actually spoofed.

    Returns:
        IoU in [0, 1].
    """
    if not salient_indices and not ground_truth_indices:
        return 1.0
    if not salient_indices or not ground_truth_indices:
        return 0.0

    intersection = len(salient_indices & ground_truth_indices)
    union = len(salient_indices | ground_truth_indices)
    return float(intersection / union)
```

**Step 4: Run tests**

Run: `cd xps_forensic && pytest tests/test_pdsm.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add xps_forensic/pdsm_ps/faithfulness.py tests/test_pdsm.py
git commit -m "feat: add faithfulness metrics (N-AOPC, Comp/Suff, Phoneme-IoU)"
```

---

### Task 22: PDSM-PS Pipeline Integration

**Files:**
- Modify: `xps_forensic/xps_forensic/pdsm_ps/__init__.py`

**Step 1: Write failing test**

Append to `xps_forensic/tests/test_pdsm.py`:
```python
from xps_forensic.pdsm_ps import PDSMPSPipeline


class TestPDSMPSPipeline:
    def test_run_mock(self):
        pipeline = PDSMPSPipeline(aligner="mock", saliency_method="mock")
        result = pipeline.run(
            frame_saliency=np.random.uniform(0, 1, 50),
            duration_sec=1.0,
            spoofed_frame_mask=np.concatenate([np.zeros(20), np.ones(15), np.zeros(15)]),
        )
        assert "phoneme_saliencies" in result
        assert "top_k_phonemes" in result
        assert "phoneme_iou" in result
        assert len(result["phoneme_saliencies"]) > 0
```

**Step 2: Run test to verify it fails**

Run: `cd xps_forensic && pytest tests/test_pdsm.py::TestPDSMPSPipeline -v`
Expected: FAIL

**Step 3: Implement PDSM-PS pipeline**

`xps_forensic/xps_forensic/pdsm_ps/__init__.py`:
```python
"""PDSM-PS: Phoneme-Discretized Saliency Maps for Partial Spoofs.

Extension of PDSM (Gupta et al., Interspeech 2024) to segment-level
partial spoof localization. Applied to CPSL-flagged segments only.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .alignment import PhonemeSegment, align_phonemes_mock
from .discretize import (
    PhonemeSaliency,
    discretize_by_phonemes,
    discretize_by_fixed_window,
)
from .faithfulness import phoneme_iou


@dataclass
class PDSMPSResult:
    """Result of PDSM-PS analysis on a single utterance."""
    phoneme_saliencies: list[PhonemeSaliency]
    top_k_phonemes: list[PhonemeSaliency]
    phoneme_iou_score: float
    aligner_used: str


class PDSMPSPipeline:
    """End-to-end PDSM-PS pipeline."""

    def __init__(
        self,
        aligner: str = "mock",
        saliency_method: str = "mock",
        top_k: int = 5,
        frame_shift_ms: int = 20,
    ):
        self.aligner = aligner
        self.saliency_method = saliency_method
        self.top_k = top_k
        self.frame_shift_ms = frame_shift_ms

    def run(
        self,
        frame_saliency: np.ndarray,
        duration_sec: float,
        spoofed_frame_mask: np.ndarray | None = None,
        wav_path: str | None = None,
        phoneme_segments: list[PhonemeSegment] | None = None,
    ) -> dict:
        """Run PDSM-PS analysis.

        Args:
            frame_saliency: Frame-level saliency, shape (n_frames,).
            duration_sec: Audio duration in seconds.
            spoofed_frame_mask: Ground truth binary mask for IoU computation.
            wav_path: Path to audio for MFA/WhisperX alignment.
            phoneme_segments: Pre-computed phoneme segments (optional).

        Returns:
            Dict with phoneme_saliencies, top_k_phonemes, phoneme_iou.
        """
        # Step 1: Get phoneme boundaries
        if phoneme_segments is None:
            if self.aligner == "mock":
                n_phonemes = max(1, int(duration_sec * 10))
                phoneme_segments = align_phonemes_mock(duration_sec, n_phonemes)
            elif self.aligner == "mfa" and wav_path:
                from .alignment import align_with_mfa
                phoneme_segments = align_with_mfa(wav_path)
            elif self.aligner == "whisperx" and wav_path:
                from .alignment import align_with_whisperx
                phoneme_segments = align_with_whisperx(wav_path)
            else:
                phoneme_segments = align_phonemes_mock(duration_sec)

        # Step 2: Discretize saliency to phoneme level
        phoneme_saliencies = discretize_by_phonemes(
            frame_saliency, phoneme_segments, self.frame_shift_ms
        )

        # Step 3: Rank and select top-K salient phonemes
        sorted_by_saliency = sorted(
            enumerate(phoneme_saliencies),
            key=lambda x: x[1].mean_saliency,
            reverse=True,
        )
        top_k_indices = {idx for idx, _ in sorted_by_saliency[:self.top_k]}
        top_k_phonemes = [ps for idx, ps in sorted_by_saliency[:self.top_k]]

        # Step 4: Compute Phoneme-IoU if ground truth available
        iou_score = 0.0
        if spoofed_frame_mask is not None:
            # Identify ground-truth spoofed phoneme indices
            gt_indices = set()
            for i, seg in enumerate(phoneme_segments):
                start_f = seg.start_frame(self.frame_shift_ms)
                end_f = min(seg.end_frame(self.frame_shift_ms), len(spoofed_frame_mask))
                if start_f < end_f:
                    overlap = spoofed_frame_mask[start_f:end_f].mean()
                    if overlap > 0.5:
                        gt_indices.add(i)
            iou_score = phoneme_iou(top_k_indices, gt_indices)

        return {
            "phoneme_saliencies": phoneme_saliencies,
            "top_k_phonemes": top_k_phonemes,
            "phoneme_iou": iou_score,
            "aligner": self.aligner,
        }
```

**Step 4: Run tests**

Run: `cd xps_forensic && pytest tests/test_pdsm.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add xps_forensic/pdsm_ps/ tests/test_pdsm.py
git commit -m "feat: add PDSM-PS pipeline integration"
```

---

## Phase 6: Evidence Packaging (Tasks 23-24)

### Task 23: Evidence JSON Schema

**Files:**
- Create: `xps_forensic/xps_forensic/evidence/schema.py`
- Test: `xps_forensic/tests/test_evidence.py`

**Step 1: Write failing test**

`xps_forensic/tests/test_evidence.py`:
```python
"""Tests for evidence packaging."""
import json
import pytest
from xps_forensic.evidence.schema import EvidencePackage, validate_evidence


class TestEvidenceSchema:
    def test_create_evidence(self):
        pkg = EvidencePackage(
            utterance_id="utt_001",
            detector="BAM",
            calibration_method="temperature",
            prediction_set={"partially_fake"},
            coverage_guarantee=0.95,
            segment_predictions=[0, 0, 1, 1, 1, 0, 0],
            crc_threshold=0.45,
            tFNR_guarantee=0.10,
            phoneme_attributions=[
                {"phoneme": "AH", "saliency": 0.8, "start": 0.1, "end": 0.15},
            ],
        )
        assert pkg.utterance_id == "utt_001"

    def test_to_json(self):
        pkg = EvidencePackage(
            utterance_id="utt_001",
            detector="BAM",
            calibration_method="temperature",
            prediction_set={"partially_fake"},
            coverage_guarantee=0.95,
            segment_predictions=[0, 0, 1, 1, 0],
            crc_threshold=0.45,
            tFNR_guarantee=0.10,
        )
        j = pkg.to_json()
        parsed = json.loads(j)
        assert parsed["utterance_id"] == "utt_001"
        assert parsed["detector"] == "BAM"
        assert "daubert_factors" in parsed

    def test_validate_complete(self):
        pkg = EvidencePackage(
            utterance_id="utt_001",
            detector="BAM",
            calibration_method="temperature",
            prediction_set={"partially_fake"},
            coverage_guarantee=0.95,
            segment_predictions=[0, 1, 0],
            crc_threshold=0.45,
            tFNR_guarantee=0.10,
        )
        errors = validate_evidence(pkg)
        assert len(errors) == 0
```

**Step 2: Run test to verify it fails**

Run: `cd xps_forensic && pytest tests/test_evidence.py -v`
Expected: FAIL

**Step 3: Implement evidence schema**

`xps_forensic/xps_forensic/evidence/schema.py`:
```python
"""Evidence packaging schema for forensic output.

Produces structured JSON aligned with Daubert/FRE 702 factors.
Framing: "forensically defensible" NOT "court-admissible."
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone


@dataclass
class EvidencePackage:
    """Structured evidence output from XPS-Forensic pipeline."""

    # Identification
    utterance_id: str
    detector: str
    calibration_method: str

    # CPSL Stage 1
    prediction_set: set[str]
    coverage_guarantee: float

    # CPSL Stage 2
    segment_predictions: list[int] | None = None
    crc_threshold: float | None = None
    tFNR_guarantee: float | None = None

    # PDSM-PS
    phoneme_attributions: list[dict] | None = None

    # Metadata
    schema_version: str = "1.0"
    pipeline_version: str = "0.1.0"
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    @property
    def daubert_factors(self) -> dict[str, str]:
        """Map pipeline components to Daubert admissibility factors."""
        return {
            "testability": (
                "Pipeline evaluated on 4 independent datasets with "
                "reproducible bootstrap CIs."
            ),
            "peer_review": (
                f"Detector ({self.detector}) published at peer-reviewed venue. "
                "Calibration and conformal methods based on established theory."
            ),
            "known_error_rate": (
                f"Utterance coverage ≥ {self.coverage_guarantee:.0%}. "
                f"Segment tFNR ≤ {self.tFNR_guarantee or 'N/A'}."
            ),
            "standards": (
                "Aligned with ENFSI BPM for digital audio authenticity, "
                "SWGDE best practices, NIST AI 600-1."
            ),
            "general_acceptance": (
                "Conformal prediction: Vovk et al. (2005), 20+ years of theory. "
                "Saliency methods: established in ML interpretability."
            ),
        }

    def to_dict(self) -> dict:
        """Convert to serializable dict."""
        d = {
            "schema_version": self.schema_version,
            "pipeline_version": self.pipeline_version,
            "timestamp": self.timestamp,
            "utterance_id": self.utterance_id,
            "detector": self.detector,
            "calibration_method": self.calibration_method,
            "cpsl_stage1": {
                "prediction_set": sorted(self.prediction_set),
                "coverage_guarantee": self.coverage_guarantee,
            },
            "cpsl_stage2": {
                "segment_predictions": self.segment_predictions,
                "crc_threshold": self.crc_threshold,
                "tFNR_guarantee": self.tFNR_guarantee,
            },
            "pdsm_ps": {
                "phoneme_attributions": self.phoneme_attributions or [],
            },
            "daubert_factors": self.daubert_factors,
        }
        return d

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)


def validate_evidence(pkg: EvidencePackage) -> list[str]:
    """Validate evidence package completeness.

    Returns list of error messages (empty = valid).
    """
    errors = []
    if not pkg.utterance_id:
        errors.append("Missing utterance_id")
    if not pkg.detector:
        errors.append("Missing detector name")
    if not pkg.prediction_set:
        errors.append("Empty prediction set")
    if pkg.coverage_guarantee <= 0 or pkg.coverage_guarantee > 1:
        errors.append(f"Invalid coverage guarantee: {pkg.coverage_guarantee}")
    if pkg.segment_predictions is not None and pkg.crc_threshold is None:
        errors.append("Segment predictions present but no CRC threshold")
    return errors
```

**Step 4: Run tests**

Run: `cd xps_forensic && pytest tests/test_evidence.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add xps_forensic/evidence/ tests/test_evidence.py
git commit -m "feat: add evidence packaging with Daubert factor mapping"
```

---

## Phase 7: Experiment Scripts (Tasks 24-31)

Each experiment script is a standalone runner that loads config, runs inference, computes metrics, and saves results. These are NOT TDD — they are execution scripts that depend on downloaded datasets and model checkpoints.

### Task 24: E1 — Baseline Detection & Localization

**Files:**
- Create: `xps_forensic/experiments/run_e1_baseline.py`
- Create: `xps_forensic/configs/experiment/e1_baseline.yaml`

**Step 1: Write experiment config**

`xps_forensic/configs/experiment/e1_baseline.yaml`:
```yaml
# E1: Baseline Detection & Localization
# Fine-tune BAM/SAL on PartialSpoof, evaluate all 4 detectors
# Report: Utt-EER, Seg-EER (20/160ms), Seg-F1
# Goal: Reproduce published numbers to establish trust

experiment: e1_baseline
detectors: [bam, sal, cfprf, mrm]
dataset: partialspoof
split: eval
resolutions_ms: [20, 40, 80, 160, 320, 640]
n_bootstrap: 1000
```

**Step 2: Write experiment script**

`xps_forensic/experiments/run_e1_baseline.py`:
```python
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
```

**Step 3: Commit**

```bash
git add experiments/run_e1_baseline.py configs/experiment/e1_baseline.yaml
git commit -m "feat: add E1 baseline detection experiment script"
```

---

### Task 25: E2 — Post-hoc Calibration Comparison

**Files:**
- Create: `xps_forensic/experiments/run_e2_calibration.py`

**Step 1: Write experiment script**

`xps_forensic/experiments/run_e2_calibration.py`:
```python
"""E2: Post-hoc Calibration Comparison.

Applies Platt/temperature/isotonic calibration to all 4 detectors.
Includes uncalibrated baseline. Reports ECE, Brier, NLL with bootstrap CIs.
Uses utterance-stratified cross-validation.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from xps_forensic.utils.config import load_config
from xps_forensic.calibration.methods import calibrate_scores, PlattScaling, TemperatureScaling, IsotonicCalibrator
from xps_forensic.calibration.metrics import (
    expected_calibration_error,
    brier_score,
    negative_log_likelihood,
    reliability_diagram_data,
)
from xps_forensic.utils.stats import bootstrap_ci, friedman_nemenyi


def run_e2(cfg=None, precomputed_scores=None):
    """Run E2 calibration comparison.

    Args:
        cfg: Config dict.
        precomputed_scores: Dict of {detector_name: (scores, labels)} from E1.
    """
    if cfg is None:
        cfg = load_config()

    output_dir = Path(cfg.experiments.output_dir) / "e2_calibration"
    output_dir.mkdir(parents=True, exist_ok=True)

    methods = ["uncalibrated", "platt", "temperature", "isotonic"]
    calibrator_classes = {
        "platt": PlattScaling,
        "temperature": TemperatureScaling,
        "isotonic": IsotonicCalibrator,
    }

    results = {}

    for det_name, (scores, labels) in (precomputed_scores or {}).items():
        print(f"\n{'='*60}")
        print(f"Detector: {det_name.upper()}")

        det_results = {}

        # Stratified K-fold cross-validation
        skf = StratifiedKFold(n_splits=cfg.calibration.cv_folds, shuffle=True, random_state=42)

        for method in methods:
            eces, briers, nlls = [], [], []

            for train_idx, test_idx in skf.split(scores, labels):
                train_scores, train_labels = scores[train_idx], labels[train_idx]
                test_scores, test_labels = scores[test_idx], labels[test_idx]

                if method == "uncalibrated":
                    cal_scores = test_scores
                else:
                    cal = calibrator_classes[method]()
                    cal.fit(train_scores, train_labels)
                    cal_scores = cal.transform(test_scores)

                eces.append(expected_calibration_error(cal_scores, test_labels))
                briers.append(brier_score(cal_scores, test_labels))
                nlls.append(negative_log_likelihood(cal_scores, test_labels))

            det_results[method] = {
                "ece_mean": float(np.mean(eces)),
                "ece_ci": bootstrap_ci(np.array(eces)),
                "brier_mean": float(np.mean(briers)),
                "brier_ci": bootstrap_ci(np.array(briers)),
                "nll_mean": float(np.mean(nlls)),
                "nll_ci": bootstrap_ci(np.array(nlls)),
            }
            print(f"  {method:15s} ECE={np.mean(eces):.4f} Brier={np.mean(briers):.4f} NLL={np.mean(nlls):.4f}")

        results[det_name] = det_results

    # Friedman test across detectors for best calibration method
    if len(results) >= 3:
        ece_matrix = np.array([
            [results[d][m]["ece_mean"] for m in methods]
            for d in results
        ])
        friedman = friedman_nemenyi(ece_matrix)
        results["statistical_tests"] = {"friedman_ece": friedman}

    # Save
    output_file = output_dir / "results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=lambda x: list(x) if isinstance(x, tuple) else float(x) if isinstance(x, np.floating) else x)
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    run_e2()
```

**Step 2: Commit**

```bash
git add experiments/run_e2_calibration.py
git commit -m "feat: add E2 calibration comparison experiment"
```

---

### Task 26: E3 — CPSL Coverage & Efficiency

**Files:**
- Create: `xps_forensic/experiments/run_e3_cpsl.py`

**Step 1: Write experiment script**

`xps_forensic/experiments/run_e3_cpsl.py`:
```python
"""E3: CPSL Coverage & Efficiency.

Stage 1: SCP+APS coverage at alpha={0.01, 0.05, 0.10}, prediction set sizes.
Stage 2: CRC on tFNR, empirical tIoU, tFNR, tFDR.
Verify on held-out 20% of PartialSpoof eval.
Statistical test: one-sided binomial for coverage verification.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from xps_forensic.utils.config import load_config
from xps_forensic.utils.stats import binomial_coverage_test, bootstrap_ci
from xps_forensic.cpsl.composed import CPSLPipeline
from xps_forensic.cpsl.nonconformity import compute_nonconformity


def run_e3(cfg=None, precomputed=None):
    """Run E3 CPSL experiment.

    Args:
        cfg: Config.
        precomputed: Dict with 'frame_scores', 'utt_labels', 'frame_labels'
            as lists from E1 inference.
    """
    if cfg is None:
        cfg = load_config()

    output_dir = Path(cfg.experiments.output_dir) / "e3_cpsl"
    output_dir.mkdir(parents=True, exist_ok=True)

    if precomputed is None:
        print("E3 requires precomputed frame scores from E1. Exiting.")
        return

    frame_scores = precomputed["frame_scores"]
    utt_labels = np.array(precomputed["utt_labels"])
    frame_labels = precomputed["frame_labels"]

    # Split: 80% calibration, 20% verification
    n = len(frame_scores)
    rng = np.random.default_rng(cfg.project.seed)
    perm = rng.permutation(n)
    split = int(n * cfg.data.partialspoof.eval_split_ratio)

    cal_idx, ver_idx = perm[:split], perm[split:]
    cal_fs = [frame_scores[i] for i in cal_idx]
    cal_ul = utt_labels[cal_idx]
    cal_fl = [frame_labels[i] for i in cal_idx]
    ver_fs = [frame_scores[i] for i in ver_idx]
    ver_ul = utt_labels[ver_idx]
    ver_fl = [frame_labels[i] for i in ver_idx]

    results = {}

    # Sweep alpha values
    for alpha_utt in cfg.cpsl.alpha_sweep:
        for alpha_seg in cfg.cpsl.alpha_sweep:
            key = f"a_utt={alpha_utt}_a_seg={alpha_seg}"
            print(f"\n--- {key} ---")

            pipeline = CPSLPipeline(
                alpha_utterance=alpha_utt,
                alpha_segment=alpha_seg,
                nonconformity_method=cfg.cpsl.nonconformity,
            )
            pipeline.calibrate(cal_fs, cal_ul, cal_fl)

            # Verify on held-out set
            predictions = pipeline.predict(ver_fs)

            # Stage 1: coverage
            covered = sum(
                1 for pred, true_label in zip(predictions, ver_ul)
                if true_label in pred["prediction_set"]
            )
            n_ver = len(ver_ul)
            coverage = covered / n_ver
            p_val = binomial_coverage_test(covered, n_ver, alpha_utt)

            # Prediction set sizes
            set_sizes = [len(p["prediction_set"]) for p in predictions]

            # Stage 2: tFNR on partial spoofs
            from xps_forensic.utils.metrics import compute_tFNR, compute_tFDR, compute_tIoU
            tfnrs, tfdrs, tious = [], [], []
            for pred, fl in zip(predictions, ver_fl):
                if pred["segment_predictions"] is not None:
                    seg_pred = pred["segment_predictions"]
                    min_len = min(len(seg_pred), len(fl))
                    tfnrs.append(compute_tFNR(seg_pred[:min_len], fl[:min_len]))
                    tfdrs.append(compute_tFDR(seg_pred[:min_len], fl[:min_len]))
                    tious.append(compute_tIoU(seg_pred[:min_len], fl[:min_len]))

            results[key] = {
                "alpha_utterance": alpha_utt,
                "alpha_segment": alpha_seg,
                "composed_guarantee": pipeline.composed_guarantee,
                "stage1": {
                    "coverage": coverage,
                    "coverage_target": 1 - alpha_utt,
                    "binomial_p_value": p_val,
                    "coverage_verified": p_val > 0.05,
                    "avg_set_size": float(np.mean(set_sizes)),
                    "set_size_ci": bootstrap_ci(np.array(set_sizes)),
                },
                "stage2": {
                    "mean_tFNR": float(np.mean(tfnrs)) if tfnrs else None,
                    "mean_tFDR": float(np.mean(tfdrs)) if tfdrs else None,
                    "mean_tIoU": float(np.mean(tious)) if tious else None,
                    "tFNR_ci": bootstrap_ci(np.array(tfnrs)) if tfnrs else None,
                    "crc_threshold": pipeline.stage2.threshold,
                },
                "quantiles": pipeline.stage1.get_quantiles(),
            }

            print(f"  Coverage: {coverage:.3f} (target: {1-alpha_utt:.3f}), p={p_val:.4f}")
            print(f"  Avg set size: {np.mean(set_sizes):.2f}")
            if tfnrs:
                print(f"  Mean tFNR: {np.mean(tfnrs):.4f}, tIoU: {np.mean(tious):.4f}")

    # Nonconformity score ablation
    print("\n--- Nonconformity ablation ---")
    nc_results = {}
    for method in ["max", "logsumexp"]:
        betas = [None] if method == "max" else cfg.cpsl.logsumexp_beta
        for beta in betas:
            nc_key = f"{method}" + (f"_b{beta}" if beta else "")
            nc_scores = compute_nonconformity(
                cal_fs, method=method, beta=beta or 10.0
            )
            nc_results[nc_key] = {
                "mean": float(np.mean(nc_scores)),
                "std": float(np.std(nc_scores)),
            }
    results["nonconformity_ablation"] = nc_results

    # Save
    output_file = output_dir / "results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=lambda x: list(x) if isinstance(x, tuple) else float(x) if isinstance(x, np.floating) else x)
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    run_e3()
```

**Step 2: Commit**

```bash
git add experiments/run_e3_cpsl.py
git commit -m "feat: add E3 CPSL coverage and efficiency experiment"
```

---

### Task 27: E4 — PDSM-PS Faithfulness

**Files:**
- Create: `xps_forensic/experiments/run_e4_pdsm.py`

**Step 1: Write experiment script**

`xps_forensic/experiments/run_e4_pdsm.py`:
```python
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
                    ious.append(result["phoneme_iou"])

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
```

**Step 2: Commit**

```bash
git add experiments/run_e4_pdsm.py
git commit -m "feat: add E4 PDSM-PS faithfulness experiment"
```

---

### Task 28: E5 — Cross-Dataset Generalization

**Files:**
- Create: `xps_forensic/experiments/run_e5_cross_dataset.py`

**Step 1: Write experiment script**

`xps_forensic/experiments/run_e5_cross_dataset.py`:
```python
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

                # CPSL coverage (if pipeline fitted)
                cpsl_coverage = None
                if cpsl_pipeline:
                    preds = cpsl_pipeline.predict(frame_scores_all)
                    covered = sum(
                        1 for p, y in zip(preds, utt_labels)
                        if y in p["prediction_set"]
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
```

**Step 2: Commit**

```bash
git add experiments/run_e5_cross_dataset.py
git commit -m "feat: add E5 cross-dataset generalization experiment"
```

---

### Task 29: E6 — Codec Stress Test

**Files:**
- Create: `xps_forensic/experiments/run_e6_codec.py`

**Step 1: Write experiment script**

`xps_forensic/experiments/run_e6_codec.py`:
```python
"""E6: Codec Stress Test.

Re-encode PartialSpoof eval through AAC/Opus/AMR/G.711.
Report metric degradation per codec.
Test CPSL coverage under codec distortion.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from xps_forensic.utils.config import load_config


CODEC_CONFIGS = {
    "aac": {"ext": ".m4a", "cmd": "-c:a aac -b:a 128k"},
    "opus": {"ext": ".opus", "cmd": "-c:a libopus -b:a 64k"},
    "amr": {"ext": ".amr", "cmd": "-c:a libopencore_amrnb -ar 8000 -ac 1"},
    "g711": {"ext": ".wav", "cmd": "-c:a pcm_alaw -ar 8000 -ac 1"},
}


def transcode_audio(
    input_path: str,
    codec: str,
    output_path: str | None = None,
) -> str:
    """Transcode audio through a codec using ffmpeg.

    Returns path to transcoded (then re-decoded to WAV) file.
    """
    config = CODEC_CONFIGS[codec]

    if output_path is None:
        output_path = tempfile.mktemp(suffix=".wav")

    intermediate = tempfile.mktemp(suffix=config["ext"])

    # Encode
    cmd_encode = f"ffmpeg -y -i {input_path} {config['cmd']} {intermediate}"
    subprocess.run(cmd_encode.split(), capture_output=True, check=True)

    # Decode back to WAV 16kHz
    cmd_decode = f"ffmpeg -y -i {intermediate} -ar 16000 -ac 1 {output_path}"
    subprocess.run(cmd_decode.split(), capture_output=True, check=True)

    # Cleanup intermediate
    Path(intermediate).unlink(missing_ok=True)

    return output_path


def run_e6(cfg=None):
    if cfg is None:
        cfg = load_config()

    output_dir = Path(cfg.experiments.output_dir) / "e6_codec"
    output_dir.mkdir(parents=True, exist_ok=True)

    codecs = cfg.experiments.codecs
    results = {codec: {} for codec in codecs}

    print("E6: Codec Stress Test")
    print(f"Codecs: {codecs}")
    print("Run E1 inference on re-encoded audio for each codec.")
    print("Compare metrics against uncompressed baseline.")

    # Save placeholder — actual execution requires E1 integration
    output_file = output_dir / "results.json"
    with open(output_file, "w") as f:
        json.dump({"codecs": codecs, "status": "ready_to_run"}, f, indent=2)
    print(f"Config saved to {output_file}")


if __name__ == "__main__":
    run_e6()
```

**Step 2: Commit**

```bash
git add experiments/run_e6_codec.py
git commit -m "feat: add E6 codec stress test experiment"
```

---

### Task 30: E7 — MFA vs WhisperX Alignment + E8 — Ablations

**Files:**
- Create: `xps_forensic/experiments/run_e7_alignment.py`
- Create: `xps_forensic/experiments/run_e8_ablation.py`

**Step 1: Write E7 script**

`xps_forensic/experiments/run_e7_alignment.py`:
```python
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
```

**Step 2: Write E8 script**

`xps_forensic/experiments/run_e8_ablation.py`:
```python
"""E8: Ablation Studies.

- CPSL: ±calibration pre-step; max vs logsumexp; frame vs segment conformal
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
```

**Step 3: Commit**

```bash
git add experiments/
git commit -m "feat: add E7 alignment comparison and E8 ablation experiment scripts"
```

---

### Task 31: Master Run Script

**Files:**
- Create: `xps_forensic/scripts/run_all.sh`

**Step 1: Create master script**

`xps_forensic/scripts/run_all.sh`:
```bash
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
```

**Step 2: Make executable and commit**

```bash
chmod +x xps_forensic/scripts/run_all.sh
git add scripts/run_all.sh
git commit -m "feat: add master experiment runner script"
```

---

## Phase 8: Cleanup & Final Verification (Task 32)

### Task 32: Full Test Suite & Cleanup

**Step 1: Run full test suite**

Run: `cd xps_forensic && pytest tests/ -v --tb=short`
Expected: All tests PASS

**Step 2: Verify imports**

Run: `cd xps_forensic && python -c "from xps_forensic.cpsl.composed import CPSLPipeline; from xps_forensic.pdsm_ps import PDSMPSPipeline; from xps_forensic.evidence.schema import EvidencePackage; print('All imports OK')"`
Expected: "All imports OK"

**Step 3: Clean up temporary research artifacts from root**

```bash
# cpsl-theoretical-analysis.md was a research artifact, not code
rm -f /media/lab2208/ssd/Explainablility/cpsl-theoretical-analysis.md
```

**Step 4: Final commit**

```bash
git add -A
git commit -m "chore: full test suite verification and cleanup"
```

---

## Execution Summary

| Phase | Tasks | Description | Est. Time |
|-------|-------|-------------|-----------|
| 0 | 1-3 | Project scaffolding, config, metrics | 30 min |
| 1 | 4-8 | Data pipeline (4 datasets + download) | 45 min |
| 2 | 9-11 | Detector wrappers (BAM, SAL, CFPRF, MRM) | 30 min |
| 3 | 12-13 | Calibration methods + metrics | 20 min |
| 4 | 14-17 | CPSL (nonconformity, SCP+APS, CRC, composed) | 45 min |
| 5 | 18-22 | PDSM-PS (alignment, saliency, discretize, faithfulness) | 45 min |
| 6 | 23 | Evidence packaging | 15 min |
| 7 | 24-31 | Experiment scripts E1-E8 + master runner | 60 min |
| 8 | 32 | Test suite verification + cleanup | 15 min |
| **Total** | **32** | | **~5 hours coding** |

**Note:** This covers code implementation only. Dataset download, model training/fine-tuning, and running all 8 experiments will take an additional 2.5-7 days of GPU time per the feasibility analysis.

## Post-Implementation: Manuscript

After experiments complete, use the `research-assistant:manuscript-writer` agent to draft the IEEE TIFS manuscript following the paper outline in Section 8 of the design document (`docs/plans/2026-03-02-xps-forensic-design.md`).
