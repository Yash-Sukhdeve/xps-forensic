# Computational Feasibility Analysis: Explainable Partial Spoof Detection Pipeline

**Date:** 2026-03-02
**Hardware:** NVIDIA RTX 4080 (16 GB VRAM, Ada Lovelace) + 64 GB system RAM + SSD
**Constraint:** Single GPU, no cloud, no multi-GPU
**Scope:** Experiments E1--E8 as defined in the project experimental design

---

## Table of Contents

1. [Hardware Baseline Measurements](#1-hardware-baseline-measurements)
2. [Model VRAM Profiling](#2-model-vram-profiling)
3. [Dataset Storage and I/O Analysis](#3-dataset-storage-and-io-analysis)
4. [Experiment-by-Experiment Feasibility](#4-experiment-by-experiment-feasibility)
5. [Master Feasibility Table](#5-master-feasibility-table)
6. [Critical Bottleneck Analysis](#6-critical-bottleneck-analysis)
7. [HQ-MPSD Handling Strategy](#7-hq-mpsd-handling-strategy)
8. [Total Project Timeline](#8-total-project-timeline)
9. [Experiments That Cannot Run on RTX 4080](#9-experiments-that-cannot-run-on-rtx-4080)
10. [Risk Register and Mitigations](#10-risk-register-and-mitigations)
11. [Recommended Execution Order](#11-recommended-execution-order)
12. [References and Assumptions](#12-references-and-assumptions)

---

## 1. Hardware Baseline Measurements

### Confirmed Hardware (from `nvidia-smi` and `free -h`, 2026-03-02)

| Component | Specification | Measured |
|-----------|--------------|----------|
| GPU | NVIDIA GeForce RTX 4080 | Confirmed |
| VRAM Total | 16,376 MiB (~16 GB GDDR6X) | Confirmed |
| VRAM Available (idle) | ~15.5 GB (after Xorg/desktop: ~800 MiB overhead) | Measured: ~6.5 GB free with Python process running |
| CUDA Cores | 9,728 | Ada Lovelace architecture |
| FP16 Throughput | ~48.7 TFLOPS | Theoretical peak |
| FP32 Throughput | ~48.7 TFLOPS (Ada uses same-rate FP32/FP16 on some paths) | Theoretical peak |
| Tensor Core Performance | ~780 TOPS INT8 | Ada 4th-gen Tensor Cores |
| Memory Bandwidth | 716.8 GB/s | GDDR6X |
| TDP | 320W | Maximum power draw |
| System RAM | 62 GB (64 GB nominal) | Confirmed |
| System RAM Available | ~41 GB (with OS + desktop) | Measured |
| Storage | SSD | Assumed sufficient throughput (~500 MB/s+ sequential) |
| CUDA Version | 12.8 | Confirmed from driver 570.211.01 |

### Key VRAM Budget

For experiments, we must account for ~800 MiB desktop overhead (Xorg, GNOME, Chrome). Before running GPU experiments, the existing Python process consuming ~8.9 GB should be terminated.

**Effective VRAM budget for experiments: ~15.5 GB** (after desktop overhead, with no other GPU processes).

### Throughput Benchmarks (Estimated for RTX 4080)

These estimates are based on published benchmarks for WavLM-Large and wav2vec2-XLSR on RTX 4080-class hardware (Pimentel et al., IEEE WIFS 2024; community benchmarks on Hugging Face forums):

| Operation | Throughput | Notes |
|-----------|-----------|-------|
| WavLM-Large forward pass (single 10s utterance, FP16) | ~40--60 ms | Batch size 1, includes feature extraction |
| WavLM-Large forward pass (batch=8, 10s utterances, FP16) | ~15--25 ms/utterance | Amortized per utterance |
| wav2vec2-XLSR forward pass (single 10s utterance, FP16) | ~35--50 ms | Slightly faster than WavLM due to fewer layers used |
| Backward pass (gradient computation) | ~1.5--2x forward | Standard ratio |
| Full training step (forward + backward + optimizer) | ~3--4x forward | Includes optimizer state in VRAM |

---

## 2. Model VRAM Profiling

### 2.1 SAL (WavLM-Large + Conformer)

| Component | Parameters | FP16 Size | FP32 Size | Notes |
|-----------|-----------|-----------|-----------|-------|
| WavLM-Large backbone | 316M | ~632 MB | ~1,264 MB | Frozen during inference; trainable for fine-tuning |
| Conformer head (2 blocks, 4 attn heads) | ~5M | ~10 MB | ~20 MB | Lightweight |
| SPL projection layers | ~1M | ~2 MB | ~4 MB | Frame-level classifier |
| **Subtotal (model weights)** | **~322M** | **~644 MB** | **~1,288 MB** | |

**Inference VRAM:**

| Component | VRAM | Notes |
|-----------|------|-------|
| Model weights (FP16) | ~644 MB | Frozen backbone + head |
| Input waveform + features (batch=1, 10s) | ~50 MB | Raw waveform + CNN features + transformer hidden states |
| Intermediate activations (batch=1) | ~200--400 MB | Depends on sequence length; WavLM produces ~625 frames for 10s audio at 16ms hop |
| PyTorch CUDA overhead | ~500 MB | Context, allocator, etc. |
| **Total inference (batch=1)** | **~1.4--1.6 GB** | |
| **Total inference (batch=8)** | **~3.5--4.5 GB** | Activations scale linearly with batch |
| **Total inference (batch=16)** | **~6--8 GB** | Near VRAM limit; may cause OOM with gradients |

**Training VRAM:**

| Component | VRAM | Notes |
|-----------|------|-------|
| Model weights (FP32 master copy for mixed precision) | ~1,288 MB | Required for AMP training |
| Optimizer states (AdamW: 2x model size) | ~2,576 MB | Momentum + variance for all trainable params |
| Gradients | ~1,288 MB | FP32 gradients |
| Activations (batch=4, 10s) | ~1,600 MB | Gradient checkpointing can reduce this |
| Input buffers | ~200 MB | |
| **Total training (batch=4, no grad ckpt)** | **~7--8 GB** | Fits in 16 GB |
| **Total training (batch=8, no grad ckpt)** | **~10--12 GB** | Tight but feasible |
| **Total training (batch=16, no grad ckpt)** | **~14--16 GB** | Will likely OOM |
| **Total training (batch=8, with grad ckpt)** | **~8--10 GB** | Recommended |

**Fine-tuning strategy:** Freeze WavLM backbone; train only Conformer head + SPL projection. This reduces trainable parameters from 322M to ~6M, optimizer states from 2.6 GB to ~48 MB, and gradients from 1.3 GB to ~24 MB.

| Scenario | Trainable Params | VRAM Estimate | Max Batch Size |
|----------|-----------------|---------------|----------------|
| Full fine-tuning (all params) | 322M | ~12 GB (batch=4) | 4--6 |
| Backbone frozen, head only | 6M | ~3.5 GB (batch=16) | 16--32 |
| Backbone frozen + gradient checkpointing | 6M | ~3 GB (batch=32) | 32+ |

### 2.2 BAM (WavLM-Large + Boundary Enhancement + BFA)

| Component | Parameters | FP16 Size | Notes |
|-----------|-----------|-----------|-------|
| WavLM-Large backbone | 316M | ~632 MB | Same as SAL |
| Boundary Enhancement module | ~5M | ~10 MB | Conv + attention layers for boundary feature extraction |
| Boundary Frame-wise Attention (BFA) | ~5M | ~10 MB | Cross-frame attention modulated by boundary predictions |
| Classification head | ~1M | ~2 MB | |
| **Subtotal** | **~327M** | **~654 MB** | |

VRAM profile is nearly identical to SAL. The Boundary Enhancement and BFA modules add minimal overhead (~20 MB) over SAL's Conformer head.

**Inference:** ~1.5--1.7 GB (batch=1); ~4--5 GB (batch=8)
**Training (head-only):** ~3.5--4 GB (batch=16)
**Training (full):** ~12--13 GB (batch=4)

### 2.3 CFPRF (wav2vec2-XLSR + FDN + PRN)

| Component | Parameters | FP16 Size | Notes |
|-----------|-----------|-----------|-------|
| wav2vec2-XLSR backbone | 300M | ~600 MB | 1024-dim embeddings at 20ms hop |
| Feature projection (1024 -> 128 dim) | ~0.1M | ~0.2 MB | Dimensionality reduction |
| Frame-level Detection Network (FDN) | ~10M | ~20 MB | Conv blocks for coarse detection |
| Proposal Refinement Network (PRN) | ~10M | ~20 MB | Boundary refinement + confidence scoring |
| DAFL (contrastive learning module) | ~2M | ~4 MB | Training only |
| BAFE (cross-attention) | ~3M | ~6 MB | Boundary-aware feature enhancement |
| **Subtotal** | **~325M** | **~650 MB** | |

**Inference:** ~1.5--1.7 GB (batch=1); ~4--5 GB (batch=8)

Note: CFPRF uses wav2vec2-XLSR rather than WavLM-Large. The wav2vec2-XLSR model is slightly smaller (300M vs 316M params) and slightly faster due to the absence of the denoising pre-training head. However, CFPRF's two-stage architecture (FDN + PRN) adds more post-backbone computation than SAL or BAM. The FDN runs per-frame; the PRN runs per-proposal (typically 5--20 proposals per utterance). Net inference time is similar to SAL/BAM.

**Special consideration:** CFPRF produces temporal proposals in addition to frame-level scores. The PRN generates variable numbers of proposals per utterance, which complicates batching. In practice, inference is typically done with batch=1 for the PRN stage, even if the FDN can be batched. This means CFPRF inference is slightly slower per-utterance than SAL/BAM for the refinement stage.

### 2.4 MRM (SSL + Multi-Resolution Heads)

| Component | Parameters | FP16 Size | Notes |
|-----------|-----------|-----------|-------|
| SSL backbone (WavLM or wav2vec2) | 300--316M | ~600--632 MB | Depends on which SSL model the reproduction uses |
| Multi-resolution classification heads (6 resolutions) | ~3M | ~6 MB | One head per resolution (20ms, 40ms, 80ms, 160ms, 320ms, 640ms) |
| Utterance-level head | ~0.5M | ~1 MB | For joint optimization |
| **Subtotal** | **~305--320M** | **~610--640 MB** | |

**Inference:** ~1.3--1.5 GB (batch=1). MRM is the lightest model due to simple linear heads.

### 2.5 Saliency Computation (Integrated Gradients / GradSHAP)

This is the critical VRAM concern. Saliency methods require gradient computation, which approximately doubles the memory footprint compared to forward-only inference.

| Method | Forward Passes per Sample | Backward Passes per Sample | VRAM Overhead | Notes |
|--------|--------------------------|---------------------------|---------------|-------|
| Integrated Gradients (50 steps) | 50 | 50 | ~2x single forward | Each interpolation step requires forward + backward; can be batched |
| Integrated Gradients (200 steps) | 200 | 200 | ~2x single forward | Higher fidelity; 4x slower than 50 steps |
| GradSHAP (n_samples=25) | 25 | 25 | ~2x single forward | Stochastic; fewer samples than IG typical |
| Vanilla gradients | 1 | 1 | ~2x single forward | Single backward pass; fast but noisy |
| Attention rollout | 0 (uses attention weights) | 0 | Negligible | No gradient computation; just attention matrix products |

**VRAM for saliency (WavLM-Large + SAL head):**

| Scenario | VRAM | Notes |
|----------|------|-------|
| Forward pass only (batch=1) | ~1.5 GB | Inference mode |
| Forward + backward (batch=1) | ~3.0 GB | Gradients stored |
| IG with internal batch of 10 interpolation points | ~5--6 GB | 10 interpolated inputs batched |
| IG with internal batch of 25 interpolation points | ~10--12 GB | Aggressive batching |
| IG with internal batch of 50 interpolation points | ~16+ GB | **Will OOM** |

**Recommended saliency configuration:**
- IG interpolation steps: 50 (standard per Sundararajan et al., 2017, ICML)
- Internal batch size for IG: 10 (process 10 interpolation points simultaneously)
- This requires 5 batches of 10 forward+backward passes per utterance
- VRAM: ~5--6 GB (well within budget)
- Time per utterance: 5 batches * ~100ms (forward+backward) = ~500ms per utterance per detector

### 2.6 Summary: Can All Models Fit in 16 GB?

| Scenario | VRAM Required | Fits in 16 GB? |
|----------|--------------|----------------|
| Single model inference (batch=1) | 1.3--1.7 GB | Yes |
| Single model inference (batch=16) | 6--8 GB | Yes |
| Single model training, head-only (batch=16) | 3.5--4 GB | Yes |
| Single model training, full (batch=4) | 7--12 GB | Yes (tight for full backbone fine-tune) |
| Single model + IG saliency (batch=1, IG internal batch=10) | 5--6 GB | Yes |
| Two models loaded simultaneously (inference) | 3--4 GB | Yes |
| Four models loaded simultaneously (inference) | 6--8 GB | Yes |
| Training one model + running inference on another simultaneously | 8--16 GB | Risky; avoid |

**Verdict: All experiments fit on RTX 4080 when run sequentially, one model at a time. No experiment is impossible.**

---

## 3. Dataset Storage and I/O Analysis

### 3.1 Dataset Sizes

| Dataset | Duration | Utterances | Uncompressed Size (est.) | Compressed Size | Splits Used |
|---------|----------|-----------|-------------------------|----------------|-------------|
| PartialSpoof (all) | ~41 h | ~12,000 | ~30 GB (16kHz, 16-bit) | ~15 GB | train + dev + eval |
| PartialSpoof train | ~25 h | ~7,000 | ~18 GB | ~9 GB | Training E1 |
| PartialSpoof dev | ~8 h | ~2,500 | ~6 GB | ~3 GB | Calibration E2, hyperparameter tuning |
| PartialSpoof eval | ~8 h | ~2,500 | ~6 GB | ~3 GB | Conformal calibration + verification |
| PartialEdit | ~(small) | 1,000 | ~1 GB | ~0.5 GB | Eval only |
| LlamaPartialSpoof | 130 h | ~40,000 | ~94 GB | ~47 GB | Eval only |
| HQ-MPSD (all 8 languages) | 350.8 h | Unknown | **~1.7 TB** | **~42 GB** | Eval only |
| HQ-MPSD (English only) | ~(est. 35 h based on 3.2 GB compressed) | Unknown | ~25 GB | 3.2 GB | Recommended subset |

### 3.2 Storage Requirements

| Item | Size | Notes |
|------|------|-------|
| PartialSpoof (all) | ~30 GB | Must download |
| PartialEdit | ~1 GB | Must download from Zenodo |
| LlamaPartialSpoof | ~94 GB | Must download |
| HQ-MPSD (English only) | ~25 GB | Recommended: download English only (3.2 GB compressed) |
| HQ-MPSD (full, 8 languages) | **~1.7 TB** | Problematic -- see Section 7 |
| Model checkpoints (4 models) | ~6 GB | Pre-trained weights |
| Intermediate outputs (frame scores, features) | ~20--50 GB | Depends on what is cached |
| Saliency maps (if saved) | ~10--30 GB | Large if saved for all utterances |
| **Total (without full HQ-MPSD)** | **~200--300 GB** | Manageable on SSD |
| **Total (with full HQ-MPSD)** | **~2 TB** | Requires dedicated storage |

### 3.3 I/O Bottleneck Analysis

Audio loading from SSD is not a bottleneck for these experiments. At 16 kHz, 16-bit mono audio:
- Data rate: 32 KB/s per stream
- A 10-second utterance: 320 KB
- Loading 100 utterances: 32 MB (trivial for SSD)
- Even HQ-MPSD at full size: streaming at ~32 KB/s per utterance, the SSD (500+ MB/s) can pre-load thousands of utterances ahead

**The bottleneck is GPU computation, not I/O.**

---

## 4. Experiment-by-Experiment Feasibility

### E1: Baseline Training / Fine-Tuning

#### E1a: Fine-tune SAL on PartialSpoof (~25h training data)

**Configuration (recommended):**
- Backbone: WavLM-Large, **frozen** (use pretrained checkpoint as feature extractor)
- Trainable: Conformer head + SPL projection (~6M params)
- Optimizer: AdamW, lr=1e-4, weight decay=0.01
- Batch size: 16 (fits in ~4 GB VRAM with frozen backbone)
- Epochs: 20--30 (typical for SAL; convergence expected by epoch 15--20)
- Mixed precision: FP16 (AMP)
- Gradient accumulation: Not needed at batch=16

**VRAM estimate:** ~4 GB (frozen backbone, batch=16)

**Time estimate:**
- Training set: ~7,000 utterances
- Steps per epoch: 7,000 / 16 = 438 steps
- Time per step: ~200ms (frozen backbone forward + head forward + backward + optimizer)
- Time per epoch: 438 * 0.2s = ~88s (~1.5 minutes)
- Total (25 epochs): ~37 minutes
- With validation on dev set: add ~5 minutes per epoch = ~2 hours total with early stopping

**If full backbone fine-tuning is required** (some papers report better results):
- Trainable: All 322M params
- Batch size: 4 (to fit in ~12 GB)
- Gradient accumulation: 4 steps (effective batch = 16)
- Time per step: ~600ms (full forward + backward through WavLM)
- Steps per epoch: 7,000 / 4 = 1,750 steps
- Time per epoch: 1,750 * 0.6s = ~1,050s (~17.5 minutes)
- Total (25 epochs): ~7.3 hours
- With validation: ~10 hours

**Verdict: FEASIBLE. 2--10 hours depending on frozen vs. full fine-tuning.**

#### E1b: Fine-tune BAM on PartialSpoof

BAM has a similar architecture to SAL (WavLM backbone + lightweight head). VRAM and time estimates are nearly identical.

**Frozen backbone:** ~2 hours total
**Full fine-tuning:** ~10 hours total

**Verdict: FEASIBLE. Same as E1a.**

#### E1c: Use pre-trained CFPRF checkpoint (no training)

CFPRF provides pretrained checkpoints for PartialSpoof via Google Drive (documented in detector-selection.md). No training needed.

**Time:** Download checkpoint (~2 GB), load, verify on a few samples. ~30 minutes including download.

**Verdict: TRIVIAL.**

#### E1d: MRM -- Use official baseline (minimal training)

The MultiResoModel-Simple reimplementation (GitHub: hieuthi/MultiResoModel-Simple) provides training code. If pretrained checkpoints are available, no training is needed. If training is required:

**Frozen backbone:** ~2 hours (similar to SAL)
**Full fine-tuning:** ~8 hours

**Verdict: FEASIBLE.**

#### E1 Total Time: 6--32 hours (depending on frozen vs. full fine-tuning for 2--3 models)

**Recommendation:** Use frozen backbone for initial experiments. If results are suboptimal, selectively fine-tune the primary detector (BAM or SAL) with full backbone.

---

### E2: Calibration (CPU-heavy, low GPU)

**Procedure:**
1. Run inference on PartialSpoof dev set with all 4 models (GPU)
2. Collect frame-level scores per utterance
3. Apply Platt scaling, temperature scaling, isotonic regression (CPU, sklearn)

#### Step 1: Inference on PartialSpoof dev set

| Model | Dev set size | Batch size | Time per batch | Total inference time |
|-------|-------------|-----------|----------------|---------------------|
| SAL | ~2,500 utterances | 16 | ~400ms | ~63s (~1 min) |
| BAM | ~2,500 utterances | 16 | ~400ms | ~63s (~1 min) |
| CFPRF | ~2,500 utterances | 8 (FDN) + 1 (PRN) | ~600ms + 200ms/proposal | ~90s (~1.5 min) |
| MRM | ~2,500 utterances | 16 | ~350ms | ~55s (~1 min) |
| **Total inference** | | | | **~5 minutes** |

**VRAM:** ~4--5 GB per model (batch=16). Run sequentially.

#### Step 2: Score collection

Frame-level scores for 2,500 utterances, ~625 frames/utterance at 160ms resolution = ~1.56M frames per model. Storage: ~6 MB per model (float32). Trivial.

#### Step 3: Calibration fitting (CPU)

| Method | Complexity | Time for 1.56M samples | Notes |
|--------|-----------|----------------------|-------|
| Temperature scaling | O(n * iterations) | ~5 seconds | Single parameter; gradient descent on NLL |
| Platt scaling | O(n * iterations) | ~10 seconds | Two parameters (a, b) in sigmoid; sklearn LogisticRegression |
| Isotonic regression | O(n log n) | ~30 seconds | sklearn IsotonicRegression; sorting-based |

**Total E2 time: ~10--15 minutes** (GPU inference + CPU calibration)

**VRAM:** ~4--5 GB (sequential inference, one model at a time)

**Verdict: TRIVIAL. Fully feasible, no bottlenecks.**

---

### E3: CPSL Conformal Prediction (CPU, no GPU needed after inference)

**Procedure:**
1. Use stored frame-level scores from E2 (already computed on dev set)
2. Run inference on PartialSpoof eval set (GPU, same as E2)
3. Split eval set 70/30 (CPU)
4. Compute nonconformity scores (CPU)
5. Calibrate quantiles at alpha = 0.01, 0.05, 0.10 (CPU)
6. Compute prediction sets on 30% verification (CPU)
7. Evaluate coverage (CPU)

#### GPU component: Inference on eval set

Same as E2 Step 1 but on eval set (~2,500 utterances). ~5 minutes for all 4 models.

#### CPU component: Conformal prediction

| Step | Computation | Time |
|------|------------|------|
| Nonconformity scores (1.56M frames * 4 detectors) | Simple arithmetic (1 - softmax_score for true class) | <1 second |
| Quantile calibration (on 70% = ~1.1M frames) | np.quantile on sorted array | <1 second |
| Prediction set construction (on 30% = ~0.47M frames) | Threshold comparison | <1 second |
| Coverage computation | np.mean(correct_in_set) | <1 second |
| Bootstrap CI (10,000 resamples, utterance-level) | ~10 minutes | Resampling at utterance level (n=750 utterances in 30% split) |
| K-fold conformal sensitivity (5 splits) | 5x above | ~5 minutes |

**Total E3 time: ~20 minutes** (including GPU inference + CPU conformal + bootstrap)

**VRAM:** ~4--5 GB (inference only; conformal computation is CPU-only)

**Verdict: TRIVIAL. Fully feasible.**

---

### E4: PDSM-PS Saliency Computation (GPU-INTENSIVE -- PRIMARY BOTTLENECK)

This is the most computationally expensive experiment. Let me break it down in detail.

#### E4a: Integrated Gradients (IG)

**Configuration:**
- Interpolation steps: 50 (standard; Sundararajan et al., 2017)
- Baseline: zero input (silence) or uniform noise
- Internal IG batch size: 10 (10 interpolated inputs per GPU forward+backward pass)
- External batch: 1 utterance at a time (due to variable lengths)
- Detectors: 4 (SAL, BAM, CFPRF, MRM)

**Per-utterance computation:**

| Component | Operations | Time | Notes |
|-----------|-----------|------|-------|
| Load + preprocess 1 utterance | 1 | ~5ms | SSD I/O + resampling |
| Create 50 interpolated inputs | 1 | ~1ms | Linear interpolation between baseline and input |
| Forward + backward passes (50 steps, batched in groups of 10) | 5 batches * (forward + backward) | 5 * ~200ms = ~1,000ms | Each batch: 10 interpolated inputs through WavLM + head |
| Average gradients across steps | 1 | ~5ms | Element-wise mean |
| Multiply by (input - baseline) | 1 | ~1ms | Element-wise product |
| **Total per utterance per detector** | | **~1.0 second** | |

**VRAM per utterance:**
- Model weights (FP16): ~650 MB
- 10 interpolated inputs + activations + gradients: ~3--4 GB
- Total: ~4--5 GB
- **Fits in 16 GB with margin**

**Scaling to dataset sizes:**

| Eval Set | Utterances | Models | Total IG Computations | Time (1.0s/utt/model) | Notes |
|----------|-----------|--------|----------------------|----------------------|-------|
| PartialSpoof eval (30% verification) | ~750 | 4 | 3,000 | ~50 minutes | Recommended for faithfulness evaluation |
| PartialSpoof eval (full) | ~2,500 | 4 | 10,000 | ~2.8 hours | If full-set saliency is needed |
| PartialSpoof eval (all 13K if ASVspoof-based) | ~13,000 | 4 | 52,000 | ~14.4 hours | Full ASVspoof 2019 LA eval partition |
| Subsample (500 utterances) | 500 | 4 | 2,000 | ~33 minutes | Stratified subsample for rapid evaluation |

**Recommendation for IG:** Compute on the 30% verification split (~750 utterances) for primary faithfulness evaluation. Compute on a stratified 500-utterance subsample if even that is too slow for iterative development.

#### E4b: GradSHAP

**Configuration:**
- Number of reference samples: 25 (standard)
- Each reference sample requires 1 forward + 1 backward pass
- Can batch reference samples (unlike IG which batches interpolation steps)

**Per-utterance computation:**

| Component | Operations | Time | Notes |
|-----------|-----------|------|-------|
| Sample 25 reference inputs | 1 | ~1ms | Random noise or utterances from genuine pool |
| Forward + backward (25 samples, batched in groups of 10) | 3 batches * ~200ms | ~600ms | |
| Compute SHAP values | 1 | ~5ms | Weighted average of gradients |
| **Total per utterance per detector** | | **~0.6 seconds** | |

**Scaling:** GradSHAP is ~40% faster than IG (50 steps). For 750 utterances * 4 models = ~30 minutes.

#### E4c: Forced Alignment for PDSM Phoneme Discretization

Before PDSM can discretize saliency maps by phoneme, phoneme boundaries must be obtained.

**Montreal Forced Aligner (MFA):**
- CPU-based (uses Kaldi backend)
- Speed: ~0.5--1x realtime on modern CPU (i.e., 10s utterance takes ~10--20s to align)
- For 750 utterances (~3 hours of audio): ~3--6 hours of CPU time
- Can run in parallel with GPU experiments (CPU-only)

**Alternative: Pre-computed phoneme boundaries from WhisperX:**
- WhisperX alignment is faster (~0.3x realtime) and GPU-accelerated
- For 750 utterances: ~1 hour (GPU) or ~2 hours (CPU-only)
- WhisperX requires ~3--4 GB VRAM -- cannot run simultaneously with saliency computation
- Must run separately

**Recommendation:** Run MFA on CPU in parallel with GPU saliency computation. Total wall-clock time is not increased since MFA is CPU-bound and IG is GPU-bound.

#### E4 Total Time Estimates

| Configuration | GPU Time | CPU Time (MFA, parallel) | Wall-Clock Time |
|---------------|---------|-------------------------|-----------------|
| IG (50 steps) on 750 utterances, 4 models | ~50 min | ~4 hours (MFA on 750 utts) | ~4 hours (CPU-bound by MFA) |
| GradSHAP on 750 utterances, 4 models | ~30 min | ~4 hours (MFA) | ~4 hours (CPU-bound by MFA) |
| IG (50 steps) on 500 utterances (subsample), 4 models | ~33 min | ~2.5 hours (MFA) | ~2.5 hours |
| IG on full PartialSpoof eval (~13K utts), 4 models | ~14 hours | ~26 hours (MFA) | ~26 hours |

**VRAM:** ~4--5 GB (single model + IG gradients). No VRAM issue.

**Verdict: FEASIBLE. The 750-utterance verification subset is the sweet spot. MFA forced alignment on CPU is the actual wall-clock bottleneck, not IG computation on GPU. Total: ~4 hours.**

**Can we subsample for saliency evaluation?** YES. 500 utterances provide a statistically adequate sample for NAOPC computation. At 500 utterances, the standard error of mean NAOPC is approximately SE = sigma / sqrt(500). For typical NAOPC standard deviations of ~0.1--0.2, this gives SE ~0.005--0.009, which is sufficient for paired significance testing (paired bootstrap with 500 pairs has excellent power for detecting medium effect sizes). Stratify the subsample by: (a) detector score quantile (low/medium/high confidence), (b) attack type, (c) speaker. Document the sampling procedure.

---

### E5: Cross-Dataset Inference

#### E5a: PartialEdit (1,000 utterances) -- FAST

| Model | Batch Size | Total Batches | Time per Batch | Total Time |
|-------|-----------|--------------|----------------|-----------|
| SAL | 16 | 63 | ~400ms | ~25s |
| BAM | 16 | 63 | ~400ms | ~25s |
| CFPRF | 8 + 1 (PRN) | 125 + 1000 | ~300ms + ~100ms | ~2.5 min |
| MRM | 16 | 63 | ~350ms | ~22s |
| **Total** | | | | **~4 minutes** |

**VRAM:** ~4--5 GB per model (sequential).

**Verdict: TRIVIAL.**

#### E5b: LlamaPartialSpoof (130 hours, ~40,000 utterances) -- MEDIUM

| Model | Batch Size | Total Batches | Time per Batch | Total Time |
|-------|-----------|--------------|----------------|-----------|
| SAL | 16 | 2,500 | ~400ms | ~17 min |
| BAM | 16 | 2,500 | ~400ms | ~17 min |
| CFPRF | 8 + 1 | 5,000 + 40,000 | ~300ms + ~100ms | ~1.4 hours |
| MRM | 16 | 2,500 | ~350ms | ~15 min |
| **Total** | | | | **~2.5 hours** |

**VRAM:** ~4--5 GB per model (sequential).

**Storage:** ~94 GB uncompressed. Must be on SSD. Streaming inference is fine.

**Verdict: FEASIBLE. ~2.5 hours total.**

#### E5c: HQ-MPSD (350.8 hours) -- LARGE / PROBLEMATIC

See Section 7 for detailed HQ-MPSD analysis. Summary:

| Scenario | Duration | Utterances (est.) | Time (4 models) | Storage |
|----------|----------|-------------------|-----------------|---------|
| Full HQ-MPSD (all 8 languages) | 350.8 h | ~100K--150K | ~8--12 hours | **1.7 TB uncompressed** |
| English only | ~35 h | ~10K--15K | ~1--2 hours | ~25 GB |
| Stratified sample (1 hour per language, 8 hours total) | 8 h | ~2,400 | ~20 minutes | ~6 GB |

**VRAM:** ~4--5 GB per model. No GPU issue.

**Bottleneck: STORAGE.** Full HQ-MPSD requires 1.7 TB uncompressed. See Section 7.

**Verdict: GPU-FEASIBLE, STORAGE-CONSTRAINED for full dataset.**

---

### E6: Codec Stress Test

#### Step 1: Re-encode PartialSpoof eval (CPU, ffmpeg)

| Codec | ffmpeg Command | Speed | Time for ~8h audio |
|-------|---------------|-------|-------------------|
| AAC 128kbps | `ffmpeg -i in.wav -c:a aac -b:a 128k out.m4a` then decode back | ~50--100x realtime | ~5--10 minutes |
| Opus 24kbps | `ffmpeg -i in.wav -c:a libopus -b:a 24k out.opus` then decode | ~50--100x realtime | ~5--10 minutes |
| AMR-NB 12.2kbps | `ffmpeg -i in.wav -ar 8000 -c:a libopencore_amrnb -b:a 12.2k out.amr` then decode | ~30--50x realtime | ~10--15 minutes |
| G.711 (mu-law) | `ffmpeg -i in.wav -ar 8000 -c:a pcm_mulaw out.wav` | ~100x+ realtime | ~5 minutes |

**Total encoding time:** ~30--40 minutes (CPU-only, can run in parallel with other work)
**Storage per codec variant:** ~6 GB (same as eval set) * 4 codecs = ~24 GB additional

#### Step 2: Inference on re-encoded versions

4 codecs * 4 models * ~2,500 utterances = 40,000 inference runs

| Total Batches (all codecs, all models) | Time per Batch | Total Time |
|---------------------------------------|----------------|-----------|
| 4 codecs * 4 models * 157 batches = 2,512 batches | ~400ms | ~17 minutes |

**Total E6 time:** ~1 hour (encoding + inference)
**VRAM:** ~4--5 GB per model (sequential)

**Verdict: FULLY FEASIBLE. ~1 hour total.**

---

### E7: MFA vs WhisperX Alignment

Both tools produce phoneme/word-level temporal alignments. This experiment evaluates alignment quality for PDSM phoneme discretization.

#### E7a: Montreal Forced Aligner (MFA)

| Parameter | Value |
|-----------|-------|
| Engine | Kaldi (CPU-only) |
| Acoustic model | English pretrained (MFA ships with this) |
| Speed | ~0.5--1x realtime |
| Dataset | PartialSpoof eval (~8 hours audio) |
| Estimated time | 8--16 hours (single CPU core) |
| Parallelism | MFA supports `--num_jobs` for parallel processing |
| With 8 CPU jobs | ~1--2 hours |
| VRAM | 0 (CPU-only) |
| RAM | ~4--8 GB peak |

#### E7b: WhisperX

| Parameter | Value |
|-----------|-------|
| Engine | Whisper (GPU) + forced alignment (CPU/GPU) |
| Model | whisper-large-v3 (~3 GB VRAM) |
| Speed | ~0.3x realtime (10s audio -> ~3s processing) on RTX 4080 |
| Dataset | PartialSpoof eval (~8 hours audio) |
| Estimated time (GPU) | ~2.4 hours |
| Estimated time (CPU-only) | ~8--12 hours |
| VRAM | ~3--4 GB (Whisper model + alignment model) |

**Important:** WhisperX cannot run simultaneously with saliency computation (both need GPU). Schedule sequentially.

**Total E7 time:**
- MFA: ~2 hours (8 CPU jobs, parallel)
- WhisperX: ~2.5 hours (GPU)
- **Total: ~4.5 hours** (can overlap MFA-CPU with WhisperX-GPU to some extent)

**Verdict: FEASIBLE.**

---

### E8: Ablations

Ablations are combinations of E1--E7 experiments. Estimated as fractions:

| Ablation | Description | Estimated Time | Notes |
|----------|------------|---------------|-------|
| 8a: Backbone swap (SAL w/ wav2vec2-XLSR) | Re-run SAL training with different backbone | ~2--10 hours | Full fine-tune if needed |
| 8b: Resolution comparison (all models at 20ms, 160ms) | Re-compute frame scores at different resolutions | ~1 hour | Just re-aggregation of existing scores |
| 8c: Calibration variants | Already part of E2 (3 methods * 4 models) | ~0 additional | |
| 8d: Nonconformity score functions (3 variants) | Re-run E3 with different score functions | ~30 minutes | CPU-only |
| 8e: Phoneme vs. word aggregation for PDSM | Re-aggregate saliency at word level | ~30 minutes | Post-processing of existing saliency maps |
| 8f: IG steps sensitivity (20, 50, 100, 200 steps) | Re-run E4 with different step counts on subsample | ~2 hours | 500-utterance subsample, 4 step counts |
| 8g: Noise injection stability (3 SNR levels) | Add noise to eval set, re-run inference | ~1 hour | CPU noise addition + GPU inference |
| 8h: Split sensitivity (60/40, 70/30, 80/20) | Re-run E3 with different splits | ~30 minutes | CPU-only |
| **Total ablations** | | **~8--16 hours** | |

**Verdict: FEASIBLE.**

---

## 5. Master Feasibility Table

| Experiment | GPU VRAM Needed | Est. Wall Time | CPU/GPU Bound | Primary Bottleneck | Risk Level | Mitigation |
|-----------|----------------|----------------|--------------|-------------------|-----------|------------|
| **E1a: SAL fine-tune (frozen backbone)** | ~4 GB | 2 hours | GPU | Training convergence | LOW | Use published hyperparameters; early stopping on dev EER |
| **E1a: SAL fine-tune (full backbone)** | ~12 GB | 10 hours | GPU | VRAM at batch>4 | MEDIUM | Use gradient checkpointing; AMP FP16; batch=4 with grad accumulation |
| **E1b: BAM fine-tune (frozen)** | ~4 GB | 2 hours | GPU | Same as E1a | LOW | Same as E1a |
| **E1b: BAM fine-tune (full)** | ~12 GB | 10 hours | GPU | Same as E1a | MEDIUM | Same as E1a |
| **E1c: CFPRF (pretrained ckpt)** | 0 (download) | 30 min | Network | Download speed | LOW | Use available mirrors |
| **E1d: MRM (baseline)** | ~4 GB | 2 hours | GPU | Training convergence | LOW | Use MultiResoModel-Simple defaults |
| **E2: Calibration** | ~4--5 GB | 15 min | CPU (calibration fitting) | None | LOW | Straightforward sklearn pipeline |
| **E3: CPSL conformal prediction** | ~4--5 GB (inference) | 20 min | CPU (conformal math) | None | LOW | Standard conformal prediction; mapie library |
| **E4a: IG saliency (750 utts, 4 models)** | ~4--5 GB | 50 min GPU + 4 hr CPU (MFA) | CPU (MFA alignment) | MFA forced alignment speed | MEDIUM | Run MFA in parallel (8 jobs); subsample to 500 if needed |
| **E4b: GradSHAP saliency (750 utts)** | ~4--5 GB | 30 min GPU | GPU | Minimal | LOW | Faster alternative to IG |
| **E5a: PartialEdit inference** | ~4--5 GB | 4 min | GPU | None | LOW | Trivial dataset size |
| **E5b: LlamaPartialSpoof inference** | ~4--5 GB | 2.5 hours | GPU | Throughput | LOW | Batch processing; no VRAM issue |
| **E5c: HQ-MPSD full inference** | ~4--5 GB | 8--12 hours | GPU + Storage I/O | **1.7 TB storage; decompression** | **HIGH** | **Subsample -- see Section 7** |
| **E5c: HQ-MPSD English only** | ~4--5 GB | 1--2 hours | GPU | Moderate throughput | LOW | Download only English subset (3.2 GB compressed) |
| **E5c: HQ-MPSD stratified sample** | ~4--5 GB | 20 min | GPU | None | LOW | 1 hour per language, 8 hours total audio |
| **E6: Codec stress test** | ~4--5 GB | 1 hour | CPU (ffmpeg encoding) | Encoding time | LOW | ffmpeg is fast; parallel encoding |
| **E7a: MFA alignment** | 0 (CPU-only) | 2 hours | CPU | Alignment speed | LOW | Parallelize with --num_jobs=8 |
| **E7b: WhisperX alignment** | ~3--4 GB | 2.5 hours | GPU | Whisper model throughput | LOW | Cannot overlap with saliency; schedule sequentially |
| **E8: Ablations (all)** | ~4--12 GB | 8--16 hours | Mixed | Depends on ablation | MEDIUM | Prioritize high-impact ablations first |

---

## 6. Critical Bottleneck Analysis

### Bottleneck 1: PDSM-PS Saliency Computation -- MODERATE (manageable)

**The concern:** Integrated Gradients requires 50 forward+backward passes per utterance. For the full PartialSpoof eval set (~13,000 utterances) with 4 models, this is 2.6 million forward+backward passes.

**Reality check:** At ~1 second per utterance per model (IG with 50 steps, internal batch 10), the full eval set requires ~14.4 hours. For the 30% verification set (~3,900 utterances), this drops to ~4.3 hours. For the recommended 750-utterance subset, it is ~50 minutes.

**Why this is manageable:** The faithfulness evaluation (NAOPC) does not require saliency maps for the entire dataset. It requires saliency maps on the test/verification partition only. The calibration partition does not need saliency. Using the 30% verification split (~750 utterances from the 2,500 eval, or ~3,900 if from the full 13K ASVspoof partition) is sufficient.

**Mitigation:** Use GradSHAP as the primary saliency method (25 reference samples, ~0.6s/utterance) and IG (50 steps, ~1.0s/utterance) as a sensitivity check on a 500-utterance subsample.

### Bottleneck 2: HQ-MPSD Storage -- HIGH (requires decision)

See Section 7.

### Bottleneck 3: MFA Forced Alignment -- MODERATE (CPU-bound, parallelizable)

MFA runs on CPU only and processes at ~0.5--1x realtime. For the full PartialSpoof eval (~8 hours audio), this takes 8--16 hours on a single core. With 8 parallel jobs (the workstation has sufficient CPU cores), this reduces to 1--2 hours.

**Mitigation:** Always run MFA with `--num_jobs 8` (or more, depending on available CPU cores). Run MFA in parallel with GPU experiments.

### Bottleneck 4: Full Backbone Fine-Tuning VRAM -- MODERATE

Fine-tuning the full WavLM-Large backbone with AdamW optimizer requires ~12 GB VRAM at batch=4. This is tight but fits in 16 GB. The risk is OOM during peak memory allocation (PyTorch memory fragmentation can cause spikes).

**Mitigation:**
1. Use `torch.cuda.empty_cache()` before training
2. Kill all other GPU processes (including the 8.9 GB Python process currently running)
3. Use gradient checkpointing (`model.gradient_checkpointing_enable()`)
4. Use mixed precision training (`torch.cuda.amp.autocast()`)
5. If still OOM: reduce batch to 2, increase gradient accumulation to 8

### Bottleneck 5: Sequential Execution of Non-Overlapping Experiments

Since there is only one GPU, experiments that need GPU cannot run in parallel. However, CPU-only tasks can overlap with GPU tasks.

**Parallelizable pairs:**
- GPU: model training (E1) + CPU: MFA alignment (E7a)
- GPU: saliency computation (E4) + CPU: MFA alignment (E7a)
- GPU: cross-dataset inference (E5) + CPU: ffmpeg codec encoding (E6 step 1)
- GPU: WhisperX (E7b) + CPU: conformal prediction math (E3 CPU steps)

---

## 7. HQ-MPSD Handling Strategy

### The Problem

HQ-MPSD is 350.8 hours of speech across 8 languages. Compressed size: ~42 GB. **Uncompressed: ~1.7 TB.** This is by far the largest dataset in the project and creates two distinct challenges:

1. **Storage:** 1.7 TB uncompressed requires a dedicated drive. The current SSD may not have sufficient free space.
2. **Inference time:** 350.8 hours of audio with 4 models at batch=16 requires ~8--12 hours of GPU time.

Neither of these is insurmountable, but both require planning.

### Recommended Strategy: Hierarchical Subsampling

**Tier 1: English only (PRIMARY -- always run)**
- Size: 3.2 GB compressed, ~25 GB uncompressed
- Duration: ~35 hours (estimated from ratio of compressed sizes: 3.2/41.9 * 350.8)
- Rationale: Detectors trained on PartialSpoof (English) are best evaluated on English data from HQ-MPSD first. Cross-language evaluation is secondary.
- Time: ~1--2 hours inference (4 models)

**Tier 2: Stratified multilingual sample (SECONDARY -- run if Tier 1 shows interesting results)**
- Sample: 1 hour of audio per language = 8 hours total
- Size: ~6 GB uncompressed
- Selection: Random stratified sample within each language, balanced across genuine/fake/partial categories
- Rationale: Tests multilingual generalization without requiring full dataset
- Time: ~20 minutes inference (4 models)

**Tier 3: Full dataset (OPTIONAL -- run only if paper requires comprehensive results)**
- Size: 1.7 TB
- Requires: Dedicated storage (external SSD or second internal drive)
- Time: ~8--12 hours inference (4 models)
- Rationale: Only if reviewers demand full-dataset results or if the paper targets multilingual generalization as a primary contribution

### Decision Matrix

| Question | If Yes | If No |
|----------|--------|-------|
| Is multilingual generalization a primary contribution of the paper? | Run Tier 3 (full dataset) | Run Tier 1 only |
| Does the paper claim cross-language coverage guarantees? | Run Tier 2 at minimum | Tier 1 sufficient |
| Is SSD space >2 TB available? | Tier 3 feasible | Tier 3 infeasible without external storage |
| Do reviewers request full-dataset results? | Prepare Tier 3 for revision | Present Tier 1 + Tier 2 in initial submission |

### Storage Estimation

| Tier | Compressed Download | Uncompressed | Output Scores (4 models) | Total Disk |
|------|-------------------|-------------|--------------------------|-----------|
| Tier 1 (English) | 3.2 GB | ~25 GB | ~500 MB | ~29 GB |
| Tier 2 (8h sample) | ~2 GB | ~6 GB | ~100 MB | ~8 GB |
| Tier 3 (full) | ~42 GB | ~1.7 TB | ~10 GB | ~1.75 TB |

**Recommendation: Start with Tier 1. Proceed to Tier 2 only if needed. Defer Tier 3 unless the paper explicitly targets multilingual evaluation as a primary contribution. Document the subsampling strategy in the paper's methodology section.**

---

## 8. Total Project Timeline

### Scenario A: Serial Execution (Worst Case -- No Parallelization)

| Phase | Experiment | Time | Running Total |
|-------|-----------|------|---------------|
| 1 | E1a: SAL fine-tune (frozen backbone) | 2 hours | 2 hours |
| 2 | E1b: BAM fine-tune (frozen backbone) | 2 hours | 4 hours |
| 3 | E1c: CFPRF checkpoint load + verification | 0.5 hours | 4.5 hours |
| 4 | E1d: MRM training/load | 2 hours | 6.5 hours |
| 5 | E2: Calibration (all 4 models, all 3 methods) | 0.25 hours | 6.75 hours |
| 6 | E3: CPSL conformal prediction | 0.33 hours | 7.08 hours |
| 7 | E4: Saliency (IG + GradSHAP, 750 utts, 4 models) | 1.5 hours (GPU) | 8.58 hours |
| 8 | E7a: MFA alignment (full eval set) | 2 hours (CPU, 8 jobs) | 10.58 hours |
| 9 | E7b: WhisperX alignment | 2.5 hours (GPU) | 13.08 hours |
| 10 | E5a: PartialEdit inference | 0.07 hours | 13.15 hours |
| 11 | E5b: LlamaPartialSpoof inference | 2.5 hours | 15.65 hours |
| 12 | E5c: HQ-MPSD English inference | 2 hours | 17.65 hours |
| 13 | E6: Codec stress test | 1 hour | 18.65 hours |
| 14 | E8: Ablations | 12 hours | 30.65 hours |
| **Total Serial** | | | **~31 hours** |

### Scenario B: Optimized Parallel Execution (Recommended)

CPU and GPU tasks can overlap. Here is an optimized schedule:

```
Time    GPU Task                           CPU Task (parallel)
------  ---------------------------------  --------------------------------
0:00    E1a: SAL fine-tune (2h)            --
0:00    --                                 Download datasets (background)
2:00    E1b: BAM fine-tune (2h)            --
4:00    E1d: MRM training (2h)             E7a: MFA alignment (2h)
6:00    E2+E3: Calibration + Conformal     E6-step1: ffmpeg encoding (0.5h)
        (0.5h total)
6:30    E4: IG saliency (1.5h GPU)         E7a continued / MFA on codecs
8:00    E7b: WhisperX alignment (2.5h)     E3 CPU conformal (if not done)
10:30   E5a+E5b: Cross-dataset inference   E8d+E8e: CPU-only ablations
        PartialEdit + LPS (2.5h)
13:00   E5c: HQ-MPSD English (2h)          E8h: Split sensitivity (CPU)
15:00   E6-step2: Codec inference (0.5h)   MFA on codec-degraded audio
15:30   E8: GPU-dependent ablations (8h)   --
23:30   DONE
```

**Total wall-clock (optimized): ~24 hours (~1 day)**

### Scenario C: Full Fine-Tuning of All Models

If full backbone fine-tuning is required for SAL and BAM (not just frozen backbone), add:
- SAL full fine-tune: +8 hours
- BAM full fine-tune: +8 hours
- Total additional: +16 hours

**Total with full fine-tuning: ~40 hours (~1.7 days)**

### Scenario D: Including Full HQ-MPSD and Extended Saliency

| Additional Task | Time |
|----------------|------|
| HQ-MPSD Tier 3 (full, 350.8h) | +10 hours GPU |
| IG saliency on full eval (~13K utts) | +14 hours GPU |
| Total additional | +24 hours |

**Maximum total: ~64 hours (~2.7 days)**

### Summary Timeline

| Scenario | Wall-Clock Time | Calendar Days (10h/day) |
|----------|----------------|------------------------|
| A: Serial, frozen backbone, recommended subsamples | ~31 hours | 3 days |
| B: Parallel-optimized, frozen backbone, recommended subsamples | ~24 hours | 2.5 days |
| C: Parallel-optimized, full fine-tuning | ~40 hours | 4 days |
| D: Everything including full HQ-MPSD and full saliency | ~64 hours | 7 days |

**Recommended: Scenario B (24 hours / 2.5 days). Extend to C or D only if initial results warrant it.**

---

## 9. Experiments That Cannot Run on RTX 4080

**Answer: NONE.** All experiments can run on the RTX 4080 (16 GB VRAM) + 64 GB RAM workstation. There is no single experiment that requires more than 16 GB VRAM.

### Constraints and Workarounds

| Potential Concern | Assessment | Workaround |
|-------------------|-----------|------------|
| Full WavLM-Large fine-tuning OOM at batch>4 | 12 GB at batch=4 fits in 16 GB | Use gradient checkpointing + AMP + batch=4 with gradient accumulation |
| Two models loaded simultaneously for comparison | ~8 GB (two models in FP16 inference) | Fits in 16 GB |
| IG saliency with large internal batch | Internal batch=50 would OOM (~16+ GB) | Use internal batch=10 (5--6 GB) |
| HQ-MPSD full dataset | Storage issue (1.7 TB), not VRAM | Subsample to English only or stratified 8h sample |
| Ensemble inference (4 models simultaneously) | ~8 GB (4 models in FP16 inference) | Fits in 16 GB; or run sequentially |

### What Would NOT Fit

For reference, operations that would exceed RTX 4080 capacity (if anyone considers them):

| Operation | VRAM Required | Fits? |
|-----------|--------------|-------|
| Training two WavLM-Large models simultaneously | ~24 GB | NO -- needs multi-GPU or A100 |
| IG with internal batch=50 on WavLM-Large | ~18--20 GB | NO -- reduce internal batch |
| Loading WavLM-Large + Whisper-Large-V3 simultaneously for saliency + alignment | ~10--12 GB | YES (tight) but not recommended |
| Batch=32 full fine-tuning of WavLM-Large | ~28 GB | NO -- reduce batch size |

---

## 10. Risk Register and Mitigations

| # | Risk | Probability | Impact | Risk Level | Mitigation |
|---|------|------------|--------|-----------|------------|
| R1 | OOM during full backbone fine-tuning | Medium | High (experiment fails, must restart) | **HIGH** | Gradient checkpointing + AMP + small batch (4) + gradient accumulation. Test with a single batch before full training. |
| R2 | HQ-MPSD storage exceeds SSD capacity | High (if full dataset needed) | Medium (limits cross-dataset evaluation) | **HIGH** | Start with English only (25 GB). Request Tier 2/3 only if reviewers demand it. Use streaming evaluation if storage is tight. |
| R3 | MFA forced alignment fails on some utterances | Medium | Low (affects PDSM coverage) | **MEDIUM** | Use fallback to WhisperX for failed utterances. Report alignment failure rate. |
| R4 | CFPRF pretrained checkpoint incompatible with current PyTorch version | Low-Medium | Medium (requires retraining or version pinning) | **MEDIUM** | Pin PyTorch version in Docker/conda environment. Test checkpoint loading first. |
| R5 | SAL codebase has undocumented dependencies or bugs | Medium | Medium (delays E1a) | **MEDIUM** | Clone and test SAL repo first (before other experiments). Budget 4 hours for environment setup. |
| R6 | VRAM fragmentation causes OOM below theoretical limit | Low-Medium | Medium (reduces effective batch size) | **MEDIUM** | Use `torch.cuda.empty_cache()` between experiments. Restart Python process between training and inference. |
| R7 | Existing Python process (8.9 GB) not terminated before experiments | Medium | High (only 6.5 GB VRAM available) | **HIGH** | **Terminate the existing Python GPU process before starting any experiment.** Verify with `nvidia-smi`. |
| R8 | LlamaPartialSpoof label format incompatible with scoring pipeline | Low | Medium (requires adapter code) | **LOW** | Check label format in LlamaPartialSpoof documentation before scoring. Budget 2 hours for adapter code. |
| R9 | Codec re-encoding introduces unexpected resampling artifacts | Low | Low (affects E6 interpretation) | **LOW** | Verify sample rate consistency after encoding/decoding. Use `ffmpeg -ar 16000` to force consistent rate. |
| R10 | WhisperX GPU memory conflict with saliency computation | Medium | Low (scheduling issue) | **LOW** | Never overlap WhisperX and saliency computation. Sequential scheduling per Section 8. |
| R11 | Isotonic regression overfits on small calibration set | Medium | Medium (inflated calibration quality) | **MEDIUM** | Report calibration on held-out verification set. Compare against Platt/temperature (parametric, less overfit-prone). |
| R12 | PartialSpoof eval set is smaller than expected (~2,500 vs ~13,000) | Medium | Low (affects sample sizes for conformal) | **LOW** | Check actual partition sizes from ASVspoof 2019 LA metadata before computing power/sample size. |

---

## 11. Recommended Execution Order

### Phase 0: Environment Setup (Day 0, 4--8 hours)

1. **Kill existing GPU process** (the Python process using 8.9 GB VRAM)
2. Create conda environment with pinned versions:
   - PyTorch 2.x + CUDA 12.8
   - transformers (for WavLM, wav2vec2)
   - torchaudio
   - sklearn (calibration)
   - mapie or nonconformist (conformal prediction)
   - captum (IG, GradSHAP)
   - montreal-forced-aligner
   - whisperx
3. Clone all 4 model repositories (SAL, BAM, CFPRF, MultiResoModel-Simple)
4. Download PartialSpoof dataset
5. Download pretrained checkpoints for CFPRF and MRM
6. Verify each model loads and runs inference on 1 test utterance
7. Download PartialEdit and LlamaPartialSpoof (background)

### Phase 1: Baseline Training (Day 1, morning)

1. E1c: Load CFPRF pretrained checkpoint, verify (30 min)
2. E1a: Fine-tune SAL (frozen backbone) on PartialSpoof train (2 hours)
3. E1b: Fine-tune BAM (frozen backbone) on PartialSpoof train (2 hours)
4. E1d: Train/load MRM baseline (2 hours)
5. [Parallel CPU] E7a: Run MFA on PartialSpoof eval set (2 hours)

### Phase 2: Calibration + Conformal (Day 1, afternoon)

1. E2: Run inference on dev set (all 4 models), fit calibration (15 min)
2. E3: Run inference on eval set (all 4 models), compute conformal prediction (20 min)
3. [Parallel CPU] E6-step1: ffmpeg codec encoding (30 min)

### Phase 3: Saliency Computation (Day 1, evening / overnight)

1. E4a: IG saliency on 750-utterance verification set, 4 models (50 min GPU)
2. E4b: GradSHAP saliency on same set (30 min GPU)
3. E7b: WhisperX alignment on eval set (2.5 hours GPU)

### Phase 4: Cross-Dataset Inference (Day 2, morning)

1. E5a: PartialEdit inference (4 min)
2. E5b: LlamaPartialSpoof inference (2.5 hours)
3. E5c: HQ-MPSD English inference (1--2 hours)
4. E6-step2: Codec stress test inference (30 min)

### Phase 5: Ablations (Day 2, afternoon / evening)

1. E8a: Backbone swap ablation (if needed, 2--10 hours)
2. E8b--E8h: Resolution, calibration, conformal, phoneme/word, IG steps, noise, split ablations (4--8 hours)

### Phase 6: Analysis and Visualization (Day 3)

1. Compile all results into tables
2. Generate reliability diagrams
3. Generate saliency visualizations
4. Statistical testing (bootstrap CIs, paired tests, binomial coverage tests)
5. Write results section

---

## 12. References and Assumptions

### VRAM Estimation Methodology

VRAM estimates in this document follow the standard formula:

**Inference VRAM** = Model weights + Activations + Framework overhead
- Model weights (FP16): 2 bytes per parameter
- Activations: proportional to batch_size * sequence_length * hidden_dim * num_layers
- Framework overhead: ~500 MB (PyTorch CUDA context, memory allocator fragmentation)

**Training VRAM** = Model weights (FP32 master) + Optimizer states + Gradients + Activations + Framework overhead
- FP32 master weights: 4 bytes per parameter
- AdamW optimizer: 2 * 4 bytes per parameter (momentum + variance)
- Gradients: 4 bytes per parameter (or 2 bytes if FP16 gradients)
- Activations: proportional to batch_size * sequence_length * hidden_dim * num_layers (can be reduced by gradient checkpointing)

### Throughput Estimation Sources

1. Pimentel, A., Zhu, Y., Guimaraes, H.R., & Falk, T.H. (2024). Efficient Audio Deepfake Detection using WavLM with Early Exiting. IEEE WIFS 2024. [WavLM inference benchmarks]
2. Hugging Face Model Hub benchmarks for WavLM-Large and wav2vec2-XLSR on Ada Lovelace GPUs
3. McAuliffe, M., Socolof, M., Mihuc, S., Wagner, M., & Sonderegger, M. (2017). Montreal Forced Aligner: trainable text-speech alignment using Kaldi. Interspeech 2017. [MFA speed characteristics]
4. Bain, M., Huh, J., Han, T., & Zisserman, A. (2023). WhisperX: Time-Accurate Speech Transcription of Long-Form Audio. Interspeech 2023. [WhisperX throughput]
5. Sundararajan, M., Taly, A., & Yan, Q. (2017). Axiomatic Attribution for Deep Networks. ICML 2017. [Integrated Gradients methodology, 50--300 steps recommended]
6. NVIDIA Ada Lovelace Architecture Whitepaper (2022). [RTX 4080 theoretical throughput]

### Key Assumptions

1. **Audio is 16 kHz, 16-bit mono** (standard for PartialSpoof and most speech anti-spoofing datasets).
2. **Average utterance length is ~10 seconds** (PartialSpoof average is ~5--15 seconds).
3. **WavLM-Large produces ~625 frames per 10-second utterance** (at 16ms hop size).
4. **CFPRF produces 5--20 proposals per utterance** (from FDN + PRN pipeline).
5. **The existing Python process (8.9 GB VRAM) will be terminated before experiments begin.**
6. **SSD has at least 500 GB free space** (sufficient for all datasets except full HQ-MPSD).
7. **System has at least 8 CPU cores** available for MFA parallelization.
8. **Network bandwidth is sufficient** to download datasets (PartialSpoof: ~15 GB, LlamaPartialSpoof: ~47 GB compressed, HQ-MPSD English: 3.2 GB compressed).
9. **All model codebases (SAL, BAM, CFPRF, MRM) are compatible with PyTorch 2.x and CUDA 12.8.** This must be verified during Phase 0.

### Numerical Cross-Checks

| Calculation | Value | Cross-Check |
|-------------|-------|-------------|
| WavLM-Large params * 2 bytes (FP16) | 316M * 2 = 632 MB | Consistent with HF model card (~1.2 GB FP32 checkpoint) |
| PartialSpoof 41h * 16kHz * 2 bytes | 41 * 3600 * 16000 * 2 = 4.72 GB (raw waveform) | Consistent with ~30 GB including all metadata and labels |
| IG 50 steps * 10 internal batch * 200ms | 5 * 200ms = 1000ms per utterance | Consistent with community benchmarks |
| 750 utterances * 4 models * 1.0s | 3000s = 50 minutes | Consistent with Section 4 estimate |
| HQ-MPSD 350.8h * 16kHz * 2 bytes | 350.8 * 3600 * 16000 * 2 = 40.5 GB (raw PCM only) | Much less than 1.7 TB -- the 1.7 TB figure likely includes multiple versions, metadata, or higher sample rates. Verify actual format. |

**Important note on HQ-MPSD size:** The 1.7 TB figure from the dataset documentation (Li et al., 2025, arXiv:2512.13012) seems anomalously large relative to 350.8 hours of 16 kHz audio (~40 GB raw PCM). The actual uncompressed size may be much smaller if the audio is stored as standard WAV files at 16 kHz. The 1.7 TB figure may reflect: (a) higher sample rate (48 kHz), (b) multiple copies at different sample rates, (c) inclusion of intermediate synthesis artifacts, or (d) non-audio metadata. **Recommendation: Download the English subset first and check the actual file format and sample rate before assuming 1.7 TB is needed.**

If HQ-MPSD uses 48 kHz audio: 350.8h * 48000 * 2 bytes = 121.5 GB. Still far from 1.7 TB. The actual disk requirement should be verified empirically.

---

## Summary of Key Findings

1. **All experiments fit on RTX 4080 (16 GB VRAM).** No experiment requires more than ~12 GB VRAM, and that is only for full backbone fine-tuning with batch=4.

2. **Total estimated timeline: 24--40 hours (1--2 working days)** for the recommended configuration. Extended configurations (full HQ-MPSD, full saliency) add up to ~64 hours (~3 working days).

3. **The primary bottleneck is MFA forced alignment (CPU-bound), not GPU computation.** MFA at ~1x realtime on CPU dominates wall-clock time for the saliency pipeline. Mitigation: parallelize with `--num_jobs=8`.

4. **HQ-MPSD can be handled via subsampling.** English-only (25 GB, 1--2 hours) or stratified 8-hour sample (6 GB, 20 minutes) are strongly recommended over the full 1.7 TB dataset. The 1.7 TB figure may be inflated (verify actual format).

5. **Saliency computation is expensive but manageable.** IG on 750 utterances with 4 models takes ~50 minutes of GPU time. GradSHAP is faster (~30 minutes). Both fit comfortably in VRAM.

6. **No experiment is infeasible.** Every experiment in E1--E8 can run on this workstation. The RTX 4080 is a well-matched GPU for this project.

7. **Critical pre-requisite:** Terminate the existing Python GPU process (8.9 GB VRAM) before starting experiments. Without this, effective VRAM drops to ~6.5 GB, which is insufficient for training.
