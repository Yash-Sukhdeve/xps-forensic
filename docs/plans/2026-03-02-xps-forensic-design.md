# XPS-Forensic: Research Design Document

**Title:** XPS-Forensic: A Calibrated Explainability Pipeline for Partial Audio Spoof Localization with Conformal Coverage Guarantees

**Target:** IEEE Transactions on Information Forensics and Security (IEEE TIFS)

**Format:** 12-14 pages, double column, ~8000-10000 words

**Date:** 2026-03-02

**Status:** Approved for implementation

---

## 1. Core Thesis

Existing partial spoof detection achieves strong in-domain performance (0.59% utterance EER, 3.00% segment EER on PartialSpoof) but produces (1) unexplainable black-box decisions, (2) uncalibrated scores that cannot be interpreted as probabilities, and (3) no formal uncertainty quantification. These gaps make current systems unsuitable for judicial proceedings under Daubert/FRE 702 standards.

We propose **XPS-Forensic**, integrating three complementary post-hoc layers on top of pre-trained frame-level detectors:

- **Post-hoc score calibration** (systematic comparison of Platt/temperature/isotonic scaling)
- **CPSL** (Conformalized Partial Spoof Localization): a two-stage conformal architecture providing distribution-free coverage guarantees at both utterance and segment levels
- **PDSM-PS** (Phoneme-Discretized Saliency Maps for Partial Spoofs): extending phoneme-level saliency attribution to segment-level localization

We evaluate across four partial spoof datasets spanning different threat models and assess both detection performance and explanation faithfulness under domain shift.

---

## 2. Novel Contributions

### C1: CPSL — Two-Stage Conformal Architecture (Core Novelty)

**Stage 1 — Utterance-Level Conformal Classification:**
- Split Conformal Prediction (SCP) with Adaptive Prediction Sets (APS, Romano et al., NeurIPS 2020)
- Nonconformity score: max frame score (primary) + log-sum-exp ablation
- Ternary classification: {real, partially_fake, fully_fake} with ordinal contiguity constraint
- Class-conditional calibration for handling imbalance
- Guarantee: P(Y ∈ C(X)) ≥ 1-α (marginal coverage, finite-sample, distribution-free)

**Stage 2 — Segment-Level Conformal Risk Control:**
- CRC (Angelopoulos et al., ICLR 2024) on temporal false negative rate (tFNR)
- Guarantee: E[tFNR] ≤ α_segment (expected temporal recall bound)
- Dual calibration: tFNR + tFDR for precision control
- tIoU reported as empirical metric (non-monotone, not directly CRC-compatible)

**Composed guarantee:** P(Stage 1 correct AND Stage 2 correct) ≥ 1 − α₁ − α₂ (Bonferroni bound). The product bound (1 − α₁)(1 − α₂) would require independence between stages, which is not assumed here. See Angelopoulos & Bates (2023) for discussion of conformal coverage composition.

**Novelty verification:** Exhaustive search confirms NO prior work applies conformal prediction to audio spoof/deepfake detection at any level. CONCH (Hore & Ramdas, 2026) exists for changepoint localization but targets single changepoints, not dense frame-level binary classification. Ernez et al. (2023) apply CP to wav2vec ASR — different task, must be cited and distinguished.

**Theoretical validity:** Utterance-level exchangeability holds when calibrating on a random partition of PartialSpoof eval (utterances are independent draws). Frame-level exchangeability does NOT hold (temporal autocorrelation via SSL context windows). The design addresses this by applying CP at the utterance level and CRC at the segment level, not frame level.

**Must disclose in paper:**
- Marginal vs conditional coverage distinction (cannot guarantee per-utterance coverage)
- Composed guarantee under joint Stage 1+2: Bonferroni bound 1 − α₁ − α₂; product bound requires independence and is not generally valid in this pipeline
- Cite Barber & Pananjady (2025) on coverage under temporal dependence

### C2: PDSM-PS — Phoneme-Discretized Saliency for Partial Spoofs

- Extends PDSM (Gupta et al., Interspeech 2024) from utterance-level TTS detection to segment-level partial spoof localization
- Applied to CPSL-flagged segments only
- Saliency methods: Integrated Gradients (primary) + GradSHAP (comparison)
- Phoneme boundaries: Montreal Forced Aligner (primary) + WhisperX (neural baseline)
- Faithfulness metrics: Normalized AOPC (Edin et al., ACL 2025) + Comprehensiveness/Sufficiency + Phoneme-IoU against ground truth
- Baselines: raw continuous saliency + fixed-window discretization (50/100ms)

**Novelty verification:** Confirmed novel. PDSM authors have not published follow-up. Liu et al. (Interspeech 2024) used Grad-CAM on PartialSpoof but NOT phoneme-discretized. Survey by He et al. (2025, arXiv:2506.14396) identifies the gap: "existing methods typically answer 'what' but lack a response to 'why'."

**Known risk:** MFA alignment may degrade on synthesized segments (domain mismatch). Mitigated by: (a) quantifying MFA failure rate, (b) WhisperX neural aligner comparison, (c) phoneme alignment confidence filtering.

### C3: Systematic Calibration Comparison

- First systematic comparison of Platt scaling, temperature scaling, and isotonic regression on audio CM scores
- Applied to BAM, SAL, CFPRF, MRM frame-level outputs
- Metrics: ECE, Brier score, reliability diagrams, negative log-likelihood
- Uncalibrated baseline included to demonstrate calibration value
- Utterance-stratified cross-validation for calibration metrics
- Reports calibration drift under cross-dataset evaluation

---

## 3. System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    XPS-Forensic Pipeline                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Input: Audio waveform x ∈ ℝ^T                             │
│                                                             │
│  ┌──────────────────────────────────┐                       │
│  │ Layer 0: Pre-trained Detectors   │                       │
│  │  PRIMARY: BAM (Interspeech 2024) │                       │
│  │  SOTA:    SAL (arXiv 2026)       │                       │
│  │  SECONDARY: CFPRF (ACM MM 2024) │                       │
│  │  BASELINE:  MRM (TASLP 2023)    │                       │
│  │  → Frame-level scores f(x_t)    │                       │
│  └──────────┬───────────────────────┘                       │
│             │                                               │
│  ┌──────────▼───────────────────────┐                       │
│  │ Layer 1: Post-hoc Calibration   │ ← PREREQUISITE        │
│  │  Platt / Temperature / Isotonic  │                       │
│  │  vs. uncalibrated baseline       │                       │
│  │  → Calibrated P(fake|frame)      │                       │
│  └──────────┬───────────────────────┘                       │
│             │                                               │
│  ┌──────────▼───────────────────────┐                       │
│  │ Layer 2: CPSL                   │ ← CORE NOVELTY        │
│  │  Stage 1: SCP + APS (utterance)  │                       │
│  │    → Prediction set over         │                       │
│  │      {real, partial, full}       │                       │
│  │    → Coverage: P(Y∈C) ≥ 1-α     │                       │
│  │                                  │                       │
│  │  Stage 2: CRC on tFNR (segment)  │                       │
│  │    → Localization threshold λ    │                       │
│  │    → E[tFNR] ≤ α_segment        │                       │
│  └──────────┬───────────────────────┘                       │
│             │                                               │
│  ┌──────────▼───────────────────────┐                       │
│  │ Layer 3: PDSM-PS               │ ← INTERPRETABILITY     │
│  │  Applied to CPSL-flagged segs    │                       │
│  │  IG + GradSHAP → phoneme bounds  │                       │
│  │  MFA + WhisperX alignment        │                       │
│  │  Faithfulness: N-AOPC +          │                       │
│  │    Comp/Suff + Phoneme-IoU       │                       │
│  └──────────┬───────────────────────┘                       │
│             │                                               │
│  ┌──────────▼───────────────────────┐                       │
│  │ Layer 4: Evidence Packaging     │                       │
│  │  Structured JSON output          │                       │
│  │  Daubert-aligned metadata        │                       │
│  │  Calibration quality + coverage  │                       │
│  │  PDSM phoneme attributions       │                       │
│  │  Model/version provenance        │                       │
│  └──────────────────────────────────┘                       │
│                                                             │
│  Output: Evidence JSON + explanation artifacts              │
└─────────────────────────────────────────────────────────────┘
```

---

## 4. Detectors

### 4.1 BAM — Primary Detector

- **Paper:** Zhong, Li, Yi. "Enhancing Partially Spoofed Audio Localization with Boundary-aware Attention Mechanism." Interspeech 2024. arXiv:2407.21611
- **Architecture:** WavLM-Large → Boundary Enhancement → Boundary Frame-wise Attention → frame-level predictions
- **Performance:** Seg-EER 3.58% (160ms), multi-resolution 20-640ms
- **Code:** https://github.com/media-sec-lab/BAM
- **Citations:** 17
- **Selection rationale:** Peer-reviewed at top speech venue; multi-resolution evaluation essential for forensic claims at different granularities; boundary-aware attention creates informative ablation with SAL's boundary-debiased approach

### 4.2 SAL — SOTA Comparison

- **Paper:** Mao, Huang, Qian. "Localizing Speech Deepfakes Beyond Transitions via Segment-Aware Learning." arXiv:2601.21925, 2026
- **Architecture:** WavLM-Large/wav2vec2-XLSR → Conformer → Segment Positional Labeling + Cross-Segment Mixing → frame-level predictions
- **Performance:** Seg-EER 3.00% (160ms), Seg-F1 97.09%, cross-dataset 36.60% on LlamaPartialSpoof
- **Code:** https://github.com/SentryMao/SAL (MIT)
- **Selection rationale:** Best segment-level performance; best cross-dataset generalization; CSM addresses boundary-artifact bias; SAL vs BAM comparison reveals whether model detects content or boundaries

### 4.3 CFPRF — Secondary Detector

- **Paper:** Wu et al. "Coarse-to-Fine Proposal Refinement Framework for Audio Temporal Forgery Detection and Localization." ACM MM 2024. arXiv:2407.16554
- **Architecture:** wav2vec2-XLSR → FDN → PRN + DAFL + BAFE → proposals
- **Performance:** Seg-EER 7.41% (20ms), mAP 55.22%, independently reproduced by Luong et al.
- **Code:** https://github.com/ItzJuny/CFPRF (MIT)
- **Selection rationale:** Top-tier venue; proposal-based localization provides complementary evidence; mAP metric directly maps to forensic question ("did the system correctly identify the time range?")

### 4.4 MRM — Baseline

- **Paper:** Zhang et al. "The PartialSpoof Database and Countermeasures." IEEE/ACM TASLP 2023. arXiv:2204.05177
- **Architecture:** SSL + multi-resolution heads (20-640ms)
- **Performance:** Seg-EER 13.72% (160ms), Utt-EER 0.77%
- **Code:** https://github.com/hieuthi/MultiResoModel-Simple (MIT)
- **Selection rationale:** Foundational model by PartialSpoof authors; 98 citations; establishes performance lower bound

---

## 5. Datasets

| Dataset | Size | Spoof type | Labels | Role |
|---------|------|-----------|--------|------|
| **PartialSpoof** | ~41h, English | Concatenation (TTS/VC splice) | Binary, 6 resolutions (20-640ms) | Training + calibration + primary evaluation |
| **PartialEdit** | 1K utts, English | Neural speech editing (VoiceCraft, A3T) | Binary segment-level | Cross-attack (no boundary artifacts) |
| **HQ-MPSD** | 350.8h, 8 langs | Forced-alignment splicing | Ternary (genuine/deepfake/transition), 30ms | Cross-language + artifact-controlled (English subset) |
| **LlamaPartialSpoof** | 130h, English | LLM-driven + voice cloning | Segment-level | Cross-domain (attacker-perspective) |

**Calibration protocol:** Random 80/20 partition of PartialSpoof eval set → 80% for conformal/CRC calibration, 20% for coverage verification. Other datasets: evaluation-only (zero training).

**HQ-MPSD constraint:** Use English-only subset (~3.2 GB compressed, ~25 GB uncompressed) due to storage limitations (full dataset: 1.7 TB).

---

## 6. Experimental Protocol

### E1: Baseline Detection & Localization
- Fine-tune BAM on PartialSpoof (frozen WavLM backbone, ~3h)
- Fine-tune SAL on PartialSpoof (same)
- Use pre-trained CFPRF checkpoint
- MRM: use official baseline
- Report: Utt-EER, Seg-EER at 20/160ms, Seg-F1
- Reproduce published numbers to establish trust

### E2: Post-hoc Calibration Comparison
- Apply Platt/temperature/isotonic to frame-level scores from all 4 detectors
- **Include uncalibrated baseline** to demonstrate calibration adds value
- Calibration set: PartialSpoof dev
- Metrics: ECE, Brier score, reliability diagrams, NLL
- Utterance-stratified cross-validation

### E3: CPSL Coverage & Efficiency
- Stage 1 (utterance): SCP + APS on 80% PartialSpoof eval
  - Coverage at α = {0.01, 0.05, 0.10}
  - Prediction set size (efficiency)
  - Per-class coverage verification
  - Nonconformity score ablation: max vs log-sum-exp(β) sweep
- Stage 2 (segment): CRC on tFNR
  - λ calibration for E[tFNR] ≤ α_segment
  - Empirical tIoU, tFNR, tFDR
- Verify on held-out 20% of PartialSpoof eval
- Statistical test: one-sided binomial for coverage verification

### E4: PDSM-PS Faithfulness
- Apply IG + GradSHAP to WavLM features on CPSL-flagged segments
- Phoneme boundaries: MFA (primary) + WhisperX (neural baseline)
- **Baselines:** raw continuous saliency + fixed-window (50/100ms) discretization
- Faithfulness: Normalized AOPC (N-AOPC)
- Localization alignment: Phoneme-IoU against ground-truth manipulated segments
- Comprehensiveness/Sufficiency (DeYoung et al., 2020)
- Subsample: ~750 utterances for saliency computation
- MFA failure rate quantification on spoofed audio

### E5: Cross-Dataset Generalization
- Run all 4 detectors on PartialEdit, HQ-MPSD (English), LlamaPartialSpoof
- Report per dataset:
  - Detection: Seg-EER, Seg-F1
  - Calibration drift: ECE before/after recalibration
  - CPSL coverage validity: empirical coverage vs nominal
  - PDSM-PS faithfulness stability: N-AOPC under domain shift
- Cite Tibshirani et al. (NeurIPS 2019) for covariate-shift conformal context

### E6: Codec Stress Test
- Re-encode PartialSpoof eval through AAC/Opus/AMR/G.711 (ffmpeg)
- Run inference on re-encoded versions
- Report metric degradation per codec
- Test CPSL coverage under codec distortion

### E7: MFA vs WhisperX Alignment Quality
- Run MFA on PartialSpoof eval set
- Run WhisperX on same data
- Compare phoneme boundaries against VAD-derived segment boundaries
- Quantify alignment error rate on bona fide vs synthesized segments
- Report impact on PDSM-PS faithfulness

### E8: Ablation Studies
- CPSL: ±calibration pre-step; max vs log-sum-exp nonconformity; frame-level vs segment-level conformal
- PDSM-PS: IG vs GradSHAP; phoneme vs word-level aggregation; MFA vs WhisperX
- Detectors: BAM vs SAL saliency comparison (boundary-focused vs boundary-debiased)
- Pipeline: single detector vs ensemble agreement

---

## 7. Metrics

| Category | Metrics | Source |
|----------|---------|--------|
| Detection | Utterance EER, Segment EER (20/160ms), Segment F1, mAP | PartialSpoof/CFPRF standard |
| Calibration | ECE, Brier, reliability diagrams, NLL, uncalibrated baseline | Guo et al. 2017, Dimitri et al. 2025 |
| Conformal (Stage 1) | Empirical coverage, prediction set size, per-class coverage | Angelopoulos & Bates 2022 |
| Conformal (Stage 2) | Empirical tFNR, tFDR, tIoU, CRC threshold λ | Angelopoulos et al. ICLR 2024 |
| Explainability | N-AOPC, Comprehensiveness/Sufficiency, Phoneme-IoU | Edin et al. ACL 2025, DeYoung et al. 2020 |
| Robustness | All above under domain shift + codec distortion | Novel evaluation |

### Statistical Rigor
- Bootstrap 95% CIs on all reported metrics
- Paired Friedman + Nemenyi tests for calibration method comparison
- One-sided binomial test for conformal coverage verification
- Holm-Bonferroni correction for multiple comparisons

---

## 8. Paper Outline

```
I. INTRODUCTION (1.5 pages)
   Problem: partial spoofs in forensic contexts need more than detection
   Gap: no integrated XAI + calibration + conformal pipeline
   Contributions: C1 (CPSL), C2 (PDSM-PS), C3 (calibration comparison)
   Unified thesis: calibration enables CPSL; CPSL tells WHERE; PDSM tells WHY

II. RELATED WORK (2 pages)
   A. Partial spoof detection & localization
   B. Explainability for audio deepfake detection
   C. Calibration and uncertainty in spoof detection
   D. Conformal prediction foundations
   E. Forensic audio standards and legal admissibility

III. PROPOSED METHOD: XPS-FORENSIC (3 pages)
   A. Problem formulation and notation
   B. Post-hoc calibration layer
   C. CPSL Stage 1: Utterance-level conformal classification
      - APS for ternary prediction sets
      - Nonconformity score design (max + log-sum-exp)
      - Class-conditional calibration
   D. CPSL Stage 2: Segment-level CRC on tFNR
      - tFNR monotonicity proof
      - Dual tFNR + tFDR calibration
      - Composed guarantee via Bonferroni (1 − α₁ − α₂); conditions required for product bound
   E. PDSM-PS: Phoneme-discretized saliency for localization
   F. Evidence packaging (JSON schema)

IV. EXPERIMENTAL SETUP (1.5 pages)
   A. Datasets and splits
   B. Detectors and training protocol
   C. Evaluation metrics and statistical tests
   D. Implementation details

V. RESULTS AND ANALYSIS (3 pages)
   A. Baseline reproduction (E1)
   B. Calibration comparison (E2)
   C. CPSL coverage and efficiency (E3)
   D. PDSM-PS faithfulness (E4)
   E. Cross-dataset generalization (E5)
   F. Codec robustness (E6)
   G. MFA vs WhisperX alignment (E7)
   H. Ablations (E8)

VI. DISCUSSION (1 page)
   A. Forensic implications (Daubert factor mapping)
   B. Marginal vs conditional coverage (honest disclosure)
   C. Limitations and failure modes
   D. When the pipeline should NOT be trusted

VII. CONCLUSION (0.5 pages)

APPENDIX
   A. Full JSON evidence schema
   B. Proof of tFNR monotonicity and composed coverage theorem
   C. Additional cross-dataset results
   D. MFA failure analysis
```

---

## 9. Mandatory Citations

These papers MUST be cited and discussed to satisfy reviewers:

### Conformal Prediction Theory
- Vovk, Gammerman, Shafer (2005) — Algorithmic Learning in a Random World
- Angelopoulos & Bates (2022) — Gentle Introduction to CP (FnTML)
- Romano, Sesia, Candes (NeurIPS 2020) — APS for classification
- Angelopoulos et al. (ICLR 2024) — Conformal Risk Control
- Barber et al. (Annals of Statistics, 2023) — CP beyond exchangeability
- Barber & Pananjady (2025) — Split conformal under temporal dependence
- Tibshirani et al. (NeurIPS 2019) — CP under covariate shift
- Ernez et al. (PMLR 2023) — CP for wav2vec ASR (must distinguish)
- Hore & Ramdas (2026) — CONCH changepoint localization (must distinguish)

### Audio Deepfake Detection
- Zhang et al. (TASLP 2023) — PartialSpoof database
- Zhong et al. (Interspeech 2024) — BAM
- Mao et al. (arXiv 2026) — SAL
- Wu et al. (ACM MM 2024) — CFPRF
- Yi et al. (2023) — ADD 2023
- He et al. (2025) — Survey on partial deepfake audio localization

### Explainability
- Gupta et al. (Interspeech 2024) — PDSM
- Liu et al. (Interspeech 2024) — Grad-CAM on PartialSpoof
- Grinberg et al. (ICASSP 2025) — Relevancy-based XAI
- Edin et al. (ACL 2025) — Normalized AOPC
- Ge, Todisco, Evans (2022) — SHAP for spoofing

### Calibration & Uncertainty
- Wang et al. (Interspeech 2024) — Score calibration for SASV
- Kang et al. (ICASSP 2025) — FADEL evidential deep learning
- Pascu et al. (Interspeech 2024) — Calibrated audio deepfake detection

### Forensic Standards
- ENFSI BPM Digital Audio Authenticity (2022)
- SWGDE Best Practices for Digital Audio Authentication
- NIST AI 600-1 (GAI risk profile)
- FRE 702 / Daubert v. Merrell Dow (1993)

### Datasets
- Zhang et al. (Interspeech 2025) — PartialEdit
- Li et al. (2025) — HQ-MPSD
- Luong et al. (ICASSP 2025) — LlamaPartialSpoof
- Luong et al. (APSIPA 2025) — Cross-dataset metrics

---

## 10. Deliverables

| Deliverable | Description |
|-------------|-------------|
| Paper manuscript | IEEE TIFS format, 12-14 pages |
| Code repository | XPS-Forensic pipeline (Python, PyTorch), MIT license |
| Pre-trained checkpoints | Calibrated models on PartialSpoof |
| Conformal calibration artifacts | Stored quantiles for α = {0.01, 0.05, 0.10}, CRC thresholds |
| Evidence JSON schema | Reusable court-oriented output format |
| Reproducibility package | Conda env + single-command evaluation script |

---

## 11. Feasibility

| Resource | Requirement | Available | Status |
|----------|-------------|-----------|--------|
| GPU | RTX 4080 (16GB VRAM) | RTX 4080 (16GB) | Sufficient |
| RAM | 64GB for large dataset processing | 64GB | Sufficient |
| Storage | ~100GB for datasets + models | SSD | Sufficient (HQ-MPSD English-only) |
| Wall time | 2.5-7 days for all experiments | N/A | Feasible |

---

## 12. Risk Register

| Risk | Severity | Probability | Mitigation |
|------|----------|-------------|------------|
| SAL rejected in peer review | Medium | 30% | BAM is primary; SAL is comparison only |
| Cross-dataset CPSL coverage fails | High | 60% | Report honestly as limitation; cite covariate-shift CP theory |
| MFA alignment degrades on spoofed audio | Medium | 70% | WhisperX baseline; quantify failure rate; confidence filtering |
| Competitor publishes CPSL-like paper | Low | 10% | Publish quickly; priority established by arXiv preprint |
| Calibration comparison scooped by ASVspoof 5 | Medium | 25% | Calibration is C3 (minor contribution); CPSL+PDSM are core |
| Reviewers reject "forensic" framing | Low | 15% | IEEE TIFS is a forensics journal; framing is appropriate |

---

## 13. Important Framing Decisions

1. **"Forensically defensible"** NOT "court-admissible" — admissibility is a legal determination, not a measurable metric
2. **"Designed to address Daubert factors"** with explicit enumeration of which factors the pipeline satisfies
3. **Marginal coverage** NOT per-utterance confidence — must be stated clearly
4. **Novel application** of established conformal methods, NOT "novel conformal method"
5. **Honest disclosure** of cross-dataset degradation — this is a feature (honest uncertainty), not a bug

---

## 14. Research Workflow (UWS Phases)

| UWS Phase | Content | Status |
|-----------|---------|--------|
| hypothesis | XAI + calibration + CP pipeline for forensic partial spoof detection | Complete |
| literature_review | Dataset research, XAI research, detector selection, novelty verification | Complete |
| experiment_design | This document | Complete |
| data_collection | Download/prepare PartialSpoof, PartialEdit, HQ-MPSD, LlamaPartialSpoof | Pending |
| analysis | Run experiments E1-E8, statistical analysis | Pending |
| peer_review | Internal review, iterate on manuscript | Pending |
| publication | Submit to IEEE TIFS | Pending |

---

## Appendix A: Supporting Research Artifacts

| File | Contents |
|------|----------|
| `.workflow/dataset-research.md` | Detailed analysis of 5 partial spoof datasets with comparison tables |
| `.workflow/xai-research.md` | 9 XAI methods, 4 calibration papers, 5 UQ papers, 9 watermarking papers |
| `.workflow/detector-selection.md` | 10 candidate detectors evaluated against 4 requirements |
| `.workflow/experiment-review.md` | Experimental rigor review: statistical tests, confounds, missing controls |
| `.workflow/gap-analysis.md` | 12 identified gaps ranked by severity with mitigations |
| `.workflow/feasibility-analysis.md` | Computational feasibility for all experiments on RTX 4080 |
| `.workflow/scoop-analysis.md` | Competitor monitoring: zero scoops found, window open |
| `.workflow/detector-risk-analysis.md` | BAM vs SAL as primary: risk/benefit analysis |
