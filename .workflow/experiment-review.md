# Experimental Design Review: Explainable Partial Spoof Detection Pipeline

**Review Date:** 2026-03-02
**Reviewer role:** Experimental design specialist
**Target:** Journal paper on integrated XAI + calibration + conformal prediction for partial audio spoof localization

---

## 0. Summary Verdict

The proposed experimental design is ambitious and addresses a genuine research gap (no published work combines XAI + calibration + conformal prediction for audio spoof localization as of March 2026). The overall structure is sound. However, the design contains **seven methodological issues** ranging from moderate to serious that must be addressed before submission. The most critical are: (1) the conformal calibration/verification split procedure, (2) the lack of a non-XAI faithfulness baseline, (3) confounded detector comparisons, and (4) insufficient statistical testing beyond point estimates.

---

## 1. Conformal Calibration Split: Is 70/30 Statistically Sound?

### Assessment: MODERATE CONCERN -- addressable but requires justification and sensitivity analysis

**The 70/30 split of the PartialSpoof eval set is defensible but requires explicit defense.**

**Context:** The PartialSpoof eval set contains approximately 13,000 utterances (based on the ASVspoof 2019 LA eval partition from which PartialSpoof is derived). A 70/30 split yields roughly 9,100 utterances for conformal calibration and 3,900 for coverage verification.

**Is 30% enough for reliable coverage verification?**

The answer depends on the desired precision of the coverage estimate. For conformal prediction with target miscoverage alpha, the coverage rate on the verification set follows a Binomial(n, 1-alpha) distribution. For n=3,900 and alpha=0.05:

- Expected coverage: 95%
- Standard error of coverage estimate: sqrt(0.05 * 0.95 / 3900) = 0.35%
- 95% CI for observed coverage: [94.3%, 95.7%]

This is tight enough for a journal paper. **However, the concern is at the segment level, not utterance level.** If coverage is evaluated per-frame (at 160ms resolution), a single utterance may contribute 20-50 frames, inflating the apparent sample size while introducing within-utterance correlation. If coverage is evaluated per-utterance (does the conformal set contain the true label for every frame?), n=3,900 is adequate.

**Recommendations:**

1. **Report the effective sample size** -- if evaluating frame-level coverage, account for within-utterance correlation using a cluster bootstrap (utterances as clusters). Simply reporting n = (number of frames) overstates precision.
2. **Conduct a split sensitivity analysis** -- repeat the conformal procedure with 60/40 and 80/20 splits and report whether the coverage guarantee holds. If it does, the 70/30 choice is robust. If coverage degrades at 60/40, the calibration set may be too small; if it holds at 80/20, reviewer concerns about the small verification set are preempted.
3. **Use random repeated splits (K-fold conformal)** -- instead of a single 70/30 split, use 5 random splits and report coverage mean and standard deviation. This is recommended by Vovk et al. (2005) and Angelopoulos & Bates (2023) for finite-sample conformal evaluation. Barber et al. (2021, "Predictive Inference with the Jackknife+") and Romano et al. (2019, "Conformalized Quantile Regression") both emphasize that coverage guarantees are marginal guarantees; a single split may show coverage fluctuation that repeated splits can average out.
4. **Explicitly state the exchangeability assumption** -- conformal prediction requires that calibration and test data are exchangeable. If the PartialSpoof eval set was not constructed with i.i.d. sampling (e.g., if certain attack types or speakers are clustered), this assumption may be violated. Document what is known about the eval set construction.

**Data leakage concern (related):** The PartialSpoof eval set was designed as a test set for the PartialSpoof challenge. Using 70% of it for conformal calibration means those utterances are no longer "unseen." This is acceptable if the conformal calibration step does not retrain any model parameters (it should not -- conformal prediction is a post-hoc wrapper). However, if any hyperparameters of the conformal procedure (e.g., choice of nonconformity score function) are tuned on the 70% calibration set and then coverage is verified on the 30%, there is an implicit model selection bias. **Recommendation:** Fix the nonconformity score function a priori (pre-register it) or use a three-way split: 50% calibration, 20% nonconformity score selection, 30% verification.

---

## 2. Missing Baselines and Controls

### Assessment: SERIOUS CONCERN -- multiple missing baselines

**2.1 Missing: Raw (uncalibrated, no-XAI) baseline**

The paper must include results where the detector's raw sigmoid/softmax scores are used directly, without any calibration or conformal prediction. This serves as the "null treatment" against which all improvements are measured. Without it, the reader cannot tell whether calibration actually improved anything or whether the raw scores were already well-calibrated.

**Specific requirement:** Report ECE, Brier score, and reliability diagrams for raw detector outputs before any calibration. This is standard in the calibration literature (Guo et al., 2017, "On Calibration of Modern Neural Networks," ICML; Niculescu-Mizil & Caruana, 2005).

**2.2 Missing: Non-phoneme saliency baseline for PDSM-PS**

PDSM discretizes saliency by phoneme boundaries. To demonstrate that phoneme discretization adds value, you need a baseline that uses:
- (a) Raw continuous saliency (Integrated Gradients or GradSHAP applied directly to the spectrogram/SSL features, without phoneme aggregation)
- (b) Fixed-window discretization (e.g., aggregate saliency in 50ms or 100ms windows, matching the approximate phoneme duration, but without linguistic alignment)

Without (a), you cannot show that discretization helps. Without (b), you cannot show that *phoneme-aligned* discretization is better than *arbitrary temporal* discretization. The original PDSM paper (Gupta et al., 2024, Interspeech) does compare against continuous saliency but does not compare against fixed-window discretization.

**2.3 Missing: Conformal prediction alternative -- evidential deep learning**

FADEL (Kang et al., 2025, ICASSP) applies Dirichlet-based evidential deep learning to audio spoof detection and provides uncertainty estimates. Since the paper claims conformal prediction fills a research gap, comparing against FADEL's uncertainty quantification would strengthen the argument. At minimum, discuss why conformal prediction was chosen over evidential deep learning (distribution-free coverage guarantee vs. model-dependent uncertainty).

**2.4 Missing: Ensemble/agreement baseline**

The detector selection document describes a three-detector ensemble (SAL + CFPRF + MRM). If the paper evaluates each detector independently through the XAI + calibration + conformal pipeline, it should also evaluate the ensemble. If it does not use an ensemble, justify why.

**2.5 Missing: BFC-Net**

From the detector selection comparison table (Section 1.2 of detector-selection.md), BFC-Net achieves Seg-EER 2.73% and Seg-F1 96.69% on PartialSpoof with WavLM backbone -- better than SAL on Seg-EER. If BFC-Net has public code, its omission needs justification. If it lacks public code, state this explicitly in the paper.

---

## 3. Cross-Dataset Evaluation Design Fairness

### Assessment: MODERATE CONCERN -- design is defensible but needs framing

**Training only on PartialSpoof is a reasonable choice but introduces a known confound.**

PartialSpoof uses concatenation-based splicing from ASVspoof 2019 LA TTS/VC systems. The three out-of-domain datasets use fundamentally different manipulation methods:

| Dataset | Manipulation Method | Expected Artifact Type |
|---------|-------------------|----------------------|
| PartialSpoof (training) | Concatenation (TTS/VC splice) | Boundary/transition artifacts |
| PartialEdit | Neural speech editing (VoiceCraft, A3T) | **No boundary artifacts** |
| HQ-MPSD | Forced-alignment-based coherent splicing | **Minimal boundary artifacts** |
| LlamaPartialSpoof | LLM-driven + voice cloning concatenation | Boundary artifacts (but different TTS) |

**The problem:** Cross-dataset degradation conflates two distinct failure modes:
1. Domain shift in recording conditions, speakers, languages (a general robustness issue)
2. Mismatch in manipulation method (concatenation vs. editing -- a fundamental detection paradigm difference)

**Why this matters for the paper's claims:** If the explainability pipeline produces correct explanations on PartialSpoof but incorrect explanations on PartialEdit, is that a failure of the XAI method or a failure of the underlying detector? The paper must disentangle these.

**Recommendations:**

1. **Report detector accuracy AND explanation quality separately for each cross-dataset condition.** If the detector fails (high Seg-EER), explanation faithfulness is moot -- you cannot faithfully explain a wrong prediction. Show a contingency: "When the detector is correct, is the explanation faithful? When the detector is wrong, does the explanation correctly indicate low confidence?"

2. **Include a condition where detectors are fine-tuned on each target dataset (oracle upper bound).** This reveals the ceiling performance for each dataset and shows how much of the cross-dataset degradation is due to fundamental detector limitations vs. distribution shift.

3. **Stratify cross-dataset results by manipulation type.** On HQ-MPSD, separately report results for "genuine," "deepfake," and "transition" frames (exploiting the ternary labels). On PartialEdit, separately report word replacement, insertion, and deletion. This prevents averaging over heterogeneous conditions.

4. **Acknowledge the asymmetry explicitly.** LlamaPartialSpoof uses concatenation (like PartialSpoof) but with different TTS systems and LLM-driven content selection. Performance on LlamaPartialSpoof tests TTS generalization. Performance on PartialEdit tests manipulation-method generalization. Performance on HQ-MPSD tests both language generalization and artifact-quality generalization. These are qualitatively different evaluations and should not be collapsed into a single "cross-dataset" number.

---

## 4. Confounds in Comparing Four Detectors

### Assessment: SERIOUS CONCERN -- systematic confounds threaten internal validity

The four detectors differ along multiple axes simultaneously:

| Detector | SSL Backbone | Architecture | Training Objective | Resolution | Peer Review |
|----------|-------------|-------------|-------------------|------------|-------------|
| SAL | WavLM-Large | Conformer + SPL + CSM | Frame-level (segment positional) | 160ms | arXiv only |
| BAM | WavLM-Large | Boundary Enhancement + BFA | Frame-level (boundary-aware) | 160ms (multi-res available) | Interspeech 2024 |
| CFPRF | wav2vec2-XLSR | FDN + PRN + DAFL + BAFE | Frame-level + proposal refinement | 20ms | ACM MM 2024 |
| MRM | SSL (unspecified in repro) | Multi-resolution heads | Multi-task (utt + seg, multi-res) | 20-640ms | TASLP 2023 |

**The confounds:**

**4.1 Backbone confound (SAL/BAM use WavLM-Large; CFPRF uses wav2vec2-XLSR; MRM uses unspecified SSL)**

WavLM-Large (316M parameters, trained on 94K hours including denoising objectives) is a substantially more powerful feature extractor than wav2vec2-XLSR (300M parameters, trained on 56K hours, multilingual). Any performance differences between SAL/BAM and CFPRF may be attributable to the backbone rather than the detection architecture. MRM's backbone is unclear in the reproduction.

**Recommendation:** Run SAL and BAM with wav2vec2-XLSR backbones (SAL's codebase supports this per the detector-selection document) AND run CFPRF with WavLM-Large if the architecture permits. Report backbone-controlled comparisons alongside the "best configuration" comparisons.

**4.2 Resolution confound**

CFPRF operates at 20ms frame resolution; SAL and BAM at 160ms; MRM at multiple resolutions. Comparing Seg-EER across different resolutions is known to be misleading (Zhang et al., 2023; Luong et al., 2025). A model evaluated at 160ms resolution will appear to have lower Seg-EER than the same model evaluated at 20ms resolution because coarser resolution smooths out frame-level errors.

**Recommendation:** Evaluate all detectors at a common resolution (160ms is the natural choice since SAL and BAM report at 160ms). For CFPRF and MRM, aggregate their finer-resolution predictions to 160ms before computing metrics. Additionally, report at 20ms for all models that support it, to show the resolution-accuracy tradeoff.

**4.3 Training procedure confound**

Each detector uses different training procedures (different augmentations, loss functions, learning rate schedules, batch sizes). If you use the authors' pretrained checkpoints, you are comparing "checkpoint A trained with procedure A" against "checkpoint B trained with procedure B." Differences may stem from hyperparameter tuning rather than architectural merit.

**Recommendation:** Since retraining all four detectors with identical training procedures is likely infeasible on a single RTX 4080, use the authors' official checkpoints and **state explicitly that the comparison evaluates "released systems" rather than "architectures."** This reframes the paper from "architecture X is better than architecture Y" to "system X performs better than system Y as released," which is the more honest and more useful claim for practitioners.

**4.4 The XAI pipeline may interact differently with different architectures**

Integrated Gradients, GradSHAP, and PDSM may produce qualitatively different results on different architectures. A Conformer-based model (SAL) may distribute gradients differently than a boundary-attention model (BAM) or a proposal-refinement model (CFPRF). This is not a bug -- it is an important empirical question -- but it must be analyzed rather than swept under "detector comparison."

**Recommendation:** For faithfulness evaluation (Normalized AOPC), report per-detector results and test whether there is a significant detector-by-XAI-method interaction (two-way ANOVA or equivalent).

---

## 5. Statistical Tests Required

### Assessment: SERIOUS CONCERN -- the design as described relies entirely on point estimates

Point estimates (Seg-EER, F1, ECE, Brier score, coverage rate, NAOPC) are necessary but **not sufficient** for a rigorous journal paper. The following statistical analyses are required:

**5.1 Confidence intervals for all primary metrics**

- **Bootstrap confidence intervals** (BCa, 10,000 resamples) for Seg-EER, F1, ECE, Brier score, NAOPC, and conformal coverage. Bootstrap at the utterance level (not frame level) to account for within-utterance correlation.
- **Report 95% CIs** in tables alongside point estimates.
- Reference: Efron & Tibshirani (1993), "An Introduction to the Bootstrap," Chapman & Hall.

**5.2 Paired statistical tests for calibration method comparisons**

Platt scaling, temperature scaling, and isotonic regression are applied to the same data from the same detectors. Use **paired tests**:
- McNemar's test or paired bootstrap test for comparing calibrated vs. uncalibrated classification performance.
- Paired DeLong test for comparing AUCs if ROC analysis is included (DeLong et al., 1988).
- For ECE and Brier score comparisons: paired permutation test (10,000 permutations) since these metrics lack standard parametric test assumptions.
- Reference: Demsar (2006), "Statistical Comparisons of Classifiers over Multiple Datasets," JMLR 7:1-30. Use the Friedman test with post-hoc Nemenyi test when comparing 3+ calibration methods across 4 detectors.

**5.3 Coverage hypothesis test for conformal prediction**

The coverage guarantee is the core claim of the conformal prediction experiment. Do not merely report observed coverage -- test it:
- H0: Coverage >= 1-alpha (the conformal guarantee holds)
- H1: Coverage < 1-alpha (the guarantee is violated)
- Use a one-sided binomial test (or the exact Clopper-Pearson interval).
- Report the p-value for each alpha level (0.01, 0.05, 0.10).
- Reference: Angelopoulos & Bates (2023), "Conformal Prediction: A Gentle Introduction," Foundations and Trends in Machine Learning.

**5.4 Prediction set size analysis**

Conformal prediction trades off coverage for prediction set size. Report:
- Mean and median prediction set size (with IQR).
- Prediction set size conditional on correct vs. incorrect detector prediction.
- Prediction set size as a function of the detector's raw score (to show that uncertain predictions produce larger sets, which is the desired behavior for forensic applications).
- A Wilcoxon signed-rank test comparing prediction set sizes across calibration methods.

**5.5 Faithfulness significance testing**

Normalized AOPC values should be compared across XAI methods and detectors with:
- Paired bootstrap test (since NAOPC is computed on the same test utterances for different methods).
- Report effect sizes (Cohen's d) alongside p-values.
- For the phoneme vs. word aggregation ablation: paired t-test or Wilcoxon test on per-utterance NAOPC values.

**5.6 Cross-dataset degradation significance**

For each detector, test whether cross-dataset performance is significantly worse than in-domain performance:
- Unpaired bootstrap test comparing in-domain Seg-EER/F1 to out-of-domain Seg-EER/F1.
- Report the degradation magnitude with CI.

**5.7 Multiple comparisons correction**

With 4 detectors, 3 calibration methods, 3 alpha levels, 4 datasets, and 2+ XAI methods, the total number of comparisons is large. Apply:
- Holm-Bonferroni correction for the family of primary hypotheses.
- Report both corrected and uncorrected p-values.
- Clearly designate which comparisons are confirmatory (pre-specified) and which are exploratory.

---

## 6. Data Leakage Risk Assessment

### Assessment: LOW-TO-MODERATE CONCERN -- but requires explicit protocol

**6.1 Conformal calibration / coverage verification split**

The primary leakage risk is that the conformal calibration set (70%) and coverage verification set (30%) are drawn from the same PartialSpoof eval partition. If the eval partition has structure (e.g., certain speakers appear in both splits, certain attack types are clustered), the calibration set may "learn" distributional properties that inflate coverage on the verification set.

**Mitigation:** Perform the 70/30 split **stratified by speaker and attack type** (the PartialSpoof eval set metadata should contain this information). Verify that no speaker appears in both splits. Report the stratification procedure.

**6.2 Hyperparameter tuning on eval data**

If any hyperparameters of the pipeline (e.g., temperature scaling temperature, Platt scaling coefficients, nonconformity score function, phoneme aggregation window) are tuned on the PartialSpoof eval set, this constitutes use of test data for model selection. The PartialSpoof dev set should be used for all hyperparameter tuning.

**Recommendation:** Use the PartialSpoof dev set for:
- Selecting the best calibration method (Platt vs. temperature vs. isotonic)
- Tuning any conformal prediction hyperparameters
- Selecting the nonconformity score function

Then apply the selected configuration to the eval set (70% for conformal calibration, 30% for verification). This preserves the eval set's integrity.

**6.3 Detector pretrained checkpoints**

All four detectors were trained on the PartialSpoof training set. The eval set is held out from training. Using their released checkpoints on the eval set is standard practice and does not constitute leakage. However, if any detector's authors also used the eval set during development (e.g., for early stopping or architecture search), there may be implicit leakage in the reported numbers. This is a community-wide issue, not specific to this paper, but should be acknowledged.

**6.4 Cross-dataset evaluation**

PartialEdit, HQ-MPSD, and LlamaPartialSpoof are evaluation-only datasets not seen during training. There is no leakage risk for these, assuming no pretrained checkpoint was exposed to them. Verify this for each detector's published training procedure.

---

## 7. Threats to Validity

### 7.1 Internal Validity

| Threat | Severity | Description | Mitigation |
|--------|----------|-------------|------------|
| **Confounded detector comparison** | High | Backbone, resolution, and training procedure differ across detectors (Section 4) | Backbone-controlled ablation; common resolution evaluation; frame comparisons as "released systems" |
| **Calibration set contamination** | Medium | If dev set is used for calibration method selection AND eval set is used for conformal calibration, the overall pipeline has seen no truly held-out data for calibration evaluation | Use dev set for method selection, eval set for conformal calibration/verification only |
| **Phoneme alignment error propagation** | Medium | PDSM depends on forced alignment; errors in phoneme boundaries propagate into saliency aggregation and may create spurious phoneme-level attributions | Report Phoneme Error Rate (PER) of the aligner on the test data; show sensitivity of NAOPC to alignment perturbation |
| **Single random seed** | Low-Medium | If the 70/30 split, any bootstraps, or any stochastic components use a single seed, results may be seed-dependent | Use multiple seeds; report variance across seeds |
| **Maturation threat (software versions)** | Low | Over the course of the project, library versions (PyTorch, torchaudio, transformers) may change, affecting reproducibility | Pin all versions in a requirements.txt; use Docker/Singularity container; report exact versions |

### 7.2 External Validity

| Threat | Severity | Description | Mitigation |
|--------|----------|-------------|------------|
| **English-centric training** | High | PartialSpoof is English-only. HQ-MPSD provides 8 languages but the detectors are not trained on them. Cross-language generalization is known to fail (>80% degradation per HQ-MPSD paper, Li et al., 2025) | Acknowledge as a limitation; do not claim language-general applicability; report per-language results on HQ-MPSD |
| **Concatenation-bias in training data** | High | All detectors are trained on PartialSpoof's concatenation-based spoofs. Neural speech editing (PartialEdit) and high-quality forced-alignment splicing (HQ-MPSD) produce fundamentally different artifacts. Cross-dataset results will reflect this mismatch | Frame as "distribution shift" experiment, not "generalization" claim; acknowledge that the pipeline is validated for concatenation-type spoofs specifically |
| **Codec/channel conditions not tested** | Medium | The design does not include ASVspoof 2021 DF-style codec stress testing. Real-world evidence is often compressed (WhatsApp, Telegram, phone calls). The deep-research-report.md explicitly recommends codec robustness evaluation | Add a codec stress test: take PartialSpoof eval utterances, re-encode through common codecs (AAC-128kbps, Opus-24kbps, AMR-NB-12.2kbps, G.711), and evaluate the full pipeline. This is computationally cheap and forensically essential |
| **Single-GPU computational constraint** | Low-Medium | RTX 4080 (16GB VRAM) may limit batch size for WavLM-Large inference, potentially requiring evaluation in smaller batches. This does not affect results but may extend experiment time | See Section 8 for detailed hardware analysis |
| **No adversarial robustness evaluation** | Medium | The design does not test whether explanations are stable under adversarial perturbation of the input audio. An adversary could add imperceptible noise that changes the explanation without changing the detector's decision, undermining the explanation's forensic value | At minimum, add a noise injection experiment (Gaussian noise at SNR 40dB, 30dB, 20dB) and report whether saliency maps and conformal prediction sets are stable. Full adversarial robustness (e.g., C&W attack adapted to audio) is a separate study |
| **Temporal granularity mismatch with legal needs** | Medium | The finest resolution is 20ms (CFPRF). Courts may need word-level or phrase-level localization (typically 200-500ms). The paper should demonstrate that frame-level results can be aggregated to word/phrase level and that aggregated results are still meaningful | Include word-level aggregation in ablations (already partially planned via phoneme vs. word aggregation). Show that word-level conformal sets have appropriate coverage |

### 7.3 Construct Validity

| Threat | Severity | Description | Mitigation |
|--------|----------|-------------|------------|
| **"Court-admissible" is not empirically testable** | High | The paper cannot empirically validate whether explanations would actually be admitted under FRE 702/Daubert or Frye. "Court-admissible" is a legal determination made by judges, not a measurable metric | Reframe claims: do not claim "court-admissible." Instead claim "designed to address Daubert factors" or "satisfies necessary conditions for admissibility." Enumerate which Daubert factors the pipeline addresses (testability, error rates, peer review, standards, acceptance) and which it does not (judicial discretion, case-specific relevance) |
| **Faithfulness != Explanation quality** | Medium | Normalized AOPC measures perturbation-based faithfulness (does removing high-saliency regions degrade the model's output?). This does not measure whether a human expert can understand the explanation, act on it correctly, or communicate it to a jury | Add a small human evaluation study (even 3-5 forensic audio experts rating explanation quality) or explicitly acknowledge this gap and defer to future work. Grinberg et al. (2025a, ICASSP) warn that "XAI results obtained from a limited set of utterances do not necessarily hold when evaluated on large datasets" |
| **Calibration != Probability** | Medium | Even well-calibrated scores are empirical frequencies, not physical probabilities. A calibrated p=0.9 means "in our validation data, 90% of instances with this score were truly spoofed." This does not mean "there is a 90% probability this specific audio is spoofed" (the latter requires a Bayesian prior) | Discuss the frequentist interpretation explicitly in the paper. For forensic contexts, consider reporting likelihood ratios (LRs) rather than or in addition to calibrated probabilities, following ENFSI guidelines. Wang et al. (2024, Interspeech) specifically argue for log-likelihood ratios in CM score fusion |

---

## 8. Hardware Sufficiency Analysis: RTX 4080

### Assessment: SUFFICIENT with constraints

**RTX 4080 specifications:**
- 16 GB GDDR6X VRAM
- 9,728 CUDA cores
- ~48.7 TFLOPS FP16

**Memory requirements per detector:**

| Component | VRAM Estimate | Notes |
|-----------|--------------|-------|
| WavLM-Large (forward pass) | ~4.0 GB | 316M params, FP16 inference |
| wav2vec2-XLSR (forward pass) | ~3.5 GB | 300M params, FP16 inference |
| SAL detection head | ~0.3 GB | Lightweight Conformer (2 blocks) |
| BAM detection head | ~0.3 GB | Boundary Enhancement + BFA |
| CFPRF (FDN + PRN) | ~1.0 GB | Two-stage network |
| MRM multi-res heads | ~0.5 GB | Multiple resolution heads |
| Saliency computation (IG/GradSHAP) | ~2x forward pass | Requires gradient storage |
| Batch of audio (16 utterances, ~10s each) | ~0.5 GB | Waveform + intermediate features |

**Worst case (WavLM-Large + SAL + saliency):** ~4.0 + 0.3 + 4.0 (grad) + 0.5 = ~8.8 GB. This fits in 16 GB with headroom.

**Bottleneck:** Saliency computation (Integrated Gradients requires multiple forward passes, typically 50-300 interpolation steps). For 3,900 verification utterances * 4 detectors * 50 IG steps = 780,000 forward passes. At ~50ms per forward pass (WavLM-Large on RTX 4080), this is approximately 10.8 hours for saliency alone.

**Total estimated compute time:**

| Experiment | Estimated Time |
|------------|---------------|
| Baseline reproduction (4 detectors, PartialSpoof eval) | 2-4 hours |
| Calibration comparison (3 methods, 4 detectors) | 1-2 hours |
| Conformal prediction (3 alpha levels, 4 detectors) | 1-2 hours |
| PDSM-PS saliency (IG, 50 steps, 4 detectors, ~17K utterances) | 30-50 hours |
| Cross-dataset evaluation (3 extra datasets, 4 detectors) | 8-16 hours |
| Ablations | 10-20 hours |
| **Total** | **~50-94 hours (2-4 days)** |

**Verdict:** The RTX 4080 is sufficient. The main constraint is saliency computation time, which can be reduced by:
- Using GradSHAP (single backward pass) instead of Integrated Gradients (50+ forward passes) as the primary saliency method, with IG as a sensitivity check on a subset.
- Computing saliency only on the 30% verification set (3,900 utterances) rather than the full eval set, since faithfulness metrics only need to be evaluated on the test partition.
- Using FP16 inference throughout.

**RAM (64 GB):** Sufficient for all operations. The largest dataset (HQ-MPSD at 350.8 hours) does not need to be loaded entirely into memory; streaming evaluation is standard.

**Risk:** If saliency computation proves too slow, the paper can report IG results on a stratified subsample (e.g., 1,000 utterances) and GradSHAP on the full set, with a note that the subsample was selected to represent the distribution of scores, speakers, and attack types.

---

## 9. Additional Recommendations

### 9.1 Pre-registration

Given that this paper addresses multiple research gaps (conformal prediction for audio spoof detection, systematic calibration comparison, PDSM applied to partial spoof localization), **pre-register the primary hypotheses and analysis plan on OSF before running experiments.** This is especially important because:
- The design has many degrees of freedom (4 detectors, 3 calibration methods, 3 alpha levels, 4 datasets, 2+ XAI methods).
- Without pre-registration, reviewers will (correctly) worry about cherry-picking the best combinations.
- Pre-registration of the primary hypotheses distinguishes confirmatory from exploratory analyses.

**Primary hypotheses to pre-register:**
1. Post-hoc calibration (best method TBD on dev set) significantly reduces ECE compared to raw scores across all 4 detectors (paired test, alpha=0.05).
2. Conformal prediction achieves coverage >= 1-alpha on the 30% verification set for all three alpha levels (binomial test).
3. PDSM-PS achieves higher Normalized AOPC than continuous saliency (paired bootstrap, alpha=0.05) on at least 3 of 4 detectors.

### 9.2 Effect size reporting

For every comparison, report effect sizes alongside p-values. Cohen's d for paired comparisons, eta-squared for ANOVA-like analyses. This follows APA guidelines and increasingly strict journal requirements.

### 9.3 Reproducibility package

Plan to release:
- All code (training scripts unnecessary if using pretrained checkpoints; evaluation and pipeline code required)
- The exact 70/30 split indices (or the random seed + stratification procedure to regenerate them)
- Conformal calibration parameters (the quantile thresholds learned on the calibration set)
- Calibration model parameters (Platt scaling coefficients, temperature values, isotonic regression mappings)
- Saliency computation configuration (number of IG steps, baseline choice, GradSHAP parameters)
- Docker/Singularity container with pinned dependencies

### 9.4 Reporting standards

Follow **CONSORT-AI** (Liu et al., 2020, Nature Medicine) for the experimental reporting structure where applicable, and the **REFORMS** checklist (Kapoor & Narayanan, 2023, Science Advances, "Reforms needed to improve reproducibility in ML research") for ML-specific reproducibility.

### 9.5 Missing experiment: calibration drift under distribution shift

Calibration is performed on PartialSpoof data. When the calibrated model is applied to HQ-MPSD or PartialEdit (cross-dataset), the calibration may degrade (ECE increases). This is calibration drift, and it is a known problem (Ovadia et al., 2019, "Can You Trust Your Model's Uncertainty? Evaluating Predictive Uncertainty Under Dataset Shift," NeurIPS). **Report ECE on each cross-dataset condition to show whether calibration transfers or degrades.** This is a single additional table and adds substantial scientific value.

### 9.6 Missing experiment: conformal prediction under distribution shift

The conformal coverage guarantee is conditional on exchangeability between calibration and test data. When the test data is from a different dataset (PartialEdit, HQ-MPSD, LlamaPartialSpoof), the exchangeability assumption is violated, and coverage may degrade. **Report the actual coverage on each cross-dataset condition.** Expected finding: coverage will be below the nominal level on out-of-domain data. This is an honest and informative result that characterizes the method's limitations.

Recent work on conformal prediction under distribution shift (Tibshirani et al., 2019, "Conformal Prediction Under Covariate Shift," NeurIPS; Barber et al., 2022) provides theoretical grounding for expected coverage degradation. Cite this literature.

---

## 10. Summary of Required Actions

| Priority | Issue | Section | Action |
|----------|-------|---------|--------|
| **Critical** | No statistical tests specified | 5 | Add bootstrap CIs, paired tests, binomial coverage test, multiple comparison correction |
| **Critical** | Confounded detector comparison | 4 | Backbone-controlled ablation; common resolution; frame as "released systems" comparison |
| **Critical** | Missing raw baseline | 2.1 | Report uncalibrated ECE/Brier/reliability for all detectors |
| **High** | Missing non-phoneme saliency baseline | 2.2 | Add continuous saliency and fixed-window baselines for PDSM-PS |
| **High** | No codec stress test | 7.2 | Re-encode PartialSpoof eval through AAC/Opus/AMR/G.711; evaluate full pipeline |
| **High** | Calibration drift not measured | 9.5 | Report ECE on all cross-dataset conditions |
| **High** | Conformal coverage under shift not measured | 9.6 | Report actual coverage on all cross-dataset conditions |
| **Medium** | Split sensitivity analysis needed | 1 | Repeat conformal with 60/40, 80/20 splits; or use K-fold conformal |
| **Medium** | Three-way split for nonconformity score selection | 1 | Fix score function a priori or use 50/20/30 split |
| **Medium** | Pre-registration recommended | 9.1 | Register primary hypotheses on OSF before data collection |
| **Medium** | Speaker stratification in split | 6.1 | Stratify 70/30 split by speaker and attack type |
| **Medium** | "Court-admissible" overclaim risk | 7.3 | Reframe to "designed to address Daubert factors" |
| **Low** | BFC-Net omission | 2.5 | Justify exclusion or include |
| **Low** | Human evaluation of explanations | 7.3 | Add expert evaluation or acknowledge gap |
| **Low** | Adversarial noise stability | 7.2 | Add noise injection test at 3 SNR levels |

---

## 11. Revised Experiment Matrix (Recommended)

For reference, the complete set of experiments after incorporating the above recommendations:

| # | Experiment | Detectors | Datasets | Metrics | Statistical Tests |
|---|-----------|-----------|----------|---------|-------------------|
| 1 | Baseline reproduction | SAL, BAM, CFPRF, MRM | PartialSpoof eval | Seg-EER, Seg-F1, Utt-EER (all at 160ms) | Bootstrap 95% CI |
| 2a | Raw calibration assessment | All 4 | PartialSpoof eval | ECE, Brier, reliability diagrams (uncalibrated) | -- |
| 2b | Post-hoc calibration comparison | All 4 * 3 methods | PartialSpoof dev (tune) + eval (report) | ECE, Brier, reliability diagrams (calibrated) | Friedman + Nemenyi across methods; paired bootstrap for each detector |
| 3 | CPSL: conformal prediction | All 4 * 3 alpha | PartialSpoof eval (70/30, speaker-stratified) | Coverage, prediction set size | One-sided binomial test per alpha; Wilcoxon for set size comparison; 5-fold split sensitivity |
| 4a | Continuous saliency baseline | All 4 * IG + GradSHAP | PartialSpoof eval (30% verification) | Normalized AOPC | Paired bootstrap per detector |
| 4b | Fixed-window saliency baseline | All 4 * 50ms/100ms windows | Same | Normalized AOPC | Paired bootstrap per detector |
| 4c | PDSM-PS | All 4 * phoneme aggregation | Same | Normalized AOPC, localization alignment | Paired bootstrap vs. 4a and 4b |
| 5a | Cross-dataset: in-domain metrics | All 4 | PartialEdit, HQ-MPSD, LlamaPartialSpoof | Seg-EER, Seg-F1 | Unpaired bootstrap vs. in-domain |
| 5b | Cross-dataset: calibration drift | All 4 * best calibration | Same | ECE (cross-domain) | -- |
| 5c | Cross-dataset: conformal coverage | All 4 * alpha=0.05 | Same | Actual coverage | Binomial test (expect violation) |
| 5d | Cross-dataset: explanation quality | All 4 * PDSM-PS | Same (subset) | Normalized AOPC | Report conditional on correct/incorrect detection |
| 6a | Ablation: calibration variants | All 4 | PartialSpoof eval | ECE, Brier | Friedman test |
| 6b | Ablation: nonconformity scores | All 4 | PartialSpoof eval | Coverage, set size | Paired comparison |
| 6c | Ablation: phoneme vs. word aggregation | All 4 | PartialSpoof eval | NAOPC | Paired Wilcoxon |
| 6d | Ablation: backbone swap (SAL w/ w2v2-XLSR, CFPRF w/ WavLM if possible) | SAL, CFPRF | PartialSpoof eval | Seg-EER, Seg-F1 | Paired bootstrap |
| 7 | Codec stress test | All 4 | PartialSpoof eval re-encoded (4 codecs) | Seg-EER, F1, ECE, coverage | Bootstrap CI; test degradation vs. clean |
| 8 | Noise stability | All 4 | PartialSpoof eval + noise (3 SNR levels) | NAOPC stability, coverage stability | Paired test vs. clean |

**Estimated total experiments:** ~40 conditions (manageable on RTX 4080 in 1-2 weeks).

---

## References Cited in This Review

1. Angelopoulos, A.N. & Bates, S. (2023). Conformal Prediction: A Gentle Introduction. Foundations and Trends in Machine Learning, 16(4), 494-591.
2. Barber, R.F., Candes, E.J., Ramdas, A., & Tibshirani, R.J. (2021). Predictive Inference with the Jackknife+. Annals of Statistics, 49(1), 486-507.
3. DeLong, E.R., DeLong, D.M., & Clarke-Pearson, D.L. (1988). Comparing the Areas under Two or More Correlated Receiver Operating Characteristic Curves: A Nonparametric Approach. Biometrics, 44(3), 837-845.
4. Demsar, J. (2006). Statistical Comparisons of Classifiers over Multiple Datasets. JMLR, 7, 1-30.
5. Efron, B. & Tibshirani, R.J. (1993). An Introduction to the Bootstrap. Chapman & Hall/CRC.
6. Guo, C., Pleiss, G., Sun, Y., & Weinberger, K.Q. (2017). On Calibration of Modern Neural Networks. ICML 2017.
7. Gupta, S., Ravanelli, M., Germain, P., & Subakan, C. (2024). Phoneme Discretized Saliency Maps for Explainable Detection of AI-Generated Voice. Interspeech 2024.
8. Edin, J., et al. (2024). Normalized AOPC: Fixing Misleading Faithfulness Metrics for Feature Attribution Explainability. arXiv:2408.08137 (accepted ACL 2025).
9. Kang, J.Y., et al. (2025). FADEL: Uncertainty-aware Fake Audio Detection with Evidential Deep Learning. ICASSP 2025.
10. Kapoor, S. & Narayanan, A. (2023). Leakage and the Reproducibility Crisis in Machine-Learning-Based Science. Science Advances, 9(34).
11. Liu, X., et al. (2020). Reporting Guidelines for Clinical Trial Reports for Interventions Involving Artificial Intelligence: The CONSORT-AI Extension. Nature Medicine, 26, 1364-1374.
12. Luong, H.-T., et al. (2025). Robust Localization of Partially Fake Speech: Metrics and Out-of-Domain Evaluation. arXiv:2507.03468.
13. Niculescu-Mizil, A. & Caruana, R. (2005). Predicting Good Probabilities with Supervised Learning. ICML 2005.
14. Ovadia, Y., et al. (2019). Can You Trust Your Model's Uncertainty? Evaluating Predictive Uncertainty Under Dataset Shift. NeurIPS 2019.
15. Romano, Y., Patterson, E., & Candes, E. (2019). Conformalized Quantile Regression. NeurIPS 2019.
16. Tibshirani, R.J., Foygel Barber, R., Candes, E., & Ramdas, A. (2019). Conformal Prediction Under Covariate Shift. NeurIPS 2019.
17. Vovk, V., Gammerman, A., & Shafer, G. (2005). Algorithmic Learning in a Random World. Springer.
18. Wang, X., Kinnunen, T., Lee, K.A., Noe, P.-G., & Yamagishi, J. (2024). Revisiting and Improving Scoring Fusion for Spoofing-aware Speaker Verification Using Compositional Data Analysis. Interspeech 2024.
19. Zhang, L., Wang, X., Cooper, E., Evans, N., & Yamagishi, J. (2023). The PartialSpoof Database and Countermeasures. IEEE/ACM TASLP.

---

*This review was prepared for internal use before finalizing the experimental protocol. All critical and high-priority items should be resolved before running experiments.*
