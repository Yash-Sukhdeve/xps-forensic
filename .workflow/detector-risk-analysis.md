# Detector Risk Analysis: BAM as Primary vs. SAL as Primary for IEEE TIFS Submission

**Date:** 2026-03-02
**Scope:** Risk assessment of promoting BAM (Interspeech 2024) to primary detector and demoting SAL (arXiv 2026) to SOTA comparison
**Target venue:** IEEE Transactions on Information Forensics and Security (IEEE TIFS)
**Decision context:** Gap Analysis (Gap 5) identified SAL's preprint status as HIGH severity, blocking submission

---

## 0. Executive Summary

**Recommendation: BAM as primary detector, SAL as SOTA comparison, with the paper framed to demonstrate detector-agnostic forensic wrapping.**

The analysis below evaluates six specific questions about the BAM-primary / SAL-comparison swap. The conclusion is that BAM as primary is scientifically safer and strategically stronger for IEEE TIFS. The 0.58 percentage-point performance gap (3.58% vs. 3.00% Seg-EER) is not material for the paper's claims, which are about the calibration/conformal/saliency pipeline, not about building a better detector. BAM's boundary-aware attention does create a documented interaction with PDSM-PS, but this interaction is informative rather than problematic -- it generates a publishable ablation that strengthens the paper's forensic argument. The alternative of using CFPRF as primary is viable but suboptimal due to its higher Seg-EER (7.41%) and different operating resolution (20ms vs. 160ms).

---

## 1. How Does the Experimental Narrative Change?

### 1.1 Model-Agnostic Pipeline Claim

**Verdict: The model-agnostic claim becomes stronger, not weaker, with BAM as primary.**

The current detector-selection document (Section 5) describes a pipeline architecture where the detection layer is modular. The paper's core contribution is the forensic wrapping (calibration + conformal prediction + PDSM-PS saliency), not the detection model itself. If both BAM (peer-reviewed, boundary-focused) and SAL (preprint, boundary-debiased) produce usable results through the same pipeline, the model-agnostic claim is empirically demonstrated rather than merely asserted.

**Narrative structure with BAM as primary:**

> "We evaluate our forensic pipeline using BAM (Zhong et al., Interspeech 2024) as the primary frame-level detector, selected for its peer-reviewed status and multi-resolution evaluation capability. To demonstrate the pipeline's detector-agnostic design, we additionally evaluate with SAL (Mao et al., 2026), which achieves current best segment-level performance but has not yet undergone peer review. The fact that the pipeline's calibration, conformal coverage, and saliency properties hold across both architecturally distinct detectors -- one boundary-focused (BAM), one boundary-debiased (SAL) -- validates the forensic wrapping as a general-purpose layer independent of the detection model."

This is a stronger narrative than the current one. The original framing ("SAL is our primary detector") forces the paper to defend SAL. The revised framing ("our pipeline works with any frame-level detector; here are results with two architecturally complementary systems") forces the paper to defend the pipeline -- which is where the actual contribution lies.

**Key requirement for model-agnostic claim:** The paper must show that the calibration quality (ECE), conformal coverage (empirical vs. nominal), and saliency faithfulness (NAOPC) do not vary significantly across detectors. If they do vary, the paper should explain why (e.g., "conformal coverage is tighter for SAL because SAL's scores have lower variance due to CSM regularization") and characterize the detector-dependence quantitatively.

### 1.2 Performance Story: 3.58% vs. 3.00% Seg-EER

**Verdict: The 0.58 percentage-point gap is not material for this paper.**

Evidence:

1. **The paper's contribution is not a detection model.** IEEE TIFS reviewers will evaluate the paper on whether the forensic pipeline (calibration + CPSL + PDSM-PS) is technically sound and novel, not on whether the underlying detector has the absolute lowest Seg-EER. The gap analysis (Gap 7) already established that using pre-trained detectors is defensible because the forensic wrapping is the contribution.

2. **The 0.58pp gap is within bootstrap confidence intervals.** At ~13,000 eval utterances in PartialSpoof, the standard error of a segment EER estimate (assuming ~50 frames per utterance at 160ms for ~10s audio) is approximately 0.1-0.3 percentage points depending on within-utterance correlation. A 0.58pp gap is likely significant but of small practical effect size. More importantly, neither detector has been evaluated with bootstrap CIs in the original papers, so the "true" gap may be larger or smaller.

3. **SAL's superiority is entirely documented in a preprint with 0 citations.** Until SAL's results are independently reproduced or the paper is accepted at a venue, the 3.00% figure has lower evidential weight than BAM's 3.58% (peer-reviewed at Interspeech, 17 citations, independently compared in the SAL paper itself). From a Daubert perspective, the peer-reviewed result has higher credibility than the non-peer-reviewed result, regardless of the numerical value.

4. **Multi-resolution evaluation is more valuable than 0.58pp.** BAM reports Seg-EER at six resolutions (20ms: 5.20%, 40ms: 4.90%, 80ms: 4.32%, 160ms: 3.58%, 320ms: 2.71%, 640ms: 2.28%). This multi-resolution profile is directly useful for forensic applications: a court may require localization at different granularities depending on the evidentiary question ("was this word replaced?" requires ~200-500ms; "was this phoneme spliced?" requires ~40-80ms). SAL reports only at 160ms.

**Summary: The performance story does not materially change. The paper gains more from BAM's multi-resolution profile and peer-reviewed credibility than it loses from the 0.58pp gap.**

---

## 2. Cross-Dataset Evaluation: Weakness or Strength?

### 2.1 The Factual Situation

BAM has no published cross-dataset results. SAL reports:
- PS -> LlamaPartialSpoof (LPS): 36.60% Seg-EER (WavLM)
- SAL is the best-performing system on this transfer task

If BAM is evaluated on LPS, HQ-MPSD, and PartialEdit, three outcomes are possible:

| Scenario | BAM on LPS | Interpretation |
|----------|-----------|----------------|
| A: BAM comparable to SAL | ~36-40% | Cross-dataset difficulty is fundamental, not BAM-specific |
| B: BAM worse than SAL | >42% | BAM's boundary bias causes worse generalization |
| C: BAM much worse than SAL | >50% | BAM heavily overfits PartialSpoof boundary artifacts |

### 2.2 Analysis of Each Scenario

**Scenario A (BAM ~= SAL cross-dataset):** This is the best outcome for the paper. It shows that cross-dataset degradation is dominated by the domain shift and generation method mismatch, not by the detector architecture. The paper can report: "Both peer-reviewed (BAM) and SOTA (SAL) detectors degrade substantially on out-of-domain data, confirming that cross-dataset generalization remains an open challenge for the community. Our forensic pipeline detects and reports this degradation via calibration drift (ECE increase) and conformal undercoverage, providing court-suitable uncertainty communication."

**Scenario B (BAM moderately worse):** This is still acceptable. The paper frames this as: "BAM's boundary-focused attention provides strong in-domain performance but reduces generalization compared to SAL's boundary-debiased design. Our pipeline surfaces this difference through per-dataset calibration analysis and conformal coverage reporting, enabling practitioners to select detectors based on the expected attack type." This actually strengthens Gap 10's recommendation (boundary-artifact confounding experiment) by providing empirical evidence.

**Scenario C (BAM much worse):** This requires careful handling. Two sub-options:

- **(C1) PartialEdit specifically:** BAM will almost certainly perform poorly on PartialEdit (neural speech editing with no splice boundaries), because BAM's architecture explicitly focuses on boundary features. Expected Seg-EER: likely >45-50%. **This is informative, not problematic.** The paper can document: "BAM, which explicitly models splice boundaries, fails on PartialEdit attacks that lack splice boundaries. This demonstrates that boundary-aware detection is fundamentally limited to concatenation-based spoofs. Our forensic pipeline honestly reports this limitation via calibration drift and conformal undercoverage, rather than producing overconfident predictions on unseen attack types."

  From a forensic/Daubert perspective, this is the correct behavior. A court-suitable system must know its own limitations and report them. A system that achieves 3.58% EER on PartialSpoof but produces no false confidence on PartialEdit is more defensible than a system that achieves 3.00% EER on PartialSpoof and implicitly claims generality it does not have.

- **(C2) LlamaPartialSpoof (which does have boundaries):** If BAM degrades substantially more than SAL even on LPS (which uses concatenation like PartialSpoof), this suggests BAM overfits to specific boundary patterns in PartialSpoof rather than learning general boundary detection. This would be a genuine weakness. **Mitigation:** The paper would need to acknowledge that BAM's boundary attention overfits to training-distribution boundary characteristics, and use SAL's better LPS performance as evidence that boundary debiasing (CSM augmentation) improves generalization.

### 2.3 Verdict: Cross-dataset evaluation strengthens the paper regardless of outcome

**The paper's contribution is the forensic pipeline, not the detector.** Every cross-dataset scenario produces a publishable result:

- Good generalization: "Pipeline works with this detector cross-dataset"
- Poor generalization: "Pipeline correctly identifies and reports detector failure"
- Mixed: "Pipeline enables detector selection based on empirical evidence"

The key design principle is that the conformal prediction and calibration layers are designed to be honest about uncertainty. If BAM's scores become poorly calibrated on PartialEdit, the pipeline should produce wider conformal prediction sets (lower precision but maintained coverage) or report coverage violations (if the shift is too severe). **Reporting this behavior is itself a contribution to the forensic AI literature.**

**Specific recommendation:** For PartialEdit evaluation, compute and report:
1. Detector-level: Seg-EER and Seg-F1 (expect poor results for BAM)
2. Calibration-level: ECE before and after calibration (expect calibration drift)
3. Conformal-level: empirical coverage vs. nominal (expect undercoverage)
4. Saliency-level: NAOPC (report only if detector achieves above-chance performance; saliency on random predictions is meaningless)

---

## 3. BAM's Boundary-Aware Attention vs. PDSM-PS

### 3.1 The Interaction

BAM explicitly models splice boundaries via its Boundary Enhancement module and Boundary Frame-wise Attention (BFA). PDSM-PS produces phoneme-discretized saliency maps that attribute detector confidence to specific phonemes. The interaction creates two distinct concerns:

**Concern A: PDSM-PS on BAM will highlight boundary phonemes, not synthetic phonemes.**

If BAM attends primarily to boundary artifacts, then saliency maps (IG/GradSHAP) applied to BAM will show high attribution at frames near splice boundaries. PDSM-PS will aggregate this attribution into the phonemes that happen to straddle the boundary. The resulting "explanation" would be: "phonemes X and Y are suspicious" -- but the actual explanation is "the boundary between phonemes X and Y is suspicious." This is a distinction that matters in court: "this phoneme was synthesized" vs. "a splice was detected between these two phonemes" are different forensic claims.

**Concern B: On PartialEdit (no boundaries), BAM-based PDSM-PS will produce uninformative saliency.**

Since PartialEdit manipulations are generated by neural speech editing without creating splice artifacts, BAM's boundary attention has nothing meaningful to attend to. The resulting saliency maps will either (a) be uniformly low (indicating the model has no basis for classification), or (b) show spurious patterns driven by noise in the gradient computation. Neither outcome produces useful forensic evidence.

### 3.2 Why This Interaction Is Informative, Not Problematic

**The BAM-vs-SAL saliency comparison is itself a publishable experiment** that directly addresses Gap 10 from the gap analysis. The experiment design:

| Condition | Detector | Dataset | Expected Saliency Pattern |
|-----------|----------|---------|--------------------------|
| 1 | BAM | PartialSpoof | High saliency at boundary phonemes |
| 2 | SAL | PartialSpoof | More distributed saliency across manipulated segment (due to CSM debiasing) |
| 3 | BAM | PartialEdit | Uninformative/random saliency |
| 4 | SAL | PartialEdit | Potentially more informative (SAL is less boundary-dependent) |
| 5 | BAM | HQ-MPSD | Reduced saliency at boundaries (forced-alignment splicing minimizes artifacts) |
| 6 | SAL | HQ-MPSD | Similar or better than BAM |

The comparison between Conditions 1 and 2 tests whether PDSM-PS saliency is detector-dependent. The comparison between Conditions 1 and 3 tests whether PDSM-PS can distinguish "the model found something meaningful" from "the model is guessing." The comparison between Conditions 1 and 5 tests whether boundary-focused saliency transfers to high-quality splicing.

**This experiment generates a table that no prior paper has produced.** It directly informs the forensic question: "When I look at this saliency map, can I trust it?" The answer depends on both the detector and the attack type, and quantifying this dependence is a novel contribution.

### 3.3 Recommendation

Include the BAM-vs-SAL saliency comparison as an ablation study. Frame it as: "We analyze how detector architecture influences explanation quality by comparing PDSM-PS saliency between a boundary-focused detector (BAM) and a boundary-debiased detector (SAL) across datasets with different manipulation characteristics."

**For the primary results in the paper, report PDSM-PS faithfulness (NAOPC) separately for BAM and SAL.** If the results differ significantly, this is a finding, not a flaw. The forensic implication is that practitioners should consider the interaction between detector architecture and explanation method when selecting a pipeline configuration.

---

## 4. Should Both BAM and SAL Be Co-Primary Detectors?

### 4.1 Pros and Cons

| Factor | Co-Primary | Single Primary (BAM) + SOTA Comparison (SAL) |
|--------|-----------|----------------------------------------------|
| **Experiment matrix size** | 2x (doubled for every pipeline experiment) | 1x primary + 1x comparison (selective) |
| **Narrative clarity** | Weaker -- "which detector should readers use?" | Stronger -- "here is the validated system; here is how it compares to SOTA" |
| **Model-agnostic claim** | Stronger empirically (two equal evaluations) | Adequate if SAL comparison shows pipeline transfers |
| **Page budget** | Requires ~2-3 extra pages of tables/figures | Fits within IEEE TIFS 14-page limit |
| **Reviewer cognitive load** | Higher -- must track two parallel result streams | Lower -- clear hierarchy |
| **RTX 4080 compute time** | ~100-190 hours (from experiment-review.md estimate, doubled) | ~50-94 hours |

### 4.2 Recommendation: Single Primary (BAM) + Comparison (SAL)

The co-primary approach is not recommended for three reasons:

1. **Gap 3 from the gap analysis already identifies "three contributions in one paper" as a structural risk.** Adding a co-primary detector exacerbates this by creating a fourth axis of variation (BAM results vs. SAL results for every pipeline component). The paper becomes a combinatorial matrix rather than a focused scientific argument.

2. **The compute time is non-trivial.** Saliency computation (the bottleneck per the experiment review, Section 8) takes 30-50 hours for four detectors on one dataset. Doubling the primary detector experiments pushes the timeline from 2-4 days to 4-8 days on a single RTX 4080. With cross-dataset evaluation (3 additional datasets), codec stress tests, and ablations, the total could approach 2-3 weeks of compute.

3. **IEEE TIFS prefers depth over breadth.** Showing complete, rigorous results for one primary detector (BAM: all calibration methods, all alpha levels, all datasets, all saliency methods, all ablations) plus selective comparison on key metrics for SAL is more convincing than showing partial results for two co-primary detectors.

### 4.3 What to Show for SAL as SOTA Comparison

| Experiment | BAM (Primary) | SAL (Comparison) | CFPRF (Secondary) | MRM (Baseline) |
|------------|--------------|------------------|-------------------|----------------|
| Baseline reproduction (PS) | Full | Full | Full | Full |
| Calibration comparison | Full (3 methods) | Best method only | Best method only | Best method only |
| CPSL conformal (3 alpha) | Full | alpha=0.05 only | alpha=0.05 only | alpha=0.05 only |
| PDSM-PS saliency | Full | Full (for detector comparison ablation) | Selected | -- |
| Cross-dataset (3 datasets) | Full | Full | Selected | Selected |
| Codec stress test | Full | Selected | -- | -- |
| Multi-resolution eval | Full (6 resolutions) | 160ms only | 20ms only | Full (6 resolutions) |

This selective matrix keeps the total experiment count manageable (~60-70 conditions) while providing BAM-vs-SAL comparisons on every dimension that matters for the narrative.

---

## 5. What Would an IEEE TIFS Reviewer Prefer?

### 5.1 Reviewer Profile Analysis

IEEE TIFS attracts reviewers from three sub-communities:

1. **Signal processing / audio forensics researchers** (~40% of likely reviewers): These reviewers know the ASVspoof/PartialSpoof literature and will check benchmark numbers. They value validated, peer-reviewed baselines and will flag preprint-only dependencies. BAM at Interspeech 2024 with 17 citations is in their comfort zone. SAL with 0 citations is not.

2. **ML/deep learning researchers** (~35%): These reviewers care about methodological rigor and SOTA performance. They may be more tolerant of preprints (arXiv culture in ML is more established) but will demand that the paper's novelty is in the pipeline, not in citing the newest preprint. They will appreciate the model-agnostic framing.

3. **Security/forensics researchers** (~25%): These reviewers focus on practical threat models, deployment considerations, and evidence integrity. They will value the forensic pipeline framing and the Daubert-factor compliance discussion. They may not know the specific detectors but will check that the chosen system is well-established.

### 5.2 Preference Ranking

| Option | SP/Audio Reviewers | ML Reviewers | Security/Forensic Reviewers | Overall |
|--------|-------------------|-------------|----------------------------|---------|
| **(a) Peer-reviewed only (BAM)** | Preferred | Acceptable | Preferred | **Best** |
| **(b) SOTA only (SAL preprint)** | Reject risk | Acceptable | Concerns | **Worst** |
| **(c) Both, with clear hierarchy** | Acceptable | Preferred | Acceptable | **Second best** |
| **(d) CFPRF instead of either** | Acceptable | May question Seg-EER gap | Preferred (ACM MM) | **Third** |

**Analysis:**

- **Option (a): BAM-primary, SAL-comparison.** Signal processing reviewers are satisfied by the peer-reviewed primary. ML reviewers see SAL's better numbers as a comparison point and recognize the model-agnostic contribution. Security reviewers see a validated system. This is the safest choice.

- **Option (c): Both as co-primary.** ML reviewers appreciate completeness but SP reviewers may question the diluted focus. The risk is moderate because the paper is already structurally complex (three contributions per Gap 3).

- **Option (b): SAL-only.** Reviewer 2 from the SP sub-community writes: "The primary detector (SAL, arXiv:2601.21925) has not been peer-reviewed and has 0 citations as of the submission date. The authors' claim of 'forensically defensible' pipeline credibility is undermined by reliance on an unvalidated detector component. Major revision required: use a peer-reviewed detector as primary." This reviewer comment alone forces a revision cycle.

### 5.3 Verdict: Option (a) is optimal for a first submission

Use BAM as primary. Include SAL as SOTA comparison. If SAL is published at a peer-reviewed venue by the time of camera-ready or resubmission, elevate it to co-primary in the revision. This strategy is robust to SAL's publication timeline while avoiding reviewer objections at initial submission.

---

## 6. CFPRF as Alternative Primary?

### 6.1 CFPRF Profile

| Attribute | Value | Comparison to BAM |
|-----------|-------|-------------------|
| Venue | ACM Multimedia 2024 (h5-index 105) | Stronger venue |
| Citations | 14 | Comparable (BAM: 17) |
| Seg-EER (160ms, PS) | 7.41% | Worse (BAM: 3.58%) |
| Seg-F1 (160ms, PS) | 93.89% | Worse (BAM: 96.09%) |
| Operating resolution | 20ms (frame-level) + proposals | Different from BAM (160ms) |
| Cross-dataset (PS->LPS) | 43.25% EER (Luong et al. repro) | No BAM comparison available |
| Architecture | Coarse proposal -> refined boundaries | Two-stage, more interpretable pipeline |
| mAP (temporal localization) | 55.22% (PS), 99.23% (HAD) | Unique metric -- BAM does not report mAP |
| Backbone | wav2vec2-XLSR (300M) | Different from BAM (WavLM-Large, 316M) |

### 6.2 Arguments For CFPRF as Primary

1. **Strongest venue.** ACM MM (h5-index 105) outranks Interspeech (h5-index 67). For IEEE TIFS reviewers evaluating the credibility chain, ACM MM carries more weight in the multimedia/AI community.

2. **Two-stage architecture maps naturally to forensic reasoning.** CFPRF's FDN (coarse detection) -> PRN (boundary refinement) mirrors how a forensic examiner thinks: first identify suspicious regions, then precisely locate boundaries. This architectural transparency is valuable for the forensic pipeline narrative.

3. **mAP metric provides interval-level evaluation.** CFPRF reports temporal localization mAP (mean Average Precision for detected intervals overlapping ground truth). This is the metric most directly aligned with the forensic question "did the system correctly identify the manipulated time range?" BAM does not report this metric.

4. **Independent reproduction exists.** Luong et al. (ICASSP 2025) independently reproduced CFPRF and evaluated it cross-dataset, providing third-party validation that BAM lacks.

### 6.3 Arguments Against CFPRF as Primary

1. **The Seg-EER gap is material.** CFPRF (7.41%) vs. BAM (3.58%) is a 3.83 percentage-point gap -- over 2x the relative error. Reviewers will question why the "primary" detector is not the best available. Unlike the BAM-vs-SAL gap (0.58pp), this gap is large enough that calibration and conformal prediction will need to compensate for substantially worse raw scores.

2. **Resolution mismatch complicates CPSL.** CFPRF operates at 20ms frame resolution and produces proposals (variable-length temporal regions). BAM and SAL operate at 160ms. The conformal prediction (CPSL) design in the experiment review (Section 1) assumes frame-level scores at a consistent resolution. Using CFPRF as primary requires either (a) applying CPSL at the proposal level (which changes the exchangeability unit), or (b) aggregating CFPRF's 20ms predictions to 160ms for consistency with other detectors (which discards CFPRF's advantage).

3. **Backbone confound with SAL comparison.** CFPRF uses wav2vec2-XLSR; BAM and SAL use WavLM-Large. Performance differences between CFPRF and SAL may be attributable to the backbone (WavLM-Large is stronger) rather than the detection architecture. This confound was identified in the experiment review (Section 4.1) and complicates interpretation. If BAM is primary and SAL is comparison, both use WavLM-Large -- the backbone is controlled.

4. **Cross-dataset results are worse than SAL.** CFPRF: 43.25% EER on LPS (Luong reproduction) vs. SAL: 36.60%. CFPRF is not the best available on the generalization axis that the forensic pipeline claims to address.

### 6.4 Verdict: CFPRF is better as secondary detector, not primary

**CFPRF should retain its secondary detector role** as specified in the original detector-selection document. Its strengths (strongest venue, two-stage interpretability, mAP metric, independent reproduction) complement BAM's strengths (best peer-reviewed Seg-EER, multi-resolution evaluation, same backbone as SAL). Using CFPRF as primary would sacrifice 3.83 percentage points of in-domain performance and introduce a backbone confound with the SAL comparison.

However, CFPRF's proposal-based output format provides a distinct type of forensic evidence (temporal proposals with boundaries and confidence) that BAM's frame-level output does not. The pipeline should use CFPRF's proposals as a secondary evidence stream alongside BAM's frame-level predictions, as originally designed.

---

## 7. Consolidated Recommendation

### 7.1 Detector Hierarchy

| Role | Detector | Justification |
|------|----------|---------------|
| **Primary** | **BAM** (Zhong et al., Interspeech 2024) | Peer-reviewed at top speech venue (17 citations); best peer-reviewed Seg-EER (3.58%); multi-resolution evaluation (20-640ms); WavLM-Large backbone (shared with SAL for controlled comparison); public code |
| **SOTA Comparison** | **SAL** (Mao et al., arXiv 2026) | Best segment EER (3.00%); best cross-dataset generalization (36.60% on LPS); addresses boundary bias via CSM; cited as preprint with explicit status disclosure; demonstrates pipeline detector-agnosticism |
| **Secondary** | **CFPRF** (Wu et al., ACM MM 2024) | Strongest venue (h5-index 105); proposal-based architecture provides complementary evidence type; mAP metric; independent reproduction by Luong et al. |
| **Baseline** | **MRM** (Zhang et al., TASLP 2023) | Foundational model (98 citations); official PartialSpoof baseline; multi-resolution reference; establishes performance floor |

### 7.2 How to Present This in the Paper

**Introduction/Methodology:**

> "We select BAM (Zhong et al., 2024) as our primary detector for the forensic pipeline evaluation, based on three criteria: (1) peer-reviewed publication at a top speech venue (Interspeech 2024), (2) multi-resolution segment-level evaluation spanning 20ms to 640ms, and (3) public code with reproducible results. To demonstrate the pipeline's detector-agnostic design, we additionally evaluate SAL (Mao et al., 2026), which achieves the current lowest reported segment EER on PartialSpoof (3.00% vs. BAM's 3.58%) but has not yet undergone peer review. CFPRF (Wu et al., ACM MM 2024) provides a complementary proposal-based localization architecture as a secondary detector, and MRM (Zhang et al., IEEE/ACM TASLP 2023) serves as the community-standard baseline."

**Results Section Structure:**

1. Primary results tables: BAM through the full pipeline
2. Detector comparison table: BAM vs. SAL vs. CFPRF vs. MRM on key metrics
3. Detector-agnostic analysis: does calibration ECE, conformal coverage, or NAOPC vary significantly across detectors? (Statistical test: Friedman test with post-hoc Nemenyi)
4. Saliency interaction analysis: BAM (boundary-focused) vs. SAL (boundary-debiased) PDSM-PS patterns

### 7.3 Risk Matrix After the Swap

| Risk | Before (SAL Primary) | After (BAM Primary) | Change |
|------|---------------------|---------------------|--------|
| Reviewer rejects due to preprint primary | HIGH | ELIMINATED | Major improvement |
| In-domain Seg-EER weaker than SOTA | N/A (was SOTA) | LOW (0.58pp gap, within noise) | Acceptable cost |
| No cross-dataset results | LOW (SAL has LPS results) | MEDIUM (BAM has none; must evaluate) | Mitigated by running experiments |
| Boundary-artifact overfitting | LOW (SAL debiased via CSM) | MEDIUM (BAM is boundary-focused) | Mitigated by SAL comparison and PartialEdit experiment |
| Model-agnostic claim | MODERATE (one primary only) | LOW (BAM + SAL comparison demonstrates agnosticism) | Improvement |
| PDSM-PS saliency confound | MODERATE (all detectors may have it) | INFORMATIVE (BAM-vs-SAL ablation) | Scientific contribution |
| Pipeline credibility at TIFS | MODERATE (preprint undermines forensic claim) | HIGH (peer-reviewed pipeline) | Major improvement |

### 7.4 New Experiments Required by the Swap

| Experiment | Estimated Time (RTX 4080) | Priority |
|------------|---------------------------|----------|
| BAM baseline reproduction on PartialSpoof eval | 1-2 hours | CRITICAL (before anything else) |
| BAM cross-dataset: LlamaPartialSpoof | 2-4 hours | HIGH |
| BAM cross-dataset: PartialEdit | 2-4 hours | HIGH |
| BAM cross-dataset: HQ-MPSD | 4-8 hours (large dataset) | MEDIUM |
| BAM calibration (3 methods) | 1-2 hours | HIGH |
| BAM conformal prediction (3 alpha) | 1-2 hours | HIGH |
| BAM PDSM-PS saliency (IG, 50 steps) | 8-12 hours | HIGH |
| BAM vs. SAL saliency comparison (PS + PartialEdit) | 4-6 hours (incremental to SAL saliency) | MEDIUM |
| BAM multi-resolution pipeline (6 resolutions) | 4-6 hours | MEDIUM |
| **Total new experiments** | **~27-46 hours** | |

These experiments are incremental to the existing plan. The total experiment timeline increases from 50-94 hours (experiment-review.md estimate) to approximately 77-140 hours (1-2 weeks on a single RTX 4080). This is feasible.

### 7.5 Contingency Plan

**If BAM performs unexpectedly poorly on cross-dataset evaluation (Scenario C2: much worse than SAL on LPS):**

- Report honestly. Frame as: "BAM's boundary-focused architecture shows strong in-domain performance but reduced generalization to out-of-domain concatenation attacks. SAL's Cross-Segment Mixing augmentation demonstrably improves generalization. The forensic pipeline correctly identifies this limitation via calibration drift and conformal undercoverage."
- Consider adding a sentence in the conclusion: "For deployments targeting unknown manipulation types, practitioners should prefer boundary-debiased detectors (e.g., SAL) once independently validated."

**If BAM's code does not reproduce the reported 3.58% Seg-EER:**

- The pretrained checkpoint is available via Google Drive (per detector-selection.md).
- If reproduction yields significantly different numbers, report the reproduced numbers and cite the original paper's numbers in parentheses. This is standard practice and adds transparency.
- Fallback to CFPRF as primary if reproduction fails entirely.

**If SAL is accepted at a peer-reviewed venue before camera-ready:**

- Elevate SAL to co-primary in the revision. Update the narrative from "BAM-primary, SAL-comparison" to "BAM and SAL as co-primary, demonstrating pipeline generality across architecturally distinct detectors."

---

## 8. Summary Decision

**Promote BAM to primary detector. Demote SAL to SOTA comparison.**

The scientific justification rests on three pillars:

1. **Forensic credibility.** The paper targets IEEE TIFS and claims forensic defensibility. A peer-reviewed primary detector (BAM, Interspeech 2024, 17 citations) is more defensible under Daubert-factor scrutiny than a zero-citation preprint (SAL). The paper's own framing -- "court-suitable explainability" -- demands that every pipeline component meet minimum scientific validation standards.

2. **Narrative coherence.** The paper's contribution is the forensic pipeline (calibration + conformal prediction + saliency), not the detector. Using BAM as primary and SAL as comparison demonstrates detector-agnosticism, which is a stronger contribution than achieving the absolute lowest Seg-EER on one benchmark.

3. **Risk management.** The 0.58pp Seg-EER cost is negligible. The reduction in reviewer-rejection risk is substantial. The new BAM-vs-SAL saliency comparison generates a novel ablation study that strengthens the paper. There is no plausible scenario where BAM-primary produces a weaker paper than SAL-primary for IEEE TIFS submission.

---

## References

1. Zhong, J., Li, B., & Yi, J. (2024). Enhancing Partially Spoofed Audio Localization with Boundary-aware Attention Mechanism. Interspeech 2024. arXiv:2407.21611. Code: https://github.com/media-sec-lab/BAM

2. Mao, Y., Huang, W., & Qian, Y. (2026). Localizing Speech Deepfakes Beyond Transitions via Segment-Aware Learning. arXiv:2601.21925. Code: https://github.com/SentryMao/SAL

3. Wu, J., Lu, W., Luo, X., Yang, R., Wang, Q., & Cao, X. (2024). Coarse-to-Fine Proposal Refinement Framework for Audio Temporal Forgery Detection and Localization. ACM Multimedia 2024. arXiv:2407.16554. Code: https://github.com/ItzJuny/CFPRF

4. Zhang, L., Wang, X., Cooper, E., Evans, N., & Yamagishi, J. (2023). The PartialSpoof Database and Countermeasures for the Detection of Short Fake Speech Segments Embedded in an Utterance. IEEE/ACM TASLP, 31, 813-825. arXiv:2204.05177.

5. Luong, H.-T., Rimon, I., Permuter, H., Lee, K.A., & Chng, E.S. (2025). Robust Localization of Partially Fake Speech: Metrics and Out-of-Domain Evaluation. ICASSP 2025. arXiv:2507.03468.

6. Gupta, S., Ravanelli, M., Germain, P., & Subakan, C. (2024). Phoneme Discretized Saliency Maps for Explainable Detection of AI-Generated Voice. Interspeech 2024.

7. Grinberg, P., et al. (2025). What Does an Audio Deepfake Detector Focus on? A Study in the Time Domain. ICASSP 2025. arXiv:2501.13887.

8. Angelopoulos, A.N. & Bates, S. (2023). A gentle introduction to conformal prediction and distribution-free uncertainty quantification. arXiv:2107.07511.

9. Zhang, Y., Tian, B., Zhang, L., & Duan, Z. (2025). PartialEdit: Identifying Partial Deepfakes in the Era of Neural Speech Editing. Interspeech 2025. arXiv:2506.02958.

10. Luong, H.-T., Li, X., Zhang, L., Lee, K.A., & Chng, E.S. (2025). LlamaPartialSpoof: An LLM-Driven Fake Speech Dataset Simulating Disinformation Generation. ICASSP 2025.

---

*This analysis was prepared as part of the Phase 1 planning process. The decision is to be implemented in the experimental protocol before proceeding to Phase 2 (implementation).*
