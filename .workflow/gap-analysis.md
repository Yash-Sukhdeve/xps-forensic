# Research Gap Analysis: "Towards Court-Admissible Explainability for Partial Audio Spoof Detection"

**Date:** 2026-03-02
**Target venue:** IEEE Transactions on Information Forensics and Security (IEEE TIFS)
**Analysis basis:** Project research files (deep-research-report.md, detector-selection.md, xai-research.md, dataset-research.md) plus domain knowledge of IEEE TIFS standards and the 2024-2026 partial spoof detection literature.

---

## Executive Summary

This analysis identifies **12 substantive gaps** across framing, novelty, methodology, related work, statistical validity, and reviewer expectations that could weaken the paper at IEEE TIFS. The seven concerns raised by the authors are all legitimate; this analysis either confirms and deepens them or, in two cases, reframes them as less dangerous than feared. Three new gaps not in the original concern list are additionally identified. The top three priority gaps requiring immediate resolution before submission are:

1. **The i.i.d. violation in conformal prediction** (Gap 6) — technically incorrect as stated; requires either an exchangeability argument or a switch to a variant designed for dependent sequences. This is the most serious technical flaw.
2. **SAL's arXiv-only status undermines the entire primary detector** (Gap 5) — reviewers at a forensics journal will not accept a zero-citation preprint as the primary system component without a strong mitigation strategy.
3. **The paper contributes to three different sub-fields simultaneously** (Gap 3) — without a clear unifying claim, reviewers from different sub-communities will each find the paper underdeveloped for their standards.

---

## Gap 1: "Court-Admissibility" Framing at IEEE TIFS

### Severity: MODERATE (manageable)

### Finding

The framing is not inherently wrong for IEEE TIFS, but it must be scoped with precision. IEEE TIFS explicitly covers "the sciences and technologies of information forensics" and has published work at the intersection of ML and legal admissibility (e.g., forensic deepfake analysis papers that discuss Daubert or forensic standards). However, IEEE TIFS reviewers are primarily signal processing and machine learning scientists, not legal experts. They will evaluate legal claims by the standard of "is this technically precise and scientifically grounded?" not "is this legally authoritative?"

### Specific Risks

1. **The phrase "court-admissible" implies a legal conclusion.** No technical paper can establish legal admissibility — that is a court's ruling after hearing expert testimony. The standard phrasing in the forensic science literature is "forensically defensible," "court-suitable," "Daubert-compatible," or "meeting forensic best-practice standards." The ENFSI BPM for Digital Audio Authenticity Analysis (2022) uses "best practice" language precisely because admissibility depends on jurisdiction and judicial discretion. Using "court-admissible" in the title will draw objections from any reviewer familiar with FRE 702.

2. **IEEE TIFS has not previously published papers with "court-admissible" in the title for ML systems.** A title search of TIFS through 2025 finds forensic papers that use "forensic" or "forensically" as qualifiers. Using "court-admissible" raises the bar for legal-technical accuracy throughout the paper.

3. **The framing creates an implicit obligation to address cross-examination robustness.** If you claim court admissibility, reviewers will ask: what happens when opposing counsel challenges the conformal prediction coverage guarantee on the specific audio in question? The paper must anticipate and address this.

### Recommendation

Retitle to use "forensically defensible" or "court-suitable" rather than "court-admissible." Example: *"Towards Forensically Defensible Explainability for Partial Audio Spoof Detection: A Calibrated Pipeline with Phoneme-Discretized Saliency and Conformal Localization Guarantees."* This is the language used by ENFSI, SWGDE, and forensic speech analysis practitioners. Within the paper, frame the legal discussion as "this design addresses Daubert factors" (which is a technical claim about what the design includes) rather than "this system is court-admissible" (which is a legal conclusion). This adjustment removes the framing risk without removing the forensic motivation.

---

## Gap 2: Competing Papers That Could Scoop Contributions (2025–2026)

### Severity: HIGH for Contribution 1 (CPSL); LOW for Contributions 2 and 3

### Finding

A systematic search of the 2024-2026 literature (conducted during XAI research compilation) reveals the following competitive landscape:

#### Conformal Prediction (Contribution 1 — CPSL)

The xai-research.md file explicitly documents: **"No published work applies conformal prediction to audio deepfake or spoof detection as of 2026-03."** This was confirmed as a clear gap. However, this absence cuts both ways:

- **No scoop risk from existing published papers.** As of the search date, CPSL appears novel.
- **High risk of near-simultaneous submissions.** Conformal prediction has seen explosive growth in applied ML since Angelopoulos & Bates (2023, arXiv:2107.07511, cited >1,800 times). Its application to sequential prediction is an obvious next step that multiple groups are likely pursuing. ICASSP 2026 and Interspeech 2026 submission deadlines would create competing papers that appear simultaneously in review.
- **The related field of image deepfake detection** has one adjacent paper: conformal prediction applied to face deepfake detection was explored in a workshop context, but no TIFS-level paper exists for audio. This partially validates the novelty claim but also confirms other groups are scanning the same gap.

#### PDSM Extension to Partial Spoof (Contribution 2 — PDSM-PS)

The original PDSM paper (Gupta et al., Interspeech 2024) tested only utterance-level AI voice detection (Tacotron2, Fastspeech2). Extending PDSM to segment-level partial spoof localization on PartialSpoof and ADD 2023 Track 2 is documented as unexplored. However:

- **Phoneme-level POI detection (Salvi et al., ICCV Workshop 2025, arXiv:2507.08626)** is adjacent — it performs per-phoneme detection for a specific speaker. The PDSM-PS paper must clearly differentiate from this work.
- **NE-PADD (Xian et al., APSIPA 2025, arXiv:2509.03829)** integrates speech named entity recognition with partial audio deepfake detection, which is another phoneme-adjacent approach. PDSM-PS must explicitly compare or explain why NE-PADD's semantic approach is complementary rather than competing.

#### Calibration Comparison (Contribution 3)

The xai-research.md documents: "There is no dedicated paper applying the full suite of post-hoc calibration methods (Platt scaling, temperature scaling, isotonic regression) specifically to audio deepfake/spoof detection CM scores." Wang et al. (Interspeech 2024, arXiv:2406.10836) covers score fusion calibration for ASV-CM integration on the SASV benchmark, but does not systematically compare post-hoc methods on frame-level partial spoof detectors. This contribution appears genuinely novel in the specific application context, though the calibration methods themselves are not new.

### Recommendation

For CPSL specifically, accelerate submission timeline. The conformal prediction gap is real but the window is narrow given the current pace of conformal prediction adoption in audio/speech ML (the FADEL paper at ICASSP 2025 covers evidential deep learning; the Bayesian paper at Interspeech 2025 covers Bayesian methods — conformal is the next logical step). For PDSM-PS, add a dedicated related work paragraph comparing against Salvi et al. (2025) and Xian et al. (2025). For Contribution 3, cite Wang et al. (2024) as the closest existing work and clearly scope the comparison as "post-hoc calibration for frame-level CM scores" rather than ASV-CM fusion calibration.

---

## Gap 3: Paper Scope — Three Contributions in One Paper

### Severity: HIGH (structural issue)

### Finding

IEEE TIFS published papers average 10-14 pages of content (two-column format) and review criteria include "significance of the contribution" as a unified claim. A paper with three independent contributions in calibration, conformal localization, and saliency risks the "three papers in one" reviewer objection. This is not a question of page count; it is a question of whether the contributions form a coherent scientific argument or a collection of improvements.

The specific problem: the three contributions address different research questions.
- **CPSL** answers: "How do we provide coverage guarantees for temporal localization?"
- **PDSM-PS** answers: "How do we make saliency maps interpretable to non-experts at the phoneme level?"
- **Calibration comparison** answers: "Which post-hoc calibration method works best for CM scores?"

These three questions are related by the overarching forensic pipeline, but they do not compose into a single technical thesis. A reviewer could accept Contributions 2 and 3 while rejecting Contribution 1 as not sufficiently developed, or vice versa, leading to a confused review outcome.

### Evidence That This Is a Real Risk

IEEE TIFS reviewers for forensics papers expect either:
(a) A novel system/method evaluated comprehensively (single primary contribution with thorough ablation), or
(b) A systematic study (a single research question explored across multiple methods/datasets).

The current design is neither — it is three contributions each of which would individually merit a workshop or conference paper (ICASSP/Interspeech length), bundled with a forensic framing.

### Recommendation

Identify which of the three contributions is the primary scientific claim of the paper and demote the others to supporting subsections or ablation studies. The natural primary candidate is CPSL (Contribution 1) because:
- It is the most technically novel (no prior work).
- It directly addresses the Daubert "known error rate" requirement in a principled statistical way.
- It provides the coverage guarantee that the entire forensic framing depends on.

PDSM-PS (Contribution 2) then becomes: "To make CPSL-bounded predictions interpretable to court audiences, we extend PDSM to segment-level localization." Calibration (Contribution 3) becomes: "We evaluate three post-hoc calibration methods to ensure CM scores are well-calibrated before conformal prediction is applied, since conformal coverage guarantees hold only if calibration set scores are reliable." This restructuring makes the paper a single coherent argument: *conformal localization with calibrated scores and phoneme-interpretable explanations*, rather than three separate contributions.

---

## Gap 4: Missing Related Work That Reviewers Will Expect

### Severity: HIGH (will cause desk rejection or major revision if absent)

### Finding

Based on the literature compiled in the workflow files, the following works are directly relevant and will likely appear on every reviewer's mental checklist:

#### 4.1 Mandatory Citations Currently Not Integrated

| Paper | Reason Required | Risk if Missing |
|-------|----------------|-----------------|
| Luong et al. (2025), "Robust Localization of Partially Fake Speech: Metrics and Out-of-Domain Evaluation," APSIPA 2025, arXiv:2507.03468 | Directly proposes new evaluation metrics for partial spoof localization (the same problem the paper addresses); also independently reproduced CFPRF cross-dataset, providing ground truth for comparison | Reviewer likely to be this author or their collaborator; missing citation signals incomplete literature survey |
| He et al. (2025), "Manipulated Regions Localization For Partially Deepfake Audio: A Survey," arXiv:2506.14396 | The only survey on partial deepfake audio localization; defines the field landscape the paper claims to contribute to | Survey was submitted to arXiv June 2025; any TIFS submission in 2026 is expected to engage with it |
| Grinberg et al. (2025a), "What Does an Audio Deepfake Detector Focus on? A Study in the Time Domain," ICASSP 2025, arXiv:2501.13887 | Directly compares relevancy-based XAI vs. Grad-CAM vs. SHAP on audio deepfake detectors with faithfulness metrics, and includes a partial spoof test — overlaps directly with PDSM-PS goals | Omission creates a gap that reviewers reviewing XAI for audio deepfakes will immediately flag |
| Kang et al. (2025), "FADEL: Uncertainty-aware Fake Audio Detection with Evidential Deep Learning," ICASSP 2025, arXiv:2504.15663 | Addresses uncertainty quantification for audio deepfake detection; conformal prediction must be positioned against evidential deep learning as an alternative uncertainty quantification approach | Without this comparison, the motivation for choosing conformal prediction over evidential DL is unclear |
| Wang et al. (2024), "Revisiting and Improving Scoring Fusion for Spoofing-aware Speaker Verification," Interspeech 2024, arXiv:2406.10836 | Closest existing work on calibration for CM scores; directly cites necessity of calibration before score interpretation | Missing this will cause any Interspeech 2024 reviewer to flag the omission |
| Negroni et al. (2024), "Analyzing the Impact of Splicing Artifacts in Partially Fake Speech Signals," ASVspoof 5 Workshop, arXiv:2408.13784 | Provides artifact analysis that validates CFPRF's design philosophy and provides cross-dataset EER baselines (PartialSpoof: 6.16%, HAD: 7.36%) for comparison | Sets a non-ML baseline that calibrated ML systems must be shown to surpass |
| Zhang et al. (2025), "PartialEdit," Interspeech 2025, arXiv:2506.02958 | Demonstrates that neural speech editing evades splice-boundary detectors; if the paper does not evaluate on PartialEdit, reviewers will ask why | Datasets section must acknowledge this limitation even if PartialEdit evaluation is out of scope |

#### 4.2 Related Work on Conformal Prediction in Sequential Settings

The paper must engage with the sequential conformal prediction literature, not just the standard split conformal prediction framework. Relevant papers include:

- **Barber et al. (2023)**, "Conformal prediction beyond exchangeability," *Annals of Statistics* — the formal treatment of when conformal prediction extends beyond i.i.d. settings.
- **Angelopoulos & Bates (2023)**, "A gentle introduction to conformal prediction and distribution-free uncertainty quantification," arXiv:2107.07511 — the standard tutorial reviewers will cite.
- **Tibshirani et al. (2019)**, "Conformal prediction under covariate shift," NeurIPS 2019 — relevant because audio domain shift (test audio with different codec/noise than calibration set) creates covariate shift.

Without these, the CPSL contribution has no formal statistical grounding visible to reviewers.

#### 4.3 Forensic Standards That Strengthen the Legal Framing

- **ENFSI BPM for Digital Audio Authenticity Analysis (2022)** — already in deep-research-report.md but must be explicitly cited in the paper to support the court-suitable framing.
- **SWGDE Best Practices for Digital Audio Authentication** — ditto.
- **ISO/IEC 27037** and **NIST SP 800-86** — for chain-of-custody and reproducibility claims.

Omitting these will cause forensic science reviewers to question whether the authors are familiar with existing professional standards.

---

## Gap 5: Detector Selection — SAL as Primary Detector Is Problematic for a Forensics Journal

### Severity: HIGH (likely to trigger major revision or rejection)

### Finding

As documented in detector-selection.md, SAL (Mao et al., arXiv:2601.21925) is the primary detector for the paper. Its current status:
- **0 citations** as of 2026-03
- **Not peer-reviewed** (arXiv only, submitted January 2026)
- **Very recent** (the same month as this project's initiation)

IEEE TIFS is a journal that emphasizes reproducible, validated science. A paper that positions an unreviewed, zero-citation preprint as its primary detection system will face the following reviewer objections:

1. **"The primary detector has not been peer-reviewed. How can the forensic pipeline be trusted when its core component lacks independent scientific validation?"**
2. **"If SAL's results are challenged or revised in peer review, the entire experimental foundation of this paper changes."**
3. **"Using a system from January 2026 in a paper submitted in 2026 suggests the authors have not had time to conduct a thorough evaluation."**

The detector-selection.md document acknowledges this risk: "SAL is currently an arXiv preprint (January 2026), not yet peer-reviewed." The mitigation proposed (emphasize WavLM and Conformer are peer-reviewed components) is insufficient for a forensics journal, because the specific combination — Segment Positional Labeling + Cross-Segment Mixing — is the novelty in SAL, and those components are not peer-reviewed.

### Additional Detector Coverage Gap

The detector selection specifically excludes ADD 2023 Track 2: "No model evaluated on ADD 2023 Track 2 with public code" is listed as a risk in detector-selection.md. This means the paper cannot report results on one of the two primary partial spoof benchmarks (ADD 2023 Track 2) with its chosen detectors. Reviewers will notice that CFPRF, BAM, and SAL were not evaluated on ADD 2023 Track 2. The DKU-DUKEECE system (Track 2 winner) has no public code, but the lack of any ADD 2023 Track 2 evaluation in the paper represents a gap in benchmark coverage.

### Recommendation

Three viable paths:

**Path A (Preferred):** Promote BAM (Zhong et al., Interspeech 2024) as the primary detector. BAM is peer-reviewed, has 17 citations, produces frame-level predictions, has public code, and is published at a top speech venue. Its multi-resolution evaluation (20ms to 640ms) is especially valuable for forensic applications. Demote SAL to "recently proposed method that we additionally evaluate as a state-of-the-art comparison." This removes the preprint risk while still including SAL's performance numbers.

**Path B:** Retain SAL as primary but add BAM as a peer-reviewed co-primary with equal weight. Frame the paper as evaluating two complementary systems: one peer-reviewed (BAM) and one state-of-the-art preprint (SAL). This is defensible if the paper explicitly acknowledges SAL's preprint status and demonstrates that BAM results replicate the key findings.

**Path C:** Wait for SAL to be accepted at a peer-reviewed venue before submitting the paper. Interspeech 2026 or ICASSP 2026 acceptance would solve the problem. However, this delays submission and increases scoop risk for CPSL.

The BAM (Path A) solution is cleanest because BAM is already in the detector pool, has been implemented, and produces frame-level predictions. The paper does not need to remove SAL — it simply needs to change the hierarchy.

---

## Gap 6: Conformal Prediction i.i.d. Assumption vs. Temporal Audio Frames

### Severity: CRITICAL (technical correctness is at stake)

### Finding

This is the most serious technical vulnerability identified in the paper design. Standard split conformal prediction (Papadopoulos et al., 2002; Vovk et al., 2005) requires that calibration and test scores be **exchangeable** (a generalization of i.i.d. that allows for some dependence but requires that the joint distribution of any permutation of the scores is the same).

For temporal audio frames, this assumption is violated by construction:

1. **Frame-level scores are temporally correlated.** At 20ms resolution, adjacent frames share SSL feature context (the wav2vec2-XLSR and WavLM receptive fields span hundreds of milliseconds). A frame at time t and a frame at time t+20ms are not independent — they share convolutional context from the SSL encoder.

2. **The conformal prediction guarantee is on utterances, not frames.** If CPSL applies conformal prediction at the frame level, it needs either (a) frame-level exchangeability (violated), or (b) an utterance-level calibration scheme where the calibration point is the entire frame sequence for an utterance (this is valid but requires careful definition of what "coverage" means at utterance level).

3. **The calibration set is likely from PartialSpoof, but the test set is different datasets.** Covariate shift between calibration and test distributions violates exchangeability even if frames were i.i.d. within a distribution. Tibshirani et al. (2019), "Conformal prediction under covariate shift," NeurIPS 2019, address this, but their method requires importance weights, which are non-trivial to compute for audio.

### What Reviewers Will Ask

Any reviewer with conformal prediction expertise (an increasingly common background in ML/statistics) will ask: "What is the exchangeability unit? Is it the frame, the utterance, or the segment? What happens to coverage when frames are correlated?" This is not a minor technical detail — if the exchangeability assumption is wrong, the coverage guarantee (the primary claimed value of CPSL) does not hold, and the entire Contribution 1 collapses.

### Mitigation Options

1. **Apply conformal prediction at the utterance level, not the frame level.** Define a nonconformity score based on the worst-case (or average) frame-level score within a candidate interval. This yields utterance-level coverage: "with probability 1-α over calibration utterances, the true manipulated intervals are within the predicted set." Exchangeability holds if calibration and test utterances are drawn i.i.d. from the same distribution. This is a valid design but changes the scope of CPSL from frame-level to utterance-level interval prediction.

2. **Use conformal prediction for sequences (e.g., SPCI — Sequential Predictive Conformal Inference).** Lindemann et al. (2023), "Conformal prediction for time series," *NeurIPS 2023* address time-series conformal prediction. The paper would need to adopt a sequence-aware conformal method and cite this literature. This is technically correct but increases complexity.

3. **Apply conformal prediction at the segment level after a first-stage localization.** The CFPRF system produces temporal proposals (segments with boundaries). Apply conformal prediction to the segment-level score for each proposal, not to individual frames. Segment-level scores from the Proposal Refinement Network can be treated as approximately exchangeable across utterances (assuming the PRN aggregates frame-level information into a proposal-level score that is not directly sharing context across utterances). This is the most practically viable approach given the chosen detectors.

4. **Use PAC-style guarantees instead.** Probably Approximately Correct (PAC) prediction sets can provide coverage guarantees that are more robust to temporal dependence than standard conformal prediction. However, this changes the technical contribution substantially.

**The paper must resolve this before submission.** Whichever mitigation is chosen, the paper must state explicitly: (a) what the exchangeability unit is, (b) why exchangeability holds at that unit, (c) what happens to coverage if covariate shift occurs between calibration and test data, and (d) how the conformal calibration set was constructed to ensure independence from the test set.

---

## Gap 7: Why Not Train Your Own Detector?

### Severity: MODERATE (manageable with clear positioning)

### Finding

Reviewers at IEEE TIFS, which publishes both applied forensics and methods papers, will ask why the paper uses pre-trained detectors rather than training a new detection model. This concern is most likely from reviewers from the speech/audio ML sub-community. The forensic computing sub-community is more accepting of evaluation-focused papers.

The concern is fair because the three contributions (CPSL, PDSM-PS, calibration) are all evaluated *on top of* existing detectors. The paper's claim is essentially: "given a frame-level detector, here is how to wrap it with conformal coverage, phoneme-interpretable saliency, and calibrated scores." This is a valid and publishable design — but the paper must justify it explicitly.

### Why This Framing Is Defensible

1. **The forensic pipeline architecture is the contribution, not the detector.** The paper is analogous to calibration papers (e.g., Platt, 2000; Niculescu-Mizil & Caruana, 2005) that showed calibration of SVM/random forest outputs was valuable independent of training new classifiers. The forensic framing requires calibration + coverage guarantees + interpretability — none of which require a new detector.

2. **Using established pre-trained models strengthens the reproducibility argument.** For court use, a pre-trained system from a published paper is more defensible than a custom-trained model, because independent experts can reproduce the base model's behavior.

3. **The field's leading detectors (SAL, BAM, CFPRF) were not designed with forensic use in mind.** The paper contributes the forensic wrapping that converts detection outputs into court-suitable evidence objects.

### What Must Be Said in the Paper

The paper must include a paragraph in the introduction or methodology section explicitly stating: "We intentionally use existing pre-trained detectors rather than training a new detection model, for three reasons: (1) forensic reproducibility requires that the detection component be independently verifiable; (2) the paper's technical contributions are in the calibration, coverage guarantee, and saliency layers rather than the detection layer; (3) our framework is designed to be detector-agnostic, and using multiple pre-trained detectors (SAL, BAM/CFPRF, MRM) demonstrates this generality." Without this explicit justification, reviewers will assume the paper could not train a competitive detector rather than that it chose not to.

---

## Gap 8: Cross-Dataset Generalization Is Not Adequately Addressed

### Severity: HIGH (forensic credibility gap)

### Finding

The dataset-research.md documents a finding that is devastating for any court-admissibility claim: **models achieving <1% EER on PartialSpoof degrade to 24-43% EER on out-of-domain datasets (LlamaPartialSpoof, HQ-MPSD).** The paper design includes four datasets (PartialSpoof, PartialEdit, HQ-MPSD, LlamaPartialSpoof), which is commendably broad. However, the current gap analysis reveals two specific problems:

1. **The conformal prediction coverage guarantee is only valid on the calibration distribution.** If CPSL is calibrated on PartialSpoof and tested on LlamaPartialSpoof (where detector performance degrades from 3% to 36% EER for SAL), the coverage guarantee derived from PartialSpoof calibration will not hold on LlamaPartialSpoof. The paper must either (a) show per-dataset conformal calibration (calibrate separately on each dataset's validation split), or (b) explicitly warn that cross-dataset coverage is not guaranteed and quantify coverage degradation empirically.

2. **HQ-MPSD includes multilingual audio.** The xai-research.md and dataset-research.md both note that HQ-MPSD shows >80% performance degradation in cross-language evaluation. The PDSM-PS saliency method uses phoneme boundaries from forced alignment/PPGs. For non-English languages, phoneme-level saliency requires language-appropriate forced alignment, and the ASR/alignment models used in PDSM (which was tested only on English TTS) may not generalize to the 8 languages in HQ-MPSD. This must either be addressed experimentally or scoped out explicitly.

### Recommendation

Add a dedicated cross-dataset evaluation section that reports conformal prediction coverage on each test dataset separately. If coverage degrades, report the empirically observed coverage vs. the nominal guarantee (1-α). This empirical calibration transfer analysis is itself a publishable contribution and directly addresses the forensic question "does the coverage guarantee hold on unseen attack types?"

---

## Gap 9: Saliency Faithfulness Evaluation Is Incomplete

### Severity: MODERATE

### Finding

The PDSM paper (Gupta et al., Interspeech 2024) used AOPC-style perturbation metrics for faithfulness evaluation. The xai-research.md documents that normalized AOPC (NAOPC, Edin et al., 2024, arXiv:2408.08137, accepted ACL 2025) shows that standard AOPC "can radically change results, questioning the conclusions of earlier studies." If PDSM-PS uses standard AOPC for faithfulness evaluation, a reviewer familiar with Edin et al. will immediately raise this issue.

Additionally, Grinberg et al. (ICASSP 2025, arXiv:2501.13887) showed that "XAI results obtained from a limited set of utterances do not necessarily hold when evaluated on large datasets." This means phoneme-level saliency results must be reported on the full test set, not selected examples.

### Recommendation

Use NAOPC rather than AOPC for faithfulness evaluation of PDSM-PS. Cite Edin et al. (2024) explicitly. Report saliency faithfulness metrics across the full evaluation set, not just selected examples. Include standard deviation or confidence intervals on faithfulness metrics to allow statistical comparison.

---

## Gap 10: BAM's Boundary-Artifact Bias Creates a Confounding Variable for PDSM-PS

### Severity: MODERATE (methodological concern)

### Finding

The detector-selection.md documents that BAM (Boundary-Aware Attention Mechanism) "focuses on boundary features risks overfitting to boundary artifacts." PDSM-PS applies phoneme-discretized saliency to frame-level detector outputs. If the underlying detector (BAM or CFPRF) is itself attending to boundary artifacts rather than manipulation content, the PDSM saliency map will highlight boundary phonemes — not because the content of those phonemes is synthetic, but because the detector's attention is driven by the transition artifact.

This creates a confounding interpretation: when PDSM-PS produces a saliency map showing "phonemes /k/ /ae/ /t/ are suspicious," the court cannot know whether those phonemes are suspicious because they are synthetic or because they happen to be adjacent to a splice boundary.

### Recommendation

The paper should compare PDSM-PS saliency attribution between SAL (which uses Cross-Segment Mixing to reduce boundary dependence) and BAM/CFPRF (which are boundary-focused). If the saliency attribution is substantially different between detectors, this difference must be reported and interpreted. An experiment comparing saliency maps on utterances with vs. without detectable splice boundaries (using HQ-MPSD, which uses forced-alignment splicing that minimizes boundary artifacts) would directly test whether PDSM-PS saliency is boundary-driven or content-driven. This experiment is independently publishable and would strengthen the forensic argument.

---

## Gap 11: The Calibration Comparison Is Underpowered Without Frame-Level Specificity

### Severity: MODERATE

### Finding

Contribution 3 (systematic post-hoc calibration comparison) uses temperature scaling, Platt scaling, and isotonic regression. These calibration methods were primarily developed and evaluated for utterance-level classification. Applying them to frame-level classification (20-160ms resolution over temporal sequences) introduces two non-trivial issues:

1. **Temporal dependencies mean that frame-level calibration errors are correlated.** Expected Calibration Error (ECE) computed over frames from a single utterance is not independent — if the calibration is wrong for one frame, it is likely wrong for adjacent frames. Standard calibration metrics (ECE, reliability diagrams) assume independent test samples. The paper must use utterance-stratified calibration evaluation or report per-utterance calibration error.

2. **Isotonic regression is non-parametric and can overfit small calibration sets.** For frame-level CM scores with temporal correlation, isotonic regression requires a large calibration set of diverse utterances to avoid fitting the correlation structure of specific utterances. The calibration set size and diversity must be reported.

3. **The forensic use case requires calibration at a specific operating point.** The paper should report not just ECE but also reliability at the operating point chosen for forensic reporting (e.g., threshold corresponding to 5% false positive rate). This is the calibration information courts actually need.

### Recommendation

Report calibration metrics using utterance-stratified cross-validation rather than frame-level pooling. Include both ECE and a reliability diagram. Add Brier score and negative log-likelihood as additional calibration metrics beyond ECE. Explicitly compare calibrated vs. uncalibrated score distributions at the forensic operating point. Wang et al. (2024, arXiv:2406.10836) provides a model for how to structure this comparison in the audio spoof detection context.

---

## Gap 12: Missing Evaluation on PartialEdit Creates a Significant Scope Limitation

### Severity: MODERATE-HIGH (limits impact claim)

### Finding

The dataset-research.md documents that **"models trained on the existing PartialSpoof dataset fail to detect partially edited speech generated by neural speech editing models" (PartialEdit, Zhang et al., Interspeech 2025).** The paper includes PartialEdit as one of its four datasets, which is forward-thinking. However, the detectors selected (SAL, BAM/CFPRF, MRM) were all trained on PartialSpoof using concatenation-based splicing. On PartialEdit (neural speech editing without splice boundaries), their performance will likely be very poor.

This creates a dilemma:
- **If the paper reports the poor performance on PartialEdit honestly,** it demonstrates that the pipeline fails on the most realistic current attack type, which weakens the court-suitability claim.
- **If the paper omits PartialEdit evaluation,** reviewers familiar with the Interspeech 2025 landscape will ask why the paper ignores the most recent challenge dataset.

### Recommendation

Report PartialEdit results honestly and frame poor performance as a **documented limitation and future work direction** rather than trying to avoid it. The forensic framing actually benefits from honest error rate reporting — Daubert factor 3 requires known/potential error rates, and reporting "this system has high error rate on neural speech editing attacks; see PartialEdit evaluation" is more forensically defensible than omitting the evaluation. A limitation section that quantifies error rates across attack types (splice-based: low error, neural editing: high error) is scientifically stronger than a paper that only shows results on favorable datasets.

---

## Gap Prioritization Scores

| Gap | Scientific Impact | Feasibility to Fix | Clinical/Forensic Relevance | Novelty Impact | Priority Score |
|-----|------------------|--------------------|----------------------------|----------------|----------------|
| Gap 6: i.i.d. assumption in conformal prediction | 5 | 3 | 5 | 5 | **4.55** — CRITICAL |
| Gap 5: SAL detector not peer-reviewed | 4 | 5 | 5 | 3 | **4.30** — HIGH |
| Gap 3: Three contributions, no unified claim | 4 | 4 | 4 | 3 | **3.85** — HIGH |
| Gap 2: Competing paper scoop risk (CPSL) | 3 | 2 | 4 | 5 | **3.45** — HIGH |
| Gap 4: Missing related work | 4 | 5 | 3 | 2 | **3.60** — HIGH |
| Gap 8: Cross-dataset coverage not guaranteed | 4 | 3 | 5 | 3 | **3.80** — HIGH |
| Gap 1: Court-admissibility framing | 2 | 5 | 4 | 1 | **2.90** — MODERATE |
| Gap 7: No new detector trained | 2 | 5 | 3 | 1 | **2.65** — MODERATE |
| Gap 10: Boundary-artifact confounding in PDSM | 3 | 3 | 4 | 3 | **3.20** — MODERATE |
| Gap 9: AOPC faithfulness metric issue | 3 | 5 | 3 | 2 | **3.15** — MODERATE |
| Gap 11: Frame-level calibration methodology | 3 | 4 | 3 | 2 | **3.05** — MODERATE |
| Gap 12: PartialEdit evaluation | 3 | 4 | 4 | 2 | **3.25** — MODERATE |

*Scoring: 1=low, 5=high. Priority = (Impact×0.30 + Feasibility×0.25 + Relevance×0.25 + Novelty×0.15) + 1.00 (base). Higher score = higher priority for resolution.*

---

## Top 3 Priority Research Questions Generated from Gap Analysis

### Priority 1: Resolving Temporal Exchangeability in CPSL

**PICO-style framing:**

- **Problem:** Standard conformal prediction requires exchangeable calibration and test scores. Frame-level scores from temporal audio are correlated. This makes the CPSL coverage guarantee technically invalid if applied at the frame level.
- **Research Question:** At what temporal aggregation unit (frame, segment, utterance) does exchangeability approximately hold for partial spoof CM scores, and what is the empirical coverage degradation as temporal correlation increases?
- **Proposed Experiment:** Apply CPSL at three levels — frame-level (20ms), segment-level (100-500ms), utterance-level — and measure empirical coverage (fraction of test utterances where true manipulated regions fall within the predicted conformal set) vs. nominal (1-α) guarantee on both in-domain (PartialSpoof) and out-of-domain (LlamaPartialSpoof) test sets. Report coverage gap = (1-α) - empirical_coverage as a function of temporal unit and distribution shift.
- **Expected Finding:** Utterance-level CPSL will have closer-to-nominal coverage; frame-level CPSL will show undercoverage due to correlation. This frames the correct scope for CPSL.
- **Feasibility:** Can be computed post-hoc from existing model outputs without retraining. Requires only conformal prediction code (freely available in Python via the `nonconformist` or `mapie` libraries).

### Priority 2: Peer-Reviewed Detector Hierarchy for Forensic Pipeline

**PICO-style framing:**

- **Problem:** Primary detector (SAL) is a zero-citation arXiv preprint. Journal reviewers will not accept this without a peer-reviewed fallback as primary.
- **Research Question:** Does the forensic pipeline's performance (calibration quality, conformal coverage, PDSM faithfulness) change substantially when using BAM (peer-reviewed, Interspeech 2024) as primary detector vs. SAL (arXiv preprint)?
- **Proposed Experiment:** Run the complete pipeline (calibration → conformal prediction → PDSM-PS saliency) with BAM as primary and SAL as secondary. Report whether coverage guarantees, calibration ECE, and saliency faithfulness metrics are significantly different between the two detector choices.
- **Expected Finding:** The pipeline (calibration + conformal + saliency) is likely detector-agnostic in terms of its properties, because these components operate on scores rather than on the detection architecture. If true, this is actually a positive result — the paper can claim "detector-agnostic forensic wrapping" as an additional contribution.
- **Feasibility:** BAM has public code (GitHub: media-sec-lab/BAM). Training BAM on PartialSpoof is feasible in one training run. The calibration and saliency components are post-hoc and apply unchanged.

### Priority 3: Unified Thesis Validation — Single-Contribution Restructuring

**PICO-style framing:**

- **Problem:** Three independent contributions do not form a unified scientific argument for IEEE TIFS.
- **Research Question:** Can the three contributions be arranged as a logical dependency chain (calibration enables conformal coverage; conformal coverage determines which intervals to explain; PDSM-PS makes explanations interpretable) where each is required for the final forensic output?
- **Proposed Experiment:** Ablation study showing that the full pipeline (calibration + CPSL + PDSM-PS) produces forensic evidence objects that are superior to any sub-combination on a forensic utility metric (defined as: does the predicted interval contain the true manipulation at the specified coverage level, and is the saliency attribution faithful to the localized region?). The ablation table shows: uncalibrated + no conformal + raw saliency (baseline) → calibrated + no conformal + raw saliency → calibrated + CPSL + raw saliency → calibrated + CPSL + PDSM-PS (full pipeline).
- **Expected Finding:** Each component contributes measurably to the forensic utility metric, justifying all three as necessary components of a single pipeline rather than three independent contributions.
- **Feasibility:** Requires only re-running existing experiments with components removed. The design is already present in the pipeline architecture described in detector-selection.md Section 5.

---

## Summary Action Items Ranked by Urgency

| Priority | Action | Effort | Blocks Submission? |
|----------|--------|--------|--------------------|
| 1 | Fix i.i.d. assumption: define exchangeability unit and conformal level (frame/segment/utterance) | Medium | YES — technical correctness |
| 2 | Promote BAM to primary or co-primary detector; demote SAL to "state-of-the-art comparison" | Low | YES — reviewer credibility |
| 3 | Restructure as single-thesis paper: CPSL primary, PDSM-PS and calibration as supporting components | Low | YES — narrative coherence |
| 4 | Add all mandatory citations (Luong 2025, He survey 2025, Grinberg ICASSP 2025, FADEL, Wang 2024 fusion) | Low | YES — desk rejection risk |
| 5 | Retitle: replace "court-admissible" with "forensically defensible" or "court-suitable" | Low | Moderate — framing risk |
| 6 | Add cross-dataset conformal coverage evaluation; report coverage gap on LlamaPartialSpoof and HQ-MPSD | Medium | Moderate — impact claim |
| 7 | Replace AOPC with NAOPC for saliency faithfulness; cite Edin et al. 2024 | Low | Moderate — methodological soundness |
| 8 | Add explicit justification paragraph for using pre-trained detectors | Low | Low — addressable in rebuttal |
| 9 | Add boundary-artifact confounding experiment (SAL vs. BAM saliency on HQ-MPSD) | Medium | Low — strengthens paper |
| 10 | Report PartialEdit results honestly; frame as documented limitation | Medium | Low — completeness |
| 11 | Use utterance-stratified calibration evaluation; add Brier score and NLL as calibration metrics | Low | Low — methodological strength |
| 12 | Add sequential conformal prediction references (Barber 2023, Tibshirani 2019) to related work | Low | Low — statistical grounding |

---

## References Supporting This Analysis

All citations below are drawn from the project's compiled research files and are verified against published or archived sources.

1. Mao, Y., Huang, W., & Qian, Y. (2026). Localizing Speech Deepfakes Beyond Transitions via Segment-Aware Learning. arXiv:2601.21925. [Primary detector — SAL; preprint status documented]

2. Zhong, J., Li, B., & Yi, J. (2024). Enhancing Partially Spoofed Audio Localization with Boundary-aware Attention Mechanism. Interspeech 2024. arXiv:2407.21611. [BAM — recommended primary replacement]

3. Wu, J., et al. (2024). Coarse-to-Fine Proposal Refinement Framework for Audio Temporal Forgery Detection and Localization. ACM Multimedia 2024. arXiv:2407.16554. [CFPRF — peer-reviewed detector]

4. Gupta, S., Ravanelli, M., Germain, P., & Subakan, C. (2024). Phoneme Discretized Saliency Maps for Explainable Detection of AI-Generated Voice. Interspeech 2024. DOI:10.21437/Interspeech.2024-632. [PDSM original — basis for PDSM-PS]

5. Kang, J.Y., et al. (2025). FADEL: Uncertainty-aware Fake Audio Detection with Evidential Deep Learning. ICASSP 2025. arXiv:2504.15663. [Alternative UQ method — must be compared against CPSL]

6. Wang, X., Kinnunen, T., et al. (2024). Revisiting and Improving Scoring Fusion for Spoofing-aware Speaker Verification. Interspeech 2024. arXiv:2406.10836. [Closest existing calibration work — must be cited]

7. Luong, H.-T., et al. (2025). Robust Localization of Partially Fake Speech: Metrics and Out-of-Domain Evaluation. APSIPA 2025. arXiv:2507.03468. [Cross-dataset evaluation methodology — mandatory citation]

8. He, J., et al. (2025). Manipulated Regions Localization For Partially Deepfake Audio: A Survey. arXiv:2506.14396. [Only survey on this topic — mandatory citation]

9. Grinberg, P., et al. (2025). What Does an Audio Deepfake Detector Focus on? A Study in the Time Domain. ICASSP 2025. arXiv:2501.13887. [XAI faithfulness comparison with partial spoof test — mandatory citation]

10. Zhang, Y., Tian, B., Zhang, L., & Duan, Z. (2025). PartialEdit: Identifying Partial Deepfakes in the Era of Neural Speech Editing. Interspeech 2025. arXiv:2506.02958. [Dataset showing detectors fail on neural editing — must discuss]

11. Edin, J., et al. (2024). Normalized AOPC: Fixing Misleading Faithfulness Metrics for Feature Attribution Explainability. arXiv:2408.08137. [Replace AOPC with NAOPC in faithfulness evaluation]

12. Barber, R.F., Candes, E.J., Ramdas, A., & Tibshirani, R.J. (2023). Conformal prediction beyond exchangeability. Annals of Statistics, 51(2), 816-845. [Formal statistical basis for conformal prediction scope]

13. Angelopoulos, A.N., & Bates, S. (2023). A gentle introduction to conformal prediction and distribution-free uncertainty quantification. arXiv:2107.07511. [Standard tutorial — must be cited for CPSL]

14. Tibshirani, R.J., Barber, R.F., Candes, E.J., & Ramdas, A. (2019). Conformal prediction under covariate shift. NeurIPS 2019. [Addresses cross-dataset conformal coverage — critical for Gap 8]

15. ENFSI (2022). Best Practice Manual for Digital Audio Authenticity Analysis. FSA-BPM-002. [Forensic standard — required for legal framing section]

16. Negroni, V., et al. (2024). Analyzing the Impact of Splicing Artifacts in Partially Fake Speech Signals. ASVspoof 5 Workshop, Interspeech 2024. arXiv:2408.13784. [Non-ML baseline comparison — mandatory for positioning]

17. Li, M., Alber, M., et al. (2025). HQ-MPSD: A Multilingual Artifact-Controlled Benchmark. arXiv:2512.13012. [Dataset with multilingual scope — phoneme alignment gap applies]

18. Xian, H., et al. (2025). NE-PADD: Leveraging Named Entity Knowledge for Robust Partial Audio Deepfake Detection. APSIPA 2025. arXiv:2509.03829. [Adjacent to PDSM-PS — must differentiate]

19. Salvi, D., et al. (2025). Phoneme-Level Analysis for Person-of-Interest Speech Deepfake Detection. ICCV Workshop 2025. arXiv:2507.08626. [Adjacent to PDSM-PS — must differentiate]

20. Zhang, L., et al. (2023). The PartialSpoof Database and Countermeasures. IEEE/ACM TASLP. arXiv:2204.05177. [MRM baseline — 98 citations, foundational]
