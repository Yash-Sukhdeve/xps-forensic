# Scoop Analysis: Competitive Landscape for Our Research Contributions

**Date of Analysis**: 2026-03-02
**Search Period**: January 2025 -- March 2026
**Databases Searched**: arXiv (cs.SD, cs.CR, cs.AI, eess.AS), OpenAlex, IEEE Xplore, ISCA Archive
**Search Strategy**: 7 query combinations x 4+ databases; targeted author/group searches; venue-specific checks (ICASSP 2025/2026, Interspeech 2025, NeurIPS 2025, IEEE TIFS, IJCAI 2025, ASRU 2025, ACL 2026)

---

## Executive Summary

**No complete scoop exists for any of our four contributions.** The intersection of conformal prediction + audio deepfake/spoofing remains entirely unoccupied in the literature. However, several papers create partial overlaps that require careful positioning. The most important findings are:

| Contribution | Scoop Risk | Threat Level | Action Needed |
|---|---|---|---|
| CPSL (Conformal Prediction for Spoof Localization) | **NONE** | LOW | Clear novelty; cite Zhu et al. (2025) as analogous CP work in text domain |
| PDSM-PS (PDSM extended to partial spoof) | **NONE** | LOW-MEDIUM | Gupta/Ravanelli have not published a follow-up; Grinberg et al. offer alternative XAI but no PDSM extension |
| Systematic calibration comparison for audio CM | **PARTIAL** | MEDIUM | Kwok et al. (ICASSP 2025) and ASVspoof 5 eval (2026) touch calibration, but neither does a systematic Platt/temp/isotonic comparison |
| Integrated XAI + calibration + uncertainty forensic pipeline | **NONE** | LOW | No one has proposed this combination; closest is Grinberg/Bharaj XAI work (no calibration/uncertainty) |

---

## Contribution 1: CPSL -- Conformal Prediction for Audio Deepfake/Spoof Localization

### Search Results

**Zero papers found** applying conformal prediction to audio deepfake detection, spoofing countermeasures, or audio anti-spoofing at any level (utterance or segment).

**Confirmed zero-hit searches:**
- arXiv: "conformal prediction" AND "deepfake" -- 0 results (all years)
- arXiv: "conformal prediction" AND "spoofing" -- 0 results (all years)
- arXiv: "conformal prediction" AND "audio" AND "detection" -- 0 results
- arXiv: "distribution-free" AND "audio" AND ("deepfake" OR "spoofing") -- 0 results
- arXiv: "coverage guarantee" AND "audio" (deepfake-related) -- 0 relevant results
- OpenAlex: "conformal prediction" AND "audio deepfake" -- 0 results
- OpenAlex: "conformal prediction" AND "spoofing" AND "speech" -- 0 results
- OpenAlex: "conformal prediction" AND "countermeasure" AND "audio" -- 0 results

### Nearest Neighbors (Adjacent Work)

#### 1. Zhu et al. (2025) -- Conformal Prediction for Machine-Generated Text Detection
- **Paper**: "Reliably Bounding False Positives: A Zero-Shot Machine-Generated Text Detection Framework via Multiscaled Conformal Prediction"
- **arXiv**: 2505.05084, May 2025
- **Overlap**: Applies conformal prediction to AI-generated content detection, but for **text only**. Uses Multiscaled Conformal Prediction (MCP) to bound false positive rates.
- **Threat**: LOW. Different modality (text vs. audio). However, establishes precedent that CP is useful for generated content detection.
- **Action**: Cite as prior art showing CP's value for AI-generated content detection. Emphasize our work as the first to bring CP to the audio/speech domain.

#### 2. Jia et al. (2025) -- Coverage-Guaranteed Speech Emotion Recognition
- **Paper**: "Coverage-Guaranteed Speech Emotion Recognition via Calibrated Uncertainty-Adaptive Prediction Sets"
- **arXiv**: 2503.22712, March 2025
- **Overlap**: Applies conformal prediction / risk-controlled prediction sets to **speech** (but for emotion recognition, not deepfake detection). Uses calibration set with binary loss function for coverage guarantees.
- **Threat**: LOW-MEDIUM. Same modality (speech), same technique (CP), but completely different task (emotion recognition vs. deepfake/spoof detection). Important to cite.
- **Action**: Cite as prior art for CP in speech processing. Clearly differentiate: their task is emotion classification; ours is authenticity/spoof detection with segment-level localization, which has fundamentally different non-conformity score requirements.

#### 3. den Hengst et al. (2025) -- Hierarchical Conformal Classification
- **Paper**: "Hierarchical Conformal Classification"
- **arXiv**: 2508.13288, August 2025
- **Overlap**: Extends CP with hierarchical prediction sets; includes audio benchmarks (unspecified type, likely audio event classification).
- **Threat**: LOW. Generic CP methodology paper, not applied to deepfake detection.
- **Action**: Cite if relevant to methodology.

#### 4. Rozenfeld & Goldshtein (2025) -- Conformal Prediction for Sound Source Localization
- **Paper**: "Conformal Prediction for Manifold-based Source Localization with Gaussian Processes"
- **arXiv**: 2409.11804v2, updated January 2025
- **Overlap**: Applies CP to audio localization -- but spatial localization of sound sources in rooms, not temporal localization of spoofed segments.
- **Threat**: NEGLIGIBLE. Completely different domain (room acoustics / robot audition).
- **Action**: Cite for completeness as CP applied to audio-related tasks.

### Verdict: NO SCOOP. CPSL is novel.

The application of conformal prediction to audio deepfake/spoof detection (at any level) has never been published. The closest work is Zhu et al. (2025) for text and Jia et al. (2025) for speech emotion recognition. Our CPSL contribution is the **first distribution-free coverage guarantee for audio countermeasure decisions**, and remains clearly novel as of March 2026.

---

## Contribution 2: PDSM-PS -- Phoneme-Discretized Saliency Maps Extended to Partial Spoof

### Search Results

**Zero papers found** extending PDSM to partial spoof localization on PartialSpoof or ADD datasets.

**Confirmed zero-hit searches:**
- arXiv: "phoneme" AND "saliency" AND "deepfake" -- 0 results
- arXiv: "partial spoof" AND "saliency" -- 0 results
- OpenAlex: "phoneme saliency" AND "partial spoof" -- 0 results
- OpenAlex: "partial spoof" AND "explainab" -- 0 results
- Gupta/Ravanelli author search 2025+ -- 0 PDSM follow-up papers found

### Original PDSM Paper Status

Gupta et al. "Phoneme Discretized Saliency Maps for Explainable Detection of AI-Generated Voice" (arXiv: 2406.10422, Interspeech 2024) remains the only PDSM paper. No follow-up from the original authors has been published or posted as of March 2026.

### Nearest Neighbors (XAI for Audio Deepfake)

#### 1. Grinberg et al. (ICASSP 2025) -- Relevancy-Based XAI for Audio Deepfake Detection
- **Paper**: "What Does an Audio Deepfake Detector Focus on? A Study in the Time Domain"
- **arXiv**: 2501.13887, January 2025
- **DOI**: 10.1109/icassp49660.2025.10887568
- **Authors**: Petr Grinberg, Ankur Kumar, Surya Koppisetti, Gaurav Bharaj (Hume AI)
- **Overlap**: Proposes relevancy-based XAI for transformer audio deepfake detectors. Compares against Grad-CAM and SHAP using faithfulness metrics and a **partial spoof test**. Examines phonetic content importance.
- **Threat**: MEDIUM for the XAI component, but LOW for PDSM-PS specifically. They do NOT use PDSM. They do NOT extend to PartialSpoof/ADD localization tasks. Their partial spoof test is used as a validation mechanism for XAI faithfulness, not as a localization target.
- **Action**: Cite and compare. Their relevancy method is a competing XAI approach, but our PDSM-PS is a distinct method (phoneme-boundary-discretized saliency) applied specifically to the partial spoof localization task, not just a validation test.

#### 2. Grinberg et al. (Interspeech 2025) -- Diffusion-Based Audio Deepfake Explanations
- **Paper**: "A Data-Driven Diffusion-based Approach for Audio Deepfake Explanations"
- **arXiv**: 2506.03425, June 2025
- **Authors**: Petr Grinberg, Ankur Kumar, Surya Koppisetti, Gaurav Bharaj (Hume AI)
- **Overlap**: Uses diffusion models trained on paired real/vocoded audio to produce ground-truth explanations. Evaluates on VocV4 and LibriSeVoc.
- **Threat**: LOW for PDSM-PS. Completely different methodology (diffusion vs. saliency). Does NOT use phoneme boundaries. Does NOT evaluate on PartialSpoof/ADD.
- **Action**: Cite as an alternative XAI approach. Emphasize that our PDSM-PS is phoneme-structured (more interpretable for court contexts) and directly targets partial spoof localization.

#### 3. Negroni et al. (ICASSP 2026) -- Multi-Task Transformer with Formant-Based Explainability
- **Paper**: "Multi-Task Transformer for Explainable Speech Deepfake Detection via Formant Modeling"
- **arXiv**: 2601.14850, January 2026
- **Authors**: Viola Negroni, Luca Cuccovillo, Paolo Bestagini, Patrick Aichroth, Stefano Tubaro
- **Overlap**: Built-in explainability via formant trajectory prediction, highlighting voiced vs. unvoiced regions. Does NOT do partial spoof localization. Does NOT use saliency maps or phoneme boundaries.
- **Threat**: LOW. Different explainability mechanism (formant-based vs. phoneme-saliency-based). Complementary rather than competing.
- **Action**: Cite as a complementary XAI approach.

#### 4. Kuhlmann et al. (ICASSP 2026) -- Speech Quality-Based Partial Spoof Localization
- **Paper**: "Speech Quality-Based Localization of Low-Quality Speech and Text-to-Speech Synthesis Artefacts"
- **arXiv**: 2601.21886, January 2026
- **Overlap**: Uses frame-level speech quality scores for partial spoof localization. An interpretable approach but NOT saliency-based and NOT phoneme-discretized.
- **Threat**: LOW for PDSM-PS (different method). MEDIUM as a competing interpretable localization approach.
- **Action**: Cite and compare. Their quality-based approach is complementary; our PDSM-PS provides attribution-level explanations tied to phoneme structure.

#### 5. Govindu et al. (2025, preprint) -- Deepfake Audio Detection with XAI
- **Paper**: "Deepfake audio detection and justification with Explainable Artificial Intelligence"
- **DOI**: 10.21203/rs.3.rs-6349440/v1
- **Overlap**: Uses LIME, SHAP, Grad-CAM for audio deepfake XAI. No partial spoof localization. No PDSM.
- **Threat**: NEGLIGIBLE. Standard post-hoc XAI methods, no localization, no phoneme-level analysis.

### Verdict: NO SCOOP. PDSM-PS is novel.

No one has extended PDSM to partial spoof localization. The Grinberg/Bharaj group is the most active in audio deepfake XAI but uses completely different methods (relevancy-based, diffusion-based). Our PDSM-PS contribution -- applying phoneme-discretized saliency to PartialSpoof/ADD localization tasks -- remains unpublished by anyone.

---

## Contribution 3: Systematic Calibration Comparison for Audio Deepfake CM Scores

### Search Results

**Zero papers found** that systematically compare Platt scaling, temperature scaling, and isotonic regression specifically on audio deepfake countermeasure scores.

**Confirmed zero-hit searches:**
- arXiv: "calibration" AND "audio deepfake" -- 1 result (Pascu et al. 2023, not a systematic comparison)
- arXiv: "Platt scaling" AND ("deepfake" OR "spoof") AND "audio" -- 0 results
- arXiv: "temperature scaling" AND "spoofing" -- 1 result (Kim et al. 2025, uses temperature in training, not post-hoc calibration)
- OpenAlex: "calibration" AND "audio deepfake" AND "Platt" -- 0 results
- OpenAlex: "spoofing" AND "calibration" AND "temperature scaling" -- 0 results

### Nearest Neighbors

#### 1. Kwok et al. (ICASSP 2025) -- Ensemble Confidence Calibration for Audio Deepfake Detection
- **Paper**: "Robust Audio Deepfake Detection using Ensemble Confidence Calibration"
- **DOI**: 10.1109/icassp49660.2025.10889972
- **Authors**: Chin Yuen Kwok, Duc-Tuan Truong, Jia Qi Yip (NTU Singapore)
- **Overlap**: Proposes EOW-Softmax for calibrating ensemble predictions in audio deepfake detection. Tests on ASVspoof 2021.
- **Threat**: MEDIUM. This is the closest competitor. However, they propose a **single new method** (EOW-Softmax) for **ensemble** calibration. They do NOT systematically compare Platt scaling, temperature scaling, and isotonic regression. They do NOT study calibration across different detector architectures (AASIST, WavLM, wav2vec).
- **Action**: Cite as related work on calibration for audio deepfake. Clearly differentiate: our contribution is a **systematic empirical comparison** of standard calibration methods across multiple detector architectures, not a new calibration method for ensembles.

#### 2. ASVspoof 5 Evaluation Paper (2026) -- Calibration Study
- **Paper**: "ASVspoof 5: Evaluation of Spoofing, Deepfake, and Adversarial Attack Detection Using Crowdsourced Speech"
- **arXiv**: 2601.03944, January 2026
- **Authors**: Xin Wang, Hector Delgado, Nicholas Evans, Tomi Kinnunen, Massimiliano Todisco, Junichi Yamagishi, et al.
- **Overlap**: Mentions "a study of calibration in addition to other principal challenges." Full PDF analysis found no detailed comparison of calibration methods (no Platt, temperature, isotonic, ECE mentioned in text).
- **Threat**: MEDIUM. The calibration study in ASVspoof 5 could overlap if deep, but our PDF analysis suggests it is brief/surface-level (part of a multi-topic challenge summary paper, not a dedicated calibration paper).
- **Action**: MUST READ FULL PAPER when published in final form. If their calibration study is shallow (likely, given it is embedded in a challenge overview), our systematic comparison still adds substantial value. Position as complementary: "Building on ASVspoof 5's observation that calibration matters, we provide the first systematic comparison..."

#### 3. Wang et al. (2024) -- Score Calibration for SASV Fusion
- **Paper**: "Revisiting and Improving Scoring Fusion for Spoofing-aware Speaker Verification Using Compositional Data Analysis"
- **arXiv**: 2406.10836, June 2024 (but relevant as recent context)
- **Authors**: Xin Wang, Tomi Kinnunen, Kong Aik Lee, Paul-Gauthier Noe, Junichi Yamagishi
- **Overlap**: Demonstrates "score calibration before fusion is essential" for SASV. Uses Bosaris-style calibration.
- **Threat**: LOW-MEDIUM. This is about score-level fusion for SASV, not systematic calibration comparison for CM systems.
- **Action**: Cite as motivation. Their finding that calibration matters supports our deeper investigation.

#### 4. Pascu et al. (Interspeech 2024) -- Calibrated SSL-Based Deepfake Detection
- **Paper**: "Towards generalisable and calibrated synthetic speech detection with self-supervised representations"
- **arXiv**: 2309.05384
- **Overlap**: Claims "considerably more reliable predictions" using frozen SSL + logistic regression. Does NOT detail specific calibration methods or compare multiple approaches.
- **Threat**: LOW. Their calibration is intrinsic (logistic regression is inherently calibrated); they do NOT compare post-hoc calibration methods.
- **Action**: Cite as related work.

#### 5. Kim et al. (Interspeech 2025) -- Dynamic Temperature Scaling
- **Paper**: "Naturalness-Aware Curriculum Learning with Dynamic Temperature for Speech Deepfake Detection"
- **arXiv**: 2505.13976, May 2025
- **Overlap**: Uses "dynamic temperature scaling" but as a **training strategy** (modifying softmax temperature during training), NOT as a post-hoc calibration method.
- **Threat**: NEGLIGIBLE for our calibration comparison contribution. Different use of "temperature."
- **Action**: Cite for disambiguation; distinguish training-time temperature from post-hoc temperature scaling.

#### 6. Stourbe et al. (2024) -- Bosaris Toolkit for CM Score Calibration
- **Paper**: "Exploring WavLM Back-ends for Speech Spoofing and Deepfake Detection" (ASVspoof 5 submission)
- **arXiv**: 2409.05032, September 2024
- **Overlap**: Uses "the Bosaris toolkit for score calibration and system fusion" for ASVspoof 5.
- **Threat**: LOW. Toolkit usage, not a systematic comparison of methods.

### Verdict: NO COMPLETE SCOOP. Partial overlap exists.

The Kwok et al. (ICASSP 2025) paper is the most relevant competitor, but it proposes a single method (EOW-Softmax) rather than conducting a systematic comparison. The ASVspoof 5 paper touches calibration but likely not in depth. No paper systematically compares Platt scaling, temperature scaling, and isotonic regression across AASIST, WavLM-based, and wav2vec-based detectors. Our contribution remains novel.

**Key Risk**: If the full ASVspoof 5 evaluation paper contains a deeper calibration analysis than the abstract/PDF suggest, we may need to reposition. **Action: Obtain and read the full final version of arXiv:2601.03944.**

---

## Contribution 4: Integrated XAI + Calibration + Uncertainty Pipeline for Forensic Audio

### Search Results

**Zero papers found** proposing an integrated pipeline combining explainability, calibration, and uncertainty quantification for forensic audio deepfake analysis.

**Confirmed zero-hit searches:**
- arXiv: "forensic" AND "audio deepfake" AND "framework"/"pipeline" -- No integrated XAI+calibration+UQ papers
- arXiv: "audio deepfake" AND "court"/"admissib" -- 0 results
- arXiv: "uncertainty quantification" AND "audio deepfake" -- 0 results
- OpenAlex: "uncertainty quantification" AND "audio deepfake" -- 0 results
- OpenAlex: "forensic" AND "audio deepfake" AND "pipeline" -- Multiple detection papers, none with integrated calibration+XAI+UQ

### Nearest Neighbors

#### 1. Yang et al. (2025) -- Forensic Deepfake Audio Detection Using Segmental Features
- **Paper**: "Forensic Deepfake Audio Detection Using Segmental Speech Features"
- **arXiv**: 2505.13847, May 2025
- **Overlap**: Speaker-specific detection framework using interpretable forensic voice comparison features. Published in Forensic Science International.
- **Threat**: LOW for our integrated pipeline. Uses forensic features for detection interpretability but does NOT include calibration, uncertainty quantification, conformal prediction, or a complete court-ready pipeline.
- **Action**: Cite as complementary work on forensic-grade audio deepfake detection.

#### 2. Nguyen & Le (ACL 2026 submission) -- Forensic Auditing Framework for Audio Deepfake
- **Paper**: "Analyzing Reasoning Shifts in Audio Deepfake Detection under Adversarial Attacks: The Reasoning Tax versus Shield Bifurcation"
- **arXiv**: 2601.03615, January 2026
- **Overlap**: Proposes a "forensic auditing framework" but for evaluating robustness of Audio Language Models under adversarial attacks -- NOT for courtroom-ready evidence packaging.
- **Threat**: LOW. "Forensic auditing" here means evaluating model behavior, not forensic evidence pipeline.
- **Action**: Cite for differentiation.

#### 3. SAFE Challenge (ACM MM 2025) -- Synthetic Audio Forensics Evaluation
- **Paper**: "SAFE: Synthetic Audio Forensics Evaluation Challenge"
- **DOI**: 10.1145/3733102.3736707
- **Authors**: Trapeznikov et al.
- **Overlap**: Forensics evaluation challenge with detection tasks including laundered audio. Establishes forensic evaluation methodology.
- **Threat**: LOW. Challenge/benchmark paper, NOT an integrated pipeline with XAI+calibration+UQ.
- **Action**: Cite as related forensic evaluation work.

#### 4. Kwok et al. (ICASSP 2025) -- Ensemble Confidence Calibration
- See above (Contribution 3). Calibration only, no XAI or UQ.

#### 5. Grinberg et al. (ICASSP 2025 + Interspeech 2025) -- XAI for Audio Deepfake
- See above (Contribution 2). XAI only, no calibration or UQ.

### Verdict: NO SCOOP. Integrated pipeline is novel.

No paper combines XAI + calibration + uncertainty quantification for forensic audio deepfake analysis. Individual components exist in isolation (XAI: Grinberg et al.; calibration: Kwok et al.; forensic features: Yang et al.), but the integrated pipeline -- especially with court admissibility considerations, chain-of-custody, and LLM-as-narrator -- remains entirely novel.

---

## Additional Relevant Papers to Track

### Partial Spoof Localization -- Active Research Area

| Paper | Venue | Authors | Key Contribution | Threat |
|---|---|---|---|---|
| LOCO (2505.01880) | IJCAI 2025 | Wu et al. | Weakly-supervised temporal forgery localization via audio-language co-learning | LOW (no XAI/calibration/CP) |
| Next-Frame Prediction (2511.10212) | Under review | Anshul et al. | Multimodal deepfake temporal localization via next-frame prediction | LOW (no XAI/calibration/CP) |
| PartialEdit (2506.02958) | Interspeech 2025 | Zhang et al. | New dataset for neural speech editing detection; shows PartialSpoof models fail on neural edits | LOW (dataset, not method overlap) |
| Spoof Diarization v2 (2509.13085) | ASRU 2025 | Koo et al. | Token-based attractors for spoof diarization on PartialSpoof | LOW (different method, no XAI) |
| GNCL (10.1109/icassp49660.2025.10888281) | ICASSP 2025 | Ge et al. | GNN with consistency loss for segment-level spoof detection | LOW (no XAI/calibration/CP) |
| CompSpoof (2509.15804) | ICASSP 2026 | Zhang et al. | Component-level audio anti-spoofing dataset + framework | LOW (different task framing) |
| He et al. Survey (2506.14396) | IEEE TPAMI | Yi/Tao group | Survey of manipulated region localization for partial deepfake audio | MEDIUM (survey may identify similar gaps; must cite) |
| Kuhlmann et al. (2601.21886) | ICASSP 2026 | Kuhlmann et al. | Quality-based partial spoof localization | LOW-MEDIUM (complementary) |

### XAI for Audio Deepfake -- Growing but Fragmented

| Paper | Venue | Authors | XAI Method | Threat to PDSM-PS |
|---|---|---|---|---|
| Grinberg et al. (2501.13887) | ICASSP 2025 | Hume AI | Relevancy-based | LOW (no PDSM, no localization task) |
| Grinberg et al. (2506.03425) | Interspeech 2025 | Hume AI | Diffusion-based | LOW (different method entirely) |
| Negroni et al. (2601.14850) | ICASSP 2026 | Politecnico di Milano | Formant-based | LOW (complementary) |
| Govindu et al. (rs-6349440) | Preprint 2025 | -- | LIME/SHAP/Grad-CAM | NEGLIGIBLE |
| Liu et al. (2406.02483) | 2024 | NUS/NTU | Grad-CAM on partial spoof | LOW (2024, different method) |

### Calibration for Audio CM -- Sparse Literature

| Paper | Venue | Authors | Calibration Focus | Threat to Systematic Comparison |
|---|---|---|---|---|
| Kwok et al. (ICASSP 2025) | ICASSP 2025 | NTU | EOW-Softmax (single method) | MEDIUM |
| ASVspoof 5 eval (2601.03944) | 2026 | Yamagishi/Kinnunen et al. | Brief calibration study | MEDIUM (need full paper) |
| Pascu et al. (Interspeech 2024) | 2024 | -- | Intrinsic calibration (logistic regression) | LOW |
| Wang et al. (2024) | 2024 | Yamagishi/Kinnunen | Score fusion calibration (Bosaris) | LOW |

---

## Key Research Groups: Activity Status (Jan 2025 -- Mar 2026)

### Yamagishi Lab (NII, Japan) -- PartialSpoof authors
- **Active on**: ASVspoof 5 challenge design/evaluation, scoring fusion calibration
- **NOT working on**: PDSM, conformal prediction, XAI for partial spoof
- **Risk**: Their survey influence could highlight similar gaps

### Todisco/Evans Lab (EURECOM) -- ASVspoof organizers
- **Active on**: ASVspoof 5 challenge, general spoofing detection
- **NOT working on**: Dedicated calibration comparison, PDSM, conformal prediction
- **Risk**: LOW for our specific contributions

### Yi/Tao Lab (Chinese Academy of Sciences) -- ADD challenge organizers
- **Active on**: Survey of partial deepfake audio localization (He et al., IEEE TPAMI)
- **NOT working on**: XAI/calibration/CP for partial spoof
- **Risk**: Their survey may identify XAI and calibration as gaps, potentially inspiring competitors
- **Action**: Publish before their survey becomes widely read

### Grinberg/Bharaj (Hume AI) -- XAI for Audio Deepfake
- **Active on**: Relevancy-based XAI (ICASSP 2025), diffusion-based XAI (Interspeech 2025)
- **NOT working on**: PDSM, calibration, conformal prediction
- **Risk**: MEDIUM for XAI comparisons. They are the most active XAI group.
- **Action**: Cite both papers; position PDSM-PS as complementary (phoneme-structured vs. general temporal)

### Gupta/Ravanelli (Mila/Concordia) -- PDSM authors
- **No publications found** extending PDSM since Interspeech 2024
- **Risk**: They could still be working on a PDSM follow-up privately
- **Action**: Move quickly to establish priority on PDSM-PS

### Kinnunen Lab (University of Eastern Finland) -- Calibration work
- **Active on**: ASVspoof 5 evaluation, joint speaker-spoof detector optimization (SASV a-DCF calibration)
- **NOT found**: Dedicated calibration comparison paper for CM scores
- **Risk**: MEDIUM. Kinnunen's group has deep calibration expertise (t-DCF design, Bosaris usage). They could publish a calibration study.
- **Action**: Publish calibration comparison before Kinnunen group does

---

## Risk Assessment Matrix

### Immediate Risks (next 3 months)

1. **Gupta/Ravanelli PDSM extension**: The original PDSM authors could independently extend to partial spoof. No evidence of this yet, but the gap is obvious. **Priority: HIGH -- submit PDSM-PS quickly.**

2. **Kinnunen calibration study**: Given their calibration expertise and ASVspoof 5 involvement, a dedicated calibration paper is plausible. **Priority: HIGH -- submit calibration comparison quickly.**

3. **He et al. survey (IEEE TPAMI)**: This survey of partial deepfake audio localization will likely highlight XAI and calibration as open research gaps, potentially inspiring competitors. **Priority: MEDIUM -- cite in our work; be first to fill the gap.**

### Medium-Term Risks (3-12 months)

4. **Conformal prediction for audio**: As CP gains popularity in other domains (text: Zhu et al.; speech emotion: Jia et al.), someone will likely try it for audio deepfake detection. **Priority: MEDIUM -- we need to establish priority.**

5. **Grinberg/Bharaj expanding to partial spoof**: They already used a partial spoof test as validation (ICASSP 2025). A dedicated partial spoof XAI paper is a logical next step. **Priority: MEDIUM -- our PDSM approach is distinct enough to coexist.**

### Low-Risk but Monitor

6. **ASVspoof 5 full calibration analysis**: The evaluation paper's calibration study may be deeper than the abstract suggests. **Action: Obtain full paper.**
7. **ACL 2026 forensic framework papers**: Nguyen & Le submitted a forensic auditing paper. Others may follow. **Monitor ACL 2026 proceedings.**

---

## Differentiation Strategy

### For CPSL:
- **Clear blue ocean**: No CP in audio deepfake exists. Frame as: "first distribution-free coverage guarantees for audio countermeasure decisions"
- Cite Zhu et al. (text) and Jia et al. (speech emotion) as inspiration from adjacent domains
- Emphasize forensic relevance: coverage guarantees map directly to error rate disclosure requirements under Daubert

### For PDSM-PS:
- **Differentiate from Grinberg et al.**: PDSM provides phoneme-structured explanations (linguistically interpretable for court), while relevancy/diffusion approaches give general temporal importance
- **Differentiate from Kuhlmann et al.**: Quality-based localization detects "something wrong" but doesn't explain "what is wrong" at the attribution level
- Frame as: "extending PDSM (Gupta et al., Interspeech 2024) from utterance-level to segment-level partial spoof localization"

### For Systematic Calibration Comparison:
- **Differentiate from Kwok et al.**: They propose a single new method; we provide the first systematic empirical comparison across standard methods and detector architectures
- **Differentiate from ASVspoof 5**: Their calibration study is embedded in a challenge summary; ours is dedicated and thorough
- Frame as: "first systematic comparison of Platt scaling, temperature scaling, and isotonic regression for audio CM score calibration"

### For Integrated Pipeline:
- **Unique combination**: No one has combined XAI + calibration + UQ + forensic chain-of-custody in a single pipeline
- Frame as: "a court-ready forensic pipeline integrating explainability, calibrated confidence, and distribution-free uncertainty quantification"
- Cite NIST AI 600-1, Daubert/FRE 702, ENFSI BPM for legal grounding

---

## Recommended Actions (Priority-Ordered)

1. **IMMEDIATE**: Begin writing CPSL and PDSM-PS papers. The fields are clear but the windows may close.
2. **URGENT**: Obtain and read the full ASVspoof 5 evaluation paper (arXiv:2601.03944) to assess calibration study depth.
3. **URGENT**: Read the He et al. survey (arXiv:2506.14396) to understand what gaps they identify.
4. **HIGH**: Cite all nearest-neighbor papers identified above in our manuscripts.
5. **MEDIUM**: Monitor Gupta/Ravanelli and Kinnunen lab preprint activity monthly.
6. **LOW**: Track ICASSP 2026 camera-ready papers and Interspeech 2026 submissions for late-breaking competitors.

---

## Appendix: Complete Search Log

### Query 1: "conformal prediction" AND audio deepfake/spoofing/anti-spoofing/countermeasure
- arXiv "conformal prediction" AND "deepfake": 0 results (all years)
- arXiv "conformal prediction" AND "spoofing": 0 results
- arXiv "conformal prediction" AND "audio" AND "detection": 0 results
- OpenAlex "conformal prediction" AND "audio deepfake": 0 results
- OpenAlex "conformal prediction" AND "spoofing" AND "speech": 0 results
- OpenAlex "conformal prediction" AND "anti-spoofing": 1 result (FAIR-SIGHT, image fairness, irrelevant)
- OpenAlex "conformal prediction" AND "countermeasure" AND "audio": 0 results

### Query 2: "conformal prediction" AND speech/audio AND localization/detection
- arXiv "conformal prediction" AND "speech" AND "detection": 0 relevant
- arXiv "conformal" AND "speech" AND "uncertainty": 1 result (Rozenfeld, sound source localization, irrelevant)
- arXiv "coverage guarantee" AND "audio": 2 results (Hierarchical CC, Speech Emotion CP)
- OpenAlex "conformal prediction" AND "speech" AND "detection": 0 relevant

### Query 3: "phoneme saliency" AND partial spoof/deepfake
- arXiv "phoneme" AND "saliency" AND "deepfake": 0 results
- arXiv "partial spoof" AND "saliency": 0 results
- OpenAlex "phoneme saliency" AND "partial spoof": 0 results

### Query 4: "calibration" AND audio deepfake/spoofing AND Platt/temperature/isotonic
- arXiv "calibration" AND "audio deepfake": 1 result (Pascu 2023)
- arXiv "Platt scaling" AND ("deepfake" OR "spoof") AND "audio": 0 results
- arXiv "temperature scaling" AND "spoofing": 1 result (Kim 2025, training-time, not post-hoc)
- OpenAlex "calibration" AND "audio deepfake" AND "Platt": 0 results
- OpenAlex "calibration" AND "spoofing countermeasure": GNSS papers only + Nes2Net
- OpenAlex "audio deepfake" AND "calibration": Kwok et al. (EOW-Softmax), plus irrelevant

### Query 5: "explainable" AND "partial spoof" AND "localization"
- arXiv "explainable" AND "partial spoof": 3 results (Kuhlmann 2026, Grinberg 2025, Liu 2024)
- OpenAlex "explainable" AND "partial spoof" AND "localization": 0 results
- OpenAlex "partial spoof" AND "explainab": 0 results

### Query 6: "forensic" AND "audio deepfake" AND pipeline/framework/system
- arXiv "forensic" AND "audio deepfake" AND "framework": Yang 2025, Zhao 2025
- arXiv "audio" AND "deepfake" AND "forensic" AND "admissib": 0 results
- arXiv "audio deepfake" AND "court": 0 results
- OpenAlex "forensic" AND "audio deepfake" AND "pipeline": SAFE challenge, surveys

### Query 7: "uncertainty quantification" AND "audio deepfake"
- arXiv "uncertainty quantification" AND "audio deepfake": 0 results
- arXiv "audio deepfake" AND "uncertainty": 0 results
- OpenAlex "uncertainty quantification" AND "audio deepfake": 0 results
- OpenAlex "audio deepfake" AND "uncertainty": Kwok et al., fuzzy logic paper

### Targeted Author Searches
- Yamagishi + "partial spoof" (2025+): 0 results
- Todisco + "spoofing" (2025+): ASVspoof 5 only
- Kinnunen + "calibration" + "spoofing" (2025+): ASVspoof 5 eval, joint SASV optimization
- Kinnunen + "calibrat" + "speaker" (2025+): 0 results
- Gupta + "saliency" + "speech" (2025+): 0 PDSM follow-up
- Pascu + "calibrat" + "deepfake" (2025+): 0 results

### Venue Searches
- ICASSP 2026 + "deepfake" + "audio": 5 papers (Xiao, Shi, Zhang, Yin, Kim) -- none overlap our contributions
- ICASSP 2026 + "spoof": 8 papers (see list above) -- CompSpoof, quality localization, fine-grained frame modeling
- Interspeech 2025 + "deepfake" + "audio": 22+ papers -- PartialEdit, Grinberg diffusion XAI, multiple detection papers
- NeurIPS 2025 + "audio deepfake": 0 results

**Search completed: 2026-03-02. All queries documented above are reproducible.**
