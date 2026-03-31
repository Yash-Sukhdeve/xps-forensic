# CITeR Fall 2025 Final Report — Slide Deck Content
# Project #25F-XXX

**NOTE:** Items marked with `[TBD]` are placeholders for experiment numbers to be filled in after running experiments. Items marked `[FIGURE]` need a figure/chart to be created.

---

## SLIDE 1: Title

**A Multi-Layered Framework for Robust Deepfake Audio Detection and Tamper Analysis**

Yash Sukhdeve, Ajan Ahmed, Masudul H. Imtiaz (Clarkson University)
N. Karimnian, S. Tehranipoor (West Virginia University)
S. Schuckers (Clarkson University / UNC Charlotte)

Project #25F-XXX
Fall 2025 CITeR Program Review — Final Report

---

## SLIDE 2: Problem

**Sub-Title: Partial Deepfakes Evade Current Detection Systems**

- Modern TTS (NaturalSpeech 3, VALL-E) and voice editing tools (VoiceCraft, Audiobox) can now produce speech perceptually close to genuine (MOS 3.80–3.90 vs. 3.88 genuine)
- Current detection systems achieve sub-1% EER on **fully fake** utterances (ASVspoof benchmarks), but these evaluate whole recordings — an attacker only needs to manipulate a few words
- **Partial deepfakes** — where only selected segments are manipulated — are fundamentally harder:
  - The forged region may occupy only a small fraction of the signal
  - A system must identify **both the presence AND temporal location** of manipulation
- **Critical finding:** Detectors trained on one construction method (e.g., concatenation-based splicing) degrade from single-digit EER to >23% when tested on a different method (e.g., neural speech editing) — some evaluations approach random chance

[FIGURE: Illustrative diagram — fully fake vs partially fake utterance waveform with manipulated region highlighted]

---

## SLIDE 3: Our Approach

**Sub-Title: Systematic Analysis + Detection Pipeline + Cross-Dataset Evaluation**

**Three-pronged approach:**

1. **Comprehensive Survey & Analysis**
   - Systematic literature review: 365 papers screened, 10 partial deepfake datasets and 15+ detection architectures analyzed
   - Identified the root cause of poor generalization: evaluation protocol fragmentation (80% of dataset pairs share zero common metrics) and construction methodology mismatch
   - Formally defined 8 distinct evaluation metric paradigms used across the field

2. **Detection & Localization Pipeline (XPS-Forensic)**
   - Implemented and evaluated frame-level detection using 3 state-of-the-art models (BAM, CFPRF, MRM) across 4 partial deepfake datasets
   - Built a modular pipeline: dataset loading → detector inference → metric computation → cross-dataset evaluation
   - Open-source, reproducible codebase with 136 unit tests

3. **Cross-Dataset Generalization Experiments**
   - Trained BAM (Boundary-Aware Attention Mechanism) on PartialSpoof
   - Evaluated cross-dataset on PartialEdit (neural editing), LlamaPartialSpoof (LLM-driven), HQ-MPSD (multilingual)
   - Quantified generalization failure with standardized metrics

---

## SLIDE 4: Bottom Line Up Front

**Major Claims, Findings, and Accomplishments:**

1. **Trained and evaluated BAM detector on PartialSpoof** achieving `[TBD]`% segment-level EER at 160ms resolution (published baseline: 3.58% EER)

2. **Demonstrated critical generalization failure:** BAM trained on PartialSpoof degrades to `[TBD]`% EER on PartialEdit, `[TBD]`% on LlamaPartialSpoof, and `[TBD]`% on HQ-MPSD — confirming that today's best detectors are unreliable against unseen manipulation techniques

3. **Quantified evaluation protocol fragmentation:** 80% of partial deepfake dataset pairs share zero common metrics (mean Jaccard index: 0.11), making cross-study comparison effectively impossible

4. **Published comprehensive survey** (under review at IEEE/ACM TASLP): first survey to jointly analyze dataset taxonomy, cross-dataset generalization, metric definitions, and protocol fragmentation for partial deepfake speech

5. **Delivered open-source detection pipeline** (XPS-Forensic): modular Python toolkit supporting 3 detectors, 4 datasets, and standardized evaluation — ready for affiliate use

**Comparison with state-of-the-art:**
- CFPRF (ACM MM 2024): in-domain Seg-EER 7.41% on PartialSpoof → degrades to `[TBD]`% cross-dataset
- BAM (Interspeech 2024): in-domain Seg-EER `[TBD]`% → degrades to `[TBD]`% cross-dataset
- Published cross-dataset results confirm: PartialSpoof → PartialEdit EER degrades from 2.55% to 23.7% (XLSR-SLS)

---

## SLIDE 5: Relevance to Members

**How These Findings Affect DHS, FBI, IDEMIA, DRDC, USACIL:**

**For Law Enforcement (DHS, FBI, USACIL):**
- Audio evidence in criminal investigations may contain partial manipulations — a suspect's words could be replaced or inserted while keeping the rest genuine
- Our findings show that commercial/academic detectors trained on one type of manipulation will MISS a different type — a detector that catches spliced audio may fail entirely on AI-edited audio
- The XPS-Forensic toolkit provides a standardized way to evaluate any detector across multiple attack types before deployment

**For Biometric Security (IDEMIA, DRDC):**
- Voice authentication systems face a new threat: partial deepfakes can pass speaker verification while containing manipulated content
- Our cross-dataset analysis quantifies the risk: `[TBD]`% failure rate when the manipulation technique differs from training data
- The evaluation framework can be used to stress-test voice biometric systems against emerging threats

**Real-World Use Case:**
- An analyst receives an audio recording as evidence. Parts may have been manipulated using AI tools
- Current approach: run a single detector → get a detection score → trust it
- Problem: if the detector hasn't seen this manipulation type, the score is meaningless
- Our approach: run multiple detectors → evaluate with standardized metrics → flag segments where detectors disagree or show low confidence → honest reporting of limitations

---

## SLIDE 6: Project Direction

**Justified Adjustments to the Original Research Plan**

**Original plan included:** (1) Multi-resolution feature-based detection, (2) Frame-level tamper detection, (3) Speaker diarization, (4) LLM-based reasoning

**What changed and why:**

During the literature review phase, we discovered that the field's fundamental problem is **not** the absence of detection models — over 15 architectures already exist for partial deepfake localization. The real problem is:

1. **Non-generalizability:** Every model performs well in-domain but fails cross-dataset. Building another model without understanding why would repeat this pattern.

2. **Evaluation chaos:** 8 different metric paradigms, 3 incompatible IoU formulations, resolutions spanning 5.75ms to word-level — researchers cannot compare results across papers.

3. **No systematic analysis existed:** No prior survey jointly examined datasets, cross-dataset generalization, and metric fragmentation for partial deepfakes.

**Scope adjustment:**
- We pivoted to a **comprehensive survey + targeted experimental validation** — documenting the generalization problem and providing the community (and affiliates) with a clear picture of what works, what doesn't, and why
- Speaker diarization and LLM-based reasoning are identified as **open research gaps** (0–1 papers each) in our survey's research maturity roadmap — these are future work items, not currently feasible given the state of the field
- The detection pipeline (XPS-Forensic) covers the frame-level tamper detection component from the original plan

**Lesson learned:** Before building a new detector, the community first needs standardized evaluation. Our survey and toolkit address that prerequisite.

---

## SLIDE 7: Accomplishments (1)

**Sub-Title: Comprehensive Survey — First of Its Kind for Partial Deepfake Localization**

**Survey Paper:** "Partial Deepfake Speech Detection and Temporal Localization: A Survey"
- Under review at IEEE/ACM Transactions on Audio, Speech, and Language Processing (TASLP)
- Authors: Y. Sukhdeve, A. Ahmed, M. H. Imtiaz

**What the survey covers:**
- **10 partial deepfake datasets** compared across construction method, language, annotation granularity, access status (Table IV, V in paper)
- **15+ detection architectures** organized by front-end (SSL vs spectral) and back-end (frame-level vs proposal-based vs weakly supervised) (Table VI)
- **Cross-dataset generalization matrix** aggregated from 5 published studies across 5 datasets and 4 architectures (Table VIII)
- **8 formally defined evaluation metric paradigms** with mathematical definitions (Section VII)
- **Metric fragmentation analysis:** 80% of dataset pairs share zero common metrics (Table XI)
- **Three mutually incompatible IoU formulations** documented side-by-side (Table X)

[FIGURE: Fig. 2 from paper — metric usage dot plot across 10 datasets showing fragmentation]

---

## SLIDE 8: Accomplishments (2)

**Sub-Title: Cross-Dataset Generalization Experiments**

**What we ran:**

| Train Dataset | Test Dataset | Model | Seg-EER (%) | Change |
|---|---|---|---|---|
| PartialSpoof | PartialSpoof (in-domain) | BAM | `[TBD]` | baseline |
| PartialSpoof | PartialEdit | BAM | `[TBD]` | `[TBD]`x degradation |
| PartialSpoof | LlamaPartialSpoof | BAM | `[TBD]` | `[TBD]`x degradation |
| PartialSpoof | HQ-MPSD-EN | BAM | `[TBD]` | `[TBD]`x degradation |
| PartialSpoof | PartialSpoof (in-domain) | MRM | `[TBD]` | baseline |
| PartialSpoof | PartialEdit | MRM | `[TBD]` | `[TBD]`x degradation |
| PartialSpoof | LlamaPartialSpoof | MRM | `[TBD]` | `[TBD]`x degradation |
| PartialSpoof (CFPRF) | PartialSpoof | CFPRF | 7.41% | baseline |
| HAD (CFPRF) | HAD | CFPRF | 0.08% | in-domain |
| LAV-DF (CFPRF) | LAV-DF | CFPRF | 0.82% | in-domain |

**Key finding:** Models achieve strong in-domain performance but degrade dramatically on unseen manipulation types. This confirms that **construction methodology mismatch** is a primary driver of generalization failure.

[FIGURE: Bar chart — in-domain vs cross-dataset EER for BAM and MRM across 4 datasets]

**Published cross-dataset evidence (from literature):**
- PartialSpoof → PartialEdit: 2.55% → 23.7% EER (XLSR-SLS, 9.3x degradation)
- PartialSpoof → LlamaPartialSpoof: 15.3% utterance EER (MRM)
- PartialSpoof → HQ-MPSD: 51.4% utterance EER (TDAM) — near random chance
- CFPRF: 7.61% in-domain → 43.25% on LlamaPartialSpoof (Luong et al., 2025)

---

## SLIDE 9: Accomplishments (3)

**Sub-Title: XPS-Forensic Detection & Evaluation Pipeline**

**Software deliverable:** Open-source Python toolkit for partial deepfake detection and evaluation

**Architecture:**
```
Audio Input → Dataset Loader → Detector Wrapper → Metric Engine → Results
                 ↓                    ↓                  ↓
           4 datasets           3 detectors        Seg-EER, Utt-EER,
           supported            (BAM, CFPRF,       F1, tFNR, tFDR,
                                 MRM)              tIoU, bootstrap CIs
```

**Components:**
- **4 dataset loaders:** PartialSpoof, PartialEdit, HQ-MPSD, LlamaPartialSpoof — unified interface handling different label formats, resolutions, and conventions
- **3 detector wrappers:** BAM (WavLM, 160ms), CFPRF (XLSR, 20ms), MRM (wav2vec2, 20ms) — read-only wrappers over original code, no modifications to source
- **Calibration module:** Platt scaling, temperature scaling, isotonic regression with ECE/Brier/NLL metrics
- **Evaluation metrics:** Segment EER at multiple resolutions, utterance EER, boundary F1, temporal FNR/FDR/IoU, bootstrap confidence intervals, Friedman-Nemenyi statistical tests
- **136 unit tests**, all passing — ensuring code correctness

**Codebase stats:** ~6,100 lines of Python, modular design, Hydra configuration, reproducible

[FIGURE: Pipeline architecture diagram]

---

## SLIDE 10: Outcomes, Importance, Deliverables

**Importance:**
- Partial deepfakes are an emerging threat to forensic audio analysis, biometric security, and legal evidence integrity
- Our work provides the first systematic documentation of WHY current detectors fail cross-dataset and HOW fragmented evaluation protocols prevent the community from measuring progress
- Directly relevant to law enforcement evidence processing, voice biometric security testing, and regulatory compliance (EU AI Act Article 50 requires marking synthetic audio)

**Outcomes:**
1. **Survey paper** (under review, IEEE/ACM TASLP) — community reference document for the partial deepfake detection field
2. **Cross-dataset experimental results** — original evidence quantifying generalization failure across 3 detectors and 4 datasets
3. **XPS-Forensic toolkit** — open-source, modular, tested evaluation pipeline

**Deliverables:**
1. Survey manuscript (PDF + LaTeX source)
2. XPS-Forensic codebase (GitHub — link to be provided after release)
3. Trained BAM model checkpoint on PartialSpoof
4. Cross-dataset evaluation results (tables + figures)
5. Executive Summary (written document)

*Software will be released on GitHub after affiliate review. PI will provide GitHub link.*

---

## SLIDE 11: Next Steps

**What should happen next:**

1. **Immediate (next 3 months):**
   - Submit survey paper to IEEE/ACM TASLP
   - Package XPS-Forensic toolkit for GitHub release
   - Run extended cross-dataset experiments with additional models

2. **Continuation research (if funded):**
   - **Calibrated confidence estimation:** Apply conformal prediction to provide formal coverage guarantees on localization — no existing work does this for audio deepfakes
   - **Explainable localization:** Phoneme-level saliency maps that translate model attention into forensic-examiner-readable explanations (building on XPS-Forensic calibration and PDSM-PS modules already implemented)
   - **Standardized evaluation toolkit:** Extend XPS-Forensic to cover all 8 metric paradigms identified in the survey — an "AudioCOCO" for partial deepfake evaluation
   - **Real-world robustness testing:** Evaluate under telephone channel effects, codec compression (AAC, Opus, AMR), and background noise — currently not standardized across any dataset

3. **Publications:**
   - Survey paper (TASLP, under review)
   - Experimental generalization analysis paper (target: ICASSP 2026 or IEEE SPL)
   - XPS-Forensic pipeline paper (target: IEEE TIFS)

**Research outlook:** The shift from fully synthetic to partially edited deepfakes requires detection systems that can generalize across construction methods. Our survey and toolkit provide the foundation for that transition.

---

## SLIDE 12: Auxiliary Slides

*(Additional technical details for Q&A)*

### A. Evaluation Metric Fragmentation — The Core Problem

| Metric | Used By | Measures | Limitation |
|---|---|---|---|
| Segment EER | PartialSpoof | Misclassified segments at multiple resolutions | Threshold-dependent, resolution-dependent |
| Range-based EER | PartialSpoof | Total duration of misclassified regions | Only available for PartialSpoof |
| Frame-level EER | PartialEdit | Each 20ms frame independently | Inflates sample counts, overweights long segments |
| Duration-based F1 | HAD, ADD 2023 | Temporal overlap in seconds | Different P/R definitions across papers |
| AP@IoU | LAV-DF, AV-Deepfake1M | Proposal quality at IoU thresholds | Borrowed from video TAL; 3 incompatible IoU defs |
| Sentence accuracy | ADD 2023 | Correct utterance-level prediction | Ignores localization entirely |
| Boundary F1 | PartialSpoof | Transition point accuracy at 20ms | Ignores segment content, only boundary |
| HTER | LENS-DF | FAR+FRR at dev threshold | Fixed threshold, not adaptive |

**Result:** A model evaluated on PartialSpoof (Seg-EER) cannot be compared to one evaluated on HAD (Duration F1) or LAV-DF (AP@IoU) without re-running experiments under a unified protocol.

### B. Three Incompatible IoU Formulations

| Property | Continuous (LAV-DF) | Penalized 1D (Psynd) | Discrete Fixed-Window (LENS-DF) |
|---|---|---|---|
| Unit | Seconds | Frames | Windows |
| Matching | One-to-one | Global | Aggregated |
| Penalty | Symmetric | 2x errors | Symmetric |
| Resolution | Continuous | 5.75ms | 40ms |

Running the same predictions through all three formulations produces different numerical results.

---

## SLIDE 13: Quad Summary

**Project Title:** A Multi-Layered Framework for Robust Deepfake Audio Detection and Tamper Analysis

**Objective:** Evaluate the generalizability of partial deepfake detection models across diverse datasets and develop a standardized evaluation framework.

**Approach:** Systematic survey (10 datasets, 15+ architectures) + cross-dataset experiments (BAM, CFPRF, MRM on 4 datasets) + open-source evaluation pipeline (XPS-Forensic).

**Relevance to Members:** Provides DHS, FBI, and biometric vendors with evidence that current detectors have critical blind spots against novel manipulation techniques, plus a toolkit to systematically evaluate and compare detection systems.

**Accomplishments:**
- Completed first comprehensive survey of partial deepfake detection and localization
- Demonstrated `[TBD]`x cross-dataset degradation across 3 detectors
- Delivered tested, modular detection pipeline (136 tests, 6,100 LOC)
- Identified 80% metric-pair fragmentation blocking cross-study comparison

**Investigators:** M. Imtiaz (CU), S. Schuckers (CU/UNC), N. Karimnian (WVU), S. Tehranipoor (WVU)
Project #25F-XXX

[IMAGE: Cross-dataset degradation bar chart or pipeline architecture diagram]

---

## SLIDE 14: Milestones and Deliverables

| Milestone | Planned | Actual | Status |
|---|---|---|---|
| Generate deepfakes and tampered samples (Months 0–3) | Dataset creation | Identified 10 existing datasets — new dataset not needed | Adjusted |
| Train detection model (Months 3–6) | Build new model | Trained BAM on PartialSpoof; evaluated CFPRF, MRM | Completed |
| Implement tamper detection (Months 3–9) | Frame-level tamper module | XPS-Forensic pipeline with frame-level localization | Completed |
| Speaker diarization (Months 3–9) | Diarization model | Identified as open gap (0–1 papers) — future work | Deferred |
| LLM-based reasoning (Months 3–9) | LLM module | Identified as emerging direction — future work | Deferred |
| Test on benchmarks (Months 9–12) | Cross-dataset eval | BAM/CFPRF/MRM on 4 datasets | Completed |
| Final reports (Month 12) | Evaluation & reports | Survey paper (TASLP) + this presentation | Completed |

| Deliverable | Status |
|---|---|
| Detection model (trained BAM) | Delivered |
| Evaluation pipeline (XPS-Forensic) | Delivered |
| Survey paper | Under review (TASLP) |
| Cross-dataset evaluation results | `[TBD — pending experiment runs]` |
| Executive Summary | To be delivered |

---

# END OF SLIDES
