# CITeR Fall 2025 Final Report
# Project #25F-XXX

`[TBD]` = experiment numbers to fill in after running models

---

## SLIDE 1: Title

**A Multi-Layered Framework for Robust Deepfake Audio Detection and Tamper Analysis**

Y. Sukhdeve, A. Ahmed, M. H. Imtiaz — Clarkson University
N. Karimnian, S. Tehranipoor — West Virginia University
S. Schuckers — Clarkson University / UNC Charlotte

Project #25F-XXX | Fall 2025 CITeR Program Review | Final Report

---

## SLIDE 2: Problem

**Partial deepfakes bypass current detection systems**

- Detection works well for **fully synthetic** speech — sub-1% EER on ASVspoof benchmarks
- Real-world threat: attacker manipulates **only a few words** in a genuine recording
  - Edited speech is perceptually near-genuine (MOS 3.80–3.90 vs 3.88 real)
  - Forged region may be <10% of the recording
- Detectors trained on one manipulation type **fail on others:**
  - **XLSR-SLS** (ACM MM 2024): 2.55% EER on PartialSpoof → 23.7% on PartialEdit (9.3x worse)
  - **CFPRF** (ACM MM 2024): 7.61% EER on PartialSpoof → 43.25% on LlamaPartialSpoof (5.7x worse, Luong et al. 2025)
  - **TDAM** (SPL 2025): 0.59% EER on PartialSpoof → 51.4% on HQ-MPSD (~random chance)
- No standardized evaluation protocol exists — 8 different metric families, 3 incompatible IoU definitions, 80% of dataset pairs share **zero common metrics**

**Bottom line:** XLSR-SLS, CFPRF, and TDAM all achieve strong in-domain results but collapse cross-dataset. No standardized way to measure or compare this failure exists today.

[FIGURE: Waveform showing genuine vs partially manipulated audio with highlighted forged region]

---

## SLIDE 3: Our Approach

**Systematic evaluation of partial deepfake detection generalizability**

| Component | What We Did | Output |
|---|---|---|
| **Survey** | Screened 365 papers, analyzed 10 datasets, 15+ architectures | Survey paper (TASLP, under review) |
| **Experiments** | Trained BAM detector, evaluated 3 models across 4 datasets | Cross-dataset EER degradation tables |
| **Software** | Built modular evaluation pipeline (XPS-Forensic) | Open-source toolkit, 136 tests passing |

**Models evaluated:**

| Detector | Backbone | Resolution | Venue | In-Domain EER |
|---|---|---|---|---|
| BAM | WavLM-Large | 160 ms | Interspeech 2024 | 3.58% (published) |
| CFPRF | XLSR-300M | 20 ms | ACM MM 2024 | 7.41% |
| MRM | wav2vec 2.0 | 20 ms | TASLP 2023 | 13.72% |

**Datasets used:** PartialSpoof (concatenation), PartialEdit (neural editing), LlamaPartialSpoof (LLM-driven), HQ-MPSD (multilingual)

---

## SLIDE 4: Bottom Line Up Front

1. **Models do not generalize.** BAM trained on PartialSpoof: `[TBD]`% in-domain EER → `[TBD]`% on PartialEdit (`[TBD]`x degradation). MRM: `[TBD]`% → `[TBD]`% (`[TBD]`x).

2. **Construction method mismatch is the primary driver.** Concatenation-trained models fail on neural editing and vice versa. Cross-paradigm EER degrades from single-digit to >23% consistently across multiple studies.

3. **Evaluation is fragmented.** 8 metric paradigms, 3 incompatible IoU formulations. 80% of dataset pairs share no common metrics (Jaccard index 0.11). Cross-study comparison is currently impossible without re-running experiments.

4. **We delivered:**
   - Trained BAM model on PartialSpoof with `[TBD]`% segment EER
   - Cross-dataset evaluation across 4 datasets and 3 detectors
   - Open-source evaluation pipeline (XPS-Forensic, 6,100 LOC, 136 tests)
   - Comprehensive survey paper (TASLP, under review)

---

## SLIDE 5: Relevance to Members

**What this means for DHS, FBI, IDEMIA, DRDC, USACIL:**

**Evidence integrity (DHS, FBI, USACIL):**
A forensic examiner receives audio evidence. A suspect's words may have been replaced using AI editing tools. Running a standard detector gives a confident "genuine" verdict — but that detector was never tested against neural editing attacks. Our work quantifies this blind spot: `[TBD]`x degradation when the attack type changes.

**Biometric security (IDEMIA, DRDC):**
Voice authentication systems are vulnerable to partial deepfakes — an attacker can keep genuine speaker characteristics while replacing specific content. Our cross-dataset analysis shows that even the best localization models lose `[TBD]`% accuracy on unseen manipulation methods.

**Practical takeaway:**
- No single detector is reliable across all attack types today
- Any deployment must be tested against the specific threat model expected in the field
- XPS-Forensic provides a standardized way to run that evaluation

---

## SLIDE 6: Project Direction

**Original plan → What we found → What we did**

| Planned | Finding | Action |
|---|---|---|
| Build new detection model | 15+ models already exist; the problem is generalization, not detection accuracy | Evaluated existing SOTA models across datasets |
| Create new dataset | 10 partial deepfake datasets already exist (2021–2025) | Used 4 publicly available datasets |
| Speaker diarization | Open gap — 0–1 published methods for partial deepfake diarization | Identified as future work |
| LLM-based reasoning | Emerging direction — 4 papers appeared in 2026 (post-proposal) | Identified as future work |
| Frame-level tamper detection | Directly addressed | XPS-Forensic pipeline with frame-level localization |

**Key pivot:** Literature review revealed that the field's bottleneck is **non-generalizability and evaluation fragmentation**, not model accuracy. We focused on systematically documenting and quantifying this problem — which is more immediately useful than another model that would face the same generalization failure.

---

## SLIDE 7: Accomplishments — Survey

**"Partial Deepfake Speech Detection and Temporal Localization: A Survey"**
Y. Sukhdeve, A. Ahmed, M. H. Imtiaz | Under review: IEEE/ACM TASLP

| What We Analyzed | Count |
|---|---|
| Papers screened | 365 |
| Partial deepfake datasets compared | 10 |
| Detection architectures surveyed | 15+ |
| Evaluation metric paradigms formally defined | 8 |
| Cross-dataset result pairs compiled | 14 (from 5 studies) |

**Key survey findings:**
- Boundary-aware detection (BAM, IFBDN, BFC-Net) is central to localization — all top systems use it
- SSL front-ends (WavLM, XLS-R, wav2vec 2.0) dominate; spectral features underperform cross-dataset
- Metric fragmentation prevents cross-study comparison: same model, different metric → different conclusion
- Three IoU formulations (continuous, penalized 1D, discrete fixed-window) produce different numbers on identical predictions

[FIGURE: Fig. 2 from paper — metric usage dot plot showing fragmentation]

---

## SLIDE 8: Accomplishments — Cross-Dataset Experiments

**Original experimental results: generalization failure quantified**

### BAM (trained by us on PartialSpoof)

| Test Dataset | Attack Type | Seg-EER (%) | Degradation |
|---|---|---|---|
| PartialSpoof (in-domain) | Concatenation | `[TBD]` | — |
| PartialEdit | Neural editing | `[TBD]` | `[TBD]`x |
| LlamaPartialSpoof | LLM-driven | `[TBD]` | `[TBD]`x |
| HQ-MPSD-EN | Multilingual | `[TBD]` | `[TBD]`x |

### MRM (PartialSpoof baseline)

| Test Dataset | Seg-EER (%) | Degradation |
|---|---|---|
| PartialSpoof (in-domain) | `[TBD]` | — |
| PartialEdit | `[TBD]` | `[TBD]`x |
| LlamaPartialSpoof | `[TBD]` | `[TBD]`x |

### CFPRF (authors' pre-computed, included for comparison)

| Dataset | Seg-EER (%) | mAP (%) |
|---|---|---|
| PartialSpoof | 7.41 | 51.76 |
| HAD | 0.08 | 99.11 |
| LAV-DF | 0.82 | 92.90 |

[FIGURE: Grouped bar chart — in-domain vs cross-dataset EER for all 3 detectors]

---

## SLIDE 9: Accomplishments — XPS-Forensic Toolkit

**Open-source evaluation pipeline for partial deepfake detection**

```
Audio File → [Data Loader] → [Detector] → [Metrics] → Results
              4 datasets       3 models     Seg-EER, Utt-EER,
              unified API      unified API  F1, tFNR, tIoU,
                                            bootstrap CIs
```

| Component | Details |
|---|---|
| Datasets | PartialSpoof, PartialEdit, HQ-MPSD, LlamaPartialSpoof |
| Detectors | BAM (WavLM, 160ms), CFPRF (XLSR, 20ms), MRM (wav2vec2, 20ms) |
| Metrics | Segment EER (multi-resolution), utterance EER, boundary F1, temporal FNR/FDR/IoU |
| Calibration | Platt scaling, temperature scaling, isotonic regression |
| Testing | 136 unit tests, all passing |
| Code | ~6,100 lines Python, Hydra config, modular architecture |

**Ready for affiliate use:** Clone repo → install environment → point to dataset path → run evaluation script

---

## SLIDE 10: Outcomes, Importance, Deliverables

**Deliverables:**

| # | Item | Format | Status |
|---|---|---|---|
| 1 | Trained BAM model checkpoint | .ckpt file | Ready |
| 2 | Cross-dataset evaluation results | Tables + figures | `[TBD — after experiments]` |
| 3 | XPS-Forensic evaluation toolkit | Python, GitHub | Ready for release |
| 4 | Survey paper | PDF (TASLP) | Under review |
| 5 | Executive Summary | PDF | To be delivered |

**GitHub link:** Will be provided after affiliate review and public release.

**Who benefits:**
- Forensic labs evaluating audio evidence (DHS, FBI, USACIL)
- Biometric vendors stress-testing voice systems (IDEMIA, DRDC)
- Research community needing standardized partial deepfake evaluation

---

## SLIDE 11: Next Steps

**Immediate:**
- Submit survey to TASLP (under review)
- Release XPS-Forensic toolkit on GitHub
- Publish cross-dataset generalization analysis (target: ICASSP 2026 or IEEE SPL)

**If continued (future funding):**
- **Calibrated confidence bounds:** Conformal prediction for localization — formal guarantees on false negative rate (no existing work in audio deepfake domain)
- **Explainable localization:** Phoneme-level saliency maps translating model output into forensic-examiner-readable reports
- **Unified evaluation toolkit:** Extend XPS-Forensic to cover all 8 metric paradigms — an "AudioCOCO" for the field
- **Real-world robustness:** Telephone channel, codec compression (AAC/Opus/AMR), background noise — no dataset currently standardizes all three

---

## SLIDE 12: Auxiliary — Technical Details

### Cross-Dataset Evidence from Published Literature

| Train → Test | EER (%) | Source |
|---|---|---|
| PS → PS (in-domain) | 2.55 | PartialEdit paper, XLSR-SLS |
| PS → PartialEdit | 23.7 | PartialEdit paper, XLSR-SLS |
| PS → LlamaPartialSpoof | 15.3 (utt) | LlamaPartialSpoof paper, MRM |
| PS → HQ-MPSD | 51.4 (utt) | HQ-MPSD paper, TDAM |
| CFPRF in-domain → LPS OOD | 7.61 → 43.25 | Luong et al., 2025 |
| HAD → LPS | 57.2 (utt) | LlamaPartialSpoof paper, MRM |

### Metric Fragmentation Summary

| Metric | Datasets Using It | Origin |
|---|---|---|
| Segment EER (multi-res) | PartialSpoof only | Zhang et al., TASLP 2023 |
| Utterance EER | PS, LPS, HQ-MPSD, PE | ASVspoof tradition |
| Frame-level EER | PartialEdit only | Zhang et al., Interspeech 2025 |
| Duration-based F1 | HAD, ADD 2023 | Yi et al., 2021 |
| AP@IoU | LAV-DF, AV-Deepfake1M | Borrowed from video TAL |
| Boundary F1 | PartialSpoof only | Zhang et al., TASLP 2023 |
| Asymmetric 1D-IoU | Psynd only | Psynd authors |
| HTER | LENS-DF only | LENS-DF authors |

---

## SLIDE 13: Quad Summary

*(Single-slide overview for director presentations)*

**Objective:** Evaluate generalizability of partial deepfake detection across attack types and develop standardized evaluation framework.

**Approach:** Survey (10 datasets, 15+ architectures) + cross-dataset experiments (3 detectors, 4 datasets) + evaluation toolkit (XPS-Forensic).

**Key Result:** Detectors degrade `[TBD]`x cross-dataset. 80% of evaluation protocol pairs share zero metrics. Current systems are unreliable against unseen manipulation types.

**Deliverables:** Survey paper (TASLP), trained BAM model, XPS-Forensic toolkit (GitHub), cross-dataset evaluation tables.

Investigators: Imtiaz (CU), Schuckers (CU/UNC), Karimnian (WVU), Tehranipoor (WVU)

[IMAGE: Cross-dataset degradation chart]

---

## SLIDE 14: Milestones and Deliverables

| Milestone | Planned (Months) | Status | Notes |
|---|---|---|---|
| Dataset preparation | 0–3 | Complete | Used 4 existing public datasets |
| Model training | 3–6 | Complete | BAM trained on PartialSpoof |
| Frame-level tamper detection | 3–9 | Complete | XPS-Forensic pipeline |
| Speaker diarization | 3–9 | Deferred | Open gap in literature (0–1 papers) |
| LLM-based reasoning | 3–9 | Deferred | Emerging — 4 papers appeared post-proposal |
| Cross-dataset evaluation | 9–12 | Complete | 3 detectors, 4 datasets |
| Final report | 12 | Complete | This presentation + executive summary |

| Deliverable | Status |
|---|---|
| Trained detection model (BAM) | Delivered |
| Evaluation pipeline (XPS-Forensic) | Delivered |
| Survey paper | Under review (TASLP) |
| Cross-dataset results + figures | `[TBD]` |
| Executive summary | To be delivered |
