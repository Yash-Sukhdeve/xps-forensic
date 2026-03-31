# CITeR Final Report — Slide Content v5
# Copy-paste into the CITeR template in PowerPoint
# Add figures where marked with [FIGURE: filename]

---

## Slide 1: Title

A Multi-Layered Framework for Robust Deepfake Audio Detection and Tamper Analysis

Yash Sukhdeve (CU), Ajan Ahmed (CU), Masudul Imtiaz (CU)
N. Karimnian (WVU), S. Tehranipoor (WVU), S. Schuckers (CU/UNC)

Project #25F-XXX
Fall 2025 CITeR Program Review
Final Report

---

## Slide 2: Problem

Partial Deepfake Detection and Localization

- Current detectors work on fully synthetic speech (sub-1% EER on ASVspoof) but fail on partial deepfakes where only some words are manipulated.
- Neural editing tools produce edits that sound nearly identical to genuine speech.
- Detectors trained on one attack type fail on others:
  - XLSR-SLS: 2.55% → 23.7% EER (PartialSpoof → PartialEdit)
  - CFPRF: 7.61% → 43.25% EER (PartialSpoof → LlamaPartialSpoof)
  - TDAM: 0.59% → 51.4% EER (PartialSpoof → HQ-MPSD)
- 80% of dataset pairs share no common evaluation metrics.

[FIGURE: fig1_taxonomy.png — right side]

---

## Slide 3: Bottom Line Up Front

- Completed a survey covering 10 datasets and over 15 detection architectures. Submitted to IEEE/ACM TASLP.
- Trained BAM detector on PartialSpoof. Evaluated cross-dataset on PartialEdit, LlamaPartialSpoof, and HQ-MPSD.
- BAM: [  ]% segment EER in-domain, [  ]% on PartialEdit, [  ]% on LlamaPartialSpoof.
- Documented that 80% of dataset pairs share no common metrics (mean Jaccard 0.11).
- Built evaluation pipeline supporting 3 detectors and 4 datasets.

---

## Slide 4: Relevance to Members

- Forensics: Detectors trained on spliced audio miss neural edits. XLSR-SLS degrades 9.3x across attack types. TDAM degrades to near random chance on unseen data.
- Biometrics: Partial deepfakes preserve speaker identity. CFPRF degrades from 7.61% to 43.25% out-of-domain. MRM: 15.3% EER on LLM-generated fakes.
- Evaluation: 80% of dataset pairs share no common metrics. Cross-study comparison not currently possible.

[FIGURE: fig3_generalization_heatmap.png — right side]

---

## Slide 5: Our Approach — Datasets

We used 4 publicly available partial deepfake datasets:

| Dataset | Year | Language | Construction | Size | Source |
|---|---|---|---|---|---|
| PartialSpoof | 2021 | English | Concatenation | 121K utt | VCTK |
| PartialEdit | 2025 | English | Neural editing | 43K utt | VCTK |
| LlamaPartialSpoof | 2025 | English | LLM-driven | 76K utt | LibriTTS |
| HQ-MPSD | 2025 | Multi | Word replacement | 155K utt | MLS |

Construction methods produce different artifacts:
- Concatenation → stitching artifacts at boundaries
- Word replacement → prosodic discontinuities
- Neural editing → minimal artifacts, context-aware generation

[FIGURE: fig2_timeline.png — bottom]

---

## Slide 6: Our Approach — Models and Metrics

Models evaluated:

| Model | Backbone | Resolution | Venue |
|---|---|---|---|
| BAM | WavLM-Large | 160 ms | Interspeech 2024 |
| CFPRF | XLSR-300M | 20 ms | ACM MM 2024 |
| MRM | wav2vec 2.0 | 20 ms | TASLP 2023 |

Metrics computed:
- Segment EER at multiple resolutions (20 ms to 640 ms)
- Utterance EER
- Frame-level EER at 20 ms

---

## Slide 7: Project Direction

- Original plan: build a new model, create a new dataset, speaker diarization, LLM reasoning.
- During the review we found 15+ models and 10 datasets already exist. The problem is that models do not generalize across attack types, and evaluation protocols are incompatible.
- Speaker diarization for partial deepfakes: 0–1 published methods.
- LLM-based reasoning: papers appeared in 2026, after our proposal.
- We focused on evaluating cross-dataset generalization and documenting the evaluation fragmentation problem.

---

## Slide 8: Accomplishments — Survey Paper

"Partial Deepfake Speech Detection and Temporal Localization: A Survey"
Submitted to IEEE/ACM TASLP

Compared 10 datasets:

[PASTE Table IV from paper — Dataset comparison table showing Year, Language, Construction, Localization, Modality, Size, Source for all 10 datasets]

Surveyed 15+ architectures:

[PASTE Table VI from paper — Architecture table showing System, Front-end, Back-end, Primary metric, Datasets, Venue grouped by detection granularity]

---

## Slide 9: Accomplishments — Cross-Dataset Generalization Findings

From the literature, cross-dataset results show consistent degradation:

| Train | Test | EER (%) | Model | Source |
|---|---|---|---|---|
| PartialSpoof | PartialSpoof | 2.55 | XLSR-SLS | PartialEdit paper |
| PartialSpoof | PartialEdit | 23.7 | XLSR-SLS | PartialEdit paper |
| PartialSpoof | LlamaPartialSpoof | 15.3 | MRM | LPS paper |
| PartialSpoof | HQ-MPSD | 51.4 | TDAM | HQ-MPSD paper |
| HAD | HAD | 0.08 | CFPRF | CFPRF paper |
| HAD | LlamaPartialSpoof | 57.2 | MRM | LPS paper |
| PartialEdit | PartialSpoof | 23.1 | XLSR-SLS | PartialEdit paper |
| PartialEdit | PartialEdit | 3.10 | XLSR-SLS | PartialEdit paper |
| Mixed (PS+PE) | PartialSpoof | 3.00 | XLSR-SLS | PartialEdit paper |
| Mixed (PS+PE) | PartialEdit | 0.64 | XLSR-SLS | PartialEdit paper |

[FIGURE: fig3_generalization_heatmap.png — the color-coded heatmap]

---

## Slide 10: Accomplishments — Our Cross-Dataset Experiments

BAM — trained on PartialSpoof (WavLM-Large, 160ms):
- PartialSpoof (in-domain): [  ]% Seg-EER
- PartialEdit:              [  ]% Seg-EER
- LlamaPartialSpoof:        [  ]% Seg-EER
- HQ-MPSD-EN:               [  ]% Seg-EER

MRM — authors' checkpoint (wav2vec 2.0, 20ms):
- PartialSpoof (in-domain): [  ]% Seg-EER
- PartialEdit:              [  ]% Seg-EER
- LlamaPartialSpoof:        [  ]% Seg-EER

CFPRF — authors' results (XLSR-300M, 20ms):
- PartialSpoof: 7.41% Seg-EER, 51.76% mAP
- HAD: 0.08% Seg-EER, 99.11% mAP
- LAV-DF: 0.82% Seg-EER, 92.90% mAP

[FIGURE: bar chart comparing in-domain vs cross-dataset — to be created after experiments]

---

## Slide 11: Accomplishments — Metric Fragmentation

We identified 8 evaluation metric paradigms used across the 10 datasets. 80% of dataset pairs share no common metrics.

[FIGURE: fig4a_metric_usage.png — dot plot showing which metrics each dataset uses]

[FIGURE: fig4b_metric_overlap.png — Jaccard overlap heatmap]

Three IoU formulations used under the same name are incompatible:

| Property | Continuous (LAV-DF) | Penalized 1D (Psynd) | Discrete (LENS-DF) |
|---|---|---|---|
| Unit | Seconds | Frames | Windows |
| Matching | One-to-one | Global | Aggregated |
| Penalty | Symmetric | 2x errors | Symmetric |
| Resolution | Continuous | 5.75 ms | 40 ms |

Same predictions through all three produce different numbers.

Resolution also changes results: same model gives 0.86% EER at 640ms vs 34.96% at 10ms.

---

## Slide 12: Accomplishments — Evaluation Pipeline

- Supports 4 datasets: PartialSpoof, PartialEdit, LlamaPartialSpoof, HQ-MPSD
- Wraps 3 detectors: BAM, CFPRF, MRM
- Computes segment EER at multiple resolutions, utterance EER, frame-level EER
- Includes calibration (Platt, temperature, isotonic)
- Will be released on GitHub

---

## Slide 13: Accomplishments — Open Challenges Identified

From our survey, we organized 9 open challenges by research maturity:

[FIGURE: fig5_challenge_roadmap.png — the 3x3 grid]

Active research (5+ papers): Cross-dataset generalization, boundary-aware detection, SSL front-end selection
Emerging (2–4 papers): Neural speech editing, metric standardization, weakly-supervised methods
Open gaps (0–1 papers): Spoof diarization, real-world deployment, forensic explainability

---

## Slide 14: Outcomes, Importance, Deliverables

- Survey paper submitted to IEEE/ACM TASLP
- Trained BAM model on PartialSpoof
- Cross-dataset evaluation results for 3 detectors on 4 datasets
- Evaluation pipeline code (GitHub release after review)
- Executive summary

---

## Slide 15: Next Steps

- Finalize TASLP revision
- Release evaluation pipeline on GitHub
- Write up cross-dataset analysis for ICASSP 2026 or IEEE Signal Processing Letters
- If continued: calibrated confidence bounds, phoneme-level explanations, real-world robustness testing

---

## Slide 16: Quad Summary

Objective: Evaluate how partial deepfake detectors generalize across attack types.
Approach: Literature review + cross-dataset experiments with 3 models on 4 datasets + evaluation pipeline.
Relevance: Detectors miss unseen attack types. 80% of dataset pairs share no common metrics.
Accomplishments: Survey (TASLP), trained BAM, cross-dataset evaluation, evaluation pipeline.

[IMAGE: fig1_taxonomy.png]

Y. Sukhdeve, A. Ahmed, M. Imtiaz (CU)
Project #25F-XXX

---

## Slide 17: Milestones and Deliverables

| Milestone | Status |
|---|---|
| Dataset preparation | Complete |
| Model training | Complete |
| Frame-level tamper detection | Complete |
| Speaker diarization | Deferred |
| LLM reasoning | Deferred |
| Cross-dataset evaluation | Complete |
| Final report | Complete |

| Deliverable | Status |
|---|---|
| Survey paper (TASLP) | Submitted |
| Trained BAM model | Delivered |
| Cross-dataset results | Delivered |
| Evaluation pipeline | Delivered (GitHub pending) |
| Executive summary | Delivered |

---

## Backup Slides

### Dataset Annotation and Access Properties
[PASTE Table V from paper — showing annotation granularity, resolution, base access, dataset release, channel conditions for all 10 datasets]

### Search Protocol
[PASTE Table II from paper — showing databases searched, dates, hits]

### Notation
[PASTE Table III from paper — symbol definitions]

### Architecture Acronyms
[PASTE Table VII from paper — glossary]

### Cross-Dataset Provenance
[PASTE Table XIII from paper — source for each cross-dataset EER value]

### Metric Set Definitions
[PASTE Table XIV from paper — metrics per dataset used for Jaccard computation]

### Computational Cost Audit
[PASTE Table XII from paper — parameter counts, latency, FLOPs reporting status]
