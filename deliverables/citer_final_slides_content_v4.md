# CITeR Final Report — Slide Content (copy-paste into template)
# Match the style of your ELAD presentation

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

- Current detectors work well on fully synthetic speech (sub-1% EER on ASVspoof) but fail on partial deepfakes where only a few words are manipulated.
- Neural editing tools (VoiceCraft, Audiobox) produce edits that sound nearly identical to genuine speech.
- Detectors trained on one attack type fail on others — EER degrades from single-digit to over 23% and in some cases to near random chance.
- No standardized evaluation protocol exists across datasets — 8 different metric families, 80% of dataset pairs share no common metrics.

[FIGURE: fig1_taxonomy.png on right side]

---

## Slide 3: Bottom Line Up Front

- Completed a survey of partial deepfake detection covering 10 datasets and over 15 detection methods. Survey submitted to IEEE/ACM TASLP.
- Trained BAM detector on PartialSpoof dataset and evaluated cross-dataset generalization on PartialEdit, LlamaPartialSpoof, and HQ-MPSD.
- BAM achieves [  ]% segment EER in-domain but degrades to [  ]% on PartialEdit and [  ]% on LlamaPartialSpoof.
- Documented that 80% of dataset pairs share no common evaluation metrics, making cross-study comparison impossible.
- Built an evaluation pipeline supporting 3 detectors and 4 datasets with standardized metrics.

---

## Slide 4: Relevance to Members

- Biometric Security: Cross-dataset results show that detectors trained on one manipulation type miss others, which affects voice authentication system reliability.
- Forensics: Audio evidence may contain partial AI edits that current detectors are not tested against. Our results quantify this gap.
- Evaluation: The 80% metric fragmentation we documented means published results across different datasets cannot be directly compared.
- R&D: Evaluation pipeline and trained models will be available on GitHub for affiliates to test their own systems.

---

## Slide 5: Our Approach

Datasets:
- PartialSpoof — concatenation-based splicing, English, 121K utterances
- PartialEdit — neural speech editing (VoiceCraft, Audiobox), English, 43K utterances
- LlamaPartialSpoof — LLM-driven fake speech, English, 76K utterances
- HQ-MPSD — multilingual with RMS alignment, English subset used

Models:
- BAM — WavLM-Large backbone, 160ms resolution, Interspeech 2024
- CFPRF — XLSR-300M backbone, 20ms resolution, ACM MM 2024
- MRM — wav2vec 2.0 backbone, 20ms resolution, TASLP 2023

[FIGURE: fig2_timeline.png at bottom]

---

## Slide 6: Our Approach (2)

Evaluation Metrics:
- Segment EER at multiple resolutions (20ms to 640ms)
- Utterance EER
- Frame-level EER at 20ms

Calibration Methods:
- Platt scaling, temperature scaling, isotonic regression

Pipeline:
- Audio → Data Loader → Detector → Metrics → Results
- Common interface for all 3 detectors and 4 datasets

---

## Slide 7: Project Direction

- Original plan included building a new model, creating a new dataset, speaker diarization, and LLM-based reasoning.
- During the literature review we found 15+ existing detection models and 10 existing datasets — the problem is not accuracy but generalization.
- Speaker diarization for partial deepfakes has 0–1 published methods — not enough to build on.
- LLM-based reasoning papers appeared in 2026, after our proposal.
- We focused on evaluating cross-dataset generalization of existing models and documenting the evaluation fragmentation problem.

---

## Slide 8: Accomplishments — Survey

"Partial Deepfake Speech Detection and Temporal Localization: A Survey"
Submitted to IEEE/ACM TASLP

- Compared 10 partial deepfake datasets across construction method, language, and evaluation protocol.
- Surveyed over 15 detection architectures organized by front-end and back-end design.
- Formally defined 8 evaluation metric paradigms with mathematical formulations.
- Compiled cross-dataset results from 5 published studies showing consistent generalization failure.
- Documented 3 incompatible IoU formulations and 80% metric fragmentation across dataset pairs.

[FIGURE: fig4a_metric_usage.png and fig4b_metric_overlap.png]

---

## Slide 9: Accomplishments — Cross-Dataset Experiments

BAM — trained on PartialSpoof:
- PartialSpoof (in-domain):       [  ]% Seg-EER
- PartialEdit:                    [  ]% Seg-EER
- LlamaPartialSpoof:              [  ]% Seg-EER
- HQ-MPSD-EN:                     [  ]% Seg-EER

MRM — authors' checkpoint:
- PartialSpoof (in-domain):       [  ]% Seg-EER
- PartialEdit:                    [  ]% Seg-EER
- LlamaPartialSpoof:              [  ]% Seg-EER

CFPRF — authors' results:
- PartialSpoof: 7.41% Seg-EER, 51.76% mAP
- HAD: 0.08% Seg-EER, 99.11% mAP
- LAV-DF: 0.82% Seg-EER, 92.90% mAP

[FIGURE: fig3_generalization_heatmap.png]

---

## Slide 10: Accomplishments — Evaluation Pipeline

- Supports 4 datasets: PartialSpoof, PartialEdit, LlamaPartialSpoof, HQ-MPSD
- Wraps 3 detectors: BAM, CFPRF, MRM
- Computes segment EER at multiple resolutions, utterance EER, frame-level EER
- Includes calibration module (Platt, temperature, isotonic)
- Will be released on GitHub after review

---

## Slide 11: Outcomes, Importance, Deliverables

- Survey paper submitted to IEEE/ACM TASLP
- Trained BAM model on PartialSpoof
- Cross-dataset evaluation results for 3 detectors on 4 datasets
- Evaluation pipeline code (GitHub release after review)
- Executive summary

---

## Slide 12: Next Steps

- Finalize TASLP revision and resubmit
- Release evaluation pipeline on GitHub
- Write up cross-dataset analysis for ICASSP 2026 or IEEE Signal Processing Letters
- If continued: add calibrated confidence bounds, phoneme-level explanations, real-world robustness testing (telephone, codec, noise)

---

## Slide 13: Quad Summary

Objective: Evaluate how partial deepfake detectors generalize across attack types.
Approach: Literature review + cross-dataset experiments with 3 models on 4 datasets + evaluation pipeline.
Relevance: Detectors miss unseen attack types. Results quantify the gap for forensic and biometric applications.
Accomplishments: Survey (TASLP), trained BAM, cross-dataset evaluation, evaluation pipeline.

Y. Sukhdeve (CU), A. Ahmed (CU), M. Imtiaz (CU)
Project #25F-XXX

[IMAGE: fig1_taxonomy.png]

---

## Slide 14: Milestones and Deliverables

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
