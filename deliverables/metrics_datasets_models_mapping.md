# Complete Mapping: Metrics ↔ Datasets ↔ Models

**For:** CITeR final report preparation and committee discussions
**Sources:** Survey paper (Sukhdeve et al.), Tables IV-VI, VIII, X, XI, XIV; metric analysis agents; primary papers

---

## PART 1: Dataset → Metric Mapping

Which metric does each dataset use as its PRIMARY evaluation, and why?

| Dataset | Year | Language | Construction | PRIMARY Metric(s) | Resolution | Why This Metric |
|---|---|---|---|---|---|---|
| **PartialSpoof** | 2021 | English | Concatenation (splice) | Segment EER (6 resolutions), Range-based EER, Boundary F1 | 20–640 ms | First partial spoof dataset; needed sub-utterance evaluation; authors invented segment EER for this purpose |
| **HAD** (Half-Truth) | 2021 | Chinese | Word replacement (TTS) | Utterance EER + Duration-based F1 | Word-level | Chinese word-level manipulation; duration F1 measures temporal overlap in seconds, natural for word-sized regions |
| **ADD 2023 Track 2** | 2023 | Chinese | Word replacement | Sentence accuracy (30%) + Duration F1 (70%) | Word-level | Competition format; weighted composite balances "did you detect it?" (30%) with "where is it?" (70%) |
| **LAV-DF** | 2022 | Multi | Content-driven (AV) | AP@IoU, AR@N, mAP | 40 ms | Audiovisual dataset; borrowed AP@IoU from video temporal action localization (ActivityNet/THUMOS) — natural fit for proposal-based detection |
| **AV-Deepfake1M** | 2024 | Multi | LLM-driven (AV) | AP@IoU, AR@N, mAP | Variable | Same TAL-borrowed metrics as LAV-DF for consistency within AV community |
| **PartialEdit** | 2025 | English | Neural speech editing | Utterance EER + Frame-level EER (20 ms) | 20 ms | Neural editing produces subtle, boundary-less edits; 20ms frame-level EER matches BAM/WavLM hop size |
| **LlamaPartialSpoof** | 2025 | English | LLM + TTS crossfade | Utterance EER only | N/A (utt-level) | Focused on OOD evaluation; no sub-utterance metrics defined yet |
| **HQ-MPSD** | 2025 | Multi (8 langs) | RMS-aligned concat | Utterance EER + AUC | 30 ms | Multilingual focus; utterance-level metrics avoid resolution ambiguity across languages |
| **Psynd** | 2022 | English | Controlled concat | Asymmetric 1D-IoU | 5.75 ms | Finest resolution dataset; custom 1D-IoU penalizes errors 2x to discourage scattered predictions |
| **LENS-DF** | 2025 | English | Concatenation | Segment EER + discrete AP@IoU + HTER | 40 ms | Bridges EER and TAL paradigms; only audio-only dataset using AP@IoU variant |

---

## PART 2: Model → Dataset → Metric Mapping

Which models were evaluated on which datasets, using which metrics?

### Frame-Level Localization Models

| Model | Backbone | Venue | Evaluated On | Metrics Reported | In-Domain EER |
|---|---|---|---|---|---|
| **MRM** | wav2vec 2.0 | TASLP 2023 | PartialSpoof | Seg-EER (6 res) | 13.72% (160ms) |
| **BAM** | WavLM-Large | Interspeech 2024 | PartialSpoof, PartialEdit | Seg-EER (160ms), Frame EER (20ms) | 3.58% (160ms) |
| **IFBDN** | wav2vec 2.0 | CSL 2024 | ADD 2023, PartialSpoof | ADD Score (Sent Acc + Dur F1) | 0.6713 (ADD score) |
| **BFC-Net** | Spectral (LFCC) | Neurocomputing 2025 | PartialSpoof, ADD 2023 | Seg-EER (160ms), ADD Score | 2.73% (160ms) |
| **SAL** | SSL (flexible) | arXiv 2026 | PartialSpoof, HAD, LlamaPartialSpoof | Seg-EER, Utt-EER | 3.00% (160ms) |
| **TDAM** | SSL | SPL 2025 | PartialSpoof, HAD | Utt-EER | 0.59% (utt) |

### Proposal-Based Localization Models

| Model | Backbone | Venue | Evaluated On | Metrics Reported | Key Result |
|---|---|---|---|---|---|
| **CFPRF** | XLSR-300M | ACM MM 2024 | PartialSpoof, HAD, LAV-DF | Seg-EER (20ms), mAP, AP@IoU | 7.41% EER / 51.76% mAP (PS) |
| **UMMAFormer** | TSN + BYOL-A | ACM MM 2023 | LAV-DF, AV-Deepfake1M | mAP, AP@IoU, AR@N | — |
| **BA-TFD+** | 3D-CNN | CVIU 2023 | LAV-DF | mAP, AP@IoU | — |

### Weakly-Supervised Localization Models

| Model | Backbone | Venue | Evaluated On | Metrics Reported | Key Result |
|---|---|---|---|---|---|
| **LOCO** | XLS-R + BERT | IJCAI 2025 | HAD, LAV-DF, AV-Deepfake1M | Frame EER | SOTA on 3 datasets |

### Utterance-Level Models (context — not localization)

| Model | Backbone | Venue | Evaluated On | Metrics Reported | Key Result |
|---|---|---|---|---|---|
| **AASIST** | Spectral (graph attn) | ICASSP 2022 | ASVspoof 2019 LA, LPS | Utt-EER | 0.83% (LA) |
| **XLSR-SLS** | XLS-R 300M | ACM MM 2024 | PartialEdit, SDF Arena | Utt-EER, Frame EER | 2.55% (PS) |
| **Nes2Net** | SSL (nested) | TIFS 2025 | HQ-MPSD | Utt-EER | — |

---

## PART 3: Cross-Dataset Evidence (from published papers)

What happens when you train on one dataset and test on another?

| Train | Test | Model | Metric | In-Domain | Cross-Domain | Degradation | Source |
|---|---|---|---|---|---|---|---|
| PS | PS | XLSR-SLS | Utt-EER | 2.55% | — | baseline | PartialEdit paper |
| PS | PartialEdit | XLSR-SLS | Utt-EER | 2.55% | 23.7% | 9.3x | PartialEdit paper |
| PS | LPS | MRM | Utt-EER | — | 15.3% (partial only) | — | LPS paper |
| PS | HQ-MPSD | TDAM | Utt-EER | — | 51.4% | ~random | HQ-MPSD paper |
| PS | HQ-MPSD | Nes2Net | Utt-EER | — | 57.47% | ~random | HQ-MPSD paper |
| PE | PS | XLSR-SLS | Utt-EER | 3.10% | 23.1% | 7.5x | PartialEdit paper |
| PE | PE | XLSR-SLS | Utt-EER | 3.10% | — | baseline | PartialEdit paper |
| HAD | HAD | CFPRF | Seg-EER | 0.08% | — | baseline | CFPRF paper |
| HAD | LPS | MRM | Utt-EER | — | 57.2% | ~random | LPS paper |
| PS (CFPRF) | PS | CFPRF | Seg-EER | 7.61% | — | baseline | Luong 2025 |
| PS (CFPRF) | LPS | CFPRF | Seg-EER | 7.61% | 43.25% | 5.7x | Luong 2025 |
| PS (CFPRF) | Half-Truth | CFPRF | Seg-EER | 7.61% | 27.59% | 3.6x | Luong 2025 |
| Mixed (PS+PE) | PS | XLSR-SLS | Utt-EER | — | 3.00% | improved | PartialEdit paper |
| Mixed (PS+PE) | PE | XLSR-SLS | Utt-EER | — | 0.64% | improved | PartialEdit paper |

**Key pattern:** Cross-paradigm transfer (concatenation → editing or vice versa) consistently degrades 5–10x. Cross-language (EN → ZH) approaches random chance.

---

## PART 4: Pros and Cons of Each Metric

### 1. Segment EER (PartialSpoof primary)

**What it measures:** At a fixed temporal resolution (e.g., 160ms), divide audio into segments, label as spoof if ANY frame is spoof, compute EER over all segments.

| Pros | Cons |
|---|---|
| Familiar to anti-spoofing community (extension of utterance EER) | **Resolution-dependent:** same model gives 0.86% at 640ms vs 34.96% at 10ms |
| Threshold-free — no need to pick operating point | Threshold-free means you can't deploy it (deployment needs a threshold) |
| Multi-resolution profile gives rich picture of model behavior | "Majority-of-one" labeling: 1 fake frame out of 32 = entire segment labeled spoof |
| Open-source implementation available (partialspoof-metrics) | Only directly comparable when BOTH training and eval resolution match |
| Most widely used for PartialSpoof → largest body of comparable results | Doesn't capture boundary accuracy at all |

**Who uses it:** BAM, SAL, MRM, BFC-Net, CFPRF — all report segment EER on PartialSpoof
**Original paper:** Zhang et al., Interspeech 2021 / TASLP 2023

---

### 2. Range-Based EER (PartialSpoof secondary)

**What it measures:** Total DURATION of misclassified regions, not count of misclassified segments. Eliminates resolution dependency.

| Pros | Cons |
|---|---|
| Resolution-independent — fixes the main flaw of segment EER | Only proposed and evaluated on PartialSpoof — not adopted elsewhere |
| Measures what actually matters: how many SECONDS were wrong | Still threshold-free (same deployment blindness as segment EER) |
| Formally proven to be special case of point-based when resolution → frame duration | More complex to implement (binary search for threshold) |
| Authors showed it fairly evaluates models trained at any resolution | Limited adoption means limited comparative value |

**Who uses it:** Only PartialSpoof evaluation. BAM, SAL report it alongside segment EER.
**Original paper:** Zhang et al., Interspeech 2023

---

### 3. Frame-Level EER (PartialEdit primary)

**What it measures:** Each 20ms frame is an independent sample. Binary classification EER computed over all frames.

| Pros | Cons |
|---|---|
| Finest granularity (20ms) — catches small edits | Inflates sample count enormously (1 utterance = hundreds of frames) |
| Matches WavLM/SSL hop size directly | Overweights long segments (a 5-second fake region contributes 250 frame decisions) |
| Simple to implement | Not formally justified — adopted from BAM without explicit motivation |
| No segment boundary artifacts | Different from segment EER — not directly comparable despite similar names |

**Who uses it:** PartialEdit, BAM (at 20ms)
**Original paper:** PartialEdit (Zhang et al., Interspeech 2025), adopted from BAM

---

### 4. Duration-Based F1 (HAD, ADD 2023 primary)

**What it measures:** Precision, recall, F1 computed over temporal overlap in SECONDS between predicted and ground-truth fake regions.

| Pros | Cons |
|---|---|
| Resolution-independent (measured in seconds) | **Requires a fixed threshold** — no standard threshold agreed |
| Separates precision and recall — forensically meaningful | Precision and recall definitions in HAD paper appear to have notation inconsistency |
| Directly answers "how many seconds did you get right?" | Different from count-based F1 — name collision causes confusion |
| Threshold-dependent = reflects deployment reality | Only used by HAD and ADD 2023 — limited comparative reach |
| Luong et al. (2025) effectively endorses this approach | |

**Who uses it:** HAD, ADD 2023 Track 2
**Original paper:** Yi et al., Interspeech 2021 (HAD)

---

### 5. AP@IoU (LAV-DF, AV-Deepfake1M primary)

**What it measures:** For each predicted temporal proposal (start, end, confidence), compute IoU with ground truth. Rank proposals by confidence, compute precision-recall curve, average precision at multiple IoU thresholds.

| Pros | Cons |
|---|---|
| Evaluates both detection AND boundary precision simultaneously | Assumes proposal-based detection — incompatible with frame-level binary classifiers |
| Standard in temporal action localization (ActivityNet, THUMOS) — large community understands it | **Three incompatible IoU formulations exist** (continuous, penalized 1D, discrete fixed-window) |
| IoU thresholds (0.5, 0.75, 0.95) test at multiple strictness levels | Borrowed from video — may not suit audio's different temporal scales |
| AR@N evaluates recall at limited number of proposals | Only 3 datasets use it (LAV-DF, AV-Deepfake1M, LENS-DF) — all originally AV |

**Who uses it:** CFPRF (on LAV-DF), UMMAFormer, BA-TFD+
**Original paper (for audio):** Cai et al., DICTA 2022 (LAV-DF)

---

### 6. Asymmetric 1D-IoU (Psynd only)

**What it measures:** Frame-level accuracy with 2x penalty on incorrect frames: N_correct / (N_correct + 2 × N_incorrect)

| Pros | Cons |
|---|---|
| Finest resolution of any dataset (5.75ms) | Used by ONE dataset only — zero comparative value |
| 2x penalty discourages scattered false positives | Non-standard formula — not directly comparable to any other IoU |
| Controlled synthetic construction allows clean evaluation | |

**Who uses it:** Psynd only
**Original paper:** Psynd authors

---

### 7. Boundary F1 (PartialSpoof secondary)

**What it measures:** For each predicted boundary (real→fake transition), check if a ground-truth boundary exists within 20ms tolerance. Compute precision, recall, F1 over boundary counts.

| Pros | Cons |
|---|---|
| Directly evaluates transition detection — forensically critical | Ignores segment content entirely — a model could find all boundaries but mislabel all segments |
| 20ms tolerance is tight but fair for SSL-based models | Only relevant for concatenation-based spoofs WITH splice boundaries |
| Separates "did you find the cut?" from "did you label the region?" | Neural editing attacks (PartialEdit) may have NO sharp boundaries — metric is meaningless there |

**Who uses it:** PartialSpoof evaluation (secondary), BAM
**Original paper:** PartialSpoof (Zhang et al., TASLP 2023)

---

### 8. HTER (LENS-DF secondary)

**What it measures:** Half Total Error Rate = (FAR + FRR) / 2 at a threshold optimized on development set, then applied to test set.

| Pros | Cons |
|---|---|
| Tests generalization of threshold — dev threshold applied to test | Used by ONE dataset only |
| Practical — simulates real deployment (pick threshold, deploy, hope it works) | Fixed threshold may not be optimal for test set |
| Complementary to EER (EER = best case, HTER = realistic case) | |

**Who uses it:** LENS-DF only
**Original paper:** LENS-DF authors (borrowed from face anti-spoofing)

---

## PART 5: The Fragmentation Problem Visualized

From your survey paper Table XI — pairwise Jaccard overlap of metric sets:

```
                PS   HAD  LAV  ADD  Psy  AV1M LENS LPS  PE   HQ
PartialSpoof    —    0    0    0    0    0    0.17 0.25 0.25 0.25
HAD             0    —    0    1.00 0    0    0    0    0    0
LAV-DF          0    0    —    0    0    1.00 0    0    0    0
ADD 2023        0    1.00 0    —    0    0    0    0    0    0
Psynd           0    0    0    0    —    0    0    0    0    0
AV-Deepfake1M   0    0    1.00 0    0    —    0    0    0    0
LENS-DF         0.17 0    0    0    0    0    —    0    0    0
LlamaPS         0.25 0    0    0    0    0    0    —    0.50 1.00
PartialEdit     0.25 0    0    0    0    0    0    0.50 —    0.50
HQ-MPSD         0.25 0    0    0    0    0    0    1.00 0.50 —
```

**36 out of 45 pairs (80%) = 0.00 Jaccard. They share NOTHING.**

Three isolated islands:
- **EER island:** PartialSpoof ↔ LPS ↔ PE ↔ HQ-MPSD (connected via utterance EER, weakly)
- **TAL island:** LAV-DF ↔ AV-Deepfake1M (AP@IoU)
- **Classification island:** HAD ↔ ADD 2023 (Sentence accuracy + duration F1)

Psynd, LENS-DF are completely disconnected from everything.

---

## PART 6: What This Means for Your Presentation

**When an affiliate asks "which metric should we use?":**

> "It depends on what you're evaluating. If you need to know how many SECONDS of fake audio the system missed, use duration-based F1 (threshold-dependent, reflects deployment). If you need a resolution-independent error rate for comparing models during development, use range-based EER. If you're evaluating a proposal-based system that predicts 'this region is fake,' use AP@IoU. The problem is that no single metric covers all use cases, and 80% of dataset pairs share no common metrics at all. Our survey recommends at minimum: utterance-level EER plus one sub-utterance metric."

**When asked "why can't you just compare published numbers?":**

> "Because the same model gives EER values ranging from 0.86% to 34.96% depending on the measurement resolution alone. And three different papers all use 'IoU' but with three mathematically different formulas that produce different numbers on identical predictions. We can't compare across papers without re-running everything under one protocol — which is exactly what our XPS-Forensic toolkit enables."
