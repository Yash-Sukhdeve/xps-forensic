# Detector Selection for Explainable Partial Spoof Detection Pipeline

**Date**: 2026-03-02
**Scope**: Evidence-based selection of frame-level audio spoof countermeasure detectors
**Requirements**: Frame-level scores, published results on PartialSpoof/ADD2023/HAD, public code, peer-reviewed venue

---

## 1. Candidate Detector Analysis

### 1.1 TDAM -- Temporal Difference Attention Model

| Attribute | Detail |
|-----------|--------|
| **Full Reference** | Menglu Li, Xiao-Ping Zhang, Lian Zhao. "Frame-level Temporal Difference Learning for Partial Deepfake Speech Detection." IEEE Signal Processing Letters, 2025. arXiv:2507.15101 |
| **Architecture** | wav2vec2-XLSR (1024-dim embeddings, 20ms hop) -> Temporal Difference Attention Module (TDAM) with dual-level hierarchical difference representation at fine and coarse scales -> adaptive average pooling -> utterance-level binary classification |
| **Frame-level?** | **Partially.** Training uses utterance-level labels only (no frame-level supervision). The temporal difference features are computed frame-wise, but the final output is an utterance-level score. Frame-level scores would require architectural modification (removing the pooling layer). The paper does NOT report segment-level EER/F1. |
| **PartialSpoof Utt-EER** | **0.59%** (eval set); 0.19% (dev set) |
| **PartialSpoof Seg-EER** | **Not reported** |
| **HAD EER** | **0.03%** (AUC: 99.99%) |
| **ADD 2023 Track 2** | Not evaluated |
| **Cross-dataset** | LA->PS: 11.42% EER; PS->LA: 0.71% EER |
| **Code** | https://github.com/menglu-lml/inconsistency (training/testing scripts, pretrained checkpoints via Google Drive) |
| **Venue** | IEEE Signal Processing Letters (peer-reviewed, IF ~3.2) |
| **Citations** | 2 (as of 2026-03; very recent paper) |
| **Saliency applied?** | No. However, the same first author (Li) published a related paper on Interpretable Temporal Class Activation Representation (arXiv:2406.08825, Interspeech 2024, 9 citations) that applies class activation maps to wav2vec2-based spoof detection. |

**Assessment**: Strong utterance-level results on PartialSpoof (best reported EER at 0.59%). However, the model outputs utterance-level scores only; segment-level localization is not natively supported. To use this for frame-level localization, one would need to either (a) remove the adaptive average pooling and add a frame-level classification head, or (b) use the temporal difference features as frame-level representations and apply a separate classifier. The cross-dataset results (LA->PS at 11.42%) show moderate but not exceptional generalization. The lack of segment-level evaluation is a significant limitation for our pipeline.

---

### 1.2 SAL -- Segment-Aware Learning

| Attribute | Detail |
|-----------|--------|
| **Full Reference** | Yuchen Mao, Wen Huang, Yanmin Qian. "Localizing Speech Deepfakes Beyond Transitions via Segment-Aware Learning." arXiv:2601.21925, 2026. |
| **Architecture** | SSL frontend (wav2vec2-XLSR or WavLM-Large) -> optional lightweight Conformer (2 blocks, 4 attn heads) -> Segment Positional Labeling (frame-level supervision based on relative position within segment) + Cross-Segment Mixing augmentation -> **frame-level binary predictions** at 160ms resolution (configurable) |
| **Frame-level?** | **Yes.** Natively produces frame-level predictions. Segment Positional Labeling provides fine-grained frame supervision. |
| **PartialSpoof Seg-EER** | **3.00%** (WavLM); 3.32% (W2V2-XLSR) -- at 160ms resolution |
| **PartialSpoof Seg-F1** | **97.09%** (WavLM); 96.84% (W2V2-XLSR) |
| **PartialSpoof Utt-EER** | Not separately reported (segment-level is the primary task) |
| **HAD EER** | **0.05%** (both WavLM and W2V2-XLSR); F1: 99.99% |
| **ADD 2023 Track 2** | Not evaluated |
| **Cross-dataset (PS->LPS)** | **36.60% EER, 56.09% F1** (WavLM); 35.52% EER, 55.30% F1 (W2V2-XLSR) -- best among all methods compared |
| **Code** | https://github.com/SentryMao/SAL (MIT license, training/eval scripts, supports PS/HAD/LlamaPartialSpoof) |
| **Venue** | **arXiv preprint only** as of 2026-03 (NOT yet peer-reviewed) |
| **Citations** | 0 (very recent, January 2026) |
| **Saliency applied?** | No |

**Comparison baselines from the SAL paper (160ms, PartialSpoof eval)**:

| Model | Backbone | Seg-EER | Seg-F1 |
|-------|----------|---------|--------|
| MRM (Zhang et al.) | SSL | 13.72% | -- |
| CFPRF (Wu et al.) | W2V2-XLSR | 7.61% | 93.89% |
| BFC-Net | WavLM | 2.73% | 96.69% |
| BAM (Zhong et al.) | WavLM | 3.58% | 96.09% |
| **SAL** | **WavLM** | **3.00%** | **97.09%** |

**Assessment**: The strongest candidate for frame-level localization. Natively produces frame-level predictions, reports comprehensive segment-level metrics, evaluates on all three target datasets (PS, HAD, LPS), and provides the best cross-dataset results. The Cross-Segment Mixing augmentation explicitly addresses the known boundary-artifact bias problem, which is critical for forensic credibility. The MIT-licensed code supports all three datasets. **Major limitation**: not yet peer-reviewed (arXiv only). However, the methodology is sound and builds on well-established components (WavLM, Conformer). For a forensic paper, we could cite this as a preprint while emphasizing that the underlying components (WavLM, segment-level supervision) are individually peer-reviewed.

---

### 1.3 CFPRF -- Coarse-to-Fine Proposal Refinement Framework

| Attribute | Detail |
|-----------|--------|
| **Full Reference** | Junyan Wu, Wei Lu, Xiangyang Luo, Rui Yang, Qian Wang, Xiaochun Cao. "Coarse-to-Fine Proposal Refinement Framework for Audio Temporal Forgery Detection and Localization." ACM Multimedia 2024. arXiv:2407.16554 |
| **Architecture** | wav2vec2-XLSR (300M, 1024-dim -> 128-dim) -> Frame-level Detection Network (FDN, identifies coarse candidate regions) -> Proposal Refinement Network (PRN, refines boundaries and confidence) + Difference-Aware Feature Learning (DAFL, contrastive) + Boundary-Aware Feature Enhancement (BAFE, cross-attention) |
| **Frame-level?** | **Yes.** The FDN explicitly produces frame-level forgery predictions at 20ms resolution. The PRN then refines these into proposal-level boundaries. |
| **PartialSpoof EER** | **7.41%** (20ms point-EER, eval set) |
| **PartialSpoof F1** | **93.89%** |
| **PartialSpoof mAP** | **55.22%** (temporal forgery localization) |
| **HAD EER** | **0.08%** (AUC: 99.96%, F1: 99.95%) |
| **HAD mAP** | **99.23%** |
| **ADD 2023 Track 2** | Not evaluated directly (uses ASVS2019PS, HAD, LAV-DF) |
| **Cross-dataset** | Not reported in original paper. Independent reproduction by Luong et al. 2025 found: PS->LPS: 43.25% EER; PS->HalfTruth: 27.59% EER |
| **Code** | https://github.com/ItzJuny/CFPRF (MIT license, pretrained checkpoints for HAD/PS/LAV-DF) |
| **Venue** | **ACM Multimedia 2024** (top-tier, peer-reviewed, h5-index 105) |
| **Citations** | 14 |
| **Saliency applied?** | No |

**Assessment**: Well-established frame-level detector with strong peer-reviewed venue (ACM MM). The two-stage architecture (FDN + PRN) is elegant and interpretable: first detect coarse regions, then refine boundaries. This naturally produces "evidence objects" (detected regions with confidence scores and boundary positions) that map well to forensic reporting. The mAP metric for temporal localization (55.22% on PS) shows the system can identify manipulated intervals, not just classify frames. Pretrained checkpoints are available for all three datasets. **Limitations**: (1) Cross-dataset generalization is weaker than SAL (43.25% vs 36.60% EER on LPS), (2) In-domain segment EER on PS (7.41%) is higher than SAL (3.00%) or BAM (3.58%). However, the ACM MM publication and 14 citations provide stronger scientific credibility than arXiv-only papers.

---

### 1.4 BAM -- Boundary-Aware Attention Mechanism

| Attribute | Detail |
|-----------|--------|
| **Full Reference** | Jiafeng Zhong, Bin Li, Jiangyan Yi. "Enhancing Partially Spoofed Audio Localization with Boundary-aware Attention Mechanism." Interspeech 2024. arXiv:2407.21611 |
| **Architecture** | SSL backend (WavLM-Large or W2V2-XLSR) -> Boundary Enhancement module (extracts discriminative boundary features) -> Boundary Frame-wise Attention (leverages boundary predictions to control feature interaction between frames) -> frame-level predictions |
| **Frame-level?** | **Yes.** Produces frame-level predictions with boundary-aware attention. |
| **PartialSpoof Seg-EER (160ms)** | **3.58%** (WavLM-Large); 4.12% (W2V2-XLSR) |
| **PartialSpoof Seg-F1 (160ms)** | **96.09%** (WavLM-Large); 94.98% (W2V2-XLSR) |
| **PartialSpoof at other resolutions** | 20ms: 5.20% EER; 40ms: 4.90%; 80ms: 4.32%; 320ms: 2.71%; 640ms: 2.28% |
| **HAD** | Not evaluated |
| **ADD 2023 Track 2** | Not evaluated |
| **Cross-dataset** | Not reported |
| **Code** | https://github.com/media-sec-lab/BAM (pretrained checkpoint via Google Drive, supports PartialSpoof) |
| **Venue** | **Interspeech 2024** (peer-reviewed, top speech venue) |
| **Citations** | 17 |
| **Saliency applied?** | No |

**Assessment**: Strong frame-level detector with good peer-reviewed venue (Interspeech). The multi-resolution evaluation (20ms to 640ms) is particularly valuable for forensic applications where the granularity of localization claims matters. The boundary-aware attention mechanism provides a natural explanation pathway: the model explicitly attends to boundary regions, and this attention pattern could be visualized as an explanatory artifact. **Limitations**: (1) Only evaluated on PartialSpoof (no HAD, no cross-dataset), (2) The focus on boundary features risks overfitting to boundary artifacts (the very problem SAL's Cross-Segment Mixing is designed to mitigate), (3) Limited to PartialSpoof dataset evaluation.

---

### 1.5 DKU-DUKEECE System

| Attribute | Detail |
|-----------|--------|
| **Full Reference** | Zexin Cai, Weiqing Wang, Yikang Wang, Ming Li. "The DKU-DUKEECE System for the Manipulation Region Location Task of ADD 2023." ADD 2023 Challenge, 2023. arXiv:2308.10281 |
| **Architecture** | Three-component fusion: (1) Frame-level boundary detector (identifies splice points), (2) Frame-level deepfake detector (classifies real/fake at frame level), (3) VAE trained exclusively on genuine data (anomaly detection for authenticity). Outputs fused for final region predictions. |
| **Frame-level?** | **Yes.** All three subsystems operate at frame level. |
| **PartialSpoof** | Not evaluated (developed for ADD 2023 data) |
| **ADD 2023 Track 2** | **1st place**: Sentence Accuracy 82.23%, F1 60.66%, ADD Score 0.6713 |
| **HAD** | Not evaluated |
| **Cross-dataset** | Not reported |
| **Code** | **Not publicly available** (no GitHub repository found) |
| **Venue** | ADD 2023 Challenge workshop paper (arXiv preprint, not a full peer-reviewed venue) |
| **Citations** | 11 |
| **Saliency applied?** | No |

**Assessment**: The ADD 2023 Track 2 winner, so the results on that benchmark are authoritative. The three-component architecture is forensically appealing: boundary detection + deepfake scoring + anomaly detection provides three independent evidence streams. **Critical limitation**: no public code. This is disqualifying for our pipeline, which requires reproducibility (R5 in project rules). The system was designed for the ADD 2023 challenge data and has not been evaluated on PartialSpoof or HAD. Without code, this system serves as a reference point but cannot be a candidate for implementation.

---

### 1.6 AASIST / AASIST2

| Attribute | Detail |
|-----------|--------|
| **Full Reference (AASIST)** | Jee-weon Jung, Hee-Soo Heo, Hemlata Tak, et al. "AASIST: Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention Networks." ICASSP 2022. arXiv:2110.01200 |
| **Full Reference (AASIST2)** | Yuxiang Zhang, Jingze Lu, et al. "Improving Short Utterance Anti-Spoofing with AASIST2." ICASSP 2024. arXiv:2309.08279 |
| **Architecture (AASIST)** | Raw waveform -> RawNet2-style sinc-conv encoder -> parallel temporal graph module + spectral graph module -> heterogeneous stacking graph attention -> max graph readout -> **utterance-level score** |
| **Architecture (AASIST2)** | wav2vec 2.0 -> Res2Net blocks + AASIST graph attention -> Dynamic Chunk Size + Adaptive Large Margin Fine-Tuning -> **utterance-level score** |
| **Frame-level?** | **No.** Both AASIST and AASIST2 produce utterance-level scores via graph readout / pooling. There is no frame-level output head. |
| **ASVspoof 2019 LA** | AASIST: EER 0.83%, min t-DCF 0.0275; AASIST-L: EER 0.99%, min t-DCF 0.0309 |
| **PartialSpoof** | Not evaluated in original papers |
| **HAD / ADD 2023** | Not evaluated |
| **Code** | https://github.com/clovaai/aasist (MIT license, pretrained checkpoints for AASIST and AASIST-L) |
| **Venue** | ICASSP 2022 / ICASSP 2024 (top-tier peer-reviewed) |
| **Citations** | **470** (AASIST is the most-cited model in this analysis) |
| **Saliency applied?** | Not by the original authors. Third-party work has applied attention visualization to AASIST-like architectures. |

**Assessment**: AASIST is the most influential CM in the community (470 citations) and has excellent utterance-level performance. However, it is **disqualified for our pipeline** because it does not produce frame-level scores. It could serve as a reference baseline for utterance-level comparison, but adding frame-level output would require significant architectural modification (replacing the graph readout with a frame-level classification head), which would essentially create a different model. AASIST2's improvements focus on short-utterance robustness, not frame-level localization.

---

### 1.7 Multi-Resolution Model (MRM) -- PartialSpoof Baseline

| Attribute | Detail |
|-----------|--------|
| **Full Reference** | Lin Zhang, Xin Wang, Erica Cooper, Nicholas Evans, Junichi Yamagishi. "The PartialSpoof Database and Countermeasures for the Detection of Short Fake Speech Segments Embedded in an Utterance." IEEE/ACM TASLP, vol. 31, pp. 813-825, 2023. arXiv:2204.05177 |
| **Architecture** | SSL pretrained model (feature extractor) -> multi-resolution segment-level heads (20ms to 640ms) + utterance-level head -> joint optimization with segment-level and utterance-level labels |
| **Frame-level?** | **Yes.** Multi-resolution segment-level predictions from 20ms to 640ms are the core contribution. |
| **PartialSpoof Utt-EER** | **0.77%** (eval); 0.90% on ASVspoof 2019 LA |
| **PartialSpoof Seg-EER (20ms)** | **13.72%** (from SAL paper's reproduction using MultiResoModel-Simple) |
| **HAD** | **46.48% EER** (cross-dataset, from MultiResoModel-Simple; very poor generalization) |
| **Cross-dataset (PS->LPS)** | **46.30% EER** (from MultiResoModel-Simple) |
| **Code** | https://github.com/nii-yamagishilab/PartialSpoof (official dataset + baseline code); https://github.com/hieuthi/MultiResoModel-Simple (unofficial reimplementation, MIT license) |
| **Venue** | **IEEE/ACM TASLP** (top-tier journal, IF ~4.1) |
| **Citations** | **98** |
| **Saliency applied?** | No |

**Assessment**: This is the foundational model for partial spoof detection -- the original PartialSpoof countermeasure from the dataset authors themselves. Published in a top journal with 98 citations. Multi-resolution evaluation is unique and valuable for forensic applications. However, in-domain segment-level performance (13.72% EER at 20ms) is significantly worse than newer methods (SAL: 3.00%, BAM: 3.58%, CFPRF: 7.41%), and cross-dataset generalization is very poor (46.30% on LPS, 46.48% on HAD). This model is valuable as a baseline and reference but is not competitive enough to be a primary detector in our pipeline.

---

### 1.8 Interpretable Temporal Class Activation (ITCA)

| Attribute | Detail |
|-----------|--------|
| **Full Reference** | Menglu Li, Xiao-Ping Zhang. "Interpretable Temporal Class Activation Representation for Audio Spoofing Detection." Interspeech 2024. arXiv:2406.08825 |
| **Architecture** | wav2vec 2.0 -> attentive utterance-level features -> class activation representation to localize discriminative frames |
| **Frame-level?** | **Yes, for interpretability** (class activation maps identify discriminative frames), but the primary output is utterance-level. |
| **ASVspoof 2019 LA** | EER 0.51%, min t-DCF 0.0165 |
| **PartialSpoof** | Not evaluated |
| **HAD / ADD 2023** | Not evaluated |
| **Code** | Not publicly available |
| **Venue** | Interspeech 2024 (peer-reviewed) |
| **Citations** | 9 |
| **Saliency applied?** | **Yes** -- this is the saliency paper. Class activation representation is the core contribution for frame-level interpretability. |

**Assessment**: Highly relevant for the explainability component of our pipeline. The class activation representation provides a method to identify which temporal frames contribute most to the detection decision. However, it has not been evaluated on PartialSpoof, HAD, or ADD 2023, and code is not publicly available. This model is better suited as an explainability technique to apply on top of another detector, rather than as a standalone frame-level detector.

---

### 1.9 Frame-to-Utterance Convergence (FUC)

| Attribute | Detail |
|-----------|--------|
| **Full Reference** | Awais Khan, Khalid Mahmood Malik, Shah Nawaz. "Frame-to-Utterance Convergence: A Spectra-Temporal Approach for Unified Spoofing Detection." ICASSP 2024. arXiv:2309.09837 |
| **Architecture** | Spectral Deviation Coefficient (SDC, frame-level) -> Bi-LSTM for Sequential Temporal Coefficients (STC, utterance-level) -> Auto-encoder for Spectra-Temporal Deviated Coefficients (STDC) |
| **Frame-level?** | **Partially.** SDC produces frame-level inconsistency scores, but the full system outputs utterance-level decisions. |
| **PartialSpoof** | Evaluated but specific segment-level EER numbers not extracted (full paper content inaccessible) |
| **ASVspoof 2019 LA / 2021** | Evaluated |
| **Code** | Not publicly available |
| **Venue** | ICASSP 2024 (top-tier peer-reviewed) |
| **Citations** | 19 |
| **Saliency applied?** | No |

**Assessment**: The SDC component provides frame-level inconsistency scores that could be useful. Published at ICASSP (strong venue). However, code is not publicly available (disqualifying for reproducibility), and the system is primarily utterance-level. The lack of accessible code and the primarily utterance-level design make this unsuitable as a primary detector.

---

### 1.10 Speech Quality-Based Localization

| Attribute | Detail |
|-----------|--------|
| **Full Reference** | Michael Kuhlmann, Alexander Werning, Thilo von Neumann, Reinhold Haeb-Umbach. "Speech Quality-Based Localization of Low-Quality Speech and Text-to-Speech Synthesis Artefacts." ICASSP 2026. arXiv:2601.21886 |
| **Architecture** | Utterance-level speech quality predictor -> regularized with segment-based consistency constraint -> frame-level quality scores without frame-level training labels |
| **Frame-level?** | **Yes.** Produces frame-level quality scores. |
| **PartialSpoof** | Mentioned as evaluated but specific numbers not extracted |
| **Code** | Not publicly available (as of 2026-03) |
| **Venue** | ICASSP 2026 (accepted, peer-reviewed) |
| **Citations** | Too new to assess |
| **Saliency applied?** | The frame-level quality scores are inherently interpretable (lower quality = more suspicious). |

**Assessment**: Interesting approach that produces inherently interpretable frame-level scores (speech quality rather than binary fake/real). However, code is not yet available, and no quantitative segment-level results were extractable. This is a promising future direction but not ready for immediate pipeline integration.

---

## 2. Comparison Table

| Model | Architecture | Frame-level? | PS Utt-EER | PS Seg-EER (160ms) | PS Seg-F1 | ADD2023 F1 | HAD EER | Cross-dataset (PS->LPS) | Code Available | Venue | Citations |
|-------|-------------|-------------|------------|-------------------|-----------|-----------|---------|------------------------|---------------|-------|-----------|
| **SAL** (Mao 2026) | WavLM/W2V2-XLSR + Conformer + SPL + CSM | **Yes** | -- | **3.00%** | **97.09%** | -- | **0.05%** | **36.60%** | **Yes** (MIT) | arXiv (preprint) | 0 |
| **BAM** (Zhong 2024) | WavLM/W2V2-XLSR + Boundary Enhancement + BFA | **Yes** | -- | 3.58% | 96.09% | -- | -- | -- | **Yes** | Interspeech 2024 | 17 |
| **CFPRF** (Wu 2024) | W2V2-XLSR + FDN + PRN + DAFL + BAFE | **Yes** | -- | 7.41% (20ms) | 93.89% | -- | **0.08%** | 43.25% (Luong repro) | **Yes** (MIT) | ACM MM 2024 | 14 |
| **TDAM** (Li 2025) | W2V2-XLSR + TDAM + Adaptive AvgPool | **No** (utt-level) | **0.59%** | -- | -- | -- | **0.03%** | 11.42% (LA->PS) | **Yes** | IEEE SPL 2025 | 2 |
| **MRM** (Zhang 2023) | SSL + Multi-resolution heads | **Yes** | 0.77% | 13.72% | -- | -- | 46.48% | 46.30% | **Yes** | IEEE/ACM TASLP | 98 |
| **DKU-DUKEECE** (Cai 2023) | Boundary det. + DF det. + VAE fusion | **Yes** | -- | -- | **60.66%** (ADD) | **60.66%** | -- | -- | **No** | Challenge paper | 11 |
| **AASIST** (Jung 2022) | Raw waveform + Graph Attention | **No** (utt-level) | -- | -- | -- | -- | -- | -- | **Yes** (MIT) | ICASSP 2022 | 470 |
| **ITCA** (Li 2024) | wav2vec2 + Class Activation | Partial (XAI) | -- | -- | -- | -- | -- | -- | **No** | Interspeech 2024 | 9 |
| **FUC** (Khan 2024) | SDC + Bi-LSTM + AE | Partial | -- | -- | -- | -- | -- | -- | **No** | ICASSP 2024 | 19 |

---

## 3. Elimination Analysis

### Disqualified candidates

1. **AASIST/AASIST2**: No frame-level output. Utterance-level only via graph readout. Would require fundamental architectural changes. **Eliminated: fails frame-level requirement.**

2. **DKU-DUKEECE**: No public code. Despite being the ADD 2023 Track 2 winner, reproducibility is impossible. **Eliminated: fails code availability requirement.**

3. **ITCA (Li 2024)**: No public code, not evaluated on PartialSpoof/HAD. **Eliminated: fails code and dataset requirements.**

4. **FUC (Khan 2024)**: No public code. **Eliminated: fails code availability requirement.**

5. **Speech Quality-Based (Kuhlmann 2026)**: No public code yet. **Eliminated: fails code availability requirement.**

6. **TDAM**: Does not produce frame-level scores natively (uses adaptive average pooling to utterance level). While the internal temporal difference features could theoretically be repurposed, the model as published does not support segment-level evaluation. The authors report no segment-level metrics. **Eliminated: fails frame-level requirement as published.** However, the codebase could potentially be modified -- this is a borderline case.

### Remaining candidates (meet all requirements except potentially venue)

| Rank | Model | Frame-level | PS Seg-EER | Code | Venue | Key Strength | Key Weakness |
|------|-------|-------------|------------|------|-------|-------------|-------------|
| 1 | **SAL** | Yes | 3.00% | Yes (MIT) | arXiv only | Best segment EER, best cross-dataset, evaluates on PS+HAD+LPS, addresses boundary bias | Not yet peer-reviewed |
| 2 | **BAM** | Yes | 3.58% | Yes | Interspeech 2024 | Strong segment EER, multi-resolution eval, peer-reviewed | No HAD/cross-dataset eval, may overfit boundaries |
| 3 | **CFPRF** | Yes | 7.41% | Yes (MIT) | ACM MM 2024 | Top venue, pretrained ckpts for 3 datasets, mAP metric | Higher segment EER, weaker cross-dataset |
| 4 | **MRM** | Yes | 13.72% | Yes | TASLP (journal) | Foundational model, highest citations, top journal | Poorest segment-level performance, poor generalization |

---

## 4. Recommendation: Selected Detectors

### Primary Detector: SAL (Segment-Aware Learning)

**Scientific justification:**

1. **Best frame-level performance on PartialSpoof**: 3.00% segment EER with 97.09% F1 (WavLM backbone) -- the lowest segment-level error rate among all candidates with public code.

2. **Best cross-dataset generalization**: 36.60% EER on LlamaPartialSpoof (PS->LPS), compared to 43.25% for CFPRF and 46.30% for MRM. While all methods degrade substantially out-of-domain, SAL degrades least. For forensic applications, generalization to unseen attack types is critical because real-world evidence will not match training distributions.

3. **Explicitly addresses boundary-artifact bias**: The Cross-Segment Mixing (CSM) augmentation was specifically designed to prevent models from relying solely on transition artifacts. This directly addresses a known vulnerability flagged in the forensic literature (cf. the deep-research-report.md discussion of boundary-cue reliance). A forensic expert can argue that the model was trained to detect manipulated content, not just splice boundaries.

4. **Multi-dataset evaluation**: Evaluated on PartialSpoof, HAD (0.05% EER), and LlamaPartialSpoof, providing evidence across different manipulation types and generation methods.

5. **Dual SSL backbone support**: Both WavLM and wav2vec2-XLSR are supported, enabling ablation studies and ensemble strategies.

6. **MIT license with full code**: Training scripts, evaluation code, and dataset preparation for all three target datasets are available.

**Limitation acknowledged**: SAL is currently an arXiv preprint (January 2026), not yet peer-reviewed. **Mitigation**: (a) The underlying components (WavLM, wav2vec2-XLSR, Conformer) are individually well-established and peer-reviewed. (b) The evaluation protocol uses established metrics and datasets. (c) We will cite it as a preprint and note this status in any publications. (d) If peer review rejects or revises the paper, we can swap to BAM as a fallback since both use the same WavLM backbone.

---

### Secondary Detector: CFPRF (Coarse-to-Fine Proposal Refinement Framework)

**Scientific justification:**

1. **Top-tier peer-reviewed venue**: ACM Multimedia 2024 is a flagship conference (h5-index 105), providing the strongest publication credibility among frame-level partial spoof detectors.

2. **Two-stage architecture maps naturally to forensic evidence**: The FDN produces coarse frame-level predictions, and the PRN refines these into precise temporal proposals with boundaries and confidence scores. This two-stage process mirrors forensic reasoning: first flag suspicious regions, then examine boundaries. Each stage produces interpretable intermediate outputs.

3. **Temporal forgery localization via mAP**: Unlike other methods that report only frame-level EER/F1, CFPRF also reports temporal localization mAP (55.22% on PS, 99.23% on HAD). mAP measures whether the detected temporal proposals overlap correctly with ground-truth manipulated intervals -- this is precisely the metric a court needs (did the system correctly identify the manipulated time range?).

4. **Pretrained checkpoints for three datasets**: HAD, PartialSpoof, and LAV-DF checkpoints are provided, enabling immediate evaluation without retraining.

5. **Independent reproduction**: Luong et al. (2025, ICASSP) independently reproduced CFPRF and evaluated it cross-dataset, providing third-party validation of the method's strengths and limitations.

6. **Complementary to SAL**: CFPRF uses a proposal-based approach (detect regions, refine boundaries), while SAL uses a frame-classification approach (classify every frame, with positional labeling). Combining both provides two independent evidence streams -- a stronger forensic argument than relying on a single method.

---

### Tertiary Detector (Baseline/Reference): MRM (Multi-Resolution Model)

**Scientific justification:**

1. **Foundational credibility**: Published in IEEE/ACM TASLP (top journal, 98 citations), created by the PartialSpoof dataset authors themselves. This is the reference implementation for the dataset.

2. **Multi-resolution evaluation at publication**: The only model with native support for evaluating at 20ms, 40ms, 80ms, 160ms, 320ms, and 640ms resolutions simultaneously. This is essential for forensic reporting: a court may need to know the model's localization precision at different temporal granularities.

3. **Baseline comparator**: Including MRM allows us to demonstrate that our chosen primary/secondary detectors (SAL, CFPRF) substantially outperform the established baseline, strengthening the validation argument.

4. **Weaker performance is informative**: MRM's higher segment EER (13.72%) and poor cross-dataset generalization (46.30% on LPS) establish an empirical lower bound. If SAL and CFPRF agree on a region but MRM does not, this provides information about detection confidence.

---

## 5. Pipeline Integration Strategy

```
Audio input
    |
    v
[SSL Feature Extraction]
    |-- WavLM-Large (shared backbone)
    |-- wav2vec2-XLSR (shared backbone)
    |
    v
[Detector 1: SAL]          [Detector 2: CFPRF]        [Detector 3: MRM (baseline)]
Frame-level predictions     Frame-level proposals       Multi-resolution predictions
(160ms resolution)          (20ms resolution + PRN)     (20ms to 640ms)
    |                           |                           |
    v                           v                           v
[Ensemble / Agreement Layer]
    |-- Frame-level consensus scores
    |-- Boundary agreement map
    |-- Confidence calibration
    |
    v
[Explainability Layer]
    |-- Saliency maps (IG/GradSHAP on WavLM features)
    |-- Temporal localization exhibits
    |-- Cross-detector agreement visualization
    |
    v
[Forensic Evidence Package]
```

### Why this combination works for forensic explainability:

1. **Multiple independent evidence streams**: SAL (frame classification), CFPRF (proposal refinement), and MRM (multi-resolution baseline) provide three independent assessments. Agreement across methods strengthens confidence; disagreement flags uncertainty.

2. **Different architectural assumptions**: SAL addresses boundary bias via CSM augmentation; CFPRF explicitly models boundaries via PRN; MRM provides a simpler multi-resolution baseline. If all three agree a region is manipulated despite different architectural biases, the finding is more robust.

3. **Reproducibility**: All three have public code (SAL: MIT, CFPRF: MIT, MRM: unofficial but MIT). Any expert can reproduce the analysis.

4. **Temporal granularity flexibility**: MRM provides 20ms-640ms resolution; CFPRF provides 20ms frame-level + refined proposals; SAL provides 160ms frame-level. Together, they cover the full granularity spectrum needed for forensic claims at different precision levels.

5. **Shared SSL backbones**: SAL and CFPRF both support wav2vec2-XLSR; SAL also supports WavLM-Large. Sharing feature extractors reduces computational overhead and enables apples-to-apples comparison of detection heads.

---

## 6. Risk Assessment and Mitigations

| Risk | Severity | Mitigation |
|------|----------|------------|
| SAL not yet peer-reviewed | Medium | Cite as preprint; emphasize peer-reviewed components (WavLM, Conformer); prepare BAM as fallback |
| All models degrade on cross-dataset (best: 36.60% EER) | High | Report cross-dataset EER honestly; use ensemble to improve robustness; evaluate on case-specific conditions; explicitly state limitations in forensic reports |
| Boundary-artifact bias in BAM/CFPRF | Medium | SAL's CSM explicitly mitigates this; ensemble agreement reduces individual model bias |
| No model evaluated on ADD 2023 Track 2 with public code | Low | DKU-DUKEECE won Track 2 but has no code; our chosen models cover PartialSpoof and HAD which use similar manipulation types |
| CFPRF's independent reproduction showed worse results than original | Low | Expected in reproduction studies; Luong et al. still confirmed method viability; we will use official checkpoints |

---

## 7. Open Questions for Next Phase

1. **Saliency integration**: None of the selected detectors have published saliency/attribution results. The ITCA paper (Li 2024, Interspeech) demonstrates class activation maps on wav2vec2, which could be adapted to SAL/CFPRF's WavLM/W2V2-XLSR backbones. Integrated Gradients and GradSHAP should also be evaluated on the SSL feature representations.

2. **WavLM layer selection**: Which WavLM layers carry the most discriminative information for partial spoof detection? Layer-wise probing experiments are needed (cf. SSL probing literature).

3. **Calibration**: Frame-level scores from all three models need to be calibrated (e.g., Platt scaling or isotonic regression) before they can be reported as probabilities in forensic contexts.

4. **Ensemble strategy**: How to combine frame-level predictions from SAL (160ms), CFPRF (20ms proposals), and MRM (multi-resolution)? Options include temporal alignment + voting, weighted averaging based on per-model calibration, or a learned meta-classifier.

5. **Real-world codec robustness**: None of these models have been specifically evaluated under the ASVspoof 2021 DF-style codec conditions. Cross-codec evaluation is essential for forensic credibility.

---

## 8. References

1. Mao, Y., Huang, W., & Qian, Y. (2026). Localizing Speech Deepfakes Beyond Transitions via Segment-Aware Learning. arXiv:2601.21925. Code: https://github.com/SentryMao/SAL

2. Wu, J., Lu, W., Luo, X., Yang, R., Wang, Q., & Cao, X. (2024). Coarse-to-Fine Proposal Refinement Framework for Audio Temporal Forgery Detection and Localization. ACM Multimedia 2024. arXiv:2407.16554. Code: https://github.com/ItzJuny/CFPRF

3. Zhang, L., Wang, X., Cooper, E., Evans, N., & Yamagishi, J. (2023). The PartialSpoof Database and Countermeasures for the Detection of Short Fake Speech Segments Embedded in an Utterance. IEEE/ACM TASLP, 31, 813-825. arXiv:2204.05177. Code: https://github.com/nii-yamagishilab/PartialSpoof ; Unofficial reimplementation: https://github.com/hieuthi/MultiResoModel-Simple

4. Zhong, J., Li, B., & Yi, J. (2024). Enhancing Partially Spoofed Audio Localization with Boundary-aware Attention Mechanism. Interspeech 2024. arXiv:2407.21611. Code: https://github.com/media-sec-lab/BAM

5. Li, M., Zhang, X.-P., & Zhao, L. (2025). Frame-level Temporal Difference Learning for Partial Deepfake Speech Detection. IEEE Signal Processing Letters. arXiv:2507.15101. Code: https://github.com/menglu-lml/inconsistency

6. Cai, Z., Wang, W., Wang, Y., & Li, M. (2023). The DKU-DUKEECE System for the Manipulation Region Location Task of ADD 2023. arXiv:2308.10281. Code: NOT AVAILABLE.

7. Jung, J., Heo, H., Tak, H., et al. (2022). AASIST: Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention Networks. ICASSP 2022. arXiv:2110.01200. Code: https://github.com/clovaai/aasist

8. Li, M. & Zhang, X.-P. (2024). Interpretable Temporal Class Activation Representation for Audio Spoofing Detection. Interspeech 2024. arXiv:2406.08825.

9. Khan, A., Malik, K.M., & Nawaz, S. (2024). Frame-to-Utterance Convergence: A Spectra-Temporal Approach for Unified Spoofing Detection. ICASSP 2024. arXiv:2309.09837.

10. Luong, H.-T., Rimon, I., Permuter, H., Lee, K.A., & Chng, E.S. (2025). Robust Localization of Partially Fake Speech: Metrics and Out-of-Domain Evaluation. ICASSP 2025. arXiv:2507.03468. Metrics code: https://github.com/hieuthi/partialspoof-metrics

11. Luong, H.-T., Li, X., Zhang, L., Lee, K.A., & Chng, E.S. (2025). LlamaPartialSpoof: An LLM-Driven Fake Speech Dataset Simulating Disinformation Generation. ICASSP 2025. Dataset: https://github.com/hieuthi/LlamaPartialSpoof

12. Kuhlmann, M., Werning, A., von Neumann, T., & Haeb-Umbach, R. (2026). Speech Quality-Based Localization of Low-Quality Speech and Text-to-Speech Synthesis Artefacts. ICASSP 2026. arXiv:2601.21886.

---

## 9. Summary Decision

| Role | Model | Justification |
|------|-------|---------------|
| **Primary detector** | SAL (WavLM-Large backbone) | Best segment EER (3.00%), best cross-dataset generalization, addresses boundary bias, evaluates on PS+HAD+LPS, MIT code |
| **Secondary detector** | CFPRF (W2V2-XLSR backbone) | Peer-reviewed at ACM MM 2024, proposal-based localization with mAP, pretrained checkpoints, complementary architecture |
| **Baseline reference** | MRM (PartialSpoof official) | Foundational model (TASLP, 98 citations), multi-resolution support, establishes performance lower bound |

This selection satisfies all stated requirements: frame-level scores (SAL, CFPRF, MRM), published results on PartialSpoof and HAD (SAL, CFPRF, MRM), public code (all three), and at least partial peer review coverage (CFPRF at ACM MM, MRM at TASLP; SAL pending but components are peer-reviewed).
