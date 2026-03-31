# Latest Models and Papers on Partial Audio Spoof Detection & Localization (2024-2026)

Comprehensive literature search conducted 2026-03-29 for IEEE TIFS submission (XPS-Forensic).

**Known models (already in our pipeline):** BAM, SAL, CFPRF, MRM, BFC-Net, TDAM

---

## Category A: DIRECTLY RELEVANT -- Partial Spoof Localization Methods

### 1. SAL -- Segment-Aware Learning (NEW, already tracked)
- **Title:** Localizing Speech Deepfakes Beyond Transitions via Segment-Aware Learning
- **Authors:** Yuchen Mao, Wen Huang, Yanmin Qian
- **Venue:** arXiv preprint, January 2026
- **arXiv:** 2601.21925
- **Method:** Proposes Segment Positional Labeling (fine-grained frame supervision based on relative position within a segment) and Cross-Segment Mixing (data augmentation generating diverse segment patterns). Addresses over-reliance on boundary artifacts.
- **Benchmark:** PartialSpoof
- **Code:** Unknown
- **Status:** Already in our pipeline as SOTA comparison

### 2. BFC-Net -- Boundary-Frame Cross Graph Attention Network (KNOWN)
- **Title:** BFC-Net: Boundary-Frame cross graph attention network for partially spoofed audio localization
- **Venue:** Neurocomputing, 2025
- **DOI:** ScienceDirect S0925231225015395
- **Method:** Three core modules: Large Selective Kernel Residual Block (LSK-ResBlock), Multi-level Feature Aggregation (MFA), Boundary-Frame Cross Attention (BFCA). LSK-ResBlock enhances spectral/temporal representations; MFA aggregates local/global features for boundary prediction; BFCA controls frame interactions based on boundary prediction.
- **Performance:** EER 2.73% on PartialSpoof; 92.13% on ADD2023 Track 2 (SOTA on both)
- **Code:** Unknown
- **Status:** Already tracked

### 3. LOCO -- Progressive Audio-Language Co-learning Network (**NEW**)
- **Title:** Weakly-supervised Audio Temporal Forgery Localization via Progressive Audio-language Co-learning Network
- **Authors:** (Multiple)
- **Venue:** IJCAI 2025
- **arXiv:** 2505.01880
- **Method:** Audio-language co-learning module captures forgery consensus features via temporal/global semantic alignment with forgery-aware prompts. Progressive refinement strategy generates pseudo frame-level labels and leverages supervised semantic contrastive learning. First weakly-supervised ATFL approach using language priors.
- **Performance:** SOTA on HAD, LAV-DF, AV-Deepfake-1M. Reduces EER by 3.91%, 17.26%, 12.39% vs. second-best on these datasets.
- **Code:** Unknown
- **Relevance:** HIGH -- weakly-supervised localization is a novel direction; cite as complementary paradigm

### 4. PartialEdit -- Neural Speech Editing Partial Deepfakes (**NEW**)
- **Title:** PartialEdit: Identifying Partial Deepfakes in the Era of Neural Speech Editing
- **Authors:** You Zhang et al.
- **Venue:** Interspeech 2025
- **arXiv:** 2506.02958
- **Method:** New dataset curated using VoiceCraft, SSR-Speech, Audiobox-Speech, Audiobox. 108 speakers, 43,358 partially edited utterances per model. Key finding: models trained on PartialSpoof fail to detect partially edited speech from neural editors.
- **Dataset URL:** https://yzyouzhang.com/PartialEdit/
- **Code:** Dataset released on Zenodo
- **Relevance:** CRITICAL -- demonstrates domain shift from PartialSpoof; must cite as limitation/future work

### 5. LlamaPartialSpoof (**NEW**)
- **Title:** LlamaPartialSpoof: An LLM-Driven Fake Speech Dataset Simulating Disinformation Generation
- **Authors:** Hieu-Thi Luong et al.
- **Venue:** ICASSP 2025, Hyderabad
- **arXiv:** 2409.14743
- **Method:** 130-hour dataset with both fully and partially fake speech using LLM + voice cloning. Tests realistic out-of-domain scenarios.
- **Performance:** Best CM achieves 24.49% EER (showing poor generalization)
- **GitHub:** https://github.com/hieuthi/LlamaPartialSpoof
- **Relevance:** HIGH -- important OOD evaluation dataset for partial spoof

### 6. Robust Localization of Partially Fake Speech (**NEW**)
- **Title:** Robust Localization of Partially Fake Speech: Metrics and Out-of-Domain Evaluation
- **Authors:** Hieu-Thi Luong, Inbal Rimon, Haim Permuter, Kong Aik Lee, Eng Siong Chng
- **Venue:** arXiv preprint, July 2025
- **arXiv:** 2507.03468
- **Method:** Critically examines limitations of EER for localization; proposes reframing as sequential anomaly detection with threshold-dependent metrics (accuracy, precision, recall, F1). Tests CFPRF cross-domain.
- **Performance:** CFPRF achieves 7.61% EER (20ms) on PartialSpoof eval; degrades to 43.25% on LlamaPartialSpoof and 27.59% on Half-Truth OOD sets.
- **Relevance:** CRITICAL -- directly relevant to our metrics discussion; validates need for better evaluation frameworks

### 7. Can LLMs Help Localize Fake Words in Partially Fake Speech? (**NEW**)
- **Title:** Can LLMs Help Localize Fake Words in Partially Fake Speech?
- **Authors:** Lin Zhang, Thomas Thebaud, Zexin Cai, Sanjeev Khudanpur, Daniel Povey, et al.
- **Venue:** arXiv preprint, March 2026
- **arXiv:** 2603.11205
- **Method:** Speech LLM performing fake word localization via next token prediction. Tested on AV-Deepfake1M and PartialEdit.
- **Relevance:** HIGH -- LLM-based localization; novel paradigm; relevant to our LLM-as-narrator approach

### 8. SDE -- Unsupervised Domain Adaptation for Partially Fake Audio (**NEW**)
- **Title:** An Unsupervised Domain Adaptation Method for Locating Manipulated Region in Partially Fake Audio
- **Authors:** (Multiple)
- **Venue:** arXiv preprint, July 2024
- **arXiv:** 2407.08239
- **Method:** Mixture-of-experts inspired unsupervised method (Samples mining with Diversity and Entropy - SDE). Addresses cross-domain performance drop for manipulation region localization.
- **Relevance:** MEDIUM -- domain adaptation for partial spoof localization

### 9. Manipulated Regions Localization Survey (**NEW** -- SURVEY)
- **Title:** Manipulated Regions Localization For Partially Deepfake Audio: A Survey
- **Authors:** Jiayi He, Jiangyan Yi, Jianhua Tao, Siding Zeng, Hao Gu
- **Venue:** Submitted to IEEE TPAMI, June 2025
- **arXiv:** 2506.14396
- **Relevance:** CRITICAL -- comprehensive survey submitted to top venue; must cite. Potential competitor/complementary to our work in TIFS.

---

## Category B: TEMPORAL FORGERY LOCALIZATION (Audio-Visual & General)

### 10. GEM-TFL (**NEW**)
- **Title:** GEM-TFL: Bridging Weak and Full Supervision for Forgery Localization through EM-Guided Decomposition and Temporal Refinement
- **Authors:** Xiaodong Zhu, Yuanming Zheng, et al.
- **Venue:** arXiv preprint, March 2026
- **arXiv:** 2603.05095
- **Method:** EM-based optimization reformulates binary labels into multi-dimensional latent attributes. Training-free temporal consistency refinement. Graph-based proposal refinement.
- **Relevance:** MEDIUM -- general TFL method, bridges weak/full supervision

### 11. DDNet (**NEW**)
- **Title:** DDNet: A Dual-Stream Graph Learning and Disentanglement Framework for Temporal Forgery Localization
- **Venue:** arXiv preprint, January 2026
- **arXiv:** 2601.01784
- **Method:** Dual-stream: Temporal Distance Stream (local) + Semantic Content Stream (global). Trace Disentanglement and Adaptation (TDA) isolates forgery fingerprints.
- **Performance:** +9% AP@0.95 over existing methods on ForgeryNet/TVIL
- **Relevance:** MEDIUM -- primarily video TFL but methodology applicable

### 12. Context-aware TFL (UniCaCLF) (**NEW**)
- **Title:** Context-aware TFL: A Universal Context-aware Contrastive Learning Framework for Temporal Forgery Localization
- **Authors:** Qilin Yin, Wei Lu, Xiangyang Luo, Xiaochun Cao
- **Venue:** arXiv preprint, June 2025
- **arXiv:** 2506.08493
- **Method:** Universal context-aware contrastive learning with heterogeneous activation operation and adaptive context updater.
- **Relevance:** LOW-MEDIUM -- general TFL framework

### 13. RegQAV -- Query-Based Audio-Visual TFL (**NEW**)
- **Title:** Query-Based Audio-Visual Temporal Forgery Localization with Register-Enhanced Representation Learning
- **Venue:** ACM MM 2025
- **DOI:** 10.1145/3746027.3755563
- **Method:** Register-enhanced query-based AV framework. Modality Fusion Adapter (MFA) for multi-scale AV integration. Deepfake Queries Generation (DQG) for query initialization.
- **Relevance:** LOW-MEDIUM -- AV domain but query-based localization is relevant

### 14. Precise Temporal Forgery Localization via Quantified AV Asynchrony (**NEW**)
- **Venue:** IEEE TIFS, August 2025
- **Method:** Coupled Pyramidal Encoder (CPE), Multi-Scale Asynchrony Probe (MAP), Context-Aware Boundary Pinpointing (CBP). Quantifies AV desynchronization for precise boundary regression.
- **Relevance:** MEDIUM -- published in our target venue (IEEE TIFS); demonstrates boundary pinpointing methodology

---

## Category C: EXPLAINABILITY & INTERPRETABILITY METHODS

### 15. SLIM -- Style-Linguistics Mismatch Model (**NEW**)
- **Title:** SLIM: Style-Linguistics Mismatch Model for Generalized Audio Deepfake Detection
- **Venue:** NeurIPS 2024
- **arXiv:** 2407.18517
- **Method:** Self-supervised pretraining on real samples to learn style-linguistics dependency. Quantifies (mis)match between style and linguistic content. Provides inherent explainability.
- **Performance:** Outperforms benchmarks on OOD with frozen encoders
- **Code:** Unknown
- **Relevance:** HIGH -- inherently explainable detection; directly relevant to our explainability narrative

### 16. Multi-Task Transformer for Explainable Detection via Formant Modeling (**NEW**)
- **Title:** Multi-Task Transformer for Explainable Speech Deepfake Detection via Formant Modeling
- **Authors:** Viola Negroni et al.
- **Venue:** arXiv preprint, January 2026
- **arXiv:** 2601.14850
- **Method:** Predicts formant trajectories and voicing patterns, classifies real/fake, highlights voiced/unvoiced region reliance. Built-in explainability with fewer parameters than baseline.
- **Relevance:** HIGH -- explainable detection via acoustic features; complement to our PDSM-PS approach

### 17. WST-X Series -- Wavelet Scattering Transform (**NEW**)
- **Title:** WST-X Series: Wavelet Scattering Transform for Interpretable Speech Deepfake Detection
- **Venue:** arXiv preprint, February 2026
- **arXiv:** 2602.02980
- **Method:** 1D and 2D wavelet scattering transforms as interpretable feature extractors. Combines wavelets with nonlinearities analogous to deep CNNs.
- **Performance:** Outperforms existing front-ends by wide margin on Deepfake-Eval-2024
- **Relevance:** HIGH -- interpretable feature extraction; relevant to PDSM-PS saliency discussion

### 18. SDD-APALLM -- Acoustic Evidence Perception in Audio LLMs (**NEW**)
- **Title:** Towards Explicit Acoustic Evidence Perception in Audio LLMs for Speech Deepfake Detection
- **Authors:** Xiaoxuan Guo, Xie et al.
- **Venue:** arXiv preprint, January 2026
- **arXiv:** 2601.23066
- **Method:** Combines raw audio with structured spectrograms to expose fine-grained time-frequency evidence. Addresses bias toward semantic cues in audio LLMs.
- **Relevance:** HIGH -- LLM-based explainability for deepfake detection

### 19. FT-GRPO -- Interpretable All-Type Detection via RL (**NEW**)
- **Title:** Interpretable All-Type Audio Deepfake Detection with Audio LLMs via Frequency-Time Reinforcement Learning
- **Authors:** Yuankun Xie et al.
- **Venue:** arXiv preprint, January 2026
- **arXiv:** 2601.02983
- **Method:** Automatic annotation pipeline constructs Frequency-Time structured CoT rationales. FT-GRPO: two-stage training (SFT + GRPO under rule-based frequency/time domain constraints). Inspired by DeepSeek-R1.
- **Relevance:** HIGH -- interpretable reasoning for deepfake detection; novel RL approach

### 20. HIR-SDD -- Human-Inspired Reasoning (**NEW**)
- **Title:** Towards Robust Speech Deepfake Detection via Human-Inspired Reasoning
- **Venue:** arXiv preprint, March 2026
- **arXiv:** 2603.10725
- **Method:** Combines Large Audio Language Models (LALMs) with chain-of-thought reasoning from human-annotated dataset. Provides human-like justifications for predictions.
- **Relevance:** MEDIUM -- explainability via reasoning; relevant to forensic narratives

### 21. Closing the Explainability Gap (**NEW**)
- **Title:** Toward Robust Real-World Audio Deepfake Detection: Closing the Explainability Gap
- **Authors:** Georgia Channing, Juil Sock, Ronald Clark, Philip Torr, Christian Schroder de Witt
- **Venue:** arXiv preprint, October 2024
- **arXiv:** 2410.07436
- **Method:** Novel explainability methods for transformer-based detectors. New benchmark for real-world generalizability. Attention roll-out mechanism for improved interpretability.
- **Relevance:** HIGH -- directly addresses explainability gap; must cite in related work

### 22. Forensic Deepfake Audio Detection Using Segmental Speech Features (**NEW**)
- **Title:** Forensic deepfake audio detection using segmental speech features
- **Authors:** Tianle Yang, Chengzhe Sun, Siwei Lyu, Phil Rose
- **Venue:** Forensic Science International (ScienceDirect), 2025; arXiv May 2025
- **arXiv:** 2505.13847
- **Method:** Speaker-specific deepfake detection using segmental acoustic features from forensic voice comparison (FVC). Highly interpretable due to articulatory grounding.
- **Relevance:** CRITICAL -- forensic-focused; directly relevant to our court-admissibility narrative

### 23. PLFD -- Phoneme-Level Feature Discrepancies (**NEW**)
- **Title:** Phoneme-Level Feature Discrepancies: A Key to Detecting Sophisticated Speech Deepfakes
- **Authors:** Kuiyuan Zhang, Zhongyun Hua, Rushi Lan, Yushu Zhang, Yifang Guo
- **Venue:** AAAI 2025
- **arXiv:** 2412.12619
- **Method:** Adaptive phoneme pooling extracts sample-specific phoneme-level features from frame-level data. Graph attention network models temporal dependencies. Random phoneme substitution augmentation.
- **Performance:** Superior to SOTA on four benchmark datasets
- **GitHub:** https://github.com/RedamancyAY/PLFD-ADD
- **Relevance:** CRITICAL -- phoneme-level analysis directly relevant to our PDSM-PS phoneme-discretized saliency

### 24. Phoneme-Level Analysis for Person-of-Interest Detection (**NEW**)
- **Title:** Phoneme-Level Analysis for Person-of-Interest Speech Deepfake Detection
- **Authors:** Davide Salvi, Viola Negroni, Sara Mandelli, Paolo Bestagini, Stefano Tubaro
- **Venue:** ICCV 2025 Workshop (APAI)
- **arXiv:** 2507.08626
- **Method:** POI-based detection at phoneme level. Decomposes reference audio into phonemes for speaker profile; compares test phonemes against profile.
- **Relevance:** HIGH -- phoneme-level forensic analysis

---

## Category D: ARCHITECTURE / DETECTION ADVANCES (Utterance-Level, but Relevant)

### 25. FGFM -- Fine-Grained Frame Modeling (**NEW**)
- **Title:** Fine-Grained Frame Modeling in Multi-head Self-Attention for Speech Deepfake Detection
- **Authors:** Tuan Dat Phuong, Duc-Tuan Truong, Long-Vu Hoang, Trang Nguyen Thi Thu
- **Venue:** arXiv preprint, February 2026
- **arXiv:** 2602.04702
- **Method:** Multi-head voting (MHV) module selects most informative frames; Cross-layer refinement (CLR) enhances subtle spoofing cue detection.
- **Performance:** EER 0.90% (LA21), 1.88% (DF21), 6.64% (ITW)
- **Relevance:** MEDIUM -- frame-level attention useful for localization context

### 26. f-InfoED -- Frame-Level Information Entropy Detector (**NEW**)
- **Title:** Generalized Audio Deepfake Detection Using Frame-level Latent Information Entropy
- **Venue:** arXiv preprint, April 2025
- **arXiv:** 2504.10819
- **Method:** Extracts information entropy from latent representations at frame level. AdaLAM extends pre-trained audio models with trainable adapters. Introduces ADFF 2024 dataset.
- **Relevance:** MEDIUM -- frame-level entropy features could inform localization

### 27. XLSR-MamBo (**NEW**)
- **Title:** XLSR-MamBo: Scaling the Hybrid Mamba-Attention Backbone for Audio Deepfake Detection
- **Authors:** Kwok-Ho Ng et al.
- **Venue:** arXiv preprint, January 2026
- **arXiv:** 2601.02944
- **Method:** XLSR front-end with synergistic Mamba-Attention backbones. Evaluates Mamba, Mamba2, Hydra, Gated DeltaNet topologies.
- **Relevance:** LOW-MEDIUM -- architecture advances; Mamba for efficient long-sequence processing

### 28. Fake-Mamba (**NEW**)
- **Title:** Fake-Mamba: Real-Time Speech Deepfake Detection Using Bidirectional Mamba as Self-Attention's Alternative
- **Authors:** Xi Xuan, Zimo Zhu, et al.
- **Venue:** IEEE ASRU 2025
- **arXiv:** 2508.09294
- **Method:** XLSR front-end + bidirectional Mamba. Three encoders: TransBiMamba, ConBiMamba, PN-BiMamba.
- **Performance:** EER 0.97% (LA21), 1.74% (DF21), 5.85% (ITW)
- **GitHub:** https://github.com/xuanxixi/Fake-Mamba
- **Relevance:** LOW -- utterance-level; but real-time + open code

### 29. BiCrossMamba-ST (**NEW**)
- **Title:** BiCrossMamba-ST: Speech Deepfake Detection with Bidirectional Mamba Spectro-Temporal Cross-Attention
- **Venue:** Interspeech 2025
- **arXiv:** 2505.13930
- **Method:** Dual-branch spectro-temporal architecture. BiMamba blocks with mutual cross-attention. 2D-Attention Map for critical region focusing.
- **Performance:** 67.74% and 26.3% relative gain over AASIST on LA21 and DF21
- **Relevance:** LOW-MEDIUM -- spectro-temporal analysis methodology

### 30. Quantizer-Aware Hierarchical Neural Codec Modeling (**NEW**)
- **Title:** Quantizer-Aware Hierarchical Neural Codec Modeling for Speech Deepfake Detection
- **Authors:** Jinyang Wu et al.
- **Venue:** arXiv preprint, March 2026
- **arXiv:** 2603.16914
- **Method:** Hierarchy-aware representation learning for RVQ codec artifacts. Learnable global weighting for quantizer-level contributions. Only 4.4% additional parameters.
- **Performance:** 46.2% relative EER reduction on ASVspoof 2019; 13.9% on ASVspoof 5
- **Relevance:** MEDIUM -- codec-aware detection relevant as speech editing uses codecs

---

## Category E: DATASETS & CHALLENGES

### 31. ASVspoof 5 Challenge Results (**NEW**)
- **Title:** ASVspoof 5: Evaluation of Spoofing, Deepfake, and Adversarial Attack Detection Using Crowdsourced Speech
- **Venue:** arXiv preprint, January 2026
- **arXiv:** 2601.03944
- **Details:** 53 participating teams. Crowdsourced database. Score calibration analysis. No dedicated localization track.
- **Note:** No partial spoof / localization track in ASVspoof 5 -- this is a key gap our work addresses.

### 32. ESDD2 -- Environment-Aware Deepfake Detection Challenge (**NEW**)
- **Title:** ESDD2: Environment-Aware Speech and Sound Deepfake Detection Challenge
- **Venue:** ICME 2026
- **arXiv:** 2601.07303
- **Details:** Component-level spoofing (speech + environmental sound independently manipulated). CompSpoofV2 dataset: 250k+ samples, ~283 hours.
- **Relevance:** LOW -- different task but represents latest challenge direction

### 33. MultiAPI Spoof (**NEW**)
- **Title:** MultiAPI Spoof: A Multi-API Dataset and Local-Attention Network for Speech Anti-spoofing Detection
- **Venue:** arXiv preprint, December 2025
- **arXiv:** 2512.07352
- **Details:** ~230 hours from 30 distinct APIs. Nes2Net-LA for local context. API tracing task.
- **Relevance:** LOW -- utterance-level but relevant for generalization discussion

---

## Category F: MISCELLANEOUS RELEVANT

### 34. Amplifying Artifacts with Speech Enhancement (**NEW**)
- **Title:** Amplifying Artifacts with Speech Enhancement in Voice Anti-spoofing
- **Authors:** Thanapat Trachu et al.
- **Venue:** Interspeech 2025
- **arXiv:** 2506.11542
- **Method:** Model-agnostic pipeline: noise addition -> speech enhancement -> artifact amplification.
- **Performance:** Up to 44.44% improvement on ASVspoof2019
- **Relevance:** MEDIUM -- preprocessing technique applicable to any detector

### 35. TADA -- Training-free Attribution (**NEW**)
- **Title:** TADA: Training-free Attribution and Out-of-Domain Detection of Audio Deepfakes
- **Venue:** Interspeech 2025
- **arXiv:** 2506.05802
- **GitHub:** https://github.com/adrianastan/tada
- **Method:** kNN-based source attribution using SSL features. F1 0.93 for grouping; 0.84 for OOD.
- **Relevance:** MEDIUM -- source attribution; complementary to localization

### 36. Few-Shot Voice Spoofing Adaptation (**NEW**)
- **Title:** Rapidly Adapting to New Voice Spoofing: Few-Shot Detection of Synthesized Speech Under Distribution Shifts
- **Venue:** arXiv preprint, August 2025
- **arXiv:** 2508.13320
- **Method:** Self-attentive prototypical network. Adapts with as few as 10 samples. Up to 32% relative EER reduction.
- **Relevance:** LOW-MEDIUM -- few-shot adaptation methodology

---

## PRIORITY CITATIONS FOR XPS-FORENSIC PAPER

### Must-Cite (directly relevant to our novelty claims):
1. **PLFD (AAAI 2025)** -- phoneme-level detection; most direct comparison to PDSM-PS
2. **Robust Localization Metrics (2507.03468)** -- validates need for better evaluation beyond EER
3. **PartialEdit (Interspeech 2025)** -- shows domain shift from PartialSpoof
4. **LlamaPartialSpoof (ICASSP 2025)** -- OOD evaluation dataset
5. **Manipulated Regions Survey (2506.14396, submitted TPAMI)** -- comprehensive survey, must position against
6. **Forensic Segmental Features (2505.13847)** -- forensic framing directly relevant
7. **SLIM (NeurIPS 2024)** -- explainable detection baseline
8. **Closing Explainability Gap (2410.07436)** -- explainability benchmark
9. **ASVspoof 5 Results (2601.03944)** -- confirms no localization track; validates our contribution
10. **BFC-Net** -- current SOTA on PartialSpoof; must compare

### Should-Cite (strengthens related work):
- LOCO (IJCAI 2025) -- weakly-supervised paradigm
- Multi-Task Formant Transformer (2601.14850) -- explainable formant modeling
- WST-X (2602.02980) -- interpretable features
- FGFM (2602.04702) -- frame-level attention
- Can LLMs Localize (2603.11205) -- LLM-based localization
- Phoneme POI (ICCV 2025W) -- phoneme forensic analysis

### Confirmed: NO existing work on conformal prediction for audio deepfake detection
- Web search returned zero results for "conformal prediction audio deepfake"
- CPSL novelty remains fully intact as of 2026-03-29
