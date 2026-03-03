# XAI Research for Audio Deepfake and Partial Spoof Detection (2024--2026)

**Compiled:** 2026-03-02
**Search databases:** Semantic Scholar, arXiv, OpenAlex, ISCA Archive, OpenReview
**Search date range:** 2024-01-01 to 2026-03-02

---

## 1. PDSM -- Phoneme Discretized Saliency Maps

### Paper

- **Title:** Phoneme Discretized Saliency Maps for Explainable Detection of AI-Generated Voice
- **Authors:** Shubham Gupta, Mirco Ravanelli, Pascal Germain, Cem Subakan
- **Year:** 2024
- **Venue:** Interspeech 2024, pp. 3295--3299
- **DOI:** 10.21437/Interspeech.2024-632
- **URL:** https://www.isca-archive.org/interspeech_2024/gupta24b_interspeech.pdf

### Method

PDSM proposes a discretization algorithm for saliency maps that leverages **phoneme boundaries** (obtained via forced alignment / phoneme posterior grams, PPGs) to make post-hoc explanations of AI-generated voice detectors more interpretable. Rather than producing a raw pixel/frame-level saliency heatmap on a magnitude spectrogram, PDSM aggregates saliency values within each phoneme segment. The method is model-agnostic and operates on top of standard gradient-based saliency methods (e.g., Integrated Gradients, GradSHAP).

### Key Results

- Tested on detection of **Tacotron2** and **Fastspeech2** synthesized speech.
- PDSM produces saliency maps that result in **more faithful explanations** compared to standard post-hoc explanation methods (standard saliency on magnitude spectrograms).
- Linking saliency representations to phoneme units yields explanations that are **more understandable** than standard saliency maps.
- Faithfulness evaluation uses perturbation-based metrics: the paper measures how well removing high-saliency regions degrades the classifier's confidence, following AOPC-style methodology. PDSM-discretized maps score higher on these faithfulness metrics than continuous saliency maps.

### Limitations

- Depends on forced alignment / ASR quality: phoneme boundary errors propagate into the explanation.
- Tested on a limited set of TTS systems (Tacotron2, Fastspeech2); generalization to diverse spoofing attacks, partial spoofs, or codec-degraded audio is not demonstrated.
- No evaluation on partial spoof datasets (PartialSpoof, ADD 2023 Track 2).

### Relevance to Judicial Partial Spoof Explainability

PDSM is highly relevant because it maps ML evidence to **linguistically meaningful units** (phonemes), which can be translated to words/syllables for a court audience. For partial spoof cases, PDSM could enable claims like "the model attributes high suspicion to phonemes /k/, /ae/, /t/ in the segment from 12.4s to 13.1s, corresponding to the word 'cat'." However, the current work does not address segment-level partial spoof localization directly; it would need to be combined with a temporal localizer. The faithfulness improvement over standard saliency is critical for Daubert-style cross-examination, where opposing experts would challenge whether the explanation actually reflects what the model uses.

---

## 2. Attention Rollout for Audio Deepfake Classifiers

### Paper

- **Title:** Toward Robust Real-World Audio Deepfake Detection: Closing the Explainability Gap
- **Authors:** Georgia Channing, Juil Sock, Ronald Clark, Philip Torr, Christian Schroeder de Witt
- **Year:** 2024
- **Venue:** arXiv preprint, arXiv:2410.07436 (cs.LG)
- **Citations:** 7 (as of 2026-03)
- **URL:** https://arxiv.org/abs/2410.07436

### Method

This paper introduces **novel explainability methods for state-of-the-art transformer-based audio deepfake detectors**, explicitly aiming to narrow the "explainability gap" between transformer-based detectors and traditional (more interpretable) methods. The paper also open-sources a benchmark for evaluating real-world generalizability.

The paper's core contribution is adapting attention rollout and related transformer visualization techniques (originally from Abnar & Zuidema, 2020, for NLP/vision) to the audio deepfake detection domain. Attention rollout recursively multiplies attention matrices across layers to produce a global attribution map showing which input regions the transformer attends to for its final classification decision.

### Key Results

- Demonstrates that attention-based explainability can **build trust with human experts** by providing visualizations of what the model focuses on in spectro-temporal space.
- Opens a benchmark for testing under real-world generalizability conditions.
- The results "pave the way for unlocking the potential of citizen intelligence to overcome the scalability issue in audio deepfake detection."

### Limitations

- The "attention is not explanation" critique (Jain & Wallace, 2019; Wiegreffe & Pinter, 2019) applies: attention weights show where the model looks, not necessarily what causally drives the decision.
- Needs validation for faithfulness (perturbation-based tests) to ensure attention maps are not misleading.
- Specific quantitative faithfulness metrics and dataset-specific results are not fully detailed in the abstract/preprint.

### Relevance to Judicial Partial Spoof Explainability

Attention rollout provides a **secondary supporting visualization** that can accompany more rigorous perturbation-based explanations (SHAP, PDSM). For courts, it is best presented as "the model's attention pattern is consistent with the flagged region" rather than as primary evidence. The paper's emphasis on real-world robustness and citizen intelligence scalability is relevant for deployment scenarios where audio evidence is submitted by non-expert parties.

---

## 3. Newer XAI Methods for Audio Deepfake Detection (2024--2026)

### 3.1 Relevancy-Based XAI for Transformer Deepfake Detectors

- **Title:** What Does an Audio Deepfake Detector Focus on? A Study in the Time Domain
- **Authors:** Petr Grinberg, Ankur Kumar, Surya Koppisetti, Gaurav Bharaj
- **Year:** 2025
- **Venue:** ICASSP 2025
- **arXiv:** 2501.13887
- **Citations:** 5

**Method:** Proposes a **relevancy-based explainable AI method** to analyze predictions of transformer-based audio deepfake detection models. Benchmarks against standard Grad-CAM and SHAP-based methods using quantitative faithfulness metrics and a **partial spoof test**.

**Key Results:**
- The proposed relevancy-based XAI method **performs best overall** on a variety of faithfulness metrics compared to Grad-CAM and SHAP.
- Critical finding: "XAI results obtained from a limited set of utterances do not necessarily hold when evaluated on large datasets." This warns against cherry-picking examples for explanations.
- Analysis of speech/non-speech regions, phonetic content, and voice onsets/offsets on large-scale data.

**Judicial Relevance:** Directly addresses the fidelity of explanations at scale -- essential for court contexts where the explanation must generalize, not just work on cherry-picked examples. The partial spoof test component makes it especially relevant.

### 3.2 Diffusion-Based Audio Deepfake Explanations

- **Title:** A Data-Driven Diffusion-based Approach for Audio Deepfake Explanations
- **Authors:** Petr Grinberg, Ankur Kumar, Surya Koppisetti, Gaurav Bharaj
- **Year:** 2025
- **Venue:** Interspeech 2025
- **arXiv:** 2506.03425

**Method:** Trains a **diffusion model** on paired genuine and vocoded audio to identify artifact regions. Uses the difference in time-frequency representations between real and vocoded pairs as supervised signal for explaining which regions contain deepfake artifacts.

**Key Results:**
- Outperforms traditional XAI techniques (SHAP, LRP) both qualitatively and quantitatively.
- Validated on VocV4 and LibriSeVoc datasets.

**Judicial Relevance:** Provides **ground-truth-aligned explanations** rather than post-hoc approximations. The fact that explanations are anchored to measurable differences between real and vocoded audio strengthens reproducibility and testability arguments under Daubert.

### 3.3 Phoneme-Level Person-of-Interest Detection

- **Title:** Phoneme-Level Analysis for Person-of-Interest Speech Deepfake Detection
- **Authors:** Davide Salvi, Viola Negroni, Sara Mandelli, Paolo Bestagini, Stefano Tubaro
- **Year:** 2025
- **Venue:** ICCV Workshop on Authenticity & Provenance in Generative AI
- **arXiv:** 2507.08626

**Method:** Decomposes reference audio into phonemes to build per-phoneme speaker profiles, then compares test phonemes individually for synthetic artifact detection. This is a **person-of-interest (POI)** approach rather than a universal detector.

**Key Results:**
- Achieves comparable accuracy to traditional approaches while offering **superior robustness and interpretability**.
- Fine-grained detection supports explainability by identifying which specific phonemes appear synthetic.

**Judicial Relevance:** Directly supports "which parts of this speaker's voice show synthetic artifacts?" questions. The per-phoneme granularity maps well to linguistic units that can be presented in court. The POI framing is natural for forensic cases where a specific speaker's identity is at issue.

### 3.4 Prosodic Feature-Based Detection

- **Title:** Pitch Imperfect: Detecting Audio Deepfakes Through Acoustic Prosodic Analysis
- **Authors:** Kevin Warren, Daniel Olszewski, Seth Layton, Kevin Butler, Carrie Gates, Patrick Traynor
- **Year:** 2025
- **Venue:** arXiv:2502.14726
- **arXiv:** 2502.14726

**Method:** Uses six classical prosodic features (pitch, intonation, jitter, shimmer, etc.) for detection. Attention mechanisms reveal feature importance: **Jitter, Shimmer, and Mean Fundamental Frequency** are most impactful.

**Key Results:**
- 93% accuracy, EER of 24.7%.
- Substantially more resistant to L-infinity adversarial attacks than competing models.
- Feature-level explanations are inherently interpretable (prosodic features are well-understood by speech experts).

**Judicial Relevance:** Classical prosodic features are understood by forensic phoneticians who already testify in courts. The approach bridges ML detection with traditional forensic voice analysis vocabulary. However, the relatively high EER (24.7%) limits standalone use.

### 3.5 Audio Language Model Reasoning Forensics

- **Title:** Analyzing Reasoning Shifts in Audio Deepfake Detection under Adversarial Attacks: The Reasoning Tax versus Shield Bifurcation
- **Authors:** Binh Nguyen, Thai Le
- **Year:** 2026
- **Venue:** arXiv:2601.03615 (submitted for ACL 2026)

**Method:** Develops a **forensic auditing framework** for audio language models (ALMs) used in deepfake detection. Examines reasoning traces along three dimensions: acoustic perception, cognitive coherence, and cognitive dissonance.

**Key Results:**
- For models with robust acoustic perception, reasoning functions as a **protective mechanism** (shield).
- Other models suffer performance degradation from reasoning (tax).
- **High cognitive dissonance can serve as a "silent alarm"** even when classification accuracy fails.

**Judicial Relevance:** The "reasoning trace as auditable evidence" concept directly supports the constrained-narrator LLM architecture described in the existing report. If an ALM's reasoning can be audited for coherence, this adds a layer of verifiability.

### 3.6 Named Entity-Enhanced Partial Audio Deepfake Detection

- **Title:** NE-PADD: Leveraging Named Entity Knowledge for Robust Partial Audio Deepfake Detection via Attention Aggregation
- **Authors:** Huhong Xian, Rui Liu, Berrak Sisman, Haizhou Li
- **Year:** 2025
- **Venue:** APSIPA ASC 2025
- **arXiv:** 2509.03829

**Method:** Integrates speech named entity recognition (SpeechNER) with partial audio deepfake detection (PADD) via two attention aggregation mechanisms: Attention Fusion (AF) and Attention Transfer (AT). Built on the **PartialSpoof-NER** dataset.

**Key Results:**
- Outperforms existing baselines on partial spoof detection.
- Named entity semantics improve frame-level fake speech positioning.

**Judicial Relevance:** Named entities (proper nouns, dates, numbers) are often the forensically critical content in partial spoofs (e.g., changing a name, amount, or date). Focusing detection on these semantically important regions directly addresses the "what was manipulated and does it matter?" question courts need answered.

### 3.7 SHAP-Based Attack Analysis (Earlier Work, Still Influential)

- **Title:** Explainable Deepfake and Spoofing Detection: An Attack Analysis Using SHAP
- **Authors:** Wanying Ge, Massimiliano Todisco, Nicholas Evans
- **Year:** 2022
- **Venue:** arXiv:2202.13693

- **Title:** Explaining Deep Learning Models for Spoofing and Deepfake Detection with SHAP
- **Authors:** Wanying Ge, Jose Patino, Massimiliano Todisco, Nicholas Evans
- **Year:** 2021
- **Venue:** arXiv:2110.03309

**Method:** Uses SHapley Additive exPlanations (SHAP) to identify attack-specific artifacts in spectrographic features. Shows which frequency bands and temporal regions the classifier relies on for different spoofing attacks.

**Judicial Relevance:** SHAP provides theoretically grounded (Shapley values) attributions with well-defined properties (local accuracy, missingness, consistency). The attack-specific analysis can support arguments about which artifacts the detector is using, and whether those artifacts are relevant to the specific spoofing technique at issue in a case.

### 3.8 FakeSound2 Benchmark for Explainable Detection

- **Title:** FakeSound2: A Benchmark for Explainable and Generalizable Deepfake Sound Detection
- **Authors:** Zeyu Xie, Yaoyun Zhang, Xuenan Xu, Yongkang Yin, Chenxing Li, Mengyue Wu, Yuexian Zou
- **Year:** 2025
- **Venue:** arXiv:2509.18461

**Method:** Introduces a benchmark that evaluates deepfake sound detection across three axes: **localization, traceability, and generalization**, with explicit emphasis on explainability.

**Judicial Relevance:** Provides a standardized benchmark for evaluating explainability claims -- essential for establishing that a detection method meets "general acceptance" criteria under Frye or "testability" under Daubert.

### 3.9 Normalized AOPC for Faithfulness Evaluation

- **Title:** Normalized AOPC: Fixing Misleading Faithfulness Metrics for Feature Attribution Explainability
- **Authors:** Joakim Edin, Andreas Geert Motzfeldt, Casper L. Christensen, Tuukka Ruotsalo, Lars Maaloe, Maria Maistro
- **Year:** 2024 (accepted ACL 2025)
- **arXiv:** 2408.08137

**Method:** Identifies that standard AOPC (Area Over the Perturbation Curve) is sensitive to model variations, producing unreliable cross-model comparisons. Proposes **Normalized AOPC (NAOPC)** for consistent cross-model evaluation.

**Key Results:** Normalization can "radically change AOPC results, questioning the conclusions of earlier studies."

**Judicial Relevance:** Directly relevant to evaluating the faithfulness of any saliency-based explanation (PDSM, SHAP, Grad-CAM) used in court. If opposing experts use different faithfulness metrics, NAOPC provides a more defensible comparison standard.

---

## 4. Calibration Methods Applied to Audio Spoof Detection

### 4.1 Score Calibration for ASV-CM Fusion

- **Title:** Revisiting and Improving Scoring Fusion for Spoofing-aware Speaker Verification Using Compositional Data Analysis
- **Authors:** Xin Wang, Tomi Kinnunen, Kong Aik Lee, Paul-Gauthier Noe, Junichi Yamagishi
- **Year:** 2024
- **Venue:** Interspeech 2024
- **arXiv:** 2406.10836
- **Citations:** 11

**Method:** Revisits score-level fusion between ASV and CM systems using decision theory and compositional data analysis. Three main findings:
1. **Score calibration before fusion is essential** -- raw CM scores are not calibrated log-likelihood ratios.
2. Linear combination of calibrated log-likelihood ratios from ASV and CM systems is effective.
3. Non-linear fusion methods can also be effective but require careful implementation.

**Key Results:** All three proposed improvements (pre-fusion calibration, refined linear fusion, enhanced non-linear fusion) demonstrated effectiveness on the **SASV challenge database**. Code is publicly available.

**Judicial Relevance:** This is the most directly relevant calibration paper for audio spoof detection. It demonstrates that **uncalibrated CM scores should not be interpreted as probabilities** -- a critical point for court testimony. The decision-theoretic framework provides a principled basis for operating point selection and error rate reporting. The emphasis on log-likelihood ratios aligns with forensic evidence evaluation standards (ENFSI guidelines on likelihood ratios).

### 4.2 Post-Hoc Calibration for Synthetic Media Detection

- **Title:** Enhancing Synthetic Generated-Images Detection through Post-Hoc Calibration
- **Authors:** Giulia Dimitri, Benedetta Tondi, Mauro Barni
- **Year:** 2025
- **Venue:** IEEE/CVF WACV Workshops 2025
- **DOI:** 10.1109/WACVW65960.2025.00087

**Method:** Applies post-hoc calibration methods (**temperature scaling, Platt scaling, isotonic regression**) to AI-generated image detectors. While focused on images, the methodology is directly transferable to audio deepfake detection.

**Key Results:**
- Post-hoc calibration improves **interpretability of outputs in terms of likelihood ratios**.
- Calibration helps **adjust detection thresholds** when different AI generators are present in training vs. calibration sets.
- Demonstrates practical benefits of calibration for forensic applications without retraining.

**Judicial Relevance:** Although image-focused, this paper provides the methodological blueprint for applying temperature scaling, Platt scaling, and isotonic regression to audio CM scores. The emphasis on likelihood ratio interpretability and threshold adjustment across generator types directly addresses Daubert concerns about known/potential error rates.

### 4.3 Calibrated Lightweight Anti-Spoofing

- **Title:** Lightweight Residual Network for Anti Spoofing in Speaker Verification
- **Authors:** Kunming Wang, Xianchang Fan, Ying Lei
- **Year:** 2025
- **Venue:** ACM ICSIE 2025
- **DOI:** 10.1145/3759179.3759195

**Method:** Uses a "calibrated lightweight classifier" as part of an anti-spoofing pipeline with log-power mel-spectrograms and phase-based group delay features.

**Key Results:** EER of 0.89% on ASVspoof 2019; real-time on commodity hardware.

**Judicial Relevance:** Demonstrates that calibrated classifiers can be part of production anti-spoofing pipelines, not just research prototypes.

### 4.4 Domain-Robust Calibration

- **Title:** Spoofing-Aware Speaker Verification Robust Against Domain and Channel Mismatches
- **Authors:** Chang Zeng, Xiaoxiao Miao, Xin Wang, Erica Cooper, Junichi Yamagishi
- **Year:** 2024
- **Venue:** SLT 2024
- **arXiv:** 2409.06327

**Method:** Proposes an integrated framework using pair-wise learning and spoofing attack simulation within a meta-learning paradigm. Addresses simultaneous robustness to spoofing, channel mismatch, and domain mismatch -- including score calibration across domains.

**Key Results:** Significantly improves over traditional ASV systems on the CNComplex dataset (combined threat scenarios).

**Judicial Relevance:** Addresses the critical court challenge of "was this system validated under conditions similar to the case evidence?" by explicitly handling domain/channel mismatch.

### Gap Analysis: Calibration for Audio Spoof Detection

**Current state:** There is no dedicated paper applying the full suite of post-hoc calibration methods (Platt scaling, temperature scaling, isotonic regression) specifically to audio deepfake/spoof detection CM scores. The Wang et al. (2024) paper on score fusion is the closest, demonstrating that calibration is essential. The Dimitri et al. (2025) paper demonstrates the methods on synthetic image detection and provides a transferable methodology.

**Research gap identified:** A systematic comparison of Platt scaling, temperature scaling, and isotonic regression applied specifically to CM scores from audio deepfake detectors (AASIST, wav2vec-based, WavLM-based) across multiple evaluation conditions (ASVspoof 2019/2021, PartialSpoof, ADD 2023) would fill this gap.

---

## 5. Uncertainty Quantification in Audio Deepfake Detection

### 5.1 FADEL -- Evidential Deep Learning for Fake Audio Detection

- **Title:** FADEL: Uncertainty-aware Fake Audio Detection with Evidential Deep Learning
- **Authors:** Ju Yeon Kang, Ji Won Yoon, Semin Kim, Min Hyun Han, Nam Soo Kim
- **Year:** 2025
- **Venue:** ICASSP 2025
- **arXiv:** 2504.15663
- **DOI:** 10.1109/ICASSP49660.2025.10888053

**Method:** Models class probabilities with a **Dirichlet distribution** via evidential deep learning, incorporating model uncertainty directly into predictions. Addresses the overconfidence problem of softmax-based classifiers when encountering out-of-distribution (OOD) spoofing attacks.

**Key Results:**
- Significantly improves performance on **ASVspoof2019 LA** and **ASVspoof2021 LA** datasets.
- Demonstrates **strong correlation between average uncertainty and EER** across different spoofing algorithms: higher uncertainty correlates with higher EER (more difficult attacks), validating the uncertainty estimates.
- Models can flag "I don't know" cases rather than making overconfident wrong predictions on unseen attacks.

**Judicial Relevance:** This is the most directly relevant uncertainty quantification paper for audio spoof detection. For judicial contexts:
- The system can explicitly quantify "how confident is the model in this specific prediction?" rather than just providing a score.
- High uncertainty on a case sample can be honestly reported as "the model indicates this sample is outside its validated operating range."
- The correlation between uncertainty and error rate provides empirical backing for reliability claims.
- Aligns with the NIST AI risk framework emphasis on understanding and communicating AI system limitations.

### 5.2 Bayesian Learning for Domain-Invariant Anti-Spoofing

- **Title:** Bayesian Learning for Domain-Invariant Speaker Verification and Anti-Spoofing
- **Authors:** Jin Li, Man-Wai Mak, Johan Rohdin, Kong Aik Lee, Hynek Hermansky
- **Year:** 2025
- **Venue:** Interspeech 2025
- **arXiv:** 2506.07536

**Method:** Proposes **Bayesian weighted RFN (BWRFN)** using variational inference to model posterior distributions of frequency normalization weights. Accounts for weight uncertainty due to domain shift.

**Key Results:**
- BWRFN significantly outperforms non-Bayesian variants (WRFN, RFN) on:
  - Cross-dataset ASV
  - Cross-TTS anti-spoofing
  - Spoofing-robust ASV

**Judicial Relevance:** The Bayesian approach provides principled uncertainty estimates that can be communicated as "the system accounts for the possibility that this audio comes from a different domain than the training data." This directly addresses the domain shift challenge raised by courts ("was the system trained on conditions like this case?").

### 5.3 Early-Exit Uncertainty for Efficient Detection

- **Title:** Efficient Audio Deepfake Detection using WavLM with Early Exiting
- **Authors:** Arthur Pimentel, Yi Zhu, Heitor R. Guimaraes, Tiago H. Falk
- **Year:** 2024
- **Venue:** IEEE WIFS 2024

**Method:** Uses early exiting from WavLM transformer layers as both an efficiency mechanism and an implicit uncertainty signal. Layers that disagree on classification provide an uncertainty indicator.

**Judicial Relevance:** The layer-disagreement signal could serve as an additional confidence indicator, but this is less principled than the Dirichlet-based approach (FADEL) or explicit Bayesian methods.

### Gap Analysis: Uncertainty Quantification

**Current state:**
- **Evidential deep learning (FADEL):** Applied to audio deepfake detection, validated on ASVspoof. This is the clearest entry point.
- **Bayesian methods:** Applied to domain-invariant anti-spoofing features (BWRFN). Not yet applied to the detection decision itself.
- **Conformal prediction:** **No published work** applies conformal prediction to audio deepfake or spoof detection as of 2026-03. This is a clear research gap.
- **MC Dropout:** **No published work** specifically applies MC dropout for uncertainty estimation in audio deepfake detection.

**Research gap identified:** Conformal prediction (Vovk et al., 2005; Angelopoulos & Bates, 2023) would provide distribution-free coverage guarantees (e.g., "with 95% probability, the true label is within this prediction set"). This is especially valuable for judicial contexts because conformal prediction's guarantees hold regardless of the model and data distribution, making them defensible under Daubert's "testability" criterion. A conformal prediction wrapper around existing CM classifiers (AASIST, WavLM-based) with calibration on held-out ASVspoof data would be a direct contribution.

---

## 6. Content Provenance and Watermarking for Audio

### 6.1 AudioSeal -- Localized Watermarking for Voice Cloning Detection

- **Title:** Proactive Detection of Voice Cloning with Localized Watermarking
- **Authors:** Robin San Roman, Pierre Fernandez, Alexandre Defossez, Teddy Furon, Tuan Tran, Hady ElSahar
- **Year:** 2024
- **Venue:** ICML 2024
- **arXiv:** 2401.17264
- **Citations:** 101
- **Code:** https://github.com/facebookresearch/audioseal

**Method:** First audio watermarking technique designed specifically for **localized detection** of AI-generated speech. Uses a joint generator/detector architecture with:
- **Localization loss:** Enables watermark detection precise to individual audio samples (not just "watermarked yes/no" but "watermarked in these specific regions").
- **Perceptual loss** inspired by auditory masking for imperceptibility.
- **Single-pass detector:** Up to two orders of magnitude faster than existing models.

**Key Results:**
- State-of-the-art robustness to real-life audio manipulations.
- Superior imperceptibility on automatic and human evaluation metrics.
- Fast single-pass detection suitable for large-scale and real-time applications.

**Judicial Relevance:** AudioSeal's **sample-level localization** is directly relevant to partial spoof scenarios. If a TTS system embeds an AudioSeal watermark, a forensic examiner could verify not just "this audio contains AI-generated content" but "the AI-generated content spans these specific sample ranges." This provides proactive provenance evidence that complements reactive detection. The localization capability maps directly to the "where is the manipulation?" question.

**Limitation for forensic use:** AudioSeal is proactive -- it only works if the generative model embeds the watermark. Adversarial actors may use unwatermarked systems. Therefore, AudioSeal complements but does not replace reactive detection.

### 6.2 WavMark -- Audio Watermarking Framework

- **Title:** WavMark: Watermarking for Audio Generation
- **Authors:** Guang Chen, Yu Wu, Shujie Liu, Tao Liu, Xiaoyong Du, Furu Wei
- **Year:** 2023
- **Venue:** arXiv:2308.12770
- **Citations:** 69

**Method:** Encodes up to **32 bits of watermark within a 1-second audio snippet**. Multiple segments can be combined for enhanced robustness.

**Key Results:**
- Average Bit Error Rate (BER) of **0.48%** across ten common attacks.
- Supports combining multiple watermark segments for reliability.

**Judicial Relevance:** The 32-bit capacity can encode provenance identifiers (model ID, generation timestamp, user hash) that establish chain-of-custody for AI-generated audio. However, see the O'Reilly et al. (2025) critique below.

### 6.3 Critique: Deep Audio Watermarks are Shallow

- **Title:** Deep Audio Watermarks are Shallow: Limitations of Post-Hoc Watermarking Techniques for Speech
- **Authors:** Patrick O'Reilly, Zeyu Jin, Jiaqi Su, Bryan Pardo
- **Year:** 2025
- **Venue:** arXiv:2504.10782
- **Citations:** 7

**Method:** Demonstrates that post-hoc audio watermarks (including WavMark and AudioSeal) can be **removed with minimal audio quality degradation** through transformation-based attacks.

**Key Results:**
- Unifies and extends evaluations of audio transformations on watermark detectability.
- Shows state-of-the-art post-hoc audio watermarks can be removed **with no knowledge of the watermarking scheme**.

**Judicial Relevance:** This is a critical caveat for forensic reliance on watermarking. Courts should be informed that:
1. Presence of a watermark provides positive evidence of provenance.
2. **Absence of a watermark does not prove the audio is authentic** (watermarks can be removed).
3. Watermarking should be one layer in a multi-evidence approach, not the sole provenance mechanism.

### 6.4 XAttnMark -- Cross-Attention Audio Watermarking

- **Title:** XAttnMark: Learning Robust Audio Watermarking with Cross-Attention
- **Authors:** Yixin Liu, Lie Lu, Jihui Jin, Lichao Sun, Andrea Fanelli
- **Year:** 2025
- **Venue:** arXiv:2502.04230

**Method:** Uses cross-attention mechanisms for message retrieval, partial parameter sharing, and psychoacoustic-aligned masking loss. Addresses the WavMark/AudioSeal limitation of struggling to achieve both robust detection and accurate attribution simultaneously.

**Key Results:** Superior performance against various audio transformations including generative editing.

### 6.5 TraceableSpeech -- Proactive Watermarking from TTS

- **Title:** TraceableSpeech: Towards Proactively Traceable Text-to-Speech with Watermarking
- **Authors:** Jun Zhou, Jiangyan Yi, Tao Wang, Jianhua Tao, Ye Bai, Chu Yuan Zhang, Yong Ren, Zhengqi Wen
- **Year:** 2024
- **Venue:** Interspeech 2024
- **arXiv:** 2406.04840
- **Citations:** 17

**Method:** Directly generates watermarked speech from TTS models (embeds watermark during generation, not post-hoc). Shows advantages over post-hoc application (e.g., WavMark).

**Judicial Relevance:** If TTS providers embed watermarks at generation time, this creates a stronger provenance chain than post-hoc watermarking. Relevant to regulatory frameworks (EU AI Act) requiring synthetic content labeling.

### 6.6 AudioMarkNet -- Watermarking for Deepfake Speech Detection

- **Title:** AudioMarkNet: Audio Watermarking for Deepfake Speech Detection
- **Authors:** W. Zong, Yang-Wai Chow, Willy Susilo, J. Baek, Seyit Ahmet Camtepe
- **Year:** 2025
- **Venue:** USENIX Security 2025
- **Citations:** 2

**Method:** Audio watermarking specifically designed for deepfake speech detection use cases. Published at a top security venue.

**Judicial Relevance:** Publication at USENIX Security provides peer-review credibility for security applications. The forensic use case framing is directly relevant.

### 6.7 C2PA for Broadcast Media

- **Title:** Interoperable Provenance Authentication of Broadcast Media using Open Standards-based Metadata, Watermarking and Cryptography
- **Authors:** John C. Simmons, Joseph M. Winograd
- **Year:** 2024
- **Venue:** IBC2024 Technical Papers Programme
- **arXiv:** 2405.12336

**Method:** Analyzes the interplay of **C2PA** (cryptographically authenticated metadata) and **ATSC** (audio/video watermarking) standards for broadcast provenance. Concludes these standards are "well suited to address broadcast provenance."

**Judicial Relevance:** C2PA is the leading industry standard for content provenance and provides a framework for:
- Cryptographic signing of audio content at creation.
- Embedding manifests with creation tool, timestamp, and modification history.
- Verification of unbroken provenance chains.

For audio forensics, C2PA manifests could serve as a digital chain-of-custody layer alongside traditional forensic hashing (SHA-256) and timestamping (RFC 3161). However, C2PA adoption in audio is behind image/video, and the standard's effectiveness depends on ecosystem-wide adoption.

### 6.8 EU AI Act Content Tagging

- **Title:** EU AI-Act: Tagging GenAI Content
- **Authors:** Julian Heeger, Waldemar Berchtold, Simon Bugert, Martin Steinebach
- **Year:** 2025
- **Venue:** Electronic Imaging, Vol. 37, Issue 4
- **DOI:** 10.2352/ei.2025.37.4.mwsf-301

**Method:** Proposes using ISO ISCC (International Standard Content Code) for AI-generated media identification under EU AI Act requirements. Incorporates hashing, digital signatures, and decentralized architecture.

**Judicial Relevance:** Addresses the regulatory framework for mandatory synthetic content labeling. Relevant to EU member state courts where the AI Act's transparency requirements may create new evidentiary standards.

### 6.9 Audio Codec Augmentation for Collaborative Watermarking

- **Title:** Audio Codec Augmentation for Robust Collaborative Watermarking of Speech Synthesis
- **Authors:** Lauri Juvela, Xin Wang
- **Year:** 2024
- **Venue:** ICASSP 2025
- **arXiv:** 2409.13382
- **Citations:** 8

**Method:** Addresses watermark robustness under audio codec transformations, a critical requirement for forensic scenarios where evidence may be compressed/transcoded.

---

## Summary: Research Gaps and Recommendations for the Report

### Confirmed and Documented

| Topic | Status | Key Papers |
|-------|--------|-----------|
| PDSM | Confirmed: Interspeech 2024 | Gupta et al., 2024 |
| Attention rollout | Confirmed: arXiv 2024 | Channing et al., 2024 |
| Relevancy-based XAI | Confirmed: ICASSP 2025 | Grinberg et al., 2025a |
| Diffusion-based explanations | Confirmed: Interspeech 2025 | Grinberg et al., 2025b |
| Phoneme-level POI detection | Confirmed: ICCV-W 2025 | Salvi et al., 2025 |
| Evidential deep learning (UQ) | Confirmed: ICASSP 2025 | Kang et al. (FADEL), 2025 |
| Bayesian anti-spoofing | Confirmed: Interspeech 2025 | Li et al. (BWRFN), 2025 |
| CM score calibration | Confirmed: Interspeech 2024 | Wang et al., 2024 |
| AudioSeal | Confirmed: ICML 2024 | San Roman et al., 2024 |
| Watermark vulnerability | Confirmed: arXiv 2025 | O'Reilly et al., 2025 |
| C2PA for audio | Confirmed: IBC 2024 | Simmons & Winograd, 2024 |
| Normalized AOPC | Confirmed: ACL 2025 | Edin et al., 2024 |

### Identified Research Gaps

1. **Conformal prediction for audio spoof detection:** No published work as of 2026-03. This would provide distribution-free coverage guarantees that are highly defensible in court.

2. **MC Dropout for audio deepfake uncertainty:** Not applied to audio CM systems specifically.

3. **Systematic post-hoc calibration comparison for audio CMs:** Temperature scaling vs. Platt scaling vs. isotonic regression specifically on AASIST / wav2vec / WavLM CM scores.

4. **PDSM extended to partial spoof localization:** The current PDSM paper tests only utterance-level detection; applying phoneme-discretized saliency to segment-level localization on PartialSpoof/ADD 2023 is unexplored.

5. **C2PA specifically for audio forensic evidence packages:** The Simmons & Winograd (2024) paper covers broadcast; judicial evidence chain-of-custody integration with C2PA is not addressed.

6. **Integrated XAI + calibration + uncertainty pipeline:** No paper combines explainability methods with calibrated scores and uncertainty quantification into a single forensic-ready pipeline.

---

## Appendix: Source URLs

```text
# PDSM
https://www.isca-archive.org/interspeech_2024/gupta24b_interspeech.pdf
https://www.isca-archive.org/interspeech_2024/gupta24b_interspeech.html

# Attention rollout / explainability gap
https://arxiv.org/abs/2410.07436

# Relevancy-based XAI (Grinberg et al. 2025a)
https://arxiv.org/abs/2501.13887

# Diffusion-based explanations (Grinberg et al. 2025b)
https://arxiv.org/abs/2506.03425

# Phoneme-level POI detection
https://arxiv.org/abs/2507.08626

# Prosodic analysis
https://arxiv.org/abs/2502.14726

# ALM reasoning forensics
https://arxiv.org/abs/2601.03615

# NE-PADD partial spoof
https://arxiv.org/abs/2509.03829

# FakeSound2 benchmark
https://arxiv.org/abs/2509.18461

# SHAP for spoofing detection
https://arxiv.org/abs/2202.13693
https://arxiv.org/abs/2110.03309

# Normalized AOPC
https://arxiv.org/abs/2408.08137

# Score calibration for SASV
https://arxiv.org/abs/2406.10836

# Post-hoc calibration for synthetic media
DOI: 10.1109/WACVW65960.2025.00087

# Domain-robust SASV
https://arxiv.org/abs/2409.06327

# FADEL (evidential deep learning)
https://arxiv.org/abs/2504.15663

# Bayesian anti-spoofing
https://arxiv.org/abs/2506.07536

# WavLM early exiting
WIFS 2024

# AudioSeal
https://arxiv.org/abs/2401.17264
https://github.com/facebookresearch/audioseal

# WavMark
https://arxiv.org/abs/2308.12770

# Deep Audio Watermarks are Shallow
https://arxiv.org/abs/2504.10782

# XAttnMark
https://arxiv.org/abs/2502.04230

# TraceableSpeech
https://arxiv.org/abs/2406.04840

# AudioMarkNet
USENIX Security 2025

# C2PA for broadcast media
https://arxiv.org/abs/2405.12336

# EU AI-Act content tagging
DOI: 10.2352/ei.2025.37.4.mwsf-301

# Audio codec augmentation for watermarking
https://arxiv.org/abs/2409.13382
```
