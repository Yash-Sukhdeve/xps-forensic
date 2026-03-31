# Evaluation Metrics for Partial Deepfake Speech Detection and Temporal Localization: A Systematic Analysis

**Compiled:** 2026-03-30
**Target venue:** IEEE/ACM TASLP survey paper
**Databases searched:** OpenAlex, arXiv, Semantic Scholar, direct PDF retrieval
**Search terms:** "segment EER partial spoof", "evaluation metric audio deepfake localization", "temporal IoU audio forgery", "range-based EER", "evaluation protocol fragmentation deepfake"

---

## Executive Summary

The evaluation landscape for partial deepfake speech detection and temporal localization is severely fragmented. This analysis traces **8 metric paradigms** across **10+ datasets**, identifies their provenance, motivation, limitations, and mutual incompatibilities. The central finding is that **no two major datasets use the same evaluation protocol**, creating a comparability crisis that impedes scientific progress.

Three key structural problems emerge:
1. **Resolution dependence:** Segment EER values are not comparable across different temporal resolutions (20ms vs 640ms), with the same model producing EER values ranging from 0.86% to 34.96% depending solely on resolution choice (Zhang et al., Interspeech 2023, Table 3).
2. **Threshold-free vs threshold-dependent tension:** EER is threshold-free but obscures deployment readiness; F1/precision/recall require threshold selection but reflect operational reality (Luong et al., arXiv:2507.03468).
3. **IoU formulation inconsistency:** At least three mutually incompatible IoU definitions coexist in the literature, making cross-dataset AP comparisons meaningless.

---

## 1. Segment EER (Multi-Resolution)

### 1.1 Original Paper

**First introduced:** Zhang, Wang, Cooper, Yamagishi, Patino, and Evans, "An Initial Investigation for Detecting Partially Spoofed Audio," Interspeech 2021, pp. 4264-4268. arXiv:2104.02518.

**Fully formalized with multi-resolution:** Zhang, Wang, Cooper, Evans, and Yamagishi, "The PartialSpoof Database and Countermeasures for the Detection of Short Fake Speech Segments Embedded in an Utterance," IEEE/ACM TASLP, vol. 31, pp. 813-825, 2023. DOI: 10.1109/TASLP.2022.3233236.

### 1.2 Motivation

The authors state that "the short spoofed speech segments to be embedded by attackers are of variable length" and therefore evaluate at "six different temporal resolutions [...] ranging from as short as 20 ms to as large as 640 ms" (Zhang et al., IEEE/ACM TASLP 2023).

The motivation is explicitly tied to the partial spoof scenario: unlike fully spoofed utterances where a single utterance-level decision suffices, partial spoofs require localizing WHICH segments are fake. Segment-level EER extends the classical utterance-level EER to sub-utterance granularity.

### 1.3 Definition

The utterance is divided into non-overlapping segments of fixed duration d (from the set {20, 40, 80, 160, 320, 640} ms). Each segment receives a single binary label: **spoof if ANY frame within the segment is spoof** (majority-of-one labeling). EER is computed in the standard way over these segment-level scores and labels.

Formally (from Zhang et al., Interspeech 2023, Eq. 1-4):

- P_FP(tau) = (1/|A_N^p|) * sum_{m in A_N^p} I(s_m < tau)
- P_FN(tau) = (1/|A_P^p|) * sum_{m in A_P^p} I(s_m >= tau)
- EER = (P_FP(tau_hat) + P_FN(tau_hat)) / 2

where A_N^p and A_P^p index bona fide and spoof segments, s_m is the segment score, and tau_hat is the threshold where FPR approximately equals FNR.

### 1.4 Datasets Mandating This Metric

- **PartialSpoof** (primary, mandatory at all 6 resolutions)
- Used as secondary metric by most methods evaluated on PartialSpoof (BAM, SAL, CFPRF, BFC-Net, etc.)

### 1.5 Limitations

1. **Resolution dependence (critical):** Zhang et al. (Interspeech 2023, Table 3) demonstrate that the SAME model with the SAME predicted scores produces drastically different EER values depending on measurement resolution. For a model trained at 20ms resolution:
   - Point-based EER at 10ms: 29.78%
   - Point-based EER at 20ms: 12.84% (source resolution)
   - Point-based EER at 640ms: 4.06%
   This means segment EER values are **not comparable** unless both training and measurement resolutions are identical.

2. **Coarse resolution underestimation:** "When the temporal resolution of the point-based reference is coarser than the temporal resolution of the training data, point-based EER may be an 'underestimation' in terms of the spoof localization performance, since the reference becomes too coarse and does not reflect accurate boundary information" (Zhang et al., Interspeech 2023).

3. **Majority-of-one labeling bias:** A segment containing 1 spoofed frame out of 32 (at 640ms) receives the same "spoof" label as one that is fully spoofed. This inflates apparent difficulty at coarser resolutions.

4. **Threshold-free but deployment-blind:** Luong et al. (arXiv:2507.03468) argue that "although EER is threshold-independent, which simplifies evaluation, they can be difficult to interpret in the context of real-world deployment" because "models that achieve strong EER scores across different evaluation sets may still perform poorly in deployment which requires a single threshold."

---

## 2. Range-Based EER

### 2.1 Original Paper

**Zhang, Wang, Cooper, Evans, and Yamagishi, "Range-Based Equal Error Rate for Spoof Localization," Interspeech 2023.** arXiv:2305.17739.

### 2.2 Motivation

The authors explicitly critique point-based (segment) EER: "Such point-based measurement overly relies on this resolution and may not accurately measure misclassified ranges. To properly measure misclassified ranges and better evaluate spoof localization performance, we upgrade point-based EER to range-based EER."

They further note: "there is currently no established way of properly measuring the performance of spoof localization" and that "the use of different measurements, such as counting the number of misclassified segments with fixed temporal resolutions (10 ms, 20 ms) or measuring the duration of misclassified regions, hinders the comparison of different spoof localization methods across the literature."

### 2.3 Definition

Instead of counting misclassified segments, range-based EER measures the total DURATION of misclassified regions. The key distinction from point-based measurement:

**Point-based:** Split audio into uniform segments of fixed resolution d, assign one label per segment, count misclassified segments.

**Range-based:** Record the boundaries of bona fide and spoof regions in both references and hypotheses. Measure the overlapped duration T(r_i, r_j) between ranges:

- T(r_i, r_j) = max(0, min(t_{i+1}, t_{j+1}) - max(t_i, t_j))

Range-based FPR and FNR (Eq. 5-6):
- P_FP(tau) = (1/D_N) * sum_{i in A_N^r} sum_j I(s_j < tau) * T(r_i, r_j)
- P_FN(tau) = (1/D_P) * sum_{i in A_P^r} sum_j I(s_j >= tau) * T(r_i, r_j)

where D_N and D_P are the total durations of bona fide and spoof ranges, and A_N^r and A_P^r index bona fide and spoof ranges in references.

The EER threshold is found via an adapted binary search algorithm (Algorithm 1 in the paper).

### 2.4 Datasets Mandating This Metric

- **PartialSpoof** (proposed for, demonstrated on)
- Not yet mandated by any challenge; proposed as replacement/complement to segment EER

### 2.5 Limitations

1. **Computational complexity:** The binary search algorithm for range-based EER is more complex than classical EER computation.
2. **Not yet widely adopted:** Despite being published in 2023, most subsequent papers still report point-based segment EER for backward compatibility.
3. **Still threshold-free:** Inherits the fundamental EER limitation of not reflecting deployment reality (Luong et al. critique still applies).

### 2.6 Key Experimental Finding

Zhang et al. (Interspeech 2023, Table 3) show that range-based EER provides a more stable estimate across resolutions than point-based EER. For a model trained at 20ms:
- Range-based EER (dev): 24.39%
- Point-based EER at 20ms (dev): 0.84%
- Point-based EER at 640ms (dev): 0.77%

This reveals that range-based EER is substantially higher (more honest) because it accounts for the actual duration of misclassified regions rather than counting discrete segments.

### 2.7 Naming Conventions

The paper explicitly documents the naming confusion: "Those two levels are called 'classical' and 'durative' in [20], and 'frame-based' as well as 'boundary-based' in [21]." The authors standardize terminology as "point-based" vs "range-based."

---

## 3. Frame-Level EER

### 3.1 Original Context

Frame-level EER treats each individual frame (typically 20ms) as an independent sample for EER computation. This is effectively segment EER at the finest possible resolution.

**Used by:** PartialEdit (Zhang, Tian, Zhang, Duan, Interspeech 2025, arXiv:2506.02958).

### 3.2 Motivation

Frame-level EER provides the finest granularity of localization assessment. When the segment size equals the frame size (typically 20ms, matching the SSL model's output resolution), segment EER and frame-level EER are identical.

### 3.3 Relationship to Segment EER

Frame-level EER is a special case of segment EER where the segment duration equals the frame duration. The distinction matters because:
- Frame-level EER at 20ms is identical to segment EER at 20ms resolution
- The term "frame-level" signals that each frame is treated as an i.i.d. sample, which may not hold for temporally correlated audio

### 3.4 Limitations

1. **i.i.d. assumption violation:** Adjacent frames in speech are highly correlated; treating them as independent samples inflates the effective sample size and produces overconfident EER estimates.
2. **Resolution-locked:** Unlike multi-resolution segment EER, frame-level EER is tied to a single granularity.
3. **All EER critiques from Luong et al. apply.**

---

## 4. Duration-Based F1

### 4.1 Original Papers

**HAD dataset:** Yi, Bai, Tao, Ma, Tian, Wang, Wang, and Fu, "Half-Truth: A Partially Fake Audio Detection Dataset," Interspeech 2021, pp. 1654-1658. arXiv:2104.03617.

**ADD 2023 challenge (formalized):** Yi, Tao, Fu, Yan, Wang, Wang, Zhang, Zhang, Zhao, Ren, Xu, Zhou, Gu, Wen, Liang, Lian, Nie, and Li, "ADD 2023: the Second Audio Deepfake Detection Challenge," IJCAI DADA Workshop 2023. arXiv:2305.13774.

### 4.2 Motivation

The HAD paper introduced the concept of localizing "small fake clips in real speech audio" and argued that partially fake audio "presents much more challenging than fully fake audio for fake audio detection." Duration-based F1 was chosen because it measures temporal overlap in seconds, directly reflecting how much of the manipulated region was correctly identified.

### 4.3 Definition (from ADD 2023, Section 3.3, Eq. 6-9)

The ADD 2023 RL track defines:

- **Segment Precision:** P_segment = TP / (TP + FP)
- **Segment Recall:** R_segment = TP / (TP + FN)
- **Segment F1:** F1_segment = (2 * P * R) / (P + R)

where TP, TN, FP, FN denote the numbers (equivalently, durations) of true positive, true negative, false positive, and false negative samples. The critical distinction: these are computed **based on the duration of each segment** in seconds, not by counting discrete frames or segments.

### 4.4 Datasets Mandating This Metric

- **HAD** (Half-truth Audio Detection) -- original usage
- **ADD 2023 Track 2 (RL)** -- mandatory, combined with sentence accuracy

### 4.5 Limitations

1. **Threshold-dependent:** Requires selection of a decision threshold, unlike EER. The ADD 2023 challenge does not specify how participants should select their thresholds.
2. **Class imbalance sensitivity:** Yi et al. (Interspeech 2021) noted these metrics "require a pre-defined threshold and have a high bias on imbalanced data" (cited as ref [17] in Zhang et al., Interspeech 2023).
3. **Duration aggregation ambiguity:** It is not fully specified whether TP/FP/FN durations are summed across all utterances in the test set or computed per-utterance and then averaged. This leads to potential inconsistency across implementations.

---

## 5. Sentence Accuracy

### 5.1 Original Paper

**ADD 2023 challenge:** Yi et al., arXiv:2305.13774, Section 3.3, Eq. 5.

### 5.2 Definition (ADD 2023, Eq. 5)

A_sentence = (TP + TN) / (TP + TN + FP + FN)

This is standard binary accuracy at the utterance level: the model must correctly classify the entire utterance as either genuine or containing manipulation. No localization information is evaluated.

### 5.3 Role in ADD 2023

Sentence accuracy is combined with duration-based F1 in the final scoring formula:

**Score = alpha * A_sentence + beta * F1_segment** (ADD 2023, Eq. 9)

where alpha = 0.3 and beta = 0.7. This weights localization (F1) more heavily than detection (accuracy), reflecting the challenge's focus on manipulation region location.

### 5.4 Limitations

1. **Coarse granularity:** Binary correct/incorrect for the entire utterance provides no localization feedback.
2. **Dominated by F1 in composite score:** With only 30% weight, sentence accuracy has limited influence on final rankings.
3. **Trivially achievable:** A system that labels all utterances as "fake" can achieve high accuracy on datasets with high fake prevalence.

---

## 6. AP@IoU (Average Precision at Intersection-over-Union thresholds)

### 6.1 Original Papers

**LAV-DF:** Cai, Stefanov, Dhall, and Hayat, "Do You Really Mean That? Content Driven Audio-Visual Deepfake Dataset and Multimodal Method for Temporal Forgery Localization," DICTA 2022. DOI: 10.1109/DICTA56598.2022.10034605. arXiv:2204.06228.

**Extended LAV-DF:** Cai, Ghosh, Dhall, Gedeon, Stefanov, and Hayat, "Glitch in the Matrix: A Large Scale Benchmark for Content Driven Audio-Visual Forgery Detection and Localization," CVIU, vol. 236, 103818, 2023. DOI: 10.1016/j.cviu.2023.103818. arXiv:2305.01979.

**AV-Deepfake1M:** Cai et al., with extensions in AV-Deepfake1M++ (arXiv:2507.20579).

### 6.2 Motivation

AP@IoU is borrowed from the **temporal action localization (TAL)** literature in computer vision (ActivityNet, THUMOS). The motivation is that temporal forgery localization is structurally identical to temporal action detection: the model must propose temporal segments and these proposals are evaluated against ground-truth segments using IoU overlap.

### 6.3 Definition

For each predicted temporal segment (proposal), IoU with each ground-truth segment is computed. A proposal is a true positive if IoU exceeds the threshold; otherwise it is a false positive. Precision-recall curves are computed, and Average Precision (AP) is the area under this curve.

Standard IoU thresholds reported in the literature:
- **LAV-DF:** AP@{0.5, 0.75, 0.9, 0.95} and mean AR@{50, 100}
- **AV-Deepfake1M:** AP@{0.5, 0.75, 0.95}

Recent results from the literature:
- DiMoDif (Koutlis & Papadopoulos, arXiv:2411.10193): "outperforms the state-of-the-art by 47.88 AP@0.75 on AV-Deepfake1M"
- AuViRe (Koutlis & Papadopoulos, WACV 2026, arXiv:2511.18993): "+8.9 AP@0.95 on LAV-DF, +9.6 AP@0.5 on AV-Deepfake1M"

### 6.4 Datasets Mandating This Metric

- **LAV-DF** (primary metric)
- **AV-Deepfake1M** (primary metric)
- Various temporal forgery localization benchmarks (ForgeryNet, TVIL)

### 6.5 Limitations

1. **Proposal-based assumption:** Requires the model to output discrete temporal proposals with start/end times, not frame-level scores. This architectural constraint excludes frame-level classifiers that produce per-frame scores without explicit proposal generation.

2. **IoU formulation inconsistency (CRITICAL):** At least three mutually incompatible IoU formulations coexist in the literature:

   **Formulation A -- Standard temporal IoU (TAL-style):**
   IoU = |Intersection| / |Union| = overlap_duration / (pred_duration + gt_duration - overlap_duration)
   Used by LAV-DF, AV-Deepfake1M, borrowed from ActivityNet.

   **Formulation B -- Jaccard index over frames (point-based IoU):**
   IoU = TP / (TP + FP + FN)
   where TP, FP, FN are counted over discrete frames. Used by Zhang and Sim, "Localizing fake segments in speech," ICPR 2022, pp. 3224-3230 (ref [15] in Zhang et al., Interspeech 2023). This is frame-counting, not duration-based.

   **Formulation C -- Anomalous "detection quality" IoU:**
   IoU = (TP + TN) / (TP + TN + 2*(FP + FN))
   Identified in He et al. survey (arXiv:2506.14396, submitted IEEE TPAMI): "system will be considered as a good detector if IoU > 1/3." This includes true negatives in the numerator and double-counts errors, making it fundamentally different from standard IoU.

   These three formulations produce **incomparable values** for the same prediction/ground-truth pair. A survey claiming "IoU" without specifying which formulation is used creates systematic ambiguity.

3. **Threshold sensitivity:** AP values vary dramatically across IoU thresholds (AP@0.5 can be 10x higher than AP@0.95), making headline AP numbers misleading without specifying the threshold.

4. **Audio-visual bias:** AP@IoU was developed for and primarily used by audio-visual deepfake datasets. Its applicability to audio-only partial spoof detection (where manipulations may be scattered micro-segments rather than contiguous blocks) is questionable.

---

## 7. Utterance-Level EER

### 7.1 Original Context

Utterance-level EER is the classical metric for automatic speaker verification (ASV) anti-spoofing, dating back to the ASVspoof challenges.

**Standardized by:** Wu, Kinnunen, Evans, Yamagishi, Hanilci, Sahidullah, and Sizov, "ASVspoof 2015: The First Automatic Speaker Verification Spoofing and Countermeasures Challenge," Interspeech 2015, pp. 2037-2041. (ref [2] in Zhang et al., Interspeech 2023).

**For deepfake detection:** Todisco, Wang, Vestman, Sahidullah, Delgado, Nautsch, Yamagishi, Evans, Kinnunen, and Lee, "ASVspoof 2019: Future Horizons in Spoofed and Fake Audio Detection," Interspeech 2019. DOI: 10.21437/Interspeech.2019-2249.

### 7.2 Definition

One score per utterance; one binary label per utterance (bonafide vs spoof). EER is the threshold where FAR = FRR.

In the context of partial spoof: an utterance containing ANY spoofed segment is labeled "spoof" at the utterance level. The model must detect the PRESENCE (not location) of manipulation.

### 7.3 Datasets Mandating This Metric

- **ASVspoof 2019 LA/PA** (primary metric alongside t-DCF)
- **ASVspoof 2021 LA/DF** (primary)
- **ASVspoof 5** (primary; DOI: 10.21437/asvspoof.2024-1)
- **PartialSpoof** (reported alongside segment EER)
- **LlamaPartialSpoof** (primary; Luong et al., ICASSP 2025)

### 7.4 Limitations for Partial Spoof

1. **No localization information:** A model achieving 0% utterance EER could have zero ability to locate the spoofed segments.
2. **Trivially solvable via shortcuts:** If partially spoofed utterances have detectable concatenation artifacts at boundaries, a model can achieve low EER without understanding spoof content.
3. **Complementary but insufficient:** Zhang et al. (IEEE/ACM TASLP 2023) showed that their model achieves 0.77% utterance-level EER on PartialSpoof but segment-level EER is substantially higher, confirming that utterance-level success does not imply localization capability.

---

## 8. Boundary F1

### 8.1 Context

Boundary F1 evaluates the precision of detecting transition points between bonafide and spoofed segments, rather than classifying each frame or segment.

**Used by:** PartialEdit (Zhang, Tian, Zhang, Duan, Interspeech 2025, arXiv:2506.02958).

**Related work:** BFC-Net (Boundary-Frame Cross Graph Attention Network, Neurocomputing 2025) explicitly models boundaries via its Boundary-Frame Cross Attention module.

### 8.2 Definition

A boundary is correctly detected if a predicted transition point falls within a tolerance window (typically 20ms = 1 frame) of a ground-truth transition. Standard precision/recall/F1 are then computed over boundary events.

### 8.3 Motivation

Boundary F1 directly measures what forensic examiners care about: "where exactly does the manipulation begin and end?" This is more interpretable than frame-level classification accuracy and more directly useful for evidence presentation.

### 8.4 Limitations

1. **Tolerance sensitivity:** Performance is highly sensitive to the tolerance window. A 20ms tolerance is strict but may be too lenient for high-precision forensic applications.
2. **Ignores interior accuracy:** A model could correctly detect all boundaries but misclassify the segments between them.
3. **Undefined for gradual transitions:** Neural speech editing (as in PartialEdit) may produce smooth transitions without sharp boundaries, making boundary detection ill-defined.
4. **Sparse events:** In utterances with few manipulated regions, there are very few boundary events, leading to high-variance F1 estimates.

---

## 9. Luong et al. (arXiv:2507.03468) -- Detailed Critique of EER for Localization

### 9.1 Paper Details

**Title:** "Robust Localization of Partially Fake Speech: Metrics and Out-of-Domain Evaluation"
**Authors:** Hieu-Thi Luong, Inbal Rimon, Haim Permuter, Kong Aik Lee, Eng Siong Chng
**Venue:** arXiv preprint, July 2025

### 9.2 Core Argument

Luong et al. make five interconnected arguments against EER as the primary localization metric:

**Argument 1 -- EER obscures deployment readiness:**
"Equal Error Rate (EER), which often obscures generalization and deployment readiness." The fundamental problem is that EER is computed at the OPTIMAL threshold for each test set independently. In deployment, a single threshold must be chosen.

**Argument 2 -- In-domain EER is misleading:**
CFPRF achieves "7.61% EER on the in-domain PartialSpoof evaluation set, but 43.25% and 27.59% on the LlamaPartialSpoof and Half-Truth out-of-domain test sets." The 7.61% headline number creates false confidence.

**Argument 3 -- Threshold instability across domains:**
Figure 1 in their paper shows CFPRF's EER threshold shifts from 0.0880 (in-domain) to 0.8828 (out-of-domain). This means "each result corresponds to different decision thresholds," making EER comparisons across domains meaningless from a deployment perspective.

**Argument 4 -- Over-optimization for in-domain EER:**
"Over-optimizing for in-domain EER...can lead to models that perform poorly in real-world scenarios." Their reproduced CFPRF shows "worse on in-domain data (9.84%) but better on the out-of-domain sets (41.72% and 14.98%)," demonstrating an inverse relationship between in-domain EER and cross-domain generalization.

**Argument 5 -- Class definition ambiguity:**
"EER and accuracy remains the same, while precision, recall, and F1 differ depending on which class is defined as the positive class." At 20ms resolution with 95% recall requirement, CFPRF achieves 81.35% precision vs MRM's 53.95% -- a distinction completely invisible in EER.

### 9.3 Proposed Alternative: Sequential Anomaly Detection Framing

Luong et al. reframe partial spoof localization as "a sequential anomaly detection problem" where "fake segment defined as the positive class, in alignment with standard anomaly detection practices." They advocate for:
- Accuracy (overall correctness)
- Precision (what fraction of predicted-fake is actually fake)
- Recall (what fraction of actual-fake is detected)
- F1-score (harmonic mean)

These are threshold-DEPENDENT metrics, requiring explicit threshold selection, which the authors argue is a FEATURE not a bug: it forces researchers to confront the threshold selection problem that EER hides.

### 9.4 Training Data Paradox

A critical finding: "adding more bona fide or fully synthetic utterances to the training data often degrades performance, whereas adding partially fake utterances improves it." This challenges the assumption that larger, more diverse training sets always help and has implications for metric selection (diverse training should improve threshold-dependent metrics more than EER).

---

## 10. Cross-Metric Comparisons and Fragmentation

### 10.1 Papers Explicitly Comparing Metrics

**Zhang et al. (Interspeech 2023, arXiv:2305.17739):** The most systematic comparison, evaluating point-based EER vs range-based EER across all 6 resolutions on PartialSpoof. Key finding: range-based EER is resolution-independent and generally yields higher (more conservative) estimates than point-based EER at coarse resolutions.

**Luong et al. (arXiv:2507.03468):** Compares EER against precision/recall/F1, demonstrating that EER rankings do not necessarily agree with threshold-dependent metric rankings, especially cross-domain.

**He et al. (arXiv:2506.14396, submitted IEEE TPAMI):** The survey identifies and catalogs all metric paradigms but notes that "performance comparisons remain difficult because methods report results 'with PartialSpoof train set' versus 'trained on the training set of the LAV-DF dataset,' creating non-comparable evaluations."

### 10.2 Metric Fragmentation Map

| Dataset | Primary Metric(s) | Resolution | Paper |
|---------|-------------------|------------|-------|
| PartialSpoof | Segment EER (6 resolutions) | 20-640ms | Zhang et al., TASLP 2023 |
| PartialSpoof (proposed) | Range-based EER | continuous | Zhang et al., IS 2023 |
| HAD | Duration-based P/R/F1 | seconds | Yi et al., IS 2021 |
| ADD 2023 Track 2 | 0.3*Acc + 0.7*F1_segment | seconds | Yi et al., DADA 2023 |
| LAV-DF | AP@{0.5,0.75,0.9,0.95} | proposals | Cai et al., DICTA 2022 |
| AV-Deepfake1M | AP@{0.5,0.75,0.95} | proposals | Cai et al., CVIU 2023 |
| ASVspoof 2019-2021 | Utterance EER + t-DCF | utterance | Todisco et al., IS 2019 |
| ASVspoof 5 | Utterance EER + a-DCF | utterance | Wang et al., 2024 |
| LlamaPartialSpoof | Utterance EER | utterance | Luong et al., ICASSP 2025 |
| PartialEdit | Frame EER + Boundary F1 | 20ms | Zhang et al., IS 2025 |

**Critical observation:** No two localization-focused datasets (PartialSpoof, HAD, ADD 2023 RL, LAV-DF, AV-Deepfake1M, PartialEdit) use the same primary evaluation metric. This makes cross-dataset performance comparison fundamentally impossible.

### 10.3 The Three Incompatible IoU Formulations

**Formulation A -- Standard temporal IoU (proposal-level):**
IoU(p, g) = |p intersect g| / |p union g|
- Used by: LAV-DF, AV-Deepfake1M (borrowed from temporal action localization)
- Operates on temporal proposals (start, end) pairs
- Standard in computer vision TAL literature

**Formulation B -- Jaccard index over frames (frame-level):**
IoU = TP / (TP + FP + FN)
- Used by: Zhang and Sim, ICPR 2022
- Operates on discrete frame-level predictions
- Equivalent to Jaccard similarity over binary frame vectors
- Referenced as "point-based IoU" in Zhang et al. (Interspeech 2023)

**Formulation C -- Detection quality IoU (global):**
IoU = (TP + TN) / (TP + TN + 2*(FP + FN))
- Documented in He et al. survey (arXiv:2506.14396)
- Includes true negatives (unusual for IoU)
- Double-weights errors
- "System will be considered as a good detector if IoU > 1/3"
- Provenance unclear; appears to be from Chinese audio forensics community

These three formulations are mutually incompatible. For a given prediction/ground-truth pair:
- Formulation A operates on temporal proposals (continuous)
- Formulation B operates on binary frame arrays (discrete)
- Formulation C incorporates TN (fundamentally different semantic meaning)

A paper reporting "IoU = 0.75" without specifying the formulation is ambiguous and potentially misleading.

---

## 11. Evaluation Protocol Fragmentation

### 11.1 Structural Sources of Fragmentation

1. **Community divide:** The partial spoof community (ASVspoof lineage) uses EER variants; the temporal forgery localization community (computer vision lineage) uses AP@IoU. These communities largely do not cite each other's metrics papers.

2. **Challenge-driven balkanization:** Each challenge (ASVspoof, ADD, DFGC, 1M-Deepfakes) defines its own evaluation protocol. Challenge organizers prioritize backward compatibility with their own prior editions over cross-challenge comparability.

3. **Audio-only vs audio-visual split:** Audio-only datasets (PartialSpoof, HAD) use frame/segment-level metrics; audio-visual datasets (LAV-DF, AV-Deepfake1M) use proposal-based AP@IoU. The underlying task is similar but the metrics are architecturally incompatible.

### 11.2 Papers Discussing Fragmentation

**He et al. (arXiv:2506.14396, submitted IEEE TPAMI 2025):** The most comprehensive survey on manipulated regions localization for partially deepfake audio. Catalogs all metrics but identifies that "different datasets employ different metrics across various resolutions (10ms, 20ms, 40ms, 160ms), preventing direct performance comparison."

**Li, Ahmadiadli, and Zhang, "A Survey on Speech Deepfake Detection," ACM Computing Surveys, 2024.** arXiv:2404.13914. Reviews 200+ papers; notes evaluation metric inconsistency as a challenge but does not propose solutions.

**Luong et al. (arXiv:2507.03468):** Most pointed critique, arguing the field needs to move beyond EER toward deployment-relevant metrics.

**Zhang et al. (Interspeech 2023, arXiv:2305.17739):** Demonstrates that even within the EER family, point-based vs range-based produce non-comparable values; proposes range-based EER as standardization.

### 11.3 Standardization Proposals

| Proposal | Source | Metric | Status |
|----------|--------|--------|--------|
| Range-based EER | Zhang et al., IS 2023 | Resolution-independent EER | Proposed; not widely adopted |
| Threshold-dependent metrics | Luong et al., 2025 | Precision/Recall/F1 at fixed threshold | Proposed; open-source library released |
| Composite score | ADD 2023 | 0.3*Acc + 0.7*F1 | Challenge-specific; not generalizable |
| AP@IoU standardization | LAV-DF/AV-Deepfake1M | TAL-style AP | De facto standard for AV domain only |

**No cross-community standardization proposal exists as of March 2026.**

---

## 12. New Metrics Proposed (2024-2026)

### 12.1 Luong et al. Threshold-Dependent Framework (2025)

As detailed in Section 9 above. Open-source evaluation library for precision/recall/F1/accuracy with explicit threshold selection.

### 12.2 He et al. Survey Taxonomy (2025)

The He et al. survey (arXiv:2506.14396) attempts to create a unified taxonomy of metrics for the field but does not propose a new metric per se. Their categorization:
- Segment-level: EER variants, P/R/F1
- Duration-based: Range EER, duration F1
- Proposal-based: AP@IoU, AR@N
- Utterance-level: EER, accuracy, t-DCF

### 12.3 Composite Metrics in Practice

The ADD 2023 composite (0.3*Acc + 0.7*F1) represents an attempt to jointly evaluate detection and localization. However, the specific weight choice (0.3/0.7) is arbitrary and the formula has not been adopted outside the ADD challenge.

---

## 13. Summary Table: Metric Paradigm Comparison

| Metric | Granularity | Threshold? | Resolution-Dep? | Localization? | Original Source | Datasets |
|--------|-------------|------------|------------------|---------------|-----------------|----------|
| Segment EER | Segment | Free | YES (critical) | Yes | Zhang et al., IS 2021 | PartialSpoof |
| Range-based EER | Continuous | Free | No | Yes | Zhang et al., IS 2023 | PartialSpoof (proposed) |
| Frame-level EER | Frame (20ms) | Free | Fixed at frame | Yes | PartialEdit 2025 | PartialEdit |
| Duration-based F1 | Continuous | Required | No | Yes | Yi et al., IS 2021 | HAD, ADD 2023 |
| Sentence Accuracy | Utterance | Required | N/A | No | ADD 2023 | ADD 2023 |
| AP@IoU | Proposal | Varies | No | Yes | Cai et al., DICTA 2022 | LAV-DF, AV-Deepfake1M |
| Utterance EER | Utterance | Free | N/A | No | ASVspoof 2015 | All ASVspoof, PS |
| Boundary F1 | Event | Required | Tolerance-dep | Boundary only | PartialEdit 2025 | PartialEdit |

---

## 14. Recommendations for Survey Paper

### 14.1 Key Claims to Make (Evidence-Backed)

1. **Metric fragmentation is the primary barrier to progress.** No two localization datasets use the same metric, preventing meaningful cross-dataset comparison. (Evidence: Section 10.2 table; He et al. survey; Zhang et al. range-based EER paper.)

2. **EER is necessary but insufficient for localization evaluation.** EER provides a threshold-free baseline but obscures deployment-relevant behavior. (Evidence: Luong et al. full argument, Section 9.)

3. **Resolution dependence invalidates naive segment EER comparison.** The same model produces EER from 0.86% to 34.96% depending on resolution. (Evidence: Zhang et al., Interspeech 2023, Table 3.)

4. **Three incompatible IoU formulations prevent AP@IoU standardization.** (Evidence: Section 10.3; Zhang et al. ref [15]; He et al. survey.)

5. **The field needs a multi-metric evaluation protocol** combining threshold-free (EER for benchmarking), threshold-dependent (F1 at operational thresholds), and localization-specific (boundary F1, AP@IoU) metrics.

### 14.2 Open Questions

1. Should the community converge on range-based EER or threshold-dependent metrics (or both)?
2. How should IoU be standardized for audio-only partial spoof (point-based frames vs continuous ranges)?
3. Should boundary detection be evaluated separately from segment classification?
4. What is the appropriate multi-metric protocol for a unified leaderboard?

---

## 15. Complete Bibliography

### Primary Metric Papers

1. Zhang, L., Wang, X., Cooper, E., Yamagishi, J., Patino, J., Evans, N. "An Initial Investigation for Detecting Partially Spoofed Audio." Interspeech 2021, pp. 4264-4268. arXiv:2104.02518. [First segment-level evaluation for partial spoof]

2. Zhang, L., Wang, X., Cooper, E., Evans, N., Yamagishi, J. "The PartialSpoof Database and Countermeasures for the Detection of Short Fake Speech Segments Embedded in an Utterance." IEEE/ACM TASLP, vol. 31, pp. 813-825, 2023. DOI: 10.1109/TASLP.2022.3233236. [Multi-resolution segment EER formalization]

3. Zhang, L., Wang, X., Cooper, E., Evans, N., Yamagishi, J. "Range-Based Equal Error Rate for Spoof Localization." Interspeech 2023. arXiv:2305.17739. [Range-based EER proposal]

4. Yi, J., Bai, Y., Tao, J., Ma, H., Tian, Z., Wang, C., Wang, T., Fu, R. "Half-Truth: A Partially Fake Audio Detection Dataset." Interspeech 2021, pp. 1654-1658. arXiv:2104.03617. [Duration-based F1 for HAD]

5. Yi, J., Tao, J., Fu, R., et al. "ADD 2023: the Second Audio Deepfake Detection Challenge." IJCAI DADA Workshop, 2023. arXiv:2305.13774. [Composite score: 0.3*Acc + 0.7*F1; sentence accuracy formalization]

6. Cai, Z., Stefanov, K., Dhall, A., Hayat, M. "Do You Really Mean That? Content Driven Audio-Visual Deepfake Dataset and Multimodal Method for Temporal Forgery Localization." DICTA 2022. DOI: 10.1109/DICTA56598.2022.10034605. arXiv:2204.06228. [AP@IoU for LAV-DF]

7. Cai, Z., Ghosh, S., Dhall, A., Gedeon, T., Stefanov, K., Hayat, M. "Glitch in the Matrix: A Large Scale Benchmark for Content Driven Audio-Visual Forgery Detection and Localization." CVIU, vol. 236, 103818, 2023. DOI: 10.1016/j.cviu.2023.103818. arXiv:2305.01979. [Extended LAV-DF with BA-TFD+]

8. Luong, H.-T., Rimon, I., Permuter, H., Lee, K.A., Chng, E.S. "Robust Localization of Partially Fake Speech: Metrics and Out-of-Domain Evaluation." arXiv:2507.03468, July 2025. [EER critique; threshold-dependent metrics proposal]

9. Zhang, Y., Tian, B., Zhang, L., Duan, Z. "PartialEdit: Identifying Partial Deepfakes in the Era of Neural Speech Editing." Interspeech 2025. arXiv:2506.02958. [Frame-level EER + Boundary F1]

### Foundational Metric Papers

10. Wu, Z., Kinnunen, T., Evans, N., Yamagishi, J., Hanilci, C., Sahidullah, M., Sizov, A. "ASVspoof 2015: The First Automatic Speaker Verification Spoofing and Countermeasures Challenge." Interspeech 2015, pp. 2037-2041. [Utterance-level EER for anti-spoofing]

11. Todisco, M., Wang, X., Vestman, V., Sahidullah, M., Delgado, H., Nautsch, A., Yamagishi, J., Evans, N., Kinnunen, T., Lee, K.A. "ASVspoof 2019: Future Horizons in Spoofed and Fake Audio Detection." Interspeech 2019. DOI: 10.21437/Interspeech.2019-2249. [EER + t-DCF standardization]

12. Wang, X., Delgado, H., Tak, H., et al. "ASVspoof 5: Crowdsourced Speech Data, Deepfakes, and Adversarial Attacks at Scale." 2024. DOI: 10.21437/asvspoof.2024-1. [Latest ASVspoof evaluation]

13. Kinnunen, T., Lee, K.A., Delgado, H., Evans, N., Todisco, M., Sahidullah, M., Yamagishi, J., Reynolds, D.A. "t-DCF: A Tandem Detection Cost Function for the ASVspoof 2019." Proc. Odyssey 2018. [t-DCF metric]

### Surveys

14. He, J., Yi, J., Tao, J., Zeng, S., Gu, H. "Manipulated Regions Localization For Partially Deepfake Audio: A Survey." Submitted to IEEE TPAMI, June 2025. arXiv:2506.14396. [Most comprehensive metric taxonomy]

15. Li, M., Ahmadiadli, Y., Zhang, X.-P. "A Survey on Speech Deepfake Detection." ACM Computing Surveys, 2024. arXiv:2404.13914. [200+ paper review; notes metric inconsistency]

### IoU Reference Papers

16. Zhang, B. and Sim, T. "Localizing fake segments in speech." ICPR 2022, pp. 3224-3330. [Point-based IoU / Jaccard index for frame-level spoof localization]

17. Luong, H.-T., et al. "LlamaPartialSpoof: An LLM-Driven Fake Speech Dataset Simulating Disinformation Generation." ICASSP 2025. arXiv:2409.14743. [OOD evaluation with utterance EER]

### Recent Temporal Forgery Localization

18. Koutlis, C. and Papadopoulos, S. "DiMoDif: Discourse Modality-information Differentiation for Audio-visual Deepfake Detection and Localization." arXiv:2411.10193. [AP@0.75 on AV-Deepfake1M]

19. Koutlis, C. and Papadopoulos, S. "AuViRe: Audio-visual Speech Representation Reconstruction for Deepfake Temporal Localization." WACV 2026. arXiv:2511.18993. [AP@0.95 on LAV-DF]

---

*End of analysis. All claims are evidence-backed with specific paper citations and, where available, direct quotations from the source material.*
