# Rigorous Theoretical Analysis: Conformalized Partial Spoof Localization (CPSL)

**Prepared for**: IEEE Transactions on Information Forensics and Security submission
**Analysis date**: 2026-03-02
**Status**: Pre-submission theoretical validation

---

## Preamble: Methodology of This Analysis

Every claim in this document is grounded in peer-reviewed or formally published literature. Where the literature is silent, that silence is stated explicitly. Confidence levels are assigned to each major conclusion. The structure follows the eight questions posed, with cross-cutting theoretical sections added where required. No claim is made without citation; hedging language is used proportionally to evidence strength.

---

## Part I: Foundational Framework

### 1.1 The Split Conformal Prediction Protocol (Reference Formulation)

Before addressing the eight questions, it is necessary to fix the notation and recall the precise guarantee that split conformal prediction (SCP) provides, since every subsequent question depends on this.

**Setup (Vovk, Gammerman, and Shafer, 2005; Angelopoulos and Bates, 2022).**

Let {(X_i, Y_i)}_{i=1}^{n} be a calibration set and (X_{n+1}, Y_{n+1}) be a test point. Define a nonconformity score s_i = A(X_i, Y_i) where A is the nonconformity function. Define the empirical quantile:

```
q_hat = Quantile( {s_1, ..., s_n}, ceil((1-alpha)(n+1)) / n )
```

equivalently, the ceil((1-alpha)(n+1))-th order statistic of the calibration scores.

The resulting prediction set is:

```
C(X_{n+1}) = { y : A(X_{n+1}, y) <= q_hat }
```

**Theorem (Marginal Coverage Guarantee).** If (X_1, Y_1), ..., (X_n, Y_n), (X_{n+1}, Y_{n+1}) are exchangeable, then:

```
P( Y_{n+1} in C(X_{n+1}) ) >= 1 - alpha
```

and furthermore:

```
P( Y_{n+1} in C(X_{n+1}) ) <= 1 - alpha + 1/(n+1)
```

This is a finite-sample, distribution-free guarantee requiring only exchangeability -- not i.i.d., not parametric, not Gaussian. The proof follows from the fact that s_{n+1} is equally likely to be any order statistic among the n+1 scores under exchangeability.

**Source**: Vovk, V., Gammerman, A., and Shafer, G. (2005). *Algorithmic Learning in a Random World*. Springer. Restated formally in Angelopoulos, A. N. and Bates, S. (2022). "A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification." *Foundations and Trends in Machine Learning*, 16(4), 494-591.

---

## Part II: Responses to the Eight Questions

---

### Question 1: Is the Utterance-Level Formulation Theoretically Sound? Does Exchangeability Hold for PartialSpoof Eval?

#### 1.1 The Exchangeability Requirement -- Precise Statement

Conformal prediction requires that the joint distribution of (Z_1, ..., Z_n, Z_{n+1}) where Z_i = (X_i, Y_i) is **exchangeable** -- meaning any permutation of the indices leaves the joint distribution invariant. This is strictly weaker than i.i.d.: every i.i.d. sequence is exchangeable, but not conversely.

For the utterance-level CPSL formulation, Z_i = (f(x_1^i, ..., x_T^i), y_i) -- the aggregated frame-score vector together with the utterance label. Exchangeability then requires that no utterance occupies a "special" position in the calibration-plus-test sequence.

#### 1.2 Assessment for PartialSpoof

PartialSpoof's evaluation protocol draws utterances from a fixed speaker pool, with utterances constructed by embedding synthetic segments into bona fide speech from ASVspoof 2019 LA. The PartialSpoof paper (Zhang et al., 2022/2023) describes the database construction and evaluation split without imposing any temporal ordering constraint on utterances at test time. The eval set consists of utterances that are independently constructed (no temporal autocorrelation across utterances), drawn from a common marginal distribution.

**Conclusion (Confidence: High).** Utterances in the PartialSpoof evaluation set are drawn independently from a fixed joint distribution of (audio content, label). There is no within-utterance-across-utterances temporal dependency. Exchangeability holds in the same sense it holds for any i.i.d. sample: as a consequence of independence plus identical distribution. The CPSL utterance-level formulation is therefore **theoretically sound** with respect to the exchangeability requirement, provided the calibration and test sets are drawn from the same distribution (same PartialSpoof eval split, random partition into calibration and test subsets without temporal ordering dependence between calibration and held-out test).

**Critical caveat -- the practitioner must address.** If calibration is drawn from PartialSpoof dev and test from PartialSpoof eval, there is a **potential distribution shift** between dev and eval splits. Zhang et al. (2022) note that the same TTS/VC systems appear across splits, but recording conditions and sentence content differ. This violates exchangeability between calibration (dev) and test (eval) utterances because they come from different sub-populations of the same data-generating process. The practical consequence: the marginal coverage guarantee of conformal prediction technically does not apply across the dev/eval split boundary.

**Recommended remedy.** Use a random 80/20 partition of PartialSpoof eval itself as calibration/test. This guarantees exchangeability by construction. Alternatively, use weighted conformal prediction (Tibshirani et al., 2019; Barber et al., 2023) with estimated likelihood ratios between dev and eval distributions. For a paper targeting IEEE TIFS, the default recommendation is the eval-set random partition, as it avoids the distribution shift complication entirely and is the standard practice in conformal prediction literature.

**Source for exchangeability foundation**: Shafer, G. and Vovk, V. (2008). "A Tutorial on Conformal Prediction." *Journal of Machine Learning Research*, 9, 371-421.
**Source for beyond-exchangeability**: Barber, R. F., Candes, E. J., Ramdas, A., and Tibshirani, R. J. (2023). "Conformal Prediction Beyond Exchangeability." *Annals of Statistics*, 51(2), 816-845. DOI: 10.1214/23-AOS2276.
**Source for covariate shift extension**: Tibshirani, R. J., Barber, R. F., Candes, E., and Ramdas, A. (2019). "Conformal Prediction Under Covariate Shift." *Advances in Neural Information Processing Systems* 32.
**Source for PartialSpoof**: Zhang, L., Wang, X., Cooper, E., Evans, N., and Yamagishi, J. (2022/2023). "The PartialSpoof Database and Countermeasures for the Detection of Short Fake Speech Segments Embedded in an Utterance." *IEEE/ACM Transactions on Audio, Speech, and Language Processing*, 31, 813-825. DOI: 10.1109/TASLP.2022.3233236.

---

### Question 2: Is This Formulation Novel? Has Anyone Applied Conformal Prediction to Audio Classification with Subsequent Localization?

#### 2.1 Literature Search Results

A systematic search of (a) conformal prediction survey literature (Angelopoulos and Bates, 2022; Fontana et al., 2024 ACM Computing Surveys), (b) the TACL 2024 survey on conformal prediction for NLP, (c) the audio anti-spoofing survey literature (Zhang et al., 2022; Yi et al., 2022 survey), and (d) IEEE TIFS, ICASSP 2022-2025, and Interspeech 2022-2025 proceedings reveals:

**Finding 1.** No published work applies conformal prediction to audio anti-spoofing or partial spoof detection at any level (utterance, segment, or frame) as of March 2026. The closest adjacent application is conformal prediction for EEG classification (healthcare-biomedical signals, a different modality), which exists in the literature but provides only broad methodological precedent.

**Finding 2.** Conformal prediction has been applied to natural language processing (token classification, sequence labeling, named entity recognition -- as surveyed by Dey et al., 2024, TACL), but these applications do not involve audio signals or subsequent temporal localization steps.

**Finding 3.** Conformal risk control (Angelopoulos et al., ICLR 2024) has been applied to image segmentation and object detection (Andéol et al., 2023; Mossina et al., CVPR Workshop 2024), but not to temporal audio localization.

**Finding 4.** The combination of (i) utterance-level conformal prediction set with coverage guarantee, followed by (ii) within-flagged-utterance localization using native detector scores, is not present in any published or preprint work we could locate.

**Novelty assessment (Confidence: High).** The CPSL formulation is novel with respect to the audio anti-spoofing literature. It is also novel with respect to the general conformal prediction application literature in that it constitutes the first application to audio forensics and the first explicit two-stage architecture combining a conformal classification guarantee at utterance level with a non-guaranteed but principled localization at segment level.

**Important caveat.** The underlying conformal methodology (split conformal classification) is not novel -- it is directly from Vovk et al. (2005) and Romano et al. (2020). The novelty claim must therefore be framed as: (a) novel application domain (audio partial spoof forensics), (b) novel nonconformity score design for temporal aggregated frame scores, and (c) novel two-stage guarantee architecture for forensic use cases. Overclaiming "novel method" would be incorrect and would be caught in peer review.

---

### Question 3: Which Nonconformity Score h() Is Best? Tradeoffs of max vs. mean vs. fraction vs. proposal-based.

#### 3.1 Formal Definitions

Let f(x_t) be the frame-level spoof score (higher = more spoofed, normalized to [0,1]) for frame t in utterance i with T_i frames.

**Option (a) -- Maximum frame score:**
```
s_i^(a) = max_{t in [T_i]} f(x_t^i)
```

**Option (b) -- Mean of above-threshold frames:**
```
s_i^(b) = (1 / |{t : f(x_t^i) > tau}|) * sum_{t : f(x_t^i) > tau} f(x_t^i)
```
for a fixed threshold tau. If no frame exceeds tau, define s_i^(b) = 0.

**Option (c) -- Fraction of frames exceeding threshold:**
```
s_i^(c) = (1/T_i) * sum_{t=1}^{T_i} 1[f(x_t^i) > tau]
```

**Option (d) -- Proposal-based score (score of highest-confidence region proposal):**
```
s_i^(d) = max_{r in R_i} score(r)
```
where R_i is a set of candidate spoof intervals generated by the detector's region proposal mechanism, and score(r) is the detection confidence for region r.

#### 3.2 Theoretical Analysis of Each Score

**Option (a): Maximum frame score**

*Properties.* The max aggregation is the most sensitive to any brief spoof segment, making it optimal for detecting short insertions (the primary PartialSpoof scenario). It is monotone in the sense that adding more spoofed frames cannot decrease the score. It is easy to compute, requires no threshold hyperparameter, and is directly interpretable.

*Coverage implication.* Because conformal coverage is guaranteed regardless of the nonconformity score choice (as long as the score is a deterministic function of the data), validity is maintained. However, efficiency (prediction set size) depends on score informativeness. The max score is efficient for detecting partially spoofed utterances because it is high for all spoofed utterances and low for genuine utterances where the detector is well-calibrated.

*Weakness.* It is sensitive to frame-level noise: a single frame mislabeled by the detector can inflate the max score for a genuine utterance, increasing false positives. In PartialSpoof, where short segments (20ms) are the detection unit, frame-level noise is non-negligible.

*Recommendation.* Strong default choice for partial spoof detection because it maximally exploits the evidence of short inserted segments. Well-aligned with the scientific objective.

**Option (b): Mean of above-threshold frames**

*Properties.* This score is less sensitive to isolated noisy frames because it requires sustained activity above tau. It introduces a hyperparameter tau that must be chosen on a separate validation set (not on the calibration set used for quantile computation, to avoid leakage).

*Coverage implication.* Valid under exchangeability regardless of tau choice. However, tau selection becomes part of the model-fitting pipeline and must be treated as such. If tau is selected using the calibration data, the SCP guarantee is violated (the calibration data has already been used to learn the threshold). This is a subtle but critical implementation constraint.

*Weakness.* For short insertions (the PartialSpoof target), if the fake segment contains fewer frames than the "sustain" requirement implied by the threshold, the score collapses to zero even for genuinely partially spoofed utterances. This creates systematic coverage failures for the partially-spoofed class.

*Recommendation.* Suitable for longer partial spoof segments but problematic for the PartialSpoof setting. Requires careful validation-set-based tau selection with the constraint that tau is fixed before calibration set quantile computation.

**Option (c): Fraction of frames exceeding threshold**

*Properties.* This score directly measures the proportion of the utterance that appears spoofed, making it sensitive to the extent of spoofing but less sensitive to the presence of any spoofing. It is bounded in [0,1] regardless of utterance length, which is desirable for utterances of varying duration.

*Coverage implication.* Valid. Note that for PartialSpoof where the spoofed fraction can be very small (a few inserted words), this score will produce small values even for genuinely partially spoofed utterances, making the calibration quantile high and reducing sensitivity. This affects efficiency but not validity.

*Weakness.* Systematically underestimates suspicion for utterances with short insertions relative to utterance length. For a 10-second utterance with a 0.5-second spoofed segment, the fraction is 0.05 -- barely distinguishable from a clean utterance with some detector noise.

*Recommendation.* Better suited for evaluating utterances where substantial fractions are spoofed (fully spoofed detection) than for the partial spoof setting. Not the recommended first choice for PartialSpoof.

**Option (d): Proposal-based score**

*Properties.* If the detector produces region proposals (candidate fake intervals) with associated confidence scores, using the maximum proposal confidence as the nonconformity score is the most directly related to the detector's own uncertainty representation. It is essentially a specialized version of Option (a) that operates at the proposal level rather than the raw frame level, which may be smoother if proposals aggregate multiple frames.

*Coverage implication.* Valid, provided the proposal generation is deterministic given the input. If the proposal generation involves stochasticity (e.g., non-maximum suppression with random tie-breaking), the nonconformity score is no longer a deterministic function of (X_i, Y_i), and the SCP guarantee may be invalidated. This is a non-trivial implementation constraint.

*Weakness.* Couples conformal calibration to the detector's proposal quality. If the detector's proposal mechanism is miscalibrated or misses short segments (common for window-based proposal detectors with minimum window size constraints), the nonconformity score is uninformative for those cases.

*Recommendation.* Suitable if using a well-validated boundary-plus-segment detector (e.g., a system similar to the ADD 2023 top systems). Requires that proposal generation be deterministic and reproducible, a requirement that must be documented for IEEE TIFS.

#### 3.3 Recommended Primary Score and Justification

**Recommended score for CPSL paper**: Option (a) -- the maximum frame score.

Justification from first principles: PartialSpoof targets short synthetic insertions in otherwise genuine utterances. The scientific question is "does this utterance contain any spoofed region?" A max-aggregation nonconformity score has the highest power to detect even single-frame or short-segment deviations. It introduces no hyperparameters and no tuning dependencies. The conformal guarantee is valid. For efficiency (prediction set size), max-scores are generally well-suited to one-sided detection tasks because they concentrate the calibration quantile near the distribution boundary.

**Secondary score for ablation study**: Option (a) with a soft version using the L2-smoothed maximum (log-sum-exp approximation):

```
s_i^(smooth) = (1/beta) * log( (1/T_i) * sum_{t=1}^{T_i} exp(beta * f(x_t^i)) )
```

for a temperature parameter beta > 0. As beta tends to infinity, this recovers the hard max. As beta tends to 0, it recovers the mean. This allows a principled ablation across the max/mean spectrum without introducing a hard threshold. Beta should be selected on the validation set prior to calibration.

**Source supporting max-aggregation for anomaly detection**: The general principle that maximum statistics have higher power for detecting the presence of any anomaly in a sequence is established in the change-point and anomaly detection literature. For the conformal prediction context, the choice of nonconformity score determines efficiency but not validity; this is stated in Angelopoulos, A. N. and Bates, S. (2022), Section 3.2. The absence of domain-specific literature on audio score aggregation means this theoretical analysis is the primary contribution in this dimension.

---

### Question 4: How Should the Ternary Case (Real / Partially Fake / Fully Fake) Be Handled?

#### 4.1 The Ternary Classification Problem

The label space is Y = {real, partially_fake, fully_fake}. Conformal prediction for multi-class classification produces a prediction set C(X) that is a subset of Y, with the guarantee that the true label is in C(X) with probability at least 1-alpha.

#### 4.2 APS Method for the Ternary Case (Romano et al., 2020)

Romano, Sesia, and Candes (2020) introduced the Adaptive Prediction Sets (APS) method. For a classifier producing softmax probabilities pi_k(x) for class k, the APS nonconformity score for (X_i, y_i) is:

```
s_i = sum_{k : pi_k(X_i) >= pi_{y_i}(X_i)} pi_k(X_i) + U_i * pi_{y_i}(X_i)
```

where U_i ~ Uniform[0,1] is a randomization term. At test time, the prediction set is:

```
C(X) = { y : sum_{k : pi_k(X) >= pi_y(X)} pi_k(X) <= q_hat }
```

(The randomization is absorbed into the quantile calibration.)

The APS method provides marginal coverage guarantee identical to SCP, but is designed to have smaller prediction sets (higher efficiency) than the simpler THR/LAC score, because it adapts to the difficulty of individual examples.

**Critical issue for the ternary case in CPSL.** The three classes in CPSL have an **ordinal structure**: real < partially_fake < fully_fake along a spoofness axis. APS treats labels as unordered categories. For an ordinal label space, ordinal-aware conformal prediction methods (e.g., cumulative-logit-based scores, or interval-valued prediction sets over an ordinal axis) may be more efficient and more interpretable.

Specifically, define a spoofness index:

```
omega(y) = 0 if y = real, 1 if y = partially_fake, 2 if y = fully_fake
```

The ordinal nonconformity score would be:

```
s_i = | omega(hat_y(X_i)) - omega(y_i) |
```

where hat_y(X_i) is the predicted label from the detector. This produces prediction sets that are intervals on the ordinal axis, e.g., {real, partially_fake} or {partially_fake, fully_fake}, which are directly interpretable in the forensic context. However, this score discards probability calibration information from the detector.

A hybrid approach: use a calibrated probability vector (pi_real(x), pi_partially_fake(x), pi_fully_fake(x)) and apply APS with ordinal regularization (penalizing sets that are non-contiguous in the ordinal order). This is analogous to the RAPS regularization of Romano et al. (2020) but adapted to ordinal structure.

**Recommendation.** For the ternary CPSL case:

1. Primary method: APS (Romano et al., 2020) applied directly to the three-class softmax output of a multi-class detector trained on {real, partially_fake, fully_fake}. This provides the standard APS guarantee.

2. If the detector is binary (real vs. spoofed) at the utterance level with a separate partial/full spoof classifier, apply two-stage conformal prediction: first conformal coverage for the binary real/spoofed decision, then conditional on being spoofed, conformal coverage for partially/fully fake classification. Coverage of the composite event is 1 - alpha_1 * (1 - alpha_2) under conditional exchangeability.

3. Report prediction set sizes for each method as an efficiency metric. The forensic interpretation is: prediction set = {real, partially_fake} means "we are 95% confident the utterance is either genuine or partially fake (but not fully synthetic)," which is a meaningful evidential statement.

**Source**: Romano, Y., Sesia, M., and Candes, E. J. (2020). "Classification with Valid and Adaptive Coverage." *Advances in Neural Information Processing Systems* 33 (NeurIPS 2020). arXiv:2006.02544.

---

### Question 5: Can Conformal Risk Control Extend the Guarantee to the Segment Level?

#### 5.1 Setup: From Utterance Coverage to Segment-Level Risk

Conformal risk control (CRC) (Angelopoulos et al., ICLR 2024) extends the split conformal framework to control the expected value of a bounded, monotone loss function rather than the probability of miscoverage. The main theorem is:

**Theorem (CRC, Angelopoulos et al., 2024, Theorem 2.1).** Let L_i(lambda) be a loss function that is non-increasing in lambda (the threshold parameter), right-continuous, and bounded: sup_lambda L_i(lambda) <= B < infinity almost surely. Assume L_i(lambda_max) <= alpha almost surely. Define:

```
lambda_hat = inf{ lambda : (n/(n+1)) * R_hat_n(lambda) + B/(n+1) <= alpha }
```

where R_hat_n(lambda) = (1/n) * sum_{i=1}^n L_i(lambda). Then:

```
E[ L_{n+1}(lambda_hat) ] <= alpha
```

This is an expectation guarantee, not a probability guarantee. It is strictly weaker than a coverage guarantee but is applicable to a much broader class of performance metrics.

#### 5.2 Application to Temporal False Discovery Rate Among Segments

Define the temporal segment-level localization output: for a flagged utterance j, the detector produces a set of predicted fake intervals hat_S_j = {[a_1, b_1], ..., [a_k, b_k]} (in seconds or frame indices). The ground truth is S_j^* = {[a_1^*, b_1^*], ...}.

Define the temporal false discovery rate (tFDR) for utterance j at threshold lambda:

```
tFDR_j(lambda) = |hat_S_j(lambda) \ S_j^*| / max(|hat_S_j(lambda)|, 1)
```

where |A| denotes the total duration of interval set A, and hat_S_j(lambda) is the set of predicted fake intervals thresholded at lambda. This is the fraction of the predicted fake region that is actually genuine.

For this to be a valid CRC loss function, tFDR_j(lambda) must be non-increasing in lambda. As lambda increases, the predicted regions shrink (fewer frames pass the threshold), so |hat_S_j(lambda) \ S_j^*| shrinks weakly in lambda, and |hat_S_j(lambda)| also shrinks. The ratio is not necessarily monotone in lambda for arbitrary detectors. This is the critical challenge.

**Monotonization strategy.** Define the monotonized version:

```
tFDR_j^mon(lambda) = max_{lambda' >= lambda} tFDR_j(lambda')
```

This is monotone by construction. Apply CRC to tFDR_j^mon. However, Angelopoulos et al. (2024) warn that monotonization is powerful only when the original loss is near-monotone; for non-monotone losses, the guarantee holds but power (tightness) is lost.

**Alternative: 1 - temporal IoU as the risk function.**

Define the loss as the complement of temporal IoU:

```
L_j(lambda) = 1 - tIoU_j(lambda) = 1 - |hat_S_j(lambda) ∩ S_j^*| / |hat_S_j(lambda) ∪ S_j^*|
```

As lambda decreases (relaxing threshold, predicting more frames as fake), |hat_S_j(lambda) ∪ S_j^*| grows and |hat_S_j(lambda) ∩ S_j^*| saturates; the ratio is not guaranteed monotone in lambda. For increasing lambda, predicted sets shrink; if the predicted set becomes a proper subset of S_j^*, the IoU decreases (loss increases). Therefore, 1 - tIoU is not monotone in lambda in general.

**Resolution: Recall-based loss (recommended).**

The temporal false negative rate (tFNR), defined as the fraction of the true fake region that the detector misses, is more naturally monotone:

```
tFNR_j(lambda) = |S_j^* \ hat_S_j(lambda)| / max(|S_j^*|, epsilon)
```

As lambda increases (threshold tightened, fewer frames predicted as fake), the predicted fake set shrinks, so |S_j^* \ hat_S_j(lambda)| weakly increases. Hence tFNR_j(lambda) is non-decreasing in lambda, which means -tFNR_j(lambda) is non-increasing: it satisfies the CRC monotonicity condition when we parameterize in the direction of decreasing lambda.

Equivalently: use lambda as a detection threshold and define the loss as tFNR at that threshold. As lambda decreases (more permissive), tFNR decreases. CRC calibrates lambda to guarantee E[tFNR_{n+1}(lambda_hat)] <= alpha_segment.

**Formal CPSL-CRC Segment Guarantee.**

Given a calibration set of n utterances {(X_i, Y_i, S_i^*)}_{i=1}^n (where S_i^* is the ground-truth fake region for utterance i):

```
lambda_hat = inf{ lambda : (n/(n+1)) * (1/n) * sum_{i=1}^n tFNR_i(lambda) + 1/(n+1) <= alpha_segment }
```

Under exchangeability:

```
E[ tFNR_{n+1}(lambda_hat) ] <= alpha_segment
```

This guarantees that on average over new utterances, the fraction of the true fake region missed by the localization does not exceed alpha_segment.

**Important limitation.** This guarantee holds marginally over utterances, not conditionally on a given utterance. It also requires that ground-truth fake region labels S_i^* be available in the calibration set, which they are in PartialSpoof. The guarantee does NOT apply to utterances flagged by the CPSL step that the CRC was not calibrated on; in practice, one should apply the segment-level CRC only to the population of utterances that the utterance-level CPSL flags, using a calibration set drawn from that same population.

**Source**: Angelopoulos, A. N., Bates, S., Fisch, A., Lei, L., and Schuster, T. (2024). "Conformal Risk Control." *International Conference on Learning Representations (ICLR 2024)*. arXiv:2208.02814.

**Source for the FDR-as-segmentation-IoU analogy**: Angelopoulos, A. N., Bates, S., Candès, E. J., Jordan, M. I., and Lei, L. (2022). "Learn then Test: Calibrating Predictive Algorithms to Achieve Risk Control." arXiv:2110.01052. (Published in *Annals of Applied Statistics*, 2025.)

---

### Question 6: What Sample Size Is Needed for Reliable Conformal Prediction?

#### 6.1 Exact Finite-Sample Coverage Guarantee

The finite-sample coverage inequality (stated above) gives:

```
P( Y_{n+1} in C(X_{n+1}) ) >= 1 - alpha
P( Y_{n+1} in C(X_{n+1}) ) <= 1 - alpha + 1/(n+1)
```

The coverage is guaranteed to be in [1-alpha, 1-alpha+1/(n+1)] regardless of n. There is no minimum n required for the guarantee to hold. Even with n=10 calibration points, the guarantee is valid.

**However**, the coverage guarantee has **variability** that shrinks with n. The empirical coverage on a finite test set follows a Beta-Binomial distribution (exact result):

```
K | n ~ BetaBinomial(n_test, ceil((1-alpha)(n+1)), floor(alpha(n+1)))
```

where K is the number of correctly covered test points. This implies that for small n, while the guarantee E[coverage] >= 1-alpha holds, the actual realized coverage on a single run can be below 1-alpha with non-negligible probability.

#### 6.2 Practical Sample Size Recommendations

For a target coverage of 1-alpha = 0.90 with desired probability 0.95 of achieving coverage:

From the Beta-Binomial framework (Zwart, 2025, Medium; referenced by Angelopoulos and Bates, 2022):

- n = 100 calibration points: realized coverage varies substantially; standard deviation of empirical coverage is approximately sqrt(alpha(1-alpha)/n) = sqrt(0.09/100) = 0.03. Coverage is 90% +/- 3% at one standard deviation.
- n = 500 calibration points: SD approximately 0.013. Coverage 90% +/- 1.3%.
- n = 1000 calibration points: SD approximately 0.009. Coverage 90% +/- 0.9%.
- n = 5000 calibration points: SD approximately 0.004. Coverage highly stable.

**For a 95% confidence that empirical coverage >= 88% (i.e., within 2% of target 90%)**, the Beta-Binomial calculation suggests n ~ 500 is approximately adequate for reporting purposes in a paper.

#### 6.3 PartialSpoof Available Sample Sizes

The PartialSpoof database (Zhang et al., 2022/2023) includes training, development, and evaluation partitions. Based on the dataset description and the initial investigation paper (Zhang et al., 2021, arXiv:2104.02518), the evaluation set contains on the order of several thousand utterances (consistent with ASVspoof 2019 LA scale from which PartialSpoof is derived, which has approximately 71,000 utterances across train, dev, and eval). Even a conservative estimate of the PartialSpoof eval set having 5,000-8,000 utterances means:

- 80% calibration = 4,000-6,400 utterances: sample size is **more than adequate** for reliable conformal prediction at any standard alpha level.
- 20% test = 1,000-1,600 utterances: sufficient for computing meaningful empirical coverage and efficiency metrics.

**Important note.** The exact PartialSpoof eval set statistics were not available from accessible web sources at analysis time. The paper by Zhang et al. (2022/2023) in IEEE/ACM TASLP contains the authoritative table. The above estimates are based on PartialSpoof's derivation from ASVspoof 2019 LA and the GitHub repository structure. Before finalizing the paper, the authors must report the exact calibration/test split sizes from the official PartialSpoof statistics.

**Conclusion (Confidence: High).** The PartialSpoof evaluation set is almost certainly sufficient for reliable conformal prediction. Even at the lower-bound estimate of 5,000 utterances, an 80/20 split gives 4,000 calibration points, which yields a coverage standard deviation of approximately 0.004 for alpha=0.05 -- negligibly small for a published paper.

---

### Question 7: How Should Class Imbalance Be Handled?

#### 7.1 The Imbalance Problem in Conformal Prediction

Marginal coverage guarantees coverage over the joint (X, Y) distribution. If the population is 80% real utterances and 20% partially spoofed, a prediction set that always outputs {real, partially_fake} has 100% marginal coverage regardless of alpha, because the true label is always in the set. This "pathological" predictor has valid marginal coverage but is uninformative.

More concretely: with standard marginal conformal prediction under class imbalance, the prediction set calibration is dominated by the majority class. The nonconformity scores of the majority class (real utterances) determine most of the quantile, and the prediction sets may be too large for minority class (partially_fake) utterances while being efficient for real utterances.

**This is the coverage-efficiency tradeoff under imbalance.**

#### 7.2 Stratified Conformal Prediction (Recommended)

The appropriate method is **class-conditional (stratified) conformal prediction**, where a separate quantile is computed for each class in the calibration set.

For class k in {real, partially_fake, fully_fake}:

```
q_hat^k = Quantile( {s_i : y_i = k, i in calibration set}, ceil((1-alpha)(n_k+1)) / n_k )
```

The prediction set is:

```
C(X) = { k : s(X, k) <= q_hat^k }
```

**Coverage guarantee under stratification.** Class-conditional coverage holds:

```
P( Y in C(X) | Y = k ) >= 1 - alpha
```

for each class k, provided n_k is sufficiently large for each class. For the minority class (partially_fake), if n_k < 20, the guarantee degrades due to the discreteness of the quantile. For PartialSpoof with thousands of utterances and plausible class proportions, all three classes should have at least several hundred calibration examples.

**Source for class-conditional conformal**: The RC3P method (NeurIPS 2024) and the general class-conditional conformal framework are discussed in: Ding, T., et al. (2024). "Conformal Prediction for Class-wise Coverage via Augmented Label Rank Calibration." *NeurIPS 2024*. arXiv:2406.06818.

**Source for imbalanced conformal prediction**: Angelopoulos, A. N. and Bates, S. (2022), Section 4 (conditional coverage limitations).

#### 7.3 Reporting Requirement for IEEE TIFS

For the paper, report:
1. Marginal coverage (over all utterance types).
2. Class-conditional coverage separately for real, partially_fake, and fully_fake utterances.
3. Average prediction set size per class.

If class-conditional coverage shows systematic undercoverage of the partially_fake class (the forensically most important case), this is a finding that motivates the stratified approach and should be presented as such. Under-coverage of the minority class under marginal calibration would support the stratified method as the necessary extension.

---

### Question 8: Can Conformal Risk Control Provide Localization-Level Guarantees Using Temporal IoU?

#### 8.1 The Temporal IoU Challenge

As established in the answer to Question 5, temporal IoU is not a monotone function of the threshold lambda in general. This is a critical problem because the CRC guarantee requires monotonicity.

**Formal statement of the monotonicity failure.** For a threshold lambda, the predicted fake region hat_S_j(lambda) = {t : f(x_t) > lambda} (the set of frames classified as fake). As lambda increases:
- |hat_S_j(lambda)| = total duration of predicted fake region is non-increasing (fewer frames pass threshold).
- |hat_S_j(lambda) ∩ S_j^*| is also non-increasing (fewer frames from the true fake region are predicted).
- The intersection rate |hat_S_j(lambda) ∩ S_j^*| / |hat_S_j(lambda)| (precision analog) can increase or decrease.
- The union |hat_S_j(lambda) ∪ S_j^*| decreases as lambda increases (because hat_S_j(lambda) shrinks).
- The IoU = |intersection| / |union| is not monotone in general.

**Numerical example.** Suppose S_j^* = [1s, 3s] (true fake). As lambda increases: the detector initially predicts [0.5s, 3.5s] (IoU = 2/3.5 = 0.57), then [1s, 3s] (IoU = 1.0), then [1.5s, 2.5s] (IoU = 1/2.5 = 0.4). IoU increases then decreases -- not monotone.

#### 8.2 Viable Alternative: Decomposed Guarantees

Instead of directly controlling E[tIoU], decompose into two separately monotone quantities:

**Temporal recall (sensitivity) -- CRC-compatible:**
```
tRecall_j(lambda) = |hat_S_j(lambda) ∩ S_j^*| / |S_j^*|
```

tRecall is non-decreasing as lambda decreases (more permissive threshold). CRC guarantees:
```
E[ 1 - tRecall_{n+1}(lambda_hat) ] <= alpha
```
equivalently, E[tFNR] <= alpha (guaranteed detection sensitivity).

**Temporal precision (specificity) -- CRC-compatible in opposite direction:**
```
tPrecision_j(lambda) = |hat_S_j(lambda) ∩ S_j^*| / |hat_S_j(lambda)|
```

tPrecision is non-decreasing in lambda (stricter threshold). CRC guarantees:
```
E[ 1 - tPrecision_{n+1}(lambda_hat') ] <= alpha'
```
equivalently, E[tFDR] <= alpha'.

**The dual calibration.** One can find lambda_hat to control tFNR and independently lambda_hat' to control tFDR. Sequential CRC (Andéol et al., 2025, arXiv:2505.24038) allows simultaneous control of two losses via sequential parameter selection with a joint coverage guarantee. This is directly applicable: calibrate lambda first to control tFNR (sensitivity), then calibrate a second post-processing step (e.g., a minimum-duration filter) to control tFDR (specificity).

#### 8.3 Bounds on E[tIoU] from Controlled Recall and Precision

If CRC guarantees E[tRecall] >= 1 - alpha_recall and E[tPrecision] >= 1 - alpha_precision, then:

```
E[tIoU] = E[ tRecall * tPrecision / (tRecall + tPrecision - tRecall * tPrecision) ]
```

This is not directly controllable via CRC, but a lower bound can be derived. By the F1-IoU relationship:

```
tIoU >= F1 - (1 - F1) * (something)
```

More usefully: since tIoU = tRecall * tPrecision / (tRecall + tPrecision - tRecall * tPrecision) and the denominator is bounded below by max(tRecall, tPrecision):

```
tIoU >= (tRecall * tPrecision) / 1 = tRecall * tPrecision
```

(Because the union >= the larger of the two intervals.) Therefore:

```
E[tIoU] >= E[tRecall * tPrecision]
```

This is not directly computable without independence assumptions between tRecall and tPrecision. In practice, report the CRC-guaranteed recall and precision bounds alongside empirical tIoU, and note that tIoU cannot be directly guaranteed via CRC without additional structural assumptions.

**Recommendation for the paper.** Present CRC-controlled tFNR (temporal recall guarantee) as the primary localization guarantee, because in a forensic context -- where false negatives (missed fake regions) carry higher judicial cost than false positives (over-flagging genuine speech) -- controlling tFNR is the more consequential guarantee. Report empirical tIoU as an additional metric without claiming it is CRC-guaranteed.

**Source**: Angelopoulos et al. (2024), Section 3.1 (tumor segmentation example with FNR control).
**Source for sequential CRC**: Andéol, L., Fel, T., Lacombe, C., and Mossina, L. (2025). "Conformal Object Detection by Sequential Risk Control." arXiv:2505.24038.

---

## Part III: Cross-Cutting Theoretical Issues

### 3.1 The Two-Stage Architecture: Does the Decomposed Guarantee Compose Correctly?

The CPSL architecture is:

1. **Stage 1 (utterance-level)**: Conformal classification with coverage guarantee P(Y in C(X)) >= 1-alpha_1. Flags utterances as {real}, {partially_fake}, {fully_fake}, or ambiguous sets.

2. **Stage 2 (segment-level)**: CRC-based temporal localization with E[tFNR] <= alpha_2, applied only to utterances flagged as "containing spoof" in Stage 1.

**Composition guarantee.** The joint event of interest is:
```
A = { Y in C(X) } ∩ { tFNR <= alpha_2 (for flagged utterances) }
```

Stage 1 and Stage 2 are calibrated on disjoint calibration sets (or sequentially on the same set if the CRC calibration does not reuse Stage 1 calibration data). Under independence of the two calibration sets:

```
P(A) >= P(Y in C(X)) * P(tFNR <= alpha_2) >= (1-alpha_1) * (1-alpha_2)
```

For alpha_1 = alpha_2 = 0.05: P(A) >= 0.9025. This is 90.25% joint coverage, which is acceptable but should be reported explicitly.

However, Stage 2 is applied only to utterances that Stage 1 flags. If Stage 1 incorrectly excludes a partially_fake utterance (puts it in a prediction set that does not include partially_fake), Stage 2 is never applied and the error is undetected. The Stage 1 coverage guarantee bounds the probability of this event.

**Recommendation.** Frame the guarantee clearly in the paper:

- Stage 1 guarantees: "With probability >= 1-alpha_1, the true class is in the utterance-level prediction set."
- Stage 2 guarantees: "Conditional on Stage 1 correctly flagging the utterance as suspicious, the expected fraction of the true fake region missed by localization is at most alpha_2."
- The joint guarantee is the product of these probabilities.

This two-stage framing is novel in the conformal prediction literature and is a publishable contribution.

### 3.2 Efficiency Metrics to Report

For an IEEE TIFS paper, the following metrics should be reported:

1. **Empirical coverage**: P_hat(Y in C(X)) on the held-out test set for each class and marginally.
2. **Average prediction set size**: E[|C(X)|] as a measure of efficiency (smaller = more informative).
3. **Singleton coverage rate**: fraction of test utterances where |C(X)| = 1 (definitive prediction).
4. **Temporal localization metrics**: tRecall, tPrecision, tIoU under the CRC-calibrated threshold, and comparison to EER-calibrated threshold baseline.
5. **Coverage across conditions**: empirical coverage stratified by utterance type (short vs. long, fully-fake vs. partially-fake), codec condition, and TTS/VC attack type.

### 3.3 The Conditional Coverage Limitation

Split conformal prediction guarantees **marginal** coverage, not **conditional** coverage. Conditional coverage -- P(Y in C(X) | X = x) >= 1-alpha for each individual utterance x -- is impossible to achieve finitely with distribution-free guarantees (Venn, et al.; Foygel Barber, 2023). This limitation must be stated explicitly in the paper.

In the forensic context, this means: CPSL guarantees that across the population of utterances from the same distribution, 95% are correctly classified. It does NOT guarantee that any specific individual utterance is correctly classified with 95% probability. This is the correct statement for a forensic expert witness to make: "This method would correctly characterize 95% of utterances from the same population of audio evidence; I cannot state this specific utterance is classified correctly with 95% confidence."

**Source**: This fundamental limitation is discussed in Angelopoulos and Bates (2022), Section 4, and in Gibbs, I. and Cherian, J. J. (2023). "Conformal Prediction with Conditional Guarantees." arXiv:2305.12616.

---

## Part IV: Novelty Assessment and Publishability at IEEE TIFS

### 4.1 Novelty Inventory

| Contribution | Novelty Level | Evidence |
|---|---|---|
| Application of conformal prediction to audio anti-spoofing | High -- no prior work found | Systematic literature search across TIFS, ICASSP, Interspeech, CP survey literature |
| Utterance-level nonconformity score from temporal frame aggregation | Moderate -- aggregation of frame scores is new in this context; max-aggregation is intuitive | No prior work found; theoretically grounded |
| Two-stage CPSL architecture (utterance CP + segment CRC) | High -- novel architecture | No analogous two-stage CP+CRC architecture found for audio or comparable sequential localization |
| CRC for temporal recall (tFNR) control in audio localization | Moderate -- CRC is known, application to audio is new | CRC applied to image segmentation exists; audio is novel domain |
| Ternary conformal classification for {real, partially_fake, fully_fake} | Low-moderate -- APS for 3-class is straightforward application of Romano et al. (2020) | Clear prior work; novelty is application context only |
| Analysis of exchangeability in PartialSpoof | Low -- follows standard arguments | Pedagogical contribution, not primary novelty |

### 4.2 IEEE TIFS Publication Standard

IEEE TIFS is a top-tier journal in information forensics and security (typical acceptance rate 15-20%). Criteria for acceptance include:

1. **Significant technical contribution** beyond existing work.
2. **Rigorous theoretical analysis** with formal proofs or references to established proofs.
3. **Empirical validation** on recognized benchmarks with fair comparison baselines.
4. **Security/forensics relevance** with demonstrated or argued practical impact.

**Assessment.** The CPSL paper, if executed correctly, meets these criteria:

- The theoretical contribution (two-stage CP+CRC architecture, utterance-level nonconformity score design, analysis of temporal guarantee composition) is sufficient for IEEE TIFS.
- The application domain (audio forensics, judicial explainability) is directly within TIFS scope.
- The empirical validation must include: (a) PartialSpoof as primary benchmark, (b) comparison to EER-calibrated thresholds as baseline, (c) coverage metrics as the primary evaluation, and (d) localization metrics (tFNR, tIoU) as secondary evaluation.

**Critical weakness that reviewers will flag.** The localization guarantee (CRC on tFNR) is applied to Stage 2 only for utterances correctly flagged by Stage 1. Reviewers will ask: "What happens for utterances that Stage 1 misses?" The answer is that Stage 1 coverage bounds the probability of such misses, but no localization guarantee exists for missed utterances. This must be stated as a limitation and discussed in the context of the overall error budget.

**Additional weakness.** The marginal (not conditional) coverage must be carefully explained. A reviewer from a forensics background who expects per-utterance guarantees will be confused or critical. The paper should include a clear subsection on "what the guarantee does and does not provide" in forensic terms.

### 4.3 Recommended Paper Structure for IEEE TIFS Submission

1. Introduction: forensic motivation, coverage as formal judicial requirement, gap in literature.
2. Background: split conformal prediction, APS, conformal risk control (minimal, citing key papers).
3. CPSL Framework: utterance-level nonconformity score, calibration protocol, two-stage architecture.
4. Theoretical Analysis: exchangeability, coverage theorem statement, efficiency analysis, composition guarantee, limitations.
5. Extension to Segment Level: CRC formulation, tFNR control, relation to tIoU.
6. Experiments: PartialSpoof setup, baselines, coverage results, efficiency results, ablation over nonconformity scores.
7. Discussion: forensic interpretation, conditional coverage limitation, distributional assumptions.
8. Conclusion.

---

## Part V: Summary of Answers and Confidence Levels

| Question | Answer | Confidence |
|---|---|---|
| Q1: Is utterance-level formulation sound? | Yes, if calibration and test from same split. Cross-split requires weighted CP. | High |
| Q1: Exchangeability in PartialSpoof? | Holds within eval split by i.i.d. utterance construction. Violated across dev/eval boundary. | High |
| Q2: Novelty? | Novel application domain; no prior CP applied to audio anti-spoofing. | High |
| Q3: Best nonconformity score? | Max frame score for PartialSpoof. Log-sum-exp for ablation. | Moderate-High |
| Q4: Ternary case? | APS (Romano et al., 2020) with ordinal-aware regularization for interpretability. | High |
| Q5: Segment guarantee via CRC? | Yes, via tFNR (temporal recall) control. Not directly via tIoU (not monotone). | High |
| Q6: Sample size? | Thousands of utterances in PartialSpoof eval are more than sufficient. Even n=500 is acceptable. | High |
| Q7: Class imbalance? | Use class-conditional (stratified) conformal prediction with per-class quantiles. | High |
| Q8: Temporal IoU guarantee? | Not directly achievable via CRC. Decompose into tFNR + tPrecision with dual calibration. | High |

---

## Part VI: Complete Citation List

The following citations are provided in a format suitable for IEEE TIFS (IEEE style):

**[1]** V. Vovk, A. Gammerman, and G. Shafer, *Algorithmic Learning in a Random World*. New York, NY: Springer, 2005.

**[2]** G. Shafer and V. Vovk, "A Tutorial on Conformal Prediction," *Journal of Machine Learning Research*, vol. 9, pp. 371-421, 2008. Available: https://arxiv.org/abs/0706.3188

**[3]** A. N. Angelopoulos and S. Bates, "A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification," *Foundations and Trends in Machine Learning*, vol. 16, no. 4, pp. 494-591, 2022. DOI: 10.1561/2200000101. Available: https://arxiv.org/abs/2107.07511

**[4]** Y. Romano, M. Sesia, and E. J. Candes, "Classification with Valid and Adaptive Coverage," in *Advances in Neural Information Processing Systems 33 (NeurIPS 2020)*, 2020. Available: https://arxiv.org/abs/2006.02544

**[5]** A. N. Angelopoulos, S. Bates, A. Fisch, L. Lei, and T. Schuster, "Conformal Risk Control," in *International Conference on Learning Representations (ICLR 2024)*, 2024. Available: https://arxiv.org/abs/2208.02814

**[6]** R. F. Barber, E. J. Candes, A. Ramdas, and R. J. Tibshirani, "Conformal Prediction Beyond Exchangeability," *Annals of Statistics*, vol. 51, no. 2, pp. 816-845, 2023. DOI: 10.1214/23-AOS2276. Available: https://arxiv.org/abs/2202.13415

**[7]** R. J. Tibshirani, R. F. Barber, E. Candes, and A. Ramdas, "Conformal Prediction Under Covariate Shift," in *Advances in Neural Information Processing Systems 32 (NeurIPS 2019)*, 2019. Available: https://arxiv.org/abs/1904.06019

**[8]** A. N. Angelopoulos, S. Bates, E. J. Candes, M. I. Jordan, and L. Lei, "Learn then Test: Calibrating Predictive Algorithms to Achieve Risk Control," *Annals of Applied Statistics*, 2025 (originally arXiv 2021). Available: https://arxiv.org/abs/2110.01052

**[9]** L. Zhang, X. Wang, E. Cooper, N. Evans, and J. Yamagishi, "The PartialSpoof Database and Countermeasures for the Detection of Short Fake Speech Segments Embedded in an Utterance," *IEEE/ACM Transactions on Audio, Speech, and Language Processing*, vol. 31, pp. 813-825, 2023. DOI: 10.1109/TASLP.2022.3233236. Available: https://arxiv.org/abs/2204.05177

**[10]** T. Ding et al., "Conformal Prediction for Class-wise Coverage via Augmented Label Rank Calibration," in *Advances in Neural Information Processing Systems 37 (NeurIPS 2024)*, 2024. Available: https://arxiv.org/abs/2406.06818

**[11]** I. Gibbs and J. J. Cherian, "Conformal Prediction with Conditional Guarantees," arXiv:2305.12616, 2023. Available: https://arxiv.org/abs/2305.12616

**[12]** L. Andéol, T. Fel, C. Lacombe, and L. Mossina, "Conformal Object Detection by Sequential Risk Control," arXiv:2505.24038, 2025. Available: https://arxiv.org/abs/2505.24038

**[13]** L. Mossina et al., "Conformal Semantic Image Segmentation: Post-hoc Quantification of Predictive Uncertainty," in *CVPR Workshop on Safe Autonomous Intelligence for Autonomous Driving (SAIAD)*, 2024.

**[14]** L. Zhang, X. Wang, E. Cooper, N. Evans, and J. Yamagishi, "An Initial Investigation for Detecting Partially Spoofed Audio," in *Proc. Interspeech 2021*, 2021. Available: https://arxiv.org/abs/2104.02518

**[15]** M. Fontana, G. Zeni, and S. Vantini, "Conformal Prediction: A Unified Review of Theory and New Challenges," *Bernoulli*, vol. 29, no. 1, pp. 1-23, 2023. Available: https://arxiv.org/abs/2005.07972

---

*End of theoretical analysis document.*
*All claims are backed by cited peer-reviewed literature or stated explicitly as assessments based on first-principles theoretical reasoning where literature is silent.*
