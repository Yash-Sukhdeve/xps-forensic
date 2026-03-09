# XPS-Forensic Bug Tracker

Authoritative list of issues to resolve one-by-one, with fixes validated by tests and scientific citations. We will not proceed to the next item until the current one is fixed and verified.

Status legend: [OPEN], [SOLVED]

1) [SOLVED] E2 calibration ranking orientation (Friedman test)
- Problem: `experiments/run_e2_calibration.py` feeds Expected Calibration Error (ECE) means directly into the Friedman/Nemenyi ranking, which assumes “higher is better”. For ECE (and NLL, Brier), lower values are better, so current rankings are inverted.
- Impact: Misleading claim about the best calibration method.
- Validation basis: ECE is defined as a weighted absolute gap between accuracy and confidence; lower is better (Guo et al., ICML 2017).
  - Source: https://arxiv.org/abs/1706.04599
- Fix applied (robust Option B): Added `higher_is_better` flag to `friedman_nemenyi` (default True for backward compatibility) and passed `higher_is_better=False` from `run_e2_calibration.py` for ECE. This generalizes to other lower-is-better metrics (Brier, NLL) without matrix negation.
- Test evidence: Entire test suite passes (102 tests). Orientation is now explicit and correct for ECE (Guo et al., 2017), avoiding inadvertent ranking inversions.

2) [SOLVED] Frame resolution mismatch (10 ms labels vs 20 ms scores)
- Problem: Dataset loaders (e.g., PartialSpoof, HQ-MPSD, LlamaPartialSpoof) produce 10 ms frame labels, while detector wrappers output 20 ms frame scores. Metrics (E1, E3) and CRC calibration compare arrays of different temporal granularity, biasing results.
- Impact: Segment EER, F1, and Stage-2 CRC guarantees can be systematically wrong.
- Validation basis: Temporal metrics must operate on time-aligned segments; aggregating labels/scores to a common resolution avoids bias.
  - General evaluation practice in speech spoofing (Zhang et al., 2023 PartialSpoof): https://arxiv.org/abs/2204.05177
- Fix applied (Option B — canonical windows + on-the-fly alignment):
  - Added mixed-resolution utilities in `utils/metrics.py`:
    - `compute_segment_eer_mixed(...)` pools 20 ms scores (mean) and 10 ms labels (majority) into canonical window sizes (20–640 ms) before EER computation, matching PartialSpoof protocol (Zhang et al., 2023).
    - `upsample_binary_predictions_to_label_grid(...)` repeats binary predictions (e.g., 20→10 ms) to align with label resolution for temporal metrics (tFNR/tFDR/tIoU).
  - Updated E1 to use `compute_segment_eer_mixed` and to align frame-level predictions to 10 ms for F1.
  - Updated CPSL pipeline/E3 so that Stage-2 CRC calibrates and evaluates temporal metrics on the 10 ms label grid by upsampling predictions, preserving label fidelity while keeping stored labels untouched.
  - This approach is consistent with PartialSpoof’s multi-resolution segment evaluation and preserves comparability (Zhang et al., 2023); BAM and CFPRF report multi-resolution metrics in this paradigm (Zhong et al., 2024: https://arxiv.org/abs/2407.21611; Wu et al., 2024: https://arxiv.org/abs/2407.16554).
- Test evidence: All tests pass (102 tests). Experiments are scaffolds; functional changes are localized and backward-compatible for unit tests.

3) [SOLVED] Stage-1 “APS” naming vs implementation
- Problem: `cpsl/scp_aps.py` implements class-conditional split conformal with a single scalar nonconformity, not Adaptive Prediction Sets (APS) per Romano et al. (NeurIPS 2020).
- Impact: Method mislabeling; potential reviewer confusion.
- Validation basis: APS constructs adaptive sets from per-class scores; the current code thresholds a single score per sample.
  - Source: https://arxiv.org/abs/2009.14193
- Fix applied:
  - Updated module/class docstrings to clearly state this is split conformal prediction with class-conditional (Mondrian) quantiles, not APS; cited the CP tutorial (Angelopoulos & Bates, 2023: https://arxiv.org/abs/2302.08112) and APS paper (Romano et al., 2020).
  - Added an ordinal contiguity constraint in `predict()`: if {real, fully_fake} are both included without {partial}, add {partial}. This only enlarges sets and therefore cannot reduce marginal coverage; it prevents unnatural non-contiguous label sets in an ordered class setting.
  - Kept class name for API stability; documentation now accurately reflects the method.
- Test evidence: Full test suite passes (102 tests). Coverage guarantees are unaffected since sets are only enlarged by the contiguity post-process.

4) [SOLVED] Composed guarantee text in design doc
- Problem: Design doc claims product bound (1−α1)(1−α2), but code correctly uses the Bonferroni bound 1−α1−α2 absent independence.
- Impact: Overstated guarantee in documentation.
- Validation basis: Bonferroni inequality is the safe composition without independence (Angelopoulos & Bates, 2023).
  - Source: https://arxiv.org/abs/2302.08112
- Fix applied: Edited `docs/plans/2026-03-02-xps-forensic-design.md` to state the Bonferroni bound (1 − α₁ − α₂) and explicitly note that the product bound requires independence between stages and is not assumed in this pipeline. Also updated the “Must disclose” and paper outline sections to reflect this.

5) [OPEN] PDSM-PS saliency integration and perturbation protocol
- Problem: IG/GradSHAP are scaffolded but not integrated with detector feature hooks; perturbation schemes for N-AOPC and comp/sufficiency are unspecified.
- Impact: E4 cannot produce scientifically meaningful faithfulness metrics.
- Validation basis: IG (Sundararajan et al., 2017) and SHAP (Lundberg & Lee, 2017) require model gradients/expectations; N-AOPC/comp/sufficiency require consistent perturbations.
  - Sources: https://arxiv.org/abs/1703.01365, https://arxiv.org/abs/1705.07874
- Plan: Integrate Captum IG/GradientShap with detector feature outputs; define frame/phoneme masking protocol to compute N-AOPC and ERASER metrics (DeYoung et al., 2020: https://arxiv.org/abs/1911.03429).

6) [OPEN] E6/E7/E8 are placeholders
- Problem: Scripts output configs/placeholders only.
- Impact: No empirical results for robustness/alignment/ablations.
- Plan: Wire detectors, calibration, CPSL, and saliency to compute real metrics and save results.

7) [OPEN] Evidence schema claims and disclosures
- Problem: Ensure evidence JSON and docs clearly state marginal coverage (not per-utterance) and risk guarantees’ scope.
- Validation basis: Conformal prediction guarantees are marginal without additional assumptions (Angelopoulos & Bates, 2023).
- Plan: Update schema usage docs and examples to reflect limitations and disclosures.
