# XPS-Forensic Bug Tracker

**Last Updated**: 2026-04-02

---

## Open Bugs

### BUG-E1-1: Sub-Native Resolution Temporal Misalignment (CRITICAL)
**Found**: 2026-04-02 by senior-research-scientist agent
**Component**: `experiments/run_e1_baseline.py` lines 329-362
**Description**: When evaluating a detector at a resolution finer than its native frame shift (e.g., BAM at 160ms evaluated at 20ms), `_pool_scores_to_windows` produces score windows that span the detector's native interval but are compared to label windows at the evaluation resolution. Score window `i` covers `[i*160ms, (i+1)*160ms]` but label window `i` covers `[i*20ms, (i+1)*20ms]` — temporal misalignment by a factor of `native/eval`.
**Impact**: All sub-native Seg-EER values for BAM (20/40/80ms) are **invalid** (40.4% is noise, not localization quality). MRM (20ms native) is unaffected.
**Fix**: For detectors with `frame_shift_ms > resolution_ms`, upsample scores to label grid first via `np.repeat`, then pool both to target resolution. Or skip sub-native resolutions with a warning.
**Status**: OPEN

### BUG-E1-2: Label Aggregation Rule Mismatch (HIGH)
**Found**: 2026-04-02 by senior-research-scientist agent
**Component**: `experiments/run_e1_baseline.py` lines 340-343
**Description**: E1 uses `_pool_scores_to_windows` with `agg="mean"` then `>= 0.5` for label pooling (majority vote). The PartialSpoof evaluation protocol (Zhang et al., TASLP 2023, Section IV-B) uses the "any" rule: a segment is spoof if ANY frame within is spoof. This inflates Seg-EER (15.83% vs expected ~8.43% at 160ms for BAM).
**Impact**: All segment-level EER values are computed with wrong label rule. Not comparable to published numbers.
**Fix**: Replace `_pool_scores_to_windows` for labels with `_pool_labels_to_windows(fl, LABEL_FRAME_SHIFT_MS, res, rule="any")`.
**Status**: OPEN

### BUG-E1-3: Frame-Level EER Truncates Continuous Scores (HIGH)
**Found**: 2026-04-02 by senior-research-scientist agent
**Component**: `experiments/run_e1_baseline.py` lines 368-378
**Description**: `upsample_binary_predictions_to_label_grid` calls `.astype(int)` internally, truncating continuous scores in [0,1) to 0 and scores at exactly 1.0 to 1. The frame-level EER is computed on integer-quantized {0,1} values, producing degenerate ROC behavior (EER~0.48, threshold=1.0, F1=0.0%).
**Impact**: Frame-level F1 is 0.0% for all detectors — completely broken.
**Fix**: Use `np.repeat(scores, ratio)` without `.astype(int)` for EER computation. Only apply `.astype(int)` after thresholding for F1.
**Status**: OPEN

### BUG-5: PDSM-PS Captum Integration (HIGH)
**Found**: 2026-03-15 (original)
**Component**: `xps_forensic/pdsm_ps/saliency.py`
**Description**: IG/GradSHAP are hand-rolled, not integrated with detector feature hooks. Perturbation loop for N-AOPC/comprehensiveness/sufficiency was missing.
**Partial Fix**: `perturbation.py` added (2026-04-01) with `compute_saliency_from_detector` (Captum first, manual fallback), `perturb_and_score`, `compute_faithfulness_suite`.
**Remaining**: Wire perturbation.py into E4 experiment runner. Test with real detector.
**Status**: PARTIALLY FIXED

### BUG-6: E6/E7/E8 Experiment Scaffolds (MEDIUM)
**Description**: Codec, alignment, and ablation experiment runners are placeholders without full inference loops.
**Status**: OPEN

### BUG-7: Evidence Schema Coverage Disclosure (MEDIUM)
**Description**: Evidence schema doesn't clearly state marginal (not per-utterance) coverage scope.
**Status**: OPEN

### BUG-CFPRF-2: CFPRF Path Resolution (MEDIUM)
**Found**: 2026-04-02
**Description**: `os.chdir()` during BAM loading changes cwd, causing CFPRF's relative `sys.path.insert` to fail with "No module named 'models.FDN'".
**Fix Applied**: 2026-04-01 — `self.external_dir.resolve()` for absolute path.
**Fix Verification**: NOT verified in E1 — CFPRF still failed in latest run. The `.resolve()` fix may not have been saved before E1 restarted, or there's an additional issue (XLSR pretrained model path).
**Status**: NEEDS RE-VERIFICATION

---

## Fixed Bugs

| ID | Description | Fix | Date |
|---|---|---|---|
| BUG-BAM-1 | BAM score inversion: probs[:,1] = bonafide, not spoof | Changed to probs[:,0] | 2026-04-01 |
| BUG-CFPRF-1 | CFPRF `weights_only=True` rejects checkpoint in PyTorch>=2.6 | Changed to `weights_only=False` | 2026-04-01 |
| BUG-E1-A2 | Segment EER computed per-utterance and averaged (wrong protocol) | Pooled across all utterances | 2026-04-01 |
| BUG-E1-B2 | No NaN guard on detector scores | Added NaN check + skip | 2026-04-01 |
| BUG-PE-1 | PartialEdit loader expects metadata.json; disk has CSV | Added CSV layout support | 2026-04-01 |
| BUG-1 | Calibration ranking orientation | `higher_is_better` flag | 2026-03-20 |
| BUG-2 | Frame resolution mismatch (20ms vs 10ms) | `np.repeat` upsampling | 2026-03-20 |
| BUG-3 | Stage-1 "APS" naming confusion | Docstring updated | 2026-03-20 |
| BUG-4 | Composed guarantee: product bound → Bonferroni | Fixed throughout | 2026-03-20 |
| BUG-FARA-NaN | FARA training NaN at epoch 9 from float16 cdist overflow | float32 guards in cmoe, learnable_mask, GCL | 2026-03-29 |
| BUG-FARA-CIT | Fabricated "Xia et al. 2025 NAACL" citation | Removed, documented as assumption | 2026-03-27 |
