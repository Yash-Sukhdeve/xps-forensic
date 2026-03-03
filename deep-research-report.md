Do deep research on explainability for audio deepfakes and partial audio spoofs only.

# Explainability for Audio Deepfakes and Partial Audio Spoofs in Judicial Contexts

## Executive summary

Audio deepfakes and **partial audio spoofs** (e.g., voice-cloned word insertions, splicing/partial replacement, short synthetic segments embedded into otherwise bona fide speech) shift the hard problem from “is it fake?” to “**what exactly is fake, where, and why**”—under realistic channel effects (compression/codecs), noise, and imperfect provenance. Benchmarks explicitly reflect this shift: **ASVspoof 2021** adds a deepfake task designed to test robustness under *compression/codecs used in social media* and cross-channel conditions, while **PartialSpoof** and **ADD 2023 Track 2** focus on **segment-level** detection/localization of manipulated intervals. citeturn15view0turn9view0turn5search1turn8view0turn8view3

State-of-the-art “countermeasures” (CMs) increasingly rely on (a) **self-supervised speech encoders** and/or (b) **end-to-end neural CMs** that operate on raw waveforms (e.g., graph-attention architectures), because handcrafted cepstral features (CQCC/LFCC) remain valuable baselines but often struggle with fast-evolving generation methods and domain shift. citeturn15view1turn14search2turn14search13turn10view0turn6search4

For **partial** spoofs, localization methods cluster into three technical patterns: (1) **frame/segment-wise classification** (sliding window or frame-level heads), (2) **boundary/change-point detection** as a first stage, followed by segment scoring, and (3) **multi-task / multi-resolution supervision** (joint utterance-level + segment-level, sometimes multiple temporal resolutions down to tens of milliseconds). These patterns are visible in PartialSpoof’s research line (multi-resolution segment labels) and top ADD 2023 Track 2 systems that fuse boundary detection with frame-level deepfake detection and additional authenticity modeling (e.g., outlier models trained on bona fide). citeturn8view1turn11view1turn11view0turn9view2turn8view0

Explainability that is **court-suitable** is best treated as a **layered evidence stack**: (i) *forensic audio authenticity signals* (metadata/format traces, ENF consistency, discontinuity cues) plus (ii) *ML localization evidence* (time-stamped segments and calibrated scores) plus (iii) *human-understandable mapping* of ML evidence to speech structure (phoneme/word-level saliency or boundary-aligned evidence). Notably, recent work proposes phoneme-discretized saliency maps to make saliency more understandable and (claimed) more faithful for AI-generated voice detection. citeturn11view3turn12view0turn12view3turn10view1

**LLMs can help—but only if demoted to “narrator.”** In judicial contexts, an LLM should *not* be the primary detector. Its appropriate role is to generate a constrained, auditable, and cross-examinable **explanatory narrative** from **structured forensic outputs** (JSON) and immutable evidence references (hashes, timestamps, tool versions). This design directly addresses generative AI risks (e.g., confabulation) emphasized in NIST’s generative AI risk profile and synthetic content risk guidance. citeturn13view0turn13view1turn19search1turn19search0

Admissibility pressure points are predictable: under **FRE 702 / Daubert**, the court cares about testability, peer review, known/potential error rates, standards/controls, and general acceptance; **Frye** jurisdictions emphasize general acceptance in the relevant scientific community. For audio deepfake explainability, this translates into (a) benchmarked error rates under conditions resembling the case (codec/noise/domain), (b) reproducible pipelines, (c) defensible calibration, and (d) disciplined chain-of-custody and authenticity handling consistent with forensic best-practice manuals. citeturn13view3turn4search1turn20search3turn12view0turn11view3turn3search2

## Scope and assumptions

This report is limited to **audio** deepfakes and **partial audio spoofs** (voice cloning, splicing/partial replacement, short inserted or replaced segments), including scenarios where audio segments are generated to align with video (“lip-synced audio segments”), but the technical focus remains on audio-side detection/localization/explainability. citeturn5search1turn21search1turn8view3

Jurisdictional assumption (explicit): examples are framed for **entity["country","United States","federal and state courts"]** federal/state evidentiary practice and **entity["organization","European Union","member state courts"]**-relevant instruments/guidance (noting EU criminal evidence admissibility is not fully harmonized; practices vary by Member State). citeturn13view3turn20search0turn20search5turn20search13

Key operational assumption: “Explainability” here means **human-understandable evidence and reasoning** that supports three forensic questions:

- **Detection**: is there manipulated/synthetic audio?
- **Localization**: where (time ranges) is manipulation likely?
- **Evidentiary explanation**: what observable artifacts or model-attributed cues support that conclusion, and are they reproducible?

This aligns with how partial spoofing research explicitly frames segment-level tasks (“locate short spoofed segments embedded in an utterance”). citeturn9view2turn8view1turn18view0

## Benchmarks and datasets

The most “load-bearing” datasets for partial spoof explainability are those with **segment labels** (or explicit region-location tasks), because explainability claims about “where the fake is” can be quantitatively evaluated.

**ASVspoof (2019–2021)** remains foundational. ASVspoof 2019 brought in a tandem, ASV-centric evaluation design (t-DCF) and provided official CQCC-GMM and LFCC-GMM baselines; ASVspoof 2021 adds a **DeepFake (DF) task** emphasizing **compressed/manipulated speech with codecs typical of online/social channels**, explicitly to test robustness beyond “clean lab audio.” citeturn16view2turn14search1turn15view0turn9view0turn21search3turn21search11

**PartialSpoof** is designed around the partial spoof scenario (“synthesized/transformed speech segments embedded into a bona fide utterance”) and was extended to include **segment labels at multiple temporal resolutions**, reportedly down to tens of milliseconds. This multi-resolution labeling matters for explainability because “the model flagged 4.8–5.1s” can be too coarse if the true replacement is sub-phonemic; conversely, too-fine resolution can inflate false alarms and complicate evaluation. citeturn8view1turn5search1turn9view2turn16view3

**ADD 2023** explicitly operationalizes partial-fake realism. It goes beyond binary detection to include “identification of manipulated intervals” and provides a **Manipulation Region Location** track (Track 2). Its evaluation combines **sentence accuracy** (did you correctly decide that the utterance is partially fake?) with **segment F1** (did you localize the fake frames/segments?), with specific weighting (reported as 0.3 / 0.7 in the challenge paper). That is effectively a benchmarked definition of what “explainable localization” should support. citeturn8view0turn8view3turn11view1turn6search19turn3search7

**WaveFake** and **In-the-Wild** primarily target **generalization**. WaveFake delivers curated deepfake audio samples spanning multiple synthesis architectures and languages; In-the-Wild explicitly evaluates how detectors trained on lab datasets generalize to realistic public-figure audio collected from online sources. These are indispensable for court suitability because opposing experts will challenge whether a detector was validated under “conditions substantially similar” to the evidence (codec/noise/domain). citeturn21search2turn9view3turn6search4turn6search7turn10view0

**FakeAVCeleb (audio subset)** contains synthesized, lip-synced fake audio paired with video. While multimodal, it is relevant here because it explicitly includes synthesized fake audios created for realistic scenarios; an “audio-only” evaluation subset can stress detectors under content that was built to match visual timing. citeturn21search1turn21search5

**LENS-DF** (newer) is best understood as a *recipe* to generate more realistic evaluation conditions: longer duration audio, noise, and multi-speaker scenarios, along with detection and temporal localization protocols. This directly addresses a major mismatch between common 2–10s single-speaker lab benchmarks and real evidentiary recordings (calls, meetings). citeturn18view1turn15view0

A practical “court validation suite” therefore typically needs: (1) ASVspoof (especially 2021 DF) for codec/channel stress, (2) PartialSpoof and/or ADD 2023 Track 2 for segment localization, and (3) In-the-Wild (plus additional case-like audio) for domain shift. citeturn9view0turn8view0turn10view0turn6search4

## Detection and localization methods for partial audio spoofs

Audio deepfake detection pipelines are often discussed as “front-end features + back-end classifier,” but partial spoofing pushes systems toward explicit temporal modeling (frame/segment scoring, boundaries, and multi-resolution targets).

Handcrafted spectral/cepstral features remain important baselines:

- **CQCC** (constant-Q cepstral coefficients) uses a constant-Q transform with variable time–frequency resolution and is widely used as a spoofing countermeasure; its design is argued to capture “tell-tale signs of manipulation artefacts.” citeturn15view3turn14search2  
- **LFCC** is an MFCC-like cepstral feature set built on linearly-spaced triangular filterbanks (rather than mel-spaced). ASVspoof organizers provided LFCC-GMM baselines alongside CQCC-GMM baselines, and ASVspoof materials explicitly reference these baselines and implementations. citeturn14search1turn14search13turn16view0

In practice, these baselines are often used as **auditable reference points**: they are simple, reproducible, and easier to explain to courts than deep neural embeddings—though they may underperform on modern attacks and domain shift. citeturn6search4turn15view0

Neural and self-supervised approaches dominate top performance, but with explainability tradeoffs:

- **Raw-waveform neural CMs** like AASIST model temporal and spectral relationships via graph attention, starting from a RawNet2-style encoder and parallel temporal/spectral graph modules. This class of model is widely cited as strong on spoof detection benchmarks. citeturn15view1turn15view2  
- **Self-supervised encoders** (e.g., wav2vec 2.0 / WavLM-style embeddings) are repeatedly reported as effective front-ends for spoof detection and for partial spoof localization systems (including in ADD Track 2 system descriptions that fuse WavLM/Wav2vec-based models with other components). citeturn11view1turn23view1turn10view3

However, partial spoofing creates two “gotchas” that matter for both performance and explainability:

1) **Duration mismatch**: many CMs assume fixed-length training/evaluation segments (often ≥4 seconds), and performance can degrade on short utterances. Work explicitly addresses this by architectural changes (multi-scale blocks) and training strategies (dynamic chunk size, fine-tuning strategies) to improve short-utterance evaluation. This matters legally because call snippets or short voice notes are common evidence. citeturn23view1turn15view0

2) **Dataset bias toward boundary cues**: partial spoof datasets often contain biases in where the spoof occurs (position), how long it lasts, or what transition artifacts exist. Recent localization work explicitly warns that models can over-rely on transition/boundary artifacts, and proposes segment-aware learning and cross-segment mixing augmentation to reduce overfitting to recurring patterns. citeturn11view0turn9view2turn8view1

Localization approaches used in partial spoof systems typically take one of these forms:

- **Frame- or segment-level scoring**: slide a window (e.g., 20–40 ms hop) and output a time series of spoof probabilities; then binarize into suspicious intervals. PartialSpoof research explicitly treats segment-level detection as a core task and supports training/evaluation at multiple resolutions. citeturn8view1turn9view2turn5search1  
- **Boundary → segment authenticity**: detect cut points and then evaluate each segment’s authenticity. ADD 2023 winner-style systems formalize this pattern: boundary detection (splicing boundaries), plus frame-level deepfake scoring, plus an additional model trained on bona fide data to catch outliers, fused into a final score and predicted regions. citeturn11view1turn8view0  
- **Multi-task / multi-resolution learning**: jointly optimize utterance-level classification and segment localization, sometimes at several temporal resolutions (e.g., 20 ms up to hundreds of ms). This directly supports explainability by enabling claims like “the model is confident the utterance is partially spoofed, and it localizes the spoof to these ranges,” rather than forcing a brittle post-hoc localization. citeturn8view1turn16view3turn10view3

Evaluation for localization is itself nontrivial. ADD 2023 uses segment precision/recall/F1 plus sentence accuracy and a weighted final score. PartialSpoof researchers argue that point-based segment EER depends heavily on the chosen temporal resolution and propose **range-based EER** to better measure misclassified *ranges* instead of misclassified points/segments. citeturn8view0turn18view0turn17search0

## Explainability techniques and LLM integration as constrained narrator

A court-usable explanation must be **reproducible, bounded, and evidence-linked**. For partial audio spoofs, “explainability” is most defensible when it is anchored to (a) **measurable forensic traces** and (b) **time-localized model outputs** that can be independently re-run.

The table below compares major explanation methods relevant to audio partial spoof cases, emphasizing what signal evidence they rely on, what role (if any) an LLM should play, and how suitable they tend to be for judicial contexts.

| Method (detection/localization/explanation) | Signal features used | LLM role | Strengths | Weaknesses | Judicial applicability | Primary sources |
|---|---|---|---|---|---|---|
| CQCC-GMM baseline CM | CQCC (constant-Q cepstral) + GMM | None (LLM not needed) | Simple, reproducible baseline; grounded in signal processing; explainable “feature pipeline.” citeturn15view3turn14search2 | May lag on new generators/domain shift; limited localization unless adapted. citeturn6search4 | Useful as baseline sanity check and for methodology transparency | citeturn15view3turn14search2turn14search1 |
| LFCC-GMM baseline CM | LFCC + GMM | None | Simple; official baseline lineage from ASVspoof; easier court explanation than deep embeddings. citeturn14search1turn14search13 | Same limitations as above; performance variability across attacks. citeturn16view0 | Baseline comparator; can support “method not cherry-picked” arguments | citeturn14search1turn14search13turn16view0 |
| AASIST-class neural CM | Raw waveform → RawNet2-style encoder + spectro/temporal graph attention | None | Strong benchmark performance; captures complex cues beyond handcrafted features. citeturn15view1turn15view2 | Harder to explain; needs careful calibration and robustness testing | Admissible only with strong validation + error rate disclosure | citeturn15view1turn15view2turn9view0 |
| SSL front-end (wav2vec2/WavLM) + classifier | Self-supervised embeddings (plus simple back-end) | None | Often improves robustness; widely used in modern systems; adaptable to partial spoof pipelines. citeturn11view1turn23view1turn18view1 | Heavy models; susceptible to dataset bias; still requires explainable localization layer | Strong if validated under realistic codecs/noise and case-like data | citeturn23view1turn11view1turn18view1 |
| Multi-task / multi-resolution partial spoof CM | Joint utterance+segment objectives; segment labels at multiple resolutions | None | Built-in localization supports explainability; aligns output with “where is fake?” question. citeturn8view1turn5search1 | Needs careful eval metrics; risk of overfitting boundary artifacts | High potential if validated with appropriate localization metrics | citeturn8view1turn18view0turn16view3 |
| Boundary detection + frame-level authenticity + fusion | Boundary cues + deepfake frame scoring + bona fide outlier model | None | Naturally produces “evidence objects”: boundaries + suspicious frames; matches splicing narrative. citeturn11view1turn8view0 | Complex; multi-stage error compounding; needs rigorous auditing | Strong if each stage validated and fusion rules documented | citeturn11view1turn8view0turn8view3 |
| Segment-aware learning (reduce boundary-cue reliance) | Segment-level features; augmentation via cross-segment mixing | None | Addresses known bias risk: boundary reliance; improves generalization argument. citeturn11view0 | Newer; must be independently replicated | Helpful for arguing robustness beyond dataset artifacts | citeturn11view0turn18view0 |
| Saliency on spectrogram (IG/LRP/GradSHAP) | Spectrogram magnitude/learned features + gradients | LLM optional: summarize saliency *already computed* | Direct “heatmap” evidence; can be aligned to time ranges | Often hard to interpret; faithfulness can be misleading | Use cautiously; must quantify faithfulness and sanity-check | citeturn10view1turn7search2turn23view0 |
| Phoneme Discretized Saliency Maps (PDSM) | Saliency discretized by phoneme boundaries/PPGs | LLM optional: translate phoneme-level evidence into narrative | Designed to improve understandability and (claimed) faithfulness; closer to “words/phones are suspicious.” citeturn10view1 | Depends on ASR/forced alignment errors; phoneme mapping adds uncertainty | Promising, but requires reporting phoneme alignment error + robustness | citeturn10view1turn7search2 |
| Transformer attention roll-out explainability | Attention rollout for transformer audio classifiers | LLM optional: summarize attention evidence | Tailored to transformer audio; aims to improve explainability for deepfake audio classifiers citeturn10view2 | Attention ≠ explanation pitfall; needs validation | Supportive as a secondary explanation, not sole evidence | citeturn10view2turn23view0 |
| LLM as constrained narrator (recommended pattern) | **No raw signal ingestion required**; consumes structured forensic JSON + evidence links | **Primary role**: produce court-readable narrative from validated outputs | Standardizes reporting; supports traceable “chain-of-evidence”; can generate cross-exam-ready summaries | Risk: hallucination/confabulation if unconstrained | High only if output is bounded, citeable, reproducible, and audit-logged | citeturn13view0turn13view1turn11view3turn12view0 |

Two major takeaways from this table:

1) The **best judicial posture** is to treat explainability as “**structured evidence + disciplined reporting**,” not “pretty heatmaps.” Saliency/attention can help but must be validated for faithfulness and stability (and should be presented as *supporting* rather than *dispositive*). citeturn10view1turn23view0turn7search1  
2) LLMs are most defensible as **report generators** operating under **hard constraints**—a design decision aligned with NIST’s emphasis on managing generative AI risks and synthetic content transparency. citeturn13view0turn13view1turn19search1

### Reference architecture: forensic pipeline with an LLM narrator

```mermaid
flowchart TB
  A[Evidence intake\n(original file + device/context)] --> B[Preservation\nhash + timestamp + chain-of-custody log]
  B --> C[Forensic audio authenticity checks\nmetadata/format, ENF cues, discontinuities]
  B --> D[ML inference\n(1) detector + (2) temporal localizer]
  D --> E[Calibration + thresholding\n(case-specific operating point)]
  E --> F[Evidence objects\nsegments, scores, model/version, plots]
  C --> F
  F --> G[Immutable evidence store\nWORM / signed logs / audit trail]
  G --> H[LLM narrator (constrained)\nJSON-in → report-out]
  H --> I[Courtroom package\nmethods, error rates, exhibits]
```

The “preservation” block is not optional for court contexts: forensic audio best practices define authenticity analysis in terms of traces left by recording/editing and explicitly discuss cryptographic hashes and metadata as integrity and context elements, and recommend maintaining chain-of-custody where possible. citeturn11view3turn20search6  
For hash/time anchoring, common practice can leverage standards like RFC 3161 (trusted timestamp protocol) and digital evidence handling guidance like ISO/IEC 27037 and NIST SP 800-86. citeturn19search3turn19search0turn19search1

### Example explainable outputs (structured JSON + narrative template)

The key design choice is: **the LLM never invents evidence**; it only verbalizes what the pipeline already produced.

**JSON schema (illustrative, court-oriented)**

```json
{
  "case_metadata": {
    "case_id": "2026-XX-123",
    "examiner": {"name": "A. Expert", "organization": "Lab"},
    "jurisdiction_assumption": "US federal/state + EU examples",
    "toolchain": [
      {"name": "localizer_model", "version": "v3.2.1", "hash": "sha256:..."},
      {"name": "feature_extractor", "version": "wav2vec2-base", "hash": "sha256:..."},
      {"name": "ffmpeg", "version": "6.1", "hash": "sha256:..."}
    ]
  },
  "evidence_integrity": {
    "original_file": {
      "filename": "Exhibit_12A.wav",
      "sha256": "…",
      "bytes": 12345678,
      "claimed_provenance": "phone recording",
      "acquisition_notes": "received via counsel on 2026-02-10"
    },
    "timestamps": [
      {"type": "rfc3161", "tsa": "ExampleTSA", "token_id": "…"}
    ],
    "chain_of_custody": [
      {"event": "received", "datetime_utc": "2026-02-10T14:22:00Z", "by": "Lab intake"},
      {"event": "hashed", "datetime_utc": "2026-02-10T14:24:00Z", "by": "Lab intake"}
    ]
  },
  "audio_properties": {
    "duration_s": 47.83,
    "sample_rate_hz": 16000,
    "channels": 1,
    "detected_codec": "PCM",
    "compression_artifacts_flag": false
  },
  "model_outputs": {
    "utterance_level": {"p_fake": 0.93, "calibrated": true},
    "segments": [
      {
        "t_start_s": 12.40,
        "t_end_s": 13.05,
        "p_fake_segment": 0.88,
        "evidence": {
          "score_trace_ref": "store://scores/segtrace_001.csv",
          "saliency_ref": "store://plots/saliency_001.png",
          "phoneme_alignment_ref": "store://align/ppg_001.json"
        }
      }
    ],
    "boundary_events": [
      {"t_s": 12.39, "p_boundary": 0.91, "method": "boundary_detector"}
    ]
  },
  "validation_context": {
    "benchmarks_used": [
      {"name": "ADD2023_track2", "metric": "segment_F1 + sentence_acc", "result": "see report"},
      {"name": "PartialSpoof", "metric": "range_EER", "result": "see report"}
    ],
    "operating_point": {
      "chosen_for": "minimize false accusations",
      "threshold_segment": 0.80
    }
  },
  "interpretation_constraints": {
    "llm_policy": "Narrate only from JSON fields; no new claims; cite evidence refs; state uncertainty."
  }
}
```

This format explicitly supports Daubert-style cross-examination topics (testability, standards, error rates, operational controls) by packaging: versioned tools, benchmark context, and operating points. citeturn13view3turn4search1turn18view0turn8view0

**Natural-language narrative template (LLM narrator output)**

> “I examined Exhibit 12A (SHA-256 …) under preserved chain-of-custody events documented in the case log. The automated localization model reports a high utterance-level probability of partial manipulation (p=0.93, calibrated). The system flagged one contiguous interval from 12.40s to 13.05s (p=0.88). A boundary detector independently indicates a probable splice boundary at 12.39s (p=0.91).  
>  
> These outputs indicate the recording contains at least one segment whose acoustic characteristics are inconsistent with the surrounding speech under the model’s training/validation conditions. Supporting material includes the segment score trace and saliency/phoneme-aligned explanation artifacts referenced in the evidence store.  
>  
> Limitations: The phoneme alignment used to summarize saliency has known error (PER reported in the alignment model documentation), and the system has reduced reliability under severe re-encoding or unseen generators. Reported conclusions therefore attach to the flagged interval(s) only, not to the entire recording.”

This template is consistent with forensic audio best-practice language that emphasizes “assessment,” “interpretation guidelines,” and credibility/robustness rather than absolutist claims. citeturn11view3turn12view1turn12view0

### Metrics for detection, localization, and explanation quality

Detection metrics used in the dominant benchmarks include:

- **EER** (equal error rate), used for ASVspoof DF and widely elsewhere. citeturn9view0turn21search3  
- **min t-DCF** (minimum tandem detection cost function) for LA/PA tasks where spoofing countermeasures are evaluated with ASV considerations. citeturn9view0turn9view1turn16view2  
- ADD 2023 uses EER-derived measures for some tracks and explicitly defines Track 2’s final score as a weighted combination of **sentence accuracy** and **segment F1**. citeturn8view0turn8view3

Localization metrics include:

- **Segment precision/recall/F1** (ADD Track 2). citeturn8view0turn5search6  
- **Range-based EER** for spoof localization, proposed specifically because point-based localization evaluation can be misleading at different temporal resolutions. citeturn18view0turn17search0  
- **IoU** is standard in temporal/spatial localization in other deepfake domains; for audio partial spoof localization, IoU can be computed, but the most “native” metrics in the PartialSpoof/ADD ecosystem are segment F1 and range-based/point-based EER variants. citeturn18view0turn8view0

Explanation evaluation metrics (for saliency/attribution) should separate **human interpretability** from **faithfulness**:

- Foundational interpretability evaluation taxonomies emphasize that “interpretability” is context-dependent and should be evaluated via task-grounded experiments, not only informal inspection. citeturn7search1  
- For perturbation-based faithfulness (AOPC-style), recent work shows AOPC can be misleading across models and proposes normalized AOPC for more meaningful comparisons. citeturn23view0  
- Audio-specific explainability work (PDSM) claims more faithful and more understandable explanations by discretizing saliency using phoneme boundaries and provides an explicit framework for evaluating faithfulness in that setting. citeturn10view1

## Privacy, security, deployment patterns, and courtroom readiness

### Evidence integrity and chain of custody for audio

Forensic audio authenticity best practices provide the vocabulary and process discipline courts expect:

- The ENFSI digital audio authenticity BPM defines authenticity analysis as grounded in “traces left within the recording during recording and subsequent editing,” defines context/metadata and cryptographic hashing functions, and frames robust, unbiased conclusions as core objectives. citeturn11view3turn20search2  
- SWGDE guidance distinguishes audio authentication (consistency with alleged manner of production) and notes that “authentication” has a legal meaning in establishing foundation for admission. citeturn12view0turn12view1  
- ENFSI ENF guidelines emphasize ENF analysis as one technique to assist authenticity determinations and explicitly discuss producing court-presentable results. citeturn12view3turn5search10

For digital evidence handling generally (relevant to audio files as digital artifacts), ISO/IEC 27037 explicitly covers identification, collection, acquisition, and preservation, while NIST SP 800-86 provides guidance for establishing forensic capability and procedures. citeturn19search0turn19search1

A practical anti-tamper stack for audio evidence packages typically includes:

- **Immutable identifiers**: cryptographic hash of original and derived working copies; ENFSI explicitly frames hash values as substantiating integrity. citeturn11view3  
- **Time anchoring**: trusted timestamp protocols like RFC 3161 (standardized request/response time-stamping). citeturn19search3  
- **Audit-logged workflows** aligned to forensic procedure guidance (ISO/NIST) and lab best practices (ENFSI/SWGDE). citeturn19search0turn19search1turn12view1

### On‑prem vs cloud, real-time vs batch

From a forensic defensibility perspective (not a cost perspective), the deployment choice should be justified by **data sensitivity and reproducibility**:

- **On‑prem** is often preferred when evidentiary audio contains sensitive personal data and when labs must guarantee toolchain immutability and restricted access; this aligns with the general “establish a forensic capability with policies and procedures” guidance in NIST SP 800-86 and with forensic lab BPMs emphasizing validated methods and quality assurance. citeturn19search1turn11view3turn12view1  
- **Cloud** can be appropriate for non-sensitive or consented corpora and for scalable benchmarking, but requires strict logging, cryptographic integrity controls, and clear documentation to survive admissibility challenges (who accessed the data, what changed, which model version ran). NIST’s synthetic content guidance and GAI risk profile emphasize provenance, testing, and governance actions that map well onto this need. citeturn13view1turn13view0  
- **Real-time detection** (call centers, fraud prevention) pressures latency and short-utterance performance. Research explicitly notes that spoof CMs often assume longer fixed durations and degrade on short utterances, motivating changes like AASIST2 and duration-mismatch strategies. citeturn23view1  
- **Batch forensic analysis** is typical for court: prioritize accuracy, repeatability, and documentation over latency. ENFSI/SWGDE practice documents emphasize method selection, quality, and interpretation rather than real-time constraints. citeturn11view3turn12view1

### Legal admissibility implications for audio explainability

In **US federal practice**, expert testimony admissibility hinges on FRE 702 reliability/helpfulness, with Daubert-style factors (testing, peer review, error rates, standards, acceptance) and the judge’s gatekeeping role. Your explainability package should therefore explicitly include: benchmark results and error rates in relevant conditions (codecs/noise), tool/version controls, and documented standards/controls. citeturn13view3turn4search1turn9view0turn15view0turn8view0

In **Frye** jurisdictions, “general acceptance” remains a key hurdle; relying on community benchmarks (ASVspoof, ADD) and forensic best-practice manuals (ENFSI/SWGDE) supports arguments that a technique is not idiosyncratic. citeturn20search3turn9view0turn12view0turn11view3

For **EU contexts**, evidentiary rules vary by Member State; however, for digital processes, there are EU-level instruments relevant to **electronic timestamps**. eIDAS Article 41 states that electronic timestamps should not be denied legal effect/admissibility solely due to electronic form, and qualified electronic time stamps carry a presumption of accuracy/integrity of the bound data. This is directly relevant to building an “immutable evidence package” using hash + timestamp tokens. citeturn20search0turn20search4turn20search24

Finally, **avoid putting an LLM in the chain of scientific inference**. If an LLM is used, courts will likely treat it as (at best) an automated report-writing aid and (at worst) a source of unverifiable assertions. NIST’s generative AI profile explicitly frames GAI risk management around governance, provenance, testing, and incident disclosure—concepts that strongly suggest constrained generation and auditable provenance for any LLM output used in legal settings. citeturn13view0turn13view1

### Courtroom readiness checklist for partial audio deepfake cases

**Collection & preservation**
- Record and preserve **original** audio in the best available form; compute and record cryptographic hashes for originals and working copies. citeturn11view3turn19search0  
- Maintain a **chain-of-custody log** (who, when, what). ENFSI explicitly recommends chain-of-custody where possible. citeturn20search6turn11view3  
- If using timestamps, use standardized trusted timestamp protocols (e.g., RFC 3161) and retain timestamp tokens. citeturn19search3turn19search1  

**Method validation & error rates**
- Report detection and localization performance on **relevant benchmarks** (ASVspoof 2021 DF for codecs; ADD 2023 Track 2 or PartialSpoof for localization). citeturn15view0turn8view0turn8view1turn9view0  
- Use localization metrics that match the claim: segment F1 and/or range-based EER (don’t hide behind a single threshold). citeturn8view0turn18view0  
- Demonstrate robustness to **domain shift** (e.g., train on ASVspoof/WaveFake, test on In-the-Wild), because overfitting between progress/eval partitions is a known issue even in ASVspoof 2021 DF. citeturn10view0turn9view0turn15view0turn21search2  

**Explainability artifacts**
- Provide time-localized exhibits: flagged intervals, boundary events, and supporting plots (spectrogram slices, saliency aligned to phoneme/word where feasible). citeturn10view1turn11view1turn8view0  
- If using saliency, quantify faithfulness (prefer normalized/robust metrics over naive AOPC comparisons) and emphasize limitations. citeturn23view0turn7search1  

**LLM usage controls (if any)**
- LLM must produce **bounded narratives** from evidence JSON only; no free-form speculation; log prompts, model version, and outputs; store as part of the evidence package. citeturn13view0turn13view1turn19search1  

**Cross-examination readiness**
- Be prepared to answer: What is the known/potential error rate under **codec/noise** conditions like the case? What standards/controls govern threshold choice? How was the tool validated? How do you know the model isn’t keying on boundary artifacts? citeturn13view3turn11view0turn18view0turn15view0  

### Recommended evaluation experiments (to support judicial-grade claims)

A court-facing validation package should include, at minimum:

- **Codec/channel robustness**: evaluate on ASVspoof 2021 DF conditions emphasizing codecs used in social media; include re-encoding tests and telephony-like compression. citeturn15view0turn21search23  
- **Partial spoof localization**: train/evaluate on PartialSpoof and/or ADD 2023 Track 2; report segment F1 and range-based EER to avoid temporal-resolution artifacts. citeturn8view0turn18view0turn5search1  
- **Boundary-artifact resistance**: evaluate models with and without segment-aware augmentation (cross-segment mixing) and report whether localization performance holds when boundaries are smoothed/cross-faded. citeturn11view0turn11view1  
- **Domain shift**: train on lab datasets (ASVspoof/WaveFake) and evaluate on In-the-Wild; document performance degradation and calibration drift. citeturn10view0turn21search2turn6search7  
- **Long-form / multi-speaker realism**: use LENS-DF-style conditions (longer clips, noise, multiple speakers) to test whether localization remains stable outside typical short single-speaker benchmarks. citeturn18view1turn23view1  
- **Adversarial / anti-forensic tests**: include noise injection, re-recording (speaker playback + re-capture), time-stretch/pitch shift, denoising, and cross-codec laundering; treat these as foreseeable attacks given the emphasis on compression/transmission realism in ASVspoof 2021 DF. citeturn15view0turn9view0  

## Appendix: primary sources URLs

```text
ASVspoof 2021 challenge paper (metrics, EER, min t-DCF): https://arxiv.org/pdf/2109.00537
ASVspoof 2021 DF realism/codecs discussion: https://arxiv.org/pdf/2210.02437
ASVspoof 2021 evaluation plan: https://www.asvspoof.org/asvspoof2021/asvspoof2021_evaluation_plan.pdf
ASVspoof 2019 evaluation plan: https://datashare.ed.ac.uk/bitstream/handle/10283/3336/asvspoof2019_evaluation_plan.pdf
ASVspoof 2019 database paper: https://arxiv.org/pdf/1911.01601
CQCC paper: https://www.asvspoof.org/papers/CSL_CQCC.pdf
t-DCF paper: https://arxiv.org/abs/1804.09618
AASIST paper (AASIST): https://arxiv.org/abs/2110.01200
AASIST2 short-utterance paper: https://arxiv.org/pdf/2309.08279

PartialSpoof (arXiv): https://arxiv.org/abs/2204.05177
PartialSpoof GitHub: https://github.com/nii-yamagishilab/PartialSpoof
Range-based EER (Interspeech 2023): https://www.isca-archive.org/interspeech_2023/zhang23v_interspeech.pdf

ADD 2023 challenge paper: https://arxiv.org/pdf/2305.13774
ADD 2023 analysis paper: https://arxiv.org/abs/2408.04967
ADD 2023 Track 2 (competition page): https://codalab.lisn.upsaclay.fr/competitions/11361
ADD 2023 Track 2 top system description (arXiv): https://arxiv.org/pdf/2308.10281

WaveFake paper: https://arxiv.org/abs/2111.02813
WaveFake dataset (Zenodo): https://zenodo.org/records/5642694
In-the-Wild generalization paper: https://arxiv.org/pdf/2203.16263
In-the-Wild dataset (Hugging Face): https://huggingface.co/datasets/mueller91/In-The-Wild
FakeAVCeleb paper (NeurIPS D&B): https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/file/d9d4f495e875a2e075a1a4a6e1b9770f-Paper-round2.pdf

PDSM explainability paper (Interspeech 2024): https://www.isca-archive.org/interspeech_2024/gupta24b_interspeech.pdf
Attention roll-out explainability paper (OpenReview PDF): https://openreview.net/pdf/e7538d6cb10fefbb335617742b271ff0af53c441.pdf
Normalized AOPC (faithfulness metric): https://arxiv.org/abs/2408.08137
Interpretability evaluation taxonomy: https://arxiv.org/abs/1702.08608

ENFSI BPM Digital Audio Authenticity Analysis: https://enfsi.eu/wp-content/uploads/2022/12/FSA-BPM-002_BPM-for-Digital-Audio-Authenticity-Analysis.pdf
ENFSI ENF guidelines (2009): https://enfsi.eu/wp-content/uploads/2016/09/forensic_speech_and_audio_analysis_wg_-_best_practice_guidelines_for_enf_analysis_in_forensic_authentication_of_digital_evidence_0.pdf
SWGDE Best Practices for Digital Audio Authentication: https://www.swgde.org/documents/published-complete-listing/15-a-001-swgde-best-practices-for-digital-audio-authentication/
SWGDE Best Practices for Forensic Audio (v2.5): https://www.swgde.org/wp-content/uploads/2023/11/2022-06-09-SWGDE-Best-Practices-for-Forensic-Audio_v2.5.pdf

NIST AI 600-1 (Generative AI profile): https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.600-1.pdf
NIST AI 100-4 (Reducing risks posed by synthetic content): https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.100-4.pdf
NIST SP 800-86 (forensic capability guidance): https://nvlpubs.nist.gov/nistpubs/legacy/sp/nistspecialpublication800-86.pdf
ISO/IEC 27037 overview: https://www.iso.org/standard/44381.html
RFC 3161 Time-Stamp Protocol: https://www.ietf.org/rfc/rfc3161.txt

FRE 702 (LII): https://www.law.cornell.edu/rules/fre/rule_702
Daubert opinion (U.S. Reports PDF): https://tile.loc.gov/storage-services/service/ll/usrep/usrep509/usrep509579/usrep509579.pdf
Frye v. United States (PDF): https://fpamed.com/wp-content/uploads/2013/09/Frye-v-US-1923.pdf

eIDAS Regulation (EU) 910/2014 (PDF): https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=CELEX%3A32014R0910
European Commission eSignature FAQ (timestamps): https://ec.europa.eu/digital-building-blocks/sites/spaces/DIGITAL/pages/880312429/eSignature%2BFAQ
```