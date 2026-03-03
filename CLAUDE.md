# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **research documentation project** on explainability for audio deepfakes and partial audio spoofs in judicial contexts. The primary deliverable is `deep-research-report.md` — a comprehensive technical synthesis covering detection, localization, and court-suitable explainability.

## Domain Context

The report bridges four disciplines:
- **Machine learning**: Detection/localization methods (AASIST, wav2vec 2.0, WavLM, frame-level classifiers, boundary detectors)
- **Forensic science**: Audio authenticity analysis, chain-of-custody, ENFSI/SWGDE standards
- **Legal procedure**: FRE 702/Daubert factors, Frye general acceptance, EU eIDAS
- **AI governance**: NIST AI risk framework, generative AI risk profiles

Key architectural concept: LLMs serve as **constrained narrators** (JSON-in → report-out), never as primary detectors. Explainability is a **layered evidence stack** (forensic signals → ML localization → human-understandable mapping).

## Key Benchmarks & Datasets Referenced

- **ASVspoof 2019/2021**: Foundational CM evaluation (codec robustness focus in 2021)
- **PartialSpoof**: Multi-resolution segment-level labeling for partial spoof detection
- **ADD 2023 Track 2**: Manipulation region localization
- **WaveFake / In-the-Wild / FakeAVCeleb / LENS-DF**: Generalization and domain-shift testing

## Working with This Repository

- No build system, tests, or dependencies — this is documentation-only
- The report uses Markdown with Mermaid diagrams, JSON schema examples, and comparison tables
- Citations use `citeturn[X]view[Y]` format from the research sourcing tool
- All claims must be backed by scientific evidence with cited sources (per project rules)
- When extending the report, maintain the existing structure: Executive Summary → Scope → Benchmarks → Methods → Explainability → Architecture → Examples → Metrics → Legal/Deployment → Appendix

## Reference Architecture (from the report)

The prescribed forensic pipeline flows: Evidence intake → Preservation (hash + timestamp + chain-of-custody) → Forensic audio checks → ML inference (detector + temporal localizer) → Calibration → Evidence objects → Immutable evidence store → LLM narrator → Courtroom package.

<!-- UWS-BEGIN -->
## UWS Workflow System

This project uses UWS (Universal Workflow System) for context persistence across sessions.

### Commands
- `/uws` - Show all available UWS commands
- `/uws-status` - Show current workflow state
- `/uws-checkpoint "msg"` - Create checkpoint
- `/uws-recover` - Full context recovery after break
- `/uws-handoff` - Prepare handoff before ending session
- `/uws-sdlc <action>` - Manage SDLC phases (status/start/next/goto/fail/reset)
- `/uws-research <action>` - Manage research phases (status/start/next/goto/reject/reset)

### Workflow Files
- `.workflow/state.yaml` - Current phase and checkpoint
- `.workflow/handoff.md` - Human-readable context (READ THIS ON SESSION START)
- `.workflow/checkpoints.log` - Checkpoint history

### Session Workflow
1. **Start**: Context is automatically loaded via SessionStart hook
2. **During**: Create checkpoints at milestones with `/uws-checkpoint`
3. **End**: Run `/uws-handoff` to update context for next session

### Auto-Checkpoint
UWS automatically creates checkpoints before context compaction to prevent state loss.
<!-- UWS-END -->

