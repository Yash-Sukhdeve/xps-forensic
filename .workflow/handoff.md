# Workflow Handoff

**Last Updated**: 2026-03-02
**Phase**: experiment_design (complete) → implementation (next)
**Checkpoint**: CP_IMPLEMENTATION_PLAN_COMPLETE

---

## Current Status

Research design AND implementation plan for XPS-Forensic paper complete.

**Design document:** `docs/plans/2026-03-02-xps-forensic-design.md`
**Implementation plan:** `docs/plans/2026-03-02-xps-forensic-implementation.md`

## What Was Accomplished

1. UWS research workflow initiated (hypothesis → literature_review → experiment_design)
2. Comprehensive literature review across 4 domains (ML, forensics, legal, AI governance)
3. Novelty claims verified (CPSL, PDSM-PS — all confirmed novel)
4. 4 detectors selected with evidence (BAM primary, SAL SOTA, CFPRF secondary, MRM baseline)
5. 8 experiments designed with statistical rigor
6. Critical theoretical issues identified and resolved (exchangeability, conditional coverage)
7. Feasibility confirmed on RTX 4080 (2.5-7 days wall time)
8. Scoop risk assessed (all clear, but publish quickly)
9. **Implementation plan written** — 32 TDD tasks across 8 phases, complete code

## Key Design Decisions

- **BAM as primary detector** (peer-reviewed, Interspeech 2024) — SAL (arXiv) as SOTA comparison
- **Utterance-level conformal prediction** (not frame-level) — addresses exchangeability
- **Two-stage CPSL**: SCP+APS (utterance) + CRC on tFNR (segment)
- **PDSM-PS**: phoneme-discretized saliency applied to CPSL-flagged segments only
- **Framing**: "forensically defensible" NOT "court-admissible"

## Next Actions

- [x] Transition to implementation planning (writing-plans skill)
- [ ] Execute implementation plan (32 tasks, ~5h coding)
- [ ] Download/prepare all 4 datasets
- [ ] Run experiments E1-E8 (2.5-7 days GPU time)
- [ ] Write manuscript

## Research Artifacts

| File | Description |
|------|-------------|
| `docs/plans/2026-03-02-xps-forensic-design.md` | Full design document |
| `docs/plans/2026-03-02-xps-forensic-implementation.md` | 32-task implementation plan |
| `deep-research-report.md` | Background literature synthesis |
| `.workflow/dataset-research.md` | Dataset analysis |
| `.workflow/xai-research.md` | XAI methods research |
| `.workflow/detector-selection.md` | Detector evaluation |
| `.workflow/experiment-review.md` | Experimental rigor review |
| `.workflow/gap-analysis.md` | Gap analysis |
| `.workflow/feasibility-analysis.md` | Computational feasibility |
| `.workflow/scoop-analysis.md` | Competitor analysis |
| `.workflow/detector-risk-analysis.md` | BAM vs SAL risk analysis |

## Blockers

None currently.

## Context

This project uses UWS for maintaining context across Claude Code sessions.

### Quick Commands
- `/uws` - Show all available commands
- `/uws-status` - Check current workflow state
- `/uws-checkpoint "message"` - Create a checkpoint
- `/uws-recover` - Full context recovery
- `/uws-handoff` - Prepare for session end
