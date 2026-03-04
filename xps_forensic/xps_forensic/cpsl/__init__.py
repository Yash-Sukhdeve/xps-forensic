"""CPSL: Conformalized Partial Spoof Localization.

Two-stage conformal prediction framework:
- Stage 1: SCP+APS for utterance-level ternary classification
- Stage 2: CRC on tFNR for segment-level localization
- Composed guarantee: P(both correct) >= (1-alpha1)(1-alpha2)

Key design decision: CP at utterance level ONLY (NOT frame-level --
temporal autocorrelation violates exchangeability).
"""
from .nonconformity import (
    max_score,
    logsumexp_score,
    compute_nonconformity,
)
from .scp_aps import SCPAPS
from .crc import ConformalRiskControl
from .composed import CPSLPipeline, CPSLResult

__all__ = [
    "max_score",
    "logsumexp_score",
    "compute_nonconformity",
    "SCPAPS",
    "ConformalRiskControl",
    "CPSLPipeline",
    "CPSLResult",
]
