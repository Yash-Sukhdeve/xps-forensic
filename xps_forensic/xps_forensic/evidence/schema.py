"""Evidence packaging schema for forensic output.

Produces structured JSON aligned with Daubert/FRE 702 factors.
Framing: "forensically defensible" NOT "court-admissible."
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone


@dataclass
class EvidencePackage:
    """Structured evidence output from XPS-Forensic pipeline."""

    # Identification
    utterance_id: str
    detector: str
    calibration_method: str

    # CPSL Stage 1
    prediction_set: set[str]
    coverage_guarantee: float

    # CPSL Stage 2
    segment_predictions: list[int] | None = None
    crc_threshold: float | None = None
    tFNR_guarantee: float | None = None

    # PDSM-PS
    phoneme_attributions: list[dict] | None = None

    # Metadata
    schema_version: str = "1.0"
    pipeline_version: str = "0.1.0"
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    @property
    def daubert_factors(self) -> dict[str, str]:
        """Map pipeline components to Daubert admissibility factors."""
        return {
            "testability": (
                "Pipeline evaluated on 4 independent datasets with "
                "reproducible bootstrap CIs."
            ),
            "peer_review": (
                f"Detector ({self.detector}) published at peer-reviewed venue. "
                "Calibration and conformal methods based on established theory."
            ),
            "known_error_rate": (
                f"Utterance coverage ≥ {self.coverage_guarantee:.0%}. "
                f"Segment tFNR ≤ {self.tFNR_guarantee or 'N/A'}."
            ),
            "standards": (
                "Aligned with ENFSI BPM for digital audio authenticity, "
                "SWGDE best practices, NIST AI 600-1."
            ),
            "general_acceptance": (
                "Conformal prediction: Vovk et al. (2005), 20+ years of theory. "
                "Saliency methods: established in ML interpretability."
            ),
        }

    def to_dict(self) -> dict:
        """Convert to serializable dict."""
        d = {
            "schema_version": self.schema_version,
            "pipeline_version": self.pipeline_version,
            "timestamp": self.timestamp,
            "utterance_id": self.utterance_id,
            "detector": self.detector,
            "calibration_method": self.calibration_method,
            "cpsl_stage1": {
                "prediction_set": sorted(self.prediction_set),
                "coverage_guarantee": self.coverage_guarantee,
            },
            "cpsl_stage2": {
                "segment_predictions": self.segment_predictions,
                "crc_threshold": self.crc_threshold,
                "tFNR_guarantee": self.tFNR_guarantee,
            },
            "pdsm_ps": {
                "phoneme_attributions": self.phoneme_attributions or [],
            },
            "daubert_factors": self.daubert_factors,
        }
        return d

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)


def validate_evidence(pkg: EvidencePackage) -> list[str]:
    """Validate evidence package completeness.

    Returns list of error messages (empty = valid).
    """
    errors = []
    if not pkg.utterance_id:
        errors.append("Missing utterance_id")
    if not pkg.detector:
        errors.append("Missing detector name")
    if not pkg.prediction_set:
        errors.append("Empty prediction set")
    if pkg.coverage_guarantee <= 0 or pkg.coverage_guarantee > 1:
        errors.append(f"Invalid coverage guarantee: {pkg.coverage_guarantee}")
    if pkg.segment_predictions is not None and pkg.crc_threshold is None:
        errors.append("Segment predictions present but no CRC threshold")
    return errors
