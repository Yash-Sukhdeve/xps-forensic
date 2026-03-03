"""Detector wrappers for spoof detection models.

Each wrapper imports from an external detector repository cloned into
``external/`` via sys.path manipulation. The external source code is
never modified; all adaptation logic lives in the wrapper classes.

Available detectors:
    - BAMDetector: Boundary-aware Attention Mechanism (Interspeech 2024)
    - SALDetector: Segment-Aware Learning (arXiv 2026)
    - CFPRFDetector: Coarse-to-Fine Proposal Refinement (ACM MM 2024)
    - MRMDetector: Multi-Resolution Model baseline (IEEE/ACM TASLP 2023)
"""
from xps_forensic.detectors.base import BaseDetector, DetectorOutput
from xps_forensic.detectors.bam import BAMDetector
from xps_forensic.detectors.sal import SALDetector
from xps_forensic.detectors.cfprf import CFPRFDetector
from xps_forensic.detectors.mrm import MRMDetector

__all__ = [
    "BaseDetector",
    "DetectorOutput",
    "BAMDetector",
    "SALDetector",
    "CFPRFDetector",
    "MRMDetector",
]
