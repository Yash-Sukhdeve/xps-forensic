"""Tests for evidence packaging."""
import json
from xps_forensic.evidence.schema import EvidencePackage, validate_evidence


class TestEvidenceSchema:
    def test_create_evidence(self):
        pkg = EvidencePackage(
            utterance_id="utt_001",
            detector="BAM",
            calibration_method="temperature",
            prediction_set={"partially_fake"},
            coverage_guarantee=0.95,
            segment_predictions=[0, 0, 1, 1, 1, 0, 0],
            crc_threshold=0.45,
            tFNR_guarantee=0.10,
            phoneme_attributions=[
                {"phoneme": "AH", "saliency": 0.8, "start": 0.1, "end": 0.15},
            ],
        )
        assert pkg.utterance_id == "utt_001"

    def test_to_json(self):
        pkg = EvidencePackage(
            utterance_id="utt_001",
            detector="BAM",
            calibration_method="temperature",
            prediction_set={"partially_fake"},
            coverage_guarantee=0.95,
            segment_predictions=[0, 0, 1, 1, 0],
            crc_threshold=0.45,
            tFNR_guarantee=0.10,
        )
        j = pkg.to_json()
        parsed = json.loads(j)
        assert parsed["utterance_id"] == "utt_001"
        assert parsed["detector"] == "BAM"
        assert "daubert_factors" in parsed

    def test_validate_complete(self):
        pkg = EvidencePackage(
            utterance_id="utt_001",
            detector="BAM",
            calibration_method="temperature",
            prediction_set={"partially_fake"},
            coverage_guarantee=0.95,
            segment_predictions=[0, 1, 0],
            crc_threshold=0.45,
            tFNR_guarantee=0.10,
        )
        errors = validate_evidence(pkg)
        assert len(errors) == 0

    def test_validate_catches_errors(self):
        pkg = EvidencePackage(
            utterance_id="",
            detector="",
            calibration_method="temperature",
            prediction_set=set(),
            coverage_guarantee=1.5,
            segment_predictions=[0, 1],
            crc_threshold=None,
        )
        errors = validate_evidence(pkg)
        assert len(errors) >= 4  # missing id, detector, empty set, invalid coverage, no threshold
