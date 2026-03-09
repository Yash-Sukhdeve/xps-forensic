"""Tests for configuration loading."""
from xps_forensic.utils.config import load_config


def test_load_default_config():
    cfg = load_config()
    assert cfg.project.name == "xps-forensic"
    assert cfg.data.partialspoof.sample_rate == 16000
    assert cfg.calibration.methods == ["platt", "temperature", "isotonic"]
    assert cfg.cpsl.alpha_utterance == 0.05
    assert cfg.cpsl.alpha_segment == 0.10


def test_config_device_detection():
    cfg = load_config()
    assert cfg.device in ("cuda", "cpu")


def test_config_variable_resolution():
    cfg = load_config()
    assert cfg.data.partialspoof.path == f"{cfg.data.root}/PartialSpoof"
    assert cfg.data.hqmpsd.path == f"{cfg.data.root}/HQ-MPSD-EN"


def test_config_override():
    cfg = load_config(overrides={"project": {"seed": 99}})
    assert cfg.project.seed == 99
    # Other values should remain
    assert cfg.project.name == "xps-forensic"


def test_config_dotdict_access():
    cfg = load_config()
    # Nested dot access
    assert cfg.detectors.bam.name == "BAM"
    assert cfg.detectors.sal.backbone == "wavlm-large"
    assert cfg.cpsl.classes == ["real", "partially_fake", "fully_fake"]
