"""E6: Codec Stress Test.

Re-encode PartialSpoof eval through AAC/Opus/AMR/G.711.
Report metric degradation per codec.
Test CPSL coverage under codec distortion.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from xps_forensic.utils.config import load_config


CODEC_CONFIGS = {
    "aac": {"ext": ".m4a", "cmd": "-c:a aac -b:a 128k"},
    "opus": {"ext": ".opus", "cmd": "-c:a libopus -b:a 64k"},
    "amr": {"ext": ".amr", "cmd": "-c:a libopencore_amrnb -ar 8000 -ac 1"},
    "g711": {"ext": ".wav", "cmd": "-c:a pcm_alaw -ar 8000 -ac 1"},
}


def transcode_audio(
    input_path: str,
    codec: str,
    output_path: str | None = None,
) -> str:
    """Transcode audio through a codec using ffmpeg.

    Returns path to transcoded (then re-decoded to WAV) file.
    """
    config = CODEC_CONFIGS[codec]

    if output_path is None:
        output_path = tempfile.mktemp(suffix=".wav")

    intermediate = tempfile.mktemp(suffix=config["ext"])

    # Encode
    cmd_encode = f"ffmpeg -y -i {input_path} {config['cmd']} {intermediate}"
    subprocess.run(cmd_encode.split(), capture_output=True, check=True)

    # Decode back to WAV 16kHz
    cmd_decode = f"ffmpeg -y -i {intermediate} -ar 16000 -ac 1 {output_path}"
    subprocess.run(cmd_decode.split(), capture_output=True, check=True)

    # Cleanup intermediate
    Path(intermediate).unlink(missing_ok=True)

    return output_path


def run_e6(cfg=None):
    if cfg is None:
        cfg = load_config()

    output_dir = Path(cfg.experiments.output_dir) / "e6_codec"
    output_dir.mkdir(parents=True, exist_ok=True)

    codecs = cfg.experiments.codecs
    results = {codec: {} for codec in codecs}

    print("E6: Codec Stress Test")
    print(f"Codecs: {codecs}")
    print("Run E1 inference on re-encoded audio for each codec.")
    print("Compare metrics against uncompressed baseline.")

    # Save placeholder — actual execution requires E1 integration
    output_file = output_dir / "results.json"
    with open(output_file, "w") as f:
        json.dump({"codecs": codecs, "status": "ready_to_run"}, f, indent=2)
    print(f"Config saved to {output_file}")


if __name__ == "__main__":
    run_e6()
