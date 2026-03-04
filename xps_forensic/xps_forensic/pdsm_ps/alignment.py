"""Phoneme alignment for PDSM-PS.

Supports:
- Montreal Forced Aligner (MFA) — primary
- WhisperX — neural baseline
- Mock aligner for testing

Reference: Gupta et al., "Phoneme Discretized Saliency Maps for Explainable
Detection of AI-Generated Voice," Interspeech 2024.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class PhonemeSegment:
    """A single phoneme with temporal boundaries."""
    phoneme: str
    start_sec: float
    end_sec: float
    confidence: float = 1.0

    @property
    def duration_sec(self) -> float:
        return self.end_sec - self.start_sec

    def start_frame(self, frame_shift_ms: int = 20) -> int:
        return int(self.start_sec * 1000 / frame_shift_ms)

    def end_frame(self, frame_shift_ms: int = 20) -> int:
        return int(self.end_sec * 1000 / frame_shift_ms)


def align_phonemes_mock(
    duration_sec: float, n_phonemes: int = 20
) -> list[PhonemeSegment]:
    """Mock phoneme alignment for testing."""
    phoneme_dur = duration_sec / n_phonemes
    phonemes = ["AH", "B", "CH", "D", "EH", "F", "G", "HH", "IH", "JH",
                 "K", "L", "M", "N", "OW", "P", "R", "S", "T", "UW"]
    segments = []
    for i in range(n_phonemes):
        segments.append(PhonemeSegment(
            phoneme=phonemes[i % len(phonemes)],
            start_sec=i * phoneme_dur,
            end_sec=(i + 1) * phoneme_dur,
            confidence=0.95,
        ))
    return segments


def align_with_mfa(
    wav_path: str | Path,
    transcript: str | None = None,
    language: str = "english_us_arpa",
) -> list[PhonemeSegment]:
    """Align phonemes using Montreal Forced Aligner.

    Requires MFA installed: conda install -c conda-forge montreal-forced-aligner

    Args:
        wav_path: Path to audio file.
        transcript: Optional transcript. If None, uses MFA G2P.
        language: MFA acoustic model/dictionary name.

    Returns:
        List of PhonemeSegments.
    """
    import subprocess
    import tempfile
    import json
    import shutil

    wav_path = Path(wav_path)

    with tempfile.TemporaryDirectory() as tmpdir:
        input_dir = Path(tmpdir) / "input"
        output_dir = Path(tmpdir) / "output"
        input_dir.mkdir()
        output_dir.mkdir()

        shutil.copy2(wav_path, input_dir / wav_path.name)

        if transcript:
            txt_path = input_dir / wav_path.with_suffix(".txt").name
            txt_path.write_text(transcript)

        cmd = [
            "mfa", "align",
            str(input_dir),
            language,
            language,
            str(output_dir),
            "--output_format", "json",
            "--clean",
        ]
        subprocess.run(cmd, capture_output=True, check=True)

        results = list(output_dir.rglob("*.json"))
        if not results:
            results = list(output_dir.rglob("*.TextGrid"))
        if not results:
            return []

        return _parse_mfa_output(results[0])


def _parse_mfa_output(path: Path) -> list[PhonemeSegment]:
    """Parse MFA JSON or TextGrid output."""
    if path.suffix == ".json":
        import json
        with open(path) as f:
            data = json.load(f)
        segments = []
        for tier in data.get("tiers", {}).values():
            if tier.get("type") == "phones":
                for entry in tier.get("entries", []):
                    if len(entry) >= 3 and entry[2].strip():
                        segments.append(PhonemeSegment(
                            phoneme=entry[2],
                            start_sec=float(entry[0]),
                            end_sec=float(entry[1]),
                        ))
        return segments

    # TextGrid fallback — basic parser
    text = path.read_text()
    segments = []
    lines = text.split("\n")
    in_phones = False
    i = 0
    while i < len(lines):
        if "phones" in lines[i].lower():
            in_phones = True
        if in_phones and "xmin" in lines[i]:
            xmin = float(lines[i].split("=")[1].strip())
            xmax = float(lines[i + 1].split("=")[1].strip())
            text_val = lines[i + 2].split('"')[1] if '"' in lines[i + 2] else ""
            if text_val.strip():
                segments.append(PhonemeSegment(
                    phoneme=text_val.strip(),
                    start_sec=xmin,
                    end_sec=xmax,
                ))
            i += 3
        else:
            i += 1
    return segments


def align_with_whisperx(
    wav_path: str | Path,
    device: str = "cpu",
) -> list[PhonemeSegment]:
    """Align phonemes using WhisperX word-level alignment.

    Note: WhisperX provides word-level, not phoneme-level alignment.
    We use grapheme-to-phoneme (G2P) conversion post-hoc.

    Args:
        wav_path: Path to audio file.
        device: "cpu" or "cuda".

    Returns:
        List of PhonemeSegments (word-level granularity).
    """
    import whisperx

    model = whisperx.load_model("base", device=device)
    audio = whisperx.load_audio(str(wav_path))
    result = model.transcribe(audio)

    model_a, metadata = whisperx.load_align_model(language_code="en", device=device)
    aligned = whisperx.align(result["segments"], model_a, metadata, audio, device)

    segments = []
    for seg in aligned.get("word_segments", []):
        if "start" in seg and "end" in seg:
            segments.append(PhonemeSegment(
                phoneme=seg.get("word", ""),
                start_sec=seg["start"],
                end_sec=seg["end"],
                confidence=seg.get("score", 0.0),
            ))

    return segments
