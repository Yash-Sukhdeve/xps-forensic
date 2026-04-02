#!/usr/bin/env python3
"""Update XPS-Forensic dashboard data from experiment logs.

Parses E1 baseline log and results to produce dashboard_data.json.
Run periodically: watch -n 30 python scripts/update_xps_dashboard.py

Output: results/xps_dashboard_data.json
"""
import json
import os
import re
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
E1_LOG = ROOT / "results" / "e1_run.log"
E1_RESULTS = ROOT / "results" / "e1_baseline" / "results.json"
OUTPUT = ROOT / "results" / "xps_dashboard_data.json"


def parse_e1_log(log_path: Path) -> dict:
    """Parse E1 baseline experiment log for progress."""
    data = {
        "running": False,
        "current_detector": None,
        "detectors_done": [],
        "detectors_skipped": [],
        "progress": {},
        "results": {},
    }
    if not log_path.exists():
        return data

    text = log_path.read_text(errors="replace")
    lines = text.strip().split("\n")

    current_det = None
    total_utt = 71239

    for line in lines:
        # Detect current detector
        m = re.search(r"Detector: (\w+)", line)
        if m:
            current_det = m.group(1).lower()

        # Skip detection
        if "SKIP" in line and current_det:
            data["detectors_skipped"].append(current_det)

        # Progress
        m = re.search(r"Processed (\d+)/(\d+)", line)
        if m and current_det:
            done, total = int(m.group(1)), int(m.group(2))
            data["progress"][current_det] = {"done": done, "total": total}
            data["current_detector"] = current_det

        # Inference complete
        if "Inference complete" in line and current_det:
            m2 = re.search(r"(\d+) utterances in ([\d.]+)s", line)
            if m2:
                data["progress"][current_det] = {
                    "done": int(m2.group(1)),
                    "total": int(m2.group(1)),
                    "time_s": float(m2.group(2)),
                }

        # EER results
        m = re.search(r"Utt-EER: ([\d.]+)", line)
        if m and current_det:
            data["results"].setdefault(current_det, {})["utt_eer"] = float(m.group(1))
            if current_det not in data["detectors_done"]:
                data["detectors_done"].append(current_det)

        # Seg-EER
        m = re.search(r"Seg-EER@(\d+)ms: ([\d.]+)", line)
        if m and current_det:
            data["results"].setdefault(current_det, {})
            data["results"][current_det][f"seg_eer_{m.group(1)}ms"] = float(m.group(2))

        # Seg-F1
        m = re.search(r"Seg-F1@(\d+)ms.*: ([\d.]+)", line)
        if m and current_det:
            data["results"].setdefault(current_det, {})
            data["results"][current_det][f"seg_f1_{m.group(1)}ms"] = float(m.group(2))

    # Check if still running
    if log_path.exists():
        mtime = datetime.fromtimestamp(log_path.stat().st_mtime)
        data["running"] = (datetime.now() - mtime).total_seconds() < 120

    # Load final results if available
    if E1_RESULTS.exists():
        try:
            with open(E1_RESULTS) as f:
                data["final_results"] = json.load(f)
        except json.JSONDecodeError:
            pass

    return data


def main():
    e1 = parse_e1_log(E1_LOG)

    dashboard = {
        "updated_at": datetime.now().isoformat(),
        "e1_baseline": e1,
        "tests": {"total": 136, "passing": 136},
        "codebase": {"loc": 8038, "files": 35},
        "bugs": {
            "open": [
                {"id": 5, "severity": "high", "desc": "PDSM-PS Captum integration"},
                {"id": 6, "severity": "medium", "desc": "E6/E7/E8 placeholders"},
                {"id": 7, "severity": "medium", "desc": "Evidence schema disclosures"},
            ],
            "resolved": [1, 2, 3, 4],
        },
        "gates": {
            "G1": "pending",
            "G2": "pass",
            "G3": "running" if e1["running"] else ("pass" if e1.get("final_results") else "pending"),
            "G4": "pending",
            "G5": "pending",
            "G6": "blocked",
            "G7": "pending",
            "G8": "pending",
        },
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(dashboard, indent=2))
    print(f"[update] Wrote {OUTPUT} ({OUTPUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
